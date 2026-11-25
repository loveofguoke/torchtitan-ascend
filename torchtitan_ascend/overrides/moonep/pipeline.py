# SPDX-License-Identifier: Apache-2.0
"""Adapt MoonEP to native input/weight backward actions and checkpoint replay."""

from collections import defaultdict
from contextlib import contextmanager
from functools import partial, wraps
from unittest.mock import patch

import torch
import torch._functorch.config as functorch_config
from torch.distributed.pipelining import stage as pipeline_stage
from torch.distributed.pipelining.schedules import BACKWARD_WEIGHT
from torch_remat import _api as remat_api

from .experts import MoonEPRoutedExperts


def configure_pipeline_backward(schedule) -> bool:
    """Select split expert backward from the native schedule's W actions."""
    weight_microbatches: dict[int, set[int]] = defaultdict(set)
    # Single-stage schedules have no action table and always use full backward.
    for actions in (getattr(schedule, "pipeline_order", None) or {}).values():
        for action in actions:
            if action is not None and action.computation_type == BACKWARD_WEIGHT:
                weight_microbatches[action.stage_index].add(action.microbatch_index)
    if not weight_microbatches:
        return False
    for stage in schedule._stages:
        _configure_stage(
            stage, weight_microbatches[stage.stage_index], schedule._n_microbatches
        )
    return True


def _configure_stage(stage, weight_microbatches, num_microbatches):
    experts = [m for m in stage.submod.modules() if isinstance(m, MoonEPRoutedExperts)]
    # Native PP skips I on stage 0 and performs full backward during W instead.
    split_microbatches = frozenset(weight_microbatches if stage.stage_index else ())
    uniform_split = len(split_microbatches) == num_microbatches
    for expert in experts:
        expert.split_backward = uniform_split
    if not split_microbatches or uniform_split:
        return
    forward = stage.forward_one_chunk

    @wraps(forward)
    def forward_one_chunk(chunk_id, *args, **kwargs):
        # DualPipeV mixes full B and I/W in the same stage.
        for expert in experts:
            expert.split_backward = chunk_id in split_microbatches
        try:
            return forward(chunk_id, *args, **kwargs)
        finally:
            for expert in experts:
                expert.split_backward = False

    stage.forward_one_chunk = forward_one_chunk


@contextmanager
def split_backward_context(*, remat_enabled: bool):
    """Keep native I/W state alive until all W parameter groups finish."""
    with functorch_config.patch(donated_buffer=False):
        if not remat_enabled:
            yield
            return
        load_saved_input = remat_api._load_saved_input

        @wraps(load_saved_input)
        def load(record, region_state, saved_input):
            value = load_saved_input(record, region_state, saved_input)
            # W resumes below the replay trigger and may read the same input again.
            if torch._C._autograd._get_current_graph_task_keep_graph():
                saved_input.value = value
            return value

        # Different W parameter groups can replay the same checkpoint independently.
        with (
            patch.object(remat_api, "_load_saved_input", load),
            patch.object(
                pipeline_stage,
                "stage_backward_weight",
                partial(pipeline_stage.stage_backward_weight, retain_graph=True),
            ),
        ):
            yield
