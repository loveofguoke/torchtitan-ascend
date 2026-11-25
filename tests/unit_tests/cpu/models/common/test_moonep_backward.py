# SPDX-License-Identifier: Apache-2.0
"""Native PP I/W gradients over real MoonEP autograd and CPU reference transport."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch_remat as remat
from torch.distributed.pipelining import stage as pipeline_stage
from torch.distributed.pipelining._backward import stage_backward_input

from torchtitan_ascend import setup

setup()

from torchtitan_ascend.overrides.moonep import ops


class ReferenceRuntime:
    ep_rank = 0
    num_local_experts = 2
    num_experts = 4
    communication_options = {}

    def __init__(self):
        self.phase = "forward"
        self.events = {}

    def record(self, event):
        assert not remat.is_recomputing(), "MoonEP communication must not replay"
        key = (self.phase, event)
        self.events[key] = self.events.get(key, 0) + 1

    def call(self, method, *args, **kwargs):
        self.record(method)
        if method == "dispatch":
            if "plan" in kwargs:
                return args[0].clone(), None, None, None
            return (
                args[0].clone(),
                args[1].flatten().clone(),
                torch.tensor([2, 4, 4, 4, 4, 4]),
                object(),
            )
        if method == "combine":
            score_grad = kwargs.get("route_weights_nvs")
            return (
                kwargs["hidden_nvsh"].clone(),
                None if score_grad is None else score_grad[:, None].clone(),
                None,
            )
        raise AssertionError(method)

    def prefetch(self, plan, projections):
        self.record("prefetch")
        return iter(projections)

    def reduce(self, plan, grads, dtypes):
        self.record("reduce")
        return tuple(value[:2].to(dtype).clone() for value, dtype in zip(grads, dtypes))

    def trace(self, event):
        pass


def reference_rows_mm(x, projection, ends, *, transpose=True):
    weights, _ = projection
    pieces = []
    start = 0
    for index, end in enumerate(ends.tolist()):
        matrix = weights[index].T if transpose else weights[index]
        pieces.append(x[start:end] @ matrix)
        start = end
    return torch.cat(pieces)


def activation(x, *, offsets):
    return torch.nn.functional.silu(x[:, 0]) * x[:, 1]


@pytest.mark.parametrize("recompute_dispatch", [False, True])
@pytest.mark.parametrize("remat_enabled", [False, True])
def test_native_split_defers_expert_weight_work(
    monkeypatch, recompute_dispatch, remat_enabled
):
    from torchtitan_ascend.overrides.moonep.pipeline import split_backward_context

    monkeypatch.setattr(ops, "_plans", {})
    monkeypatch.setattr(ops, "_expert_calls", {})
    runtime = ReferenceRuntime()

    def weight_grad(dy, x, ends):
        runtime.record("weight_grad")
        pieces = []
        start = 0
        for end in ends.tolist():
            pieces.append(dy[start:end].T @ x[start:end])
            start = end
        return torch.stack(pieces)

    monkeypatch.setattr(ops, "rows_mm", reference_rows_mm)
    monkeypatch.setattr(ops, "weight_grad", weight_grad)
    monkeypatch.setattr(ops, "current_runtime", lambda: runtime)
    generator = torch.Generator().manual_seed(42)
    initial = [
        torch.randn(shape, generator=generator)
        for shape in [(4, 3), (3, 3), (1, 3), (2, 2, 4, 3), (2, 3, 4), (3,), (3,)]
    ]
    dy = torch.randn(4, 3, generator=generator)
    results = []
    for split in [False, True]:
        runtime.events = {}
        runtime.phase = "forward"
        x, dense, router, w13, w2, scale1, scale2 = [
            torch.nn.Parameter(value.clone()) for value in initial
        ]

        def forward(x):
            hidden = remat.region(
                torch.nn.functional.linear, "hidden", recompute=False
            )(x.cos(), dense)
            remat.recompute_needs_tensor(hidden)
            scores = torch.nn.functional.linear(hidden, router).sigmoid()
            projections = (
                (torch.cat([w13.flatten(1, 2), torch.zeros(2, 8, 3)]), None),
                (torch.cat([w2, torch.zeros(2, 3, 4)]), None),
            )
            ids = torch.zeros(4, 1, dtype=torch.int64)
            counts = torch.tensor([2, 2, 0, 0])
            if recompute_dispatch:
                y = ops.recomputed_experts(
                    activation,
                    projections,
                    hidden,
                    scores,
                    ids,
                    counts,
                    w13,
                    w2,
                    "moe",
                    split_backward=split,
                )
            else:
                dispatched, routes, ends, plan_id = ops.dispatch_tokens(
                    hidden, scores, ids, counts, "dispatch"
                )
                weighted = ops.routed_experts(
                    activation,
                    projections,
                    dispatched,
                    routes,
                    w13,
                    w2,
                    ends,
                    plan_id,
                    "experts",
                    split_backward=split,
                )
                y = ops.combine_tokens(weighted, plan_id, "combine")
            return y.sin() + hidden.sin() * scale1 + hidden.cos() * scale2

        y = (remat.checkpoint()(forward) if remat_enabled else forward)(x).clone()
        parameters = [dense, router, w13, w2, scale1, scale2]
        if split:
            runtime.phase = "input"
            with split_backward_context(remat_enabled=remat_enabled):
                _, groups = stage_backward_input((y,), [dy], [x], iter(parameters))
            assert all(parameter.grad is None for parameter in parameters)
            assert runtime.events.get(("input", "weight_grad"), 0) == 0
            assert runtime.events.get(("input", "reduce"), 0) == 0
            assert runtime.events["input", "dispatch"] == (
                2 if recompute_dispatch else 1
            )
            runtime.phase = "weight"
            with split_backward_context(remat_enabled=remat_enabled):
                pipeline_stage.stage_backward_weight(iter(parameters), groups)
            assert runtime.events.get(("weight", "weight_grad"), 0) == 2
            assert runtime.events.get(("weight", "reduce"), 0) == 1
            assert runtime.events.get(("weight", "prefetch"), 0) == 0
            assert runtime.events.get(("weight", "dispatch"), 0) == 0
            assert runtime.events.get(("weight", "combine"), 0) == 0
        else:
            runtime.phase = "full"
            y.backward(dy)
        assert all(value.grad is not None for value in [x, *parameters])
        results.append([value.grad.clone() for value in [x, *parameters]])
        assert not ops._plans
    for expected, actual in zip(results[0], results[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def make_stage(stage_index):
    from torchtitan_ascend.overrides.moonep.experts import MoonEPRoutedExperts

    expert = cast(MoonEPRoutedExperts, object.__new__(MoonEPRoutedExperts))
    torch.nn.Module.__init__(expert)
    expert.split_backward = False
    return SimpleNamespace(
        stage_index=stage_index,
        submod=expert,
        forward_one_chunk=lambda chunk_id: expert.split_backward,
    )


@pytest.mark.parametrize(
    "schedule_name,num_local_stages",
    [
        ("1F1B", 1),
        ("GPipe", 1),
        ("Interleaved1F1B", 6),
        ("InterleavedZeroBubble", 2),
        ("InterleavedZeroBubble", 6),
        ("ZBVZeroBubble", 2),
        ("DualPipeV", 2),
    ],
)
def test_native_schedule_selects_split_per_microbatch(schedule_name, num_local_stages):
    from torch.distributed.pipelining.schedules import (
        BACKWARD_WEIGHT,
        get_schedule_class,
    )

    from torchtitan_ascend.overrides.moonep.pipeline import configure_pipeline_backward

    schedule_class = get_schedule_class(schedule_name)
    schedule = schedule_class.__new__(schedule_class)
    schedule.pp_group_size = schedule.microbatches_per_round = 2
    schedule._n_microbatches = 8
    schedule.n_local_stages = num_local_stages
    schedule.num_stages = 2 * num_local_stages
    schedule._stages = [make_stage(index) for index in range(schedule.num_stages)]
    if num_local_stages > 1:
        schedule.pipeline_order = {
            rank: schedule._calculate_single_rank_operations(rank) for rank in range(2)
        }
    else:
        schedule.pipeline_order = None
    weight_actions = {
        (action.stage_index, action.microbatch_index)
        for actions in (getattr(schedule, "pipeline_order", None) or {}).values()
        for action in actions
        if action is not None and action.computation_type == BACKWARD_WEIGHT
    }
    forwards = [stage.forward_one_chunk for stage in schedule._stages]
    assert configure_pipeline_backward(schedule) == bool(weight_actions)
    for stage, original_forward in zip(schedule._stages, forwards, strict=True):
        expected = [
            stage.stage_index != 0 and (stage.stage_index, index) in weight_actions
            for index in range(8)
        ]
        assert [stage.forward_one_chunk(index) for index in range(8)] == expected
        assert stage.submod.split_backward == all(expected)
        assert (stage.forward_one_chunk is original_forward) == (
            all(expected) or not any(expected)
        )


def test_mixed_split_flag_is_restored_after_forward_error():
    from torchtitan_ascend.overrides.moonep.pipeline import _configure_stage

    stage = make_stage(1)

    def forward(chunk_id):
        assert stage.submod.split_backward is True
        raise RuntimeError("forward failed")

    stage.forward_one_chunk = forward
    _configure_stage(stage, {1}, 2)
    with pytest.raises(RuntimeError, match="forward failed"):
        stage.forward_one_chunk(1)
    assert stage.submod.split_backward is False
