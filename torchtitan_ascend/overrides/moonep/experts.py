# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from pytorch/torchtitan PR #4751. See LICENSE-BSD-3-Clause.
"""MoonEP dispatch and routed-expert integration."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import BaseEPTokenDispatcher

from . import ops
from .runtime import get_runtime
from .settings import MoonEPConfig


@dataclass(frozen=True)
class MoonEPDispatchMetadata:
    """Retain each call's plan and original token count for combine."""

    plan_id: torch.Tensor
    route_weights: torch.Tensor
    group_end_offsets: torch.Tensor
    num_tokens: int
    source_input: torch.Tensor | None = None


def _checkpoint_expert_weights(module, state_dict, prefix, local_metadata):
    """Keep checkpoint staging away from unmapped portions of VMM storage."""
    for name in ("w13.weight", "w2.weight"):
        key = prefix + name
        if key in state_dict:
            state_dict[key] = state_dict[key].detach().clone()


class MoonEPTokenDispatcher(BaseEPTokenDispatcher):
    """Pad variable multimodal batches to MoonEP's static token capacity."""

    num_experts: int
    top_k: int
    ep_mesh: Any
    remat_region_name: Callable[[str], str]

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config, MoonEPConfig):
        hidden_dim: int
        num_max_tokens_per_rank: int

        def __post_init__(self):
            self.validate()
            if (
                min(
                    self.hidden_dim,
                    self.num_max_tokens_per_rank,
                    cast(BaseEPTokenDispatcher.Config, self).top_k,
                )
                <= 0
            ):
                raise ValueError(
                    "MoonEP width, token capacity and top-k must be positive"
                )

    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.communication_options = config.options()
        self.runtime = None

    def init_buffer(self):
        """Resolve the ordinary expert-parallel group before model execution."""
        self.ep_device_mesh = self.ep_mesh
        if self.ep_device_mesh is None:
            raise ValueError("MoonEP requires expert parallelism")
        self.runtime = get_runtime(
            self.ep_device_mesh.get_group(),
            self.num_max_tokens_per_rank,
            self.hidden_dim,
            self.top_k,
            self.num_experts,
            **self.communication_options,
        )

    def _prepare_dispatch(
        self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
    ):  # noqa: N803
        """Validate and pad every physical route, including zero-weight padding routes."""
        del num_local_tokens_per_expert_E
        num_tokens = x_TD.shape[0]
        if x_TD.dtype != torch.bfloat16:
            raise ValueError("MoonEP expert inputs must be BF16")
        if num_tokens > self.num_max_tokens_per_rank:
            raise ValueError(
                f"MoonEP token capacity {self.num_max_tokens_per_rank} exceeded by {num_tokens}"
            )
        if self.communication_options["validate_routing"]:
            if topk_expert_ids_TK.shape != (
                num_tokens,
                self.top_k,
            ) or topk_scores_TK.shape != (num_tokens, self.top_k):
                raise ValueError(
                    "MoonEP routing ids/scores must have shape [tokens, top_k]"
                )
            if topk_expert_ids_TK.dtype not in (torch.int32, torch.int64):
                raise ValueError("MoonEP routing ids must be integers")
            if topk_expert_ids_TK.numel() and not bool(
                (
                    (topk_expert_ids_TK >= 0) & (topk_expert_ids_TK < self.num_experts)
                ).all()
            ):
                raise ValueError(
                    f"MoonEP routing ids must lie in [0, {self.num_experts})"
                )
        num_padding_tokens = self.num_max_tokens_per_rank - num_tokens
        source_input = functional.pad(x_TD, (0, 0, 0, num_padding_tokens))
        topk_scores = functional.pad(topk_scores_TK, (0, 0, 0, num_padding_tokens))
        topk_expert_ids = functional.pad(
            topk_expert_ids_TK, (0, 0, 0, num_padding_tokens)
        )
        # Router counts omit masked tokens; planning must cover all physical IDs.
        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.int32, device=topk_expert_ids.device
        )
        tokens_per_expert.scatter_add_(
            0,
            topk_expert_ids.flatten().long(),
            torch.ones_like(topk_expert_ids.flatten(), dtype=torch.int32),
        )
        return source_input, topk_scores, topk_expert_ids, tokens_per_expert

    def dispatch(
        self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
    ):  # noqa: N803
        num_tokens = x_TD.shape[0]
        (
            source_input,
            topk_scores,
            topk_expert_ids,
            tokens_per_expert,
        ) = self._prepare_dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        (
            dispatched_input,
            route_weights,
            group_end_offsets,
            plan_id,
        ) = ops.dispatch_tokens(
            source_input,
            topk_scores,
            topk_expert_ids,
            tokens_per_expert,
            self.remat_region_name("ep_communication.dispatch"),
        )
        return (
            dispatched_input,
            group_end_offsets.diff(prepend=group_end_offsets.new_zeros(1)),
            MoonEPDispatchMetadata(
                plan_id,
                route_weights,
                group_end_offsets,
                num_tokens,
                source_input
                if self.communication_options["recompute_dispatch"]
                else None,
            ),
        )

    def combine(self, routed_output_RD, metadata, x_TD):  # noqa: N803
        """Combine already weighted expert outputs and remove static padding."""
        return ops.combine_tokens(
            routed_output_RD,
            metadata.plan_id,
            self.remat_region_name("ep_communication.combine"),
        )[: metadata.num_tokens].to(x_TD.dtype)


class MoonEPRoutedExperts(RoutedExperts):
    """Keep GroupedLinear names and shapes over resident, EP-sharded symmetric homes."""

    w13: GroupedLinear
    w2: GroupedLinear
    token_dispatcher: MoonEPTokenDispatcher
    activation_fn: Callable
    remat_region_name: Callable[[str], str]
    weight_projections: tuple

    def __init__(self, config):
        super().__init__(config)
        cast(torch.nn.Module, self).register_state_dict_post_hook(
            _checkpoint_expert_weights
        )

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        def __post_init__(self):
            RoutedExperts.Config.__post_init__(self)
            config = cast(RoutedExperts.Config, self)
            if (
                type(config.w13) is not GroupedLinear.Config
                or type(config.w2) is not GroupedLinear.Config
            ):
                raise ValueError("MoonEP requires plain GroupedLinear experts")
            if config.output_postprocess is not None:
                raise ValueError("MoonEP does not support expert output_postprocess")

    def _init_self_buffers(self, *, buffer_device=None):
        if self.w2.weight.device.type != "npu":
            raise ValueError("MoonEP expert pools require NPU initialization")
        super()._init_self_buffers(buffer_device=buffer_device)
        runtime = self.token_dispatcher.runtime
        assert runtime is not None
        runtime.allocate(
            (
                (2 * self.w13.out_features, self.w13.in_features),
                (self.w2.out_features, self.w2.in_features),
            )
        )

    def forward(
        self,
        x_TD,  # noqa: N803
        topk_scores_TK,  # noqa: N803
        topk_expert_ids_TK,  # noqa: N803
        num_local_tokens_per_expert_E,  # noqa: N803
        *,
        padding_mask_T=None,  # noqa: N803
    ):
        del padding_mask_T
        if self.token_dispatcher.communication_options["recompute_dispatch"]:
            dispatch_inputs = self.token_dispatcher._prepare_dispatch(
                x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
            )
            with maybe_set_sparse_mesh():
                combined_output = ops.recomputed_experts(
                    self.activation_fn,
                    self.weight_projections,
                    *dispatch_inputs,
                    self.w13.weight.to_local(),
                    self.w2.weight.to_local(),
                    self.remat_region_name("moonep_experts"),
                )
            return combined_output[: x_TD.shape[0]].to(x_TD.dtype)
        dispatched_input, _, metadata = self.token_dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        with maybe_set_sparse_mesh():
            weighted_expert_output = ops.routed_experts(
                self.activation_fn,
                self.weight_projections,
                dispatched_input,
                metadata.route_weights,
                self.w13.weight.to_local(),
                self.w2.weight.to_local(),
                metadata.group_end_offsets,
                metadata.plan_id,
                self.remat_region_name("moonep_experts"),
                source_input=metadata.source_input,
            )
        return self.token_dispatcher.combine(weighted_expert_output, metadata, x_TD)


@override(
    target=RoutedExperts.Config,
    description="Ascend MoonEP expert dispatch, computation and combine",
)
def npu_moonep(config):
    """Select MoonEP experts through the override API."""
    if not isinstance(config.token_dispatcher, MoonEPTokenDispatcher.Config):
        raise ValueError(
            "Configure MoonEP through MoonEPTrainer.Config before applying its override"
        )
    return derive(config, MoonEPRoutedExperts.Config)
