# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from TorchTitan retain its BSD-3-Clause license;
# see LICENSE-BSD-3-Clause in the repository root.
"""Ascend MoE token dispatch and combine with fused NPU permutations."""

from collections.abc import Callable
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
)

from torchtitan_ascend.overrides.permute import permute, unpermute

# T: input tokens, K: selected experts, D: hidden size, E: global experts,
# R: received token copies, e: local experts, N: local token copies (T*K).


class AscendAllToAllTokenDispatcher(AllToAllTokenDispatcher):
    """Use paired permutation maps for local and EP token routing."""

    ep_mesh: DeviceMesh | None
    num_experts: int
    remat_region_name: Callable[[str], str]
    remat_should_recompute: Callable[[str], bool]
    _token_count_exchange: Callable[..., torch.Tensor]
    _sync_token_count_exchange: Callable[
        ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    _dispatch_token_exchange: Callable[..., torch.Tensor]
    _combine_token_exchange: Callable[..., torch.Tensor]

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pass

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, LocalDispatchMetadata | AllToAllDispatchMetadata
    ]:
        """Distribute token copies to experts and group them for computation.

        Without EP, only a local permutation is needed. With EP, permute by
        global expert, exchange tokens across ranks, then group by local expert.

        Args:
            x_TD: Input token features [T, D] from the current rank.
            topk_scores_TK: Routing weights [T, K], saved for combine().
            topk_expert_ids_TK: Selected global expert IDs [T, K], in [0, E).
            num_local_tokens_per_expert_E: Counts [E] of this rank's token
                assignments to every global expert, including remote experts.

        Returns:
            routed_input_RD: Expert-grouped token features [R, D]. Without
                EP, R=T*K; with EP, R is the number of received token copies.
            num_global_tokens_per_local_expert_e: Counts [e] for local experts,
                summed across source ranks. Without EP, e=E.
            metadata: Routing weights, inverse permutation maps and all-to-all
                splits needed by combine() to restore and merge expert outputs.
        """
        if spmd.is_type_checking():
            spmd.mutate_type(
                num_local_tokens_per_expert_E,
                src=spmd.P,
                dst={"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
            )

        # EP=1: local dispatch — no all-to-all needed.
        if self.ep_mesh is None:
            routed_input_ND, local_sorted_indices_N = permute(
                x_TD, topk_expert_ids_TK.to(torch.int32)
            )
            return (
                routed_input_ND,
                num_local_tokens_per_expert_E,
                LocalDispatchMetadata(
                    token_indices_experts_sorted_N=local_sorted_indices_N,
                    topk_scores_experts_sorted_N=topk_scores_TK,
                ),
            )

        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            local_sorted_indices_N,
            ep_sorted_indices_R,
            input_splits,
            output_splits,
        ) = remat.region(
            self._dispatch,
            self.remat_region_name("dispatch"),
            recompute=self.remat_should_recompute("dispatch"),
        )(
            x_TD, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        remat.recompute_needs_tensor(
            num_global_tokens_per_local_expert_e, input_splits, output_splits
        )
        input_splits_list = input_splits.tolist()
        output_splits_list = output_splits.tolist()
        return (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            AllToAllDispatchMetadata(
                token_indices_experts_sorted_N=local_sorted_indices_N,
                topk_scores_experts_sorted_N=topk_scores_TK,
                input_shape=torch.Size((sum(output_splits_list), x_TD.shape[-1])),
                permuted_indices=ep_sorted_indices_R,
                input_splits=input_splits_list,
                output_splits=output_splits_list,
            ),
        )

    def _dispatch(
        self,
        x_TD: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return tensors only from the complete dispatch remat region."""
        assert self.ep_mesh is not None
        ep_size = self.ep_mesh.size()
        routed_input_ND, local_sorted_indices_N = permute(
            x_TD, topk_expert_ids_TK.to(torch.int32)
        )
        with maybe_set_sparse_mesh():
            pg = "ep"
            if spmd.is_type_checking():
                num_local_tokens_per_expert_E = spmd.reinterpret_mesh(
                    num_local_tokens_per_expert_E, spmd.current_mesh()
                )
                routed_input_ND = spmd.reinterpret_mesh(
                    routed_input_ND, spmd.current_mesh()
                )

            # Generate the input splits and output splits for all-to-all
            with torch.no_grad():
                num_global_tokens_per_local_expert_EP_e = self._token_count_exchange(
                    num_local_tokens_per_expert_E, pg, ep_size
                )
                (
                    num_global_tokens_per_local_expert_E,
                    input_splits,
                    output_splits,
                ) = self._sync_token_count_exchange(
                    num_local_tokens_per_expert_E,
                    num_global_tokens_per_local_expert_EP_e,
                    ep_size,
                )

            routed_input_RD = self._dispatch_token_exchange(
                routed_input_ND, pg, output_splits.tolist(), input_splits.tolist()
            )
            # Received blocks are rank-major. Label each row with its local
            # expert so a second permutation groups them by expert.
            counts_EP_e = num_global_tokens_per_local_expert_E.reshape(ep_size, -1)
            num_local_experts = self.num_experts // ep_size
            _expert_ids_per_ep_rank = (
                torch.arange(
                    self.num_experts, dtype=torch.int32, device=routed_input_RD.device
                )
                % num_local_experts
            )
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                _expert_ids_per_ep_rank, num_global_tokens_per_local_expert_E.ravel()
            )
            routed_input_RD, ep_sorted_indices_R = permute(
                routed_input_RD, global_input_tokens_local_experts_indices
            )
            num_global_tokens_per_local_expert_e = counts_EP_e.sum(dim=0)

        return (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            local_sorted_indices_N,
            ep_sorted_indices_R,
            input_splits,
            output_splits,
        )

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Restore token order and merge expert outputs with routing weights.

        Without EP, locally unpermute and merge top-K outputs. With EP, undo
        grouping by local expert, return outputs to their source ranks, then
        restore the original token order and merge top-K outputs.

        Args:
            routed_output_RD: Expert outputs [R, D] in dispatch order. Without
                EP, R=T*K; with EP, R is the number of received token copies.
            metadata: Routing weights [T, K], inverse permutation maps and
                all-to-all splits returned by dispatch(). Without EP, local
                metadata contains only the weights and first inverse map.
            x_TD: Original input token features [T, D]. Kept for dispatcher
                interface compatibility; unused by this implementation.

        Returns:
            out_TD: Routing-weighted sum of top-K expert outputs [T, D],
                arranged in the original local token order.
        """
        if self.ep_mesh is None:
            remat.recompute_needs_tensor(routed_output_RD)
            return unpermute(
                routed_output_RD,
                metadata.token_indices_experts_sorted_N,
                metadata.topk_scores_experts_sorted_N,
            )
        out_TD = remat.region(
            self._combine,
            self.remat_region_name("combine"),
            recompute=self.remat_should_recompute("combine"),
        )(
            routed_output_RD,
            metadata.topk_scores_experts_sorted_N,
            metadata.token_indices_experts_sorted_N,
            metadata.permuted_indices,
            metadata.input_splits,
            metadata.output_splits,
        )
        return out_TD

    def _combine(
        self,
        routed_output_RD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        local_sorted_indices_N: torch.Tensor,
        ep_sorted_indices_R: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        with maybe_set_sparse_mesh():
            routed_output_RD = unpermute(routed_output_RD, ep_sorted_indices_R)
            routed_output_RD = self._combine_token_exchange(
                routed_output_RD, "ep", input_splits, output_splits
            )
        if spmd.is_type_checking():
            routed_output_RD = spmd.reinterpret_mesh(
                routed_output_RD, spmd.current_mesh()
            )
        return unpermute(routed_output_RD, local_sorted_indices_N, topk_scores_TK)


@override(
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description="NPU fused npu_moe_token_permute/unpermute for MoE dispatch/combine",
)
def npu_token_dispatcher(
    cfg: AllToAllTokenDispatcher.Config,
) -> AscendAllToAllTokenDispatcher.Config:
    return derive(cfg, AscendAllToAllTokenDispatcher.Config)
