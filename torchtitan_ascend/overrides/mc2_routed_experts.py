# SPDX-License-Identifier: Apache-2.0
"""Ascend MC2 routed experts with fused communication and grouped matmul."""

from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist
import torch_remat as remat
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_sparse_mesh
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher

from torchtitan_ascend.overrides.gemm_mc2 import (
    all2all_grouped_matmul,
    grouped_matmul_all2all,
)
from torchtitan_ascend.overrides.permute import permute, unpermute


class AscendMC2RoutedExperts(RoutedExperts):
    """Replace the native RoutedExperts EP path with MC2 fused kernels.

    Fuse dispatch, expert regrouping and w13 in all2all_grouped_matmul;
    fuse w2, inverse regrouping and return in grouped_matmul_all2all.
    Local token permute/unpermute and activation remain separate.
    Read inherited weights directly, preserving native parameter layouts,
    and use custom autograd instead of the GroupedLinear forward path.

    With no EP or EP size 1, use the native forward. The MC2 path does not
    support output_postprocess or SPMD typing yet. The fused dispatch/w13
    and w2/combine segments follow the dispatcher remat policy as whole regions.
    The default policy retains both, including w13 fused with dispatch.
    """

    output_postprocess: torch.nn.Module | None
    token_dispatcher: AllToAllTokenDispatcher
    w13: GroupedLinear
    w2: GroupedLinear
    activation_fn: torch.nn.Module

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Permute -> dispatch/w13 -> activation -> w2/return -> unpermute.

        Padding rows participate in communication and expert computation, as
        in upstream RoutedExperts; the mask does not remove physical rows.
        """
        sparse_mesh = spmd_sparse_mesh()
        ep_mesh = None if sparse_mesh is None else sparse_mesh["ep"]
        if ep_mesh is None or ep_mesh.size() == 1:
            return super().forward(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
                padding_mask_T=padding_mask_T,
            )

        if self.output_postprocess is not None:
            raise NotImplementedError(
                "MC2 does not support output_postprocess because w2 and return communication are fused."
            )

        (
            gate_up_R2F,
            local_inverse_indices_N,
            send_counts,
            recv_counts,
            offsets_e,
        ) = remat.region(
            self._dispatch_w13,
            self.token_dispatcher.remat_region_name("dispatch"),
            recompute=self.token_dispatcher.remat_should_recompute("dispatch"),
        )(
            x_TD, topk_expert_ids_TK, self.w13.weight, num_local_tokens_per_expert_E
        )
        remat.recompute_needs_tensor(gate_up_R2F, offsets_e)

        with maybe_set_sparse_mesh():
            hidden_RF = self.activation_fn(gate_up_R2F, offsets=offsets_e)

        out_TD = remat.region(
            self._w2_combine,
            self.token_dispatcher.remat_region_name("combine"),
            recompute=self.token_dispatcher.remat_should_recompute("combine"),
        )(
            hidden_RF,
            self.w2.weight,
            send_counts,
            recv_counts,
            local_inverse_indices_N,
            topk_scores_TK,
        )
        if out_TD.dtype != x_TD.dtype:
            remat.recompute_needs_tensor(out_TD)
        return out_TD.type_as(x_TD)

    def _dispatch_w13(
        self,
        x_TD: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        weight_e2FD: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Count exchange and permutations must be saved with the fused collective.
        ep_mesh = spmd_sparse_mesh()["ep"]
        ep_size = ep_mesh.size()
        ep_rank = ep_mesh.get_local_rank()
        ep_group = ep_mesh.get_group()
        num_experts = self.token_dispatcher.num_experts
        num_local_experts = num_experts // ep_size
        send_counts = num_local_tokens_per_expert_E
        routed_input_ND, local_inverse_indices_N = permute(x_TD, topk_expert_ids_TK)

        with maybe_set_sparse_mesh():
            num_global_tokens_per_expert = torch.zeros(
                ep_size, num_experts, dtype=send_counts.dtype, device=send_counts.device
            )
            dist.all_gather_into_tensor(
                num_global_tokens_per_expert, send_counts, group=ep_group
            )
            start_idx = ep_rank * num_local_experts
            end_idx = start_idx + num_local_experts
            recv_counts_EP_e = num_global_tokens_per_expert[:, start_idx:end_idx]
            recv_counts = recv_counts_EP_e.reshape(-1)
            offsets_e = recv_counts_EP_e.sum(dim=0).cumsum(dim=0, dtype=torch.int32)
            weight_eD2F = weight_e2FD.flatten(1, 2).transpose(-2, -1)
            gate_up_R2F = cast(
                torch.Tensor,
                all2all_grouped_matmul(
                    inputs=routed_input_ND.bfloat16(),
                    weights=weight_eD2F.bfloat16().contiguous(),
                    group=ep_group,
                    send_counts=send_counts,
                    recv_counts=recv_counts,
                ),
            )
            gate_up_R2F = gate_up_R2F.unflatten(-1, (2, self.w13.out_features))

        return gate_up_R2F, local_inverse_indices_N, send_counts, recv_counts, offsets_e

    def _w2_combine(
        self,
        hidden_RF: torch.Tensor,
        weight_eDF: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
        local_inverse_indices_N: torch.Tensor,
        topk_scores_TK: torch.Tensor,
    ) -> torch.Tensor:
        ep_group = spmd_sparse_mesh()["ep"].get_group()
        with maybe_set_sparse_mesh():
            weight_eFD = weight_eDF.transpose(-2, -1)
            routed_output_ND = grouped_matmul_all2all(
                inputs=hidden_RF,
                weights=weight_eFD.bfloat16().contiguous(),
                group=ep_group,
                send_counts=recv_counts,
                recv_counts=send_counts,
            )
        return unpermute(
            routed_output_ND,
            local_inverse_indices_N,
            probs=topk_scores_TK,
        )


@override(
    target=RoutedExperts.Config,
    exact=True,
    description="Ascend MC2 routed experts with fused communication and grouped matmul",
)
def npu_mc2_routed_experts(
    cfg: RoutedExperts.Config,
) -> AscendMC2RoutedExperts.Config:
    """Replace the parent config while retaining its existing child configs.

    Activate this instead of the token-dispatcher override: TorchTitan rejects
    separate overrides claiming a parent and its descendant in the same tree.
    """
    return derive(cfg, AscendMC2RoutedExperts.Config)
