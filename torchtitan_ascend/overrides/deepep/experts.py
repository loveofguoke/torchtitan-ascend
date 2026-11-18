# SPDX-License-Identifier: Apache-2.0
"""DeepEP-Ascend expanded token dispatch/combine for MoE routed experts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_remat as remat
from torchtitan.config import derive, override
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    BaseEPTokenDispatcher,
)

from torchtitan_ascend.device import require_device

# Importing ops registers the dispatch/combine custom ops.
from torchtitan_ascend.overrides.deepep import ops as _ops, runtime

# T = source tokens, D = hidden size, K = selected experts, R = received
# token-expert rows, e = local experts.


@dataclass
class DispatchState:
    """Per-call handle and per-row scores carried from dispatch to combine."""

    handle_id_1: torch.Tensor  # CPU int64 tensor indexing the handle cache
    recv_scores_R: torch.Tensor  # noqa: N815


class AscendDeepEPTokenDispatcher(BaseEPTokenDispatcher):
    """Dispatch through DeepEP-Ascend's expand-only EPBuffer.

    The expanded output is already expert-major, so RoutedExperts computes on it
    directly; routing scores are applied before the combine op so their gradient
    stays in plain autograd.
    """

    num_experts: int
    top_k: int
    ep_mesh: Any
    _remat_module_fqn: str | None
    remat_region_name: Callable[[str], str]
    remat_should_recompute: Callable[[str], bool]

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        num_max_tokens_per_rank: int
        hidden_dim: int
        expert_alignment: int = 1

        def __post_init__(self) -> None:
            if self.num_max_tokens_per_rank <= 0 or self.hidden_dim <= 0:
                raise ValueError(
                    "DeepEP token capacity and hidden_dim must be positive"
                )
            if self.expert_alignment != 1:
                raise ValueError("DeepEP-Ascend supports expert_alignment=1 only")

    def __init__(self, config: Config):
        super().__init__(config)
        require_device("Ascend950", feature="DeepEP-Ascend")
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.hidden_dim = config.hidden_dim
        # Deferred to the first dispatch: EPBuffer's constructor barrier would
        # race FSDP parameter-init collectives (DeepSeek V4.1 EP8).
        self._buffer_entry: runtime._BufferEntry | None = None

    def _replays_communication(self) -> bool:
        """Whether activation remat replays dispatch/combine during backward."""
        return bool(self._remat_module_fqn) and self.remat_should_recompute(
            "ep_communication"
        )

    def init_buffer(self) -> None:
        """Resolve the EP group; the buffer itself is created on first dispatch."""
        if self.ep_mesh is None:
            raise ValueError("DeepEP requires expert parallel degree > 1")
        if self.ep_mesh.size() % 2:
            raise ValueError("DeepEP requires an even expert parallel degree")

    def _ensure_buffer(self) -> None:
        """Create the buffer at the first dispatch and keep this entry active.

        All EP ranks reach the first dispatch together, so EPBuffer's
        constructor barrier cannot deadlock.
        """
        if self._buffer_entry is None:
            self._buffer_entry = runtime.get_buffer(
                self.ep_mesh.get_group(),
                hidden=self.hidden_dim,
                num_max_tokens_per_rank=self.num_max_tokens_per_rank,
                num_topk=self.top_k,
                deterministic=torch.are_deterministic_algorithms_enabled()
                or self._replays_communication(),
            )
        else:
            runtime.set_active_buffer(self._buffer_entry)

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
        """Dispatch through the EPBuffer; received counts come from its handle."""
        # Ignore the router-provided counts; DeepEP returns per-local-expert
        # received counts in its handle.
        del num_local_tokens_per_expert_E
        if x_TD.dtype != torch.bfloat16:
            raise ValueError(f"DeepEP dispatch requires BF16 input, got {x_TD.dtype}")
        if x_TD.shape[0] > self.num_max_tokens_per_rank:
            raise ValueError(
                f"DeepEP token count {x_TD.shape[0]} exceeds the preallocated "
                f"capacity {self.num_max_tokens_per_rank}"
            )

        self._ensure_buffer()
        recompute = self._replays_communication()
        entry = self._buffer_entry
        assert entry is not None
        if recompute and not entry.buffer.deterministic:
            raise RuntimeError(
                "DeepEP dispatch replay under activation checkpointing requires "
                "a deterministic EPBuffer"
            )

        # Mask zero-score selections; DeepEP uses -1 for "no selection".
        topk_idx_TK = topk_expert_ids_TK.contiguous()
        topk_weights_TK = topk_scores_TK.contiguous()
        topk_idx_TK = topk_idx_TK.masked_fill(topk_weights_TK == 0, -1)
        if topk_weights_TK.dtype != torch.float32:
            topk_weights_TK = topk_weights_TK.float()

        region = remat.region(
            _ops.dispatch_op,
            self.remat_region_name("ep_communication.dispatch"),
            recompute=recompute,
        )
        recv_x_RD, recv_scores_R, num_recv_per_expert_e, handle_id_1 = region(
            x_TD,
            topk_idx_TK,
            topk_weights_TK,
            num_experts=self.num_experts,
            num_tokens_per_rank=x_TD.shape[0],
        )
        # The saved region is skipped during replay; these outputs are still
        # needed by their original consumers.
        remat.recompute_needs_tensor(
            recv_x_RD, recv_scores_R, num_recv_per_expert_e, handle_id_1
        )

        num_tokens_per_expert_e = num_recv_per_expert_e.to(recv_x_RD.device)
        state = DispatchState(handle_id_1=handle_id_1, recv_scores_R=recv_scores_R)
        return recv_x_RD, num_tokens_per_expert_e, state

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DispatchState,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Apply routing scores, then reduce expert outputs to source tokens."""
        # Required by the dispatcher interface; EPBuffer.combine restores the
        # source order itself.
        del x_TD
        self._ensure_buffer()
        recompute = self._replays_communication()
        # The score multiply is read by the combine region with bare ops.
        remat.recompute_needs_tensor(routed_output_RD)

        will_backward = torch.is_grad_enabled()
        num_recv = routed_output_RD.shape[0]
        if num_recv == 0:
            # Keep the collective on empty receives so all EP ranks stay in sync.
            weighted_RD = routed_output_RD
        else:
            per_row_score_R1 = metadata.recv_scores_R.reshape(num_recv, -1).sum(
                dim=-1, keepdim=True
            )
            weighted_RD = routed_output_RD * per_row_score_R1.to(routed_output_RD.dtype)

        region = remat.region(
            _ops.combine_op,
            self.remat_region_name("ep_communication.combine"),
            recompute=recompute,
        )
        combined_TD = region(weighted_RD, metadata.handle_id_1, will_backward)
        # The caller consumes this output outside another remat region.
        remat.recompute_needs_tensor(combined_TD)
        return combined_TD


@override(
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description="DeepEP-Ascend expanded token dispatch/combine for MoE routed experts (EP>1)",
)
def npu_deepep(
    cfg: AllToAllTokenDispatcher.Config,
    *,
    num_max_tokens_per_rank: int,
    hidden_dim: int,
    expert_alignment: int = 1,
) -> AscendDeepEPTokenDispatcher.Config:
    """Replace the dispatcher node; the recipe supplies the capacity kwargs."""
    return derive(
        cfg,
        AscendDeepEPTokenDispatcher.Config,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden_dim=hidden_dim,
        expert_alignment=expert_alignment,
    )
