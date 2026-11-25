# SPDX-License-Identifier: Apache-2.0
"""DeepEP-Ascend dispatch/combine custom ops.

Dispatch backward is a combine of the gradients; combine backward is a cached
dispatch reusing the original handle. The opaque EPHandle crosses the
tensor-only op boundary through a CPU int64 handle-id and the runtime cache.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from torchtitan_ascend.overrides.deepep import runtime

_lib = torch.library.Library("torchtitan_ascend", "DEF")

# recv_scores_R is the per-received-row routing weight [R]; num_recv_per_expert_e
# is the per-local-expert count [e]. Ascend dispatch is expand-only, so no
# recv_topk_idx is returned.
_lib.define(
    "deepep_dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "int num_experts, int num_tokens_per_rank) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
# will_backward is the caller's outer grad state: autograd disables grad inside
# the op forward, so it is the only signal deciding who frees the handle.
_lib.define("deepep_combine(Tensor x, Tensor handle_id, bool will_backward) -> Tensor")

# Ordered effects so selective checkpointing saves both ops instead of replaying
# them; a replay would need a deterministic buffer (enforced in experts.py).
_lib._register_effectful_op(
    "torchtitan_ascend::deepep_dispatch", torch.library.EffectType.ORDERED
)
_lib._register_effectful_op(
    "torchtitan_ascend::deepep_combine", torch.library.EffectType.ORDERED
)

# Op handles used by the dispatcher; importing this module registers them.
dispatch_op = torch.ops.torchtitan_ascend.deepep_dispatch
combine_op = torch.ops.torchtitan_ascend.deepep_combine


@torch.library.impl(_lib, "deepep_dispatch", "PrivateUse1")
def _dispatch_impl(
    x_TD: torch.Tensor,
    topk_idx_TK: torch.Tensor,
    topk_weights_TK: torch.Tensor,
    num_experts: int,
    num_tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expanded dispatch on the current stream, host-synced."""
    entry = runtime._ACTIVE_BUFFER
    assert entry is not None, "EPBuffer must be initialized before dispatch"
    buffer = entry.buffer

    # Immediate mode returns the completed tuple, not the EventOverlap of a
    # deferred epilogue; narrow the public union at this boundary.
    dispatched = buffer.dispatch(
        x_TD,
        topk_idx=topk_idx_TK,
        topk_weights=topk_weights_TK,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_tokens_per_rank,
        expert_alignment=1,
        do_expand=True,
        do_zero_padding=True,
        do_cpu_sync=True,
    )
    recv_x_RD, _recv_topk_idx, recv_scores_R, handle, _event = cast(
        tuple[torch.Tensor, Any, torch.Tensor, Any, Any], dispatched
    )
    handle_id_1 = runtime.register_handle(entry, handle)
    # Per-local-expert unaligned counts stay on device for the grouped-GEMM offsets.
    num_recv_per_expert_e = handle.num_unaligned_recv_tokens_per_expert.to(torch.int32)
    return recv_x_RD, recv_scores_R, num_recv_per_expert_e, handle_id_1


def _dispatch_setup_context(ctx, inputs, output):
    x_TD, *_ = inputs
    *_, handle_id_1 = output
    ctx.input_dtype = x_TD.dtype
    entry = runtime._ACTIVE_BUFFER
    assert entry is not None, "EPBuffer must be initialized before dispatch"
    ctx.entry = entry
    ctx.saved_handle = entry.handles.get(handle_id_1.item())


def _dispatch_backward(
    ctx,
    grad_recv_x_RD,
    grad_recv_scores_R,
    _grad_num_recv_per_expert_e,
    _grad_handle_id_1,
):
    """A combine of the gradients: reduces grad_x and the routing-score grads."""
    if grad_recv_x_RD is None:
        return None, None, None, None, None
    entry = ctx.entry
    buffer = entry.buffer
    handle = ctx.saved_handle
    assert handle is not None

    combined = buffer.combine(
        grad_recv_x_RD,
        handle=handle,
        topk_weights=(
            grad_recv_scores_R.float() if grad_recv_scores_R is not None else None
        ),
    )
    grad_x_TD, grad_topk_weights_TK, _event = cast(
        tuple[torch.Tensor, torch.Tensor | None, Any], combined
    )
    grad_x_TD = grad_x_TD.to(ctx.input_dtype)
    # Keep the combine's FP32 reduction of the router-score gradients; the
    # autograd graph casts back to the original score dtype.
    return grad_x_TD, None, grad_topk_weights_TK, None, None


@torch.library.impl(_lib, "deepep_combine", "PrivateUse1")
def _combine_impl(
    x_RD: torch.Tensor, handle_id_1: torch.Tensor, will_backward: bool
) -> torch.Tensor:
    """Pure reduction; the routing scores were applied upstream."""
    entry = runtime._ACTIVE_BUFFER
    assert entry is not None, "EPBuffer must be initialized before combine"
    buffer = entry.buffer

    # Without a backward the handle is freed here; with one, setup_context
    # transfers it to the combine backward.
    handle = runtime.take_handle(entry, handle_id_1, consume=not will_backward)
    combined = buffer.combine(x_RD, handle=handle, topk_weights=None)
    combined_TD, _combined_weights, _event = cast(
        tuple[torch.Tensor, Any, Any], combined
    )
    return combined_TD


def _combine_setup_context(ctx, inputs, output):
    _, handle_id_1, _will_backward = inputs
    entry = runtime._ACTIVE_BUFFER
    assert entry is not None, "EPBuffer must be initialized before combine"
    ctx.entry = entry
    ctx.saved_handle = runtime.take_handle(entry, handle_id_1, consume=True)


def _combine_backward(ctx, grad_combined_TD):
    """A cached dispatch of the gradient, reusing the forward handle's layout."""
    entry = ctx.entry
    buffer = entry.buffer
    handle = ctx.saved_handle
    assert handle is not None

    # Cached dispatch reuses topk_idx from the handle and skips the host sync.
    dispatched = buffer.dispatch(
        grad_combined_TD, handle=handle, do_expand=True, do_cpu_sync=False
    )
    grad_x_RD, _idx, _scores, _handle, _event = cast(
        tuple[torch.Tensor, Any, Any, Any, Any], dispatched
    )
    return grad_x_RD, None, None


torch.library.register_autograd(
    "torchtitan_ascend::deepep_dispatch",
    _dispatch_backward,
    setup_context=_dispatch_setup_context,
)
torch.library.register_autograd(
    "torchtitan_ascend::deepep_combine",
    _combine_backward,
    setup_context=_combine_setup_context,
)
