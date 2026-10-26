# SPDX-License-Identifier: Apache-2.0
"""Autograd and fake-tensor boundary for the temporary varlen_attn patch.

Import only when applying the patch; no TorchTitan model imports are needed.
"""

from itertools import pairwise

import torch

_INT_MAX = 2147483647


def _ends(cu, total, maximum):
    offsets = cu.cpu().tolist()
    lengths = [b - a for a, b in pairwise(offsets)]
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != total or any(n <= 0 or n > maximum for n in lengths):
        raise ValueError("NPU varlen_attn requires positive sequence lengths covering all tokens within max_q/max_k")
    return offsets[1:]


def _sequence_ends(q, k, cu_q, cu_k, max_q, max_k):
    ends_q = _ends(cu_q, q.shape[0], max_q)
    ends_k = ends_q if cu_q is cu_k and q.shape[0] == k.shape[0] and max_q == max_k else _ends(cu_k, k.shape[0], max_k)
    return ends_q, ends_k


def _mask_args(q, left, right):
    mode = 4 if left >= 0 else (3 if right == 0 else 0)
    # Construct on the caller's stream; no cross-stream cached-tensor dependency.
    mask = None if mode == 0 else torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu(1)
    return {
        "atten_mask": mask,
        "sparse_mode": mode,
        "pre_tockens": _INT_MAX if left == -1 else left,
        "next_tockens": _INT_MAX if right == -1 else 0,
        "gen_mask_parallel": False,
        "sync": True,
    }


def _lse(smax, ssum, heads, ends):
    # Legacy TND softmax statistics are head-major within each document.
    values = smax.reshape(-1, 8)[:, 0] + ssum.reshape(-1, 8)[:, 0].log()
    pieces, start = [], 0
    for end in ends:
        pieces.append(values[start * heads : end * heads].reshape(heads, end - start))
        start = end
    return torch.cat(pieces, dim=1)


@torch.library.custom_op("torchtitan_ascend::varlen_attention", mutates_args=())
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
    left: int,
    right: int,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch_npu

    if not torch.backends.cuda.flash_sdp_enabled():
        raise NotImplementedError("NPU varlen_attn patch requires FLASH_ATTENTION enabled")
    ends_q, ends_k = _sequence_ends(q, k, cu_q, cu_k, max_q, max_k)
    if len(ends_q) != len(ends_k):
        raise ValueError("NPU varlen_attn Q/K must have the same number of sequences")
    out, smax, ssum, *_ = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        "TND",
        scale=scale,
        actual_seq_qlen=ends_q,
        actual_seq_kvlen=ends_k,
        **_mask_args(q, left, right),
    )
    lse = _lse(smax, ssum, q.shape[1], ends_q) if return_lse else smax.new_empty((0,))
    return out, lse, smax, ssum


@attention.register_fake
def _(q, k, v, cu_q, cu_k, max_q, max_k, scale, left, right, return_lse=True):
    return (
        q.new_empty((q.shape[0], q.shape[1], v.shape[-1])),
        q.new_empty((q.shape[1], q.shape[0]) if return_lse else (0,), dtype=torch.float32),
        q.new_empty((*q.shape[:2], 8), dtype=torch.float32),
        q.new_empty((*q.shape[:2], 8), dtype=torch.float32),
    )


@torch.library.custom_op("torchtitan_ascend::varlen_attention_backward", mutates_args=())
def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    smax: torch.Tensor,
    ssum: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
    left: int,
    right: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch_npu

    ends_q, ends_k = _sequence_ends(q, k, cu_q, cu_k, max_q, max_k)
    dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        grad_out.contiguous(),
        q.shape[1],
        "TND",
        scale_value=scale,
        softmax_max=smax,
        softmax_sum=ssum,
        attention_in=out,
        actual_seq_qlen=ends_q,
        actual_seq_kvlen=ends_k,
        **_mask_args(q, left, right),
    )
    return dq, dk, dv


@backward.register_fake
def _(q, k, v, out, grad_out, smax, ssum, cu_q, cu_k, max_q, max_k, scale, left, right):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, cu_q, cu_k, *ctx.params = inputs[:-1]
    out, lse, smax, ssum = output
    ctx.save_for_backward(q, k, v, out, smax, ssum, cu_q, cu_k)
    ctx.mark_non_differentiable(lse, smax, ssum)


def _backward(ctx, grad_out, _dlse, _dsmax, _dssum):
    q, k, v, out, smax, ssum, cu_q, cu_k = ctx.saved_tensors
    grads = backward(q, k, v, out, grad_out, smax, ssum, cu_q, cu_k, *ctx.params)
    return (*grads, None, None, None, None, None, None, None, None)


attention.register_autograd(_backward, setup_context=_setup_context)
