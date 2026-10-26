# SPDX-License-Identifier: Apache-2.0
"""Temporary NPU varlen_attn bridge (NPU-1 / Ascend/pytorch #4439).

Upstream implementation: https://gitcode.com/Ascend/op-plugin/merge_requests/6079
Remove once the supported dependency versions provide native ATen FlashAttention
and the public varlen_attn forward/backward pass the same NPU tests.
This replaces the Python API only; it does not register or replace ATen kernels.
"""

from functools import wraps

from torchtitan_ascend.patches import patch


@patch(target="torch.nn.attention.varlen:varlen_attn", kind="replace")
def apply(original):
    import torch

    from ._attention import attention

    @wraps(original)
    def varlen_attn(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        *,
        return_aux=None,
        scale=None,
        window_size=(-1, -1),
        enable_gqa=False,
        seqused_k=None,
        block_table=None,
        num_splits=None,
    ):
        if seqused_k is not None or block_table is not None:
            raise NotImplementedError("NPU varlen_attn patch does not support seqused_k or block_table")
        # The CANN training path does not split KV across separate attention calls.
        if num_splits not in (None, 1):
            raise NotImplementedError("NPU varlen_attn patch only supports num_splits=None or 1")
        if any(t.dim() != 3 for t in (query, key, value)):
            raise ValueError("NPU varlen_attn requires packed 3D Q/K/V tensors")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("NPU varlen_attn requires float16 or bfloat16")
        if any(t.dtype != query.dtype or t.device != query.device for t in (key, value)):
            raise ValueError("NPU varlen_attn requires matching Q/K/V dtype and device")
        if key.shape[:2] != value.shape[:2] or query.shape[-1] != key.shape[-1]:
            raise ValueError("NPU varlen_attn requires compatible Q/K/V shapes")
        hq, hk = query.shape[1], key.shape[1]
        if hk == 0 or hq == 0 or hq % hk or (hq != hk and not enable_gqa):
            raise ValueError("NPU varlen_attn requires enable_gqa=True for grouped query heads")
        for cu in (cu_seq_q, cu_seq_k):
            if cu is None or cu.ndim != 1 or cu.dtype not in (torch.int32, torch.int64):
                raise ValueError("NPU varlen_attn requires 1D integer cumulative lengths")
            if cu.device != query.device:
                raise ValueError("NPU varlen_attn cumulative lengths must be on the query device")
        left, right = window_size
        if left < -1 or right not in (-1, 0):
            raise NotImplementedError("NPU varlen_attn supports full, causal and left-window attention")
        return_lse = return_aux is not None and return_aux.lse
        out, lse, _, _ = attention(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            query.shape[-1] ** -0.5 if scale is None else scale,
            left,
            right,
            return_lse,
        )
        return (out, lse) if return_lse else out

    return varlen_attn
