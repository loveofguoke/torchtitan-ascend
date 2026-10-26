# Copyright (c) 2026 Tianjin University
# SPDX-License-Identifier: BSD-3-Clause
"""Packed causal convolution shared by GDN and KDA."""

import torch
import torch.nn.functional as functional


class _CausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, offsets):
        from fla_npu.ops import ascendc

        y = ascendc.causal_conv1d_fn(
            x,
            weight,
            None,
            query_start_loc_cpu=offsets,
            activation=None,
        )
        ctx.save_for_backward(x, weight)
        ctx.offsets = offsets
        return y

    @staticmethod
    def backward(ctx, grad_output):
        from fla_npu.ops import ascendc

        x, weight = ctx.saved_tensors
        dx, dw, *_ = ascendc.causal_conv1d_bwd(
            x,
            None,
            weight,
            grad_output.contiguous(),
            query_start_loc=ctx.offsets,
            activation=0,
            input_layout="TND",
        )
        return dx, dw, None


def causal_conv1d(x, weight, cu_seqlens):
    offsets = cu_seqlens.tolist() if cu_seqlens is not None else [0, x.shape[0]]
    weight = weight[:, 0].T.to(x.dtype).contiguous()
    return functional.silu(_CausalConv1d.apply(x.contiguous(), weight, offsets))
