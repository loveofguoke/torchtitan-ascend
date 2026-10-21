# Copyright (c) 2026 Tianjin University
# SPDX-License-Identifier: BSD-3-Clause
# Operator composition adapted from flash-linear-attention-npu's GDN example.
# See LICENSE-BSD-3-Clause.

"""TorchTitan GDN adapters for fla-npu's AscendC training operators."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.config import derive, override
from torchtitan.models.common.attention.gdn import GDNKernel, InnerGDN

from torchtitan_ascend.device import require_device


def _chunk_indices(offsets, chunk_size):
    if offsets is None:
        return None
    return [
        index
        for seq, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
        for chunk in range((end - begin + chunk_size - 1) // chunk_size)
        for index in (seq, chunk)
    ]


class _ChunkGDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, offsets):
        from fla_npu.ops import ascendc

        # The AscendC training operators require 64-token chunks.
        ctx.options = dict(
            layout="BSND",
            chunk_size=64,
            cu_seqlens=offsets,
            chunk_indices=_chunk_indices(offsets, 64),
            use_qk_l2norm_in_kernel=True,
        )
        ctx.scale = q.shape[-1] ** -0.5
        out, _, cumulative_g, inverse, _, _, q_hat, k_hat, q_rstd, k_rstd = (
            ascendc.chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale=ctx.scale, **ctx.options)
        )
        ctx.save_for_backward(q_hat, k_hat, v, cumulative_g, beta, inverse, q_rstd, k_rstd)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from fla_npu.ops import ascendc

        q, k, v, g, beta, inverse, q_rstd, k_rstd = ctx.saved_tensors
        dq, dk, dv, dbeta, dg, *_ = ascendc.chunk_gated_delta_rule_bwd(
            q,
            k,
            v,
            g,
            beta,
            inverse,
            grad_output.contiguous(),
            ctx.scale,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
            **ctx.options,
        )
        return dq, dk, dv, dg, dbeta, None


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


class FlaGDNKernel(GDNKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(GDNKernel.Config):
        pass

    def __init__(self, config: Config):
        require_device("Ascend950", feature="fla-npu GDN")
        super().__init__(config)

    def forward(self, q, k, v, g, beta, *, cu_seqlens=None):
        offsets = cu_seqlens.tolist() if cu_seqlens is not None else None
        tensors = [t.unsqueeze(0).contiguous() for t in (q, k, v, g.float(), beta.float())]
        return _ChunkGDN.apply(*tensors, offsets).squeeze(0)


class FlaInnerGDN(InnerGDN):
    @dataclass(kw_only=True, slots=True)
    class Config(InnerGDN.Config):
        pass

    def __init__(self, config: Config):
        require_device("Ascend950", feature="fla-npu GDN convolution")
        super().__init__(config)

    def short_convolution(self, x, weight, initial_state=None, *, cu_seqlens, routing):
        if initial_state is not None or routing is not None:
            raise NotImplementedError("fla-npu convolution does not support context-parallel state")
        offsets = cu_seqlens.tolist() if cu_seqlens is not None else [0, x.shape[0]]
        weight = weight[:, 0].T.to(x.dtype).contiguous()
        return F.silu(_CausalConv1d.apply(x.contiguous(), weight, offsets))


@override(
    target=InnerGDN.Config, exact=True, description="GDN training with fla-npu AscendC operators"
)
def npu_gated_delta_net(cfg: InnerGDN.Config) -> FlaInnerGDN.Config:
    new = derive(cfg, FlaInnerGDN.Config)
    new.kernel = derive(cfg.kernel, FlaGDNKernel.Config)
    return new
