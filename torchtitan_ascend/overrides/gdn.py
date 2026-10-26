# Copyright (c) 2026 Tianjin University
# SPDX-License-Identifier: BSD-3-Clause
# Operator composition adapted from flash-linear-attention-npu's GDN example.
# See LICENSE-BSD-3-Clause.

"""TorchTitan GDN adapters for fla-npu's AscendC training operators."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torchtitan.config import derive, override
from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel, InnerGatedDeltaNet

from torchtitan_ascend.device import require_device
from torchtitan_ascend.models.common.linear_attention import causal_conv1d


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
        out, _, cumulative_g, inverse, _, _, q_hat, k_hat, q_rstd, k_rstd = ascendc.chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale=ctx.scale, **ctx.options
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


class FlaGDNKernel(GatedDeltaKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        pass

    def __init__(self, config: Config):
        require_device("Ascend950", feature="fla-npu GDN")
        super().__init__(config)

    def forward(self, q, k, v, g, beta, *, cu_seqlens=None):
        offsets = cu_seqlens.tolist() if cu_seqlens is not None else None
        tensors = [t.unsqueeze(0).contiguous() for t in (q, k, v, g.float(), beta.float())]
        return _ChunkGDN.apply(*tensors, offsets).squeeze(0)


class FlaInnerGDN(InnerGatedDeltaNet):
    kernel: torch.nn.Module

    @dataclass(kw_only=True, slots=True)
    class Config(InnerGatedDeltaNet.Config):
        pass

    def __init__(self, config: Config):
        require_device("Ascend950", feature="fla-npu GDN convolution")
        super().__init__(config)

    def forward(
        self,
        query_TC,
        key_TC,
        value_TC,
        a_TH,
        b_TH,
        conv_q_weight_C1W,
        conv_k_weight_C1W,
        conv_v_weight_C1W,
        A_log_H,
        dt_bias_H,
        cu_seqlens,
        *,
        key_head_dim,
        value_head_dim,
    ):
        q = causal_conv1d(query_TC, conv_q_weight_C1W, cu_seqlens).unflatten(-1, (-1, key_head_dim))
        k = causal_conv1d(key_TC, conv_k_weight_C1W, cu_seqlens).unflatten(-1, (-1, key_head_dim))
        v = causal_conv1d(value_TC, conv_v_weight_C1W, cu_seqlens).unflatten(-1, (-1, value_head_dim))
        g = -torch.exp(A_log_H.float()) * functional.softplus(a_TH.float() + dt_bias_H)
        beta = torch.sigmoid(b_TH)
        return self.kernel(q, k, v, g, beta, cu_seqlens=cu_seqlens)


@override(target=InnerGatedDeltaNet.Config, exact=True, description="GDN training with fla-npu AscendC operators")
def npu_gated_delta_net(cfg: InnerGatedDeltaNet.Config) -> FlaInnerGDN.Config:
    new = derive(cfg, FlaInnerGDN.Config)
    new.kernel = derive(cfg.kernel, FlaGDNKernel.Config)
    return new
