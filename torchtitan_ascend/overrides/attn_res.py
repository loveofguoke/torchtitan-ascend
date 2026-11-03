# SPDX-License-Identifier: Apache-2.0
"""Opt-in Triton attention residual using MindSpeed-MM's FP32 formula.

No atomics are used. Weight gradients reduce per-token contributions with
PyTorch's deterministic reduction; token-local work is fused in Triton.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.kimi_k3.model import AttentionResidual

__all__ = [
    "attention_residual",
    "TritonAttentionResidual",
    "triton_attn_res",
    "enable_triton_attn_res",
]

_eager_attention_residual = AttentionResidual(AttentionResidual.Config())


@triton.jit
def _scores(
    V,
    Gamma,
    Project,
    Rstd,
    Scores,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    v = tl.load(V + row * D + d, d < D, 0).to(tl.float32)
    gamma = tl.load(Gamma + d, d < D, 0).to(tl.float32)
    project = tl.load(Project + d, d < D, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(v * v, 0) / D + EPS)
    score = tl.sum((v * rstd) * (gamma * project), 0)
    tl.store(Rstd + row, rstd)
    tl.store(Scores + row, score)


@triton.jit
def _mix(
    V,
    Scores,
    Prob,
    Out,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    t, tile = tl.program_id(0), tl.program_id(1)
    n = tl.arange(0, BN)
    d = tile * BD + tl.arange(0, BD)
    scores = tl.load(Scores + t * N + n, n < N, -float("inf"))
    exp_scores = tl.exp(scores - tl.max(scores, 0))
    prob = exp_scores / tl.sum(exp_scores, 0)
    v = tl.load(
        V + (t * N + n[:, None]) * D + d[None, :],
        (n[:, None] < N) & (d[None, :] < D),
        0,
    ).to(tl.float32)
    output = tl.sum(prob[:, None] * v, 0)
    tl.store(Out + t * D + d, output, d < D)
    if tile == 0:
        tl.store(Prob + t * N + n, prob, n < N)


@triton.jit
def _value_dot(V, DY, Dots, N: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    t = row // N
    d = tl.arange(0, BD)
    v = tl.load(V + row * D + d, d < D, 0).to(tl.float32)
    dy = tl.load(DY + t * D + d, d < D, 0).to(tl.float32)
    tl.store(Dots + row, tl.sum(v * dy, 0))


@triton.jit
def _grad_values(
    V,
    Gamma,
    Project,
    DY,
    Rstd,
    Scores,
    Prob,
    Dots,
    DV,
    DW,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    t, tile = tl.program_id(0), tl.program_id(1)
    n = tl.arange(0, BN)
    d = tile * BD + tl.arange(0, BD)
    mask = (n[:, None] < N) & (d[None, :] < D)
    v = tl.load(V + (t * N + n[:, None]) * D + d[None, :], mask, 0).to(tl.float32)
    dy = tl.load(DY + t * D + d, d < D, 0).to(tl.float32)
    gamma = tl.load(Gamma + d, d < D, 0).to(tl.float32)
    project = tl.load(Project + d, d < D, 0).to(tl.float32)
    prob = tl.load(Prob + t * N + n, n < N, 0)
    dots = tl.load(Dots + t * N + n, n < N, 0)
    rstd = tl.load(Rstd + t * N + n, n < N, 0)
    scores = tl.load(Scores + t * N + n, n < N, 0)
    ds = prob * (dots - tl.sum(prob * dots, 0))
    dv = prob[:, None] * dy[None, :] + (ds * rstd)[:, None] * (
        (gamma * project)[None, :] - v * (rstd * scores / D)[:, None]
    )
    dw = tl.sum((ds * rstd)[:, None] * v, 0)
    tl.store(DV + (t * N + n[:, None]) * D + d[None, :], dv, mask)
    tl.store(DW + t * D + d, dw, d < D)


class _AttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, gamma, project, eps):
        values = values.contiguous()
        gamma, project = gamma.contiguous(), project.contiguous()
        tokens, blocks, dim = values.shape
        rstd = torch.empty((tokens, blocks), device=values.device, dtype=torch.float32)
        scores, prob = torch.empty_like(rstd), torch.empty_like(rstd)
        output = torch.empty((tokens, dim), device=values.device, dtype=values.dtype)
        block_dim = triton.next_power_of_2(dim)
        block_count = triton.next_power_of_2(blocks)
        _scores[(tokens * blocks,)](
            values, gamma, project, rstd, scores, blocks, dim, eps, block_dim
        )
        _mix[(tokens, triton.cdiv(dim, 128))](
            values, scores, prob, output, blocks, dim, block_count, 128
        )
        ctx.save_for_backward(values, gamma, project, rstd, scores, prob)
        return output

    @staticmethod
    def backward(ctx, dy):
        values, gamma, project, rstd, scores, prob = ctx.saved_tensors
        tokens, blocks, dim = values.shape
        dy = dy.contiguous()
        dots = torch.empty_like(scores)
        dv = torch.empty_like(values)
        dw = torch.empty((tokens, dim), device=values.device, dtype=torch.float32)
        _value_dot[(tokens * blocks,)](
            values, dy, dots, blocks, dim, triton.next_power_of_2(dim)
        )
        # The Ascend graph pass corrupts the parameter-gradient reduction.
        _grad_values[(tokens, triton.cdiv(dim, 128))](
            values,
            gamma,
            project,
            dy,
            rstd,
            scores,
            prob,
            dots,
            dv,
            dw,
            blocks,
            dim,
            triton.next_power_of_2(blocks),
            128,
            enable_graph_optimize=False,
        )
        dw = dw.sum(0)
        return (
            dv,
            (dw * project.float()).to(gamma.dtype),
            (dw * gamma.float()).to(project.dtype),
            None,
        )


def attention_residual(partial_block_TD, block_residual_TND, projection, norm):
    values = (
        block_residual_TND
        if partial_block_TD is None
        else torch.cat((block_residual_TND, partial_block_TD.unsqueeze(1)), dim=1)
    )
    if values.shape[1] == 0 or values.shape[0] == 0:
        return _eager_attention_residual(
            partial_block_TD, block_residual_TND, projection, norm
        )
    if norm.eps is None or norm.weight is None:
        raise ValueError(
            "Attention residual requires weighted RMSNorm and explicit epsilon"
        )
    return _AttentionResidual.apply(
        values, norm.weight, projection.weight.squeeze(0), norm.eps
    )


class TritonAttentionResidual(AttentionResidual):
    @dataclass(kw_only=True, slots=True)
    class Config(AttentionResidual.Config):
        pass

    def __call__(self, partial_block_TD, block_residual_TND, projection, norm):
        return attention_residual(
            partial_block_TD, block_residual_TND, projection, norm
        )


@override(
    target=AttentionResidual.Config,
    exact=True,
    description="Triton attention residual on Ascend",
)
def triton_attn_res(config):
    return derive(config, TritonAttentionResidual.Config)


def enable_triton_attn_res():
    """Return the override target for explicit activation in a training config."""
    return f"{triton_attn_res.__module__}.{triton_attn_res.__name__}"
