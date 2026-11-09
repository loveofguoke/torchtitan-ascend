# SPDX-License-Identifier: Apache-2.0
"""Persistent Triton attention residual with online softmax.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.kimi_k3.model import AttentionResidual

from torchtitan_ascend.device import get_num_vector_cores, require_device

__all__ = [
    "attention_residual",
    "TritonAttentionResidual",
    "triton_attn_res",
    "enable_triton_attn_res",
]

_eager_attention_residual = AttentionResidual(AttentionResidual.Config())


@triton.jit
def _forward(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    out_ptr,
    saved_score_ptr,
    saved_rstd_ptr,
    saved_max_ptr,
    saved_sumexp_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    TOKENS_PER_CORE: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tok0 = pid * TOKENS_PER_CORE
    tok1 = tl.minimum(tok0 + TOKENS_PER_CORE, T)

    cols = tl.arange(0, H)  # Full hidden dimension, including non-power-of-two widths
    tok_offs = tl.arange(0, SUB_BLOCK_SIZE)  # sub-block token offsets

    # Fused mul: score_weight = norm_w * proj_w (computed once, resident)
    norm_w = tl.load(norm_w_ptr + cols).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols).to(tl.float32)
    w = norm_w * proj_w

    br_stride = N * H  # block_residual row stride
    inv_H = 1.0 / H
    num_streams = N + HAS_PREFIX

    for tok_base in range(tok0, tok1, SUB_BLOCK_SIZE):
        cur_toks = tok_offs + tok_base  # [SUB_BLOCK_SIZE]
        tok_mask = cur_toks < tok1  # boundary mask
        mask2d = tl.expand_dims(tok_mask, 1)  # [SUB_BLOCK_SIZE, 1]

        # Online softmax: single-pass fused score + weighted sum (no stream reload)
        max_score = tl.full([SUB_BLOCK_SIZE], -float("inf"), dtype=tl.float32)
        exp_acc = tl.zeros([SUB_BLOCK_SIZE], dtype=tl.float32)
        out_acc = tl.zeros([SUB_BLOCK_SIZE, H], dtype=tl.float32)

        for s in range(N):
            ptrs = (
                block_residual_ptr
                + cur_toks[:, None] * br_stride
                + s * H
                + cols[None, :]
            )
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            vv_sum = tl.sum(v * v, axis=1)
            vw_sum = tl.sum(v * w[None, :], axis=1)
            rstd = tl.rsqrt(vv_sum * inv_H + EPS)
            score = vw_sum * rstd

            tl.store(saved_score_ptr + cur_toks * num_streams + s, score, mask=tok_mask)
            tl.store(saved_rstd_ptr + cur_toks * num_streams + s, rstd, mask=tok_mask)

            m_new = tl.maximum(max_score, score)
            correction = tl.exp(max_score - m_new)
            exp_score = tl.exp(score - m_new)
            out_acc = out_acc * correction[:, None] + exp_score[:, None] * v
            exp_acc = exp_acc * correction + exp_score
            max_score = m_new

        if HAS_PREFIX:
            # prefix_sum stream (s == N)
            ptrs = prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :]
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            vv_sum = tl.sum(v * v, axis=1)
            vw_sum = tl.sum(v * w[None, :], axis=1)
            rstd = tl.rsqrt(vv_sum * inv_H + EPS)
            score = vw_sum * rstd

            tl.store(saved_score_ptr + cur_toks * num_streams + N, score, mask=tok_mask)
            tl.store(saved_rstd_ptr + cur_toks * num_streams + N, rstd, mask=tok_mask)

            m_new = tl.maximum(max_score, score)
            correction = tl.exp(max_score - m_new)
            exp_score = tl.exp(score - m_new)
            out_acc = out_acc * correction[:, None] + exp_score[:, None] * v
            exp_acc = exp_acc * correction + exp_score
            max_score = m_new

        tl.store(saved_max_ptr + cur_toks, max_score, mask=tok_mask)
        tl.store(saved_sumexp_ptr + cur_toks, exp_acc, mask=tok_mask)

        out = out_acc / exp_acc[:, None]

        tl.store(
            out_ptr + cur_toks[:, None] * H + cols[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask2d,
        )


@triton.jit
def _backward(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    grad_out_ptr,
    grad_block_residual_ptr,
    grad_prefix_sum_ptr,
    grad_w_partials_ptr,
    saved_score_ptr,
    saved_rstd_ptr,
    saved_max_ptr,
    saved_sumexp_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOKENS_PER_CORE: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    NB: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tok0 = pid * TOKENS_PER_CORE
    tok1 = tl.minimum(tok0 + TOKENS_PER_CORE, T)

    cols = tl.arange(0, H)
    s_idx = tl.arange(0, NB)
    tok_offs = tl.arange(0, SUB_BLOCK_SIZE)

    norm_w = tl.load(norm_w_ptr + cols).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols).to(tl.float32)
    w = norm_w * proj_w

    br_stride = N * H
    inv_H = 1.0 / H
    num_streams = N + HAS_PREFIX
    grad_w_local = tl.zeros([H], dtype=tl.float32)

    for tok_base in range(tok0, tok1, SUB_BLOCK_SIZE):
        cur_toks = tok_offs + tok_base
        tok_mask = cur_toks < tok1
        mask2d = tl.expand_dims(tok_mask, 1)

        g = tl.load(
            grad_out_ptr + cur_toks[:, None] * H + cols[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)

        # Load saved softmax reduction values from forward (avoids recomputing
        # vv_sum, vw_sum, rstd, score, scores_max, sum_exp).
        scores_max = tl.load(
            saved_max_ptr + cur_toks, mask=tok_mask, other=-float("inf")
        )
        sum_exp = tl.load(saved_sumexp_ptr + cur_toks, mask=tok_mask, other=1.0)

        # Pass 1: compute gv_s only; use saved scores for weighted_gv accumulation.
        gv_arr = tl.full([NB, SUB_BLOCK_SIZE], 0.0, dtype=tl.float32)
        weighted_gv = tl.zeros([SUB_BLOCK_SIZE], dtype=tl.float32)

        for s in range(N):
            ptrs = (
                block_residual_ptr
                + cur_toks[:, None] * br_stride
                + s * H
                + cols[None, :]
            )
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            gv_s = tl.sum(g * v, axis=1)
            score_s = tl.load(
                saved_score_ptr + cur_toks * num_streams + s,
                mask=tok_mask,
                other=-float("inf"),
            )
            weighted_gv += tl.exp(score_s - scores_max) * gv_s
            gv_arr = tl.where(s_idx[:, None] == s, gv_s[None, :], gv_arr)

        if HAS_PREFIX:
            ptrs = prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :]
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            gv_s = tl.sum(g * v, axis=1)
            score_s = tl.load(
                saved_score_ptr + cur_toks * num_streams + N,
                mask=tok_mask,
                other=-float("inf"),
            )
            weighted_gv += tl.exp(score_s - scores_max) * gv_s
            gv_arr = tl.where(s_idx[:, None] == N, gv_s[None, :], gv_arr)

        weighted_gv = weighted_gv / sum_exp

        # Pass 2: compute gradients using saved rstd, score (no recompute).
        for s in range(N):
            ptrs = (
                block_residual_ptr
                + cur_toks[:, None] * br_stride
                + s * H
                + cols[None, :]
            )
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            rstd_s = tl.load(
                saved_rstd_ptr + cur_toks * num_streams + s, mask=tok_mask, other=0.0
            )
            score_s = tl.load(
                saved_score_ptr + cur_toks * num_streams + s, mask=tok_mask, other=0.0
            )
            gv_s = tl.sum(tl.where(s_idx[:, None] == s, gv_arr, 0.0), axis=0)

            k = v * rstd_s[:, None]
            prob = tl.exp(score_s - scores_max) / sum_exp
            delta = gv_s - weighted_gv
            dscore_dv = (
                rstd_s[:, None] * w[None, :]
                - inv_H * v * rstd_s[:, None] * rstd_s[:, None] * score_s[:, None]
            )
            grad_v = prob[:, None] * (g + delta[:, None] * dscore_dv)

            tl.store(
                grad_block_residual_ptr
                + cur_toks[:, None] * br_stride
                + s * H
                + cols[None, :],
                grad_v.to(grad_block_residual_ptr.dtype.element_ty),
                mask=mask2d,
            )
            grad_w_local += tl.sum(prob[:, None] * delta[:, None] * k, axis=0)

        if HAS_PREFIX:
            ptrs = prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :]
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            rstd_s = tl.load(
                saved_rstd_ptr + cur_toks * num_streams + N, mask=tok_mask, other=0.0
            )
            score_s = tl.load(
                saved_score_ptr + cur_toks * num_streams + N, mask=tok_mask, other=0.0
            )
            gv_s = tl.sum(tl.where(s_idx[:, None] == N, gv_arr, 0.0), axis=0)

            k = v * rstd_s[:, None]
            prob = tl.exp(score_s - scores_max) / sum_exp
            delta = gv_s - weighted_gv
            dscore_dv = (
                rstd_s[:, None] * w[None, :]
                - inv_H * v * rstd_s[:, None] * rstd_s[:, None] * score_s[:, None]
            )
            grad_v = prob[:, None] * (g + delta[:, None] * dscore_dv)

            tl.store(
                grad_prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :],
                grad_v.to(grad_prefix_sum_ptr.dtype.element_ty),
                mask=mask2d,
            )
            grad_w_local += tl.sum(prob[:, None] * delta[:, None] * k, axis=0)

    tl.store(grad_w_partials_ptr + pid * H + cols, grad_w_local)


class _AttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, partial_block_TD, block_residual_TND, gamma, project, eps, num_cores
    ):
        blocks_TND = block_residual_TND.contiguous()
        prefix_TD = (
            partial_block_TD.contiguous() if partial_block_TD is not None else None
        )
        gamma, project = gamma.contiguous(), project.contiguous()
        num_tokens, num_blocks, dim = blocks_TND.shape
        has_prefix = prefix_TD is not None
        num_streams = num_blocks + has_prefix
        dtype = (
            torch.promote_types(blocks_TND.dtype, prefix_TD.dtype)
            if prefix_TD is not None
            else blocks_TND.dtype
        )
        device = blocks_TND.device
        output_TD = torch.empty((num_tokens, dim), device=device, dtype=dtype)
        scores_TN = torch.empty(
            (num_tokens, num_streams), device=device, dtype=torch.float32
        )
        rstd_TN = torch.empty_like(scores_TN)
        max_score_T = torch.empty((num_tokens,), device=device, dtype=torch.float32)
        sum_exp_T = torch.empty_like(max_score_T)

        tokens_per_core = triton.cdiv(num_tokens, num_cores)
        grid = (triton.cdiv(num_tokens, tokens_per_core),)
        _forward[grid](
            blocks_TND,
            prefix_TD if has_prefix else blocks_TND,
            gamma,
            project,
            output_TD,
            scores_TN,
            rstd_TN,
            max_score_T,
            sum_exp_T,
            T=num_tokens,
            H=dim,
            N=num_blocks,
            EPS=eps,
            TOKENS_PER_CORE=tokens_per_core,
            HAS_PREFIX=has_prefix,
            SUB_BLOCK_SIZE=1,
            multibuffer=True,
        )
        ctx.save_for_backward(
            blocks_TND,
            prefix_TD,
            gamma,
            project,
            scores_TN,
            rstd_TN,
            max_score_T,
            sum_exp_T,
        )
        ctx.tokens_per_core = tokens_per_core
        return output_TD

    @staticmethod
    def backward(ctx, grad_out_TD):
        (
            blocks_TND,
            prefix_TD,
            gamma,
            project,
            scores_TN,
            rstd_TN,
            max_score_T,
            sum_exp_T,
        ) = ctx.saved_tensors
        num_tokens, num_blocks, dim = blocks_TND.shape
        has_prefix = prefix_TD is not None
        num_programs = triton.cdiv(num_tokens, ctx.tokens_per_core)
        grad_blocks_TND = torch.empty_like(blocks_TND)
        grad_prefix_TD = torch.empty_like(prefix_TD) if has_prefix else None
        grad_weight_CD = torch.empty(
            (num_programs, dim), device=blocks_TND.device, dtype=torch.float32
        )
        # FP32 streams and single-token loops can exhaust the Ascend UB when
        # double buffering is enabled.
        use_multibuffer = grad_out_TD.dtype in (torch.bfloat16, torch.float16) and (
            ctx.tokens_per_core > 1
        )
        _backward[(num_programs,)](
            blocks_TND,
            prefix_TD if has_prefix else blocks_TND,
            gamma,
            project,
            grad_out_TD.contiguous(),
            grad_blocks_TND,
            grad_prefix_TD if has_prefix else grad_blocks_TND,
            grad_weight_CD,
            scores_TN,
            rstd_TN,
            max_score_T,
            sum_exp_T,
            T=num_tokens,
            H=dim,
            N=num_blocks,
            TOKENS_PER_CORE=ctx.tokens_per_core,
            HAS_PREFIX=has_prefix,
            NB=max(triton.next_power_of_2(num_blocks + has_prefix), 8),
            SUB_BLOCK_SIZE=1,
            multibuffer=use_multibuffer,
        )
        # Each program writes a separate partial, so no floating-point atomics
        # are needed for the weight gradient.
        grad_weight_D = grad_weight_CD.sum(0)
        return (
            grad_prefix_TD,
            grad_blocks_TND,
            (grad_weight_D * project.float()).to(gamma.dtype),
            (grad_weight_D * gamma.float()).to(project.dtype),
            None,
            None,
        )


def _num_attn_res_cores() -> int:
    require_device("Ascend950", feature="Triton attention residual")
    return get_num_vector_cores()


def attention_residual(
    partial_block_TD, block_residual_TND, projection, norm, *, num_cores=None
):
    num_tokens, num_blocks, _ = block_residual_TND.shape
    if not num_tokens or (num_blocks == 0 and partial_block_TD is None):
        return _eager_attention_residual(
            partial_block_TD, block_residual_TND, projection, norm
        )
    if norm.eps is None or norm.weight is None:
        raise ValueError(
            "Attention residual requires weighted RMSNorm and explicit epsilon"
        )
    return _AttentionResidual.apply(
        partial_block_TD,
        block_residual_TND,
        norm.weight,
        projection.weight.squeeze(0),
        norm.eps,
        _num_attn_res_cores() if num_cores is None else num_cores,
    )


class TritonAttentionResidual(AttentionResidual):
    @dataclass(kw_only=True, slots=True)
    class Config(AttentionResidual.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_cores = _num_attn_res_cores()

    def __call__(self, partial_block_TD, block_residual_TND, projection, norm):
        return attention_residual(
            partial_block_TD,
            block_residual_TND,
            projection,
            norm,
            num_cores=self.num_cores,
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
