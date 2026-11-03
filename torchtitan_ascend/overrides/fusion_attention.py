# SPDX-License-Identifier: Apache-2.0
"""Opt-in Ascend C attention using torch_npu's forward and backward."""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from torchtitan.config import derive, override
from torchtitan.models.common.attention import FlexInnerAttention

__all__ = [
    "AscendCFlexInnerAttention",
    "fusion_attention",
    "ascendc_fusion_attention",
]

# T = packed tokens, H = heads, K = query/key head dim, V = value head dim.


def _get_attn_mask(attention_metadata, q_THK, k_THK):
    """Share dense masks between layers using the same immutable batch metadata.

    A cache belongs to its BlockMask, so new packed documents get a new cache
    and finished batches release their masks. Shape alone does not identify a
    document mask. Callers must rebuild metadata when captured positions change.
    """
    mask_mod = attention_metadata.mask_mod
    cached_mod, masks = getattr(attention_metadata, "_ascend_dense_masks", (None, {}))
    if cached_mod is not mask_mod:
        masks = {}
        attention_metadata._ascend_dense_masks = (mask_mod, masks)
    key = (q_THK.device, q_THK.shape[1], q_THK.shape[0], k_THK.shape[0])
    if key not in masks:
        # Flex uses True for visible entries; CANN uses True for masked entries.
        masks[key] = (~create_mask(mask_mod, 1, *key[1:], device=key[0])).contiguous()
    return masks[key]


class _OffloadedFusionAttention(torch.autograd.Function):
    """Save FA output and softmax statistics in CPU memory for backward.

    Mirrors MindSpeed-MM's skip_recompute_flash_attention saved intermediates.
    Q/K/V stay on device. Blocking transfers keep ownership and stream ordering
    explicit without depending on MM's global layer/recompute manager.
    """

    @staticmethod
    def forward(ctx, q_BHTK, k_BHSK, v_BHSK, mask_BHTS, scale):
        import torch_npu

        output_BHTK, softmax_max, softmax_sum, *_ = torch_npu.npu_fusion_attention(
            q_BHTK,
            k_BHSK,
            v_BHSK,
            head_num=q_BHTK.shape[1],
            input_layout="BNSD",
            atten_mask=mask_BHTS,
            scale=scale,
            keep_prob=1.0,
            sparse_mode=0,
        )
        ctx.save_for_backward(
            q_BHTK,
            k_BHSK,
            v_BHSK,
            output_BHTK,
            softmax_max,
            softmax_sum,
        )
        ctx.mask = mask_BHTS
        ctx.scale = scale
        return output_BHTK

    @staticmethod
    def backward(ctx, grad_BHTK):
        import torch_npu

        (
            q_BHTK,
            k_BHSK,
            v_BHSK,
            output_BHTK,
            softmax_max,
            softmax_sum,
        ) = ctx.saved_tensors
        grad_q, grad_k, grad_v, *_ = torch_npu.npu_fusion_attention_grad(
            q_BHTK,
            k_BHSK,
            v_BHSK,
            grad_BHTK.contiguous(),
            head_num=q_BHTK.shape[1],
            input_layout="BNSD",
            atten_mask=ctx.mask,
            scale_value=ctx.scale,
            keep_prob=1.0,
            sparse_mode=0,
            attention_in=output_BHTK,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
        )
        return grad_q, grad_k, grad_v, None, None


spmd.register_local_autograd_function(_OffloadedFusionAttention)


def fusion_attention(
    q_THK, k_THK, v_THV, attention_metadata, scale=None, *, cpu_offload=True
):
    """Preserve the token mask and MLA's unequal query/value head dims."""
    import torch_npu

    mask_1HSS = _get_attn_mask(attention_metadata, q_THK, k_THK)
    value_dim = v_THV.shape[-1]
    # Match the fused kernel head width; padding leaves attention scores unchanged.
    # Slice off the extra output channels below to restore MLA value dimensions.
    v_THK = F.pad(v_THV, (0, q_THK.shape[-1] - value_dim))
    q_1HTK = q_THK.transpose(0, 1).unsqueeze(0).contiguous()
    k_1HSK = k_THK.transpose(0, 1).unsqueeze(0).contiguous()
    v_1HSK = v_THK.transpose(0, 1).unsqueeze(0).contiguous()
    scale = q_THK.shape[-1] ** -0.5 if scale is None else scale
    if (
        cpu_offload
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in (q_1HTK, k_1HSK, v_1HSK))
    ):
        out_1HTK = _OffloadedFusionAttention.apply(
            q_1HTK, k_1HSK, v_1HSK, mask_1HSS, scale
        )
    else:
        out_1HTK = torch_npu.npu_fusion_attention(
            q_1HTK,
            k_1HSK,
            v_1HSK,
            head_num=q_THK.shape[1],
            input_layout="BNSD",
            atten_mask=mask_1HSS,
            scale=scale,
            keep_prob=1.0,
            sparse_mode=0,
        )[0]
    return out_1HTK.squeeze(0).transpose(0, 1)[..., :value_dim]


class AscendCFlexInnerAttention(FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        pass

    def forward(
        self,
        q_THK,
        k_THK,
        v_THV,
        *,
        attention_metadata,
        score_mod=None,
        scale=None,
        enable_gqa=False,
        out_transform=None,
        **kwargs,
    ):
        # Score modifiers and differentiable LSE epilogues retain Flex semantics.
        if (
            q_THK.dtype not in (torch.float16, torch.bfloat16)
            or score_mod is not None
            or out_transform is not None
            or v_THV.shape[-1] > q_THK.shape[-1]
        ):
            return super().forward(
                q_THK,
                k_THK,
                v_THV,
                attention_metadata=attention_metadata,
                score_mod=score_mod,
                scale=scale,
                enable_gqa=enable_gqa,
                out_transform=out_transform,
                **kwargs,
            )
        return fusion_attention(q_THK, k_THK, v_THV, attention_metadata, scale)

    if TYPE_CHECKING:
        __call__ = forward


@override(
    target=FlexInnerAttention.Config, exact=True, description="Ascend C attention"
)
def ascendc_fusion_attention(config):
    return derive(config, AscendCFlexInnerAttention.Config)
