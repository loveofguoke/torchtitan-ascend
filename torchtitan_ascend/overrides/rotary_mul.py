"""Opt-in vision RoPE following MindSpeed-MM's FP32 interleaved route."""

from __future__ import annotations

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.kimi_k2_7 import vision_encoder

__all__ = [
    "rotary_mul",
    "enable_ascendc_rotary_mul",
]

# T = packed tokens, H = heads, K = head dim.


def rotary_mul(q_THK, k_THK, rope_cache, *, inverse=False):
    """Rotate adjacent pairs in FP32, then restore each input's dtype."""
    import torch_npu

    if inverse:
        rope_cache = rope_cache.conj().resolve_conj()
    cos_T1K = rope_cache.real.repeat_interleave(2, dim=-1).contiguous()
    sin_T1K = rope_cache.imag.repeat_interleave(2, dim=-1).contiguous()
    q_THK = torch_npu.npu_rotary_mul(
        q_THK.float(), cos_T1K, sin_T1K, rotary_mode="interleave"
    ).type_as(q_THK)
    if k_THK is None:
        return q_THK
    k_THK = torch_npu.npu_rotary_mul(
        k_THK.float(), cos_T1K, sin_T1K, rotary_mode="interleave"
    ).type_as(k_THK)
    return q_THK, k_THK


class _VisionRoPE:
    @staticmethod
    def apply_rotary_emb(query, key, rope_cache, *, inverse=False):
        return rotary_mul(query, key, rope_cache, inverse=inverse)


def enable_ascendc_rotary_mul():
    # Vision RoPE is called through a module lookup, without a Config node.
    # Rebind that lookup for this process, leaving shared ComplexRoPE unchanged.
    vision_encoder.ComplexRoPE = _VisionRoPE
