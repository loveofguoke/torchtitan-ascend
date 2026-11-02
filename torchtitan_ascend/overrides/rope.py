"""Rotary embeddings with real-valued caches and Ascend rotary multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_npu import npu_rotary_mul as _npu_rotary_mul
from torchtitan.config import derive, override
from torchtitan.models.common.rope import (
    ComplexRoPE,
    CosSinRoPE,
    _maybe_check_max_pos,
    _reshape_for_broadcast,
)

from torchtitan_ascend.device import require_device


def _rotary_kernel(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str
) -> torch.Tensor:
    """``npu_rotary_mul`` needs 4-D input and computes in the input dtype (bf16 rounding)."""
    return _npu_rotary_mul(
        x.unsqueeze(0),
        cos.to(x.dtype).unsqueeze(0),
        sin.to(x.dtype).unsqueeze(0),
        rotary_mode=mode,
    ).squeeze(0)


class AscendComplexRoPE(ComplexRoPE):
    """``ComplexRoPE`` semantics with a real ``(L, dim/2, 2)`` [cos, sin] cache."""

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def __init__(self, config: Config) -> None:
        require_device("Ascend", feature="Ascend ComplexRoPE")
        super().__init__(config)

    def _precompute_cache(self) -> torch.Tensor:
        return torch.view_as_real(super()._precompute_cache()).contiguous()

    def _reshape_cache(
        self, query: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        num_tokens = query.shape[0]
        if query.ndim >= 3:
            return _reshape_for_broadcast(
                self.cache.flatten(-2), query.shape, positions
            ).unflatten(-1, (-1, 2))
        cache = self.cache[:num_tokens] if positions is None else self.cache[positions]
        return cache.view(num_tokens, cache.shape[-2], 2)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """(a + ib)(c + is) = (ac - bs) + i(as + bc) on adjacent (even, odd) pairs.
        """
        cos = rope_cache[..., 0]
        sin = rope_cache[..., 1]
        if inverse:
            sin = -sin

        if query.ndim == 3 and not inverse:
            cos2 = cos.repeat_interleave(2, dim=-1)
            sin2 = sin.repeat_interleave(2, dim=-1)
            if key is None:
                return _rotary_kernel(query, cos2, sin2, "interleave")
            return (
                _rotary_kernel(query, cos2, sin2, "interleave"),
                _rotary_kernel(key, cos2, sin2, "interleave"),
            )

        def rot(x: torch.Tensor) -> torch.Tensor:
            xf = x.float().reshape(*x.shape[:-1], -1, 2)
            a, b = xf[..., 0], xf[..., 1]
            out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
            return out.flatten(-2).type_as(x)

        if key is None:
            return rot(query)
        return rot(query), rot(key)


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="ComplexRoPE with a real-valued cache (torch_npu cannot index complex tensors)",
)
def real_cache_rope(cfg: ComplexRoPE.Config) -> AscendComplexRoPE.Config:
    return derive(cfg, AscendComplexRoPE.Config)


class AscendCosSinRoPE(CosSinRoPE):
    """Apply upstream ``CosSinRoPE`` rotate-half math with the NPU rotary kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    def __init__(self, config: Config) -> None:
        require_device("Ascend", feature="Ascend CosSinRoPE")
        super().__init__(config)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inverse:
            return CosSinRoPE.apply_rotary_emb(query, key, rope_cache, inverse=inverse)
        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        query_out = _rotary_kernel(query, cos, sin, "half")
        if key is None:
            return query_out
        return query_out, _rotary_kernel(key, cos, sin, "half")


@override(
    target=CosSinRoPE.Config,
    exact=True,
    description="CosSinRoPE rotation via torch_npu.npu_rotary_mul (rotate-half)",
)
def npu_rotary_cossin(cfg: CosSinRoPE.Config) -> AscendCosSinRoPE.Config:
    return derive(cfg, AscendCosSinRoPE.Config)
