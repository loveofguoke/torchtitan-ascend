"""Token-first TorchTitan ComplexRoPE support for Ascend NPU."""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.rope import ComplexRoPE, _maybe_check_max_pos


@spmd.local_map(
    out_types=(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
        spmd.PartitionSpec(("dp", "cp"), None, None),
    )
)
def npu_reshape_for_broadcast(
    rope_cache: torch.Tensor,
    query_shape: torch.Size | tuple[int, ...],
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a token-first cache without complex NPU gather."""
    cache_width = rope_cache.shape[-1]
    num_tokens = query_shape[0]
    if positions is None:
        selected = rope_cache[:num_tokens]
    elif torch.is_complex(rope_cache):
        selected = torch.complex(
            rope_cache.real[positions],
            rope_cache.imag[positions],
        )
    else:
        selected = rope_cache[positions]
    return selected.view(num_tokens, 1, cache_width)


def _wrap_dtensor_like(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if not isinstance(reference, DTensor):
        return output
    return DTensor.from_local(
        output,
        device_mesh=reference.device_mesh,
        placements=reference.placements,
        run_check=False,
    )


def npu_apply_complex_rope(
    query: torch.Tensor,
    key: torch.Tensor | None,
    rope_cache: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply token-first complex RoPE with the fused Ascend kernel."""
    from torch.distributed.tensor import DTensor

    query_local = query.to_local() if isinstance(query, DTensor) else query
    key_local = key.to_local() if isinstance(key, DTensor) else key
    cache_local = (
        rope_cache.to_local() if isinstance(rope_cache, DTensor) else rope_cache
    )

    cos_1T1H = cache_local.real.repeat_interleave(2, dim=-1).unsqueeze(0)
    sin_1T1H = cache_local.imag.repeat_interleave(2, dim=-1).unsqueeze(0)
    if inverse:
        sin_1T1H = -sin_1T1H

    query_1TNH = query_local.float().unsqueeze(0)
    query_out_TNH = torch_npu.npu_rotary_mul(
        query_1TNH,
        cos_1T1H,
        sin_1T1H,
        rotary_mode="interleave",
    ).squeeze(0)
    query_result = _wrap_dtensor_like(
        query_out_TNH.type_as(query_local),
        query,
    )
    if key_local is None:
        return query_result

    key_1TMH = key_local.float().unsqueeze(0)
    key_out_TMH = torch_npu.npu_rotary_mul(
        key_1TMH,
        cos_1T1H,
        sin_1T1H,
        rotary_mode="interleave",
    ).squeeze(0)
    return (
        query_result,
        _wrap_dtensor_like(key_out_TMH.type_as(key_local), key),
    )


class AscendComplexRoPE(ComplexRoPE):
    """ComplexRoPE with NPU-safe cache selection and fused rotation."""

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def _reshape_cache(
        self, query: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
        return npu_reshape_for_broadcast(
            self.cache,
            complex_query_shape,
            positions,
        )

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from torch.distributed.tensor import DTensor

        query_local = query.to_local() if isinstance(query, DTensor) else query
        if query_local.device.type != "npu":
            return ComplexRoPE.apply_rotary_emb(
                query,
                key,
                rope_cache,
                inverse=inverse,
            )
        return npu_apply_complex_rope(
            query,
            key,
            rope_cache,
            inverse=inverse,
        )


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="ComplexRoPE with NPU-safe cache selection and fused rotation",
)
def npu_complex_rope(cfg: ComplexRoPE.Config) -> AscendComplexRoPE.Config:
    """Select the Ascend-compatible ComplexRoPE implementation."""
    return derive(cfg, AscendComplexRoPE.Config)
