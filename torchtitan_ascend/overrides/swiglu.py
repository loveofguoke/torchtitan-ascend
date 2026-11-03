# SPDX-License-Identifier: Apache-2.0
"""SwiGLU activation via ``torch_npu.npu_swiglu``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_npu import npu_swiglu
from torchtitan.config import derive, override
from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU


class AscendSwiGLU(BinaryActivationFn):
    """SwiGLU via ``torch_npu.npu_swiglu``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        gate_up: torch.Tensor,
        *,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return npu_swiglu(gate_up.flatten(-2).contiguous(), dim=-1)

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor, /) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up


@override(
    target=SwiGLU.Config,
    exact=True,
    description="SwiGLU via torch_npu.npu_swiglu (replaces the compiled silu*mul region)",
)
def npu_fused_swiglu(cfg: SwiGLU.Config) -> AscendSwiGLU.Config:
    return derive(cfg, AscendSwiGLU.Config)
