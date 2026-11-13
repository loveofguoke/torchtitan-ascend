# SPDX-License-Identifier: Apache-2.0
"""Opt-in Ascend C RMSNorm, following MindSpeed-MM's npu_rms_norm route."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.norm import GatedRMSNorm

__all__ = [
    "AscendCRMSNorm",
    "AscendCGatedRMSNorm",
    "rmsnorm",
    "ascendc_rmsnorm",
    "ascendc_gated_rmsnorm",
]


def rmsnorm(x, weight, eps):
    """Keep the original parameter, epsilon, and output shape."""
    import torch_npu

    if weight is None:
        weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    # CANN requires FP32 gamma when the input is FP32.
    if x.dtype == torch.float32:
        weight = weight.float()
    epsilon = eps if eps is not None else torch.finfo(x.dtype).eps
    return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]


class AscendCRMSNorm(RMSNorm):
    weight: torch.Tensor | None
    eps: float | None

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, input):
        return rmsnorm(input, self.weight, self.eps)


@override(target=RMSNorm.Config, exact=True, description="Ascend C RMSNorm")
def ascendc_rmsnorm(config):
    return derive(config, AscendCRMSNorm.Config)


class AscendCGatedRMSNorm(GatedRMSNorm):
    weight: torch.Tensor | None
    eps: float | None
    activation_fn: Callable[[torch.Tensor], torch.Tensor]

    @dataclass(kw_only=True, slots=True)
    class Config(GatedRMSNorm.Config):
        pass

    def forward(self, input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normalized = rmsnorm(input.float(), self.weight, self.eps)
        return (normalized * self.activation_fn(gate.float())).to(input.dtype)


@override(target=GatedRMSNorm.Config, exact=True, description="Ascend C gated RMSNorm")
def ascendc_gated_rmsnorm(config: GatedRMSNorm.Config) -> AscendCGatedRMSNorm.Config:
    return derive(config, AscendCGatedRMSNorm.Config)
