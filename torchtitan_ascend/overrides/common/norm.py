# SPDX-License-Identifier: Apache-2.0
"""AscendC RMSNorm and gated RMSNorm using torch_npu.npu_rms_norm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.norm import GatedRMSNorm

__all__ = [
    "AscendRMSNorm",
    "AscendGatedRMSNorm",
    "rmsnorm_forward",
    "npu_rmsnorm",
    "npu_gated_rmsnorm",
]


def rmsnorm_forward(x, weight, eps):
    """Keep the original parameter, epsilon, and output shape."""
    import torch_npu

    # CANN requires FP32 gamma when the input is FP32.
    if x.dtype == torch.float32:
        weight = weight.float()
    epsilon = eps if eps is not None else torch.finfo(x.dtype).eps
    return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]


class AscendRMSNorm(RMSNorm):
    weight: torch.Tensor | None
    eps: float | None

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, input):
        weight = self.weight
        if weight is None:
            normalized_shape = self.normalized_shape
            if tuple(input.shape[-len(normalized_shape) :]) != normalized_shape:
                raise ValueError(
                    f"Expected trailing dimensions {normalized_shape}, "
                    f"got input shape {tuple(input.shape)}"
                )
            weight = torch.ones(
                normalized_shape, dtype=input.dtype, device=input.device
            )
        return rmsnorm_forward(input, weight, self.eps)


@override(target=RMSNorm.Config, exact=True, description="Ascend C RMSNorm")
def npu_rmsnorm(config):
    return derive(config, AscendRMSNorm.Config)


class AscendGatedRMSNorm(GatedRMSNorm):
    weight: torch.Tensor | None
    eps: float | None
    activation_fn: Callable[[torch.Tensor], torch.Tensor]

    @dataclass(kw_only=True, slots=True)
    class Config(GatedRMSNorm.Config):
        pass

    def forward(self, input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normalized = rmsnorm_forward(input.float(), self.weight, self.eps)
        return (normalized * self.activation_fn(gate.float())).to(input.dtype)


@override(target=GatedRMSNorm.Config, exact=True, description="Ascend C gated RMSNorm")
def npu_gated_rmsnorm(config: GatedRMSNorm.Config) -> AscendGatedRMSNorm.Config:
    return derive(config, AscendGatedRMSNorm.Config)
