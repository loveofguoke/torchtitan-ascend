# SPDX-License-Identifier: Apache-2.0
"""RMSNorm on ``torch_npu.npu_rms_norm``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_npu import npu_rms_norm as _npu_rms_norm
from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm


class AscendRMSNorm(RMSNorm):

    weight: torch.Tensor | None
    eps: float | None

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return super().forward(input)
        eps = self.eps if self.eps is not None else torch.finfo(input.dtype).eps
        return _npu_rms_norm(input, self.weight, eps)[0]


@override(
    target=RMSNorm.Config,
    exact=True,
    description="RMSNorm via torch_npu.npu_rms_norm",
)
def npu_rms_norm(cfg: RMSNorm.Config) -> AscendRMSNorm.Config:
    return derive(cfg, AscendRMSNorm.Config)
