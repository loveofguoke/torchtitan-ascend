# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Adapt HcPre for Single-Pass pre-mix propagation;
# 2. Match V4.1 Sinkhorn normalization;
# 3. Reuse the upstream HcPost implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-3-Clause file in the root directory of this source tree.

"""Single-Pass mHC for DeepSeek V4.1.

V4.1 shares V4's Sinkhorn-balanced mixing math, so ``HcPre``/``HcPost``
inherit the upstream modules. The only architectural difference: V4.1
threads the pre-mixing coefficients across sublayers (the "single pass"),
so ``HcPre.forward`` collapses the branches with the *incoming* ``pre_mix``
and emits its own coefficients for the next sublayer to consume.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from torchtitan.models.deepseek_v4.mhc import HcPost, HcPre, HcSplitSinkhorn

__all__ = ["HcPre", "HcPost"]


if TYPE_CHECKING:
    from torchtitan.models.base.sharding import ShardingConfig


class _V41Sinkhorn(HcSplitSinkhorn):

    hc_mult: int
    eps: float
    sinkhorn_iters: int

    @dataclass(kw_only=True, slots=True)
    class Config(HcSplitSinkhorn.Config):
        pass

        if TYPE_CHECKING:
            # Inherited from TorchTitan; keep runtime dataclass fields unchanged.
            param_init: dict | None = None
            sharding_config: ShardingConfig | None = None
            hc_mult: int = 4
            sinkhorn_iters: int = 20
            eps: float = 1e-06

    def forward(self, mixes, hc_scale, hc_base):
        """Apply V4.1 Sinkhorn normalization with the official first-row epsilon semantics."""
        hc = self.hc_mult
        eps = self.eps
        iters = self.sinkhorn_iters
        pre, post, comb = mixes.split([hc, hc, hc * hc], dim=-1)
        comb = comb.unflatten(-1, (hc, hc))
        pre = (
            torch.sigmoid(
                pre * hc_scale[0] + hc_base[:hc].view(*([1] * (pre.ndim - 1)), hc)
            )
            + eps
        )
        post = 2 * torch.sigmoid(
            post * hc_scale[1] + hc_base[hc : 2 * hc].view(*([1] * (post.ndim - 1)), hc)
        )
        comb = comb * hc_scale[2] + hc_base[2 * hc :].view(
            *([1] * (comb.ndim - 2)), hc, hc
        )
        comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
        comb = comb / comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb


class HcPre(HcPre):
    """V4.1 HcPre: collapse with the incoming ``pre_mix``, emit the next one."""

    norm_eps: float
    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        # Redeclared so Configurable wires build() to this subclass.
        pass

        if TYPE_CHECKING:
            # Inherited from TorchTitan; keep runtime dataclass fields unchanged.
            param_init: dict | None = None
            sharding_config: ShardingConfig | None = None
            hc_mult: int = 4
            dim: int
            sinkhorn_iters: int = 20
            eps: float = 1e-06
            norm_eps: float = 1e-06

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.sinkhorn = _V41Sinkhorn.Config(
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.sinkhorn_iters,
            eps=config.eps,
        ).build()

    def forward(
        self, x: torch.Tensor, pre_mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse the branches and emit the coefficients the next step consumes.

        Args:
            x: Hidden states of shape ``[..., hc_mult, D]``.
            pre_mix: Incoming pre-mixing coefficients of shape
                ``[..., hc_mult]`` produced by the previous sublayer.

        Returns:
            ``(y, pre, post, comb)`` where ``y`` has shape ``[..., D]`` and
            ``pre`` is handed to the next sublayer while ``post``/``comb``
            are consumed by ``HcPost``.
        """
        flat = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn.float()) * rsqrt
        pre, post, comb = self.sinkhorn(
            mixes.float(), self.hc_scale.float(), self.hc_base.float()
        )
        return self.collapse(x, pre_mix), pre, post, comb

    @staticmethod
    def identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
        """Return the one-hot stream mix used at the input of the stack."""
        pre = torch.zeros(
            (*x.shape[:-2], hc_mult),
            dtype=torch.float32,
            device=x.device,
        )
        pre[..., 0] = 1.0
        return pre

    @staticmethod
    def collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the branches into one sublayer input."""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2)
        return y.to(x.dtype)

    if TYPE_CHECKING:
        __call__ = forward
