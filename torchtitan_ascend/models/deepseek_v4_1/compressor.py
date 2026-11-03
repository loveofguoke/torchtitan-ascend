# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Remove absolute positional embeddings and overlap candidates;
# 2. Add source/reuse ownership and pre-RoPE latent output;
# 3. Support packed documents and non-empty ratio-1 compression pools.
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

"""Compressed main KV for DeepSeek V4.1 (CSA2).

Inherits upstream V4's :class:`Compressor`: same softmax-gated pooling over
each compression group, same packed ``cu_seqlens`` handling (groups restart at
document boundaries), same pre-norm RoPE split. V4.1 differs in three ways,
all expressed as overrides:

- No absolute positional embedding (``ape``) and no overlap candidate, so
  ``__init__`` is redeclared without them.
- Layers are *source* or *reuse*: a source pools its own hidden states, a
  reuse layer is handed the shared compressed KV and returns it unchanged.
- forward returns ``(cmp_k, latent)``: the RoPE-rotated KV for the attention
  and the pre-RoPE latent the indexer projects its keys from.

Shape suffixes: ``T`` = packed tokens, ``G`` = compressed groups,
``R`` = compression ratio, ``D`` = model dim, ``Dk`` = compressed head dim.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_remat as remat

from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v4.compressor import Compressor


if TYPE_CHECKING:
    from torchtitan.models.base.sharding import ShardingConfig


class Compressor(Compressor):
    """V4.1 compressor: V4's pooling, source/reuse wiring, no ape/overlap."""

    @dataclass(kw_only=True, slots=True)
    class Config(Compressor.Config):
        # Redeclared so Configurable wires build() to this subclass. Source
        # layers own the projections; reuse layers hold no weights.
        is_source: bool = True
        rope: RoPE.Config | None = None
        wkv: Linear.Config | None = None
        wgate: Linear.Config | None = None
        norm: RMSNorm.Config | None = None

        if TYPE_CHECKING:
            # Inherited from TorchTitan; keep runtime dataclass fields unchanged.
            param_init: dict | None = None
            sharding_config: ShardingConfig | None = None
            head_dim: int = 512
            rope_head_dim: int = 64
            compress_ratio: int = 4

    def __init__(self, config: Config):
        # Bypass V4's __init__, which always builds wkv/wgate/norm and the
        # V4-only ape parameter; V4.1 reuse layers carry no weights at all.
        Module.__init__(self)
        self.compress_ratio = config.compress_ratio
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.is_source = config.is_source
        if not self.is_source:
            return
        if config.rope is None or config.wkv is None or config.norm is None:
            raise ValueError(
                "A Compressor source layer requires rope, wkv and norm configs."
            )
        self.wkv = config.wkv.build()
        if config.compress_ratio > 1:
            if config.wgate is None:
                raise ValueError(
                    "A Compressor with compress_ratio > 1 requires a wgate config."
                )
            self.wgate = config.wgate.build()
        self.norm = config.norm.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        cmp_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pool hidden states into compressed KV, or pass the shared one through.

        Args:
            x: Hidden states of shape ``[T, D]``.
            positions: Position IDs of shape ``[T]``.
            cu_seqlens: Packed document offsets ``[N + 1]``; pooling groups
                restart at each document boundary.
            cmp_k: The shared compressed KV when this layer is not a source.

        Returns:
            ``(cmp_k, latent)``: the RoPE-rotated compressed KV
            ``[G, head_dim]`` and its pre-RoPE latent (which the indexer
            projects its keys from). A reuse layer returns ``(cmp_k, None)``.
        """
        if self.compress_ratio > 0 and not self.is_source:
            assert cmp_k is not None, (
                "A layer that reuses the compressed KV must receive it: no source "
                f"layer precedes this one (compress_ratio={self.compress_ratio})."
            )
        if not self.is_source:
            return cmp_k, None

        ratio = self.compress_ratio
        rd = self.rope_head_dim
        dtype = x.dtype
        seqlen = x.size(0)

        if ratio == 1:
            # One token per group: a plain projection, no gate and no fp32.
            latent = self.norm(self.wkv(x))
            comp_positions = positions
        else:
            with torch.autocast(device_type=x.device.type, dtype=torch.float32):
                kv = self.wkv(x)
                score = self.wgate(x)
            remat.recompute_needs_tensor(kv, score)
            if cu_seqlens is None:
                if seqlen % ratio != 0:
                    raise ValueError(
                        f"seqlen ({seqlen}) must be divisible by compress_ratio ({ratio})"
                    )
                kv = kv.unflatten(0, (-1, ratio))
                score = score.unflatten(0, (-1, ratio))
                comp_positions = (
                    positions[::ratio]
                    if positions is not None
                    else torch.arange(0, seqlen, ratio, device=x.device)
                )
            else:
                if seqlen < ratio:
                    return (
                        x.new_zeros(0, self.head_dim),
                        x.new_zeros(0, self.head_dim),
                    )
                # Groups restart at each document boundary; a document's
                # trailing partial group forms no entry, matching Attention
                # Gym's packed cu_seqlens_k convention.
                from torchtitan_ascend.models.compression import (
                    compression_group_layout,
                )

                slot_G = torch.arange(
                    seqlen // ratio, dtype=torch.int32, device=x.device
                )
                cmp_cu, doc_G, start_G = compression_group_layout(
                    cu_seqlens, ratio, slot_G
                )
                group_G = slot_G - cmp_cu[doc_G]
                start_G = torch.where(slot_G < cmp_cu[-1], start_G, 0)
                token_GR = start_G.unsqueeze(1) + torch.arange(ratio, device=x.device)
                kv, score = kv[token_GR], score[token_GR]
                comp_positions = (
                    positions[start_G] if positions is not None else group_G * ratio
                )
            latent = self.norm((kv * score.softmax(dim=1)).sum(dim=1).to(dtype))

        # Entry j stands for the group starting at token j * R, so it rotates at
        # that token's position. RoPE applies to the trailing rope slice only;
        # the latent is returned un-rotated for the indexer.
        latent_nope, latent_rope = torch.split(latent, [self.head_dim - rd, rd], dim=-1)
        latent_rope = self.rope(latent_rope.unsqueeze(1), positions=comp_positions)
        rotated = torch.cat([latent_nope, latent_rope.squeeze(1)], dim=-1)
        return rotated, latent

    if TYPE_CHECKING:
        __call__ = forward
