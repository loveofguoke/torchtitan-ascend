# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
#
# Modification points:
# 1. Add per-document compression group layout;
# 2. Share group-start positions between compressed KV and index keys;
# 3. Preserve padding and incomplete-group semantics.
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
"""Compression-group geometry for packed documents.

Each document contributes ``floor(len_d / ratio)`` complete groups. Trailing
partial groups produce no valid entries, and padded slots are not selectable.
Compressed KV and index keys must use the same group-start positions for RoPE.
"""

import torch

from torchtitan.models.deepseek_v4.compressor import compressed_cu_seqlens

__all__ = ["compression_group_layout"]


def compression_group_layout(cu_seqlens: torch.Tensor, ratio: int, slots=None):
    """Return compressed offsets, document IDs, and group-start token indices.

    Args:
        cu_seqlens: Token-level packed document offsets of shape ``[N + 1]``.
        ratio: Compression ratio applied independently to each document.
        slots: Group slots to query; ``None`` selects all valid groups.

    Returns:
        Tuple ``(cmp_cu, doc_of_slot, start_token_of_slot)`` containing the
        compressed offsets of shape ``[G + 1]``, the document ID for each slot,
        and the absolute token index at which each group starts. The start
        indices are used for gathering and group-level RoPE positions.
    """
    cmp_cu = compressed_cu_seqlens(cu_seqlens, ratio)
    if slots is None:
        slots = torch.arange(
            int(cmp_cu[-1]), dtype=torch.int32, device=cu_seqlens.device
        )
    doc = torch.searchsorted(cmp_cu[1:], slots, right=True, out_int32=True)
    start = cu_seqlens[doc] + (slots - cmp_cu[doc]) * ratio
    return cmp_cu, doc, start
