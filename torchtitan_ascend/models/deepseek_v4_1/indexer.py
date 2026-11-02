# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
#
# Modification points:
# 1. Add FULL/REINDEX/REUSE hierarchical indexer modes;
# 2. Add an eager candidate-aware selector and differentiable student scores;
# 3. Support packed documents and reuse upstream compressed gather helpers.
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

"""Hierarchical sparse indexer for DeepSeek V4.1 (CSA2).

Inherits upstream V4's :class:`Indexer` for the projection side (query
projection ``wq_b``, per-head ``weights_proj``, RoPE), and overrides the rest
for V4.1's cross-layer design. Where V4's indexer is per-layer and self-contained
(each one owns a compressor and hands ``(idx_q, idx_k, idx_w)`` to the CSA,
which selects), V4.1's indexer is hierarchical across CSA2's three static
modes:

- ``FULL`` owns the compressed KV and projects its own index keys from the
  compressor latent, then scores and selects.
- ``REINDEX`` rescores the shared index keys with its own query.
- ``REUSE`` carries the shared top-k forward without computing anything.

The selection itself is a small eager implementation (``:class:``Selector``):
plain einsum scoring, visibility/candidate masking and top-k, instead of the
fused ``lightning_indexer`` kernel V4's ``Indexer.select`` calls.

Shape suffixes: ``T`` = packed tokens, ``H`` = index heads, ``Di`` = index head
dim, ``N`` = compressed pool size, ``K`` = selected slots.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v4.compressor import (
    compressed_cu_seqlens,
    compressed_gather_indices,
)
from torchtitan.models.deepseek_v4.compressor import Indexer

if TYPE_CHECKING:
    from torchtitan.models.common.attention import VarlenAttentionMetadata


class IndexerMode(StrEnum):
    """CSA2's static per-layer modes."""

    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


FULL = IndexerMode.FULL
REINDEX = IndexerMode.REINDEX
REUSE = IndexerMode.REUSE


def _document_intervals(
    cu_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor, *, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(pool_start, pool_end, local)`` per query: its own document's packed
    compressed-entry interval, and its position inside that document.
    """
    positions = torch.arange(num_tokens, device=cu_seqlens.device)
    documents = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    pool_start = cu_seqlens_k.index_select(0, documents).unsqueeze(-1)
    pool_end = cu_seqlens_k.index_select(0, documents + 1).unsqueeze(-1)
    local = (positions - cu_seqlens.index_select(0, documents)).unsqueeze(-1)
    return pool_start, pool_end, local


def _selection_mask(
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    num_tokens: int,
    num_cmp: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Document isolation plus causal completeness over the compressed axis.

    A compressed entry becomes visible once the query has passed its last
    token: entry ``j`` covers tokens ``[j * ratio, (j + 1) * ratio)`` of its
    document, so a query at local position ``t`` sees entries ``j`` with
    ``(j + 1) * ratio <= t + 1``.
    """
    pool_start, pool_end, local = _document_intervals(
        cu_seqlens, cu_seqlens_k, num_tokens=num_tokens
    )
    ends = torch.minimum(pool_end, pool_start + (local + 1) // compress_ratio)
    entry = torch.arange(num_cmp, device=cu_seqlens.device).unsqueeze(0)
    visible = (entry >= pool_start) & (entry < ends)
    return visible, ends - 1, ends > pool_start, pool_start


def _student_logits(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    weights: torch.Tensor,
    local_TK: torch.Tensor,
    cu_seqlens: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """The per-head scores at a document-local selection: the pass that
    carries gradient (selection itself is discrete and carries none).
    """
    selected_TKD = idx_k[
        compressed_gather_indices(local_TK, cu_seqlens, compress_ratio)
    ]
    logits_THK = torch.einsum("thd,tkd->thk", idx_q, selected_TKD)
    logits_THK = logits_THK.relu() * weights.unsqueeze(-1)
    return logits_THK.sum(dim=1).masked_fill(local_TK < 0, -torch.inf)


class Selector(Module):
    """The selection node: scores every visible compressed entry and picks the
    top-k. A small eager implementation standing in for the fused kernel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mode: IndexerMode
        compress_ratio: int
        num_index_heads: int
        index_head_dim: int
        index_topk: int
        candidate_topk_blocks: int = -1
        candidate_block_size: int = -1

    def __init__(self, config: Config):
        super().__init__()
        self.mode = config.mode
        self.compress_ratio = config.compress_ratio
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size

    @property
    def has_candidate_pool(self) -> bool:
        return self.candidate_topk_blocks > 0

    @staticmethod
    def select_candidate_blocks(
        scores_TN: torch.Tensor,
        newest_T1: torch.Tensor,
        newest_valid_T1: torch.Tensor,
        topk_blocks: int,
        block_size: int,
    ) -> torch.Tensor:
        """The candidate source's blockwise selection: keep the best-scoring
        blocks, always pinning the block that holds the newest position.
        """
        width = scores_TN.size(-1)
        if width % block_size != 0:
            scores_TN = F.pad(scores_TN, (0, -width % block_size), value=-torch.inf)
        block_scores_TB = scores_TN.unflatten(-1, (-1, block_size)).amax(dim=-1)
        num_blocks = block_scores_TB.size(-1)

        last = newest_T1 // block_size
        pin_TB = torch.arange(num_blocks, device=scores_TN.device).unsqueeze(0) == last
        block_scores_TB = block_scores_TB.masked_fill(pin_TB & newest_valid_T1, torch.inf)

        top = block_scores_TB.topk(min(topk_blocks, num_blocks), dim=-1)
        keep_TB = torch.zeros_like(block_scores_TB, dtype=torch.bool).scatter_(
            -1, top.indices, top.values > -torch.inf
        )
        return keep_TB.repeat_interleave(block_size, dim=-1)[..., :width]

    def _select_candidate_blocks_by_document(
        self,
        scores_TN: torch.Tensor,
        newest_T1: torch.Tensor,
        newest_valid_T1: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.zeros_like(scores_TN, dtype=torch.bool)
        for doc in range(cu_seqlens.numel() - 1):
            q_begin, q_end = int(cu_seqlens[doc]), int(cu_seqlens[doc + 1])
            k_begin, k_end = int(cu_seqlens_k[doc]), int(cu_seqlens_k[doc + 1])
            if q_begin == q_end or k_begin == k_end:
                continue
            candidates[q_begin:q_end, k_begin:k_end] = self.select_candidate_blocks(
                scores_TN[q_begin:q_end, k_begin:k_end],
                newest_T1[q_begin:q_end] - k_begin,
                newest_valid_T1[q_begin:q_end],
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        return candidates

    def forward(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        weights: torch.Tensor,
        attention_metadata: "VarlenAttentionMetadata | None",
        *,
        candidates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Score the visible compressed entries and select the top-k.

        Returns ``(topk_indices, topk_scores, candidates)``: document-local
        indices ``[T, K]`` with -1 in unusable slots, the live indexer logits
        for those entries, and the candidate pool this layer built (FULL mode
        with a candidate pool only).
        """
        num_cmp = idx_k.size(0)
        max_seqlen = idx_q.size(0) if attention_metadata is None else attention_metadata.max_k
        cu_seqlens = (
            torch.tensor([0, idx_q.size(0)], dtype=torch.int32, device=idx_q.device)
            if attention_metadata is None
            else attention_metadata.cu_seq_q
        )
        cu_seqlens_k = compressed_cu_seqlens(cu_seqlens, self.compress_ratio)
        if not self.has_candidate_pool:
            topk = min(self.index_topk, max_seqlen // self.compress_ratio, num_cmp)
            indices, scores = Indexer.select(
                idx_q,
                idx_k,
                weights,
                max_seqlen=max_seqlen,
                ratio=self.compress_ratio,
                topk=self.index_topk,
                cu_seqlens=cu_seqlens,
                return_scores=True,
            )
            return indices[:, :topk], scores[:, :topk], candidates
        visible_TN, newest_T1, newest_valid_T1, pool_start_T1 = _selection_mask(
            cu_seqlens,
            cu_seqlens_k,
            num_tokens=idx_q.size(0),
            num_cmp=num_cmp,
            compress_ratio=self.compress_ratio,
        )

        with torch.no_grad():
            scores_THN = torch.einsum("thd,nd->thn", idx_q, idx_k)
            scores_TN = (scores_THN.relu() * weights.unsqueeze(-1)).sum(dim=1)
            scores_TN = scores_TN.masked_fill(~visible_TN, -torch.inf)

            if self.mode is FULL and self.has_candidate_pool:
                candidates = self._select_candidate_blocks_by_document(
                    scores_TN,
                    newest_T1,
                    newest_valid_T1,
                    cu_seqlens,
                    cu_seqlens_k,
                )
            elif self.mode is REINDEX and self.has_candidate_pool:
                assert candidates is not None, (
                    "A Reindex Mode indexer with candidate_topk_blocks set "
                    "searches the pool the candidate source built."
                )
                scores_TN = scores_TN.masked_fill(~candidates, -torch.inf)

            topk = min(self.index_topk, max_seqlen // self.compress_ratio, num_cmp)
            selected_TK = scores_TN.topk(topk, dim=-1, sorted=False).indices.to(torch.int32)
            selected_valid_TK = torch.isfinite(scores_TN.gather(-1, selected_TK))
            local_TK = torch.where(selected_valid_TK, selected_TK - pool_start_T1, -1)
            topk_indices_TK = local_TK.sort(dim=-1, descending=True).values

        topk_scores_TK = _student_logits(
            idx_q,
            idx_k,
            weights,
            topk_indices_TK,
            cu_seqlens,
            self.compress_ratio,
        )
        return topk_indices_TK, topk_scores_TK, candidates


class HierarchicalIndexer(Indexer):
    """The Hierarchical Sparse Indexer across CSA2's layer modes.

    Inherits V4's indexer projections (``wq_b``/``weights_proj``/RoPE); the
    per-layer self-contained compressor is dropped (V4.1's index keys come
    from the shared compressor latent or the shared chain), and forward is
    overridden for the three modes plus eager selection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Indexer.Config):
        # Re-declared so Configurable wires build() to this subclass.
        mode: IndexerMode = REUSE
        compress_ratio: int = 0
        index_topk: int = 0
        selector: Selector.Config | None = None
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None
        # V4.1 indexers never own a compressor: FULL projects the shared
        # latent, REINDEX/REUSE read the shared chain.
        compressor: None = None  # pyrefly: ignore [bad-override]
        rope: RoPE.Config | None = None  # pyrefly: ignore [bad-override]
        wq_b: Linear.Config | None = None  # pyrefly: ignore [bad-override]
        weights_proj: Linear.Config | None = None  # pyrefly: ignore [bad-override]

    def __init__(self, config: Config):
        # Bypass V4's __init__, which always builds the per-layer compressor.
        Module.__init__(self)
        self.mode = config.mode
        self.compress_ratio = config.compress_ratio
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        if config.selector is None:
            raise ValueError("A HierarchicalIndexer requires a selector config.")
        self.selector = config.selector.build()
        if self.mode is REUSE:
            return
        if config.rope is None or config.wq_b is None or config.weights_proj is None:
            raise ValueError(
                "A Full or Reindex Mode indexer requires rope, wq_b and "
                "weights_proj configs."
            )
        self.rope = config.rope.build()
        self.wq_b = config.wq_b.build()
        self.weights_proj = config.weights_proj.build()
        if self.mode is FULL:
            if config.wk is None or config.k_norm is None:
                raise ValueError(
                    "A Full Mode indexer requires wk and k_norm configs to "
                    "project its own index keys."
                )
            self.wk = config.wk.build()
            self.k_norm = config.k_norm.build()

    def _rope_partial(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the trailing ``rope_head_dim`` channels of an indexer tensor."""
        rd = self.rope_head_dim
        x_nope, x_rope = torch.split(x, [self.index_head_dim - rd, rd], dim=-1)
        x_rope = self.rope(x_rope, positions=positions)
        return torch.cat([x_nope, x_rope], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: "VarlenAttentionMetadata | None",
        *,
        latent: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """The mode adapter: emit or reuse the shared index keys and top-k."""
        if self.mode is REUSE:
            if self.compress_ratio > 0:
                assert topk_indices is not None, (
                    "A Reuse Mode indexer must be handed the shared top-k: no "
                    f"index source precedes this one (compress_ratio="
                    f"{self.compress_ratio})."
                )
            return idx_k, topk_indices, topk_scores, candidates

        if self.mode is FULL:
            assert latent is not None, (
                "A Full Mode indexer projects its keys from the compressor latent."
            )
            idx_k = self.k_norm(self.wk(latent))
            # Entry j stands for the group starting at token j * ratio; index
            # keys rotate at the same per-slot group starts as the compressed
            # KV (per-document full groups; fixed-capacity padding slots carry
            # position 0 and are unselectable via cu_seqlens_k).
            if attention_metadata is not None:
                from torchtitan_ascend.models.compression import (
                    compression_group_layout,
                )

                num_slots = latent.size(0)
                slots = torch.arange(
                    num_slots, dtype=torch.int32, device=idx_k.device
                )
                cmp_cu, _, start_tokens = compression_group_layout(
                    attention_metadata.cu_seq_q, self.compress_ratio, slots
                )
                start_tokens = torch.where(slots < cmp_cu[-1], start_tokens, 0)
                group_positions = positions[start_tokens]
            else:
                group_positions = positions[:: self.compress_ratio]
            idx_k = self._rope_partial(
                idx_k.unsqueeze(1), group_positions
            ).squeeze(1)
        else:
            assert idx_k is not None, (
                "A Reindex Mode indexer rescores the shared index keys, which "
                "no preceding Full Mode layer produced."
            )

        idx_q = self.wq_b(qr).unflatten(-1, (self.num_index_heads, self.index_head_dim))
        idx_q_nope, idx_q_rope = torch.split(
            idx_q, [self.index_head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1
        )
        remat.recompute_needs_tensor(idx_q_nope, idx_q_rope)
        idx_q_rope = self.rope(idx_q_rope, positions=positions)
        idx_q = torch.cat([idx_q_nope, idx_q_rope], dim=-1)
        weights = self.weights_proj(x)
        remat.recompute_needs_tensor(weights)
        weights = weights * (
            self.index_head_dim**-0.5 * self.num_index_heads**-0.5
        )

        topk_indices, topk_scores, candidates = self.selector(
            idx_q,
            idx_k,
            weights,
            attention_metadata,
            candidates=candidates,
        )
        return idx_k, topk_indices, topk_scores, candidates
