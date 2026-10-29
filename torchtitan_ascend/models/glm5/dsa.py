# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""GLM DSA parameters and packed-document metadata, independent of kernels."""

from dataclasses import dataclass
from functools import cached_property

import torch
from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear
from torchtitan.models.common.cp_attention import CPInnerAttention
from torchtitan.protocols.module import Module


@dataclass(frozen=True)
class DsaPacking:
    """Gather indices and cumulative lengths for causal TND kernel calls."""

    queries: tuple[int, ...]
    keys: tuple[int, ...]
    query_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]
    query_count: int


@dataclass(frozen=True)
class DsaMetadata:
    """Global valid document tokens and this CP rank's query range.

    Metadata is prepared outside model execution. No quadratic mask is built.
    The order maps CP-permuted token rows to original document positions.
    """

    documents: tuple[tuple[int, ...], ...]
    token_count: int
    order: tuple[int, ...] = ()
    query_start: int = 0
    query_end: int | None = None

    @classmethod
    def from_positions(cls, positions: torch.Tensor, padding_mask: torch.Tensor | None = None):
        positions_cpu = positions.detach().cpu().tolist()
        padding = [False] * len(positions_cpu) if padding_mask is None else padding_mask.detach().cpu().tolist()
        if len(padding) != len(positions_cpu):
            raise ValueError("DSA padding and position lengths must match")
        documents: list[list[int]] = []
        for row, (position, ignored) in enumerate(zip(positions_cpu, padding, strict=True)):
            if position == 0 or not documents:
                documents.append([])
            if not ignored:
                documents[-1].append(row)
        return cls(tuple(tuple(document) for document in documents if document), len(positions_cpu))

    @cached_property
    def packing(self) -> DsaPacking:
        """Pack each contiguous local query run with its document's KV prefix.

        Splitting runs also preserves causality under CP token permutations:
        right-down causal kernels align a run with the end of its KV prefix.
        """
        order = self.order or tuple(range(self.token_count))
        end = self.token_count if self.query_end is None else self.query_end
        local = {row: index for index, row in enumerate(order[self.query_start : end])}
        global_rows = {row: index for index, row in enumerate(order)}
        queries, keys, q_lengths, k_lengths = [], [], [], []
        for document in self.documents:
            selected = [index for index, row in enumerate(document) if row in local]
            cursor = 0
            while cursor < len(selected):
                stop = cursor + 1
                while stop < len(selected) and selected[stop] == selected[stop - 1] + 1:
                    stop += 1
                queries.extend(local[document[index]] for index in selected[cursor:stop])
                keys.extend(global_rows[row] for row in document[: selected[stop - 1] + 1])
                q_lengths.append(len(queries))
                k_lengths.append(len(keys))
                cursor = stop
        return DsaPacking(tuple(queries), tuple(keys), tuple(q_lengths), tuple(k_lengths), end - self.query_start)


class DsaAttention(Module):
    """Config seam for a backend implementing GLM token-selected attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention_dropout: float = 0.0
        enable_cp: bool = False

    def __init__(self, config):
        super().__init__()

    def forward(self, query, key, indices, qr, kr, qlens, klens, scale) -> tuple[torch.Tensor, ...]:
        raise RuntimeError("Select the GLM NPU DSA override before model construction.")


class DsaCPAttention(CPInnerAttention[DsaMetadata, DsaMetadata], DsaAttention):
    """CP config marker resolved to a native backend before model construction."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, DsaAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        raise RuntimeError("Select the GLM NPU DSA override before CP preprocessing.")


class Glm5DsaIndexer(Module):
    """Checkpoint-compatible indexer projections; execution belongs to a backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        n_heads: int
        head_dim: int
        qk_rope_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: ComplexRoPE.Config
        trainable: bool = False

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build().float()
        self.rope = config.rope.build()
        self.requires_grad_(config.trainable)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.weights_proj.float()
        return self

    def forward(self, hidden_states_TD, q_resid_TR, positions_T) -> tuple[torch.Tensor, ...]:
        raise RuntimeError("Select the GLM NPU DSA override before model construction.")
