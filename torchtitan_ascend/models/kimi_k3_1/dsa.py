# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-3-Clause file in the root directory of this source tree.

"""Kimi K3.1 gated sparse attention for packed training."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.base.module import Module
from torchtitan.models.common import LayerNorm, Linear, RMSNorm
from torchtitan.models.common.attention import (
    Attention,
    create_varlen_metadata_for_document,
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3.attention import KimiMLAAttention


class IndexerFeatures(NamedTuple):
    """Projected query (T,H,Di), key (T,Di), and head weights (T,H)."""

    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor


@cache
def _hadamard_matrix(
    dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    return F.linear(x, _hadamard_matrix(dim, x.dtype, x.device)) * dim**-0.5


def _document_ids(offsets: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(offsets[1:], positions, right=True, out_int32=True)


def select_topk(
    q_THDi: torch.Tensor,
    k_SDi: torch.Tensor,
    weights_TH: torch.Tensor,
    *,
    top_k: int,
    attention_metadata: VarlenAttentionMetadata | None,
    padding_mask_T: torch.Tensor | None,
    return_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return global packed indices, using -1 for slots without a valid key."""
    num_queries, num_keys = q_THDi.shape[0], k_SDi.shape[0]
    num_selected_slots = min(top_k, num_keys)
    if num_selected_slots == 0:
        return q_THDi.new_empty((num_queries, 0), dtype=torch.int32), None

    with torch.no_grad():
        scores_TS = (
            torch.einsum(
                "thd,sd->ths", q_THDi.float(), k_SDi.float()  # codespell:ignore ths
            )
            .relu()
            .mul(weights_TH.float().unsqueeze(-1))
            .sum(dim=1)
        )
        query_positions_T = torch.arange(num_queries, device=scores_TS.device)
        key_positions_S = torch.arange(num_keys, device=scores_TS.device)
        valid_TS = torch.le(
            key_positions_S.unsqueeze(0), query_positions_T.unsqueeze(1)
        )
        if attention_metadata is not None:
            query_docs_T = _document_ids(
                attention_metadata.cu_seq_q, query_positions_T.to(torch.int32)
            )
            key_docs_S = _document_ids(
                attention_metadata.cu_seq_k, key_positions_S.to(torch.int32)
            )
            valid_TS = valid_TS & torch.eq(
                query_docs_T.unsqueeze(1), key_docs_S.unsqueeze(0)
            )
        if padding_mask_T is not None:
            valid_TS = valid_TS & ~padding_mask_T.unsqueeze(1)
            valid_TS = valid_TS & ~padding_mask_T.unsqueeze(0)
        scores_TS = scores_TS.masked_fill(~valid_TS, -torch.inf)
        values_TK, indices_TK = torch.topk(
            scores_TS, num_selected_slots, dim=-1, sorted=True
        )
        indices_TK = indices_TK.to(torch.int32).masked_fill(~values_TK.isfinite(), -1)

    if not return_scores:
        return indices_TK, None
    return indices_TK, _selected_index_scores(q_THDi, k_SDi, weights_TH, indices_TK)


def _gather_selected_rows(
    source_SD: torch.Tensor, indices_TK: torch.Tensor
) -> torch.Tensor:
    # Keep masked slots spread across rows: a long repeated padding index can
    # lose that row's FP32 gradient in the Ascend index_select backward kernel.
    padding_T = torch.arange(indices_TK.size(0), device=indices_TK.device)
    padding_T = padding_T.clamp_max(source_SD.size(0) - 1)
    safe_TK = torch.where(torch.ge(indices_TK, 0), indices_TK, padding_T[:, None])
    return source_SD.index_select(0, safe_TK.long().flatten()).unflatten(
        0, indices_TK.shape
    )


def _selected_index_scores(
    q_THDi: torch.Tensor,
    k_SDi: torch.Tensor,
    weights_TH: torch.Tensor,
    indices_TK: torch.Tensor,
) -> torch.Tensor:
    # Only selected scores need gradients for distillation.
    selected_keys_TKDi = _gather_selected_rows(k_SDi, indices_TK)
    selected_logits_THK = torch.einsum(
        "thd,tkd->thk", q_THDi.float(), selected_keys_TKDi.float()
    )
    return (
        selected_logits_THK.relu()
        .mul(weights_TH.float().unsqueeze(-1))
        .sum(dim=1)
        .masked_fill(indices_TK < 0, -torch.inf)
    )


class Indexer(Module):
    """Project shared keys and choose each DSA layer's own causal Top-K."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_index_heads: int
        index_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        apply_hadamard: bool = True

        def __post_init__(self) -> None:
            if min(self.num_index_heads, self.index_head_dim, self.index_topk) <= 0:
                raise ValueError("Kimi DSA index geometry must be positive.")
            if self.apply_hadamard and self.index_head_dim & (self.index_head_dim - 1):
                raise ValueError(
                    "Hadamard index head dimension must be a power of two."
                )
            if self.weights_proj.bias:
                raise ValueError(
                    "Kimi DSA requires a bias-free indexer head projection."
                )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.apply_hadamard = config.apply_hadamard
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()

    def _select(
        self,
        q_THDi: torch.Tensor,
        k_TDi: torch.Tensor,
        weights_TH: torch.Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata | None,
        padding_mask_T: torch.Tensor | None,
        return_scores: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return select_topk(
            q_THDi,
            k_TDi,
            weights_TH,
            top_k=self.index_topk,
            attention_metadata=attention_metadata,
            padding_mask_T=padding_mask_T,
            return_scores=return_scores,
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        q_latent_TA: torch.Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata | None,
        padding_mask_T: torch.Tensor | None,
        return_scores: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, IndexerFeatures]:
        q_THDi = self.wq_b(q_latent_TA).unflatten(
            -1, (self.num_index_heads, self.index_head_dim)
        )
        k_TDi = self.wk(x_TD)
        remat.recompute_needs_tensor(q_THDi, k_TDi)
        k_TDi = self.k_norm(k_TDi)
        weights_TH = F.linear(x_TD.float(), self.weights_proj.weight.float()) * (
            self.num_index_heads**-0.5 * self.index_head_dim**-0.5
        )
        if self.apply_hadamard:
            q_THDi = _hadamard(q_THDi)
            k_TDi = _hadamard(k_TDi)
        remat.recompute_needs_tensor(q_THDi, k_TDi)
        indices_TK, scores_TK = self._select(
            q_THDi,
            k_TDi,
            weights_TH,
            attention_metadata=attention_metadata,
            padding_mask_T=padding_mask_T,
            return_scores=return_scores,
        )
        return indices_TK, scores_TK, IndexerFeatures(q_THDi, k_TDi, weights_TH)


class KimiDSALatentAttention(InnerAttention):
    """Reference attention over selected shared MLA latent K/V entries."""

    __call__: Callable[..., torch.Tensor]
    training: bool
    requires_index_scores: bool = True

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        n_heads: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wkv_b: Linear.Config
        aux_loss: SparseIndexerLoss.Config | None = None

        def __post_init__(self) -> None:
            if self.wkv_b.bias:
                raise ValueError(
                    "Kimi DSA latent attention requires a bias-free wkv_b."
                )
            if (
                self.wkv_b.in_features != self.kv_lora_rank
                or self.wkv_b.out_features
                != self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
            ):
                raise ValueError(
                    "Kimi DSA latent attention has inconsistent wkv_b geometry."
                )
            if self.aux_loss is not None and self.aux_loss.compress_ratio != 1:
                raise ValueError("Kimi DSA indexer loss requires compress_ratio=1.")

        @property
        def is_full_attention(self) -> bool:
            return True

        @property
        def softmax_scale(self) -> float:
            return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5

        build_attention_metadata = staticmethod(create_varlen_metadata_for_document)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = config.softmax_scale
        self.wkv_b = config.wkv_b.build()
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def _attend(
        self,
        q_latent_THL: torch.Tensor,
        q_shared_THR: torch.Tensor,
        kv_latent_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        q_compact_THC = torch.cat((q_latent_THL, q_shared_THR), dim=-1).float()
        k_compact_TC = torch.cat((kv_latent_TL, k_shared_TR), dim=-1).float()
        selected_keys_TKC = _gather_selected_rows(k_compact_TC, topk_indices_TK)
        logits_THK = (
            torch.einsum("thc,tkc->thk", q_compact_THC, selected_keys_TKC)
            * self.softmax_scale
        )
        valid_TK = torch.ge(topk_indices_TK, 0)
        logits_THK = logits_THK.masked_fill(~valid_TK.unsqueeze(1), -torch.inf)
        lse_TH = torch.logsumexp(logits_THK, dim=-1)
        safe_lse_TH = torch.where(torch.isfinite(lse_TH), lse_TH, 0.0)
        probs_THK = torch.exp(logits_THK - safe_lse_TH.unsqueeze(-1))
        selected_values_TKL = _gather_selected_rows(
            kv_latent_TL.float(), topk_indices_TK
        )
        out_latent_THL = torch.einsum("thk,tkl->thl", probs_THK, selected_values_TKL)
        return out_latent_THL, lse_TH, q_compact_THC, k_compact_TC

    def _apply_aux_loss(
        self,
        q_compact_THC: torch.Tensor,
        k_compact_TC: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor | None,
        lse_TH: torch.Tensor,
        *,
        carrier: torch.Tensor,
        denominator: torch.Tensor,
        padding_mask_T: torch.Tensor | None,
        attention_metadata: VarlenAttentionMetadata | None,
        indexer_features: IndexerFeatures | None,
    ) -> torch.Tensor:
        assert self.aux_loss is not None and topk_scores_TK is not None
        return self.aux_loss(
            q_compact_THC.detach(),
            k_compact_TC.detach(),
            topk_indices_TK,
            topk_scores_TK,
            lse_TH.detach(),
            None,
            carrier=carrier,
            denominator=denominator,
            padding_mask_T=padding_mask_T,
        )

    def forward(
        self,
        q_THE: torch.Tensor,
        kv_latent_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        *,
        topk_indices_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor | None,
        attention_metadata: VarlenAttentionMetadata | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
        padding_mask_T: torch.Tensor | None = None,
        indexer_features: IndexerFeatures | None = None,
    ) -> torch.Tensor:
        with spmd.no_typecheck():
            num_tokens, num_heads = q_THE.shape[:2]
            if topk_indices_TK.shape[-1] == 0:
                out_THV = q_THE.new_zeros((num_tokens, num_heads, self.v_head_dim))
            else:
                q_nope_THN, q_shared_THR = q_THE.split(
                    (self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1
                )
                wkv_HPL = self.wkv_b.weight.view(
                    num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                    self.kv_lora_rank,
                )
                w_k_HNL, w_v_HVL = wkv_HPL.split(
                    (self.qk_nope_head_dim, self.v_head_dim), dim=1
                )

                # Absorb the key up-projection into Q. Project values only after reducing
                # the selected latent entries, preserving the original MLA weights.
                q_latent_THL = torch.einsum(
                    "thn,hnl->thl", q_nope_THN, w_k_HNL  # codespell:ignore thn
                )
                out_latent_THL, lse_TH, q_compact_THC, k_compact_TC = self._attend(
                    q_latent_THL,
                    q_shared_THR,
                    kv_latent_TL,
                    k_shared_TR,
                    topk_indices_TK,
                    attention_metadata,
                )
                out_THV = torch.einsum(
                    "thl,hvl->thv", out_latent_THL, w_v_HVL.float()
                ).to(q_THE.dtype)
                if self.training and self.aux_loss is not None:
                    if aux_loss_denominator is None:
                        raise ValueError(
                            "SparseIndexerLoss requires aux_loss_denominator."
                        )
                    assert q_compact_THC is not None and k_compact_TC is not None
                    out_THV = self._apply_aux_loss(
                        q_compact_THC,
                        k_compact_TC,
                        topk_indices_TK,
                        topk_scores_TK,
                        lse_TH,
                        carrier=out_THV,
                        denominator=aux_loss_denominator,
                        padding_mask_T=padding_mask_T,
                        attention_metadata=attention_metadata,
                        indexer_features=indexer_features,
                    )
        if spmd.is_type_checking():
            spmd.assert_type(
                out_THV, spmd.get_local_type(q_THE), spmd.get_partition_spec(q_THE)
            )
        return out_THV


class KimiDSAAttention(KimiMLAAttention):
    """Reuse the K3 MLA projections and gate with per-layer DSA selection."""

    __call__: Callable[..., torch.Tensor]
    training: bool
    q_head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    wq_a: Linear
    q_norm: RMSNorm
    wq_b: Linear
    wkv_a: Linear
    kv_norm: RMSNorm
    inner_attention: KimiDSALatentAttention
    gate: Linear
    wo: Linear

    @dataclass(kw_only=True, slots=True)
    class Config(KimiMLAAttention.Config):
        indexer: Indexer.Config
        inner_attention: KimiDSALatentAttention.Config

        def __post_init__(self) -> None:
            Attention.Config.__post_init__(self)
            if not isinstance(self.inner_attention, KimiDSALatentAttention.Config):
                raise ValueError("Kimi DSA requires KimiDSALatentAttention.Config.")

        @property
        def softmax_scale(self) -> float:
            return self.inner_attention.softmax_scale

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.indexer = config.indexer.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        *,
        aux_loss_denominator: torch.Tensor | None = None,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if not isinstance(attention_metadata, VarlenAttentionMetadata):
            raise ValueError("Kimi DSA requires packed varlen attention metadata.")
        if spmd_mesh_size("tp") > 1 or spmd_mesh_size("cp") > 1:
            raise ValueError("Kimi DSA reference attention requires TP=CP=1.")

        x_TD = maybe_gather_tp_input(self, x_TD)
        q_latent_TA = self.wq_a(x_TD)
        remat.recompute_needs_tensor(q_latent_TA)
        q_latent_TA = self.q_norm(q_latent_TA)
        q_THE = local_head_split(
            self.wq_b(q_latent_TA), self.q_head_dim, cp_shard_dim=0
        )
        remat.recompute_needs_tensor(q_THE)

        compressed_kv_TC = self.wkv_a(x_TD)
        remat.recompute_needs_tensor(compressed_kv_TC)
        kv_latent_TL, k_shared_TR = compressed_kv_TC.split(
            (self.kv_lora_rank, self.qk_rope_head_dim), dim=-1
        )
        kv_latent_TL = self.kv_norm(kv_latent_TL)

        inner_attention = self.inner_attention
        with spmd.no_typecheck():
            topk_indices_TK, topk_scores_TK, indexer_features = self.indexer(
                x_TD.detach(),
                q_latent_TA.detach(),
                attention_metadata=attention_metadata,
                padding_mask_T=padding_mask_T,
                return_scores=(
                    self.training
                    and inner_attention.aux_loss is not None
                    and inner_attention.requires_index_scores
                ),
            )
        out_THV = inner_attention(
            q_THE,
            kv_latent_TL,
            k_shared_TR,
            topk_indices_TK=topk_indices_TK,
            topk_scores_TK=topk_scores_TK,
            attention_metadata=attention_metadata,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask_T,
            indexer_features=indexer_features,
        )
        gate_TF = self.gate(x_TD)
        remat.recompute_needs_tensor(out_THV, gate_TF)
        return self.wo(out_THV.flatten(-2) * torch.sigmoid(gate_TF))
