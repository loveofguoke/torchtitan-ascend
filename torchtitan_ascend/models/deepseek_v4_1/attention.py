# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Adapt DeepSeek V4 attention projections to CSA2;
# 2. Thread compressed KV and indexer state across layers;
# 3. Preserve packed metadata, LSE, aux-loss denominator and TP/SP contracts.
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

"""DeepSeek V4.1 attention: CSA2 on top of V4's shared CSA core.

The CSA family shares one core, upstream V4's ``DSV4InnerAttention._gather_attn``:
a causal sliding window over the uncompressed KV, the gathered compressed KV
entries listed in ``cmp_topk``, and a learned per-head attention sink, computed
by Attention Gym's ``gather_attn``; the indexer distillation loss rides the
kernel's LSE through ``SparseIndexerLoss``.

The two variants differ only in where the top-k selection comes from:

- CSA1 (upstream V4 ``CompressedSparseAttention``) selects in-layer: its own
  ``Indexer`` projects the queries/keys and ``Indexer.select`` picks the top-k.
- CSA2 (``CompressedSparseAttention2`` below) receives the selection from
  V4.1's cross-layer :class:`HierarchicalIndexer`, which produces one shared
  top-k per compression group instead of one per layer.

The whole stream is packed ``[T, *]``. Document offsets come from upstream
``VarlenAttentionMetadata`` (``cu_seq_q``); compressed consumers derive their pool
offsets per ratio with ``compressed_cu_seqlens(cu_q, ratio)``.
"""

from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.common.attention import VarlenAttentionMetadata
from torchtitan.models.common.attn_gym import AuxRequest, gather_attn
from torchtitan.models.deepseek_v4.attention import (
    _assert_spmd_attention_type,
    DeepSeekV4Attention,
    DSV4InnerAttention,
)
from torchtitan.models.deepseek_v4.compressor import compressed_cu_seqlens

from .compressor import Compressor
from .indexer import HierarchicalIndexer


if TYPE_CHECKING:
    from torchtitan.models.common import Linear, RMSNorm, RoPE
    from torchtitan.models.deepseek_v4.attention import SlidingWindowAttention
    from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss


class V41InnerAttention(DSV4InnerAttention):
    """Base for V4.1 inner attention on the packed varlen contract.

    Upstream V4 reads ``VarlenAttentionMetadata.cu_seq_q`` and derives the compressed
    pool offsets as ``compressed_cu_seqlens(cu, ratio)`` -- except at
    ``ratio == 1``, where V4's SWA path substitutes zeros because its
    compressed pool is empty. V4.1's ratio-1 CSA2 layers own a non-empty pool,
    so every ratio derives its real per-document pool offsets here.
    """

    aux_loss: "SparseIndexerLoss | None"
    training: bool
    window_size: int
    softmax_scale: float
    compress_ratio: int

    def _gather_attn(
        self,
        q_THD,
        swa_k_TD,
        cmp_k_SD,
        cmp_topk_TK,
        attn_sink,
        cu_seqlens,
        *,
        cu_seqlens_k=None,
        topk_scores_TK=None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ) -> torch.Tensor:
        """Same as V4's core, with the pool offsets taken from the caller.

        ``cmp_topk_TK`` holds document-local indices into ``cmp_k_SD``; -1
        marks unused slots.
        """
        with spmd.no_typecheck():
            aux_loss = self.aux_loss if self.training else None
            out_1HTD, aux = gather_attn(
                q_THD.transpose(0, 1).unsqueeze(0),
                swa_k_TD[None, None],
                cmp_k_SD[None, None],
                cmp_topk_TK.unsqueeze(0),
                attention_sink=attn_sink,
                sliding_window_size=self.window_size,
                cu_seqlens=cu_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                scale=self.softmax_scale,
                impl="fused" if q_THD.device.type == "cuda" else "reference",
                return_aux=AuxRequest(lse=aux_loss is not None),
            )
            out_THD = out_1HTD.squeeze(0).transpose(0, 1)
            if aux_loss is not None:
                assert topk_scores_TK is not None and aux.lse is not None
                if aux_loss_denominator is None:
                    raise RuntimeError(
                        "SparseIndexerLoss requires aux_loss_denominator."
                    )
                out_THD = aux_loss(
                    q_THD.detach(),
                    cmp_k_SD.detach(),
                    cmp_topk_TK,
                    topk_scores_TK,
                    aux.lse.squeeze(0).transpose(0, 1).detach(),
                    cu_seqlens,
                    carrier=out_THD,
                    denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask_T,
                )
        # Kernel output is opaque to SPMD typechecking; it keeps q's layout.
        if spmd.is_type_checking():
            spmd.assert_type(
                out_THD, spmd.get_local_type(q_THD), spmd.get_partition_spec(q_THD)
            )
        return out_THD


class CompressedSparseAttention2(V41InnerAttention):
    """CSA2: the CSA core fed by V4.1's cross-layer hierarchical indexer.

    Unlike CSA1, nothing is selected here: ``topk_indices_TK``/``topk_scores_TK``
    arrive from the :class:`HierarchicalIndexer` chain and are consumed as-is.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(V41InnerAttention.Config):
        pass

        if TYPE_CHECKING:
            # Inherited from TorchTitan; keep runtime dataclass fields unchanged.
            window_size: int
            compress_ratio: int
            softmax_scale: float
            index_topk: int
            aux_loss: SparseIndexerLoss.Config | None = None

    def forward(
        self,
        q_THD,
        swa_k_TD,
        cmp_k_SD,
        topk_indices_TK,
        topk_scores_TK,
        attn_sink,
        *,
        attention_metadata: VarlenAttentionMetadata | None = None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ) -> torch.Tensor:
        cu_seqlens = None
        cu_seqlens_k = None
        if attention_metadata is not None:
            cu_seqlens = attention_metadata.cu_seq_q
            cu_seqlens_k = compressed_cu_seqlens(cu_seqlens, self.compress_ratio)
        return self._gather_attn(
            q_THD,
            swa_k_TD,
            cmp_k_SD,
            topk_indices_TK,
            attn_sink,
            cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            topk_scores_TK=topk_scores_TK,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask_T,
        )

    if TYPE_CHECKING:
        __call__ = forward


class Attention(DeepSeekV4Attention):
    """V4.1 attention wrapper: V4's projections, V4.1's cross-layer state.

    ``__init__`` and the projection submodules are inherited from upstream V4.
    ``forward`` is overridden because V4.1's compressor and hierarchical
    indexer are wired across layers (a source layer emits the compressed KV /
    index keys / top-k, later layers reuse them), while V4's are per-layer.

    Everything runs on the packed ``[T, *]`` stream; there is no ``[B, L]``
    fold/restore.
    """

    rope_head_dim: int
    head_dim: int
    norm_eps: float
    compress_ratio: int
    n_groups: int
    n_heads: int
    o_lora_rank: int
    wq_a: "Linear"
    wq_b: "Linear"
    wkv: "Linear"
    wo_a: "Linear"
    wo_b: "Linear"
    attn_sink: "Linear"
    q_norm: "RMSNorm"
    kv_norm: "RMSNorm"
    rope: "RoPE"
    compressor: Compressor | None
    indexer: HierarchicalIndexer | None
    inner_attention: "CompressedSparseAttention2 | SlidingWindowAttention"

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Attention.Config):
        inner_attention: V41InnerAttention.Config
        compressor: Compressor.Config | None = None
        indexer: HierarchicalIndexer.Config | None = None

        if TYPE_CHECKING:
            # Inherited from TorchTitan; keep runtime dataclass fields unchanged.
            n_heads: int
            dim: int
            rope: RoPE.Config
            head_dim: int = 512
            rope_head_dim: int = 64
            q_lora_rank: int = 1024
            o_lora_rank: int = 1024
            n_groups: int = 8
            compress_ratio: int = 1
            norm_eps: float = 1e-06
            index_n_heads: int = 64
            index_head_dim: int = 128
            n_layers: int = 4
            layer_id: int = 0
            mask_type: str = "causal"
            wq_a: Linear.Config
            q_norm: RMSNorm.Config
            wq_b: Linear.Config
            wkv: Linear.Config
            kv_norm: RMSNorm.Config
            wo_a: Linear.Config
            wo_b: Linear.Config
            attn_sink: Linear.Config
            compressor_128: Compressor.Config | None = None

    def forward(
        self,
        x,
        attention_metadata: VarlenAttentionMetadata | None = None,
        positions=None,
        aux_loss_denominator=None,
        padding_mask=None,
        *,
        cmp_k=None,
        idx_k=None,
        topk_indices=None,
        topk_scores=None,
        candidates=None,
    ):
        """Apply one V4.1 attention layer.

        Returns ``(o, cmp_k, idx_k, topk_indices, topk_scores, candidates)``:
        the attention output plus the cross-layer sparse-indexer state.
        """
        rd = self.rope_head_dim

        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            x = spmd.redistribute(
                x,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )
        num_tokens = x.size(0)
        if positions is None:
            positions = torch.arange(num_tokens, dtype=torch.int32, device=x.device)
            if attention_metadata is not None:
                cu_seqlens = attention_metadata.cu_seq_q
                positions = positions - torch.repeat_interleave(
                    cu_seqlens[:-1], torch.diff(cu_seqlens), output_size=num_tokens
                )

        qr = self.wq_a(x)
        # q_norm reads the wq_a projection output with bare ops.
        remat.recompute_needs_tensor(qr)
        qr = self.q_norm(qr)
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(num_tokens, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(1))
        # The rescale reads the wq_b projection output with bare ops.
        remat.recompute_needs_tensor(q)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.wkv(x)
        # kv_norm reads the wkv projection output with bare ops.
        remat.recompute_needs_tensor(kv)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        # V4.1's cross-layer sparse state: source layers emit, later reuse.
        cu_seqlens = None if attention_metadata is None else attention_metadata.cu_seq_q
        if self.compressor is not None:
            cmp_k, latent = self.compressor(
                x, positions, cu_seqlens=cu_seqlens, cmp_k=cmp_k
            )
        else:
            latent = None
        if self.indexer is not None:
            idx_k, topk_indices, topk_scores, candidates = self.indexer(
                x.detach(),
                qr.detach(),
                positions,
                attention_metadata,
                latent=latent,
                idx_k=idx_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
                candidates=candidates,
            )

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if self.compress_ratio > 0:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                topk_indices,
                topk_scores,
                attn_sink_param,
                attention_metadata=attention_metadata,
                aux_loss_denominator=aux_loss_denominator,
                padding_mask_T=padding_mask,
            )
        else:
            o = cast("SlidingWindowAttention", self.inner_attention)(
                q,
                kv,
                attn_sink_param,
                attention_metadata=attention_metadata,
            )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)

        with spmd.local():
            n_local_heads = o.shape[1]
            n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
            o = o.view(num_tokens, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
            wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            if spmd.is_type_checking():
                spmd.assert_type(
                    wo_a,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)},
                )
        o = torch.einsum("tgd,grd->tgr", o, wo_a)
        with spmd.local():
            o = o.reshape(num_tokens, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
        o = self.wo_b(o)
        return o, cmp_k, idx_k, topk_indices, topk_scores, candidates

    if TYPE_CHECKING:
        __call__ = forward


__all__ = [
    "Attention",
    "CompressedSparseAttention2",
]
