# SPDX-License-Identifier: Apache-2.0
"""Packed sparse flash attention."""

from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenAttentionMetadata

from torchtitan_ascend.device import require_device
from torchtitan_ascend.models.kimi_k3_1.dsa import (
    _document_ids,
    _selected_index_scores,
    Indexer,
    KimiDSALatentAttention,
)

_CAUSAL = 3
_ABSORBED_MLA = 2
_TOKEN_BLOCK = 1


class _SparseFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, q_tail, k_tail, indices, cu_q, cu_k, scale):
        import torch_npu

        out, maximum, total = torch_npu.npu_sparse_flash_attention(
            q,
            kv,
            kv,
            indices,
            scale,
            actual_seq_lengths_query=cu_q,
            actual_seq_lengths_kv=cu_k,
            query_rope=q_tail,
            key_rope=k_tail,
            sparse_block_size=_TOKEN_BLOCK,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=_CAUSAL,
            attention_mode=_ABSORBED_MLA,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(
            q, kv, q_tail, k_tail, indices, cu_q, cu_k, out, maximum, total
        )
        ctx.scale = scale
        ctx.mark_non_differentiable(maximum, total)
        return out, maximum, total

    @staticmethod
    def backward(ctx, d_out, _d_maximum, _d_total):
        import torch_npu

        (
            q,
            kv,
            q_tail,
            k_tail,
            indices,
            cu_q,
            cu_k,
            out,
            maximum,
            total,
        ) = ctx.saved_tensors
        dq, dk, dv, dq_tail, dk_tail = torch_npu.npu_sparse_flash_attention_grad(
            q,
            kv,
            kv,
            indices,
            d_out.contiguous(),
            out,
            maximum,
            total,
            ctx.scale,
            _TOKEN_BLOCK,
            query_rope=q_tail,
            key_rope=k_tail,
            actual_seq_qlen=cu_q,
            actual_seq_kvlen=cu_k,
            layout="TND",
            sparse_mode=_CAUSAL,
            attention_mode=_ABSORBED_MLA,
        )
        # Passing value=None on CANN 9.2 and 9.3 drops the value contribution.
        dkv = (dk.float() + dv.float()).to(kv.dtype)
        return dq, dkv, dq_tail, dk_tail, None, None, None, None


def _packed_key_offsets(
    num_queries: int, metadata: VarlenAttentionMetadata
) -> torch.Tensor:
    positions = torch.arange(
        num_queries, dtype=torch.int32, device=metadata.cu_seq_q.device
    )
    return metadata.cu_seq_k.index_select(
        0, _document_ids(metadata.cu_seq_q, positions)
    )


def _local_sparse_indices(
    indices_TK: torch.Tensor,
    valid_TK: torch.Tensor,
    metadata: VarlenAttentionMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return document-local key indices, empty rows, and query positions."""
    empty_T = ~valid_TK.any(dim=-1)
    offsets_T = _packed_key_offsets(indices_TK.size(0), metadata)
    local_TK = (indices_TK - offsets_T[:, None]).masked_fill(~valid_TK, -1)
    positions_T = torch.arange(
        indices_TK.size(0), device=indices_TK.device, dtype=torch.int32
    )
    query_offsets_T = metadata.cu_seq_q.index_select(
        0, _document_ids(metadata.cu_seq_q, positions_T)
    )
    return local_TK, empty_T, positions_T - query_offsets_T


class AscendDSALatentAttention(KimiDSALatentAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(KimiDSALatentAttention.Config):
        def __post_init__(self) -> None:
            KimiDSALatentAttention.Config.__post_init__(self)
            if (self.kv_lora_rank, self.qk_rope_head_dim) != (512, 64):
                raise ValueError("SFA requires a 512-wide latent and 64-wide tail.")

    def __init__(self, config: Config) -> None:
        require_device("Ascend950", feature="Sparse flash attention")
        super().__init__(config)

    def _attend(
        self,
        q_latent_THL: torch.Tensor,
        q_shared_THR: torch.Tensor,
        kv_latent_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if attention_metadata is None:
            raise ValueError("SFA requires packed varlen metadata.")
        top_k = topk_indices_TK.size(1)
        if not 1024 <= top_k <= 8192 or top_k % 1024:
            raise ValueError("SFAG requires Top-K to be a multiple of 1024 up to 8192.")
        if q_latent_THL.device.type != "npu" or q_latent_THL.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError("SFA requires BF16 or FP16 Ascend tensors.")

        valid_TK = torch.ge(topk_indices_TK, 0)
        local_TK, empty_T, query_positions_T = _local_sparse_indices(
            topk_indices_TK, valid_TK, attention_metadata
        )
        slots_K = torch.arange(
            top_k, device=query_positions_T.device, dtype=torch.int32
        )
        safe_TK = slots_K.expand(local_TK.shape).masked_fill(
            torch.gt(slots_K[None, :], query_positions_T[:, None]), -1
        )
        local_TK = torch.where(empty_T[:, None], safe_TK, local_TK)
        out_THL, maximum_1TH, total_1TH = _SparseFlashAttention.apply(
            q_latent_THL.contiguous(),
            kv_latent_TL.unsqueeze(1).contiguous(),
            q_shared_THR.contiguous(),
            k_shared_TR.unsqueeze(1).contiguous(),
            local_TK.unsqueeze(1).contiguous(),
            attention_metadata.cu_seq_q[1:].contiguous(),
            attention_metadata.cu_seq_k[1:].contiguous(),
            self.softmax_scale,
        )
        out_THL = out_THL.masked_fill(empty_T[:, None, None], 0)
        lse_TH = (maximum_1TH[0] + total_1TH[0].log()).masked_fill(
            empty_T[:, None], -torch.inf
        )
        q_compact_THC = k_compact_TC = None
        if self.training and self.aux_loss is not None:
            q_compact_THC = torch.cat(
                (q_latent_THL.detach(), q_shared_THR.detach()), dim=-1
            )
            k_compact_TC = torch.cat(
                (kv_latent_TL.detach(), k_shared_TR.detach()), dim=-1
            )
        return out_THL.float(), lse_TH, q_compact_THC, k_compact_TC


@override(
    target=KimiDSALatentAttention.Config,
    exact=True,
    description="Packed Ascend sparse flash attention and gradient",
)
def ascendc_sfa(
    config: KimiDSALatentAttention.Config,
) -> AscendDSALatentAttention.Config:
    return derive(config, AscendDSALatentAttention.Config)


class AscendLightningIndexer(Indexer):
    @dataclass(kw_only=True, slots=True)
    class Config(Indexer.Config):
        def __post_init__(self) -> None:
            Indexer.Config.__post_init__(self)
            if self.index_head_dim != 128 or self.num_index_heads not in (
                8,
                16,
                24,
                32,
                64,
            ):
                raise ValueError("Unsupported Ascend 950 Lightning Indexer geometry.")
            if self.index_topk > 2048 and (
                self.index_topk > 8192 or self.index_topk % 1024
            ):
                raise ValueError("Unsupported Ascend Lightning Indexer Top-K.")

    def __init__(self, config: Config) -> None:
        require_device("Ascend950", feature="Lightning Indexer")
        super().__init__(config)

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
        import torch_npu

        if attention_metadata is None:
            raise ValueError("Lightning Indexer requires packed varlen metadata.")
        if q_THDi.device.type != "npu" or q_THDi.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError("Lightning Indexer requires BF16 or FP16 Ascend tensors.")
        with torch.no_grad():
            offsets_T = _packed_key_offsets(q_THDi.size(0), attention_metadata)
            if padding_mask_T is not None:
                # Packed loaders represent padding as separate document segments.
                # LI cannot exclude an individual padded key inside a live document.
                torch._assert_async(
                    torch.eq(
                        padding_mask_T, padding_mask_T.index_select(0, offsets_T)
                    ).all(),
                    "Lightning Indexer requires document-aligned padding.",
                )
            local_T1K, _ = torch_npu.npu_lightning_indexer(
                q_THDi.contiguous(),
                k_TDi.unsqueeze(1).contiguous(),
                weights_TH.to(q_THDi.dtype).contiguous(),
                actual_seq_lengths_query=attention_metadata.cu_seq_q[1:].contiguous(),
                actual_seq_lengths_key=attention_metadata.cu_seq_k[1:].contiguous(),
                layout_query="TND",
                layout_key="TND",
                sparse_count=self.index_topk,
                sparse_mode=_CAUSAL,
                return_value=False,
            )
            local_TK = local_T1K[:, 0, : min(self.index_topk, k_TDi.size(0))]
            valid_TK = torch.ge(local_TK, 0)
            if padding_mask_T is not None:
                valid_TK = valid_TK & ~padding_mask_T[:, None]
            indices_TK = (local_TK + offsets_T[:, None]).masked_fill(~valid_TK, -1)

        scores_TK = None
        if return_scores:
            scores_TK = _selected_index_scores(q_THDi, k_TDi, weights_TH, indices_TK)
        return indices_TK, scores_TK


@override(
    target=Indexer.Config,
    exact=True,
    description="Packed Ascend Lightning Indexer selection",
)
def ascendc_lightning_indexer(
    config: Indexer.Config,
) -> AscendLightningIndexer.Config:
    return derive(config, AscendLightningIndexer.Config)
