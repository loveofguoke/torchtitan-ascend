# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi indexer KL and gradients using detached SFA teacher statistics.

Shapes: T = packed tokens, H = heads, C = compact channels, Di = index channels,
K = selected slots, L = latent channels, R = shared tail channels. The native
teacher uses supported padded head count G.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenAttentionMetadata

from torchtitan_ascend.models.kimi_k3_1.dsa import (
    IndexerFeatures,
    KimiDSALatentAttention,
)
from torchtitan_ascend.overrides.kimi_k3_1.dsa import (
    _local_sparse_indices,
    AscendDSALatentAttention,
)

_SUPPORTED_HEADS = (32, 48, 64, 128)


class _IndexerKL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, qi, ki, weights, q, k, qr, kr, indices, maximum, cu_q, cu_k, scale
    ):
        import torch_npu

        dq, dk, dw, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            q.contiguous(),
            k.unsqueeze(1).contiguous(),
            qi.contiguous(),
            ki.unsqueeze(1).contiguous(),
            weights.contiguous(),
            indices.unsqueeze(1).contiguous(),
            maximum.unsqueeze(0).contiguous(),
            torch.ones_like(maximum).unsqueeze(0),
            scale,
            query_rope=qr.contiguous(),
            key_rope=kr.unsqueeze(1).contiguous(),
            actual_seq_qlen=cu_q[1:].tolist(),
            actual_seq_klen=cu_k[1:].tolist(),
            layout="TND",
            sparse_mode=3,
        )
        ctx.save_for_backward(dq, dk.squeeze(1), dw)
        return loss.sum()

    @staticmethod
    def backward(ctx, gradient):
        dq, dk, dw = ctx.saved_tensors
        # AuxLoss.inject supplies coeff and the step-global denominator once.
        return (
            dq * gradient,
            dk * gradient,
            dw * gradient,
            *([None] * 9),
        )


class AscendLIGAttention(AscendDSALatentAttention):
    requires_index_scores: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(AscendDSALatentAttention.Config):
        def __post_init__(self) -> None:
            AscendDSALatentAttention.Config.__post_init__(self)
            if not 1 <= self.n_heads <= max(_SUPPORTED_HEADS):
                raise ValueError("Kimi LIG_KL requires 1 to 128 teacher heads.")
            if self.aux_loss is None or self.aux_loss.mass_weighted:
                raise ValueError("Kimi LIG_KL requires unweighted SparseIndexerLoss.")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_heads = config.n_heads
        self.num_kernel_heads = next(n for n in _SUPPORTED_HEADS if n >= config.n_heads)

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
        assert self.aux_loss is not None
        assert attention_metadata is not None and indexer_features is not None
        qi_THDi, ki_TDi, weights_TH = indexer_features
        valid_TK = torch.ge(topk_indices_TK, 0)
        if padding_mask_T is not None:
            valid_TK = valid_TK & ~padding_mask_T[:, None]
        local_TK, empty_T, query_positions_T = _local_sparse_indices(
            topk_indices_TK, valid_TK, attention_metadata
        )
        counts_T = (query_positions_T + 1).clamp_max(local_TK.size(1))
        slots_K = torch.arange(local_TK.size(1), device=local_TK.device)
        safe_TK = slots_K.expand(local_TK.shape).masked_fill(
            torch.ge(slots_K[None, :], counts_T[:, None]), -1
        )
        local_TK = torch.where(empty_T[:, None], safe_TK, local_TK).int()

        # Empty rows use identical uniform teacher/student distributions. Their
        # zero query and weight also make every indexer gradient contribution zero.
        teacher_THC = q_compact_THC.detach().masked_fill(empty_T[:, None, None], 0)
        qi_THDi = qi_THDi.masked_fill(empty_T[:, None, None], 0)
        weights_TH = weights_TH.masked_fill(empty_T[:, None], 0)
        maximum_TH = torch.where(
            empty_T[:, None], counts_T.float().log()[:, None], lse_TH.detach()
        )
        # Native LIG averages G heads without renormalizing missing head mass.
        # Zero-probability padding therefore also needs a G/H teacher correction.
        maximum_TH = maximum_TH - math.log(self.num_kernel_heads / self.num_heads)
        pad = self.num_kernel_heads - self.num_heads
        teacher_TGC = F.pad(teacher_THC, (0, 0, 0, pad))
        maximum_TG = F.pad(maximum_TH, (0, pad), value=torch.inf)
        q_TGL, qr_TGR = teacher_TGC.split(
            (self.kv_lora_rank, self.qk_rope_head_dim), -1
        )
        k_TL, kr_TR = k_compact_TC.detach().split(
            (self.kv_lora_rank, self.qk_rope_head_dim), -1
        )
        raw_sum = _IndexerKL.apply(
            qi_THDi,
            ki_TDi,
            weights_TH,
            q_TGL,
            k_TL,
            qr_TGR,
            kr_TR,
            local_TK,
            maximum_TG,
            attention_metadata.cu_seq_q,
            attention_metadata.cu_seq_k,
            self.softmax_scale,
        )
        return self.aux_loss.inject(raw_sum, carrier=carrier, denominator=denominator)


@override(
    target=KimiDSALatentAttention.Config,
    exact=True,
    description="Packed SFA with fused indexer gradients and KL loss",
)
def npu_lig_kl(
    config: KimiDSALatentAttention.Config,
) -> AscendLIGAttention.Config:
    return derive(config, AscendLIGAttention.Config)
