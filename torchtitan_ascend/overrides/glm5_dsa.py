# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ascend fused GLM DSA, including dense warmup and sparse indexer training."""

from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch_remat as remat
from torchtitan.config import derive, override
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.cp_attention import CPInnerAttention
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import maybe_gather_tp_input

from torchtitan_ascend.device import require_device
from torchtitan_ascend.models.glm5.dsa import DsaAttention, DsaMetadata, Glm5DsaIndexer
from torchtitan_ascend.models.glm5.model import Glm5Attention
from torchtitan_ascend.overrides.rope import npu_complex_rope

_SPARSE_KL_WIDTHS = tuple(range(1024, 8193, 1024))


def _gather(tensor, axis, dim):
    group = spmd_mesh_group(axis)
    if group is None:
        return tensor
    return spmd.redistribute(
        tensor,
        group,
        src=spmd.S(dim),
        dst=spmd.R,
        backward_options={"op_dtype": tensor.dtype},
    )


def _sparse_width(count):
    return max(1024, ((count + 1023) // 1024) * 1024)


def _sparse_kl_width(count):
    if count > 8192:
        raise ValueError("CANN Sparse KL supports at most 8192 selected keys per query")
    return _sparse_width(count)


def _all_causal_keys(packing, width, device):
    """Select every key in each packed document, with invalid slots at the end."""
    rows = []
    qstart = kstart = 0
    for qend, kend in zip(packing.query_lengths, packing.key_lengths, strict=True):
        qlen, klen = qend - qstart, kend - kstart
        keys = torch.arange(width, device=device, dtype=torch.int32)
        limits = torch.arange(qlen, device=device) + klen - qlen + 1
        rows.append(torch.where(keys[None, :] < limits[:, None], keys, -1))
        qstart, kstart = qend, kend
    return torch.cat(rows).unsqueeze(1).contiguous()


class _IndexerKL(torch.autograd.Function):
    """Use CANN's fused loss/gradient kernels as one differentiable KL objective."""

    @staticmethod
    def forward(ctx, qi, ki, weights, query, key, qr, kr, maximum, total, indices, qlens, klens, scale, dense):
        import torch_npu

        common = dict(
            query_rope=qr,
            key_rope=kr,
            actual_seq_qlen=list(qlens),
            actual_seq_klen=list(klens),
            layout="TND",
            sparse_mode=3,
        )
        if dense:
            imax, isum = torch_npu.npu_dense_lightning_indexer_softmax_lse(
                qi,
                ki,
                weights,
                actual_seq_qlen=list(qlens),
                actual_seq_klen=list(klens),
                layout="TND",
            )
            dqi, dki, dw, loss = torch_npu.npu_dense_lightning_indexer_grad_kl_loss(
                query,
                key,
                qi,
                ki,
                weights,
                maximum,
                total,
                imax,
                isum,
                scale,
                **common,
            )
        else:
            # The kernel can mask a causal KV tail, but cannot represent an
            # arbitrary smaller top-k by padding its selection with -1.
            if indices.shape[-1] not in _SPARSE_KL_WIDTHS:
                raise ValueError("CANN Sparse KL requires an index width of 1024..8192 in multiples of 1024")
            dqi, dki, dw, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                query,
                key,
                qi,
                ki,
                weights,
                indices,
                maximum,
                total,
                scale,
                **common,
            )
        ctx.save_for_backward(dqi, dki, dw)
        return loss.float().sum()

    @staticmethod
    def backward(ctx, grad):
        dqi, dki, dw = ctx.saved_tensors
        return dqi * grad, dki * grad, dw * grad, *(None for _ in range(11))


class AscendDsaIndexer(Glm5DsaIndexer):
    """Project indexer features without differentiating discrete top-k indices."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm5DsaIndexer.Config):
        pass

    def forward(self, hidden_states_TD, q_resid_TR, positions_T):
        # The indexer has its own objective. Its inputs and teacher are detached.
        x, q_resid = hidden_states_TD.detach(), q_resid_TR.detach()
        q = self.wq_b(q_resid).view(-1, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).unsqueeze(1)
        r = self.qk_rope_head_dim
        qr, kr = self.rope(q[..., :r], k[..., :r], positions_T)
        q = torch.cat((qr, q[..., r:]), -1)
        k = torch.cat((kr, k[..., r:]), -1)
        weights = self.weights_proj(x.float()).float()
        # The fused indexer has no scale argument. This is the original score
        # scale, including the per-head mixing normalization.
        weights = weights * (self.n_heads * self.head_dim) ** -0.5
        return q, k, weights


class AscendDsaAttention(DsaAttention):
    """SparseFlashAttention and packed-query metadata for CP KV all-gather."""

    @dataclass(kw_only=True, slots=True)
    class Config(DsaAttention.Config):
        pass

    def forward(self, query, key, indices, qr, kr, qlens, klens, scale):
        import torch_npu

        return torch_npu.npu_sparse_flash_attention(
            query.contiguous(),
            key.contiguous(),
            key.contiguous(),
            indices,
            scale,
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_kv=klens,
            query_rope=qr.contiguous(),
            key_rope=kr.contiguous(),
            layout_query="TND",
            layout_kv="TND",
            sparse_block_size=1,
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )


class AscendCPDsaAttention(CPInnerAttention[DsaMetadata, DsaMetadata], AscendDsaAttention):
    """CP variant with native DSA execution and packed metadata preparation."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, AscendDsaAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            return attention_metadata
        count = attention_metadata.token_count
        if count % group.size():
            raise ValueError("DSA token count must divide evenly across CP ranks")
        order = attention_metadata.order
        if permutation is not None:
            order = tuple(permutation.detach().cpu().reshape(-1).tolist())
            if len(order) != count or set(order) != set(range(count)):
                raise ValueError("DSA CP requires a permutation of the packed token axis")
        local = count // group.size()
        start = dist.get_rank(group) * local
        return replace(attention_metadata, order=order, query_start=start, query_end=start + local)


class DsaIndexerLoss(AuxLoss):
    """Per-token KL objective normalized by the trainer's valid-token count."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        coeff: float
        reduce_mesh: str = "loss"


class AscendGlm5Attention(Glm5Attention):
    """Checkpoint-compatible MLA absorption with native NPU DSA kernels."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm5Attention.Config):
        indexer_stage: str = "sparse"
        indexer_loss: DsaIndexerLoss.Config | None = None
        index_topk: int = 2048

    def __init__(self, config):
        require_device("Ascend910B", "Ascend910_93", feature="GLM fused DSA")
        super().__init__(config)
        self.indexer_stage = config.indexer_stage
        self.index_topk = config.index_topk
        self.indexer_loss = config.indexer_loss.build() if config.indexer_loss is not None else None
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError("CANN GLM DSA requires kv_lora_rank=512 and qk_rope_head_dim=64")
        if self.indexer is not None and self.indexer.head_dim != 128:
            raise ValueError("CANN LightningIndexer requires indexer head_dim=128")
        if self.indexer_loss is not None:
            assert self.indexer is not None
            if self.n_heads not in (32, 64, 128):
                raise ValueError("CANN DSA indexer KL requires 32, 64 or 128 global attention heads")
            if self.indexer.n_heads not in (8, 16, 32, 64):
                raise ValueError("CANN DSA indexer KL requires 8, 16, 32 or 64 indexer heads")
        torch.nn.Module.register_buffer(self, "causal_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def _init_self_buffers(self, *, buffer_device=None):
        if buffer_device is None:
            buffer_device = self.causal_mask.device
        if self.indexer_stage == "warmup" and self.qk_nope_head_dim == 128:
            self.causal_mask = torch.ones(2048, 2048, dtype=torch.bool, device=buffer_device).triu(1)
        else:
            self.causal_mask = torch.empty(0, dtype=torch.bool, device=buffer_device)

    def forward(
        self,
        x_TD,
        attention_masks,
        positions=None,
        *,
        topk_indices_TS=None,
        return_indices=False,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        x_TD = maybe_gather_tp_input(self, x_TD)
        q_resid = self.q_norm(self.wq_a(x_TD))
        q = self.wq_b(q_resid)
        kv = self.wkv_a(x_TD)
        remat.recompute_needs_tensor(q_resid, q, kv)
        # The fused backend is an opaque local-tensor compute boundary. Module
        # configs still own parameter sharding and the input/output collectives.
        with spmd.no_typecheck():
            num_tokens = x_TD.shape[0]
            q = q.view(num_tokens, -1, self.qk_head_dim)
            q_nope, qr = q.split((self.qk_nope_head_dim, self.qk_rope_head_dim), -1)
            latent, kr = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
            latent = self.kv_norm(latent)
            if positions is None:
                positions = torch.arange(num_tokens, device=x_TD.device)
            qr, kr = self.rope(qr, kr.unsqueeze(1), positions)
            kv_weight = self.wkv_b.weight.view(
                -1,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            wk, wv = kv_weight.split((self.qk_nope_head_dim, self.v_head_dim), 1)
            q_abs = torch.einsum("tnp,npc->tnc", q_nope, wk)
            latent = _gather(latent, MeshAxisName.CP, 0).unsqueeze(1)
            kr = _gather(kr, MeshAxisName.CP, 0)
            features = None
            if self.indexer is not None:
                qi, ki, weights = self.indexer(x_TD, q_resid, positions)
                ki = _gather(ki, MeshAxisName.CP, 0)
                features = (qi, ki, weights)
            output, indices = self._fused_attention(
                q_abs,
                latent,
                qr,
                kr,
                q_nope,
                wk,
                attention_masks,
                features,
                topk_indices_TS,
            )
            output = torch.einsum("tnc,nvc->tnv", output, wv).flatten(1)
        remat.recompute_needs_tensor(output)
        output = self.wo(output)
        return (output, indices) if return_indices else output

    def _fused_attention(self, query, key, qr, kr, q_nope, wk, metadata, features, previous_indices):
        import torch_npu

        packing = metadata.packing
        device = query.device
        qidx = torch.tensor(packing.queries, dtype=torch.int64, device=device)
        kidx = torch.tensor(packing.keys, dtype=torch.int64, device=device)
        width = self.index_topk
        if not packing.queries:
            # Empty CP ranks must still participate in every KV all-gather's
            # backward, including RoPE and the independently trained indexer.
            dependency = key.sum() + qr.sum() + kr.sum()
            if features is not None:
                dependency = dependency + sum(feature.sum() for feature in features)
            zeros = query * 0 + dependency * 0
            indices = torch.full((packing.query_count, width), -1, device=device, dtype=torch.int32)
            return zeros, indices
        q = query.index_select(0, qidx).contiguous()
        k = key.index_select(0, kidx).contiguous()
        qrope, krope = qr.index_select(0, qidx).contiguous(), kr.index_select(0, kidx).contiguous()
        qlens = torch.tensor(packing.query_lengths, device=device, dtype=torch.int32)
        klens = torch.tensor(packing.key_lengths, device=device, dtype=torch.int32)
        selected = None
        qi = ki = weights = None
        if features is not None:
            qi = features[0].index_select(0, qidx).contiguous()
            ki = features[1].index_select(0, kidx).contiguous()
            weights = features[2].index_select(0, qidx).contiguous()
            selected, _ = torch_npu.npu_lightning_indexer(
                qi.detach(),
                ki.detach(),
                weights.detach(),
                actual_seq_lengths_query=qlens,
                actual_seq_lengths_key=klens,
                layout_query="TND",
                layout_key="TND",
                sparse_count=width,
                sparse_mode=3,
            )
        elif previous_indices is not None:
            selected = previous_indices.index_select(0, qidx).unsqueeze(1).contiguous()
        else:
            raise ValueError("A shared-indexer layer requires the producer's DSA indices")

        indices = torch.full((packing.query_count, width), -1, device=device, dtype=torch.int32)
        indices = indices.index_copy(0, qidx, selected.squeeze(1))
        kmain = None
        selected_for_attention = selected
        dense = self.indexer_stage == "warmup" and self.qk_nope_head_dim == 128
        if dense:
            qmain = torch.cat((q_nope.index_select(0, qidx), qrope), -1).contiguous()
            kmain = torch.einsum("tjc,npc->tnp", k, wk)
            kfull = torch.cat((kmain, krope.expand(-1, kmain.shape[1], -1)), -1).contiguous()
            _, maximum, total, *_ = torch_npu.npu_fusion_attention(
                qmain,
                kfull,
                kmain.contiguous(),
                qmain.shape[1],
                "TND",
                atten_mask=self.causal_mask,
                scale=self.softmax_scale,
                actual_seq_qlen=list(packing.query_lengths),
                actual_seq_kvlen=list(packing.key_lengths),
                sparse_mode=3,
                softmax_layout="TND",
            )
            maximum, total = maximum.transpose(0, 1)[..., :1].contiguous(), total.transpose(0, 1)[..., :1].contiguous()
            longest = max(b - a for a, b in zip((0, *packing.key_lengths[:-1]), packing.key_lengths, strict=True))
            full_indices = _all_causal_keys(packing, _sparse_width(longest), device)
            out, _, _ = self.inner_attention(
                q,
                k,
                full_indices,
                qrope,
                krope,
                qlens,
                klens,
                self.softmax_scale,
            )
        else:
            if self.indexer_stage == "warmup":
                # CANN dense KL supports D=128 only. Full causal selection
                # gives the same dense objective for GLM's D=192 MLA.
                longest = max(b - a for a, b in zip((0, *packing.key_lengths[:-1]), packing.key_lengths, strict=True))
                selected_for_attention = _all_causal_keys(packing, _sparse_kl_width(longest), device)
            else:
                selected_for_attention = selected
            out, maximum, total = self.inner_attention(
                q,
                k,
                selected_for_attention,
                qrope,
                krope,
                qlens,
                klens,
                self.softmax_scale,
            )
        if self.training and self.indexer_loss is not None:
            # Indexer targets sum over all attention heads. Gather teacher
            # heads, including under TP4 where a local shard has only 16 heads.
            if dense:
                assert kmain is not None
                teacher_q = _gather(q_nope.detach(), MeshAxisName.TP, 1).index_select(0, qidx).contiguous()
                teacher_k = _gather(kmain.detach(), MeshAxisName.TP, 1)
                teacher_qr = _gather(qrope.detach(), MeshAxisName.TP, 1)
                teacher_kr = krope.expand(-1, teacher_k.shape[1], -1).contiguous()
                maximum = _gather(maximum.detach(), MeshAxisName.TP, 0)
                total = _gather(total.detach(), MeshAxisName.TP, 0)
                kl_indices = selected
            else:
                teacher_q = _gather(q.detach(), MeshAxisName.TP, 1)
                teacher_k = k.detach()
                teacher_qr = _gather(qrope.detach(), MeshAxisName.TP, 1)
                teacher_kr = krope.detach()
                maximum = _gather(maximum.detach(), MeshAxisName.TP, 2)
                total = _gather(total.detach(), MeshAxisName.TP, 2)
                kl_indices = selected_for_attention
            loss = _IndexerKL.apply(
                qi,
                ki,
                weights,
                teacher_q,
                teacher_k,
                teacher_qr,
                teacher_kr,
                maximum,
                total,
                kl_indices,
                packing.query_lengths,
                packing.key_lengths,
                self.softmax_scale,
                dense,
            )
            out = self.indexer_loss.inject(loss, carrier=out)
        restored = query.new_zeros(query.shape).index_copy(0, qidx, out)
        return restored, indices


@override(target=Glm5Attention.Config, description="Ascend fused MLA/DSA with two-stage indexer training")
def npu_glm5_dsa(cfg, *, stage="sparse", indexer_loss_coeff=1.0):
    if stage not in ("frozen", "warmup", "sparse"):
        raise ValueError("DSA stage must be frozen, warmup or sparse")
    if indexer_loss_coeff < 0:
        raise ValueError("DSA indexer_loss_coeff must be non-negative")
    if not (1 <= cfg.index_topk <= 2048 or cfg.index_topk in (3072, 4096, 5120, 6144, 7168, 8192)):
        raise ValueError("DSA top-k is outside the fixed CANN stack's supported range")
    if stage == "sparse" and cfg.index_topk not in _SPARSE_KL_WIDTHS:
        raise ValueError(
            "Sparse indexer training requires index_topk=1024..8192 in multiples of 1024; padding is unsupported"
        )
    indexer = cfg.indexer
    if indexer is not None:
        indexer = derive(
            indexer,
            AscendDsaIndexer.Config,
            rope=npu_complex_rope(indexer.rope),
            trainable=stage != "frozen",
            sharding_config=None,
        )
    result = derive(
        cfg,
        AscendGlm5Attention.Config,
        rope=npu_complex_rope(cfg.rope),
        indexer=indexer,
        inner_attention=derive(
            cfg.inner_attention,
            AscendCPDsaAttention.Config if cfg.inner_attention.enable_cp else AscendDsaAttention.Config,
            sharding_config=None,
        ),
        indexer_stage=stage,
        indexer_loss=(
            DsaIndexerLoss.Config(coeff=indexer_loss_coeff) if indexer is not None and stage != "frozen" else None
        ),
        index_topk=cfg.index_topk,
    )
    result.sharding_config = replace(
        result.sharding_config,
        state_shardings={**result.sharding_config.state_shardings, "causal_mask": dense_param_placement(tp=spmd.R)},
    )
    return result
