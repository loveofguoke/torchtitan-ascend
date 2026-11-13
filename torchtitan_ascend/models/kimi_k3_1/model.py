# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-3-Clause file in the root directory of this source tree.

"""Kimi K3.1 decoder with packed KDA and gated DSA layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_remat as remat
from torchtitan.models.common import Linear, RMSNorm
from torchtitan.models.common.attention import AttentionMetadata
from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.flops import quadratic_attention_flops_per_token
from torchtitan.models.kimi_k3.model import (
    AttentionResidual,
    KimiK3Model,
    KimiK3TransformerBlock,
)
from torchtitan.models.kimi_k3.moe import KimiLatentMoE

from .dsa import KimiDSAAttention
from .state_dict_adapter import KimiK31StateDictAdapter

# T = packed tokens, D = model dimension, N = attention-residual entries.


class KimiK31TransformerBlock(KimiK3TransformerBlock):
    """Pass the training metadata through to a DSA attention layer."""

    first_layer_in_block: bool
    attention: KimiDSAAttention
    feed_forward: DenseFeedForward | KimiLatentMoE
    attention_norm: RMSNorm
    ffn_norm: RMSNorm
    attention_res_norm: RMSNorm | None
    attention_res_proj: Linear | None
    ffn_res_norm: RMSNorm
    ffn_res_proj: Linear
    attention_residual: AttentionResidual

    if TYPE_CHECKING:

        def remat_region_name(self, name: str) -> str:
            ...

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3TransformerBlock.Config):
        pass

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_metadata: AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.first_layer_in_block:
            block_residual_TND = torch.cat(
                (block_residual_TND, x_TD.unsqueeze(1)), dim=1
            )
            partial_block_TD = None
        else:
            partial_block_TD = x_TD

        if self.attention_res_proj is None:
            h_TD = x_TD
        else:
            assert self.attention_res_norm is not None
            h_TD = self.attention_residual(
                partial_block_TD,
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            )
        h_TD = self.attention_norm(h_TD)
        h_TD = self.attention(
            h_TD,
            attention_metadata,
            positions,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask,
        )
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(h_TD)
        prefix_sum_TD = h_TD if self.first_layer_in_block else x_TD + h_TD

        h_TD = self.attention_residual(
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if isinstance(self.feed_forward, KimiLatentMoE):
            h_TD = self.feed_forward(
                h_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            h_TD = self.feed_forward(h_TD)
        # The trailing add saves nothing for backward, so replay can skip it.
        out_TD = remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(prefix_sum_TD, h_TD)
        return out_TD, block_residual_TND


class KimiK31Model(KimiK3Model):
    """Reuse K3 input packing and residuals with the DSA loss step hook."""

    state_dict_adapter_cls = KimiK31StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3Model.Config):
        layers: list[KimiK3TransformerBlock.Config]

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """Replace the dense MLA estimate with absorbed DSA and index work."""
            nparams, flops = KimiK3Model.Config.get_nparams_and_flops(
                self, model, seq_len
            )
            for layer in self.layers:
                attention = layer.attention
                if isinstance(attention, KimiDSAAttention.Config):
                    qk_tail = attention.qk_rope_head_dim
                    latent = attention.kv_lora_rank
                    indexer = attention.indexer
                    flops -= quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.qk_nope_head_dim + qk_tail,
                        v_head_dim=attention.v_head_dim,
                        seq_len=seq_len,
                    )
                    flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=latent + qk_tail,
                        v_head_dim=latent,
                        seq_len=min(seq_len, indexer.index_topk),
                    )
                    flops += (
                        6 * indexer.num_index_heads * indexer.index_head_dim * seq_len
                    )
            return nparams, flops

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
