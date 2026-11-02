# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
#
# Modification points:
# 1. Implement V4.1 block and model forward with cross-layer CSA2 state;
# 2. Remove HcHead and MTP, and disable unsupported PP/CP paths;
# 3. Route attention metadata, aux-loss denominators and model FLOPs.
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

"""DeepSeek V4.1 backbone: the text stack.

Both the transformer block and the model inherit upstream V4
(``torchtitan.models.deepseek_v4.model``). The block reuses V4's submodule
structure (HC pre/post mixing around attention and the MoE feed-forward)
and overrides ``forward`` for V4.1's single-pass mHC and the cross-layer
sparse-indexer state. The model reuses V4's skeleton (optimizer hooks, the
FSDP application, TP/CP validation, document metadata construction and the
decoder config fields) and overrides the pieces V4.1 does not share: no
HcHead merge, no MTP stack.

The whole stream is packed ``[T, *]``: tokens/positions/padding are
``[T]``, the mHC stream is ``[T, hc, D]`` and the attention core is the
packed ``[T, H, Dk]`` stream. Document offsets come from upstream
``VarlenAttentionMetadata`` (``cu_seq_q``); compressed consumers derive their
per-ratio pool offsets with ``compressed_cu_seqlens(cu_q, ratio)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn
from torchtitan.models.base.module import ModuleList
from torchtitan.models.common.attention import AttentionMetadataMap, VarlenAttentionMetadata
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.deepseek_v4.mhc import HcHead
from torchtitan.models.deepseek_v4.model import (
    DeepSeekV4Model,
    DeepSeekV4TransformerBlock,
)

from .indexer import IndexerMode  # noqa: F401  (re-exported for flavors)
from .mhc import HcPost, HcPre
from .state_dict_adapter import DeepSeekV41StateDictAdapter

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE

    from .attention import Attention


def compression_alignment(compress_ratios: tuple[int, ...]) -> int:
    """The ``A`` a packed document length must be a multiple of, ``1`` when none."""
    alignment = 1
    for ratio in compress_ratios:
        if ratio > 0:
            alignment = math.lcm(alignment, ratio)
    return alignment


class DeepSeekV41TransformerBlock(DeepSeekV4TransformerBlock):
    """V4.1 block: inherits V4's Config/__init__, overrides forward.

    V4.1 threads the mHC pre-mixing coefficients and the sparse-indexer
    state (compressed KV, index keys, top-k selection, candidate pool)
    across layers, so forward takes and returns them explicitly.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4TransformerBlock.Config):
        attention: "Attention.Config"  # pyrefly: ignore [bad-override]
        feed_forward: "MoE.Config"  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        pre_mix: torch.Tensor,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(x, pre_mix, cmp_k, idx_k, topk_indices, topk_scores, candidates)``."""
        residual = x
        x, attn_pre, post, comb = self.hc_attn_pre(x, pre_mix)
        x, cmp_k, idx_k, topk_indices, topk_scores, candidates = self.attention(
            self.attention_norm(x),
            attention_metadata,
            positions,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask=padding_mask,
            cmp_k=cmp_k,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, ffn_pre, post, comb = self.hc_ffn_pre(x, attn_pre)
        x = self.feed_forward(
            self.ffn_norm(x),
            padding_mask_T=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)
        return x, ffn_pre, cmp_k, idx_k, topk_indices, topk_scores, candidates


class DeepSeekV41Model(DeepSeekV4Model):
    """DeepSeek-V4.1 text backbone: CSA2 policy over the packed varlen contract.

    Inherits upstream V4's model for the shared skeleton (optimizer hooks,
    the FSDP application, TP/CP validation, document metadata construction
    via ``VarlenAttentionMetadata`` and the decoder config fields). V4.1 has no
    HcHead merge and no MTP stack, so ``Config.hc_head`` is optional and
    ``__init__``/``forward`` are overridden for the single-pass mHC chain
    and the cross-layer sparse-indexer state on the packed ``[T, *]``
    stream.
    """

    state_dict_adapter_cls = DeepSeekV41StateDictAdapter
    supports_pipeline_parallel = False

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        # V4 compiles fused activation regions that pull in inductor's
        # flex_attention kernels, which the NPU torch build does not ship;
        # V4.1 keeps the decoder default (no local compile regions).
        local_compile_regions: list[str] = field(default_factory=list)
        hc_head: HcHead.Config | None = None  # pyrefly: ignore [bad-override]
        kv_source_layers: tuple[int, ...] = ()
        index_source_layers: tuple[int, ...] = ()
        candidate_source_layer: int = 20
        candidate_topk_blocks: int = 2048
        candidate_block_size: int = 8

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """Estimate V4.1 model FLOPs per token using TorchTitan's MFU convention."""
            from typing import cast

            from torchtitan.models.common.flops import (
                get_nparams_and_active_nparams,
                quadratic_attention_flops_per_token,
            )

            deepseek_v4_1_model = cast("DeepSeekV41Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                deepseek_v4_1_model
            )

            num_flops_per_token = 6 * active_nparams
            for layer in self.layers:
                attention = layer.attention
                num_flops_per_token += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=attention.head_dim,
                    v_head_dim=attention.head_dim,
                    seq_len=seq_len,
                    sliding_window_size=attention.inner_attention.window_size,
                )
                if attention.compress_ratio > 0:
                    compressed_seq_len = seq_len // attention.compress_ratio
                    num_flops_per_token += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.head_dim,
                        v_head_dim=attention.head_dim,
                        seq_len=min(
                            attention.indexer.index_topk, compressed_seq_len
                        ),
                        sliding_window_size=0,
                    )
            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        # Reuse upstream V4's TP/CP validation, then init as a plain decoder:
        # V4's __init__ would also build the HcHead merge and the MTP stack,
        # neither of which V4.1 has (hc_head is None, mtp_layers is empty).
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        tp = spmd_mesh_size("tp")
        if tp > 1:
            for layer in config.layers[: config.n_layers]:
                num_heads = layer.attention.n_heads
                if num_heads % tp != 0:
                    raise ValueError(
                        f"n_heads ({num_heads}) must be divisible by tp ({tp})"
                    )
                num_groups = layer.attention.n_groups
                if num_groups % tp != 0:
                    raise ValueError(
                        f"n_groups ({num_groups}) must be divisible by tp ({tp})"
                    )
        if spmd_mesh_size("cp") > 1:
            raise NotImplementedError(
                "Context Parallel is not yet supported for DeepSeek V4.1 sparse attention."
            )
        Decoder.__init__(self, config)
        cfg = config
        self.hc_mult = cfg.hc_mult
        self.n_mtp_layers = 0
        self.n_main_layers = cfg.n_layers
        self.compress_ratios = tuple(cfg.compress_ratios)
        self.mtp_layers = ModuleList()

    def _embedded_inputs(
        self,
        tokens: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the input embedding stream of one forward."""
        tok_embeddings = self.tok_embeddings
        if input_embeds is not None:
            return input_embeds
        if tok_embeddings is None:
            return tokens
        return tok_embeddings(tokens)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
    ):
        """Run the packed V4.1 stack with single-pass mHC and CSA2 state."""
        hidden = self._embedded_inputs(tokens, input_embeds=input_embeds)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)

        pre_mix = HcPre.identity_pre_mix(hidden, self.hc_mult)
        cmp_k: torch.Tensor | None = None
        idx_k: torch.Tensor | None = None
        topk_indices: torch.Tensor | None = None
        topk_scores: torch.Tensor | None = None
        candidates: torch.Tensor | None = None
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            (
                hidden,
                pre_mix,
                cmp_k,
                idx_k,
                topk_indices,
                topk_scores,
                candidates,
            ) = layer(
                hidden,
                layer_attention_metadata,
                positions,
                pre_mix=pre_mix,
                cmp_k=cmp_k,
                idx_k=idx_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
                candidates=candidates,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )

        main_hidden = HcPre.collapse(hidden, pre_mix)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden
        if self._skip_lm_head or self.lm_head is None:
            return main_hidden
        return self.lm_head(main_hidden)
