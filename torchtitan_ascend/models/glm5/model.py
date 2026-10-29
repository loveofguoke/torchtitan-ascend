# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5 structure using upstream MLA, decoder, MoE, and MTP components.

NPU DSA execution is supplied by the explicitly selected config override.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch_remat as remat
from torch import nn

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common import ComplexRoPE
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.deepseek_v3.model import Attention as MLAAttention, DeepSeekV3TransformerBlock
from torchtitan.models.deepseek_v3.mtp import MTPDecoder, roll_mtp_sequence
from torchtitan.models.utils import get_nparams_and_active_nparams

from .dsa import DsaMetadata, Glm5DsaIndexer
from .state_dict_adapter import Glm5StateDictAdapter

# Tensor dimensions used in this file:
# T/Q/K: token, query-token, and key-token dimensions.
# D/N/H/R/P/V: model, head-count, head, RoPE, pass-through, and value dimensions.


class Glm5Attention(MLAAttention):
    """Upstream MLA parameter structure with a GLM token indexer.

    Ascend execution is selected through the GLM DSA config override.
    """

    # The inherited MLA contract, also visible without importing upstream types.
    n_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    softmax_scale: float
    training: bool
    wq_a: torch.nn.Module
    wq_b: torch.nn.Module
    wkv_a: torch.nn.Module
    wkv_b: torch.nn.Module
    wo: torch.nn.Module
    q_norm: torch.nn.Module
    kv_norm: torch.nn.Module
    rope: torch.nn.Module
    inner_attention: torch.nn.Module

    @dataclass(kw_only=True, slots=True)
    class Config(MLAAttention.Config):
        rope: ComplexRoPE.Config
        indexer: Glm5DsaIndexer.Config | None
        index_topk: int = 2048

        @property
        def qk_head_dim(self) -> int:
            return self.qk_nope_head_dim + self.qk_rope_head_dim

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build() if config.indexer is not None else None

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("GLM DSA execution requires the npu_glm5_dsa override from its NPU recipe.")


class Glm5TransformerBlock(DeepSeekV3TransformerBlock):
    """GLM-5 decoder block with either a dense FFN or common MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3TransformerBlock.Config):
        preserve_output_for_mtp: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.preserve_output_for_mtp = config.preserve_output_for_mtp

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: DsaMetadata,
        positions: torch.Tensor | None = None,
        topk_indices_TS: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_indices:
            attention_TD, topk_indices_TS = self.attention(
                self.attention_norm(x_TD),
                attention_masks,
                positions,
                topk_indices_TS=topk_indices_TS,
                return_indices=True,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            attention_TD = self.attention(
                self.attention_norm(x_TD),
                attention_masks,
                positions,
                topk_indices_TS=topk_indices_TS,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attention_TD)
        x_TD = x_TD + attention_TD
        normalized_TD = self.ffn_norm(x_TD)
        if self.moe_enabled:
            ffn_output_TD = self.moe(
                normalized_TD,
                padding_mask_T=padding_mask,
            )
        else:
            ffn_output_TD = self.feed_forward(normalized_TD)
        if self.preserve_output_for_mtp:
            # The saved residual add is a bare consumer of the MoE/FFN result.
            # MTP's auxiliary backward branch replays this boundary, so declare
            # the input dependency before entering the non-recomputed region.
            remat.recompute_needs_tensor(ffn_output_TD)
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        output_TD: torch.Tensor = remat.region(torch.add, self.remat_region_name("ffn_residual"), recompute=False)(
            x_TD, ffn_output_TD
        )
        if self.preserve_output_for_mtp:
            # MTP adds an extra backward branch through the main decoder. Keep
            # each saved residual boundary materialized for MTP flavors so
            # selective-AC replay never exposes a metadata placeholder to a
            # bare consumer. Non-MTP flavors leave this disabled.
            remat.recompute_needs_tensor(output_TD)
            output_TD = output_TD.clone()
        if return_indices:
            assert topk_indices_TS is not None
            return output_TD, topk_indices_TS
        return output_TD


class Glm5Model(MTPDecoder):
    """GLM-5 decoder with token-indexed DSA attention.

    The inherited decoder accepts token ids on the first PP stage and hidden
    states on later stages. A PP stage may own zero decoder layers, which is why
    mask construction and forward preserve embedding-only/output-only stages.
    """

    state_dict_adapter_cls = Glm5StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(MTPDecoder.Config):
        dim: int = 6144
        vocab_size: int = 154880
        index_sources: tuple[int, ...] = ()
        indexer_training_stage: str = "frozen"

        def set_sharding_(self, parallelism) -> None:
            from .sharding import set_glm5_sharding_config

            set_glm5_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            attention = self.layers[0].attention
            assert isinstance(attention, Glm5Attention.Config)
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            base_flops = 6 * active_nparams
            # Match the upstream parameter-based estimate, with forward-only
            # projection cost for explicitly frozen indexers.
            nparams_indexer = sum(
                parameter.numel() for name, parameter in model.named_parameters() if ".attention.indexer." in name
            )
            if self.indexer_training_stage == "frozen":
                base_flops -= 4 * nparams_indexer
            # Per token, the indexer computes N dot products of width H against
            # the full sequence, followed by a weighted reduction over N heads.
            num_dsa_layers = len(self.layers) + len(self.mtp_layers or ())
            assert attention.indexer is not None
            dsa_flops = 2 * num_dsa_layers * attention.indexer.n_heads * (attention.indexer.head_dim + 1) * seq_len
            return nparams, base_flops + dsa_flops

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        if all(part.config.indexer_training_stage == "warmup" for part in model_parts):
            register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
        else:
            DeepSeekV3Model._register_optimizer_hooks(optimizers, model_parts, parallelism_context)

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> DsaMetadata | None:
        # Pipeline partitioning may create an embedding-only or output-only
        # stage. Such a stage has no attention operation and needs no mask.
        if len(self.layers) == 0:
            return None

        if positions.ndim != 1:
            raise ValueError("GLM-5 positions must have shape [T].")
        return DsaMetadata.from_positions(positions, padding_mask)

    def __init__(self, config):
        super().__init__(config)
        self.index_sources = config.index_sources
        if config.indexer_training_stage == "warmup":
            self.requires_grad_(False)
            for module in self.modules():
                if isinstance(module, Glm5DsaIndexer):
                    module.requires_grad_(True)
        if self.index_sources:
            if len(self.index_sources) != len(config.layers):
                raise ValueError("index_sources must contain one source per layer")
            for layer, source in enumerate(self.index_sources):
                if not 0 <= source <= layer or self.index_sources[source] != source:
                    raise ValueError("index source must be a current/earlier producer layer")

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context,
        parallelism,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ):
        """Prepare document metadata and CP-shard the tensor inputs."""
        del max_num_documents, max_context_length, kwargs
        positions = input_dict.get("positions")
        if positions is not None:
            input_dict["attention_masks"] = self.get_attention_masks(
                positions,
                input_dict.get("padding_mask"),
            )

        input_shardings = decoder_input_sharding()
        depths = range(1, len(self.mtp_layers) + 1) if self.mtp_layers is not None else range(0)
        if self.mtp_layers is not None and positions is None:
            raise ValueError("GLM-5 MTP input preprocessing requires positions.")
        for depth in depths:
            assert positions is not None
            mtp_tokens, mtp_valid = roll_mtp_sequence(
                input_dict["input"],
                shift=depth,
                positions=positions,
                padding_mask=input_dict.get("padding_mask"),
                fill_value=0,
                return_valid_mask=True,
            )
            mtp_labels = roll_mtp_sequence(
                input_dict["labels"],
                shift=depth,
                positions=positions,
                padding_mask=input_dict.get("padding_mask"),
                fill_value=IGNORE_INDEX,
                return_valid_mask=False,
            )
            mtp_padding = ~mtp_valid
            if input_dict.get("padding_mask") is not None:
                mtp_padding = mtp_padding | input_dict["padding_mask"]
            input_dict[f"mtp_tokens_{depth}"] = mtp_tokens
            input_dict[f"mtp_labels_{depth}"] = mtp_labels
            input_dict[f"mtp_valid_{depth}"] = mtp_valid
            input_dict[f"mtp_attention_masks_{depth}"] = self.get_attention_masks(positions, mtp_padding)
            input_shardings[f"mtp_tokens_{depth}"] = input_shardings["input"]
            input_shardings[f"mtp_labels_{depth}"] = input_shardings["labels"]
            input_shardings[f"mtp_valid_{depth}"] = input_shardings["input"]
        if parallelism_context.cp_enabled:
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
            main_metadata = input_dict["attention_masks"]
            for depth in depths:
                key = f"mtp_attention_masks_{depth}"
                input_dict[key] = replace(
                    input_dict[key],
                    order=main_metadata.order,
                    query_start=main_metadata.query_start,
                    query_end=main_metadata.query_end,
                )
        input_dict = annotate_input_spmd_types(parallelism_context, input_dict, input_shardings)

        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        if self.mtp_layers is not None:
            inputs = (inputs, *(input_dict.pop(f"mtp_tokens_{d}") for d in depths))
            labels = (labels, *(input_dict.pop(f"mtp_labels_{d}") for d in depths))
            input_dict["mtp_input_valid_masks"] = tuple(input_dict.pop(f"mtp_valid_{d}") for d in depths)
            input_dict["mtp_attention_masks"] = tuple(input_dict.pop(f"mtp_attention_masks_{d}") for d in depths)
        return inputs, labels, input_dict

    def forward(  # pyrefly: ignore[bad-param-name-override, bad-override]
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        pipeline_indices_TS: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: DsaMetadata | None = None,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        mtp_attention_masks: tuple[DsaMetadata, ...] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # Preserve Decoder's positional ``model(tokens, positions)`` API. PP
        # metadata is [T, S], while GLM positions are required to be [T].
        if pipeline_indices_TS is not None and pipeline_indices_TS.ndim == 1:
            if positions is not None:
                raise ValueError("positions were provided twice")
            positions = pipeline_indices_TS
            pipeline_indices_TS = None
        main_tokens = tokens[0] if isinstance(tokens, tuple) else tokens
        if positions is None:
            positions = torch.arange(main_tokens.shape[0], device=main_tokens.device)
        if attention_masks is None:
            attention_masks = self.get_attention_masks(positions, padding_mask)
        hidden_TD = self.tok_embeddings(main_tokens) if self.tok_embeddings is not None else main_tokens
        index_sources = self.index_sources or tuple(range(len(self.config.layers)))
        local_layers = {int(name) for name in self.layers}
        external_sources = {index_sources[layer] for layer in local_layers if index_sources[layer] not in local_layers}
        indices_by_source: dict[int, torch.Tensor] = {}
        if external_sources:
            if pipeline_indices_TS is None:
                raise ValueError("this GLM-5 PP stage requires DSA indices")
            source = next(iter(external_sources))
            indices_by_source[source] = pipeline_indices_TS
        for name, layer in self.layers.items():
            layer_id = int(name)
            source = index_sources[layer_id]
            hidden_TD, indices_TS = layer(
                hidden_TD,
                attention_masks,
                positions,
                indices_by_source.get(source),
                return_indices=True,
                padding_mask=padding_mask,
            )
            if source == layer_id:
                indices_by_source[source] = indices_TS
        pre_norm_hidden_TD = hidden_TD
        hidden_TD = self.norm(hidden_TD) if self.norm is not None else hidden_TD
        if self.mtp_layers is not None:
            if not isinstance(tokens, tuple) or len(tokens) != len(self.mtp_layers) + 1:
                raise ValueError("GLM-5 MTP requires one shifted token tensor per depth.")
            if mtp_input_valid_masks is None or mtp_attention_masks is None:
                raise ValueError("GLM-5 MTP requires validity masks and DSA metadata.")
            if self.tok_embeddings is None or self.lm_head is None:
                raise ValueError("GLM-5 MTP is not supported on a pipeline partition.")
            mtp_outputs: list[torch.Tensor] = []
            prev_hidden = pre_norm_hidden_TD
            depth_inputs = zip(
                self.mtp_layers,
                tokens[1:],
                mtp_input_valid_masks,
                mtp_attention_masks,
                strict=True,
            )
            for layer, depth_tokens, valid_mask, depth_attention_mask in depth_inputs:
                prev_hidden = layer(
                    self.tok_embeddings(depth_tokens),
                    prev_hidden,
                    valid_mask,
                    depth_attention_mask,
                    positions,
                    padding_mask=padding_mask,
                )
                mtp_outputs.append(prev_hidden)
            outputs = (hidden_TD, *mtp_outputs)
            if self._skip_lm_head:
                return outputs
            return tuple(self.lm_head(item) for item in outputs)
        if self._skip_lm_head or self.lm_head is None:
            if local_layers:
                next_layer = max(local_layers) + 1
                if next_layer < len(index_sources):
                    source = index_sources[next_layer]
                    if source != next_layer:
                        return hidden_TD, indices_by_source[source]
            return hidden_TD
        return self.lm_head(hidden_TD)
