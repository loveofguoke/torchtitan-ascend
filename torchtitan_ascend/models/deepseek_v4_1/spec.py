# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
#
# Modification points:
# 1. Add immutable V4.1 model specifications and static layer policies;
# 2. Declare CSA2 compression and source/reuse topology;
# 3. Add static topology validation.
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

"""Immutable DeepSeek V4.1 model specifications and static layer policies.

``V41Spec`` centralizes model dimensions, numerical parameters, and the
compression topology. Flavors declare their differences with
``dataclasses.replace``. ``V41LayerPolicy`` describes static per-layer rules,
including compression ratios, KV/index ownership, FULL/REINDEX/REUSE modes,
and candidate-pool usage. ``layer_policies`` derives and validates these rules;
runtime checks for missing source state remain in the forward path.
"""

from __future__ import annotations

from dataclasses import dataclass

CANDIDATE_BLOCK_SIZE = 8

V41_CANDIDATE_SOURCE_LAYER = 20
V41_FULL_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
V41_KV_SOURCE_LAYERS = (2, 8, 14, 20)
V41_FULL_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)

V41_STRIPPED_COMPRESS_RATIOS = (2, 1, 1)
V41_STRIPPED_KV_SOURCE_LAYERS = (0, 1)
V41_STRIPPED_INDEX_SOURCE_LAYERS = (0, 1, 2)
V41_STRIPPED_CANDIDATE_SOURCE_LAYER = 1


@dataclass(frozen=True, kw_only=True, slots=True)
class V41Spec:
    """Model dimensions, numerical parameters, and compression topology."""

    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    n_groups: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    moe_inter_dim: int
    num_experts: int
    num_shared_experts: int
    top_k: int
    vocab_size: int
    window_size: int
    norm_eps: float
    hc_mult: int
    route_scale: float
    route_norm: bool
    load_balance_coeff: float
    sinkhorn_iters: int
    hc_eps: float
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    original_seq_len: int
    max_context_length: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int

    def validate_topology(self) -> None:
        """Validate static consistency of the compression topology."""
        if len(self.compress_ratios) != self.n_layers:
            raise ValueError(
                f"compress_ratios must match n_layers ({self.n_layers}), "
                f"got {len(self.compress_ratios)}"
            )
        for name, sources in (
            ("kv_source_layers", self.kv_source_layers),
            ("index_source_layers", self.index_source_layers),
        ):
            outside = sorted({s for s in sources if s < 0 or s >= self.n_layers})
            if outside:
                raise ValueError(f"{name} outside the {self.n_layers}-layer crop: {outside}")
        if any(ratio < 0 for ratio in self.compress_ratios):
            raise ValueError("compress_ratios must be non-negative")
        if not set(self.kv_source_layers).issubset(self.index_source_layers):
            raise ValueError("every KV source must also be an index source")
        for layer_id, ratio in enumerate(self.compress_ratios):
            kv_source = max((s for s in self.kv_source_layers if s <= layer_id), default=None)
            if ratio > 0 and kv_source is None:
                raise ValueError(
                    f"layer {layer_id} has compress_ratio={ratio} but no KV source precedes it"
                )
            if kv_source is not None and self.compress_ratios[kv_source] != ratio:
                raise ValueError(
                    f"layer {layer_id} (compress_ratio={ratio}) consumes the compressed KV of "
                    f"layer {kv_source} (compress_ratio={self.compress_ratios[kv_source]}); "
                    "the ratios must match"
                )
            if layer_id in self.index_source_layers and layer_id not in self.kv_source_layers:
                index_source = max(
                    (s for s in self.index_source_layers if s < layer_id), default=None
                )
                if index_source is None:
                    raise ValueError(f"re-indexing layer {layer_id} has no preceding index source")
                if self.compress_ratios[index_source] != ratio:
                    raise ValueError(
                        f"layer {layer_id} (compress_ratio={ratio}) scores the shared index keys "
                        f"of layer {index_source} (compress_ratio={self.compress_ratios[index_source]}); "
                        "the ratios must match"
                    )
        if self.candidate_source_layer not in set(self.index_source_layers):
            raise ValueError(
                "the candidate-pool source must also be an index source: "
                f"layer {self.candidate_source_layer} is not in index_source_layers"
            )
        if self.candidate_topk_blocks <= 0:
            raise ValueError(
                f"the candidate pool capacity must be positive, got {self.candidate_topk_blocks}"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class V41LayerPolicy:
    """Static parallelism and sparse-attention rules for one layer."""

    layer_id: int
    compress_ratio: int
    owns_compressor: bool
    owns_indexer: bool
    source_key: bool
    external_key: bool
    is_candidate_source: bool
    uses_candidates: bool


def layer_policies(spec: V41Spec) -> tuple[V41LayerPolicy, ...]:
    """Derive and validate static rules for every layer in the specification."""
    spec.validate_topology()
    source_key_layers = tuple(
        layer_id
        for layer_id in spec.index_source_layers
        if layer_id in spec.kv_source_layers
    )
    external_key_layers = tuple(
        layer_id
        for layer_id in spec.index_source_layers
        if layer_id not in spec.kv_source_layers
    )
    return tuple(
        V41LayerPolicy(
            layer_id=layer_id,
            compress_ratio=spec.compress_ratios[layer_id],
            owns_compressor=layer_id in spec.kv_source_layers,
            owns_indexer=layer_id in spec.index_source_layers,
            source_key=layer_id in source_key_layers,
            external_key=layer_id in external_key_layers,
            is_candidate_source=layer_id == spec.candidate_source_layer,
            uses_candidates=(
                layer_id in spec.index_source_layers
                and layer_id != spec.candidate_source_layer
                and 0 <= spec.candidate_source_layer < layer_id
            ),
        )
        for layer_id in range(spec.n_layers)
    )


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "V41LayerPolicy",
    "V41Spec",
    "V41_CANDIDATE_SOURCE_LAYER",
    "V41_FULL_COMPRESS_RATIOS",
    "V41_FULL_INDEX_SOURCE_LAYERS",
    "V41_KV_SOURCE_LAYERS",
    "V41_STRIPPED_COMPRESS_RATIOS",
    "V41_STRIPPED_CANDIDATE_SOURCE_LAYER",
    "V41_STRIPPED_INDEX_SOURCE_LAYERS",
    "V41_STRIPPED_KV_SOURCE_LAYERS",
    "layer_policies",
]
