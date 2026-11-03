# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Add V4.1 flavor and model Config builders;
# 2. Reuse the V4 MoE factory and upstream components;
# 3. Apply sharding during model Config construction.
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

"""DeepSeek V4.1 model flavors and config builders.

Builders share a ``(spec, policy, rope)`` interface. ``V41Spec`` defines
immutable model dimensions, numerical parameters, and compression topology;
``V41LayerPolicy`` defines static KV/index ownership and candidate-pool rules
for each layer. Flavors use ``dataclasses.replace`` to declare differences from
the Flash specification, and the public entry point resolves ``seq_len``
before following the same construction path. Sharding is applied during model
Config construction according to the parallelism configuration.
"""

from __future__ import annotations

import dataclasses
from functools import partial

from typing import cast, Protocol, TYPE_CHECKING

import torch.nn as nn
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    Linear,
    RMSNorm,
    RowParallelLinear,
)
from torchtitan.models.deepseek_v4.attention import SlidingWindowAttention
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.deepseek_v4.flavors import (
    _make_v4_moe_config,
    _output_linear_init,
)

from .attention import Attention, CompressedSparseAttention2
from .compressor import Compressor
from .indexer import FULL, HierarchicalIndexer, REINDEX, REUSE, Selector
from .mhc import HcPost, HcPre
from .model import DeepSeekV41Model, DeepSeekV41TransformerBlock
from .spec import (
    CANDIDATE_BLOCK_SIZE,
    layer_policies,
    V41_CANDIDATE_SOURCE_LAYER,
    V41_FULL_COMPRESS_RATIOS,
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
    V41_STRIPPED_CANDIDATE_SOURCE_LAYER,
    V41_STRIPPED_COMPRESS_RATIOS,
    V41_STRIPPED_INDEX_SOURCE_LAYERS,
    V41_STRIPPED_KV_SOURCE_LAYERS,
    V41LayerPolicy,
    V41Spec,
)

if TYPE_CHECKING:

    class _ContextSpecFactory(Protocol):
        def __call__(self, *, max_context_length: int) -> V41Spec:
            ...


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_HC_PARAM_INIT = {
    "hc_fn": partial(nn.init.trunc_normal_, std=0.02),
    "hc_base": partial(nn.init.trunc_normal_, std=0.02),
    "hc_scale": partial(nn.init.trunc_normal_, std=0.02),
}
_INDEXER_LOSS_COEFF = 0.01


def _make_compressor_config(
    spec: V41Spec, policy: V41LayerPolicy, rope: ComplexRoPE.Config | None
) -> Compressor.Config:
    return Compressor.Config(
        compress_ratio=policy.compress_ratio,
        is_source=policy.owns_compressor,
        head_dim=spec.head_dim,
        rope_head_dim=spec.rope_head_dim,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wkv=(
            Linear.Config(
                in_features=spec.dim,
                out_features=spec.head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
                inplace_wgrad_accum=False,
            )
            if policy.owns_compressor
            else None
        ),
        wgate=(
            Linear.Config(
                in_features=spec.dim,
                out_features=spec.head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
                inplace_wgrad_accum=False,
            )
            if policy.owns_compressor and policy.compress_ratio > 1
            else None
        ),
        norm=(
            RMSNorm.Config(
                normalized_shape=spec.head_dim,
                eps=spec.norm_eps,
                param_init=_NORM_INIT,
            )
            if policy.owns_compressor
            else None
        ),
    )


def _make_indexer_config(
    spec: V41Spec, policy: V41LayerPolicy, rope: ComplexRoPE.Config | None
) -> HierarchicalIndexer.Config:
    owns_index_k = policy.source_key and policy.owns_indexer
    if not policy.owns_indexer:
        mode = REUSE
    elif owns_index_k:
        mode = FULL
    else:
        mode = REINDEX
    wants_pool = policy.is_candidate_source or policy.uses_candidates
    pool_capacity = spec.candidate_topk_blocks if wants_pool else -1
    pool_block_size = CANDIDATE_BLOCK_SIZE if wants_pool else -1
    return HierarchicalIndexer.Config(
        mode=mode,
        selector=Selector.Config(
            mode=mode,
            compress_ratio=policy.compress_ratio,
            num_index_heads=spec.index_n_heads,
            index_head_dim=spec.index_head_dim,
            index_topk=spec.index_topk,
            candidate_topk_blocks=pool_capacity,
            candidate_block_size=pool_block_size,
        ),
        num_index_heads=spec.index_n_heads,
        index_head_dim=spec.index_head_dim,
        rope_head_dim=spec.rope_head_dim,
        index_topk=spec.index_topk,
        compress_ratio=policy.compress_ratio,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wq_b=(
            Linear.Config(
                in_features=spec.q_lora_rank,
                out_features=spec.index_n_heads * spec.index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if policy.owns_indexer
            else None
        ),
        weights_proj=(
            Linear.Config(
                in_features=spec.dim,
                out_features=spec.index_n_heads,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if policy.owns_indexer
            else None
        ),
        wk=(
            Linear.Config(
                in_features=spec.head_dim,
                out_features=spec.index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if owns_index_k
            else None
        ),
        k_norm=(
            RMSNorm.Config(
                normalized_shape=spec.index_head_dim,
                eps=spec.norm_eps,
                param_init=_NORM_INIT,
            )
            if owns_index_k
            else None
        ),
    )


def _make_attention_config(
    spec: V41Spec, policy: V41LayerPolicy, rope: ComplexRoPE.Config
) -> Attention.Config:
    if policy.source_key and policy.external_key:
        raise ValueError(
            "an indexer cannot own its source-key projection and consume an external key at the same time"
        )
    if policy.is_candidate_source and not policy.owns_compressor:
        raise ValueError(
            "the candidate-pool source must own the compressed KV it scores"
        )

    hd = spec.head_dim
    # The attention, compressor and indexer all split the rope slice off by
    # hand (V4 style), so every rope config here rotates a bare
    # rope_head_dim input (split=0, the default); replace() only copies the
    # config so each module gets its own cache.
    per_group_in = (spec.n_heads * hd) // spec.n_groups
    per_group_out = spec.n_groups * spec.o_lora_rank
    softmax_scale = hd**-0.5
    compressor_cfg = _make_compressor_config(
        spec,
        policy,
        dataclasses.replace(rope) if policy.owns_compressor else None,
    )
    indexer_cfg = _make_indexer_config(
        spec,
        policy,
        dataclasses.replace(rope) if policy.owns_indexer else None,
    )
    if policy.compress_ratio > 0:
        inner_attention_cfg = CompressedSparseAttention2.Config(
            window_size=spec.window_size,
            softmax_scale=softmax_scale,
            compress_ratio=policy.compress_ratio,
            index_topk=spec.index_topk,
            aux_loss=(
                SparseIndexerLoss.Config(
                    coeff=_INDEXER_LOSS_COEFF,
                    reduce_mesh="loss",
                    softmax_scale=softmax_scale,
                    num_heads=spec.n_heads,
                    compress_ratio=policy.compress_ratio,
                )
                if _INDEXER_LOSS_COEFF > 0
                else None
            ),
        )
    else:
        inner_attention_cfg = SlidingWindowAttention.Config(
            window_size=spec.window_size,
            softmax_scale=softmax_scale,
            compress_ratio=policy.compress_ratio,
            index_topk=spec.index_topk,
            aux_loss=None,
        )
    return Attention.Config(
        dim=spec.dim,
        n_heads=spec.n_heads,
        head_dim=spec.head_dim,
        rope_head_dim=spec.rope_head_dim,
        q_lora_rank=spec.q_lora_rank,
        o_lora_rank=spec.o_lora_rank,
        n_groups=spec.n_groups,
        compress_ratio=policy.compress_ratio,
        norm_eps=spec.norm_eps,
        index_n_heads=spec.index_n_heads,
        index_head_dim=spec.index_head_dim,
        n_layers=spec.n_layers,
        layer_id=policy.layer_id,
        inner_attention=inner_attention_cfg,
        rope=rope,
        wq_a=Linear.Config(
            in_features=spec.dim,
            out_features=spec.q_lora_rank,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=spec.q_lora_rank,
            eps=spec.norm_eps,
            param_init=_NORM_INIT,
        ),
        wq_b=Linear.Config(
            in_features=spec.q_lora_rank,
            out_features=spec.n_heads * hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=spec.dim,
            out_features=hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=hd,
            eps=spec.norm_eps,
            param_init=_NORM_INIT,
        ),
        wo_a=Linear.Config(
            in_features=per_group_in,
            out_features=per_group_out,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wo_b=RowParallelLinear.Config(
            in_features=per_group_out,
            out_features=spec.dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        # attn_sink uses a Linear wrapper to hold a (n_heads, 1) weight; the
        # forward path squeezes it back to (n_heads,) to match the original
        # parameter semantics.
        attn_sink=Linear.Config(
            in_features=1,
            out_features=spec.n_heads,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        compressor=compressor_cfg,
        indexer=indexer_cfg,
    )


def _make_moe_config(spec: V41Spec, layer_id: int, parallelism: ParallelismConfig):
    return _make_v4_moe_config(
        layer_id=layer_id,
        dim=spec.dim,
        moe_inter_dim=spec.moe_inter_dim,
        num_experts=spec.num_experts,
        num_shared_experts=spec.num_shared_experts,
        top_k=spec.top_k,
        vocab_size=spec.vocab_size,
        n_hash_layers=0,
        route_norm=spec.route_norm,
        route_scale=spec.route_scale,
        load_balance_coeff=spec.load_balance_coeff,
        enable_sp=parallelism.enable_sequence_parallel,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )


def _make_v41_config(
    spec: V41Spec, parallelism: ParallelismConfig
) -> DeepSeekV41Model.Config:
    policies = layer_policies(spec)

    rope = ComplexRoPE.Config(
        dim=spec.rope_head_dim,
        max_context_length=spec.max_context_length,
        theta=spec.rope_theta,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=spec.rope_head_dim,
        max_context_length=spec.max_context_length,
        theta=spec.compress_rope_theta,
        scaling="yarn",
        rope_factor=spec.rope_factor,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=spec.original_seq_len,
    )

    layers = []
    for policy in policies:
        attn_cfg = _make_attention_config(
            spec,
            policy,
            rope_compress if policy.compress_ratio >= 1 else rope,
        )
        moe_cfg = _make_moe_config(spec, policy.layer_id, parallelism)
        layers.append(
            DeepSeekV41TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=spec.dim,
                    eps=spec.norm_eps,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=spec.dim,
                    eps=spec.norm_eps,
                    param_init=_NORM_INIT,
                ),
                feed_forward=moe_cfg,
                hc_attn_pre=HcPre.Config(
                    hc_mult=spec.hc_mult,
                    dim=spec.dim,
                    sinkhorn_iters=spec.sinkhorn_iters,
                    eps=spec.hc_eps,
                    norm_eps=spec.norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_ffn_pre=HcPre.Config(
                    hc_mult=spec.hc_mult,
                    dim=spec.dim,
                    sinkhorn_iters=spec.sinkhorn_iters,
                    eps=spec.hc_eps,
                    norm_eps=spec.norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_post=HcPost.Config(),
            )
        )

    return DeepSeekV41Model.Config(
        dim=spec.dim,
        vocab_size=spec.vocab_size,
        norm_eps=spec.norm_eps,
        max_context_length=spec.max_context_length,
        tok_embeddings=Embedding.Config(
            num_embeddings=spec.vocab_size,
            embedding_dim=spec.dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(
            normalized_shape=spec.dim, eps=spec.norm_eps, param_init=_NORM_INIT
        ),
        lm_head=Linear.Config(
            in_features=spec.dim,
            out_features=spec.vocab_size,
            param_init=_output_linear_init(spec.dim),
        ),
        layers=layers,
        hc_mult=spec.hc_mult,
        compress_ratios=spec.compress_ratios,
        n_layers=spec.n_layers,
        kv_source_layers=spec.kv_source_layers,
        index_source_layers=spec.index_source_layers,
        candidate_source_layer=spec.candidate_source_layer,
        candidate_topk_blocks=spec.candidate_topk_blocks,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
    )


def _flash_spec(num_experts: int = 384, max_context_length: int = 4096) -> V41Spec:
    return V41Spec(
        dim=5120,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        n_groups=8,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        moe_inter_dim=2304,
        candidate_topk_blocks=2048,
        n_layers=40,
        vocab_size=129280,
        window_size=128,
        norm_eps=1e-20,
        hc_mult=4,
        num_shared_experts=1,
        top_k=6,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=0.001,
        sinkhorn_iters=20,
        hc_eps=1e-06,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=65536,
        max_context_length=max_context_length,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        num_experts=num_experts,
    )


def _debug_spec(num_experts: int = 16) -> V41Spec:
    # The debug model uses a fixed context length of 512, matching the flavor's
    # maximum; build_model_config enforces the same upper bound for seq_len.
    return dataclasses.replace(
        _flash_spec(num_experts=num_experts, max_context_length=512),
        dim=512,
        n_heads=8,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=128,
        o_lora_rank=128,
        n_groups=4,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=32,
        moe_inter_dim=256,
        candidate_topk_blocks=4,
    )


def _stripped_spec(num_experts: int = 4, max_context_length: int = 4096) -> V41Spec:
    return dataclasses.replace(
        _flash_spec(num_experts=num_experts, max_context_length=max_context_length),
        n_layers=len(V41_STRIPPED_COMPRESS_RATIOS),
        top_k=2,
        compress_ratios=V41_STRIPPED_COMPRESS_RATIOS,
        kv_source_layers=V41_STRIPPED_KV_SOURCE_LAYERS,
        index_source_layers=V41_STRIPPED_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_STRIPPED_CANDIDATE_SOURCE_LAYER,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debug_spec, 512),
    "deepseek_v4_1_flash": (_flash_spec, 4096),
    "deepseek_v4_1_stripped": (_stripped_spec, 4096),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    converters: list | None = None,
) -> DeepSeekV41Model.Config:
    if flavor not in MODEL_FLAVORS:
        raise ValueError(
            f"Unknown deepseek_v4_1 flavor: {flavor}. "
            f"Available: {list(MODEL_FLAVORS.keys())}"
        )
    get_spec, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    spec = (
        cast("_ContextSpecFactory", get_spec)(max_context_length=context_len)
        if flavor != "debugmodel"
        else get_spec()
    )
    config = _make_v41_config(spec, parallelism)
    from .sharding import set_deepseek_v4_1_sharding_config

    set_deepseek_v4_1_sharding_config(
        config,
        enable_sp=parallelism.enable_sequence_parallel,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    if converters is not None:
        from torchtitan.config.transform import validate_converter_compatibility

        validate_converter_compatibility(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    return config


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "V41_CANDIDATE_SOURCE_LAYER",
    "V41_FULL_COMPRESS_RATIOS",
    "V41_FULL_INDEX_SOURCE_LAYERS",
    "V41_KV_SOURCE_LAYERS",
    "V41LayerPolicy",
    "V41Spec",
    "Attention",
    "DeepSeekV41Model",
    "MODEL_FLAVORS",
    "build_model_config",
    "layer_policies",
]
