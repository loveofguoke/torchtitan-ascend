# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Add CSA2, mHC and Selector sharding contracts;
# 2. Apply sharding during model Config construction;
# 3. Reuse V4 helpers and the upstream SWA sharding contract.
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

"""Declarative sharding configs for DeepSeek V4.1 components."""

import spmd_types as spmd

from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    dense_activation_placement,
    dense_param_placement,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
    token_id_placement,
)
from torchtitan.models.deepseek_v4.sharding import (
    _hc_branch_placement,
    _hc_mix_placement,
    attention_sharding_config,
    attn_sink_sharding_config,
    dsa_inner_attention_sharding_config,
    hc_post_sharding_config,
    indexer_sharding_config,
    wo_a_sharding_config,
)

_replicated_weight = replicated_state_sharding_config("weight")
_rope_cache_sharding = rope_cache_sharding_config()


def csa2_inner_attention_sharding_config(*, compress_ratio: int) -> ShardingConfig:
    query_states = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    input_shardings = {
        "q_THD": query_states,
        "swa_k_TD": replicated_activation,
        "attn_sink": dense_param_placement(tp=spmd.S(0)),
    }
    if compress_ratio > 0:
        input_shardings.update(
            {
                "cmp_k_SD": replicated_activation,
                "topk_indices_TK": replicated_activation,
                "topk_scores_TK": replicated_activation,
                "padding_mask_T": token_id_placement(),
            }
        )
    return ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=query_states,
        out_dst_shardings=query_states,
        local_spmd=True,
    )


def selector_sharding_config() -> ShardingConfig:
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    return ShardingConfig(
        in_src_shardings={
            "idx_q": replicated_activation,
            "idx_k": replicated_activation,
            "weights": replicated_activation,
        },
        in_dst_shardings={
            "idx_q": replicated_activation,
            "idx_k": replicated_activation,
            "weights": replicated_activation,
        },
    )


def hc_pre_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    hc_branch_layout = _hc_branch_placement(enable_sp=enable_sp)
    hc_mix_layout = _hc_mix_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings=replicated_state_sharding_config(
            "hc_fn", "hc_base", "hc_scale"
        ).state_shardings,
        in_src_shardings={"x": hc_branch_layout, "pre_mix": hc_mix_layout},
        out_src_shardings=(
            residual_activation_placement(enable_sp=enable_sp),
            hc_mix_layout,
            hc_mix_layout,
            hc_branch_layout,
        ),
    )


def _set_compressor_sharding(compressor_cfg) -> None:
    if compressor_cfg.rope is not None:
        compressor_cfg.rope.sharding_config = _rope_cache_sharding
    if compressor_cfg.wkv is not None:
        compressor_cfg.wkv.sharding_config = _replicated_weight
    if compressor_cfg.wgate is not None:
        compressor_cfg.wgate.sharding_config = _replicated_weight
    if compressor_cfg.norm is not None:
        compressor_cfg.norm.sharding_config = _replicated_weight


def _set_indexer_sharding(indexer_cfg) -> None:
    indexer_cfg.sharding_config = indexer_sharding_config()
    if indexer_cfg.selector is not None:
        indexer_cfg.selector.sharding_config = selector_sharding_config()
    if indexer_cfg.rope is not None:
        indexer_cfg.rope.sharding_config = _rope_cache_sharding
    if indexer_cfg.wq_b is not None:
        indexer_cfg.wq_b.sharding_config = _replicated_weight
    if indexer_cfg.weights_proj is not None:
        indexer_cfg.weights_proj.sharding_config = _replicated_weight
    if indexer_cfg.wk is not None:
        indexer_cfg.wk.sharding_config = _replicated_weight
    if indexer_cfg.k_norm is not None:
        indexer_cfg.k_norm.sharding_config = _replicated_weight


def _set_attention_sharding(attention_cfg, *, enable_sp: bool) -> None:
    attention_cfg.sharding_config = attention_sharding_config(enable_sp=enable_sp)
    if attention_cfg.compress_ratio > 0:
        attention_cfg.inner_attention.sharding_config = (
            csa2_inner_attention_sharding_config(
                compress_ratio=attention_cfg.compress_ratio
            )
        )
    else:
        attention_cfg.inner_attention.sharding_config = (
            dsa_inner_attention_sharding_config(compress_ratio=0)
        )

    replicated_input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    attention_cfg.wq_a.sharding_config = _replicated_weight
    attention_cfg.q_norm.sharding_config = _replicated_weight
    attention_cfg.wq_b.sharding_config = colwise_sharding_config(
        input_layout=replicated_input_layout
    )
    attention_cfg.wkv.sharding_config = _replicated_weight
    attention_cfg.kv_norm.sharding_config = _replicated_weight
    attention_cfg.wo_a.sharding_config = wo_a_sharding_config()
    attention_cfg.wo_b.sharding_config = rowwise_sharding_config(
        output_layout=residual_activation_placement(enable_sp=enable_sp)
    )
    attention_cfg.attn_sink.sharding_config = attn_sink_sharding_config()
    attention_cfg.rope.sharding_config = _rope_cache_sharding

    if attention_cfg.compressor is not None:
        _set_compressor_sharding(attention_cfg.compressor)
    if attention_cfg.indexer is not None:
        _set_indexer_sharding(attention_cfg.indexer)


def set_deepseek_v4_1_sharding_config(
    config,
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    del enable_ep
    config.tok_embeddings.sharding_config = token_embedding_sharding_config(
        enable_sp=enable_sp
    )
    config.norm.sharding_config = pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
    config.lm_head.sharding_config = lm_head_sharding_config()
    for layer_cfg in config.layers:
        hc_pre = hc_pre_sharding_config(enable_sp=enable_sp)
        layer_cfg.hc_attn_pre.sharding_config = hc_pre
        layer_cfg.hc_ffn_pre.sharding_config = hc_pre
        layer_cfg.hc_post.sharding_config = hc_post_sharding_config(enable_sp=enable_sp)
        norm = norm_sharding_config(enable_sp=enable_sp)
        layer_cfg.attention_norm.sharding_config = norm
        layer_cfg.ffn_norm.sharding_config = norm
        _set_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)


__all__ = [
    "attention_sharding_config",
    "csa2_inner_attention_sharding_config",
    "hc_post_sharding_config",
    "hc_pre_sharding_config",
    "indexer_sharding_config",
    "selector_sharding_config",
    "set_deepseek_v4_1_sharding_config",
]
