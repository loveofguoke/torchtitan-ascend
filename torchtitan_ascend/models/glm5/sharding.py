# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative SPMD layouts for GLM-5-specific attention components."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    pre_lm_head_norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    token_id_placement,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def residual_activation_placement(*, enable_sp: bool) -> SpmdType:
    return dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))


def replicated_state_sharding_config(*state_names: str) -> ShardingConfig:
    return ShardingConfig(state_shardings={name: dense_param_placement(tp=spmd.R) for name in state_names})


def rope_cache_sharding_config() -> ShardingConfig:
    return replicated_state_sharding_config("cache")


def norm_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    return norm_config(enable_sp=enable_sp)


def pre_lm_head_norm_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    return pre_lm_head_norm_config(enable_sp=enable_sp)


def token_embedding_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    activation = residual_activation_placement(enable_sp=enable_sp)
    partial = dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
    tokens = token_id_placement()
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": tokens},
        in_dst_shardings={"input": tokens},
        out_src_shardings=partial,
        out_dst_shardings=activation,
        local_spmd=True,
    )


def lm_head_sharding_config() -> ShardingConfig:
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    vocab_sharded = dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": replicated},
        in_dst_shardings={"input": replicated},
        out_src_shardings=vocab_sharded,
        out_dst_shardings=vocab_sharded,
    )


def mtp_projection_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    activation = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )


def mtp_block_sharding_config(*, enable_sp: bool) -> ShardingConfig | None:
    if not enable_sp:
        return None
    return ShardingConfig(
        in_src_shardings={"mtp_input_valid_mask": token_id_placement()},
    )


def glm5_attention_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Upstream MLA's external residual contract; forward owns TP gathering."""
    residual = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x_TD": residual, "positions": token_id_placement()},
        out_src_shardings=residual,
    )


def glm5_indexer_sharding_config() -> ShardingConfig:
    """Indexer projections stay replicated over TP."""
    return ShardingConfig()


def replicated_weight_sharding_config(*, bias: bool = False) -> ShardingConfig:
    """Replicate a GLM-specific projection or norm across TP."""
    states = {"weight": dense_param_placement(tp=spmd.R)}
    if bias:
        states["bias"] = dense_param_placement(tp=spmd.R)
    return ShardingConfig(state_shardings=states)


def set_glm5_sharding_config(
    config,
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Refresh GLM-5 sharding for the selected TorchTitan parallelism."""
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    activation = residual_activation_placement(enable_sp=enable_sp)
    norm = norm_config(enable_sp=enable_sp)
    for layer in [*config.layers, *config.mtp_layers]:
        layer.attention_norm.sharding_config = norm
        layer.ffn_norm.sharding_config = norm
        layer.attention.sharding_config = glm5_attention_sharding_config(enable_sp=enable_sp)
        replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        layer.attention.wq_b.sharding_config = colwise_config(input_layout=replicated)
        layer.attention.wkv_b.sharding_config = colwise_config(input_layout=replicated)
        layer.attention.wo.sharding_config = rowwise_config(output_layout=activation)
        if layer.feed_forward is not None:
            set_dense_ffn_sharding(
                layer.feed_forward,
                attn_x_layout=activation,
                enable_sp=enable_sp,
            )
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=enable_ep,
                enable_sp=enable_sp,
            )
