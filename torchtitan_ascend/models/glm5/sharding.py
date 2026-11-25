# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative SPMD layouts for GLM-5-specific attention components."""

import spmd_types as spmd
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    residual_activation_placement,
    token_id_placement,
)


def glm5_attention_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Declare the residual layout; fused MLA gathers its input once."""
    residual = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x_TD": residual, "positions": token_id_placement()},
        out_src_shardings=residual,
    )


def glm5_indexer_sharding_config() -> ShardingConfig:
    """Keep indexer projections replicated over TP."""
    return ShardingConfig()


def replicated_weight_sharding_config(*, bias: bool = False) -> ShardingConfig:
    """Replicate an indexer projection over TP."""
    states = {"weight": dense_param_placement(tp=spmd.R)}
    if bias:
        states["bias"] = dense_param_placement(tp=spmd.R)
    return ShardingConfig(state_shardings=states)
