# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Expose only public V4.1 interfaces and move implementation details to flavors.py.
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

"""DeepSeek V4.1 model flavors and public config builders."""

from .flavors import (
    Attention,
    build_model_config,
    CANDIDATE_BLOCK_SIZE,
    DeepSeekV41Model,
    layer_policies,
    MODEL_FLAVORS,
    V41_CANDIDATE_SOURCE_LAYER,
    V41_FULL_COMPRESS_RATIOS,
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
    V41LayerPolicy,
    V41Spec,
)

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
