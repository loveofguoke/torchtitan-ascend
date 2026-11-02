# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
#
# Modification points:
# 1. Adapt V4 HF mappings to V4.1 source/reuse ownership;
# 2. Remove V4-only, MTP and hash-routing entries;
# 3. Filter unsupported HF checkpoint weights.
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

"""The V4.1 state-dict adapter: the V4.1-effective subset of the V4 mapping.

Inherits upstream V4's :class:`DeepSeekV4StateDictAdapter` for the bulk of
the HF<->torchtitan key mapping (embeddings, attention projections, norms,
MoE experts and the mHC coefficients). V4.1 drops the V4-only entries (the
HcHead merge, the MTP stack and the hash-routing ``tid2eid`` table) and
re-maps the compressor/indexer weights conditionally on V4.1's source/reuse
wiring: only source layers own compressor weights, and only Full/Reindex
indexers own projection weights, so reuse layers map nothing.
"""

from __future__ import annotations

from torchtitan.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter

from .indexer import REUSE


class DeepSeekV41StateDictAdapter(DeepSeekV4StateDictAdapter):
    """V4.1 adapter: inherits V4's mapping, overrides the V4.1 differences."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self._drop_hf_keys = set()

        for v4_only in (
            "layers.{}.enorm.weight",
            "layers.{}.hnorm.weight",
            "layers.{}.e_proj.weight",
            "layers.{}.h_proj.weight",
            "layers.{}.norm.weight",
            "layers.{}.hc_head_base",
            "layers.{}.hc_head_fn",
            "layers.{}.hc_head_scale",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
            "layers.{}.ffn.gate.tid2eid",
        ):
            self.from_hf_map.pop(v4_only, None)
            if "{}" in v4_only:
                self._drop_hf_keys.update(
                    v4_only.format(layer_id) for layer_id in range(model_config.n_layers)
                )
            else:
                self._drop_hf_keys.add(v4_only)

        for layer_id in range(model_config.n_layers):
            for v4_key in (
                f"layers.{layer_id}.attn.compressor.ape",
                f"layers.{layer_id}.attn.compressor.norm.weight",
                f"layers.{layer_id}.attn.compressor.wgate.weight",
                f"layers.{layer_id}.attn.compressor.wkv.weight",
                f"layers.{layer_id}.attn.indexer.compressor.ape",
                f"layers.{layer_id}.attn.indexer.compressor.norm.weight",
                f"layers.{layer_id}.attn.indexer.compressor.wgate.weight",
                f"layers.{layer_id}.attn.indexer.compressor.wkv.weight",
                f"layers.{layer_id}.attn.indexer.wq_b.weight",
                f"layers.{layer_id}.attn.indexer.weights_proj.weight",
            ):
                self.from_hf_map.pop(v4_key, None)
                self._drop_hf_keys.add(v4_key)

            layer_cfg = model_config.layers[layer_id]
            compressor_cfg = layer_cfg.attention.compressor
            if compressor_cfg is not None and compressor_cfg.is_source:
                self.from_hf_map[f"layers.{layer_id}.attn.compressor.norm.weight"] = (
                    f"layers.{layer_id}.attention.compressor.norm.weight"
                )
                self.from_hf_map[f"layers.{layer_id}.attn.compressor.wkv.weight"] = (
                    f"layers.{layer_id}.attention.compressor.wkv.weight"
                )
                if compressor_cfg.wgate is not None:
                    self.from_hf_map[
                        f"layers.{layer_id}.attn.compressor.wgate.weight"
                    ] = f"layers.{layer_id}.attention.compressor.wgate.weight"

            indexer_cfg = layer_cfg.attention.indexer
            if indexer_cfg is not None and indexer_cfg.mode is not REUSE:
                self.from_hf_map[f"layers.{layer_id}.attn.indexer.wq_b.weight"] = (
                    f"layers.{layer_id}.attention.indexer.wq_b.weight"
                )
                self.from_hf_map[
                    f"layers.{layer_id}.attn.indexer.weights_proj.weight"
                ] = f"layers.{layer_id}.attention.indexer.weights_proj.weight"
                if indexer_cfg.wk is not None:
                    self.from_hf_map[f"layers.{layer_id}.attn.indexer.wk.weight"] = (
                        f"layers.{layer_id}.attention.indexer.wk.weight"
                    )
                    if indexer_cfg.k_norm is not None:
                        self.from_hf_map[
                            f"layers.{layer_id}.attn.indexer.k_norm.weight"
                        ] = f"layers.{layer_id}.attention.indexer.k_norm.weight"

        self._drop_hf_keys.difference_update(self.from_hf_map)

    def from_hf(self, hf_state_dict, *, is_quantized_layout=False):
        filtered = {
            key: value
            for key, value in hf_state_dict.items()
            if not key.startswith("mtp.") and key not in self._drop_hf_keys
        }
        return super().from_hf(filtered, is_quantized_layout=is_quantized_layout)
