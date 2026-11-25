# SPDX-License-Identifier: BSD-3-Clause

"""Kimi K3.1 Hugging Face checkpoint key mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model


class KimiK31StateDictAdapter(KimiK3StateDictAdapter):
    """Extend the K3 MLA mapping with the five DSA Indexer tensors."""

    mla_from_hf_map: dict[str, str]

    def __init__(
        self,
        model_config: KimiK3Model.Config,
        hf_assets_path: str | None,
    ) -> None:
        super().__init__(model_config, hf_assets_path)
        self.mla_from_hf_map.update(
            {
                f"language_model.model.layers.{{}}.self_attn.indexer.{name}": (
                    f"layers.{{}}.attention.indexer.{name}"
                )
                for name in (
                    "wq_b.weight",
                    "wk.weight",
                    "k_norm.weight",
                    "k_norm.bias",
                    "weights_proj.weight",
                )
            }
        )
