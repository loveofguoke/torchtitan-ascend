"""Qwen3.5 text recipes and native override resolution."""

import pytest
from torchtitan.config import apply_overrides
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.gdn import InnerGDN

from torchtitan_ascend.modules.gdn import FlaGDNKernel, FlaInnerGDN
from torchtitan_ascend.recipes.qwen3_5 import (
    qwen35_0_8b_npu,
    qwen35_debugmodel_npu,
)


@pytest.mark.parametrize("recipe", [qwen35_debugmodel_npu, qwen35_0_8b_npu])
def test_text_recipe_resolves_gdn_and_preserves_native_attention(recipe):
    if recipe is qwen35_0_8b_npu:
        config = recipe(hf_assets_path="/data/tokenizer", data_path="/data/train.jsonl")
        assert config.hf_assets_path == "/data/tokenizer"
        assert config.dataloader.dataset.dataset.source.load_dataset_kwargs == {
            "data_files": "/data/train.jsonl"
        }
    else:
        config = recipe()

    original_gdn = list(config.model.traverse(InnerGDN.Config))
    original_attention = [
        (path, id(cfg)) for path, cfg, *_ in config.model.traverse(VarlenInnerAttention.Config)
    ]
    assert original_gdn and original_attention
    apply_overrides(config.override, config)
    replacement = list(config.model.traverse(FlaInnerGDN.Config))
    assert len(replacement) == len(original_gdn)
    assert all(isinstance(cfg.kernel, FlaGDNKernel.Config) for _, cfg, *_ in replacement)
    assert original_attention == [
        (path, id(cfg)) for path, cfg, *_ in config.model.traverse(VarlenInnerAttention.Config)
    ]
