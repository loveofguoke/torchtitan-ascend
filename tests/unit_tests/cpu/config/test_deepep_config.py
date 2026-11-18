# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the DeepEP dispatcher config and recipe wiring."""

import pytest
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from torchtitan_recipes.tests.models.qwen3 import qwen3_moe_debug

from torchtitan_ascend.overrides.deepep import (
    AscendDeepEPTokenDispatcher,
    expert_hidden_dim,
    npu_deepep,
)


def _trainer_config(ep: int):
    return qwen3_moe_debug(parallelism=ParallelismConfig(expert_parallel_degree=ep))


def test_factory_derives_config():
    param_init = {"weight": object()}
    cfg = AllToAllTokenDispatcher.Config(num_experts=8, top_k=4, param_init=param_init)
    out = npu_deepep(cfg, num_max_tokens_per_rank=128, hidden_dim=256)
    assert isinstance(out, AscendDeepEPTokenDispatcher.Config)
    assert (out.num_experts, out.top_k) == (8, 4)
    assert out.param_init is param_init
    assert (
        out.num_max_tokens_per_rank,
        out.hidden_dim,
        out.expert_alignment,
    ) == (128, 256, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"expert_alignment": 2},
        {"num_max_tokens_per_rank": 0},
        {"hidden_dim": 0},
    ],
)
def test_config_rejects_invalid_options(overrides):
    kwargs = {"num_max_tokens_per_rank": 128, "hidden_dim": 256, **overrides}
    with pytest.raises(ValueError):
        AscendDeepEPTokenDispatcher.Config(num_experts=8, top_k=4, **kwargs)


def test_expert_hidden_dim_from_model():
    config = _trainer_config(ep=2)
    assert expert_hidden_dim(config.model) == 256
