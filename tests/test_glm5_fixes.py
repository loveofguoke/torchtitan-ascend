from functools import partial

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.protocols.module import Module
from torchtitan_ascend.models.glm5 import build_model_config
from torchtitan_ascend.overrides.glm5 import npu_safe_trunc_normal_, replace_trunc_normal_initializers


def test_replace_glm5_trunc_normal_initializers():
    config = build_model_config(
        "glm5_2_debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=128,
    )

    count = replace_trunc_normal_initializers(config)

    assert count > 0
    for _, module_config, _, _ in config.traverse(Module.Config, recurse=True):
        param_init = getattr(module_config, "param_init", None)
        if param_init is None:
            continue
        for initializer in param_init.values():
            if isinstance(initializer, partial):
                assert initializer.func is not torch.nn.init.trunc_normal_


def test_npu_safe_trunc_normal_cpu_fallback():
    tensor = torch.empty(512)
    result = npu_safe_trunc_normal_(tensor, std=0.02)

    assert result is tensor
    assert torch.isfinite(tensor).all()
    assert tensor.min() >= -2.0
    assert tensor.max() <= 2.0
