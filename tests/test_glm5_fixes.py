from functools import partial

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.protocols.module import Module
from torchtitan_ascend.models.glm5 import build_model_config
from torchtitan_ascend.overrides.glm5 import npu_safe_trunc_normal_, replace_trunc_normal_initializers
from torchtitan_ascend.patches.vocab_parallel_loss import _local_vocab_labels


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


def test_local_vocab_labels_avoids_out_of_range_indices():
    labels = torch.tensor([-100, 0, 3, 4, 7, 8])

    local_labels, out_of_range = _local_vocab_labels(
        labels,
        vocab_start=4,
        local_vocab_size=4,
        ignore_index=-100,
    )

    torch.testing.assert_close(local_labels, torch.tensor([0, 0, 0, 0, 3, 0]))
    torch.testing.assert_close(out_of_range, torch.tensor([True, True, True, False, False, True]))
