"""Run from the pinned TorchTitan checkout with one visible NPU."""

import math
import socket
from functools import partial

import pytest
import torch
import torch.distributed as dist

from torchtitan.models.base import Module
from torchtitan_ascend import setup
from torchtitan_ascend.changes import has_override
from torchtitan_ascend.modules.glm5 import npu_safe_trunc_normal_
from torchtitan_ascend.ops.rope import npu_complex_rope
from torchtitan_ascend.recipes.glm5 import (
    glm5_2_debugmodel_mtp_npu,
    glm5_2_debugmodel_npu,
    glm5_3_debugmodel_mtp_npu,
    glm5_3_debugmodel_npu,
    glm5_debugmodel_npu,
)


@pytest.mark.parametrize(
    "recipe",
    [glm5_2_debugmodel_npu, glm5_3_debugmodel_npu],
)
def test_glm5_debugmodel_npu_config(recipe):
    config = recipe()

    assert config.model.local_compile_regions == []
    assert config.activation_remat is None
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 256
    assert config.training.max_context_length == 128
    assert config.training.steps == 10
    assert has_override(config, npu_complex_rope)
    assert any(
        isinstance(initializer, partial) and initializer.func is npu_safe_trunc_normal_
        for _, module_config, _, _ in config.model.traverse(Module.Config, recurse=True)
        for initializer in (module_config.param_init or {}).values()
    )


@pytest.mark.parametrize(
    "recipe",
    [glm5_2_debugmodel_mtp_npu, glm5_3_debugmodel_mtp_npu],
)
def test_glm5_mtp_npu_config(recipe):
    config = recipe(mtp_scale=0.125)

    assert len(config.model.mtp_layers) == 1
    assert config.dataloader.num_mtp_layers == 1
    assert config.loss.loss_fn.mtp_scale == 0.125


def test_glm5_single_card(tmp_path, monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    for key, value in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
    }.items():
        monkeypatch.setenv(key, value)

    setup()
    config = glm5_debugmodel_npu()
    config.dump_folder = str(tmp_path)
    config.debug.seed = 42
    config.training.steps = 2
    trainer = None
    try:
        trainer = config.build()
        parameter = next(trainer.engine.model_parts[0].parameters())
        before = parameter.detach().clone()
        metrics = []
        original_log = trainer.metrics_processor.log

        def capture(step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs):
            metrics.append((global_avg_loss, grad_norm))
            return original_log(step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs)

        monkeypatch.setattr(trainer.metrics_processor, "log", capture)
        trainer.train()
        torch.npu.synchronize()
        assert trainer.engine.num_completed_steps == 2
        assert len(metrics) == 2
        assert all(math.isfinite(value) for row in metrics for value in row)
        assert torch.isfinite(parameter).all().item()
        assert not torch.equal(before, parameter), "optimizer did not update the model"
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
