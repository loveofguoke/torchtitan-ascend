# SPDX-License-Identifier: Apache-2.0
"""Run from the pinned TorchTitan checkout with one visible NPU."""

import math
import socket

import pytest
import torch
import torch.distributed as dist

from torchtitan_ascend import setup
from torchtitan_ascend.recipes.qwen3 import qwen3_npu


def test_native_qwen3_updates_parameters(tmp_path, monkeypatch):
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
    config = qwen3_npu()
    config.dump_folder = str(tmp_path)
    config.debug.seed = 42
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
        assert trainer.engine.num_completed_steps == config.training.steps
        assert len(metrics) == config.training.steps
        assert all(math.isfinite(value) for row in metrics for value in row)
        assert torch.isfinite(parameter).all().item()
        assert not torch.equal(before, parameter), "optimizer did not update the model"
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
