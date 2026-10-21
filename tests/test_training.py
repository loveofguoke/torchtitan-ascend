# SPDX-License-Identifier: Apache-2.0
"""Run from the pinned TorchTitan checkout with one visible NPU."""

import math
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from torchtitan_ascend import setup
from torchtitan_ascend.recipes.qwen3 import qwen3_debugmodel_npu
from torchtitan_ascend.recipes.qwen3_5 import qwen35_0_8b_npu, qwen35_debugmodel_npu


@pytest.mark.npu
@pytest.mark.parametrize("recipe", [qwen3_debugmodel_npu, qwen35_debugmodel_npu, qwen35_0_8b_npu])
def test_training_updates_parameters(recipe, tmp_path, monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")

    if recipe is qwen35_0_8b_npu and not (
        Path("assets/hf/Qwen3.5-0.8B/tokenizer.json").is_file()
        and Path("assets/c4/en/c4-train.00000-of-01024.json.gz").is_file()
    ):
        pytest.skip("requires the Qwen3.5-0.8B tokenizer and C4 shard from docs/models/qwen3_5.md")

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

    if recipe is not qwen3_debugmodel_npu:
        pytest.importorskip("fla_npu")

    setup()
    config = recipe()
    config.dump_folder = str(tmp_path)
    config.debug.seed = 42
    trainer = None
    try:
        trainer = config.build()
        model = trainer.engine.model_parts[0]
        names = ["tok_embeddings.weight"]
        if recipe is not qwen3_debugmodel_npu:
            names.append("layers.0.attn.conv_q.weight")
        parameters = {name: model.get_parameter(name) for name in names}
        before = {name: value.detach().clone() for name, value in parameters.items()}
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
        for name, parameter in parameters.items():
            assert torch.isfinite(parameter).all().item(), name
            assert not torch.equal(before[name], parameter), f"optimizer did not update {name}"
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
