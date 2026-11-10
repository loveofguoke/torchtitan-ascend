# SPDX-License-Identifier: Apache-2.0

"""Check deterministic Kimi K3 resume from a native DCP checkpoint."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

STEPS = 10
CHECKPOINT_STEP = 5
METRICS_PREFIX = "KIMI_K3_RESUME_METRICS "


def _train(directory: Path) -> None:
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from torchtitan_ascend import setup

    setup()
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.observability.logging import init_logger

    from torchtitan_ascend.recipes.kimi_k3 import kimi_k3_debugmodel_npu

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    init_logger()
    config = kimi_k3_debugmodel_npu()
    config.dump_folder = str(directory)
    config.training.steps = STEPS
    config.debug.seed = config.dataloader.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.debug.enable_structured_logging = False
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = False
    config.checkpointer = CheckpointManager.Config(
        interval=CHECKPOINT_STEP, last_save_model_only=False
    )
    config.__post_init__()

    trainer = None
    try:
        trainer = config.build()
        metrics: list[tuple[int, float, float]] = []
        original_log = trainer.metrics_processor.log

        def capture(step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs):
            metrics.append((int(step), float(global_avg_loss), float(grad_norm)))
            return original_log(
                step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs
            )

        trainer.metrics_processor.log = capture
        trainer.train()
        torch.npu.synchronize()
        assert trainer.engine.num_completed_steps == STEPS
        print(METRICS_PREFIX + json.dumps([dist.get_rank(), metrics]), flush=True)
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_training(directory: Path, num_devices: int, steps: range):
    env = dict(os.environ, PYTHONHASHSEED="42")
    test_file = Path(__file__).resolve()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(test_file.parents[2]), str(Path.cwd()), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={num_devices}",
            str(test_file),
            str(directory),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-12000:]
    by_rank = {}
    for line in result.stdout.splitlines():
        if line.startswith(METRICS_PREFIX):
            rank, metrics = json.loads(line.removeprefix(METRICS_PREFIX))
            assert rank not in by_rank, f"duplicate metrics from rank {rank}"
            assert [row[0] for row in metrics] == list(steps), (rank, metrics)
            assert all(
                math.isfinite(loss) and math.isfinite(norm) for _, loss, norm in metrics
            )
            by_rank[rank] = metrics
    assert set(by_rank) == set(range(num_devices)), result.stdout[-12000:]
    return by_rank


def _assert_checkpoint_equal(expected, actual, path: str = "checkpoint") -> None:
    assert type(actual) is type(expected), f"{path}: different value types"
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, msg=lambda error: f"{path}: {error}"
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: different keys"
        for key in expected:
            _assert_checkpoint_equal(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: different lengths"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_checkpoint_equal(left, right, f"{path}/{index}")
    else:
        assert actual == expected, f"{path}: different values"


@pytest.mark.npu
@pytest.mark.parametrize(
    "num_devices",
    [
        pytest.param(1, id="single"),
        pytest.param(2, id="fsdp2", marks=pytest.mark.multi_gpu),
    ],
)
def test_kimi_k3_checkpoint_resume(num_devices):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or torch.npu.device_count() < num_devices:
        pytest.skip(f"requires {num_devices} visible Ascend NPU(s)")
    if not Path("run_train.sh").is_file() or not Path("torchtitan").is_dir():
        pytest.fail("run this test from the pinned TorchTitan checkout")

    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        continuous, resumed = root / "continuous", root / "resumed"
        reference_metrics = _run_training(continuous, num_devices, range(1, STEPS + 1))
        checkpoint = Path("checkpoint") / f"step-{CHECKPOINT_STEP}"
        shutil.copytree(continuous / checkpoint, resumed / checkpoint)
        resumed_metrics = _run_training(
            resumed, num_devices, range(CHECKPOINT_STEP + 1, STEPS + 1)
        )
        for rank, metrics in resumed_metrics.items():
            assert (
                metrics == reference_metrics[rank][CHECKPOINT_STEP:]
            ), f"rank {rank}: loss or grad norm differs after resume"

        states = []
        consolidated = root / "state.pt"
        for run in (continuous, resumed):
            dcp_to_torch_save(run / "checkpoint" / f"step-{STEPS}", consolidated)
            states.append(
                torch.load(consolidated, map_location="cpu", weights_only=False)
            )
        reference, actual = states
        assert {
            "optimizer",
            "lr_scheduler",
            "dataloader",
            "train_state",
        } <= reference.keys()
        assert reference["optimizer"], "expected optimizer state in the full checkpoint"
        assert reference["train_state"]["step"] == STEPS
        _assert_checkpoint_equal(reference, actual)


if __name__ == "__main__":
    _train(Path(sys.argv[1]))
