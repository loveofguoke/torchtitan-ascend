# SPDX-License-Identifier: Apache-2.0

"""Compare Kimi K3 losses using the TorchTitan scripts/loss_compare.py approach."""

import math
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STEPS = 20
RTOL = 0.002
NGPU = 2


def read_losses(path: str | Path) -> dict[int, float]:
    """Read TorchTitan's golden-result format: one <step> <loss> row per step."""
    losses: dict[int, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        step_text, loss_text = line.split()
        step, loss = int(step_text), float(loss_text)
        assert step not in losses, f"duplicate step {step}: {path}"
        assert math.isfinite(loss) and loss > 0, f"invalid loss at step {step}: {path}"
        losses[step] = loss
    assert list(losses) == list(range(1, STEPS + 1)), f"expected exactly steps 1..{STEPS}: {path}"
    return losses


def compare_losses(reference, candidate):
    """Fail CI if any step differs from the fixed reference by two per mille or more."""
    baseline, actual = read_losses(reference), read_losses(candidate)
    errors = {step: abs(value - actual[step]) / abs(value) for step, value in baseline.items()}
    worst = max(errors, key=lambda step: errors[step])
    assert errors[worst] < RTOL, (
        f"loss relative error at step {worst}: {errors[worst]:.6%} >= {RTOL:.3%}; "
        f"baseline={baseline[worst]}, actual={actual[worst]}. Log: {candidate}"
    )
    print(f"PASS: {STEPS}-step loss comparison, max relative error={errors[worst]:.6%}")


def export_losses(directory: Path) -> Path:
    """Read the same TensorBoard scalar as TorchTitan and save a golden-format log."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runs = [path for path in (directory / "tb").iterdir() if path.is_dir()]
    assert len(runs) == 1, f"expected one TensorBoard run: {directory / 'tb'}"
    events = EventAccumulator(str(runs[0]), size_guidance={"scalars": 0})
    events.Reload()
    scalars = events.Scalars("loss_metrics/global_avg_loss")
    log = directory / "loss.log"
    log.write_text("# step loss\n" + "".join(f"{event.step} {event.value!r}\n" for event in scalars), encoding="utf-8")
    read_losses(log)
    return log


def precision_recipe():
    """Keep the product recipe; fix only the twenty-step test and random seeds."""
    from torchtitan_ascend.recipes.kimi_k3 import kimi_k3_debugmodel_npu

    config = kimi_k3_debugmodel_npu()
    config.training.steps = STEPS
    config.debug.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.debug.enable_structured_logging = False
    config.dataloader.seed = 42
    config.metrics.log_freq = 1
    config.metrics.disable_color_printing = True
    config.metrics.enable_tensorboard = True
    config.metrics.save_tb_folder = "tb"
    return config


def _run_training():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available() or torch.npu.device_count() < NGPU:
        pytest.skip(f"requires {NGPU} visible Ascend NPUs")
    if not Path("run_train.sh").is_file() or not Path("torchtitan").is_dir():
        pytest.fail("run this test from the pinned TorchTitan directory containing run_train.sh and torchtitan/")
    output = ROOT / "outputs/kimi_k3"
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="fsdp2_", dir=output))
    log_path = directory / "train.log"
    env = os.environ.copy()
    env.update(
        NGPU=str(NGPU),
        LOG_RANK="0",
        ENTRYPOINT="torchtitan_ascend.train",
        MODULE="test_kimi_k3",
        CONFIG="precision_recipe",
        PYTHONHASHSEED="42",
        PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(ROOT), env.get("PYTHONPATH", "")]),
    )
    env.pop("COMM_BACKEND", None)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            ["bash", "./run_train.sh", "--output-dir", str(directory)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    assert result.returncode == 0, f"training failed (rc={result.returncode}): {log_path}"
    return export_losses(directory)


@pytest.mark.npu
def test_kimi_k3_fsdp2():
    baseline = ROOT / "tests/assets/losses/kimi_k3/kimi_k3_fsdp2.txt"
    assert baseline.is_file(), f"missing current-PR loss baseline: {baseline}"
    read_losses(baseline)
    log = _run_training()
    compare_losses(baseline, log)
