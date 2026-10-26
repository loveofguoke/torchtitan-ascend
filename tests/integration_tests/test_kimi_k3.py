# SPDX-License-Identifier: Apache-2.0

"""Check Kimi K3 loss accuracy, global gradient norms and deterministic replay."""

import math
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
STEPS = 20
FIRST_STEP_RTOL = 0.005
RTOL = 0.01
NORM_RTOL = 0.05
DETERMINISTIC_RTOL = 1e-6
NGPU = 2


def read_metrics(path: str | Path) -> tuple[dict[int, float], dict[int, float]]:
    """Read twenty ordered rows of step, loss and global gradient norm."""
    losses: dict[int, float] = {}
    norms: dict[int, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        step_text, loss_text, norm_text = line.split()
        step, loss, norm = int(step_text), float(loss_text), float(norm_text)
        assert step not in losses, f"duplicate step {step}: {path}"
        assert math.isfinite(loss) and loss > 0, f"invalid value for loss at step {step}: {path}"
        assert math.isfinite(norm) and norm >= 0, f"invalid value for grad norm at step {step}: {path}"
        losses[step], norms[step] = loss, norm
    assert list(losses) == list(range(1, STEPS + 1)), f"expected exactly steps 1..{STEPS}: {path}"
    return losses, norms


def compare_losses(reference, candidate, *, deterministic=False):
    """Check baseline loss accuracy, or deterministic replay within a small relative tolerance."""
    baseline, _ = read_metrics(reference)
    actual, _ = read_metrics(candidate)
    if deterministic:
        for step, value in baseline.items():
            assert torch.isclose(
                torch.tensor(actual[step], dtype=torch.float64),
                torch.tensor(value, dtype=torch.float64),
                rtol=DETERMINISTIC_RTOL,
                atol=0,
            ), (
                f"deterministic loss mismatch at step {step}: "
                f"baseline={value!r}, actual={actual[step]!r}, rtol={DETERMINISTIC_RTOL}, atol=0. "
                f"Log: {candidate}"
            )
        print(f"PASS: {STEPS}-step deterministic loss comparison, rtol={DETERMINISTIC_RTOL}, atol=0")
        return
    errors = {step: abs(actual[step] - value) / value for step, value in baseline.items()}
    assert errors[1] < FIRST_STEP_RTOL, (
        f"first-step loss relative error: {errors[1]:.6%} >= {FIRST_STEP_RTOL:.3%}; "
        f"baseline={baseline[1]}, actual={actual[1]}. Log: {candidate}"
    )
    worst = max(errors, key=lambda step: errors[step])
    assert errors[worst] < RTOL, (
        f"loss relative error at step {worst}: {errors[worst]:.6%} >= {RTOL:.3%}; "
        f"baseline={baseline[worst]}, actual={actual[worst]}. Log: {candidate}"
    )
    print(f"PASS: {STEPS}-step loss comparison, first relative error={errors[1]:.6%}, max={errors[worst]:.6%}")


def compare_norms(reference, candidate, *, exact=False):
    """Require every norm within +/-5%, or a small relative tolerance for deterministic replay."""
    _, baseline = read_metrics(reference)
    _, actual = read_metrics(candidate)
    for step, value in baseline.items():
        if exact:
            assert torch.isclose(
                torch.tensor(actual[step], dtype=torch.float64),
                torch.tensor(value, dtype=torch.float64),
                rtol=DETERMINISTIC_RTOL,
                atol=0,
            ), (
                f"deterministic grad norm mismatch at step {step}: "
                f"baseline={value!r}, actual={actual[step]!r}. Log: {candidate}"
            )
            continue
        if not value:
            assert torch.isclose(
                torch.tensor(actual[step], dtype=torch.float64),
                torch.tensor(0, dtype=torch.float64),
                rtol=0,
                atol=0,
            ), f"nonzero grad norm for zero baseline at step {step}: {candidate}"
            continue
        relative_change = (actual[step] - value) / value
        assert -NORM_RTOL <= relative_change <= NORM_RTOL, (
            f"grad norm relative change at step {step}: {relative_change:+.6%} outside +/-{NORM_RTOL:.1%}; "
            f"baseline={value}, actual={actual[step]}. Log: {candidate}"
        )
    criterion = f"rtol={DETERMINISTIC_RTOL}, atol=0" if exact else f"every step within +/-{NORM_RTOL:.1%}"
    print(f"PASS: {STEPS}-step global grad norm comparison, {criterion}")


def export_metrics(directory: Path) -> Path:
    """Export loss and global gradient norm into a single file aligned by step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runs = [path for path in (directory / "tb").iterdir() if path.is_dir()]
    assert len(runs) == 1, f"expected one TensorBoard run: {directory / 'tb'}"
    events = EventAccumulator(str(runs[0]), size_guidance={"scalars": 0})
    events.Reload()
    losses = events.Scalars("loss_metrics/global_avg_loss")
    norms = events.Scalars("grad_norm")
    assert [event.step for event in losses] == [event.step for event in norms], f"loss/norm step mismatch: {directory}"
    log = directory / "metrics.log"
    log.write_text(
        "# step loss grad_norm\n"
        + "".join(f"{loss.step} {loss.value!r} {norm.value!r}\n" for loss, norm in zip(losses, norms, strict=True)),
        encoding="utf-8",
    )
    read_metrics(log)
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
    return export_metrics(directory)


@pytest.mark.npu
def test_kimi_k3_fsdp2():
    baseline = ROOT / "tests/assets/losses/kimi_k3/kimi_k3_fsdp2.txt"
    assert baseline.is_file(), f"missing current-PR baseline: {baseline}"
    read_metrics(baseline)
    log = _run_training()
    compare_losses(baseline, log)
    compare_norms(baseline, log)


@pytest.mark.npu
def test_kimi_k3_fsdp2_deterministic():
    first = _run_training()
    second = _run_training()
    compare_losses(first, second, deterministic=True)
    compare_norms(first, second, exact=True)
