# SPDX-License-Identifier: Apache-2.0
r"""Keep collective results live across compiled consumers.

Run with:
    TORCHINDUCTOR_NPU_BACKEND=triton_experimental torchrun --nproc-per-node=2 \
        -m pytest -q tests/test_collective_wait.py
"""

import math
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module")
def tp_runtime():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun with at least two physical NPUs")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPUs and torch_npu")
    assert (
        os.environ.get("TORCHINDUCTOR_NPU_BACKEND") == "triton_experimental"
    ), "set TORCHINDUCTOR_NPU_BACKEND=triton_experimental to test the affected backend"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        yield torch.device("npu", local_rank)
    finally:
        if created:
            dist.destroy_process_group()


def _distributed_log_probs(logits):
    maximum = funcol.all_reduce(logits.amax(-1, keepdim=True), "max", dist.group.WORLD)
    shifted = logits - maximum
    denominator = funcol.all_reduce(
        shifted.exp().sum(-1, keepdim=True), "sum", dist.group.WORLD
    )
    return shifted - denominator.log()


@pytest.mark.parametrize("shape", [(1, 2), (64, 1024)])
def test_compiled_wait_preserves_live_collective_input(tp_runtime, shape):
    """The maximum remains live when the denominator collective is computed."""
    logits = torch.zeros(shape, device=tp_runtime)
    expected = torch.full_like(logits, -math.log(shape[1] * dist.get_world_size()))
    torch.testing.assert_close(
        _distributed_log_probs(logits), expected, rtol=1e-5, atol=1e-6
    )
    compiled = torch.compile(_distributed_log_probs, fullgraph=True)
    torch.testing.assert_close(compiled(logits), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_broadcast_accepts_transposed_input(tp_runtime, dtype):
    """The preserved native lowering must still meet HCCL's layout contract."""
    rank, world = dist.get_rank(), dist.get_world_size()
    pending_before = torch._C._distributed_c10d._get_work_registry_size()
    base = (torch.arange(1024).reshape(16, 64) % 11 / 16).to(dtype)
    values = (base + rank).t().contiguous().t().to(tp_runtime)
    assert not values.is_contiguous()

    def broadcast(x):
        return funcol.broadcast(x, world - 1, dist.group.WORLD)

    expected = base + world - 1
    eager = broadcast(values).cpu()
    actual = torch.compile(broadcast, fullgraph=True)(values).cpu()
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(values.cpu(), base + rank, rtol=0, atol=0)
    assert torch._C._distributed_c10d._get_work_registry_size() == pending_before
