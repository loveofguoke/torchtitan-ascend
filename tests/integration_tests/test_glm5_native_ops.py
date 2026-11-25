# SPDX-License-Identifier: Apache-2.0
"""Run native GLM initialization and vocab-parallel loss under torchrun."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import empty, Replicate, Shard

from torchtitan.components.loss import _LossParallelCrossEntropy

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module")
def tp_mesh():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires torchrun with at least two visible Ascend NPUs")
    device = torch.device(f"npu:{int(os.environ['LOCAL_RANK'])}")
    torch.npu.set_device(device)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("hccl", device_id=device, timeout=timedelta(seconds=180))
    try:
        yield init_device_mesh("npu", (world_size,), mesh_dim_names=("tp",))
    finally:
        if owns_group:
            dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("placement", [Replicate(), Shard(0), Shard(1)], ids=["replicate", "shard0", "shard1"])
@pytest.mark.parametrize("bound", [2.0, 0.04], ids=["glm_default_bounds", "two_sigma_bounds"])
def test_native_dtensor_trunc_normal(tp_mesh, dtype, placement, bound):
    """Check native sampling, bounds, repeatability and cross-rank RNG semantics."""

    def sample():
        torch.manual_seed(20261103)
        tensor = empty((512, 512), dtype=dtype, device_mesh=tp_mesh, placements=[placement])
        assert torch.nn.init.trunc_normal_(tensor, std=0.02, a=-bound, b=bound) is tensor
        return tensor.to_local().float().cpu()

    actual = sample()
    torch.testing.assert_close(actual, sample(), atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    cutoff = torch.tensor(bound, dtype=dtype).float().item()
    assert actual.min().item() >= -cutoff
    assert actual.max().item() <= cutoff
    expected_std = 0.02 if bound == 2.0 else 0.017592513220684797
    assert abs(actual.mean().item()) < expected_std * 0.03
    assert abs(actual.std().item() - expected_std) < expected_std * 0.05

    samples = [None] * dist.get_world_size()
    dist.all_gather_object(samples, actual.flatten()[:64].tolist())
    if isinstance(placement, Replicate):
        assert all(values == samples[0] for values in samples)
    else:
        assert len({tuple(values) for values in samples}) == len(samples)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [16, 17, 154881])
@pytest.mark.parametrize("all_ignored", [False, True], ids=["mixed_labels", "all_ignored"])
@pytest.mark.parametrize("reduction", ["sum", "none"])
def test_native_vocab_parallel_loss(tp_mesh, dtype, vocab_size, all_ignored, reduction):
    """Compare distributed CE and local logits gradients to a full CPU reference."""
    group = tp_mesh.get_group("tp")
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    chunk = (vocab_size + world_size - 1) // world_size
    start, end = rank * chunk, min(vocab_size, (rank + 1) * chunk)
    tokens = 16
    generator = torch.Generator().manual_seed(101 + vocab_size)
    fixture = (torch.randn(tokens, vocab_size, generator=generator) * 2).to(dtype)
    labels = torch.tensor([-100, 0, chunk - 1, chunk, vocab_size - 1, 1, vocab_size - 2, -100] * 2)
    if all_ignored:
        labels.fill_(-100)

    reference_logits = fixture.float().detach().requires_grad_()
    reference = torch.nn.functional.cross_entropy(reference_logits, labels, ignore_index=-100, reduction=reduction)
    upstream = torch.linspace(0.5, 1.5, tokens) if reduction == "none" else torch.tensor(0.75)
    reference.backward(upstream)

    device = torch.device(f"npu:{int(os.environ['LOCAL_RANK'])}")
    logits = fixture[:, start:end].contiguous().to(device).requires_grad_()
    actual = _LossParallelCrossEntropy.apply(logits, labels.to(device), group, vocab_size, reduction)
    torch.testing.assert_close(actual.detach().cpu(), reference.detach(), atol=1e-4, rtol=1e-5)
    actual.backward(upstream.to(device))
    gradient = logits.grad.float().cpu()
    expected = reference_logits.grad[:, start:end].to(dtype).float()
    torch.testing.assert_close(
        gradient,
        expected,
        atol=1e-6 if dtype == torch.float32 else 1e-3,
        rtol=1e-5 if dtype == torch.float32 else 1e-2,
    )
    assert torch.count_nonzero(gradient[labels == -100]).item() == 0
