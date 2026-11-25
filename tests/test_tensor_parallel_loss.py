# SPDX-License-Identifier: Apache-2.0
r"""Vocab-parallel CE forward/backward regression on 2 or 4 NPUs.

Run with:
    TORCHINDUCTOR_NPU_BACKEND=triton_experimental torchrun --nproc-per-node=2 \
        -m pytest -q tests/test_tensor_parallel_loss.py
"""

import math
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch._inductor import config as inductor_config
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.components.loss import cross_entropy_loss
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.distributed.spmd_types import set_current_spmd_mesh

pytestmark = pytest.mark.npu

TOKENS = 64
IGNORE_INDEX = -100


def _buffer_reuse_enabled():
    return bool(inductor_config.allow_buffer_reuse)


@pytest.fixture(scope="module")
def tp_runtime():
    """Own a TP process group without changing the registered global SPMD meshes."""
    if int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4):
        pytest.skip("requires torchrun --nproc-per-node=2 or 4")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPUs and torch_npu")
    assert (
        os.environ.get("TORCHINDUCTOR_NPU_BACKEND") == "triton_experimental"
    ), "set TORCHINDUCTOR_NPU_BACKEND=triton_experimental to test the affected backend"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    created_default_group = not dist.is_initialized()
    if created_default_group:
        dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    group = None
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        assert world_size in (2, 4)
        group = dist.new_group(
            ranks=list(range(world_size)),
            backend="hccl",
            timeout=timedelta(seconds=180),
        )
        mesh = DeviceMesh.from_group(group, "npu", mesh_dim_names=("tp",))
        assert _buffer_reuse_enabled(), "regression requires default buffer reuse"
        yield rank, world_size, torch.device("npu", local_rank), mesh
    finally:
        apply_local_compile([])
        if group is not None:
            dist.destroy_process_group(group)
        if created_default_group:
            dist.destroy_process_group()


def _inputs(vocab_size, pattern, all_ignored, world_size, dtype):
    if pattern == "zeros":
        logits = torch.zeros(TOKENS, vocab_size, dtype=dtype)
    else:
        logits = (
            torch.linspace(-1, 1, vocab_size).expand(TOKENS, -1)
            + torch.arange(TOKENS)[:, None] * 0.001
        ).to(dtype)
    labels = (torch.arange(TOKENS, dtype=torch.int64) * 37) % vocab_size
    labels[[0, 3, 7]] = IGNORE_INDEX
    chunk_size = (vocab_size + world_size - 1) // world_size
    # Include both sides of every shard boundary, including the last class.
    for owner in range(world_size):
        start = owner * chunk_size
        end = min(start + chunk_size, vocab_size)
        labels[16 + 2 * owner] = start
        labels[17 + 2 * owner] = end - 1
    if all_ignored:
        labels.fill_(IGNORE_INDEX)
    return logits, labels


def _assert_uniform_bf16_gradient(gradient, labels, vocab_size, start, end):
    # For V=2048, 1/V is exactly representable in BF16. Build the target
    # component in FP32, then round to the original BF16 leaf's gradient dtype.
    expected = torch.full((TOKENS, end - start), 1.0 / vocab_size, dtype=torch.float32)
    owned_rows = torch.nonzero((labels >= start) & (labels < end), as_tuple=True)[0]
    expected[owned_rows, labels[owned_rows] - start] -= 1.0
    expected[labels == IGNORE_INDEX] = 0
    torch.testing.assert_close(gradient, expected.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "vocab_size,pattern,reduction,all_ignored",
    [
        pytest.param(2048, "zeros", "sum", False, id="uniform-sum"),
        pytest.param(2048, "ramp", "sum", False, id="ramp-sum"),
        pytest.param(2049, "ramp", "none", False, id="uneven-weighted-none"),
        pytest.param(2048, "ramp", "sum", True, id="ignored-sum"),
        pytest.param(2048, "ramp", "none", True, id="ignored-none"),
    ],
)
def test_tensor_parallel_cross_entropy(
    tp_runtime, dtype, vocab_size, pattern, reduction, all_ignored, record_property
):
    """Compare production eager/compiled CE with the identical dense CPU inputs."""
    rank, world_size, device, mesh = tp_runtime
    logits, labels = _inputs(vocab_size, pattern, all_ignored, world_size, dtype)
    chunk_size = (vocab_size + world_size - 1) // world_size
    start, end = rank * chunk_size, min((rank + 1) * chunk_size, vocab_size)
    reference_logits = logits.clone().requires_grad_(True)
    reference = functional.cross_entropy(
        reference_logits.float(), labels, reduction=reduction, ignore_index=IGNORE_INDEX
    )
    cotangent = None
    if reduction == "none":
        # Exact dyadic values exercise zero, negative and unequal upstream grads.
        cotangent = torch.tensor([0.0, 0.25, -0.5, 1.0, 2.0, -1.0])[
            torch.arange(TOKENS) % 6
        ]
    reference.backward(cotangent)
    assert reference_logits.grad is not None
    expected_gradient = reference_logits.grad[:, start:end]
    record_property("tp_world_size", world_size)
    record_property("tp_rank", rank)
    record_property(
        "npu_backend", os.environ.get("TORCHINDUCTOR_NPU_BACKEND", "default")
    )

    # Six dtype/shape/reduction specializations per rank, below the default
    # Dynamo limit; labels/patterns vary only tensor values, not graph structure.
    results = []
    try:
        for compiled in (False, True):
            apply_local_compile(["loss"] if compiled else [])
            local = logits[:, start:end].contiguous().to(device).requires_grad_(True)
            assert _buffer_reuse_enabled()
            with set_current_spmd_mesh(mesh):
                observed = cross_entropy_loss(
                    local,
                    labels.to(device),
                    global_vocab_size=vocab_size,
                    reduction=reduction,
                )
                observed.backward(None if cotangent is None else cotangent.to(device))
            torch.npu.synchronize()
            # Check the actual setting after compilation/backend activation too.
            assert _buffer_reuse_enabled()
            record_property(
                "compiled_buffer_reuse" if compiled else "eager_buffer_reuse", True
            )
            actual = observed.detach().cpu()
            assert local.grad is not None
            gradient = local.grad.cpu()
            results.append((compiled, actual, gradient))
        # Finish both collective sequences before numerical assertions, so a
        # rank-specific eager mismatch cannot skip the compiled collectives.
        for compiled, actual, gradient in results:
            assert torch.isfinite(
                actual
            ).all(), f"non-finite loss with compiled={compiled}"
            assert torch.isfinite(gradient).all()
            if reduction == "sum":
                # Preserve the original probe's scalar loss gate exactly.
                assert math.isclose(
                    actual.item(), reference.item(), rel_tol=1e-5, abs_tol=1e-3
                )
            else:
                torch.testing.assert_close(
                    actual, reference.detach(), rtol=1e-5, atol=1e-6
                )
            torch.testing.assert_close(
                gradient.float(),
                expected_gradient.float(),
                rtol=1e-5 if dtype == torch.float32 else 1e-3,
                atol=1e-6 if dtype == torch.float32 else 1e-3,
            )
            assert torch.count_nonzero(gradient[labels == IGNORE_INDEX]).item() == 0
            if cotangent is not None:
                assert torch.count_nonzero(gradient[cotangent == 0]).item() == 0
            if all_ignored:
                assert torch.count_nonzero(actual).item() == 0
                assert torch.count_nonzero(gradient).item() == 0
            if pattern == "zeros":
                valid_tokens = torch.count_nonzero(labels != IGNORE_INDEX).item()
                assert math.isclose(
                    actual.item() / valid_tokens,
                    math.log(vocab_size),
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                )
                if dtype == torch.bfloat16:
                    _assert_uniform_bf16_gradient(
                        gradient, labels, vocab_size, start, end
                    )
    finally:
        apply_local_compile([])
