# SPDX-License-Identifier: Apache-2.0
"""Basic EP forward/backward smoke tests; launch with torchrun."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.distributed.spmd_types import set_spmd_meshes
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher

from torchtitan_ascend.overrides.common.moe.mc2_routed_experts import (
    npu_mc2_routed_experts,
)
from torchtitan_ascend.overrides.common.moe.token_dispatcher import npu_token_dispatcher

pytestmark = pytest.mark.npu

TOKENS = 64
DIM = 128
HIDDEN = 256
TOP_K = 2
EXPERTS_PER_RANK = 4


@pytest.fixture(scope="module")
def ep_runtime():
    """Set up the EP group."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun --nproc-per-node=2 (or 8)")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPUs and torch_npu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    created_process_group = not dist.is_initialized()
    if created_process_group:
        dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    world_size, rank = dist.get_world_size(), dist.get_rank()
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("ep",))
    # The classes obtain their EP group from this runtime registration.
    set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
    try:
        yield rank, world_size, torch.device("npu", local_rank)
    finally:
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=None, dense_sp_enabled=False)
        if created_process_group:
            dist.destroy_process_group()


def _inputs(rank, world_size, device, dtype):
    generator = torch.Generator().manual_seed(42 + rank)
    x = (
        torch.randn(TOKENS, DIM, generator=generator)
        .to(device=device, dtype=dtype)
        .requires_grad_()
    )
    scores = (
        torch.rand(TOKENS, TOP_K, generator=generator)
        .softmax(-1)
        .to(device)
        .requires_grad_()
    )
    ids = (torch.arange(TOKENS * TOP_K).reshape(TOKENS, TOP_K) + rank) % (
        world_size * EXPERTS_PER_RANK
    )
    counts = torch.bincount(ids.flatten(), minlength=world_size * EXPERTS_PER_RANK).to(
        device
    )
    return x, scores, ids.to(device=device, dtype=torch.int32), counts


def _assert_gradient(tensor):
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(tensor.grad) > 0


def test_alltoall_dispatcher_ep(ep_runtime):
    """Dispatch then combine unchanged expert rows to recover the input."""
    rank, world_size, device = ep_runtime
    dispatcher = npu_token_dispatcher(
        AllToAllTokenDispatcher.Config(
            num_experts=world_size * EXPERTS_PER_RANK, top_k=TOP_K
        )
    ).build()
    x, scores, ids, counts = _inputs(rank, world_size, device, torch.float32)
    routed, expert_counts, metadata = dispatcher.dispatch(x, scores, ids, counts)
    output = dispatcher.combine(routed, metadata, x)
    output.square().mean().backward()

    assert routed.shape[1] == DIM
    assert expert_counts.numel() == EXPERTS_PER_RANK
    assert expert_counts.sum().item() == routed.shape[0]
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-6)
    _assert_gradient(x)
    _assert_gradient(scores)


def _build_mc2_model(rank, world_size, device, *, recompute_routed_input=False):
    experts = world_size * EXPERTS_PER_RANK
    model = npu_mc2_routed_experts(
        RoutedExperts.Config(
            w13=GroupedLinear.Config(
                group_size=experts, in_features=DIM, out_features=HIDDEN, num_linears=2
            ),
            w2=GroupedLinear.Config(
                group_size=experts, in_features=HIDDEN, out_features=DIM
            ),
            token_dispatcher=AllToAllTokenDispatcher.Config(
                num_experts=experts, top_k=TOP_K
            ),
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        ),
        recompute_routed_input=recompute_routed_input,
    ).build()
    generator = torch.Generator().manual_seed(100 + rank)
    # Place only this rank's expert weights on its NPU, without FSDP setup.
    model.w13.weight = torch.nn.Parameter(
        (torch.randn(EXPERTS_PER_RANK, 2, HIDDEN, DIM, generator=generator) * 0.02).to(
            device
        )
    )
    model.w2.weight = torch.nn.Parameter(
        (torch.randn(EXPERTS_PER_RANK, DIM, HIDDEN, generator=generator) * 0.02).to(
            device
        )
    )
    model.w13.group_size = model.w2.group_size = EXPERTS_PER_RANK
    return model


def test_mc2_routed_experts_ep(ep_runtime):
    """Run fused dispatch, expert computation and combine, then backward."""
    rank, world_size, device = ep_runtime
    model = _build_mc2_model(rank, world_size, device)
    x, scores, ids, counts = _inputs(rank, world_size, device, torch.bfloat16)
    output = model(x, scores, ids, counts)
    output.float().square().mean().backward()

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    for tensor in (x, scores, model.w13.weight, model.w2.weight):
        _assert_gradient(tensor)


def test_mc2_routed_input_replay_ep(ep_runtime):
    """Compare forward and gradients under the same saved-dispatch policy."""
    rank, world_size, device = ep_runtime
    num_experts = world_size * EXPERTS_PER_RANK
    results = []
    for replay in (False, True):
        model = _build_mc2_model(
            rank, world_size, device, recompute_routed_input=replay
        )
        model.configure_remat_regions(
            ["token_dispatcher.dispatch", "token_dispatcher.combine"]
        )
        x, scores, ids, _ = _inputs(rank, world_size, device, torch.bfloat16)
        ids[::3, 0] = (ids[::3, 0] + 1) % num_experts
        counts = torch.bincount(ids.flatten().long(), minlength=num_experts)
        checkpointed = remat.checkpoint(region_name="experts")(model)
        output = checkpointed(x, scores, ids, counts)
        # The training runtime keeps recomputation on the SPMD mesh's thread.
        with torch.autograd.set_multithreading_enabled(False):
            output.float().square().mean().backward()
        grads = []
        for tensor in (x, scores, model.w13.weight, model.w2.weight):
            _assert_gradient(tensor)
            grads.append(tensor.grad)
        results.append((output.detach(), *grads))

    saved, replayed = results
    for candidate, reference in zip(replayed, saved, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=0.015625, atol=3e-4)


def test_deepep_dispatcher_ep(ep_runtime):
    """Dispatch and combine through the real DeepEP-Ascend EPBuffer."""
    from importlib.util import find_spec

    if find_spec("deep_ep") is None:
        pytest.skip("requires DeepEP-Ascend (deep_ep)")
    from torchtitan_ascend.device import get_device_name

    if not get_device_name().startswith("Ascend950"):
        pytest.skip("DeepEP-Ascend requires Ascend 950")

    from torchtitan_ascend.overrides.deepep import npu_deepep
    from torchtitan_ascend.overrides.deepep.runtime import destroy_all_buffers

    rank, world_size, device = ep_runtime
    dispatcher = npu_deepep(
        AllToAllTokenDispatcher.Config(
            num_experts=world_size * EXPERTS_PER_RANK, top_k=TOP_K
        ),
        num_max_tokens_per_rank=TOKENS,
        hidden_dim=DIM,
    ).build()
    dispatcher.init_buffer()
    x, scores, ids, counts = _inputs(rank, world_size, device, torch.bfloat16)
    try:
        routed, expert_counts, metadata = dispatcher.dispatch(
            x, scores, ids.to(torch.int64), counts
        )
        output = dispatcher.combine(routed, metadata, x)
        output.square().mean().backward()

        assert routed.shape[1] == DIM
        assert expert_counts.sum().item() == routed.shape[0]
        assert output.shape == x.shape
        torch.testing.assert_close(output, x, rtol=2e-2, atol=2e-2)
        _assert_gradient(x)
        _assert_gradient(scores)
    finally:
        torch.npu.synchronize()
        destroy_all_buffers()


@pytest.mark.parametrize(
    "saved_regions",
    [
        (),
        ("token_dispatcher.dispatch",),
        ("token_dispatcher.combine",),
        ("token_dispatcher.dispatch", "token_dispatcher.combine"),
    ],
    ids=["recompute-both", "save-dispatch", "save-combine", "save-both"],
)
@pytest.mark.parametrize(
    "recompute_routed_input", [False, True], ids=["store", "replay"]
)
def test_mc2_combine_remat_ep(ep_runtime, saved_regions, recompute_routed_input):
    """Compare mixed-dtype MC2 output and all input/weight gradients with eager."""
    rank, world_size, device = ep_runtime
    from torchtitan_ascend.device import get_device_name

    if not get_device_name().startswith("Ascend950"):
        pytest.skip("requires Ascend 950 mixed-dtype unpermute")
    config = make_routed_experts_config(
        dim=DIM,
        hidden_dim=HIDDEN,
        num_experts=world_size * EXPERTS_PER_RANK,
        top_k=TOP_K,
        param_init={},
        enable_ep=True,
    )
    config.activation_fn = SiTUGLU.Config(beta=4.0, linear_beta=25.0)
    model = npu_mc2_routed_experts(
        config, recompute_routed_input=recompute_routed_input
    ).build()
    generator = torch.Generator().manual_seed(100 + rank)
    for linear in (model.w13, model.w2):
        shape = (EXPERTS_PER_RANK, *linear.weight.shape[1:])
        linear.weight = torch.nn.Parameter(
            (torch.randn(shape, generator=generator) * 0.02).to(device)
        )
        linear.group_size = EXPERTS_PER_RANK
    assert model.allow_mixed_dtype
    model.configure_remat_regions(saved_regions)
    x, scores, ids, counts = _inputs(rank, world_size, device, torch.bfloat16)

    def run(forward):
        output = forward(x, scores, ids, counts)
        # The EP mesh used in recomputation is registered on this thread.
        with torch.autograd.set_multithreading_enabled(False):
            grads = torch.autograd.grad(
                output, (x, scores, model.w13.weight, model.w2.weight), x.detach()
            )
        return (output.detach(), *grads)

    eager = run(model)
    checkpointed = run(remat.checkpoint(region_name="experts")(model))
    torch.testing.assert_close(checkpointed, eager, rtol=0.015625, atol=3e-4)
