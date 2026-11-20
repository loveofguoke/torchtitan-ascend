# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_remat as remat
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.models.common.config_utils import make_routed_experts_config


@pytest.fixture(scope="module")
def mc2_ops():
    package = importlib.import_module("torchtitan_ascend.overrides.common.moe")
    registry = importlib.import_module("torchtitan.config.override")._REGISTRY
    backend = ModuleType("torch_npu")
    with pytest.MonkeyPatch.context() as mp, patch.dict(registry):
        # Import a CPU-only copy of the overrides, then restore their import and
        # registration state for the other modules in the same pytest batch.
        mp.setitem(sys.modules, "torch_npu", backend)
        for name in ("mc2_gmm", "permute", "mc2_routed_experts"):
            qualified = f"{package.__name__}.{name}"
            mp.setitem(sys.modules, qualified, None)
            del sys.modules[qualified]
            mp.setattr(package, name, None, raising=False)
        module = f"{package.__name__}.mc2_routed_experts"
        target = f"{module}.npu_mc2_routed_experts"
        registry.pop(target, None)
        yield importlib.import_module(module), backend


@pytest.mark.parametrize("recompute_routed_input", [False, True])
@pytest.mark.parametrize("recompute_dispatch", [False, True])
@pytest.mark.parametrize("recompute_combine", [False, True])
def test_mc2_combine_region(
    mc2_ops, monkeypatch, recompute_dispatch, recompute_combine, recompute_routed_input
):
    mc2, backend = mc2_ops
    monkeypatch.setattr(DefaultDeviceType, "get_device_type", lambda: "cpu")
    mesh = SimpleNamespace(size=lambda: 2, get_group=object)
    monkeypatch.setattr(mc2, "spmd_sparse_mesh", lambda: {"ep": mesh})
    monkeypatch.setattr(mc2, "maybe_set_sparse_mesh", nullcontext)

    events = []
    inverse = torch.tensor([2, 1, 0, 3])
    counts = torch.tensor([2, 2])

    def dispatch(x, ids, weight, local_counts, *, replay_routed_input):
        assert replay_routed_input == (
            recompute_routed_input and not recompute_dispatch
        )
        events.append("dispatch")
        routed = x.repeat_interleave(2, dim=0).index_select(0, inverse)
        return routed @ weight, inverse, counts, counts, counts

    def return_routed_tokens(*, inputs, weights, group, send_counts, recv_counts):
        events.append("combine")
        return inputs @ weights.T

    def restore(tokens, indices, probs, *, restore_shape):
        events.append("unpermute")
        assert tokens.dtype == torch.bfloat16 and probs.dtype == torch.float32
        weighted = tokens.index_select(0, indices).view(2, 2, 2).float()
        return (weighted * probs.unsqueeze(-1)).sum(1).to(tokens.dtype)

    monkeypatch.setattr(backend, "npu_moe_token_unpermute", restore, raising=False)
    x = torch.tensor([[1, 2], [3, -1]], dtype=torch.bfloat16, requires_grad=True)
    w13 = torch.nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
    w2 = torch.nn.Parameter(torch.tensor([[1, 2], [0, 1]], dtype=torch.bfloat16))
    scores = torch.tensor([[0.25, 0.75], [0.5, 0.25]], requires_grad=True)
    monkeypatch.setattr(mc2, "get_device_name", lambda: "Ascend950DT")
    config = make_routed_experts_config(
        dim=2, hidden_dim=2, num_experts=2, top_k=2, param_init={}, enable_ep=True
    )
    experts = mc2.npu_mc2_routed_experts(
        config, recompute_routed_input=recompute_routed_input
    ).build()
    experts.w13.weight, experts.w2.weight = w13, w2
    monkeypatch.setattr(experts, "_dispatch_w13", dispatch)
    monkeypatch.setattr(mc2, "grouped_matmul_all2all", return_routed_tokens)
    monkeypatch.setattr(experts, "activation_fn", lambda x, *, offsets: x.square())
    saved_regions = []
    if not recompute_dispatch:
        saved_regions.append("token_dispatcher.dispatch")
    if not recompute_combine:
        saved_regions.append("token_dispatcher.combine")
    experts.configure_remat_regions(saved_regions)
    ids = torch.tensor([[0, 1], [1, 0]])
    reference = (((x @ w13).square() @ w2).float() * scores.sum(1, keepdim=True)).to(x)
    output = remat.checkpoint(region_name="layer")(experts)(x, scores, ids, counts)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    parameters = (x, scores, w13, w2)
    grad = torch.tensor([[1, 2], [3, 4]], dtype=x.dtype)
    expected_grads = torch.autograd.grad(reference, parameters, grad)
    actual_grads = torch.autograd.grad(output, parameters, grad)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert events.count("dispatch") == 1 + recompute_dispatch
    assert events.count("combine") == 1 + recompute_combine
    assert events.count("unpermute") == 1 + recompute_combine


@pytest.mark.parametrize(
    ("token_dtype", "probs_dtype", "enabled", "expected_dtype"),
    [
        (torch.bfloat16, torch.float32, True, torch.bfloat16),
        (torch.bfloat16, torch.float32, False, torch.float32),
        (torch.float16, torch.float32, True, torch.float16),
        (torch.bfloat16, torch.float16, True, torch.bfloat16),
        (torch.float32, torch.bfloat16, False, torch.bfloat16),
    ],
)
def test_weighted_unpermute_input_dtype(
    mc2_ops, monkeypatch, token_dtype, probs_dtype, enabled, expected_dtype
):
    mc2, backend = mc2_ops

    def restore(tokens, indices, probs, *, restore_shape):
        assert tokens.dtype == expected_dtype
        assert probs.dtype == probs_dtype
        return tokens

    monkeypatch.setattr(backend, "npu_moe_token_unpermute", restore, raising=False)
    tokens = torch.ones(2, 1, dtype=token_dtype)
    output = mc2.unpermute(
        tokens,
        torch.tensor([1, 0]),
        torch.ones(2, 1, dtype=probs_dtype),
        allow_mixed_dtype=enabled,
    )
    assert output.dtype == token_dtype
