# SPDX-License-Identifier: Apache-2.0
"""Validate upstream ComplexRoPE on the A5 path without an Ascend override."""

import pytest
import torch

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan_ascend.device import get_device_name


@pytest.fixture
def npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)
    if not get_device_name().startswith("Ascend950"):
        pytest.skip("upstream ComplexRoPE regression coverage targets A5")


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("positions_kind", ["slice", "reset", "random", "boundary"])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("with_key", [False, True])
def test_upstream_rope_forward_backward(npu, dtype, positions_kind, inverse, with_key):
    torch.manual_seed(61)
    config = ComplexRoPE.Config(dim=64, max_context_length=8192, theta=8_000_000.0)
    reference = ComplexRoPE(config)
    # Share the CPU-generated cache to isolate RoPE from CPU/NPU cache-init rounding.
    actual = ComplexRoPE(config).to("npu")
    query = torch.randn(128, 32, 64, dtype=dtype, requires_grad=True)
    key = torch.randn(128, 1, 64, dtype=dtype, requires_grad=True) if with_key else None
    positions = {
        "slice": None,
        "reset": torch.arange(128) % 32,
        "random": torch.randint(config.max_context_length, (128,)),
        "boundary": torch.full((128,), config.max_context_length - 1, dtype=torch.int64),
    }[positions_kind]
    inputs = (query,) if key is None else (query, key)
    gradients = tuple(torch.randn_like(value) for value in inputs)

    expected = reference(query, key, positions, inverse=inverse)
    expected = expected if isinstance(expected, tuple) else (expected,)
    expected_gradients = torch.autograd.grad(expected, inputs, gradients)
    npu_inputs = tuple(value.detach().to("npu").requires_grad_() for value in inputs)
    npu_positions = None if positions is None else positions.to("npu")
    observed = actual(npu_inputs[0], npu_inputs[1] if with_key else None, npu_positions, inverse=inverse)
    observed = observed if isinstance(observed, tuple) else (observed,)
    observed_gradients = torch.autograd.grad(observed, npu_inputs, tuple(value.to("npu") for value in gradients))

    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    for got, want in zip((*observed, *observed_gradients), (*expected, *expected_gradients), strict=True):
        assert got.device.type == "npu"
        torch.testing.assert_close(got.detach().cpu(), want, atol=tolerance, rtol=tolerance)
