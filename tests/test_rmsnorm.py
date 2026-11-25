# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; these tests do not establish NPU accuracy."""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common.activation import Sigmoid, SiLU

import torchtitan_ascend.overrides.common.norm as MODULE


def test_rms_public_api(monkeypatch):
    import torch_npu

    calls = []

    def fake(x, weight, *, epsilon):
        calls.append(epsilon)
        return F.rms_norm(x, (x.shape[-1],), weight, epsilon), None

    monkeypatch.setattr(torch_npu, "npu_rms_norm", fake)
    x = torch.randn(2, 9, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, dtype=torch.float64, requires_grad=True)
    y = MODULE.rmsnorm_forward(x, weight, 3e-4)
    assert torch.equal(y, F.rms_norm(x, (9,), weight, 3e-4))
    assert calls == [3e-4]
    assert all(
        torch.isfinite(v).all() for v in torch.autograd.grad(y.sum(), (x, weight))
    )


def test_rms_fp32_input_retains_bf16_parameter_gradient(monkeypatch):
    import torch_npu

    def fake(x, weight, *, epsilon):
        assert x.dtype == weight.dtype == torch.float32
        return F.rms_norm(x, (x.shape[-1],), weight, epsilon), None

    monkeypatch.setattr(torch_npu, "npu_rms_norm", fake)
    model = (
        MODULE.npu_rmsnorm(MODULE.RMSNorm.Config(normalized_shape=9, eps=3e-4))
        .build()
        .bfloat16()
    )
    weight = model.weight
    x = torch.randn(2, 9, requires_grad=True)
    actual = model(input=x)
    expected = F.rms_norm(x, (9,), weight.float(), 3e-4)
    assert torch.equal(actual, expected)
    assert model.weight is weight and weight.dtype == torch.bfloat16
    for got, want in zip(
        torch.autograd.grad(actual.sum(), (x, weight)),
        torch.autograd.grad(expected.sum(), (x, weight)),
        strict=True,
    ):
        assert torch.equal(got, want)


@pytest.mark.parametrize("eps", [None, 3e-4])
def test_rms_without_affine_uses_npu_operator(monkeypatch, eps):
    import torch_npu

    calls = []

    def fake(x, weight, *, epsilon):
        calls.append(epsilon)
        assert weight.shape == (9,)
        assert weight.dtype == x.dtype and weight.device == x.device
        assert torch.equal(weight, torch.ones_like(weight))
        return F.rms_norm(x, (x.shape[-1],), weight, epsilon), None

    monkeypatch.setattr(torch_npu, "npu_rms_norm", fake)
    model = MODULE.npu_rmsnorm(
        MODULE.RMSNorm.Config(normalized_shape=9, elementwise_affine=False)
    ).build()
    model.eps = eps
    x = torch.randn(2, 9, requires_grad=True)
    actual = model(input=x)
    expected = F.rms_norm(x, (9,), None, eps)
    assert model.weight is None and not dict(model.named_parameters())
    assert calls == [eps if eps is not None else torch.finfo(x.dtype).eps]
    assert torch.equal(actual, expected)
    dy = torch.randn_like(actual)
    assert torch.equal(
        torch.autograd.grad(actual, x, dy)[0],
        torch.autograd.grad(expected, x, dy)[0],
    )


def test_rms_without_affine_rejects_shape_mismatch(monkeypatch):
    import torch_npu

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input must be rejected before calling the NPU operator")

    monkeypatch.setattr(torch_npu, "npu_rms_norm", unexpected_call)
    model = MODULE.npu_rmsnorm(
        MODULE.RMSNorm.Config(normalized_shape=9, elementwise_affine=False)
    ).build()
    with pytest.raises(ValueError, match="Expected trailing dimensions"):
        model(input=torch.randn(2, 12))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [Sigmoid, SiLU])
def test_gated_rms_forward_and_gradients(monkeypatch, dtype, activation):
    import torch_npu

    calls = []

    def fake(x, weight, *, epsilon):
        calls.append(epsilon)
        assert x.dtype == weight.dtype == torch.float32
        return F.rms_norm(x, (x.shape[-1],), weight, epsilon), None

    monkeypatch.setattr(torch_npu, "npu_rms_norm", fake)
    config = MODULE.GatedRMSNorm.Config(
        dim=9, eps=3e-4, activation_fn=activation.Config()
    )
    model = MODULE.npu_gated_rmsnorm(config).build().to(dtype=dtype)
    torch.nn.init.normal_(model.weight)
    weight = model.weight
    x = torch.randn(2, 9, dtype=dtype, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    actual = model(input=x, gate=gate)
    normalized = F.rms_norm(x.float(), (9,), weight.float(), config.eps)
    gate_fn = torch.sigmoid if activation is Sigmoid else F.silu
    expected = (normalized * gate_fn(gate.float())).to(dtype)
    assert calls == [config.eps]
    assert actual.dtype == dtype
    assert model.weight is weight and weight.dtype == dtype
    assert torch.equal(actual, expected)
    dy = torch.randn_like(actual)
    for got, want in zip(
        torch.autograd.grad(actual, (x, gate, weight), dy),
        torch.autograd.grad(expected, (x, gate, weight), dy),
        strict=True,
    ):
        assert torch.equal(got, want)
