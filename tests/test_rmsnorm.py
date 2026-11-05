# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; these tests do not establish NPU accuracy."""

import torch
import torch.nn.functional as F

import torchtitan_ascend.overrides.rmsnorm as MODULE


def test_rms_public_api(monkeypatch):
    import torch_npu

    calls = []

    def fake(x, weight, *, epsilon):
        calls.append(epsilon)
        return F.rms_norm(x, (x.shape[-1],), weight, epsilon), None

    monkeypatch.setattr(torch_npu, "npu_rms_norm", fake)
    x = torch.randn(2, 9, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, dtype=torch.float64, requires_grad=True)
    y = MODULE.rmsnorm(x, weight, 3e-4)
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
        MODULE.ascendc_rmsnorm(MODULE.RMSNorm.Config(normalized_shape=9, eps=3e-4))
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
