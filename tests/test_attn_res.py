# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; these tests do not establish NPU accuracy."""

import pytest
import torch

import torchtitan_ascend.overrides.attn_res as MODULE
from torch.testing import assert_close


@pytest.mark.parametrize("partial", [False, True])
def test_attn_res_packing_and_gradients(monkeypatch, partial):
    from torchtitan.models.common.nn_modules import RMSNorm

    norm = RMSNorm(RMSNorm.Config(normalized_shape=7, eps=3e-4)).double()
    proj = torch.nn.Linear(7, 1, bias=False).double()
    blocks = torch.randn(4, 3, 7, dtype=torch.float64, requires_grad=True)
    prefix = (
        torch.randn(4, 7, dtype=torch.float64, requires_grad=True) if partial else None
    )

    def kernel(values, gamma, project, eps):
        values_float = values.float()
        keys = values_float * torch.rsqrt(
            values_float.square().mean(-1, keepdim=True) + eps
        )
        scores = (keys * (gamma.float() * project.float())).sum(-1)
        return (
            (scores.softmax(-1).unsqueeze(1) @ values_float).squeeze(1).to(values.dtype)
        )

    monkeypatch.setattr(MODULE._AttentionResidual, "apply", kernel)
    a = MODULE.TritonAttentionResidual.Config().build()(prefix, blocks, proj, norm)
    b = MODULE._eager_attention_residual(prefix, blocks, proj, norm)
    assert torch.equal(a, b)
    inputs = [blocks, norm.weight, proj.weight] + ([prefix] if partial else [])
    ga = torch.autograd.grad(a.sum(), inputs, retain_graph=True)
    gb = torch.autograd.grad(b.sum(), inputs)
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_attn_res_override_preserves_config():
    source = MODULE.AttentionResidual.Config()
    replacement = MODULE.triton_attn_res(source)
    assert isinstance(replacement, MODULE.TritonAttentionResidual.Config)
    assert MODULE.enable_triton_attn_res().endswith(".triton_attn_res")


@pytest.mark.npu
@pytest.mark.parametrize("dim", [256, 7168])
@pytest.mark.parametrize("partial", [False, True])
def test_attn_res_npu_matches_reference(dim, partial):
    from torchtitan.models.common.nn_modules import RMSNorm

    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(42)
    norm = RMSNorm.Config(normalized_shape=dim, eps=1e-5).build().to("npu")
    proj = torch.nn.Linear(dim, 1, bias=False).to("npu")
    with torch.no_grad():
        proj.weight.copy_(torch.randn(1, dim, generator=generator) / dim**0.5)
    blocks = torch.randn(16, 3, dim, generator=generator).to("npu").requires_grad_()
    prefix = (
        torch.randn(16, dim, generator=generator).to("npu").requires_grad_()
        if partial
        else None
    )
    actual = MODULE.attention_residual(prefix, blocks, proj, norm)
    inputs = [blocks, norm.weight, proj.weight] + ([prefix] if partial else [])
    # A CPU FP64 oracle avoids NPU matmul accumulation error in projection gradients.
    reference_inputs = [
        tensor.detach().cpu().double().requires_grad_() for tensor in inputs
    ]
    values, gamma, project = reference_inputs[:3]
    if partial:
        values = torch.cat((values, reference_inputs[3].unsqueeze(1)), dim=1)
    keys = values * (values.square().mean(-1, keepdim=True) + norm.eps).rsqrt()
    scores = (keys * (gamma * project.squeeze(0))).sum(-1)
    expected = (scores.softmax(-1).unsqueeze(-1) * values).sum(-2)
    assert_close(actual.detach().cpu().double(), expected, rtol=5e-5, atol=5e-5)
    dy = torch.randn(actual.shape, generator=generator).to("npu")
    for index, (got, want) in enumerate(
        zip(
            torch.autograd.grad(actual, inputs, dy),
            torch.autograd.grad(expected, reference_inputs, dy.cpu().double()),
            strict=True,
        )
    ):
        got = got.cpu().double()
        if index == 2:
            # Projection dot products can cancel near zero even in CPU FP32.
            # Bound both total and worst-entry error relative to the gradient scale.
            error = got - want
            assert torch.isfinite(got).all()
            assert error.norm() <= 5e-5 * want.norm().clamp_min(1.0)
            assert error.abs().max() <= 5e-5 * want.abs().max().clamp_min(1.0)
        else:
            assert_close(got, want, rtol=5e-5, atol=5e-5)


def test_recipe_registers_residual_override_without_rebinding(monkeypatch):
    from torchtitan.models.kimi_k2_7 import vision_encoder
    from torchtitan_ascend.recipes.kimi_k3 import kimi_k3_debugmodel_npu

    monkeypatch.setattr(vision_encoder, "ComplexRoPE", vision_encoder.ComplexRoPE)
    original = MODULE.AttentionResidual.__call__
    config = kimi_k3_debugmodel_npu()
    assert MODULE.enable_triton_attn_res() in config.override.imports
    assert MODULE.AttentionResidual.__call__ is original
