# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the vision-only FP32 interleaved RoPE route."""

import pytest
import torch
import torchtitan_ascend.overrides.rotary_mul as MODULE
from torch.testing import assert_close

from torchtitan.models.common.rope import ComplexRoPE


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_interleaved_rope_and_gradients(monkeypatch, dtype):
    import torch_npu

    calls = []

    def fake(x, cos, sin, *, rotary_mode):
        calls.append((x.dtype, cos, sin, rotary_mode))
        even, odd = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    monkeypatch.setattr(torch_npu, "npu_rotary_mul", fake)
    torch.manual_seed(42)
    # Dyadic values make real and complex CPU multiplication exactly comparable.
    cache = torch.complex(torch.randint(-8, 9, (7, 1, 4)) / 8, torch.randint(-8, 9, (7, 1, 4)) / 8)
    inputs = [(torch.randint(-16, 17, (7, heads, 8)).to(dtype) / 16).requires_grad_() for heads in (4, 2)]
    references = [x.detach().clone().requires_grad_() for x in inputs]
    actual = MODULE.rotary_mul(*inputs, cache)
    expected = ComplexRoPE.apply_rotary_emb(*references, cache)
    for a, b in zip(actual, expected):
        assert_close(a, b, rtol=0, atol=0)
    grads = [torch.randint(-16, 17, x.shape).to(dtype) / 16 for x in actual]
    for a, b in zip(
        torch.autograd.grad(actual, inputs, grads),
        torch.autograd.grad(expected, references, grads),
    ):
        assert_close(a, b, rtol=0, atol=0)
    assert all(c[0] == torch.float32 and c[3] == "interleave" for c in calls)
    assert all(c[1].shape == (7, 1, 8) for c in calls)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("with_key", [False, True])
def test_vision_dispatch(monkeypatch, inverse, with_key):
    calls = []

    def rotary(query, key, cache, *, inverse):
        calls.append(inverse)
        return ComplexRoPE.apply_rotary_emb(query, key, cache, inverse=inverse)

    monkeypatch.setattr(MODULE, "rotary_mul", rotary)
    monkeypatch.setattr(MODULE.vision_encoder, "ComplexRoPE", ComplexRoPE)
    MODULE.enable_ascendc_rotary_mul()
    MODULE.enable_ascendc_rotary_mul()
    query = torch.randn(7, 2, 8, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True) if with_key else None
    cache = torch.polar(torch.ones(7, 1, 4), torch.randn(7, 1, 4))
    actual = MODULE.vision_encoder.ComplexRoPE.apply_rotary_emb(query, key, cache, inverse=inverse)
    expected = ComplexRoPE.apply_rotary_emb(query, key, cache, inverse=inverse)
    if with_key:
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    else:
        assert torch.equal(actual, expected)

    assert calls == [inverse]
