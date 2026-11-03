# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; these tests do not establish NPU accuracy."""

import pytest
import torch

import torchtitan_ascend.overrides.situ_glu as MODULE
from torch.testing import assert_close


def situ_reference(packed, *, beta, linear_beta, **ignored):
    gate, up = packed.chunk(2, -1)
    if linear_beta:
        up = linear_beta * torch.tanh(up / linear_beta)
    return beta * torch.tanh(gate / beta) * gate.sigmoid() * up


@pytest.mark.parametrize("linear_beta", [None, 2.0, 25.0])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_situ_layout_and_backward(monkeypatch, linear_beta, noncontiguous):
    seen = []

    def fwd(packed, **kwargs):
        seen.append((packed.shape, kwargs))
        return situ_reference(packed, **kwargs)

    def bwd(dy, packed, **kwargs):
        with torch.enable_grad():
            x = packed.detach().requires_grad_()
            return torch.autograd.grad(situ_reference(x, **kwargs), x, dy)[0]

    monkeypatch.setattr(MODULE, "_operators", lambda: (fwd, bwd))
    x = torch.randn(3, 2, 11, dtype=torch.float64)
    if noncontiguous:
        x = x.transpose(0, 2).contiguous().transpose(0, 2)
    x.requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    actual = MODULE.situ_glu(x, 4.0, linear_beta)
    expected = situ_reference(ref_x.flatten(-2), beta=4.0, linear_beta=linear_beta or 0.0)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    assert_close(actual, expected, atol=0, rtol=0)
    assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
    assert seen[0][1]["activate_left"] is True
    assert seen[0][1]["linear_beta"] == (linear_beta or 0.0)


def test_situ_rejects_wrong_layout():
    with pytest.raises(ValueError):
        MODULE.situ_glu(torch.randn(2, 3, 7), 4.0, None)
    with pytest.raises(ValueError):
        MODULE.situ_glu(torch.randn(2, 2, 7), 4.0, 0.0)


@pytest.mark.parametrize("offsets", [None, torch.tensor([0, 2, 5])])
def test_situ_accepts_packed_gate_up(monkeypatch, offsets):
    packed = torch.randn(5, 2, 7, requires_grad=True)
    calls = []

    def fused(gate_up, beta, linear_beta):
        calls.append((gate_up, beta, linear_beta))
        return gate_up[:, 0] * gate_up[:, 1]

    monkeypatch.setattr(MODULE, "situ_glu", fused)
    actual = MODULE.ascendc_situ_glu(MODULE.SiTUGLU.Config(beta=4.0, linear_beta=25.0)).build()(packed, offsets=offsets)
    expected = packed[:, 0] * packed[:, 1]
    assert calls[0][0] is packed
    assert calls[0][1:] == (4.0, 25.0)
    assert_close(actual, expected, rtol=0, atol=0)
    assert_close(torch.autograd.grad(actual.sum(), packed)[0], torch.autograd.grad(expected.sum(), packed)[0])
