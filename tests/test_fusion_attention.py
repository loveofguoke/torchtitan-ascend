# SPDX-License-Identifier: Apache-2.0
"""CPU mask/API contracts; NPU accuracy requires the integration runs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torchtitan_ascend.overrides.fusion_attention as MODULE
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.testing import assert_close


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("value_dim", [4, 8])
def test_fusion_mask_and_mla_gradients(monkeypatch, causal, value_dim):
    import torch_npu

    segments = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    def mask(b, h, q, k):
        visible = segments[q] == segments[k]
        return visible & (q >= k) if causal else visible

    calls = []

    def fake(q, k, v, **kwargs):
        calls.append(kwargs)
        return (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=~kwargs["atten_mask"], scale=kwargs["scale"]
            ),
        )

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", fake)
    torch.manual_seed(42)
    inputs = [torch.randn(7, 2, d, requires_grad=True) for d in (8, 8, value_dim)]
    references = [x.detach().clone().requires_grad_() for x in inputs]
    output = MODULE.fusion_attention(
        inputs[0],
        inputs[1],
        inputs[2],
        SimpleNamespace(mask_mod=mask),
        scale=0.7,
        cpu_offload=False,
    )
    q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in references]
    visible = mask(0, 0, torch.arange(7)[:, None], torch.arange(7)[None, :])
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=visible, scale=0.7)
    expected = expected.squeeze(0).transpose(0, 1)
    assert_close(output, expected)
    grad = torch.randn_like(output)
    for actual, reference in zip(
        torch.autograd.grad(output, inputs, grad),
        torch.autograd.grad(expected, references, grad),
    ):
        assert_close(actual, reference)
    assert calls[0]["input_layout"] == "BNSD"
    assert calls[0]["keep_prob"] == 1.0
    assert calls[0]["head_num"] == 2


def test_cpu_fallback_with_score_mod_and_lse(monkeypatch):
    def cpu_attention(
        q, k, v, *, score_mod, block_mask, scale, enable_gqa, return_aux, kernel_options
    ):
        assert score_mod is kwargs["score_mod"]
        assert block_mask is metadata
        assert not enable_gqa
        assert return_aux.lse
        scores = (q @ k.transpose(-2, -1)) * (
            q.shape[-1] ** -0.5 if scale is None else scale
        )
        b, h, qi, ki = torch.meshgrid(
            *[torch.arange(size) for size in scores.shape], indexing="ij"
        )
        scores = score_mod(scores, b, h, qi, ki)
        visible = create_mask(
            block_mask.mask_mod, *scores.shape[:2], *scores.shape[2:], device="cpu"
        )
        scores = scores.masked_fill(~visible, -float("inf"))
        return scores.softmax(-1) @ v, SimpleNamespace(lse=scores.logsumexp(-1))

    monkeypatch.setattr(MODULE.FlexInnerAttention, "_compiled_flex_attn", cpu_attention)
    base = MODULE.FlexInnerAttention(MODULE.FlexInnerAttention.Config())
    fused = MODULE.AscendCFlexInnerAttention(MODULE.AscendCFlexInnerAttention.Config())
    metadata = create_block_mask(lambda b, h, q, k: q >= k, 1, None, 7, 7, device="cpu")
    inputs = [torch.randn(7, 2, 8, requires_grad=True) for _ in range(3)]
    kwargs = dict(
        attention_metadata=metadata,
        score_mod=lambda score, b, h, q, k: score * 0.5,
        out_transform=lambda out, lse: out * lse.sigmoid().unsqueeze(-1),
    )
    actual, expected = fused(*inputs, **kwargs), base(*inputs, **kwargs)
    assert torch.equal(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.sum(), inputs),
        torch.autograd.grad(expected.sum(), inputs),
    ):
        assert torch.equal(a, b)


def test_mask_cache_separates_documents_and_shapes(monkeypatch):
    calls = []
    original = MODULE.create_mask

    def create(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(MODULE, "create_mask", create)
    q = torch.empty(7, 2, 8)
    metadata = SimpleNamespace(mask_mod=lambda b, h, q, k: q >= k)
    mask = MODULE._get_attn_mask(metadata, q, q)
    assert MODULE._get_attn_mask(metadata, q, q) is mask
    assert len(calls) == 1
    other = SimpleNamespace(mask_mod=lambda b, h, q, k: q == k)
    assert not torch.equal(MODULE._get_attn_mask(other, q, q), mask)
    assert MODULE._get_attn_mask(metadata, q[:4], q).shape == (1, 2, 4, 7)
    assert MODULE._get_attn_mask(metadata, q[:, :1], q).shape == (1, 1, 7, 7)
    metadata.mask_mod = other.mask_mod
    assert torch.equal(
        MODULE._get_attn_mask(metadata, q, q), MODULE._get_attn_mask(other, q, q)
    )
    assert len(calls) == 5


def test_mask_cache_released_with_metadata():
    import gc
    import weakref

    metadata = create_block_mask(lambda b, h, q, k: q >= k, 1, 2, 7, 7, device="cpu")
    q = torch.empty(7, 2, 8)
    mask_ref = weakref.ref(MODULE._get_attn_mask(metadata, q, q))
    assert mask_ref() is not None
    del metadata
    gc.collect()
    assert mask_ref() is None


@pytest.mark.parametrize("value_dim", [4, 8])
@pytest.mark.parametrize("requires_grad", [(True, True, True), (False, True, False)])
def test_offload_backward_matches_attention(monkeypatch, value_dim, requires_grad):
    import torch_npu

    forwards, backwards = [], []

    def forward(q, k, v, **kwargs):
        forwards.append(kwargs)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=~kwargs["atten_mask"], scale=kwargs["scale"]
        )
        return output, torch.ones(1), torch.full((1,), 2.0)

    def backward(q, k, v, grad, **kwargs):
        backwards.append(kwargs)
        assert torch.equal(kwargs["softmax_max"], torch.ones(1))
        assert torch.equal(kwargs["softmax_sum"], torch.full((1,), 2.0))
        with torch.enable_grad():
            inputs = [t.detach().requires_grad_() for t in (q, k, v)]
            output = F.scaled_dot_product_attention(
                *inputs, attn_mask=~kwargs["atten_mask"], scale=kwargs["scale_value"]
            )
            assert_close(kwargs["attention_in"], output)
            return torch.autograd.grad(output, inputs, grad)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", forward)
    monkeypatch.setattr(torch_npu, "npu_fusion_attention_grad", backward)
    inputs = [
        torch.randn(7, 2, d, requires_grad=r)
        for d, r in zip((8, 8, value_dim), requires_grad)
    ]
    references = [t.detach().clone().requires_grad_(t.requires_grad) for t in inputs]
    metadata = SimpleNamespace(mask_mod=lambda b, h, q, k: q >= k)
    actual = MODULE.fusion_attention(
        inputs[0], inputs[1], inputs[2], metadata, scale=0.7
    )
    expected = MODULE.fusion_attention(
        references[0],
        references[1],
        references[2],
        metadata,
        scale=0.7,
        cpu_offload=False,
    )
    assert_close(actual, expected)
    grad = torch.randn_like(actual)
    for _ in range(2):
        for a, b in zip(
            torch.autograd.grad(
                actual, [t for t in inputs if t.requires_grad], grad, retain_graph=True
            ),
            torch.autograd.grad(
                expected,
                [t for t in references if t.requires_grad],
                grad,
                retain_graph=True,
            ),
        ):
            assert_close(a, b)
    assert len(forwards) == 2
    assert len(backwards) == 2
    assert all(call["atten_mask"] is forwards[0]["atten_mask"] for call in backwards)


def test_offload_not_used_without_gradients(monkeypatch):
    import torch_npu

    def forward(q, k, v, **kwargs):
        return (v,)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", forward)
    metadata = SimpleNamespace(mask_mod=lambda b, h, q, k: q >= k)
    q = torch.randn(7, 2, 8, requires_grad=True)
    with torch.no_grad():
        assert MODULE.fusion_attention(q, q, q, metadata).shape == q.shape
