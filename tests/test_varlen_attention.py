# SPDX-License-Identifier: Apache-2.0
"""Validate the public varlen patch independently of native ATen registrations."""

import pytest
import torch
from torch.nn.attention import varlen


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,window", [(2, (-1, -1)), (4, (-1, 0)), (4, (8, 0))])
def test_output_lse_and_gradients(dtype, heads, window, monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected native FlashAttention or unused LSE computation")

    for name in ("_flash_attention_forward", "_flash_attention_backward"):
        monkeypatch.setattr(torch.ops.aten, name, forbidden)
    torch.manual_seed(4439)
    q = torch.randn(64, heads, 64, device="npu", dtype=dtype, requires_grad=True)
    k = torch.randn(96, 2, 64, device="npu", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cq = torch.tensor([0, 24, 64], device="npu", dtype=torch.int32)
    ck = torch.tensor([0, 40, 96], device="npu", dtype=torch.int32)
    refs = [t.detach().float().cpu().requires_grad_() for t in (q, k, v)]
    outputs, lses = [], []
    for qs, qe, ks, ke in ((0, 24, 0, 40), (24, 64, 40, 96)):
        qq = refs[0][qs:qe].transpose(0, 1)
        kk = refs[1][ks:ke].transpose(0, 1).repeat_interleave(heads // 2, 0)
        vv = refs[2][ks:ke].transpose(0, 1).repeat_interleave(heads // 2, 0)
        scores = qq @ kk.transpose(-1, -2) * 0.125
        pos_q = torch.arange(qe - qs)[:, None] + (ke - ks) - (qe - qs)
        pos_k = torch.arange(ke - ks)[None, :]
        if window[1] == 0:
            scores = scores.masked_fill(pos_k > pos_q, -float("inf"))
        if window[0] >= 0:
            scores = scores.masked_fill(pos_k < pos_q - window[0], -float("inf"))
        outputs.append((scores.softmax(-1) @ vv).transpose(0, 1))
        lses.append(scores.logsumexp(-1))
    expected = torch.cat(outputs)
    out, lse = varlen.varlen_attn(
        q,
        k,
        v,
        cq,
        ck,
        40,
        56,
        window_size=window,
        enable_gqa=heads != 2,
        return_aux=varlen.AuxRequest(lse=True),
    )
    assert out.dtype == dtype and not lse.requires_grad
    dy = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dy)
    ref_grads = torch.autograd.grad(expected, refs, dy.float().cpu())
    for got, want in zip((out, lse, *grads), (expected, torch.cat(lses, dim=1), *ref_grads), strict=True):
        torch.testing.assert_close(got.detach().float().cpu(), want, atol=0.025, rtol=0.025)

    if dtype == torch.bfloat16:
        from torchtitan_ascend.patches.varlen_attention import _attention

        monkeypatch.setattr(_attention, "_lse", forbidden)
        result = varlen.varlen_attn(q, k, v, cq, ck, 40, 56, window_size=window, enable_gqa=heads != 2)
        torch.testing.assert_close(result, out)
