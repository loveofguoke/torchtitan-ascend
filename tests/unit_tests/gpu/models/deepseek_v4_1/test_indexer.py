# SPDX-License-Identifier: Apache-2.0
"""``_student_logits`` must match the advanced-indexing reference it replaced.

The ``npu``/``float32`` combination is intentionally absent: the legacy
advanced-indexing backward is itself inaccurate there.
"""

import pytest
import torch

from torchtitan_ascend.models.deepseek_v4_1.indexer import (
    _student_logits,
    compressed_gather_indices,
)

T, N, H, D, K, RATIO = 24, 12, 3, 5, 4, 2


def reference(idx_q, idx_k, weights, local_TK, cu_seqlens):
    selected_TKD = idx_k[compressed_gather_indices(local_TK, cu_seqlens, RATIO)]
    logits_THK = torch.einsum("thd,tkd->thk", idx_q, selected_TKD)
    logits_THK = logits_THK.relu() * weights.unsqueeze(-1)
    return logits_THK.sum(dim=1).masked_fill(local_TK < 0, -torch.inf)


@pytest.mark.parametrize(
    "device, dtype, tol",
    [
        pytest.param("cpu", torch.float64, {}, id="cpu"),
        pytest.param(
            "npu",
            torch.bfloat16,
            {"atol": 1e-2, "rtol": 1e-2},
            id="npu",
            marks=pytest.mark.skipif(
                not torch.npu.is_available(), reason="NPU is unavailable"
            ),
        ),
    ],
)
def test_student_logits_matches_advanced_indexing(device, dtype, tol):
    torch.manual_seed(6)
    # Two documents of 12 tokens; ratio 2 -> 6 pool rows per document, N = 12.
    cu_seqlens = torch.tensor([0, 12, 24], dtype=torch.int32, device=device)
    local_TK = torch.randint(0, 6, (T, K), device=device)
    local_TK[0, 1] = -1  # invalid slots read row 0 and are masked downstream
    local_TK[17, 0] = -1
    local_TK[17, 3] = -1

    def inputs():
        return (
            torch.randn(T, H, D, device=device, dtype=dtype, requires_grad=True),
            torch.randn(N, D, device=device, dtype=dtype, requires_grad=True),
            torch.randn(T, H, device=device, dtype=dtype, requires_grad=True),
        )

    idx_q, idx_k, weights = inputs()
    ref_q, ref_k, ref_w = inputs()
    with torch.no_grad():
        for live, ref in ((idx_q, ref_q), (idx_k, ref_k), (weights, ref_w)):
            ref.data.copy_(live.data)

    out = _student_logits(idx_q, idx_k, weights, local_TK, cu_seqlens, RATIO)
    ref = reference(ref_q, ref_k, ref_w, local_TK, cu_seqlens)

    torch.testing.assert_close(out, ref, **tol)
    assert torch.equal(torch.isneginf(out), local_TK < 0)

    grads = torch.autograd.grad(out.sum(), (idx_q, idx_k, weights))
    ref_grads = torch.autograd.grad(ref.sum(), (ref_q, ref_k, ref_w))
    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(grad, ref_grad, **tol)
