# SPDX-License-Identifier: Apache-2.0
"""Check mixed-dtype token restoration and both input gradients on NPU."""

import pytest
import torch
import torch_npu

from torchtitan_ascend.device import get_device_name
from torchtitan_ascend.overrides.common.moe.permute import permute, unpermute

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "probs_dtype", [None, torch.bfloat16, torch.float16, torch.float32]
)
def test_unpermute_output_and_gradients(dtype, probs_dtype):
    if not torch.npu.is_available() or not get_device_name().startswith("Ascend950"):
        pytest.skip("requires Ascend 950 for mixed unpermute")
    generator = torch.Generator().manual_seed(42)
    tokens, hidden, topk = 128, 256, 8 if probs_dtype is not None else 1
    ids_TK = torch.randint(0, 64, (tokens, topk), generator=generator).to(
        device="npu", dtype=torch.int32
    )
    x_TD = torch.randn(tokens, hidden, generator=generator).to("npu", dtype)
    _, inverse_N = permute(x_TD, ids_TK)
    sorted_ND = (
        torch.randn(tokens * topk, hidden, generator=generator)
        .to("npu", dtype)
        .requires_grad_()
    )
    probs_TK = (
        torch.rand(tokens, topk, generator=generator)
        .softmax(-1)
        .to(device="npu", dtype=probs_dtype)
        .requires_grad_()
        if probs_dtype is not None
        else None
    )
    reference_ND = sorted_ND.detach().float().requires_grad_()
    reference_probs_TK = (
        probs_TK.detach().to(dtype=torch.float32, copy=True).requires_grad_()
        if probs_TK is not None
        else None
    )
    # Compare against FP32 combine, including its gradients.
    reference_TD = torch_npu.npu_moe_token_unpermute(
        reference_ND, inverse_N, reference_probs_TK
    ).to(dtype)
    result_TD = unpermute(sorted_ND, inverse_N, probs_TK, allow_mixed_dtype=True)
    assert result_TD.dtype == dtype
    grad_TD = torch.randn(tokens, hidden, generator=generator).to("npu", dtype)
    result_TD.backward(grad_TD)
    reference_TD.backward(grad_TD)
    # Low-precision products may round before accumulation.
    token_tol = {torch.bfloat16: 2e-2, torch.float16: 2e-3, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(result_TD, reference_TD, rtol=token_tol, atol=token_tol)
    torch.testing.assert_close(
        sorted_ND.grad.float(), reference_ND.grad, rtol=token_tol, atol=token_tol
    )
    if probs_TK is not None:
        assert probs_TK.grad.dtype == probs_dtype
        torch.testing.assert_close(
            probs_TK.grad,
            reference_probs_TK.grad.to(probs_dtype),
            rtol=token_tol,
            atol=token_tol,
        )
