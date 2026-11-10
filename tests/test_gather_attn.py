# SPDX-License-Identifier: Apache-2.0
"""The NPU fast path of the patched ``_packed_gather_attn`` matches upstream.

The wrap routes non-NPU tensors to the original implementation, so feeding
the same inputs through CPU (original) and NPU (patched) must agree in
outputs, LSE, and gradients of the query and both KV pools.
"""

import importlib

import pytest
import torch

# Not `import ... as`: the `gather_attn` attribute of `attn_gym.sparse` is
# shadowed by its identically named re-exported function.
reference_impl = importlib.import_module("attn_gym.sparse.gather_attn.impl.reference")

B, HQ, T, D, POOL, TOPK, WINDOW = 1, 4, 24, 8, 10, 6, 5


def _base_inputs():
    torch.manual_seed(11)
    kv_indices = torch.randint(0, 4, (B, T, TOPK))
    kv_indices[0, 3, 0] = -1  # invalid slots read row 0 and are masked downstream
    kv_indices[0, 20, 2] = -1
    kv_indices[0, 20, 4] = 40  # out-of-pool sentinel, also masked downstream
    return (
        torch.randn(B, HQ, T, D, requires_grad=True),
        torch.randn(B, 1, T, D, requires_grad=True),
        torch.randn(B, 1, POOL, D, requires_grad=True),
        kv_indices,
        torch.randn(HQ),
        torch.tensor([0, 10, 24], dtype=torch.int32),
        torch.tensor([0, 4, 10], dtype=torch.int32),
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float32, {"atol": 1e-4, "rtol": 1e-3}),
        # ~2 bf16 ulp at gradient magnitudes of order 10: the two paths sum
        # duplicated-index gradients in different orders.
        (torch.bfloat16, {"atol": 0.25, "rtol": 2e-2}),
    ],
)
def test_packed_gather_attn_matches_upstream(dtype, tol):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")

    packed = reference_impl._packed_gather_attn  # patched by conftest setup()
    base = _base_inputs()
    outputs, grads = {}, {}
    for device in ("cpu", "npu"):
        # Only the floating-point inputs follow the test dtype; cu_seqlens
        # and kv_indices keep their integer dtypes.
        args = [
            t.detach().to(device, dtype).requires_grad_(t.requires_grad)
            for t in base[:3]
        ]
        args += [
            base[3].to(device),
            base[4].to(device, dtype),
            base[5].to(device),
            base[6].to(device),
        ]
        out, lse = packed(*args, WINDOW, scale=D**-0.5)
        loss = out.sum() + torch.nan_to_num(lse, neginf=0.0).sum()
        grads[device] = torch.autograd.grad(loss, args[:3])
        outputs[device] = (out.cpu(), lse.cpu())

    for out, ref_out in zip(outputs["npu"], outputs["cpu"], strict=True):
        torch.testing.assert_close(out, ref_out, **tol)
    for grad, ref_grad in zip(grads["npu"], grads["cpu"], strict=True):
        torch.testing.assert_close(grad.cpu(), ref_grad, **tol)
