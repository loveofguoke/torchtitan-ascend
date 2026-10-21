"""Compare fla-npu convolution with PyTorch using identical cotangents."""

import pytest
import torch
import torch.nn.functional as F

from torchtitan_ascend.modules.gdn import FlaGDNKernel, FlaInnerGDN


@pytest.fixture
def npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")
    return "npu"


def _reference_convolution(x, weight, *, cu_seqlens, routing):
    offsets = cu_seqlens.tolist() if cu_seqlens is not None else [0, x.shape[0]]
    dtype = x.dtype
    # Accumulate shared-weight gradients across documents in FP32.
    x, weight = x.float(), weight.float()
    outputs = []
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        segment = x[begin:end].T.unsqueeze(0)
        output = F.conv1d(F.pad(segment, (weight.shape[-1] - 1, 0)), weight, groups=x.shape[-1])
        outputs.append(F.silu(output.to(dtype).squeeze(0).T))
    return torch.cat(outputs)


@pytest.mark.npu
@pytest.mark.parametrize("offsets", [None, [0, 2, 5, 9]])
def test_fla_convolution_matches_reference(offsets, npu):
    pytest.importorskip("fla_npu")
    gen = torch.Generator().manual_seed(3)
    inputs = [
        torch.randn(9, 256, generator=gen).bfloat16(),
        torch.randn(256, 1, 4, generator=gen).bfloat16(),
    ]
    grad_output = torch.randn(9, 256, generator=gen).bfloat16()
    kernel = FlaInnerGDN.Config(conv_kernel_size=4, kernel=FlaGDNKernel.Config()).build()
    expected, actual = [], []
    for module, device, results in [
        (_reference_convolution, "cpu", expected),
        (kernel.short_convolution, npu, actual),
    ]:
        x, weight = [t.to(device).clone().requires_grad_() for t in inputs]
        cu = torch.tensor(offsets, dtype=torch.int32, device=device) if offsets else None
        output = module(x, weight, cu_seqlens=cu, routing=None)
        output.backward(grad_output.to(device))
        results.extend([output.detach().cpu(), x.grad.cpu(), weight.grad.cpu()])
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)
