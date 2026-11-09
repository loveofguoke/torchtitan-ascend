# SPDX-License-Identifier: Apache-2.0
"""Check the compact backward mask with captured packed-document tensors."""

import pytest
import torch


@pytest.mark.npu
def test_compiled_flex_preserves_short_packed_mask_gqa_gradients():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")
    from torch._inductor import config
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.npu.set_device(0)
    torch.manual_seed(42)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_selection = config.deterministic
    try:
        config.deterministic = False
        torch.use_deterministic_algorithms(False)
        # Match the native packed-document mask: only offsets are padded. An
        # invalid lane must not read beyond the eight document_id entries.
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], device="npu")
        document_id = torch.eq(positions, 0).int().cumsum(0).to(torch.int32) - 1
        offsets = torch.full((128,), 8, dtype=torch.int32, device="npu")
        offsets[:2] = torch.tensor([0, 3], dtype=torch.int32, device="npu")

        def packed_mask(batch, head, query, key):
            return (key <= query) & (key >= offsets[document_id[query]])

        block_mask = create_block_mask(
            packed_mask, B=1, H=1, Q_LEN=8, KV_LEN=8, device="npu", BLOCK_SIZE=128, _compile=True
        )
        q = torch.randn(1, 8, 8, 32, device="npu", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 4, 8, 32, device="npu", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k).requires_grad_()
        compiled = torch.compile(flex_attention, fullgraph=True, dynamic=True, options={"max_autotune": True})
        output = compiled(q, k, v, block_mask=block_mask, enable_gqa=True)
        gradients = torch.autograd.grad(output.float().square().mean(), (q, k, v))
        reference_inputs = [x.detach().cpu().float().requires_grad_() for x in (q, k, v)]
        tokens = torch.arange(8)
        document_start = torch.tensor([0, 0, 0, 3, 3, 3, 3, 3])
        reference_mask = (tokens[None, :] <= tokens[:, None]) & (tokens[None, :] >= document_start[:, None])
        reference = torch.nn.functional.scaled_dot_product_attention(
            *reference_inputs, attn_mask=reference_mask, enable_gqa=True
        )
        reference_gradients = torch.autograd.grad(reference.square().mean(), reference_inputs)
        for actual, expected in zip((output, *gradients), (reference, *reference_gradients), strict=True):
            relative_error = (actual.detach().cpu().float() - expected).norm() / expected.norm()
            assert relative_error < 0.01
        torch.npu.synchronize()
    finally:
        config.deterministic = previous_selection
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
