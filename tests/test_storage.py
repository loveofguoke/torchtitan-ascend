# SPDX-License-Identifier: Apache-2.0
"""Validate allocation-free remat placeholders."""

import pytest
import torch


@pytest.mark.npu
def test_remat_placeholder_does_not_allocate():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")
    import torch_npu

    remat = pytest.importorskip("torch_remat._placeholder")
    source = torch.empty((1, 1, 1, 1), device="npu", dtype=torch.bfloat16)
    metadata = remat._TensorMetadata(
        (1, 16, 16384, 128), (33554432, 2097152, 128, 1), source.dtype, source.device, 67108864
    )
    before = torch.npu.memory_allocated()
    storage = remat._null_storage(metadata.storage_nbytes, source.device)
    assert storage.nbytes() == metadata.storage_nbytes
    assert not storage.resizable()
    placeholder = remat._make_placeholder_tensor(metadata, "storage regression", requires_grad=True)
    assert placeholder.shape == metadata.shape
    assert placeholder.stride() == metadata.stride
    assert placeholder.requires_grad
    with pytest.raises(RuntimeError, match="storage regression"):
        placeholder.data_ptr()
    assert torch_npu.get_npu_format(placeholder) == torch_npu.get_npu_format(source)
    assert torch.npu.memory_allocated() == before
