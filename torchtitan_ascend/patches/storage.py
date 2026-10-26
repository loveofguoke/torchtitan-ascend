# SPDX-License-Identifier: Apache-2.0
"""Use NPUStorageImpl for external storage, including remat placeholders.

Requires torch_npu placeholder storage support (Ascend/pytorch !47944).
Remove when the supported PyTorch version includes factory dispatch from
https://github.com/pytorch/pytorch/pull/194598.
"""

from functools import wraps

from torchtitan_ascend.patches import patch


@patch(target="torch._C:_construct_storage_from_data_pointer", kind="wrap")
def apply(original):
    import torch

    if not hasattr(torch, "npu"):
        return original
    import torch_npu

    @wraps(original)
    def construct_storage(data_ptr, device, size_bytes):
        device = torch.device(device)
        if device.type == "npu":
            return torch_npu._C._construct_storage_from_data_pointer(data_ptr, device, size_bytes)
        return original(data_ptr, device, size_bytes)

    return construct_storage
