# SPDX-License-Identifier: Apache-2.0
"""Shared Ascend grouped GEMM helpers."""

import torch


def weight_grad(grad_output_rows, input_rows, group_end_offsets):
    """Contract tokens in one K-split GEMM, returning [group, out, in]."""
    import torch_npu

    return torch_npu.npu_grouped_matmul(
        [input_rows.t()],
        [grad_output_rows],
        bias=None,
        group_list=group_end_offsets.to(torch.int64),
        split_item=3,
        group_type=2,
        group_list_type=0,
    )[0].transpose(-2, -1)
