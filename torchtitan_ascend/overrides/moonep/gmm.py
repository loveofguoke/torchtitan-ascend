# SPDX-License-Identifier: Apache-2.0
"""Ascend grouped GEMMs in the stored expert [group, out, in] orientation."""

import torch


def rows_mm(input_rows, weight_projection, group_end_offsets, *, transpose=True):
    """Read mapped home/slot rows; unmapped remote experts have zero tokens."""
    import torch_npu

    full_projection, group_indices = weight_projection
    group_token_counts = group_end_offsets.diff(
        prepend=group_end_offsets.new_zeros(1)
    ).to(torch.int64)
    full_group_end_offsets = (
        group_token_counts.new_zeros(full_projection.shape[0])
        .scatter_(0, group_indices, group_token_counts)
        .cumsum(0)
    )
    return torch_npu.npu_grouped_matmul(
        [input_rows],
        [full_projection.transpose(-2, -1) if transpose else full_projection],
        bias=None,
        group_list=full_group_end_offsets,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )[0]


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
