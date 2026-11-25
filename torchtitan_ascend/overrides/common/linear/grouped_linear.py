# SPDX-License-Identifier: Apache-2.0
"""Ascend grouped matmul for the ordinary expert dispatcher."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torchtitan.config import derive, override
from torchtitan.models.common.linear import GroupedLinear

from torchtitan_ascend.overrides.common.gmm import weight_grad


def _rows_mm(rows, weight, offsets):
    import torch_npu

    return torch_npu.npu_grouped_matmul(
        [rows],
        [weight],
        bias=None,
        group_list=offsets,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )[0]


class _AscendGroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, weight, offsets):
        ctx.save_for_backward(rows, weight, offsets)
        return _rows_mm(rows, weight.transpose(-2, -1), offsets)

    @staticmethod
    def backward(ctx, grad):
        rows, weight, offsets = ctx.saved_tensors
        grad = grad.contiguous()
        d_rows = _rows_mm(grad, weight, offsets) if ctx.needs_input_grad[0] else None
        d_weight = weight_grad(grad, rows, offsets) if ctx.needs_input_grad[1] else None
        return d_rows, d_weight, None


spmd.register_local_autograd_function(_AscendGroupedMatmul)


class AscendGroupedLinear(GroupedLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
        if input_RI.device.type != "npu" or input_RI.dtype != torch.bfloat16:
            raise ValueError("Ascend grouped linear requires BF16 Ascend activations.")
        if input_RI.size(0) == 0:
            zero = input_RI.sum() + weight_EOI.flatten()[:0].sum()
            return input_RI.new_empty((0, weight_EOI.size(1))) + zero.to(input_RI.dtype)
        return _AscendGroupedMatmul.apply(
            input_RI.contiguous(),
            weight_EOI.bfloat16(),
            offsets_E.to(torch.int64),
        )


@override(
    target=GroupedLinear.Config,
    exact=True,
    description="AscendC grouped matmul for ordinary routed experts",
)
def npu_grouped_linear(config: GroupedLinear.Config) -> AscendGroupedLinear.Config:
    return derive(config, AscendGroupedLinear.Config)
