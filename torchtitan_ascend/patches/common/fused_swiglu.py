# SPDX-License-Identifier: Apache-2.0
"""Dispatch the upstream fused SwiGLU Triton kernels on PrivateUse1."""

import sys

import torch

from torchtitan_ascend.patches import patch


@patch(
    target="torchtitan_recipes.overrides.fused_swiglu:silu_and_mul_op", deferred=True
)
def forward(original):
    module = sys.modules["torchtitan_recipes.overrides.fused_swiglu"]
    original.register_kernel(torch._C._get_privateuse1_backend_name())(
        module.silu_and_mul_forward_kernel
    )
    return original


@patch(
    target="torchtitan_recipes.overrides.fused_swiglu:silu_and_mul_backward_op",
    deferred=True,
)
def backward(original):
    module = sys.modules["torchtitan_recipes.overrides.fused_swiglu"]
    original.register_kernel(torch._C._get_privateuse1_backend_name())(
        module.silu_and_mul_backward_kernel
    )
    return original
