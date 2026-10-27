# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GLM-5 model configuration changes required by Ascend NPU."""

import math
import warnings
from functools import partial

import torch
from torch.distributed.tensor import DTensor

_ORIGINAL_TRUNC_NORMAL = torch.nn.init.trunc_normal_


def npu_safe_trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialize a DTensor using elementwise inverse-CDF operations."""
    if not isinstance(tensor, DTensor):
        return _ORIGINAL_TRUNC_NORMAL(tensor, mean=mean, std=std, a=a, b=b, generator=generator)
    if tensor.is_meta:
        return tensor
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    def norm_cdf(value: float) -> float:
        return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0

    lower = norm_cdf((a - mean) / std)
    upper = norm_cdf((b - mean) / std)
    with torch.no_grad():
        tensor.uniform_(2 * lower - 1, 2 * upper - 1, generator=generator)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def replace_trunc_normal_initializers(model_config) -> int:
    """Replace trunc-normal partials throughout one model config tree."""
    from torchtitan.protocols.module import Module

    replaced = 0
    for _, config, _, _ in model_config.traverse(Module.Config, recurse=True):
        if config.param_init is None:
            continue
        converted = {}
        for name, initializer in config.param_init.items():
            if isinstance(initializer, partial) and initializer.func is _ORIGINAL_TRUNC_NORMAL:
                initializer = partial(
                    npu_safe_trunc_normal_,
                    *initializer.args,
                    **(initializer.keywords or {}),
                )
                replaced += 1
            converted[name] = initializer
        config.param_init = converted
    return replaced
