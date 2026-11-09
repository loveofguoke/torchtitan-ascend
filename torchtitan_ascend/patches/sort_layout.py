# SPDX-License-Identifier: Apache-2.0
"""Constrain NPU stable-sort fallbacks to their actual contiguous output layout."""

from functools import wraps

from torchtitan_ascend.patches import patch


@patch(target="torch._inductor.ir:FallbackKernel.create", kind="wrap", deferred=True)
def stable_sort_layout(original):
    from torch import ops
    from torch._inductor import ir

    @wraps(original)
    def create(cls, kernel, *args, **kwargs):
        if kernel is ops.aten.sort.stable:
            value = args[0] if args else kwargs["self"]
            if value.get_device().type == "npu":
                # ACLNN Sort returns contiguous outputs, while FakeTensor may
                # preserve a transposed input's strides. Constrain the input
                # before fallback inference so compiled consumers read indices
                # correctly. This also covers node-level forced fallbacks,
                # which bypass the ordinary operator lowering table.
                value = ir.ExternKernel.require_contiguous(value)
                if args:
                    args = (value, *args[1:])
                else:
                    kwargs = {**kwargs, "self": value}
        return original(cls, kernel, *args, **kwargs)

    return create
