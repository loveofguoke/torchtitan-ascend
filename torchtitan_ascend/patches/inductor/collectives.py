# SPDX-License-Identifier: Apache-2.0
"""Preserve collective lifetimes in the experimental NPU compiler."""

from functools import wraps

from torchtitan_ascend.patches import patch


@patch(
    target="torch_npu._inductor.triton_experimental.lowering:_register_npu_inductor_fallbacks",
    kind="wrap",
    deferred=True,
)
def collective_lowerings(original):
    import torch
    from torch._inductor import lowering
    from torch_npu._inductor.triton_experimental import lowering as npu_lowering

    def broadcast_layout(fx_node, *args, **kwargs):
        # Native broadcast lowering calls broadcast_, which needs contiguous
        # HCCL storage. Preserve that requirement for transposed NPU inputs.
        if args[0].get_device().type == "npu":
            return lowering.require_contiguous(fx_node, *args, **kwargs)
        return lowering.constrain_to_fx_strides(fx_node, *args, **kwargs)

    def wait_layout(fx_node, *args, **kwargs):
        inp = args[0]
        if isinstance(inp, (list, tuple)):
            inp = next(iter(inp), None)
        if inp is not None and inp.get_device().type == "npu":
            # A layout copy before wait loses the storage holding the work.
            return args, kwargs
        return lowering.constrain_to_fx_strides(fx_node, *args, **kwargs)

    @wraps(original)
    def register(*args, **kwargs):
        # Collective/wait IR records mutation and volatile reads that generic
        # operator schemas cannot express. Keep both ends of that lifetime.
        keep = npu_lowering.KEEP_UPSTREAM_LOWERING
        for op in lowering.lowerings:
            if lowering.in_namespace(op, "_c10d_functional") and op not in keep:
                keep.append(op)
        result = original(*args, **kwargs)
        lowering.add_layout_constraint(
            torch.ops._c10d_functional.broadcast, broadcast_layout
        )
        wait_ops = [torch.ops._c10d_functional.wait_tensor]
        wait_tensors = getattr(torch.ops._c10d_functional, "wait_tensors", None)
        if wait_tensors is not None:
            wait_ops.append(wait_tensors)
        for wait in wait_ops:
            lowering.add_layout_constraint(wait, wait_layout)
        return result

    return register
