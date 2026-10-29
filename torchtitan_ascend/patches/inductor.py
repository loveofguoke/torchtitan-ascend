# SPDX-License-Identifier: Apache-2.0
"""Adapt NPU scheduler capabilities and reduction interfaces."""

import inspect
from functools import wraps

from torchtitan_ascend.patches import patch


@patch(
    target="torch_npu._inductor.codegen.npu_combined_scheduling:NPUCombinedScheduling.__init__",
    kind="wrap",
    deferred=True,
)
def scheduling(original):
    from torch_npu._inductor.codegen.npu_combined_scheduling import NPUTritonScheduling

    @wraps(original)
    def initialize(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        NPUTritonScheduling.supports_sub_parent_epilogue = False
        return result

    return initialize


@patch(target="torch_npu._inductor.lowering:make_reduction", kind="replace", deferred=True)
def reduction(original):
    from torch._inductor import lowering
    from torch_npu._inductor import lowering as npu

    inner_takes_type = "reduction_type" in inspect.signature(npu._make_reduction_inner).parameters
    create_takes_strict = "strict_reduction" in inspect.signature(npu.Reduction.create).parameters

    def make_reduction(reduction_type, override_return_dtype=None, *, strict_reduction=False):
        def inner(x, axis=None, keepdims=False, *, dtype=None):
            inner_kwargs = {"reduction_type": reduction_type} if inner_takes_type else {}
            kwargs = npu._make_reduction_inner(
                x,
                axis=axis,
                keepdims=keepdims,
                dtype=dtype,
                override_return_dtype=override_return_dtype,
                **inner_kwargs,
            )
            if create_takes_strict:
                kwargs["strict_reduction"] = strict_reduction
            if npu.npu_config.dump_fx_graph:
                node_name = f"reduction_{next(npu.node_id)}"
                graphs = npu.fetch_graphs([x, axis if axis is not None else list(range(len(x.get_size())))])
                kwargs.update(
                    node_name=node_name,
                    traced_graph=npu.merge_traced_graphs(
                        graphs, npu.reduction_type_to_aten_fn[reduction_type], node_name, keepdim=keepdims
                    ),
                )
            result = npu.Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)
            if isinstance(result.data.data, npu.Reduction):
                result.realize()
            return result

        return inner

    lowering.make_reduction = make_reduction
    return make_reduction
