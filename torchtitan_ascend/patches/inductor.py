# SPDX-License-Identifier: Apache-2.0
"""Match TorchTitan's PyTorch version in the NPU Inductor backend.

Remove when the supported torch_npu includes
https://gitcode.com/Ascend/pytorch/merge_requests/45528.
"""

import inspect

from torchtitan_ascend.patches import patch


@patch(
    target="torch_npu._inductor.codegen.npu_combined_scheduling:NPUCombinedScheduling.__init__",
    kind="replace",
    deferred=True,
)
def scheduling(original):
    from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
    from torch_npu._inductor.codegen.npu_combined_scheduling import (
        CATLASSScheduling,
        NPUNoLinearTritonScheduling,
        NPUTritonScheduling,
    )

    NPUTritonScheduling.supports_sub_parent_epilogue = False

    def initialize(self, scheduler):
        CUDACombinedScheduling.__init__(self, scheduler)
        self._nolinear_triton_scheduling = NPUNoLinearTritonScheduling(scheduler)
        self._triton_scheduling = NPUTritonScheduling(scheduler)
        self._catlass_scheduling = CATLASSScheduling(scheduler)

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
