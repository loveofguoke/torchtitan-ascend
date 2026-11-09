# SPDX-License-Identifier: Apache-2.0
"""Use Triton's grid handling for AutoBlockify NPU kernels."""

from functools import wraps

from torchtitan_ascend.patches import patch


@patch(
    target=(
        "torch_npu._inductor.triton_experimental.static_launcher:"
        "NPUStaticTritonCompileResult.can_statically_launch"
    ),
    kind="wrap",
    deferred=True,
)
def apply(original):
    @wraps(original)
    def can_statically_launch(binary, *, compile_meta, inductor_meta, heuristic_type):
        metadata = binary.metadata
        auto_blockify = (
            metadata.get("auto_blockify_enabled", False)
            if isinstance(metadata, dict)
            else getattr(metadata, "auto_blockify_enabled", False)
        )
        # Triton's generated launcher caps the physical grid for AutoBlockify.
        # The NPU static launcher currently launches the logical grid directly;
        # large Flex backward reductions then fail with a vector core exception.
        if auto_blockify:
            return None
        return original(
            binary,
            compile_meta=compile_meta,
            inductor_meta=inductor_meta,
            heuristic_type=heuristic_type,
        )

    return can_statically_launch
