# SPDX-License-Identifier: Apache-2.0
"""Keep experimental NPU reduction grouping stable in deterministic mode."""

from functools import wraps

from torchtitan_ascend.patches import patch


@patch(
    target="torch_npu._inductor.triton_experimental.npu_triton_heuristics:cached_autotune",
    kind="wrap",
    deferred=True,
)
def apply(original):
    from torch._inductor.runtime.hints import HeuristicType
    from torch._inductor.runtime.triton_heuristics import (
        filter_reduction_configs_for_determinism,
    )

    @wraps(original)
    def cached_autotune(
        size_hints,
        configs,
        triton_meta,
        heuristic_type,
        filename=None,
        inductor_meta=None,
        custom_kernel=False,
    ):
        metadata = inductor_meta or {}
        if heuristic_type in (
            HeuristicType.REDUCTION,
            HeuristicType.PERSISTENT_REDUCTION,
        ) and any(
            metadata.get(key, False)
            for key in (
                "deterministic",
                "are_deterministic_algorithms_enabled",
                "force_filter_reduction_configs",
            )
        ):
            configs = filter_reduction_configs_for_determinism(metadata, configs)
            # A timed cache entry must not replace the deterministic config.
            inductor_meta = {**metadata, "force_disable_caches": True}
        return original(
            size_hints,
            configs,
            triton_meta,
            heuristic_type,
            filename=filename,
            inductor_meta=inductor_meta,
            custom_kernel=custom_kernel,
        )

    return cached_autotune
