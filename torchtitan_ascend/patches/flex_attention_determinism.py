# SPDX-License-Identifier: Apache-2.0
"""Keep deterministic NPU Flex compilation independent of earlier input shapes."""

from functools import wraps

from torchtitan_ascend.patches import patch


@patch(
    target="torchtitan.distributed.utils:set_determinism", kind="wrap", deferred=True
)
def deterministic_flex(original):
    @wraps(original)
    def set_determinism(
        parallelism_context, device, debug_config, distinct_seed_mesh_axes
    ):
        result = original(
            parallelism_context, device, debug_config, distinct_seed_mesh_axes
        )
        if debug_config.deterministic:
            import torch
            from torch.compiler import config as compiler_config
            from torch.nn.attention.flex_attention import flex_attention
            from torchtitan.models.common.attention import FlexInnerAttention

            # Lowering patches are not part of the default FX graph cache key.
            # A graph generated before the sort-layout repair can otherwise
            # bypass that repair entirely on a cache hit. Version the compiler
            # caches once, preserving caller tags and normal cache reuse.
            version = "torchtitan-ascend-flex-layout-v1"
            previous = compiler_config.cache_key_tag
            if version not in previous.split("|"):
                compiler_config.cache_key_tag = (
                    f"{previous}|{version}" if previous else version
                )

            # Automatic dynamic promotion gives a cold resume a static Flex
            # kernel while a continuous run uses a dynamic one. The NPU masked
            # and divisible paths have different floating-point rounding.
            # Compile symbolically from the first call so identical shapes
            # always follow the same path, including their backward kernels.
            FlexInnerAttention._compiled_flex_attn = torch.compile(
                flex_attention,
                fullgraph=True,
                options=FlexInnerAttention.inductor_configs,
                recompile_limit=16,
                dynamic=True,
            )
        return result

    return set_determinism
