# SPDX-License-Identifier: Apache-2.0
"""Register rank-local causal convolution for SPMD type checking."""

from torchtitan_ascend.patches import patch


@patch(
    target="torchtitan_ascend.models.common.linear_attention:_CausalConv1d",
    deferred=True,
)
def apply(original):
    import spmd_types as spmd

    spmd.register_local_autograd_function(original)
    return original
