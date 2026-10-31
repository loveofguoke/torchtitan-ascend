# SPDX-License-Identifier: Apache-2.0
"""Split the complex RoPE cache before gather on NPUs without complex64 indexing.

Ascend 910B's aclnn index rejects DT_COMPLEX64, so ``rope_cache[positions]`` in
``_reshape_for_broadcast`` fails for ComplexRoPE. Index the real and imaginary
parts separately and rebuild the complex tensor; numerically identical.
"""

import torch

from torchtitan_ascend.patches import patch


@patch(
    target="torchtitan.models.common.rope:_reshape_for_broadcast",
    kind="wrap",
    deferred=True,
)
def reshape(original):
    def _reshape_for_broadcast(rope_cache, query_shape, positions=None):
        if rope_cache.is_complex():
            real = original(rope_cache.real.contiguous(), query_shape, positions)
            imag = original(rope_cache.imag.contiguous(), query_shape, positions)
            return torch.complex(real, imag)
        return original(rope_cache, query_shape, positions)

    return _reshape_for_broadcast
