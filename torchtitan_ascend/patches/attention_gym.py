# SPDX-License-Identifier: Apache-2.0
"""Preserve TorchTitan placeholders for Attention Gym's lazy CUDA imports.

Missing CUDA dependencies during attribute lookup produce backend placeholders.
Unrelated import errors propagate to the caller.
"""

import sys
from functools import wraps

from torchtitan_ascend.patches import patch


@patch(target="torchtitan.models.common.attn_gym:__getattr__", deferred=True)
def lazy_backends(original):
    module = sys.modules[original.__module__]

    @wraps(original)
    def resolve(name):
        try:
            return original(name)
        except ImportError as error:
            cause: BaseException = error
            while cause.__cause__ is not None:
                cause = cause.__cause__
            if not isinstance(cause, ModuleNotFoundError) or (cause.name or "").split(
                "."
            )[0] not in {
                "cutlass",
                "cuda",
                "cudnn",
                "nvidia",
            }:
                raise
            source = module._EXPORTS[name]
            module._BACKEND_ERRORS.setdefault(source, cause)
            value = module._missing_backend_kernel(source)
            setattr(module, name, value)
            return value

    return resolve
