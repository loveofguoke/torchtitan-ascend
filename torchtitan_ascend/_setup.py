# SPDX-License-Identifier: Apache-2.0
"""Apply registered runtime patches before entering TorchTitan."""

from __future__ import annotations

_configured: bool | None = None


def setup(*, apply_patches: bool = True) -> None:
    """Initialize once; repeated calls must use the same options."""
    global _configured
    if _configured is not None:
        if _configured != apply_patches:
            raise RuntimeError("setup() was already called with a different apply_patches value")
        return

    if apply_patches:
        from torchtitan_ascend.patches import apply_all

        apply_all()
    _configured = apply_patches
