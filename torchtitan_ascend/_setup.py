# SPDX-License-Identifier: Apache-2.0
"""Apply registered runtime fixes before entering TorchTitan."""

from __future__ import annotations

_configured: bool | None = None


def setup(*, apply_fixes: bool = True) -> None:
    """Initialize once; repeated calls must use the same options."""
    global _configured
    if _configured is not None:
        if _configured != apply_fixes:
            raise RuntimeError("setup() was already called with a different apply_fixes value")
        return

    if apply_fixes:
        from torchtitan_ascend.fixes import apply_all

        apply_all()
    _configured = apply_fixes
