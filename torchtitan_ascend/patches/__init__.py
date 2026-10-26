# SPDX-License-Identifier: Apache-2.0
"""Runtime function patches; importing this package does not apply patches."""

from torchtitan_ascend.patches.registry import apply_all, patch

__all__ = ["apply_all", "patch"]
