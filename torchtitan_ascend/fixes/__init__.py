# SPDX-License-Identifier: Apache-2.0
"""Runtime function patches; importing this package does not apply fixes."""

from torchtitan_ascend.fixes.registry import apply_all, fix

# Importing fix modules registers them; apply_all() installs the fixes.
from torchtitan_ascend.fixes import vocab_parallel_loss  # noqa: F401

__all__ = ["apply_all", "fix"]
