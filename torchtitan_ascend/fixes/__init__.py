"""Runtime function patches; importing this package does not apply fixes."""

from torchtitan_ascend.fixes.registry import apply_all, fix

__all__ = ["apply_all", "fix"]
