# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU extensions for TorchTitan.

Call ``setup()`` before entering TorchTitan to apply registered runtime patches.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__", "setup"]


def setup(*, apply_patches: bool = True):
    """Apply registered runtime patches once, before entering TorchTitan."""
    from torchtitan_ascend._setup import setup as _setup

    return _setup(apply_patches=apply_patches)
