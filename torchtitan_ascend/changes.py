"""Activate TorchTitan overrides using references to their factory functions."""

import inspect
from collections.abc import Callable

from torchtitan.config.override import OverrideImport
from torchtitan.trainer import Trainer


def override_target(override_fn: Callable) -> str:
    """Return the factory's ``module.function`` activation target."""
    if (
        not inspect.isfunction(override_fn)
        or override_fn.__qualname__ != override_fn.__name__
        or not override_fn.__name__.isidentifier()
    ):
        raise TypeError("An override factory must be a module-level function")
    if not override_fn.__module__ or override_fn.__module__ == "__main__":
        raise ValueError("An override factory must have an importable module and name")
    return f"{override_fn.__module__}.{override_fn.__name__}"


def _entry_target(entry: OverrideImport) -> str:
    return entry if isinstance(entry, str) else entry[0]


def has_override(config: Trainer.Config, override_fn: Callable) -> bool:
    """Check whether the factory's target is listed in ``override.imports``."""
    target = override_target(override_fn)
    return any(_entry_target(entry) == target for entry in config.override.imports)


def add_override(config: Trainer.Config, override_fn: Callable) -> bool:
    """Activate an override once, preserving any existing per-entry kwargs."""
    if has_override(config, override_fn):
        return False
    config.override.imports = [*config.override.imports, override_target(override_fn)]
    return True


def swap_override(config: Trainer.Config, *, remove: Callable, add: Callable) -> bool:
    """Swap activation targets, preserving an active replacement's kwargs."""
    if not has_override(config, remove):
        return False
    old_target, new_target = override_target(remove), override_target(add)
    if old_target == new_target:
        return False
    config.override.imports = [
        entry for entry in config.override.imports if _entry_target(entry) != old_target
    ]
    if not has_override(config, add):
        config.override.imports = [*config.override.imports, new_target]
    return True
