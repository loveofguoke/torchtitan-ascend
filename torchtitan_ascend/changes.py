"""Activate TorchTitan overrides using references to their factory functions."""

import inspect
from collections.abc import Callable


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
