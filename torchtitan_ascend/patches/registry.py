# SPDX-License-Identifier: Apache-2.0
"""Register and apply function patches without importing their targets at registration."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

_REGISTRY: dict[str, Callable[[], None]] = {}
_MISSING = object()


def patch(
    *, target: str, kind: Literal["wrap", "replace", "polyfill"] = "wrap", deferred: bool = False
) -> Callable[[Callable[[Any], Any]], Callable[[], None]]:
    """Register a factory and return its zero-argument application function.

    ``target`` is ``module:attribute`` or ``module:Class.method``. The factory
    receives the original function, or ``None`` for a missing polyfill target.
    Each target has one registered patch. Static and class methods retain binding.
    """
    module_name, separator, attribute = target.partition(":")
    if not separator or not all(part.isidentifier() for part in (*module_name.split("."), *attribute.split("."))):
        raise ValueError(f"Invalid patch target {target!r}; expected 'module:attribute'")
    if kind not in ("wrap", "replace", "polyfill"):
        raise ValueError(f"Unknown patch kind: {kind!r}")

    def register(factory: Callable[[Any], Any]) -> Callable[[], None]:
        if target in _REGISTRY:
            raise ValueError(f"A patch is already registered for {target}")
        installed: Any = _MISSING

        def install() -> None:
            nonlocal installed
            owner = importlib.import_module(module_name)
            parts = attribute.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part)
            name = parts[-1]
            original = inspect.getattr_static(owner, name, _MISSING)
            if original is not _MISSING:
                if kind == "polyfill" or original is installed:
                    return
            elif kind != "polyfill":
                raise AttributeError(f"Patch target {target!r} does not exist")

            descriptor = type(original) if isinstance(original, staticmethod | classmethod) else None
            function = original.__func__ if descriptor else original
            if function is not _MISSING and not callable(function):
                raise TypeError(f"Patch target {target!r} is not callable")
            patched = factory(None if function is _MISSING else function)
            if not callable(patched):
                raise TypeError(f"Patch factory for {target!r} must return a callable")
            if descriptor:
                patched = descriptor(patched)
            setattr(owner, name, patched)
            installed = patched

        @wraps(factory)
        def apply() -> None:
            if deferred and module_name not in sys.modules:
                from ._imports import defer

                defer(module_name, install)
            else:
                install()

        _REGISTRY[target] = apply
        return apply

    return register


def apply_all() -> None:
    """Discover public patch modules, then install or defer their patches."""
    from . import __path__

    for module in sorted(pkgutil.iter_modules(__path__), key=lambda module: module.name):
        if not module.name.startswith("_") and module.name != "registry":
            importlib.import_module(f"{__package__}.{module.name}")
    for apply in _REGISTRY.values():
        apply()
