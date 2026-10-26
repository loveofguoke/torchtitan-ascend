# SPDX-License-Identifier: Apache-2.0
"""Apply deferred patches before an imported module reaches its callers."""

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable


class _Loader(importlib.abc.Loader):
    def __init__(self, loader, callbacks):
        self.loader = loader
        self.callbacks = callbacks

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        for callback in self.callbacks:
            callback()
        _finder.pending.pop(module.__name__)
        if not _finder.pending:
            sys.meta_path.remove(_finder)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.pending: dict[str, list[Callable[[], None]]] = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.pending:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, self.pending[fullname])
        return spec


_finder = _Finder()


def defer(module_name, callback):
    callbacks = _finder.pending.setdefault(module_name, [])
    if callback not in callbacks:
        callbacks.append(callback)
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
