"""Compatibility shim for the legacy top-level ``providers`` package.

The canonical implementation now lives under ``module.providers``.
This shim keeps old imports working while preserving module identity so
``providers.base`` and ``module.providers.base`` resolve to the same object.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys


_ALIAS_PREFIX = "providers."
_TARGET_PREFIX = "module.providers."


class _ProvidersAliasLoader(importlib.abc.Loader):
    def __init__(self, alias_name: str, target_name: str):
        self.alias_name = alias_name
        self.target_name = target_name

    def create_module(self, spec):
        module = importlib.import_module(self.target_name)
        _bind_alias(self.alias_name, module)
        return module

    def exec_module(self, module):
        return None


class _ProvidersAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_ALIAS_PREFIX):
            return None

        target_name = _TARGET_PREFIX + fullname[len(_ALIAS_PREFIX) :]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None

        spec = importlib.util.spec_from_loader(
            fullname,
            _ProvidersAliasLoader(fullname, target_name),
            origin=target_spec.origin,
            is_package=target_spec.submodule_search_locations is not None,
        )
        if spec is not None and target_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = target_spec.submodule_search_locations
        return spec


def _install_alias_finder() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _ProvidersAliasFinder):
            return
    sys.meta_path.insert(0, _ProvidersAliasFinder())


def _alias_loaded_modules() -> None:
    for name, module in list(sys.modules.items()):
        if name.startswith(_TARGET_PREFIX):
            alias = _ALIAS_PREFIX + name[len(_TARGET_PREFIX) :]
            _bind_alias(alias, module)


def _bind_alias(alias_name: str, module) -> None:
    sys.modules[alias_name] = module
    parent_name, _, child_name = alias_name.rpartition(".")
    if not parent_name:
        return
    parent_module = sys.modules.get(parent_name)
    if parent_module is not None:
        setattr(parent_module, child_name, module)


_install_alias_finder()

_impl = importlib.import_module("module.providers")
_alias_loaded_modules()

__all__ = getattr(_impl, "__all__", [])
__doc__ = _impl.__doc__
__file__ = _impl.__file__
__path__ = _impl.__path__

for export_name in __all__:
    globals()[export_name] = getattr(_impl, export_name)
