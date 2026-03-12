"""Vision API Providers.

Providers are auto-discovered by ProviderRegistry.
Submodules stay lazy-loaded so compatibility patch paths keep working.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any


def _submodule_names() -> list[str]:
    return sorted(name for _, name, _ in pkgutil.iter_modules(__path__))


def __getattr__(name: str) -> Any:
    if name not in _submodule_names():
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_submodule_names()))


__all__ = _submodule_names()
