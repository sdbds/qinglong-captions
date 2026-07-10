from __future__ import annotations

import importlib
import sys

import pytest


_CANONICAL_UTILS_SUBMODULES = {
    name: importlib.import_module(f"utils.{name}")
    for name in ("transformer_loader", "wdtagger_opencv")
}


def _restore_canonical_submodules() -> None:
    package = sys.modules.get("utils")
    if package is None:
        return
    for child_name, canonical_module in _CANONICAL_UTILS_SUBMODULES.items():
        module_name = f"utils.{child_name}"
        sys.modules[module_name] = canonical_module
        setattr(package, child_name, canonical_module)


@pytest.fixture(autouse=True)
def _restore_package_submodule_identity():
    """Repair package attributes after tests temporarily replace submodules."""
    _restore_canonical_submodules()
    yield
    _restore_canonical_submodules()
