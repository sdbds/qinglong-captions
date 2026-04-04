from __future__ import annotations

import importlib
import sys
from pathlib import Path


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
CONFLICT_PREFIXES = ("modules", "utils", "annotators")


def ensure_vendor_imports() -> Path:
    vendor_root_str = str(VENDOR_ROOT)
    if vendor_root_str in sys.path:
        sys.path.remove(vendor_root_str)
    sys.path.insert(0, vendor_root_str)

    for prefix in CONFLICT_PREFIXES:
        for name in [key for key in list(sys.modules) if key == prefix or key.startswith(prefix + ".")]:
            del sys.modules[name]

    importlib.invalidate_caches()
    return VENDOR_ROOT

