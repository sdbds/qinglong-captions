import ast
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = ROOT / "module" / "see_through" / "extracted"
VENDOR_DIR = ROOT / "module" / "see_through" / "vendor"
VENDOR_UTILS_DIR = ROOT / "module" / "see_through" / "vendor" / "utils"
LOCAL_UTILS_DIR = ROOT / "utils"
VENDOR_PREFIX = "module.see_through.vendor"
ALLOWED_LOCAL_UTILS = {"utils.transformer_loader"}
TEST_FILES = [
    ROOT / "tests" / "test_see_through_layerdiff_core.py",
    ROOT / "tests" / "test_see_through_marigold_core.py",
]


def _parse_top_level_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            symbols.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def _build_symbol_table() -> dict[str, set[str]]:
    symbol_table: dict[str, set[str]] = {}
    for path in VENDOR_DIR.rglob("*.py"):
        relative_module = path.relative_to(VENDOR_DIR).with_suffix("")
        parts = list(relative_module.parts)
        if parts[-1] == "__init__":
            parts.pop()
        module_name = ".".join([VENDOR_PREFIX, *parts])
        symbol_table[module_name] = _parse_top_level_symbols(path)
    for path in LOCAL_UTILS_DIR.glob("*.py"):
        module_name = f"utils.{path.stem}"
        if module_name in ALLOWED_LOCAL_UTILS:
            symbol_table[module_name] = _parse_top_level_symbols(path)
    return symbol_table


def _collect_extracted_surface_imports() -> dict[Path, set[tuple[str, str | None]]]:
    imports_by_file: dict[Path, set[tuple[str, str | None]]] = {}
    for path in EXTRACTED_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: set[tuple[str, str | None]] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module.startswith(f"{VENDOR_PREFIX}.") or node.module in ALLOWED_LOCAL_UTILS)
            ):
                for alias in node.names:
                    imports.add((node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(f"{VENDOR_PREFIX}.") or alias.name in ALLOWED_LOCAL_UTILS:
                        imports.add((alias.name, None))
        imports_by_file[path] = imports
    return imports_by_file


def _collect_fake_utils_attrs(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fake_modules: dict[str, str] = {}
    attrs_by_module: dict[str, set[str]] = defaultdict(set)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
                call = node.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "ModuleType"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    fake_modules[target_name] = call.args[0].value

            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in fake_modules
                ):
                    module_name = fake_modules[target.value.id]
                    if module_name.startswith(f"{VENDOR_PREFIX}.utils."):
                        attrs_by_module[module_name].add(target.attr)

    return attrs_by_module


def test_extracted_vendor_imports_match_available_surfaces():
    symbol_table = _build_symbol_table()
    imports_by_file = _collect_extracted_surface_imports()
    failures: list[str] = []

    for path, imports in sorted(imports_by_file.items()):
        rel_path = path.relative_to(ROOT)
        for module_name, symbol_name in sorted(imports):
            if symbol_name is None:
                if module_name not in symbol_table:
                    failures.append(f"{rel_path}: module {module_name} is not available")
                continue
            if symbol_name not in symbol_table.get(module_name, set()):
                failures.append(f"{rel_path}: {module_name}.{symbol_name} is not defined")

    assert not failures, "\n".join(failures)


def test_vendor_and_extracted_sources_do_not_import_top_level_conflict_packages():
    failures: list[str] = []
    for source_dir in (VENDOR_DIR, EXTRACTED_DIR):
        for path in source_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported_modules: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imported_modules.append(node.module)
                elif isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)

                for module_name in imported_modules:
                    root_name = module_name.partition(".")[0]
                    if root_name in {"modules", "annotators"} or (
                        root_name == "utils" and module_name not in ALLOWED_LOCAL_UTILS
                    ):
                        rel_path = path.relative_to(ROOT)
                        failures.append(f"{rel_path}:{node.lineno}: {module_name}")

    assert not failures, "Top-level conflicting imports found:\n" + "\n".join(failures)


def test_vendor_bootstrap_is_removed_after_namespace_migration():
    assert not (ROOT / "module" / "see_through" / "vendor_bootstrap.py").exists()


def test_unbundled_lama_inpainter_raises_clear_optional_component_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.delitem(sys.modules, "module.see_through.vendor.utils.torchcv", raising=False)
    from module.see_through.vendor.utils.torchcv import cluster_inpaint_part

    fake_sklearn = types.ModuleType("sklearn")
    fake_sklearn.__path__ = []
    fake_cluster = types.ModuleType("sklearn.cluster")

    class FakeKMeans:
        cluster_centers_ = np.array([[0.0], [1.0]], dtype=np.float32)

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, samples):
            return self

        def predict(self, samples):
            return (np.asarray(samples).reshape(-1) >= 0.5).astype(np.int64)

    fake_cluster.DBSCAN = object
    fake_cluster.HDBSCAN = object
    fake_cluster.MeanShift = object
    fake_cluster.KMeans = FakeKMeans
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_cluster)

    depth = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[..., -1] = 255

    with pytest.raises(RuntimeError, match="optional See-Through LaMa inpainter"):
        cluster_inpaint_part(depth, mask, image, inpaint="lama")


def test_core_tests_only_fake_real_vendor_symbols():
    symbol_table = _build_symbol_table()
    failures: list[str] = []

    for path in TEST_FILES:
        attrs_by_module = _collect_fake_utils_attrs(path)
        rel_path = path.relative_to(ROOT)
        for module_name, attrs in sorted(attrs_by_module.items()):
            for attr_name in sorted(attrs):
                if attr_name not in symbol_table.get(module_name, set()):
                    failures.append(f"{rel_path}: fake {module_name}.{attr_name} does not exist in real module")

    assert not failures, "\n".join(failures)
