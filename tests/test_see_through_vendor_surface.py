import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = ROOT / "module" / "see_through" / "extracted"
VENDOR_UTILS_DIR = ROOT / "module" / "see_through" / "vendor" / "utils"
LOCAL_UTILS_DIR = ROOT / "utils"
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
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def _build_symbol_table() -> dict[str, set[str]]:
    symbol_table: dict[str, set[str]] = {}
    for path in VENDOR_UTILS_DIR.glob("*.py"):
        symbol_table[f"utils.{path.stem}"] = _parse_top_level_symbols(path)
    for path in LOCAL_UTILS_DIR.glob("*.py"):
        module_name = f"utils.{path.stem}"
        if module_name in ALLOWED_LOCAL_UTILS:
            symbol_table[module_name] = _parse_top_level_symbols(path)
    return symbol_table


def _collect_extracted_utils_imports() -> dict[Path, set[tuple[str, str | None]]]:
    imports_by_file: dict[Path, set[tuple[str, str | None]]] = {}
    for path in EXTRACTED_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: set[tuple[str, str | None]] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("utils."):
                for alias in node.names:
                    imports.add((node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("utils."):
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
                    if module_name.startswith("utils."):
                        attrs_by_module[module_name].add(target.attr)

    return attrs_by_module


def test_extracted_utils_imports_match_available_surfaces():
    symbol_table = _build_symbol_table()
    imports_by_file = _collect_extracted_utils_imports()
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
