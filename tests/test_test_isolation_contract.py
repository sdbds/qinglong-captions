from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"


def _ordinary_test_files():
    return sorted(TESTS.glob("test_*.py")) + [TESTS / "provider_v2_helpers.py"]


def _module_level_statements(tree: ast.Module):
    for statement in tree.body:
        yield statement
        if isinstance(statement, (ast.If, ast.Try)):
            yield from statement.body
            yield from statement.orelse


def _is_sys_modules_mutation(node: ast.AST) -> bool:
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "sys"
            and target.value.attr == "modules"
            for target in targets
        )
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "setdefault"
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
        and func.value.attr == "modules"
    )


def _is_sys_path_insert(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "insert"
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
        and func.value.attr == "path"
    )


def test_ordinary_tests_do_not_mutate_import_state_at_module_scope():
    violations = []
    for path in _ordinary_test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _module_level_statements(tree):
            if _is_sys_modules_mutation(node) or _is_sys_path_insert(node):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert violations == []


def _run_ordered_pytest(*test_paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *test_paths, "-q", "--strict-markers"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )


def test_preprocess_then_provider_runtime_groups_are_order_independent():
    result = _run_ordered_pytest(
        "tests/test_preprocess_resize_optimization.py",
        "tests/test_preprocess_datasets_alignment.py",
        "tests/test_unlimited_ocr_provider.py",
        "tests/test_runtime_backends.py",
    )

    assert result.returncode == 0, result.stdout


def test_provider_runtime_then_preprocess_groups_are_order_independent():
    result = _run_ordered_pytest(
        "tests/test_unlimited_ocr_provider.py",
        "tests/test_runtime_backends.py",
        "tests/test_preprocess_resize_optimization.py",
        "tests/test_preprocess_datasets_alignment.py",
    )

    assert result.returncode == 0, result.stdout
