from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pytest_registers_all_execution_class_markers():
    markers = _pyproject()["tool"]["pytest"]["ini_options"]["markers"]

    assert any(marker.startswith("compat:") for marker in markers)
    assert any(marker.startswith("optional_runtime:") for marker in markers)
    assert any(marker.startswith("gpu:") for marker in markers)
    assert any(marker.startswith("network:") for marker in markers)


def test_test_dependency_group_declares_ruff():
    dependencies = _pyproject()["dependency-groups"]["test"]

    assert any(dependency == "ruff" or dependency.startswith("ruff==") for dependency in dependencies)
