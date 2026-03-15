from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_tensorrt_cu12_libs_declares_wheel_stub_extra_build_dependency():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra_build_deps = pyproject["tool"]["uv"].get("extra-build-dependencies", {})

    assert extra_build_deps["tensorrt-cu12-libs"] == ["wheel_stub"]


def test_transformers_declares_setuptools_and_wheel_extra_build_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra_build_deps = pyproject["tool"]["uv"].get("extra-build-dependencies", {})

    assert extra_build_deps["transformers"] == ["setuptools", "wheel"]


def test_pysrt_declares_setuptools_and_wheel_extra_build_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra_build_deps = pyproject["tool"]["uv"].get("extra-build-dependencies", {})

    assert extra_build_deps["pysrt"] == ["setuptools", "wheel"]


def test_bitmath_declares_setuptools_and_wheel_extra_build_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra_build_deps = pyproject["tool"]["uv"].get("extra-build-dependencies", {})

    assert extra_build_deps["bitmath"] == ["setuptools", "wheel"]
