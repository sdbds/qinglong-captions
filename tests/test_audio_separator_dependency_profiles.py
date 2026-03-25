from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_onnx_base_depends_on_torch_base():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    onnx_base_deps = pyproject["project"]["optional-dependencies"]["onnx-base"]

    assert "qinglong-captions[torch-base]" in onnx_base_deps
    assert any(dep.startswith("onnxruntime-gpu") for dep in onnx_base_deps)


def test_uv_extra_indexes_cover_windows_onnx_cuda13_and_torch_cu130():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra_indexes = pyproject["tool"]["uv"]["extra-index-url"]

    assert (
        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/"
        in extra_indexes
    )
    assert "https://download.pytorch.org/whl/cu130" in extra_indexes


def test_vocal_midi_depends_on_onnx_base():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    vocal_midi_deps = pyproject["project"]["optional-dependencies"]["vocal-midi"]

    assert "qinglong-captions[onnx-base]" in vocal_midi_deps
    assert any(dep.startswith("mido") for dep in vocal_midi_deps)
    assert any(dep.startswith("numpy") for dep in vocal_midi_deps)
