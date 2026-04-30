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
    assert "tensorrt-cu13==10.16.1.11" in onnx_base_deps
    assert not any(dep.startswith("tensorrt==") for dep in onnx_base_deps)
    assert not any(dep.startswith("tensorrt-cu13-libs==") for dep in onnx_base_deps)


def test_tensorrt_is_not_a_separate_wdtagger_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "wdtagger-tensorrt" not in optional_deps


def test_uv_extra_indexes_cover_windows_onnx_cuda13_and_torch_cu130():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra_index_urls = {index["url"] for index in pyproject["tool"]["uv"]["index"]}

    assert (
        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/"
        in extra_index_urls
    )
    assert "https://download.pytorch.org/whl/cu130" in extra_index_urls


def test_vocal_midi_depends_on_onnx_base():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    vocal_midi_deps = pyproject["project"]["optional-dependencies"]["vocal-midi"]

    assert "qinglong-captions[onnx-base]" in vocal_midi_deps
    assert any(dep.startswith("mido") for dep in vocal_midi_deps)
    assert any(dep.startswith("numpy") for dep in vocal_midi_deps)


def test_see_through_extra_includes_bitsandbytes_for_nf4_repos():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    see_through_deps = pyproject["project"]["optional-dependencies"]["see-through"]

    assert any(dep.startswith("bitsandbytes") for dep in see_through_deps)
