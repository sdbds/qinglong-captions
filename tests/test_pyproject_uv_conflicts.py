from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def _load_conflicts():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["uv"]["conflicts"]


def _load_optional_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["optional-dependencies"]


def _load_pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _load_extra_conflict_sets():
    sets = []
    for group in _load_conflicts():
        extras = {
            item["extra"]
            for item in group
            if isinstance(item, dict) and "extra" in item and "package" not in item
        }
        if extras:
            sets.append(extras)
    return sets


def _has_extra_conflict(*extras: str) -> bool:
    expected = set(extras)
    return any(expected <= conflict_set for conflict_set in _load_extra_conflict_sets())


def test_deepseek_ocr_conflicts_with_lfm_vl_local():
    assert _has_extra_conflict("deepseek-ocr", "lfm-vl-local")


def test_penguin_vl_local_conflicts_with_lfm_vl_local():
    assert _has_extra_conflict("penguin-vl-local", "lfm-vl-local")


def test_paddleocr_conflicts_with_lfm_vl_local():
    assert _has_extra_conflict("paddleocr-native", "lfm-vl-local")


def test_paddleocr_conflicts_with_logics_ocr():
    assert _has_extra_conflict("paddleocr-native", "logics-ocr")


def test_paddleocr_conflicts_with_wdtagger_cl_tagger_v2():
    assert _has_extra_conflict("paddleocr-native", "wdtagger-cl-tagger-v2")


def test_paddleocr_conflicts_with_all_torch_stack_extras():
    torch_extras = [
        name
        for name, deps in _load_optional_dependencies().items()
        if name not in {"paddleocr", "paddleocr-onnx", "paddleocr-native"}
        and (
            any(dep.startswith(("torch", "torchvision", "torchaudio")) for dep in deps)
            or "qinglong-captions[torch-base]" in deps
            or "qinglong-captions[onnx-base]" in deps
        )
    ]

    for extra in torch_extras:
        assert _has_extra_conflict("paddleocr-native", extra)


def test_lighton_ocr_conflicts_with_translate():
    assert _has_extra_conflict("lighton-ocr", "translate")


def test_translate_extra_uses_hy_mt2_dependency_stack():
    translate_deps = _load_optional_dependencies()["translate"]

    assert "transformers[serving]>=5.6.0" in translate_deps
    assert "compressed-tensors>=0.14.0" in translate_deps


def test_lighton_ocr_conflicts_with_paddleocr():
    assert _has_extra_conflict("lighton-ocr", "paddleocr-native")


def test_paddleocr_onnx_extra_uses_onnx_base_without_native_paddle_wheels():
    deps = _load_optional_dependencies()["paddleocr-onnx"]

    assert "qinglong-captions[onnx-base]" in deps
    assert any(dep.startswith("paddleocr>=3.7.0") for dep in deps)
    assert not any("paddlepaddle" in dep for dep in deps)


def test_paddleocr_native_extra_keeps_legacy_paddle_gpu_stack():
    deps = _load_optional_dependencies()["paddleocr-native"]

    assert any("paddlepaddle-gpu" in dep for dep in deps)
    assert any(dep.startswith("paddleocr[doc-parser]") for dep in deps)


def test_dots_ocr_conflicts_with_known_transformers_incompatible_extras():
    expected_pairs = [
        ("dots-ocr", "translate"),
        ("dots-ocr", "deepseek-ocr"),
        ("dots-ocr", "penguin-vl-local"),
        ("dots-ocr", "lighton-ocr"),
        ("dots-ocr", "hunyuan-ocr"),
        ("dots-ocr", "glm-ocr"),
        ("dots-ocr", "qwen-vl-local"),
        ("dots-ocr", "music-flamingo-local"),
    ]

    for pair in expected_pairs:
        assert _has_extra_conflict(*pair)


def test_wdtagger_no_longer_conflicts_with_transformers_only_profiles():
    assert not _has_extra_conflict("wdtagger", "translate")
    assert not _has_extra_conflict("wdtagger", "dots-ocr")
    assert not _has_extra_conflict("wdtagger", "olmocr")
    assert not _has_extra_conflict("wdtagger", "music-flamingo-local")


def test_logics_ocr_conflicts_with_known_transformers_incompatible_extras():
    expected_pairs = [
        ("logics-ocr", "translate"),
        ("logics-ocr", "deepseek-ocr"),
        ("logics-ocr", "dots-ocr"),
        ("logics-ocr", "penguin-vl-local"),
        ("logics-ocr", "music-flamingo-local"),
    ]

    for pair in expected_pairs:
        assert _has_extra_conflict(*pair)


def test_eureka_audio_conflicts_with_known_transformers_incompatible_extras():
    expected_pairs = [
        ("eureka-audio-local", "music-flamingo-local"),
        ("eureka-audio-local", "translate"),
        ("eureka-audio-local", "deepseek-ocr"),
        ("eureka-audio-local", "dots-ocr"),
        ("eureka-audio-local", "penguin-vl-local"),
        ("eureka-audio-local", "qwen-vl-local"),
        ("eureka-audio-local", "hunyuan-ocr"),
        ("eureka-audio-local", "glm-ocr"),
    ]

    for pair in expected_pairs:
        assert _has_extra_conflict(*pair)


def test_acestep_transcriber_conflicts_with_translate():
    assert _has_extra_conflict("acestep-transcriber-local", "translate")


def test_cohere_transcribe_conflicts_with_translate():
    assert _has_extra_conflict("cohere-transcribe-local", "translate")


def test_marlin_2b_local_conflicts_with_known_transformers_incompatible_extras():
    expected_pairs = [
        ("marlin-2b-local", "translate"),
        ("marlin-2b-local", "deepseek-ocr"),
        ("marlin-2b-local", "dots-ocr"),
        ("marlin-2b-local", "penguin-vl-local"),
        ("marlin-2b-local", "qwen-vl-local"),
        ("marlin-2b-local", "eureka-audio-local"),
        ("marlin-2b-local", "cohere-transcribe-local"),
        ("marlin-2b-local", "music-flamingo-local"),
    ]

    for pair in expected_pairs:
        assert _has_extra_conflict(*pair)


def test_infinity_parser2_ocr_conflicts_with_known_transformers_incompatible_extras():
    expected_pairs = [
        ("infinity-parser2-ocr", "paddleocr-native"),
        ("infinity-parser2-ocr", "deepseek-ocr"),
        ("infinity-parser2-ocr", "dots-ocr"),
        ("infinity-parser2-ocr", "penguin-vl-local"),
        ("infinity-parser2-ocr", "eureka-audio-local"),
        ("infinity-parser2-ocr", "cohere-transcribe-local"),
        ("infinity-parser2-ocr", "reward-model"),
        ("infinity-parser2-ocr", "music-flamingo-local"),
    ]

    for pair in expected_pairs:
        assert _has_extra_conflict(*pair)


def test_unlimited_ocr_conflicts_with_known_transformers_incompatible_extras():
    expected_pairs = [
        ("unlimited-ocr", "paddleocr-native"),
        ("unlimited-ocr", "deepseek-ocr"),
        ("unlimited-ocr", "dots-ocr"),
        ("unlimited-ocr", "penguin-vl-local"),
        ("unlimited-ocr", "eureka-audio-local"),
        ("unlimited-ocr", "cohere-transcribe-local"),
        ("unlimited-ocr", "reward-model"),
        ("unlimited-ocr", "music-flamingo-local"),
    ]

    for pair in expected_pairs:
        assert _has_extra_conflict(*pair)


def test_flash_attn_windows_url_is_centralized_in_uv_sources():
    pyproject = _load_pyproject()
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    optional_dependencies = pyproject["project"]["optional-dependencies"]
    flash_attn_sources = pyproject["tool"]["uv"]["sources"]["flash-attn"]

    assert len(flash_attn_sources) == 1
    source = flash_attn_sources[0]
    assert source["marker"] == "sys_platform == 'win32'"
    assert source["url"].startswith("https://github.com/sdbds/flash-attention-for-windows/releases/download/")
    assert source["url"].endswith(".whl")
    assert pyproject_text.count(source["url"]) == 1
    assert pyproject_text.count("https://github.com/sdbds/flash-attention-for-windows/releases/download/") == 1

    direct_flash_attn_dependencies = [
        dep
        for deps in optional_dependencies.values()
        for dep in deps
        if dep.startswith("flash-attn @")
    ]

    assert direct_flash_attn_dependencies == []


def test_torch_213_cuda_flash_attention_wheel_is_pinned_for_windows():
    pyproject = _load_pyproject()
    torch_base = pyproject["project"]["optional-dependencies"]["torch-base"]
    sources = pyproject["tool"]["uv"]["sources"]

    assert "torch==2.13.0" in torch_base
    assert "torch==2.12.1" not in torch_base
    assert not any(dep.startswith("sageattention") for dep in torch_base)
    assert "sageattention" not in sources

    flash_url = sources["flash-attn"][0]["url"]
    assert "cu130torch2.13.0" in flash_url
