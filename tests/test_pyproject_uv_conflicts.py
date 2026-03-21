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
    assert _has_extra_conflict("paddleocr", "lfm-vl-local")


def test_paddleocr_conflicts_with_logics_ocr():
    assert _has_extra_conflict("paddleocr", "logics-ocr")


def test_paddleocr_conflicts_with_all_torch_stack_extras():
    torch_extras = [
        name
        for name, deps in _load_optional_dependencies().items()
        if name != "paddleocr" and any(dep.startswith(("torch", "torchvision", "torchaudio")) for dep in deps)
    ]

    for extra in torch_extras:
        assert _has_extra_conflict("paddleocr", extra)


def test_lighton_ocr_conflicts_with_translate():
    assert _has_extra_conflict("lighton-ocr", "translate")


def test_lighton_ocr_conflicts_with_paddleocr():
    assert _has_extra_conflict("lighton-ocr", "paddleocr")


def test_dots_ocr_conflicts_with_known_transformers_incompatible_extras():
    expected_pairs = [
        ("dots-ocr", "translate"),
        ("dots-ocr", "wdtagger"),
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
