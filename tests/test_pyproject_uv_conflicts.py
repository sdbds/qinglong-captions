from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def _load_conflicts():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["uv"]["conflicts"]


def test_deepseek_ocr_conflicts_with_lfm_vl_local():
    conflicts = _load_conflicts()

    assert [
        {"extra": "deepseek-ocr"},
        {"extra": "lfm-vl-local"},
    ] in conflicts or [
        {"extra": "lfm-vl-local"},
        {"extra": "deepseek-ocr"},
    ] in conflicts


def test_penguin_vl_local_conflicts_with_lfm_vl_local():
    conflicts = _load_conflicts()

    assert [
        {"extra": "penguin-vl-local"},
        {"extra": "lfm-vl-local"},
    ] in conflicts or [
        {"extra": "lfm-vl-local"},
        {"extra": "penguin-vl-local"},
    ] in conflicts


def test_paddleocr_conflicts_with_lfm_vl_local():
    conflicts = _load_conflicts()

    assert [
        {"extra": "paddleocr"},
        {"extra": "lfm-vl-local"},
    ] in conflicts or [
        {"extra": "lfm-vl-local"},
        {"extra": "paddleocr"},
    ] in conflicts


def test_lighton_ocr_conflicts_with_translate():
    conflicts = _load_conflicts()

    assert [
        {"extra": "lighton-ocr"},
        {"extra": "translate"},
    ] in conflicts or [
        {"extra": "translate"},
        {"extra": "lighton-ocr"},
    ] in conflicts


def test_lighton_ocr_conflicts_with_paddleocr():
    conflicts = _load_conflicts()

    assert [
        {"extra": "lighton-ocr"},
        {"extra": "paddleocr"},
    ] in conflicts or [
        {"extra": "paddleocr"},
        {"extra": "lighton-ocr"},
    ] in conflicts


def test_dots_ocr_conflicts_with_known_transformers_incompatible_extras():
    conflicts = _load_conflicts()
    expected_pairs = [
        [{"extra": "dots-ocr"}, {"extra": "translate"}],
        [{"extra": "dots-ocr"}, {"extra": "wdtagger"}],
        [{"extra": "dots-ocr"}, {"extra": "deepseek-ocr"}],
        [{"extra": "dots-ocr"}, {"extra": "penguin-vl-local"}],
        [{"extra": "dots-ocr"}, {"extra": "lighton-ocr"}],
        [{"extra": "dots-ocr"}, {"extra": "hunyuan-ocr"}],
        [{"extra": "dots-ocr"}, {"extra": "glm-ocr"}],
        [{"extra": "dots-ocr"}, {"extra": "qwen-vl-local"}],
        [{"extra": "dots-ocr"}, {"extra": "music-flamingo-local"}],
    ]

    for pair in expected_pairs:
        reversed_pair = list(reversed(pair))
        assert pair in conflicts or reversed_pair in conflicts


def test_logics_ocr_conflicts_with_known_transformers_incompatible_extras():
    conflicts = _load_conflicts()
    expected_pairs = [
        [{"extra": "logics-ocr"}, {"extra": "translate"}],
        [{"extra": "logics-ocr"}, {"extra": "deepseek-ocr"}],
        [{"extra": "logics-ocr"}, {"extra": "dots-ocr"}],
        [{"extra": "logics-ocr"}, {"extra": "penguin-vl-local"}],
        [{"extra": "logics-ocr"}, {"extra": "music-flamingo-local"}],
    ]

    for pair in expected_pairs:
        reversed_pair = list(reversed(pair))
        assert pair in conflicts or reversed_pair in conflicts
