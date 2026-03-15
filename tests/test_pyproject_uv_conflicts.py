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
