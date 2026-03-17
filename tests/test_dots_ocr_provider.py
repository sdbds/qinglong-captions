import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))

from providers.base import ProviderContext
from providers.ocr.dots import DotsOCRProvider


def make_ctx(config):
    return ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config=config,
        args=SimpleNamespace(
            ocr_model="dots_ocr",
            document_image=True,
            dir_name=False,
            openai_model_name="",
            local_runtime_backend="",
        ),
    )


def test_resolve_prompt_mode_uses_upstream_mapping(monkeypatch):
    ctx = make_ctx({"dots_ocr": {"prompt_mode": "prompt_layout_all_en"}})
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_layout_all_en": "<doc-prompt>"},
        raising=False,
    )

    prompt_mode, prompt_text = provider._resolve_prompt_mode_and_prompt()

    assert prompt_mode == "prompt_layout_all_en"
    assert prompt_text == "<doc-prompt>"


def test_provider_prompt_overrides_upstream_default(monkeypatch):
    ctx = make_ctx(
        {
            "dots_ocr": {"prompt_mode": "prompt_web_parsing", "prompt": "<override>"},
            "prompts": {"dots_ocr_prompt": "<fallback>"},
        }
    )
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_web_parsing": "<upstream>"},
        raising=False,
    )

    _, prompt_text = provider._resolve_prompt_mode_and_prompt()

    assert prompt_text == "<override>"


def test_global_prompt_fallback_applies_when_provider_prompt_empty(monkeypatch):
    ctx = make_ctx(
        {
            "dots_ocr": {"prompt_mode": "prompt_scene_spotting", "prompt": ""},
            "prompts": {"dots_ocr_prompt": "<fallback>"},
        }
    )
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_scene_spotting": "<upstream>"},
        raising=False,
    )

    _, prompt_text = provider._resolve_prompt_mode_and_prompt()

    assert prompt_text == "<fallback>"


def test_prompt_image_to_svg_selects_svg_model(monkeypatch):
    ctx = make_ctx(
        {
            "dots_ocr": {
                "prompt_mode": "prompt_image_to_svg",
                "model_id": "text-model",
                "svg_model_id": "svg-model",
            }
        }
    )
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_image_to_svg": "<svg-prompt>"},
        raising=False,
    )

    assert provider._select_model_id("prompt_image_to_svg") == "svg-model"


def test_invalid_prompt_mode_raises(monkeypatch):
    ctx = make_ctx({"dots_ocr": {"prompt_mode": "not-real"}})
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_layout_all_en": "<doc-prompt>"},
        raising=False,
    )

    with pytest.raises(ValueError):
        provider._resolve_prompt_mode_and_prompt()
