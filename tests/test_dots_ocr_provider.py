import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
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


def _write_png(path: Path):
    Image.new("RGB", (32, 32), color="white").save(path)


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


def test_text_prompt_single_image_writes_result_md(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    _write_png(image_path)

    ctx = make_ctx({"dots_ocr": {"prompt_mode": "prompt_layout_all_en"}})
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_layout_all_en": "<doc>"},
        raising=False,
    )
    monkeypatch.setattr(
        provider,
        "_run_direct_generation",
        lambda **_: "# extracted markdown",
        raising=False,
    )

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, provider.resolve_prompts(str(image_path), "image/png"))

    assert (tmp_path / "sample" / "result.md").read_text(encoding="utf-8") == "# extracted markdown"
    assert result.raw == "# extracted markdown"


def test_svg_prompt_single_image_writes_result_svg(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    _write_png(image_path)

    ctx = make_ctx({"dots_ocr": {"prompt_mode": "prompt_image_to_svg", "svg_model_id": "svg-model"}})
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_image_to_svg": "<svg>"},
        raising=False,
    )
    monkeypatch.setattr(
        provider,
        "_run_direct_generation",
        lambda **_: "<svg>ok</svg>",
        raising=False,
    )

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, provider.resolve_prompts(str(image_path), "image/png"))

    assert (tmp_path / "sample" / "result.svg").read_text(encoding="utf-8") == "<svg>ok</svg>"
    assert result.raw == "<svg>ok</svg>"


def test_svg_prompt_pdf_writes_page_svgs_and_root_markdown(monkeypatch, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    ctx = make_ctx({"dots_ocr": {"prompt_mode": "prompt_image_to_svg", "svg_model_id": "svg-model"}})
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_image_to_svg": "<svg>"},
        raising=False,
    )
    monkeypatch.setattr(
        "providers.ocr.dots.pdf_to_images_high_quality",
        lambda _: [object(), object()],
        raising=False,
    )
    monkeypatch.setattr(
        "providers.ocr.dots._save_pdf_page_image",
        lambda *args, **kwargs: str(args[1]),
        raising=False,
    )
    monkeypatch.setattr(
        provider,
        "_run_direct_generation",
        lambda **kwargs: f"<svg>{kwargs['image_path']}</svg>",
        raising=False,
    )

    media = provider.prepare_media(str(pdf_path), "application/pdf", ctx.args)
    result = provider.attempt(media, provider.resolve_prompts(str(pdf_path), "application/pdf"))

    root_dir = tmp_path / "doc"
    assert (root_dir / "page_0001" / "result.svg").exists()
    assert (root_dir / "page_0002" / "result.svg").exists()
    root_md = (root_dir / "result.md").read_text(encoding="utf-8")
    assert "page_0001/result.svg" in root_md
    assert "page_0002/result.svg" in root_md
    assert result.raw == root_md
