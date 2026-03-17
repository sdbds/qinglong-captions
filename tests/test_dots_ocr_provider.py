import io
import sys
import builtins
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))

import providers.ocr.dots as dots_module
from providers.base import ProviderContext
from providers.ocr.dots import DotsOCRProvider, _load_upstream_prompt_mapping, _resolve_model_source


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


def test_load_upstream_prompt_mapping_reads_prompts_file_without_importing_package(monkeypatch, tmp_path):
    prompts_path = tmp_path / "dots_ocr" / "utils" / "prompts.py"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(
        'dict_promptmode_to_prompt = {"prompt_layout_all_en": "<doc-prompt>"}',
        encoding="utf-8",
    )

    class FakeDistribution:
        def locate_file(self, relative_path):
            assert str(relative_path).replace("\\", "/") == "dots_ocr/utils/prompts.py"
            return prompts_path

    def fake_distribution(name):
        if name == "dots_ocr":
            return FakeDistribution()
        raise AssertionError(f"unexpected distribution lookup: {name}")

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dots_ocr" or name.startswith("dots_ocr."):
            raise AssertionError("dots_ocr package import should be bypassed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(dots_module, "importlib_metadata", SimpleNamespace(distribution=fake_distribution), raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    prompt_map = _load_upstream_prompt_mapping()

    assert prompt_map["prompt_layout_all_en"] == "<doc-prompt>"


def test_resolve_model_source_downloads_remote_snapshot(monkeypatch, tmp_path):
    resolved_snapshot = tmp_path / "snapshot"
    resolved_snapshot.mkdir()
    dots_module._resolve_model_source.cache_clear()
    monkeypatch.setattr(
        "providers.ocr.dots._download_model_snapshot",
        lambda repo_id: str(resolved_snapshot) if repo_id == "davanstrien/dots.ocr-1.5" else "",
        raising=False,
    )

    resolved = _resolve_model_source("davanstrien/dots.ocr-1.5")

    assert resolved == str(resolved_snapshot.resolve())


def test_run_direct_generation_uses_snapshot_path(monkeypatch, tmp_path):
    ctx = make_ctx({"dots_ocr": {"model_id": "davanstrien/dots.ocr-1.5"}})
    provider = DotsOCRProvider(ctx)
    resolved_snapshot = tmp_path / "models" / "dots"
    resolved_snapshot.mkdir(parents=True)
    captured = {}

    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=[[101, 102]])
            self.input_ids = [[101, 102]]

        def to(self, _device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            captured["messages"] = messages
            assert tokenize is False
            assert add_generation_prompt is True
            return "templated"

        def __call__(self, **kwargs):
            captured["processor_call"] = kwargs
            return FakeInputs()

        def batch_decode(self, generated_ids_trimmed, **kwargs):
            captured["generated_ids_trimmed"] = generated_ids_trimmed
            captured["decode_kwargs"] = kwargs
            return ["decoded-text"]

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return [[101, 102, 201, 202]]

    class FakeLoader:
        def get_or_load_processor(self, model_id, processor_cls, **kwargs):
            captured["processor_load"] = {
                "model_id": model_id,
                "processor_cls": processor_cls,
                "kwargs": kwargs,
            }
            return FakeProcessor()

        def get_or_load_model(self, model_id, model_cls, **kwargs):
            captured["model_load"] = {
                "model_id": model_id,
                "model_cls": model_cls,
                "kwargs": kwargs,
            }
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.AutoModelForCausalLM = object

    fake_qwen_vl_utils = types.ModuleType("qwen_vl_utils")
    fake_qwen_vl_utils.process_vision_info = lambda messages: (["image-input"], None)

    dots_module._resolve_model_source.cache_clear()
    monkeypatch.setattr(
        "providers.ocr.dots._resolve_model_source",
        lambda model_id: str(resolved_snapshot.resolve()) if model_id == "davanstrien/dots.ocr-1.5" else model_id,
        raising=False,
    )
    monkeypatch.setattr("providers.ocr.dots.resolve_device_dtype", lambda: ("cpu", "float32", "eager"), raising=False)
    monkeypatch.setattr(dots_module, "_TRANS_LOADER", FakeLoader(), raising=False)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_vl_utils)

    result = provider._run_direct_generation(
        image_path=str(tmp_path / "image.png"),
        prompt_mode="prompt_layout_all_en",
        prompt_text="<prompt>",
        model_id="davanstrien/dots.ocr-1.5",
        max_new_tokens=128,
    )

    assert captured["processor_load"]["model_id"] == str(resolved_snapshot.resolve())
    assert captured["model_load"]["model_id"] == str(resolved_snapshot.resolve())
    assert result == "decoded-text"


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


def test_server_backend_uses_same_resolved_prompt(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    _write_png(image_path)

    ctx = make_ctx(
        {
            "dots_ocr": {
                "prompt_mode": "prompt_web_parsing",
                "runtime_model_id": "served-dots",
            },
            "prompts": {"dots_ocr_prompt": "<override>"},
        }
    )
    ctx.args.local_runtime_backend = "openai"

    provider = DotsOCRProvider(ctx)
    captured = {}
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_web_parsing": "<upstream>"},
        raising=False,
    )
    monkeypatch.setattr(
        provider,
        "_complete_via_openai_runtime",
        lambda **kwargs: captured.setdefault("prompt", kwargs["prompt_text"]) or "# served markdown",
        raising=False,
    )

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    provider.attempt(media, provider.resolve_prompts(str(image_path), "image/png"))

    assert captured["prompt"] == "<override>"


def test_server_backend_svg_single_image_writes_svg(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    _write_png(image_path)

    ctx = make_ctx({"dots_ocr": {"prompt_mode": "prompt_image_to_svg", "svg_model_id": "served-svg"}})
    ctx.args.local_runtime_backend = "openai"
    provider = DotsOCRProvider(ctx)
    monkeypatch.setattr(
        "providers.ocr.dots._load_upstream_prompt_mapping",
        lambda: {"prompt_image_to_svg": "<svg>"},
        raising=False,
    )
    monkeypatch.setattr(
        provider,
        "_complete_via_openai_runtime",
        lambda **_: "<svg>served</svg>",
        raising=False,
    )

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, provider.resolve_prompts(str(image_path), "image/png"))

    assert (tmp_path / "sample" / "result.svg").read_text(encoding="utf-8") == "<svg>served</svg>"
    assert result.raw == "<svg>served</svg>"
