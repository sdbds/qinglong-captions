import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from rich.console import Console

GROUNDED = (
    "<|ref|>title<|/ref|><|det|>[[0,0,999,333]]<|/det|>\n# Report\n"
    "<|ref|>image<|/ref|><|det|>[[333,333,999,666],[0,0,333,333]]<|/det|>\n"
    "<|ref|>text<|/ref|><|det|>[[0,666,999,999]]<|/det|>\nFinal paragraph."
)
EOS = "<eos>"


def _responses(monkeypatch, strings, responses):
    from transformers.feature_extraction_utils import BatchFeature

    from module.providers.ocr import deepseek_native as native

    prompts = []

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=1)

        def __call__(self, *, images, text, return_tensors, **kwargs):
            prompts.append(text)
            return BatchFeature({"input_ids": torch.tensor([[91, 92]]), "pixel_values": torch.tensor([0.5])})

        def decode(self, ids, *, skip_special_tokens, **kwargs):
            assert not skip_special_tokens, "would strip ref/det tokens"
            tokens = {1: EOS, 91: "prompt", 92: "prompt", **strings}
            return "".join(tokens[int(token)] for token in ids)

    class Model:
        device = torch.device("cpu")
        dtype = torch.float32
        config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=8192))
        generation_config = SimpleNamespace(eos_token_id=1)

        def generate(self, *, input_ids, pixel_values, do_sample, max_new_tokens, **kwargs):
            assert do_sample is False
            assert input_ids.dtype == torch.long
            assert pixel_values.dtype == self.dtype
            new_ids = next(responses)
            assert len(new_ids) <= max_new_tokens
            return torch.cat((input_ids, torch.tensor([new_ids])), dim=1)

    responses = iter(responses)
    monkeypatch.setattr(native, "resolve_device_dtype", lambda **kwargs: ("cpu", torch.float32, "eager"))
    monkeypatch.setattr(native, "_load_runtime", lambda *args, **kwargs: (Processor(), Model()))
    return prompts


def _run(source, output=None, **kwargs):
    from module.providers.ocr.deepseek_native import attempt_deepseek_native

    return attempt_deepseek_native(
        uri=str(source),
        output_dir=str(output) if output else None,
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        **kwargs,
    )


def _image(tmp_path):
    path = tmp_path / "scan.png"
    with Image.new("RGB", (300, 90), "white") as image:
        image.paste("blue", (100, 30, 300, 60))
        image.save(path)
    return path


def _pdf(tmp_path):
    fitz = pytest.importorskip("fitz", reason="PDF integration requires the PyMuPDF extra")

    path = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        pdf.new_page(width=150, height=90)
        pdf.new_page(width=180, height=120)
        pdf.save(path)
    return path


def test_image_keeps_generated_raw_and_crops_all_boxes_in_page_coordinates(monkeypatch, tmp_path):
    prompts = _responses(monkeypatch, {12: GROUNDED}, [[12, 1], [12, 1]])
    source = _image(tmp_path)
    first, second = tmp_path / "a", tmp_path / "b"
    markdown = _run(source, first)
    assert _run(source, second) == markdown
    assert prompts == ["<image>\n<|grounding|>Convert the document to markdown."] * 2
    assert markdown == (first / "result.md").read_text(encoding="utf-8")
    assert (first / "raw.txt").read_text(encoding="utf-8") == GROUNDED + EOS
    assert "<|" not in markdown and EOS not in markdown and "prompt" not in markdown
    assert "# Report" in markdown and "Final paragraph." in markdown
    layout = json.loads((first / "grounding.json").read_text(encoding="utf-8"))
    assert (layout["width"], layout["height"], layout["coordinate_system"]) == (300, 90, "normalized_0_999")
    assert [item["label"] for item in layout["regions"]] == ["title", "image", "image", "text"]
    crop = layout["regions"][1]
    assert crop["bbox"] == [333, 333, 999, 666]
    assert crop["bbox_pixels"] == [100, 30, 300, 60]
    assert "(images/0.png)" in markdown and "(images/1.png)" in markdown
    with Image.open(first / "images/0.png") as pixels:
        assert pixels.size == (200, 30) and pixels.getextrema() == ((0, 0), (0, 0), (255, 255))


def test_pdf_artifacts_are_page_local_and_merged_links_resolve(monkeypatch, tmp_path):
    _responses(monkeypatch, {12: GROUNDED, 13: GROUNDED.replace("Report", "Appendix")}, [[12, 1], [13, 1]])
    out = tmp_path / "pdf-out"
    combined = _run(_pdf(tmp_path), out)
    assert combined == (out / "result.md").read_text(encoding="utf-8")
    assert combined.index("Report") < combined.index("<--- Page Split --->") < combined.index("Appendix")
    for number, size in [(1, (300, 180)), (2, (360, 240))]:
        name = f"page_{number:04d}"
        page = out / name
        assert "(images/0.png)" in (page / "result.md").read_text(encoding="utf-8")
        assert f"({name}/images/0.png)" in combined and (page / "images/0.png").is_file()
        assert "<|ref|>image<|/ref|>" in (page / "raw.txt").read_text(encoding="utf-8")
        with Image.open(page / f"{name}.png") as image:
            assert image.size == size
        layout = json.loads((page / "grounding.json").read_text(encoding="utf-8"))
        assert (layout["width"], layout["height"]) == size


def test_plain_prompt_is_not_replaced_by_default_grounding(monkeypatch, tmp_path):
    prompts = _responses(monkeypatch, {12: "The original text."}, [[12, 1]])
    out = tmp_path / "out"
    assert _run(_image(tmp_path), out, prompt_text="Free OCR.") == "The original text."
    assert prompts == ["<image>\nFree OCR."]
    assert json.loads((out / "grounding.json").read_text(encoding="utf-8"))["regions"] == []
    assert not (out / "images").exists()


def test_configured_prompt_keeps_exact_whitespace(monkeypatch, tmp_path):
    prompts = _responses(monkeypatch, {12: "Text."}, [[12, 1]])
    prompt = "<image>\n<|grounding|>Convert the document to markdown. \n"
    _run(_image(tmp_path), tmp_path / "out", prompt_text=prompt)
    assert prompts == [prompt]


@pytest.mark.parametrize(
    "marker",
    [
        "<|ref|>image<|/ref|><|det|>[[0,0,1000,999]]<|/det|>",
        "<|ref|>image<|/ref|><|det|>[[20,100,10,200]]<|/det|>",
        "<|ref|>image<|/ref|><|det|>[]<|/det|>",
        "<|ref|>image<|/ref|><|det|>not coordinates<|/det|>",
        "<|ref|>image<|/ref|><|det|>[[10,20,30,40]]",
    ],
)
def test_bad_grounding_never_publishes_invented_or_partial_crops(monkeypatch, tmp_path, marker):
    _responses(monkeypatch, {12: GROUNDED + "\n" + marker}, [[12, 1]])
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        _run(_image(tmp_path), out)
    assert not any(out.iterdir())


@pytest.mark.parametrize(
    "settings",
    [
        {"prompt_text": "<image><image>"},
        {"base_size": 2048},
        {"image_size": 256},
        {"crop_mode": False},
        {"max_new_tokens": 0},
    ],
)
def test_unsupported_geometry_and_prompt_options_do_not_silently_change_inference(tmp_path, settings):
    with pytest.raises(ValueError, match="placeholder|geometry|positive"):
        _run(_image(tmp_path), tmp_path / "out", **settings)


def test_truncated_later_page_publishes_no_partial_document(monkeypatch, tmp_path):
    _responses(monkeypatch, {12: GROUNDED}, [[12, 1], [12, 12]])
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(RuntimeError, match="truncated"):
        _run(_pdf(tmp_path), out, max_new_tokens=2)
    assert list(out.iterdir()) == []


def test_occupied_output_is_not_adopted_and_default_runs_are_isolated(monkeypatch, tmp_path):
    _responses(monkeypatch, {12: GROUNDED, 13: "New text"}, [[12, 1], [13, 1]])
    source = _image(tmp_path)
    occupied = tmp_path / "scan"
    occupied.mkdir()
    user_file = occupied / "result.md"
    user_file.write_text("mine", encoding="utf-8")
    with pytest.raises((OSError, ValueError), match="empty|occupied"):
        _run(source, occupied)
    assert user_file.read_text(encoding="utf-8") == "mine"
    _run(source)
    _run(source)
    artifacts = tmp_path / ".qinglong-ocr-artifacts"
    assert len(list(artifacts.glob("*/result.md"))) == 2
    assert len(list(artifacts.glob("*/images/0.png"))) == 1


def test_loading_pins_native_components_and_never_executes_remote_code(monkeypatch, tmp_path):
    from module.providers.ocr import deepseek_native as native

    processor, model = object(), SimpleNamespace(eval=lambda: model)
    loads = []

    def load(cls, model_id, **kwargs):
        assert kwargs["trust_remote_code"] is False
        if cls == "model":
            assert kwargs["use_safetensors"] is True
        loads.append((model_id, kwargs))
        return processor if cls == "processor" else model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor="processor",
            DeepseekOcr2ForConditionalGeneration="model",
        ),
    )
    monkeypatch.setattr(native, "load_pretrained_component", load)
    native._load_runtime.cache_clear()
    try:
        hub = "deepseek-community/DeepSeek-OCR-2"
        for _ in range(2):
            assert native._load_runtime(hub, "cpu", torch.float32, "eager") == (processor, model)
        assert len(loads) == 2
        assert all(kw["revision"] == "838dc37567aa97106fcd590e2be6a8f279881813" for _, kw in loads)
        local = tmp_path / "snapshot"
        local.mkdir()
        (local / "config.json").write_text('{"model_type":"deepseek_ocr2"}', encoding="utf-8")
        assert native._load_runtime(str(local), "cpu", torch.float32, "eager") == (processor, model)
        assert all(key == str(local) and kw.get("local_files_only") and "revision" not in kw for key, kw in loads[2:])
        with pytest.raises(ValueError, match="native|converted"):
            native._load_runtime("deepseek-ai/DeepSeek-OCR-2", "cpu", torch.float32, "eager")
        native._load_runtime.cache_clear()
        monkeypatch.chdir(tmp_path)
        Path(hub).mkdir(parents=True)
        with pytest.raises(ValueError, match="shadows|absolute"):
            native._load_runtime(hub, "cpu", torch.float32, "eager")
    finally:
        native._load_runtime.cache_clear()


def test_provider_delegates_only_native_settings_and_rebases_caption_links(monkeypatch, tmp_path):
    from module.providers.base import PromptContext, ProviderContext
    from module.providers.ocr import deepseek_native as native

    source = tmp_path / "scan.pdf"
    source.touch()
    context = ProviderContext(
        console=Console(file=io.StringIO()),
        args=SimpleNamespace(
            deepseek_base_size=1024,
            deepseek_image_size=768,
            deepseek_crop_mode=True,
            openai_model_name="deepseek-ai/DeepSeek-OCR-2",
        ),
        config={
            "deepseek_ocr": {"model_id": "deepseek-ai/DeepSeek-OCR-2", "temperature": "unused legacy setting"},
            "deepseek_ocr_native": {"model_id": "native-local-path", "max_new_tokens": 321},
        },
    )
    provider = native.DeepSeekNativeOCRProvider(context)
    media = provider.prepare_media(str(source), "application/pdf", context.args)

    def run(**kwargs):
        assert kwargs["model_id"] == "native-local-path"
        assert kwargs["prompt_text"] == "My prompt" and kwargs["max_new_tokens"] == 321
        assert (kwargs["base_size"], kwargs["image_size"], kwargs["crop_mode"]) == (1024, 768, True)
        out = Path(kwargs["output_dir"])
        assert out == media.extras["output_dir"]
        (out / "crop.png").touch()
        return "![crop](crop.png)"

    monkeypatch.setattr(native, "attempt_deepseek_native", run)
    result = provider.attempt(media, PromptContext(system="", user="My prompt"))
    out = Path(result.metadata["output_dir"])
    assert result.metadata["provider"] == "deepseek_ocr_native"
    assert result.raw == f"![crop]({out.relative_to(source.parent).as_posix()}/crop.png)"
    del context.args.deepseek_image_size
    context.config["deepseek_ocr_native"]["image_size"] = 640
    sizes = []
    monkeypatch.setattr(native, "attempt_deepseek_native", lambda **kw: sizes.append(kw["image_size"]) or "ok")
    provider.attempt(media, PromptContext(system="", user="My prompt"))
    assert sizes == [640]  # The native implementation rejects unsupported geometry, rather than ignoring it.
    context.args.local_runtime_backend = "openai"
    with pytest.raises(ValueError, match="direct|local"):
        provider.attempt(media, PromptContext(system="", user="My prompt"))
