import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


class FakeProcessor:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or ["```markdown\nHello\n```"])
        self.apply_chat_template_calls = []
        self.processor_calls = []
        self.batch_decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.apply_chat_template_calls.append((messages, kwargs))
        return "templated"

    def __call__(self, **kwargs):
        self.processor_calls.append(kwargs)
        return {"input_ids": torch.tensor([[101, 102]])}

    def batch_decode(self, generated_ids, **kwargs):
        self.batch_decode_calls.append((generated_ids, kwargs))
        return [self.outputs.pop(0) if self.outputs else ""]


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[101, 102, 103, 104]])


def _console():
    return Console(file=io.StringIO(), force_terminal=False)


def _ctx(config):
    from providers.base import ProviderContext

    return ProviderContext(
        console=_console(),
        config=config,
        args=SimpleNamespace(ocr_model="infinity_parser2_ocr"),
    )


def test_infinity_parser2_ocr_uses_flash_by_default():
    from providers.ocr.infinity_parser2 import InfinityParser2OCRProvider

    assert InfinityParser2OCRProvider.default_model_id == "infly/Infinity-Parser2-Flash"


def test_infinity_parser2_model_toml_lists_flash_and_pro():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    parsed = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))
    section = parsed["infinity_parser2_ocr"]
    model_ids = {entry["model_id"] for entry in section["model_list"].values()}

    assert section["model_id"] == "infly/Infinity-Parser2-Flash"
    assert "infly/Infinity-Parser2-Flash" in model_ids
    assert "infly/Infinity-Parser2-Pro" in model_ids


def test_infinity_parser2_prompt_priority_provider_then_prompt_config_then_default():
    from providers.ocr.infinity_parser2 import DEFAULT_DOC2MD_PROMPT, InfinityParser2OCRProvider

    provider = InfinityParser2OCRProvider(
        _ctx(
            {
                "infinity_parser2_ocr": {"prompt": "provider prompt"},
                "prompts": {"infinity_parser2_ocr_prompt": "prompt config"},
            }
        )
    )
    assert provider.get_prompts("image/png")[1] == "provider prompt"

    provider = InfinityParser2OCRProvider(
        _ctx(
            {
                "infinity_parser2_ocr": {"prompt": ""},
                "prompts": {"infinity_parser2_ocr_prompt": "prompt config"},
            }
        )
    )
    assert provider.get_prompts("image/png")[1] == "prompt config"

    provider = InfinityParser2OCRProvider(_ctx({"infinity_parser2_ocr": {"prompt": ""}, "prompts": {}}))
    assert provider.get_prompts("image/png")[1] == DEFAULT_DOC2MD_PROMPT


def test_infinity_parser2_custom_task_requires_prompt():
    from providers.ocr.infinity_parser2 import InfinityParser2OCRProvider

    provider = InfinityParser2OCRProvider(
        _ctx(
            {
                "infinity_parser2_ocr": {"task_type": "custom", "prompt": ""},
                "prompts": {"infinity_parser2_ocr_prompt": ""},
            }
        )
    )

    with pytest.raises(ValueError, match="custom task requires"):
        provider.get_prompts("image/png")


def test_infinity_parser2_rejects_unknown_task_type():
    from providers.ocr.infinity_parser2 import InfinityParser2OCRProvider

    provider = InfinityParser2OCRProvider(_ctx({"infinity_parser2_ocr": {"task_type": "doc2json"}, "prompts": {}}))

    with pytest.raises(ValueError, match="Unsupported infinity_parser2_ocr task_type"):
        provider.get_prompts("image/png")


def test_infer_single_image_uses_official_chat_template_and_generation_options():
    from providers.ocr.infinity_parser2 import _infer_single_image

    processor = FakeProcessor()
    model = FakeModel()
    captured_process = {}

    def fake_process_vision_info(messages, image_patch_size):
        captured_process["messages"] = messages
        captured_process["image_patch_size"] = image_patch_size
        return ["vision-input"], None

    result = _infer_single_image(
        pil_image=Image.new("RGB", (8, 8), "white"),
        prompt_text="extract markdown",
        processor=processor,
        model=model,
        process_vision_info_fn=fake_process_vision_info,
        min_pixels=123,
        max_pixels=456,
        image_patch_size=16,
        max_new_tokens=99,
        temperature=0.0,
        top_p=1.0,
    )

    messages, template_kwargs = processor.apply_chat_template_calls[0]
    image_part = messages[0]["content"][0]

    assert result == "Hello"
    assert template_kwargs["enable_thinking"] is False
    assert template_kwargs["tokenize"] is False
    assert image_part["min_pixels"] == 123
    assert image_part["max_pixels"] == 456
    assert captured_process["image_patch_size"] == 16
    assert processor.processor_calls[0]["do_resize"] is False
    assert model.generate_calls[0]["max_new_tokens"] == 99
    assert model.generate_calls[0]["temperature"] == 0.0
    assert model.generate_calls[0]["top_p"] == 1.0


def test_infer_single_image_rejects_empty_output():
    from providers.ocr.infinity_parser2 import _infer_single_image

    with pytest.raises(ValueError, match="returned empty output"):
        _infer_single_image(
            pil_image=Image.new("RGB", (8, 8), "white"),
            prompt_text="extract markdown",
            processor=FakeProcessor(outputs=["```markdown\n \n```"]),
            model=FakeModel(),
            process_vision_info_fn=lambda messages, image_patch_size: (["vision-input"], None),
        )


def test_attempt_infinity_parser2_pdf_writes_pages_and_root_result(tmp_path, monkeypatch):
    from providers.ocr import infinity_parser2 as mod

    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "out"
    processor = FakeProcessor(outputs=["page one", "page two"])

    monkeypatch.setattr(
        mod,
        "pdf_to_images_high_quality",
        lambda _: [Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "white")],
    )
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)

    result = mod.attempt_infinity_parser2_ocr(
        uri=str(source),
        console=_console(),
        progress=None,
        task_id=None,
        prompt_text="extract markdown",
        output_dir=str(output_dir),
        processor=processor,
        model=FakeModel(),
        process_vision_info_fn=lambda messages, image_patch_size: (["vision-input"], None),
    )

    assert result == "page one\n<--- Page Split --->\npage two"
    assert (output_dir / "page_0001" / "result.md").read_text(encoding="utf-8") == "page one"
    assert (output_dir / "page_0002" / "result.md").read_text(encoding="utf-8") == "page two"
    assert (output_dir / "result.md").read_text(encoding="utf-8") == result


def test_attempt_infinity_parser2_pdf_raises_when_all_pages_fail(tmp_path, monkeypatch):
    from providers.ocr import infinity_parser2 as mod

    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        mod,
        "pdf_to_images_high_quality",
        lambda _: [Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "white")],
    )
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="failed for all PDF pages"):
        mod.attempt_infinity_parser2_ocr(
            uri=str(source),
            console=_console(),
            progress=None,
            task_id=None,
            prompt_text="extract markdown",
            output_dir=str(output_dir),
            processor=FakeProcessor(outputs=["", ""]),
            model=FakeModel(),
            process_vision_info_fn=lambda messages, image_patch_size: (["vision-input"], None),
        )

    assert not (output_dir / "result.md").exists()
