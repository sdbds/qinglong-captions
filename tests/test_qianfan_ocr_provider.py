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

import providers.ocr.qianfan as qianfan_module
from providers.base import ProviderContext
from providers.ocr.qianfan import QianfanOCRProvider


def make_ctx(config=None):
    return ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config=config or {},
        args=SimpleNamespace(
            ocr_model="qianfan_ocr",
            document_image=True,
            dir_name=False,
            openai_model_name="",
            local_runtime_backend="",
        ),
    )


def _write_png(path: Path):
    Image.new("RGB", (32, 32), color="white").save(path)


def test_compose_question_uses_base_prompt_by_default():
    provider = QianfanOCRProvider(make_ctx({"qianfan_ocr": {}}))

    assert provider._compose_question() == "Parse this document to Markdown.<think>"


def test_compose_question_appends_custom_prompt():
    provider = QianfanOCRProvider(
        make_ctx({"qianfan_ocr": {"prompt": "Keep tables aligned.", "prompt_strategy": "append"}})
    )

    question = provider._compose_question()

    assert question.startswith("Parse this document to Markdown.")
    assert "Keep tables aligned." in question
    assert question.endswith("<think>")


def test_compose_question_replaces_base_prompt():
    provider = QianfanOCRProvider(
        make_ctx({"qianfan_ocr": {"prompt": "Extract plain text only.", "prompt_strategy": "replace"}})
    )

    assert provider._compose_question() == "Extract plain text only.<think>"


def test_compose_question_disables_think_suffix():
    provider = QianfanOCRProvider(make_ctx({"qianfan_ocr": {"think_enabled": False}}))

    assert provider._compose_question() == "Parse this document to Markdown."


def test_invalid_prompt_strategy_raises():
    provider = QianfanOCRProvider(make_ctx({"qianfan_ocr": {"prompt_strategy": "merge"}}))

    with pytest.raises(ValueError):
        provider._compose_question()


def test_clean_reasoning_output_strips_think_sections():
    provider = QianfanOCRProvider(make_ctx({"qianfan_ocr": {}}))

    assert provider._clean_reasoning_output("<think>hidden</think>\n# final") == "# final"


def test_clean_reasoning_output_rejects_empty_result():
    provider = QianfanOCRProvider(make_ctx({"qianfan_ocr": {}}))

    with pytest.raises(ValueError):
        provider._clean_reasoning_output("<think>hidden</think>")


def test_run_direct_generation_uses_qianfan_chat_interface(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    _write_png(image_path)

    ctx = make_ctx({"qianfan_ocr": {"model_id": "baidu/Qianfan-OCR"}})
    provider = QianfanOCRProvider(ctx)
    captured = {}

    class FakeTensor:
        def to(self, dtype=None):
            captured["pixel_dtype"] = dtype
            return self

    class FakeModel:
        def chat(self, tokenizer, pixel_values=None, question=None, generation_config=None):
            captured["chat"] = {
                "tokenizer": tokenizer,
                "pixel_values": pixel_values,
                "question": question,
                "generation_config": generation_config,
            }
            return "<think>hidden</think>\n# final"

    class FakeLoader:
        def get_or_load_processor(self, model_id, processor_cls, **kwargs):
            captured["processor_load"] = {
                "model_id": model_id,
                "processor_cls": processor_cls,
                "kwargs": kwargs,
            }
            return "tokenizer"

        def get_or_load_model(self, model_id, model_cls, **kwargs):
            captured["model_load"] = {
                "model_id": model_id,
                "model_cls": model_cls,
                "kwargs": kwargs,
            }
            return FakeModel()

    fake_transformers = SimpleNamespace(AutoModel=object, AutoTokenizer=object)

    monkeypatch.setattr(qianfan_module, "_load_image_tensor", lambda *_args, **_kwargs: FakeTensor(), raising=False)
    monkeypatch.setattr(qianfan_module, "resolve_device_dtype", lambda: ("cpu", "bf16", "eager"), raising=False)
    monkeypatch.setattr(qianfan_module, "_TRANS_LOADER", FakeLoader(), raising=False)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = provider._run_direct_generation(
        image_path=str(image_path),
        question="Parse this document to Markdown.<think>",
        model_id="baidu/Qianfan-OCR",
        max_new_tokens=128,
    )

    assert captured["processor_load"]["model_id"] == "baidu/Qianfan-OCR"
    assert captured["model_load"]["model_id"] == "baidu/Qianfan-OCR"
    assert captured["chat"]["question"] == "Parse this document to Markdown.<think>"
    assert captured["chat"]["generation_config"] == {"max_new_tokens": 128}
    assert result == "<think>hidden</think>\n# final"


def test_attempt_single_image_writes_cleaned_result_md(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    _write_png(image_path)

    ctx = make_ctx({"qianfan_ocr": {"max_new_tokens": 64}})
    provider = QianfanOCRProvider(ctx)
    monkeypatch.setattr(
        provider,
        "_run_direct_generation",
        lambda **_: "<think>hidden</think>\n# final",
        raising=False,
    )

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, provider.resolve_prompts(str(image_path), "image/png"))

    assert (tmp_path / "sample" / "result.md").read_text(encoding="utf-8") == "# final"
    assert result.raw == "# final"


def test_attempt_pdf_writes_cleaned_page_results_and_merged_markdown(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    ctx = make_ctx({"qianfan_ocr": {"max_new_tokens": 64}})
    provider = QianfanOCRProvider(ctx)
    monkeypatch.setattr(
        qianfan_module,
        "pdf_to_images_high_quality",
        lambda _uri: [Image.new("RGB", (32, 32), color="white"), Image.new("RGB", (32, 32), color="white")],
        raising=False,
    )
    responses = iter([
        "<think>page-1</think>\n# page 1",
        "<think>page-2</think>\n# page 2",
    ])
    monkeypatch.setattr(
        provider,
        "_run_direct_generation",
        lambda **_: next(responses),
        raising=False,
    )

    media = provider.prepare_media(str(pdf_path), "application/pdf", ctx.args)
    result = provider.attempt(media, provider.resolve_prompts(str(pdf_path), "application/pdf"))

    assert (tmp_path / "sample" / "page_0001" / "page_0001.png").exists()
    assert (tmp_path / "sample" / "page_0001" / "result.md").read_text(encoding="utf-8") == "# page 1"
    assert (tmp_path / "sample" / "page_0002" / "result.md").read_text(encoding="utf-8") == "# page 2"
    assert (tmp_path / "sample" / "result.md").read_text(encoding="utf-8") == "# page 1\n<--- Page Split --->\n# page 2"
    assert result.raw == "# page 1\n<--- Page Split --->\n# page 2"
