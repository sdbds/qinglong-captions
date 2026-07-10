import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent

from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
from tests.provider_v2_helpers import make_provider_args


def _raise_runtime_error(message: str):
    raise RuntimeError(message)


def test_stepfun_retry_exhausted_logs_full_traceback():
    from module.providers.cloud_vlm.stepfun import StepfunProvider

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    provider = StepfunProvider(
        ProviderContext(
            console=console,
            config={},
            args=make_provider_args(step_api_key="sk-test", max_retries=1, wait_time=0.01),
        )
    )

    cfg = provider.get_retry_config()

    try:
        _raise_runtime_error("stepfun exhausted")
    except RuntimeError as exc:
        cfg.on_exhausted(exc)

    output = buffer.getvalue()
    assert "StepFun exhausted" in output
    assert "RuntimeError: stepfun exhausted" in output
    assert "Traceback" in output
    assert "_raise_runtime_error" in output


def test_minimax_api_load_tags_failure_logs_full_traceback(monkeypatch):
    from module.providers.cloud_vlm.minimax_api import _load_tags_from_json

    buffer = io.StringIO()
    progress = SimpleNamespace(console=Console(file=buffer, force_terminal=False, color_system=None))

    def _raise_json_error(self, encoding="utf-8"):
        raise json.JSONDecodeError("boom", "x", 0)

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "read_text", _raise_json_error)

    result = _load_tags_from_json("E:/fake.png", progress=progress)

    output = buffer.getvalue()
    assert result == []
    assert "Error loading or parsing" in output
    assert "JSONDecodeError" in output
    assert "Traceback" in output


def test_openai_compatible_attempt_logs_full_traceback_on_api_error(monkeypatch):
    from module.providers.cloud_vlm.openai_compatible import OpenAICompatibleProvider

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("api boom")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    provider = OpenAICompatibleProvider(
        ProviderContext(
            console=console,
            config={},
            args=make_provider_args(
                openai_base_url="http://localhost:1234/v1",
                openai_api_key="sk-test",
                openai_model_name="fake-model",
                openai_json_mode=True,
            ),
        )
    )

    media = MediaContext(
        uri="/fake.png",
        mime="image/png",
        sha256hash="",
        modality=MediaModality.IMAGE,
        blob="ZmFrZQ==",
    )
    prompts = PromptContext(system="system", user="describe")

    result = provider.attempt(media, prompts)

    output = buffer.getvalue()
    assert result.raw == ""
    assert "API call failed" in output
    assert "Retry failed" in output
    assert "RuntimeError: api boom" in output
    assert "Traceback" in output


class _FakeOCRRuntime:
    mode = "fake"
    model_id = "fake-model"


class _FakeOCRBackend:
    def __init__(self, runtime):
        self.runtime = runtime

    def complete(self, messages):
        return "# OCR\n"


def _make_fake_ocr_provider(console):
    from module.providers.ocr_base import OCRProvider

    class FakeOCRProvider(OCRProvider):
        name = "fake_ocr"
        default_model_id = "fake"
        default_prompt = "ocr"

        def get_runtime_backend(self):
            return _FakeOCRRuntime()

        def attempt(self, media, prompts):
            raise NotImplementedError

    return FakeOCRProvider(ProviderContext(console=console, config={}, args=make_provider_args()))


def _ocr_media(tmp_path):
    return MediaContext(
        uri=str(tmp_path / "doc.png"),
        mime="image/png",
        sha256hash="",
        modality=MediaModality.IMAGE,
        blob="ZmFrZQ==",
        extras={"output_dir": tmp_path / "doc"},
    )


def test_ocr_markdown_write_failure_logs_full_traceback(monkeypatch, tmp_path):
    import module.providers.ocr_base as ocr_base

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    provider = _make_fake_ocr_provider(console)

    monkeypatch.setattr(ocr_base, "OpenAIChatRuntime", _FakeOCRBackend)

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(ocr_base, "write_markdown_output", fail_write)

    result = provider.attempt_via_openai_backend(_ocr_media(tmp_path), PromptContext(system="", user="ocr"))

    output = buffer.getvalue()
    assert result.raw == "# OCR\n"
    assert "Failed to write OCR markdown" in output
    assert "RuntimeError: write failed" in output
    assert "Traceback" in output


def test_ocr_display_failure_logs_full_traceback(monkeypatch, tmp_path):
    import module.providers.ocr_base as ocr_base

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    provider = _make_fake_ocr_provider(console)

    monkeypatch.setattr(ocr_base, "OpenAIChatRuntime", _FakeOCRBackend)

    def fail_display(*_args, **_kwargs):
        raise RuntimeError("display failed")

    monkeypatch.setattr("utils.parse_display.display_markdown", fail_display)

    result = provider.attempt_via_openai_backend(_ocr_media(tmp_path), PromptContext(system="", user="ocr"))

    output = buffer.getvalue()
    assert result.raw == "# OCR\n"
    assert "Failed to display OCR markdown" in output
    assert "RuntimeError: display failed" in output
    assert "Traceback" in output
