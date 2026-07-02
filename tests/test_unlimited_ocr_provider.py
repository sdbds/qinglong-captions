import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _console():
    return Console(file=io.StringIO(), force_terminal=False)


def _ctx(config):
    from module.providers.base import ProviderContext

    return ProviderContext(
        console=_console(),
        config=config,
        args=SimpleNamespace(ocr_model="unlimited_ocr"),
    )


def _pdf_page(pdf_path: Path, page_number: int, image: Image.Image, *, page_count: int = 1):
    return SimpleNamespace(
        pdf_path=pdf_path,
        page_index=page_number - 1,
        page_number=page_number,
        page_count=page_count,
        image=image,
        size=image.size,
        dpi=144,
        image_format="PNG",
    )


def test_unlimited_ocr_default_model_id():
    from module.providers.ocr.unlimited import UnlimitedOCRProvider

    assert UnlimitedOCRProvider.default_model_id == "baidu/Unlimited-OCR"


def test_unlimited_ocr_default_prompt():
    from module.providers.ocr.unlimited import UnlimitedOCRProvider

    assert UnlimitedOCRProvider.default_prompt == "<image>document parsing."


def test_unlimited_ocr_multi_page_prompt_is_fixed():
    from module.providers.ocr.unlimited import _MULTI_PAGE_PROMPT

    assert _MULTI_PAGE_PROMPT == "<image>Multi page parsing."


def test_image_mode_gundam_resolves_correct_params():
    from module.providers.ocr.unlimited import _resolve_image_mode_params

    base_size, image_size, crop_mode = _resolve_image_mode_params("gundam", None, None, None)
    assert (base_size, image_size, crop_mode) == (1024, 640, True)


def test_image_mode_base_resolves_correct_params():
    from module.providers.ocr.unlimited import _resolve_image_mode_params

    base_size, image_size, crop_mode = _resolve_image_mode_params("base", None, None, None)
    assert (base_size, image_size, crop_mode) == (1024, 1024, False)


def test_image_mode_config_overrides_mode_defaults():
    from module.providers.ocr.unlimited import _resolve_image_mode_params

    base_size, image_size, crop_mode = _resolve_image_mode_params("gundam", 2048, 999, False)
    assert (base_size, image_size, crop_mode) == (2048, 999, False)


def test_image_mode_invalid_raises():
    from module.providers.ocr.unlimited import _resolve_image_mode_params

    with pytest.raises(ValueError, match="Unsupported unlimited_ocr image_mode"):
        _resolve_image_mode_params("fast", None, None, None)


def test_prompt_priority_config_over_default():
    from module.providers.ocr.unlimited import UnlimitedOCRProvider

    provider = UnlimitedOCRProvider(
        _ctx({"prompts": {"unlimited_ocr_prompt": "config prompt"}})
    )
    assert provider.get_prompts("image/png")[1] == "config prompt"


def test_prompt_priority_default_when_no_config():
    from module.providers.ocr.unlimited import UnlimitedOCRProvider

    provider = UnlimitedOCRProvider(_ctx({"prompts": {}}))
    assert provider.get_prompts("image/png")[1] == "<image>document parsing."


def test_openai_backend_raises_not_implemented():
    from module.providers.ocr.unlimited import UnlimitedOCRProvider

    provider = UnlimitedOCRProvider(_ctx({}))

    # Mock get_runtime_backend to return is_openai=True
    mock_backend = MagicMock()
    mock_backend.is_openai = True
    with patch.object(provider, "get_runtime_backend", return_value=mock_backend):
        with pytest.raises(NotImplementedError, match="OpenAI-compatible runtime backend"):
            provider.attempt(
                SimpleNamespace(uri="test.png", mime="image/png", pixels=None, extras={}),
                SimpleNamespace(system="", user="test", character_name="", character_prompt=""),
            )


def test_attempt_single_image_calls_infer(tmp_path, monkeypatch):
    from module.providers.ocr import unlimited as mod

    img_path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), "white").save(img_path)
    output_dir = tmp_path / "out"

    mock_model = MagicMock()
    mock_model.infer.return_value = "page content"

    mock_tokenizer = MagicMock()

    monkeypatch.setattr(mod, "_TRANS_LOADER", MagicMock())
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)
    monkeypatch.setattr(mod, "write_markdown_output", lambda *a, **kw: None)

    mod._TRANS_LOADER.get_or_load_processor.return_value = mock_tokenizer
    mod._TRANS_LOADER.get_or_load_model.return_value = mock_model

    result = mod.attempt_unlimited_ocr(
        uri=str(img_path),
        console=_console(),
        progress=None,
        task_id=None,
        prompt_text="<image>document parsing.",
        output_dir=str(output_dir),
        image_mode="gundam",
    )

    # No result.md file written by mock → falls back to return value
    assert result == "page content"
    mock_model.infer.assert_called_once()
    call_kwargs = mock_model.infer.call_args
    assert call_kwargs.kwargs["prompt"] == "<image>document parsing."
    assert call_kwargs.kwargs["base_size"] == 1024
    assert call_kwargs.kwargs["image_size"] == 640
    assert call_kwargs.kwargs["crop_mode"] is True
    assert call_kwargs.kwargs["save_results"] is True


def test_attempt_pdf_calls_infer_multi(tmp_path, monkeypatch):
    from module.providers.ocr import unlimited as mod

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "out"

    mock_model = MagicMock()
    mock_model.infer_multi.return_value = "combined content"

    mock_tokenizer = MagicMock()

    monkeypatch.setattr(mod, "_TRANS_LOADER", MagicMock())
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(
        mod,
        "iter_pdf_pages_high_quality",
        lambda _path: [
            _pdf_page(pdf_path, 1, Image.new("RGB", (8, 8), "white"), page_count=2),
            _pdf_page(pdf_path, 2, Image.new("RGB", (8, 8), "white"), page_count=2),
        ],
    )
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)
    monkeypatch.setattr(mod, "write_markdown_output", lambda *a, **kw: None)

    mod._TRANS_LOADER.get_or_load_processor.return_value = mock_tokenizer
    mod._TRANS_LOADER.get_or_load_model.return_value = mock_model

    result = mod.attempt_unlimited_ocr(
        uri=str(pdf_path),
        console=_console(),
        progress=None,
        task_id=None,
        output_dir=str(output_dir),
    )

    # No result.md file written by mock → falls back to return value (string)
    assert result == "combined content"
    mock_model.infer_multi.assert_called_once()
    call_kwargs = mock_model.infer_multi.call_args
    assert call_kwargs.kwargs["prompt"] == "<image>Multi page parsing."
    assert call_kwargs.kwargs["image_size"] == 1024
    assert call_kwargs.kwargs["save_results"] is True
    assert len(call_kwargs.kwargs["image_files"]) == 2
    # infer should not be called for PDF
    mock_model.infer.assert_not_called()


def test_attempt_pdf_chunks_when_over_budget(tmp_path, monkeypatch):
    from module.providers.ocr import unlimited as mod

    pdf_path = tmp_path / "big.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "out"

    mock_model = MagicMock()
    mock_model.infer_multi.return_value = "chunk text"
    mock_tokenizer = MagicMock()

    monkeypatch.setattr(mod, "_TRANS_LOADER", MagicMock())
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(
        mod,
        "iter_pdf_pages_high_quality",
        lambda _path: [
            _pdf_page(pdf_path, page, Image.new("RGB", (8, 8), "white"), page_count=5)
            for page in range(1, 6)
        ],
    )
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)
    monkeypatch.setattr(mod, "write_markdown_output", lambda *a, **kw: None)
    mod._TRANS_LOADER.get_or_load_processor.return_value = mock_tokenizer
    mod._TRANS_LOADER.get_or_load_model.return_value = mock_model

    result = mod.attempt_unlimited_ocr(
        uri=str(pdf_path),
        console=_console(),
        progress=None,
        task_id=None,
        output_dir=str(output_dir),
        page_budget=2,
    )

    # 5 pages / budget 2 -> 3 chunks -> 3 infer_multi calls, joined by page split
    assert mock_model.infer_multi.call_count == 3
    assert result.count("<--- Page Split --->") == 2
    for call in mock_model.infer_multi.call_args_list:
        assert len(call.kwargs["image_files"]) <= 2


def test_attempt_pdf_budget_zero_forces_single_call(tmp_path, monkeypatch):
    from module.providers.ocr import unlimited as mod

    pdf_path = tmp_path / "big.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "out"

    mock_model = MagicMock()
    mock_model.infer_multi.return_value = "all pages"
    mock_tokenizer = MagicMock()

    monkeypatch.setattr(mod, "_TRANS_LOADER", MagicMock())
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(
        mod,
        "iter_pdf_pages_high_quality",
        lambda _path: [
            _pdf_page(pdf_path, page, Image.new("RGB", (8, 8), "white"), page_count=5)
            for page in range(1, 6)
        ],
    )
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)
    monkeypatch.setattr(mod, "write_markdown_output", lambda *a, **kw: None)
    mod._TRANS_LOADER.get_or_load_processor.return_value = mock_tokenizer
    mod._TRANS_LOADER.get_or_load_model.return_value = mock_model

    result = mod.attempt_unlimited_ocr(
        uri=str(pdf_path),
        console=_console(),
        progress=None,
        task_id=None,
        output_dir=str(output_dir),
        page_budget=0,
    )

    # Budget disabled -> all 5 pages in one call, no page-split marker
    mock_model.infer_multi.assert_called_once()
    assert len(mock_model.infer_multi.call_args.kwargs["image_files"]) == 5
    assert result == "all pages"


def test_attempt_empty_output_raises(tmp_path, monkeypatch):
    from module.providers.ocr import unlimited as mod

    img_path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), "white").save(img_path)
    output_dir = tmp_path / "out"

    mock_model = MagicMock()
    mock_model.infer.return_value = "   "

    mock_tokenizer = MagicMock()

    monkeypatch.setattr(mod, "_TRANS_LOADER", MagicMock())
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)
    monkeypatch.setattr(mod, "write_markdown_output", lambda *a, **kw: None)

    mod._TRANS_LOADER.get_or_load_processor.return_value = mock_tokenizer
    mod._TRANS_LOADER.get_or_load_model.return_value = mock_model

    with pytest.raises(RuntimeError, match="returned empty output"):
        mod.attempt_unlimited_ocr(
            uri=str(img_path),
            console=_console(),
            progress=None,
            task_id=None,
            output_dir=str(output_dir),
        )


def test_attempt_single_image_reads_result_md(tmp_path, monkeypatch):
    """When model writes result.md, read it instead of using return value."""
    from module.providers.ocr import unlimited as mod

    img_path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), "white").save(img_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "result.md").write_text("# Real markdown content", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.infer.return_value = None  # model returns None, writes file instead

    monkeypatch.setattr(mod, "_TRANS_LOADER", MagicMock())
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)
    monkeypatch.setattr(mod, "write_markdown_output", lambda *a, **kw: None)

    mod._TRANS_LOADER.get_or_load_processor.return_value = MagicMock()
    mod._TRANS_LOADER.get_or_load_model.return_value = mock_model

    result = mod.attempt_unlimited_ocr(
        uri=str(img_path),
        console=_console(),
        progress=None,
        task_id=None,
        output_dir=str(output_dir),
    )

    assert result == "# Real markdown content"


def test_attempt_pdf_reads_result_md_with_tuple_return(tmp_path, monkeypatch):
    """infer_multi returns (outputs, output_tokens) tuple; result.md is the source of truth."""
    from module.providers.ocr import unlimited as mod

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "result.md").write_text("# PDF markdown", encoding="utf-8")

    mock_model = MagicMock()
    # infer_multi returns a tuple, not a string
    mock_model.infer_multi.return_value = ("outputs text", [1, 2, 3])

    monkeypatch.setattr(mod, "_TRANS_LOADER", MagicMock())
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(
        mod,
        "iter_pdf_pages_high_quality",
        lambda _path: [_pdf_page(pdf_path, 1, Image.new("RGB", (8, 8), "white"))],
    )
    monkeypatch.setattr(mod, "display_markdown", lambda **kwargs: None)
    monkeypatch.setattr(mod, "write_markdown_output", lambda *a, **kw: None)

    mod._TRANS_LOADER.get_or_load_processor.return_value = MagicMock()
    mod._TRANS_LOADER.get_or_load_model.return_value = mock_model

    result = mod.attempt_unlimited_ocr(
        uri=str(pdf_path),
        console=_console(),
        progress=None,
        task_id=None,
        output_dir=str(output_dir),
    )

    assert result == "# PDF markdown"


def test_no_self_reference_bug_in_attempt_unlimited_ocr():
    """Ensure the deepseek self-reference bug is not replicated."""
    import inspect

    from module.providers.ocr.unlimited import attempt_unlimited_ocr

    source = inspect.getsource(attempt_unlimited_ocr)
    assert "self." not in source, "attempt_unlimited_ocr must not reference self (deepseek bug)"
