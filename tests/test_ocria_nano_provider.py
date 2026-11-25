import io
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console

from module.providers.base import CaptionStatus, ProviderContext
from module.providers.catalog import route_choices
from module.providers.registry import get_registry


def test_ocria_route_is_opt_in_for_pdf_and_document_images():
    assert "ocria_nano" in route_choices("ocr_model")
    assert "ocria_nano_1" in route_choices("ocr_model", include_aliases=True)
    assert "ocria_nano" not in route_choices("vlm_image_model")
    for route in ("ocria_nano", "ocria_nano_1"):
        args = SimpleNamespace(ocr_model=route, document_image=True)
        cls = get_registry().find_provider(args, "application/pdf")
        assert cls.name == "ocria_nano"
        assert get_registry().find_provider(args, "image/png") is cls
        assert not cls.can_handle(args, "application/zip")
        args.document_image = False
        assert not cls.can_handle(args, "image/jpeg")
        assert not cls.can_handle(SimpleNamespace(), "application/pdf")


def test_gui_selection_installs_only_the_cpu_extra(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "gui"))
    from gui.wizard.step4_caption import CaptionStep

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="ocria_nano")
    step.vlm_image_model = step.alm_model = SimpleNamespace(value="")
    assert step._build_ocr_labels()["ocria_nano"] == "OCRia-nano-1 (CPU)"
    assert step._build_local_extra_args() == ["--extra", "ocria-nano"]
    extras = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]["optional-dependencies"]
    assert [dep.split("==")[0] for dep in extras["ocria-nano"]] == ["liteparse"]


def _provider(**settings):
    from module.providers.ocr.ocria_nano import OCRiaNanoProvider

    return OCRiaNanoProvider(ProviderContext(
        console=Console(file=io.StringIO()), args=SimpleNamespace(ocr_model="ocria_nano", document_image=True),
        config={"ocria_nano": settings},
    ))


@pytest.fixture
def sdk(monkeypatch):
    from module.providers.ocr.ocria_nano import _get_parser

    _get_parser.cache_clear()
    calls = []
    pages = [SimpleNamespace(page_num=1, text="First", markdown="# First"),
             SimpleNamespace(page_num=2, text="Second", markdown="**Second**")]
    result = SimpleNamespace(text="# First\n\n-----\n\n**Second**", pages=pages, total_pages=2)
    installed = [SimpleNamespace(name="ocria-nano", complete=True)]

    def parse(source):
        calls.append(Path(source))
        return result

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parse=parse)

    monkeypatch.setitem(sys.modules, "liteparse", SimpleNamespace(
        LiteParse=factory, models=SimpleNamespace(list=lambda: installed),
    ))
    yield SimpleNamespace(result=result, installed=installed, calls=calls)
    _get_parser.cache_clear()


def test_parse_original_file_once_and_publish_markdown(tmp_path, sdk):
    provider = _provider(dpi=180, keep_headers_footers=True, max_pages=30)
    pdf = tmp_path / "scan.pdf"
    result = provider.execute(str(pdf), "application/pdf", "hash")
    assert result.raw == "# First\n\n-----\n\n**Second**"
    assert result.metadata["model_id"] == "llamaindex/OCRia-nano-1"
    out = Path(result.metadata["output_dir"])
    assert out.parent == tmp_path / ".qinglong-ocr-artifacts"
    assert (out / "result.md").read_text("utf-8") == result.raw
    assert (out / "page_0002" / "result.md").read_text("utf-8") == "**Second**"
    options, source = sdk.calls
    assert source == pdf
    assert options["ocr_backend"] == "ocria-nano"
    assert options["output_format"] == "markdown"
    assert (options["dpi"], options["keep_headers_footers"], options["max_pages"]) == (180, True, 30)

    image = tmp_path / "receipt.png"
    Image.new("RGB", (32, 32), "white").save(image)
    _provider(dpi=180, keep_headers_footers=True, max_pages=30).execute(str(image), "image/png", "hash2")
    assert sdk.calls[2:] == [image]  # One parser for the batch; no JPEG/base64 conversion.


@pytest.mark.parametrize(("filename", "mime"), [("scan.png", "image/png"), ("scan.pdf", "application/pdf")])
def test_caption_pipeline_preserves_literal_document_markdown(tmp_path, sdk, filename, mime):
    from module.caption_pipeline.postprocess import postprocess_caption_content
    from module.lanceexport import save_caption

    markdown = "Invoice\n\n### First item\n\n**Subtotal**: 3*4\n\n### Total: 12"
    sdk.result.text = markdown
    provider = _provider()
    provider.ctx.args.mode = "long"
    provider.ctx.args.image_prompt_template = "rating"
    source = tmp_path / filename
    if mime == "image/png":
        Image.new("RGB", (32, 32), "white").save(source)
    result = provider.execute(str(source), mime, "hash")
    published = postprocess_caption_content(result, source, provider.ctx.args, provider.ctx.console)
    assert published.text == markdown
    export = tmp_path / "export" / filename
    assert save_caption(str(export), [published.to_dataset_caption()], mime.split("/")[0])
    assert export.with_suffix(".md").read_text("utf-8") == markdown


def test_export_does_not_interpret_scanned_json_as_a_caption(tmp_path, sdk):
    from module.caption_pipeline.postprocess import postprocess_caption_content
    from module.lanceexport import save_caption

    document = '{"description": "literal document field", "amount": 12}'
    sdk.result.text = document
    provider = _provider()
    source = tmp_path / "scan.pdf"
    result = provider.execute(str(source), "application/pdf", "hash")
    published = postprocess_caption_content(result, source, provider.ctx.args, provider.ctx.console)
    assert save_caption(str(source), [published.to_dataset_caption()], "application")
    assert source.with_suffix(".md").read_text("utf-8") == document
    assert json.loads(published.to_dataset_caption())["markdown"] == document


@pytest.mark.parametrize(("filename", "image_format", "mime"), [
    ("scan.tif", "TIFF", "image/tiff"),
    ("scan.gif", "GIF", "image/gif"),
    ("scan.png", "PNG", "image/png"),
    ("scan.webp", "WEBP", "image/webp"),
    ("scan.jpg", "MPO", "image/jpeg"),
])
def test_multiframe_scans_are_not_silently_truncated(tmp_path, sdk, filename, image_format, mime):
    source = tmp_path / filename
    with Image.new("RGB", (32, 32), "white") as first, Image.new("RGB", (32, 32), "black") as second:
        first.save(source, format=image_format, save_all=True, append_images=[second])

    with pytest.raises(ValueError, match="multi-page PDF"):
        _provider().execute(str(source), mime, "hash")
    assert not list(tmp_path.rglob("result.md"))


def test_partial_and_blank_results_are_not_published(tmp_path, sdk):
    provider = _provider()
    source = str(tmp_path / "scan.pdf")
    sdk.result.total_pages = 3
    with pytest.raises(ValueError, match="2 of 3"):
        provider.execute(source, "application/pdf", "hash")

    sdk.result.total_pages = 2
    for page in sdk.result.pages:
        page.text, page.markdown = "", "```text\n\n```"
    sdk.result.text = "```text\n\n```\n\n-----\n\n```text\n\n```"
    assert provider.execute(source, "application/pdf", "hash").status is CaptionStatus.SKIPPED
    assert not list(tmp_path.rglob("result.md"))


def test_page_separators_are_not_published_for_removed_headers(tmp_path, sdk):
    for page in sdk.result.pages:
        page.text, page.markdown = "Repeating page header", ""
    sdk.result.text = "\n\n-----\n\n"
    result = _provider().execute(str(tmp_path / "headers.pdf"), "application/pdf", "hash")
    assert result.status is CaptionStatus.SKIPPED
    assert not list(tmp_path.rglob("result.md"))


def test_missing_weights_do_not_fall_back_or_download(tmp_path, sdk):
    sdk.installed[0].complete = False
    with pytest.raises(RuntimeError, match="lit models pull ocria-nano"):
        _provider().execute(str(tmp_path / "scan.png"), "image/png", "hash")
    assert sdk.calls == []


def test_missing_sdk_names_the_optional_extra(tmp_path, sdk, monkeypatch):
    monkeypatch.setitem(sys.modules, "liteparse", None)
    with pytest.raises(ImportError, match="--extra ocria-nano"):
        _provider().execute(str(tmp_path / "scan.pdf"), "application/pdf", "hash")


def test_explicit_remote_backend_is_rejected_before_inference(tmp_path, sdk):
    provider = _provider()
    provider.ctx.args.local_runtime_backend = "openai"
    with pytest.raises(ValueError, match="direct"):
        provider.execute(str(tmp_path / "scan.pdf"), "application/pdf", "hash")
    assert sdk.calls == []


@pytest.mark.optional_runtime
def test_installed_ocria_recognizes_scanned_pdf_without_network(tmp_path):
    from PIL import ImageDraw, ImageFont

    models = pytest.importorskip("liteparse.models")
    if not any(model.name == "ocria-nano" and model.complete for model in models.list()):
        pytest.skip("Requires provisioned OCRia-nano-1 weights")

    pdf = tmp_path / "scan.pdf"
    font = ImageFont.load_default(size=44)
    with Image.new("RGB", (960, 320), "white") as first, Image.new("RGB", (960, 320), "white") as second:
        ImageDraw.Draw(first).text((60, 65), "Invoice number 12345", font=font, fill="black")
        ImageDraw.Draw(second).text((60, 65), "Payment received 67890", font=font, fill="black")
        first.save(pdf, save_all=True, append_images=[second], resolution=150)

    result = _provider().execute(str(pdf), "application/pdf", "hash")
    assert result.raw.index("12345") < result.raw.index("67890")
    assert result.metadata["page_count"] == 2
    with pytest.raises(ValueError, match="1 of 2"):
        _provider(max_pages=1).execute(str(pdf), "application/pdf", "hash")
