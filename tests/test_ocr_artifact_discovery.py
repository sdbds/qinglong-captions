import base64
import io
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from PIL import Image
from rich.console import Console

from module.caption_pipeline.postprocess import postprocess_caption_content
from module.lanceImport import load_data
from module.providers.base import PromptContext, ProviderContext
from module.providers.ocr import deepseek, deepseek_native, paddle, unlimited
from module.providers.ocr.document import rebase_generated_images
from module.providers.ocr.dots import DotsOCRProvider
from module.providers.ocr.glm import GLMOCRProvider
from module.providers.ocr.infinity_parser2 import InfinityParser2OCRProvider
from module.providers.ocr.ovis_ocr2 import OvisOCR2Provider
from module.providers.vision_api import pixtral


pytestmark = pytest.mark.compat
ARTIFACT_ROOT = ".qinglong-ocr-artifacts"


def _provider(cls):
    return cls(ProviderContext(console=Console(file=io.StringIO()), config={}, args=SimpleNamespace()))


@pytest.mark.parametrize("cls", [GLMOCRProvider, DotsOCRProvider, InfinityParser2OCRProvider, OvisOCR2Provider])
def test_provider_preparation_reserves_separate_directories_for_complete_sources(tmp_path, cls):
    legacy_dir = tmp_path / "scan"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "result.md"
    legacy_file.write_text("existing user file", encoding="utf-8")
    provider = _provider(cls)

    output_dirs = []
    for name in ("scan.pdf", "scan.xps", "scan.pdf"):
        source = tmp_path / name
        source.touch()
        media = provider.prepare_media(str(source), "application/pdf", provider.ctx.args)
        output_dir = Path(media.extras["output_dir"])
        assert output_dir.is_dir()
        output_dirs.append(output_dir)
        (output_dir / "result.md").write_text(name, encoding="utf-8")

    assert len(set(output_dirs)) == 3
    assert all(output_dir.is_relative_to(tmp_path / ARTIFACT_ROOT) for output_dir in output_dirs)
    assert [path.joinpath("result.md").read_text(encoding="utf-8") for path in output_dirs] == [
        "scan.pdf", "scan.xps", "scan.pdf",
    ]
    assert legacy_file.read_text(encoding="utf-8") == "existing user file"


def test_provider_does_not_adopt_an_unowned_reserved_directory(tmp_path):
    source = tmp_path / "scan.pdf"
    source.touch()
    root = tmp_path / ARTIFACT_ROOT
    root.mkdir()
    existing_file = root / "result.md"
    existing_file.write_text("existing user file", encoding="utf-8")
    provider = _provider(GLMOCRProvider)

    with pytest.raises(OSError, match="OCR artifact"):
        provider.prepare_media(str(source), "application/pdf", provider.ctx.args)

    assert list(root.iterdir()) == [existing_file]


def test_mistral_pdf_preparation_leaves_legacy_stem_path_alone(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    source.touch()
    legacy_file = tmp_path / "scan"
    legacy_file.write_text("existing user file", encoding="utf-8")
    monkeypatch.setattr(pixtral, "_create_mistral_client", lambda args: None)
    provider = _provider(pixtral.MistralOCRProvider)

    media = provider.prepare_media(str(source), "application/pdf", SimpleNamespace(max_retries=0))

    assert media.uri == str(source)
    assert legacy_file.read_text(encoding="utf-8") == "existing user file"


def _page(index, data):
    img = SimpleNamespace(id="../figure #1.png", image_base64=base64.b64encode(data).decode("ascii"))
    return SimpleNamespace(index=index, markdown="![figure](../figure #1.png)", images=[img])


def test_postprocessed_image_links_preserve_same_stem_sources_and_repeated_page_names(tmp_path):
    args = SimpleNamespace(document_image=True, ocr_model="mistral_ocr")
    console = Console(file=io.StringIO())
    links = []
    for suffix in ("jpg", "png"):
        source = tmp_path / f"scan.{suffix}"
        source.touch()
        pages = [_page(0, f"{suffix}-one".encode()), _page(1, f"{suffix}-two".encode())]
        result = postprocess_caption_content(pages, source, args, console)
        links.extend(re.findall(r"!\[figure\]\(([^)]*)\)", result.raw))

    paths = [tmp_path / unquote(link) for link in links]
    assert len(set(paths)) == 4
    assert all(path.is_relative_to(tmp_path / ARTIFACT_ROOT) for path in paths)
    assert all("%23" in link for link in links)
    assert [path.read_bytes() for path in paths] == [b"jpg-one", b"jpg-two", b"png-one", b"png-two"]
    assert not (tmp_path / "scan").exists()


def test_rebase_only_changes_links_to_existing_local_assets(tmp_path):
    artifact = tmp_path / "artifact output"
    images_dir = artifact / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "figure #1.png").touch()
    (tmp_path / "unrelated.png").touch()
    markdown = (
        "![crop](images/figure%20%231.png)\n"
        '<img src="images/figure #1.png" alt="crop">\n'
        "![outside](../unrelated.png)\n"
        "![missing](images/missing.png)\n"
        "![external](https://example.test/figure.png)"
    )

    rebased = rebase_generated_images(markdown, artifact, tmp_path)

    assert rebased == markdown.replace("images/figure%20%231.png", "artifact%20output/images/figure%20%231.png").replace(
        "images/figure #1.png", "artifact%20output/images/figure%20%231.png",
    )


@pytest.mark.parametrize(("engine", "label", "filename"), [("legacy", "crop", "crop.png"), ("native", "image", "0.png")])
def test_deepseek_preserves_links_to_page_local_images_in_both_markdown_outputs(tmp_path, monkeypatch, engine, label, filename):
    source = tmp_path / "scan.pdf"
    source.touch()

    def infer(_tokenizer, *, output_path, **kwargs):
        output = Path(output_path)
        (output / "images").mkdir()
        (output / "images" / "crop.png").touch()
        (output / "result.mmd").write_text("![crop](images/crop.png)", encoding="utf-8")

    if engine == "legacy":
        mod = deepseek
        provider = _provider(deepseek.DeepSeekLegacyOCRProvider)
        provider.ctx.config["deepseek_ocr"] = {"model_id": str(tmp_path / "legacy-fixture")}
        monkeypatch.setattr(mod, "version", lambda _: "4.46.3")
        monkeypatch.setattr(mod, "_TRANS_LOADER", SimpleNamespace(
            get_or_load_processor=lambda *args, **kwargs: None,
            get_or_load_model=lambda *args, **kwargs: SimpleNamespace(infer=infer),
        ))
    else:
        mod = deepseek_native
        provider = _provider(deepseek.DeepSeekOCRProvider)
        prediction = "<|ref|>image<|/ref|><|det|>[[0,0,999,999]]<|/det|>"
        monkeypatch.setattr(mod, "_load_runtime", lambda *_: (None, None))
        monkeypatch.setattr(mod, "_generate_page", lambda *_: (prediction, prediction))
    monkeypatch.setattr(mod, "resolve_device_dtype", lambda **_: ("cpu", None, "eager"))
    monkeypatch.setattr(mod, "iter_pdf_pages_high_quality", lambda *_: [
        SimpleNamespace(page_number=1, image=Image.new("RGB", (2, 2))),
    ])
    monkeypatch.setattr(mod, "display_markdown", lambda **_: None)
    media = provider.prepare_media(str(source), "application/pdf", provider.ctx.args)

    result = provider.attempt(media, PromptContext(system="", user=""))

    output = media.extras["output_dir"]
    page_dir = output / "page_0001"
    assert (page_dir / "result.md").read_text(encoding="utf-8") == f"![{label}](images/{filename})"
    assert (output / "result.md").read_text(encoding="utf-8") == f"![{label}](page_0001/images/{filename})"
    link = re.search(rf"!\[{label}\]\(([^)]+)\)", result.raw)[1]
    assert (source.parent / unquote(link)) == page_dir / "images" / filename


@pytest.mark.parametrize("provider_name, mod, cls, attempt_name", [
    ("unlimited_ocr", unlimited, unlimited.UnlimitedOCRProvider, "attempt_unlimited_ocr"),
    ("paddle_ocr", paddle, paddle.PaddleOCRProvider, "attempt_paddle_vl_native"),
])
def test_legacy_provider_rebases_emitted_image_links(tmp_path, monkeypatch, provider_name, mod, cls, attempt_name):
    source = tmp_path / "scan.pdf"
    source.touch()
    provider = _provider(cls)
    provider.ctx.config[provider_name] = {"model_version": "1", "backend": "paddle_vl_native"}

    def mock_inference(*, output_dir, **kwargs):
        artifact = Path(output_dir)
        (artifact / "images").mkdir()
        (artifact / "images" / "crop.png").touch()
        markdown = "![crop](images/crop.png)"
        return (markdown, {}) if provider_name == "paddle_ocr" else markdown

    monkeypatch.setattr(mod, attempt_name, mock_inference)
    media = provider.prepare_media(str(source), "application/pdf", provider.ctx.args)

    result = provider.attempt(media, PromptContext(system="", user=""))

    link = re.search(r"!\[crop\]\(([^)]+)\)", result.raw)[1]
    assert (source.parent / unquote(link)) == media.extras["output_dir"] / "images" / "crop.png"


@pytest.mark.parametrize("external_captions", [False, True])
def test_import_ignores_ocr_artifacts_but_keeps_user_page_folders(tmp_path, external_captions):
    dataset = tmp_path / "dataset"
    originals = dataset / "page_0001"
    originals.mkdir(parents=True)
    source = originals / "scan.pdf"
    source.touch()
    user_page = originals / "page_0001.png"
    user_page.touch()
    artifacts = originals / ARTIFACT_ROOT / "scan" / "page_0001"
    artifacts.mkdir(parents=True)
    (artifacts / "page_0001.png").touch()
    (artifacts / "result.md").write_text("generated OCR", encoding="utf-8")
    captions = tmp_path / "captions"
    captions.mkdir()

    caption_dir = str(captions) if external_captions else None
    assert {Path(row["file_path"]) for row in load_data(str(dataset), caption_dir)} == {source, user_page}
    assert load_data(str(artifacts), caption_dir) == []
