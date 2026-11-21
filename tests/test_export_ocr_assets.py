from __future__ import annotations

import base64
import io
import os
import re
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from rich.console import Console

from module.caption_pipeline.postprocess import postprocess_caption_content
from module.lanceexport import extract_from_lance
from tests.test_export_integrity import _dataset, _png
from utils.ocr_artifacts import relative_artifact_url, reserve_ocr_artifact_dir
from utils.output_writer import write_caption_output


@pytest.mark.parametrize("extract_images, route", [(True, "mistral_ocr"), (False, "")])
def test_mistral_pdf_preserves_pages_and_requested_images_through_postprocessing(tmp_path, monkeypatch, extract_images, route):
    from unittest.mock import Mock

    from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from module.providers.vision_api import pixtral

    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF")
    image_bytes = _png("green")
    payload = base64.b64encode(image_bytes).decode() if extract_images else None
    images = [SimpleNamespace(id=name, image_base64=payload) for name in ("fig #1.png", "fig(2).png")]
    pages = [
        SimpleNamespace(index=0, markdown='### A heading\n![chart](fig #1.png)\n![plot](fig(2).png "Plot")'
                                          '\n`![example](fig #1.png)`', images=images),
        SimpleNamespace(index=1, markdown="", images=[]),
        SimpleNamespace(index=2, markdown="Final page", images=[]),
    ]
    client = Mock()
    client.ocr.process.return_value = SimpleNamespace(pages=pages)
    monkeypatch.setattr(pixtral, "_create_mistral_client", lambda args: client)
    args = SimpleNamespace(mistral_api_key="test", ocr_model=route, document_image=extract_images, mode="short")
    console = Console(file=io.StringIO())
    provider = pixtral.MistralOCRProvider(ProviderContext(args=args, config={}, console=console))
    media = MediaContext(uri=str(source), mime="application/pdf", sha256hash="", modality=MediaModality.DOCUMENT,
                         extras={"signed_url": "https://example.invalid/scan.pdf"})

    response = provider.attempt(media, PromptContext(system="", user=""))
    result = postprocess_caption_content(response, source, args, console)

    assert result.is_persistable and result.metadata["provider"] == "mistral_ocr"
    caption, _ = write_caption_output(source, result, media.mime)
    markdown = caption.read_text(encoding="utf-8")
    assert markdown.count("<header") == 3 and markdown.count("<footer") == 3
    assert markdown.index("### A heading") < markdown.index(" Page 2 ") < markdown.index("Final page")
    link = unquote(re.search(r"!\[chart\]\(([^)]+)\)", markdown).group(1))
    if extract_images:
        assert (source.parent / link).read_bytes() == image_bytes
        plot = re.search(r'!\[plot\]\(([^ ]+) "Plot"\)', markdown).group(1)
        assert (source.parent / unquote(plot)).read_bytes() == image_bytes
        assert "`![example](fig #1.png)`" in markdown
    else:
        assert not list(tmp_path.rglob("*.png"))
    assert client.ocr.process.call_args.kwargs["include_image_base64"] is extract_images


def test_blank_ocr_pages_cannot_replace_a_caption_with_only_headers(tmp_path):
    from module.providers.base import PagedOCRResult

    response = PagedOCRResult(raw="", pages=[SimpleNamespace(index=0, markdown="  ", images=[])])
    result = postprocess_caption_content(response, tmp_path / "empty.pdf", SimpleNamespace(), Console(file=io.StringIO()))
    assert not result.is_persistable


@pytest.mark.parametrize("extract", [True, False])
def test_image_only_ocr_page_cannot_publish_unreferenced_figures(tmp_path, extract):
    from module.providers.base import PagedOCRResult

    data = _png("green")
    figure = SimpleNamespace(id="only figure.png", image_base64=base64.b64encode(data).decode())
    page = SimpleNamespace(index=0, markdown="", images=[figure])
    result = postprocess_caption_content(
        PagedOCRResult(raw="", pages=[page]), tmp_path / "scan.pdf", SimpleNamespace(document_image=extract),
        Console(file=io.StringIO()),
    )
    if extract:
        assert result.is_persistable
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", result.raw)
        assert len(links) == 1 and (tmp_path / unquote(links[0])).read_bytes() == data
    else:
        assert not result.is_persistable and not list(tmp_path.rglob("*.png"))


def test_ocr_links_prefer_the_complete_decoded_id_over_an_ambiguous_filename(tmp_path):
    source = tmp_path / "scan.pdf"
    images = [
        SimpleNamespace(id=f"scans {label}/diagram.png", image_base64=base64.b64encode(_png(color)).decode())
        for label, color in (("A", "red"), ("B", "blue"))
    ]
    page = SimpleNamespace(index=0, images=images, markdown=(
        "![a](scans%20A/diagram.png)\n![b](scans%20B/diagram.png)\n![ambiguous](diagram.png)"
    ))
    result = postprocess_caption_content([page], source, SimpleNamespace(document_image=True), Console(file=io.StringIO()))
    for label, color in (("a", "red"), ("b", "blue")):
        link = re.search(rf"!\[{label}\]\(([^)]+)\)", result.raw).group(1)
        assert (source.parent / unquote(link)).read_bytes() == _png(color)
    assert "![ambiguous](diagram.png)" in result.raw


def _ocr_document(tmp_path, *, name="scan.pdf", color="blue"):
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / name
    source.write_bytes(b"document")
    image_bytes = _png(color)
    figure = SimpleNamespace(id="figure #1.png", image_base64=base64.b64encode(image_bytes).decode("ascii"))
    page = SimpleNamespace(index=0, markdown="![figure](figure #1.png)", images=[figure])
    args = SimpleNamespace(document_image=True, ocr_model="mistral_ocr")
    markdown = postprocess_caption_content([page], source, args, Console(file=io.StringIO())).raw
    assert ".qinglong-ocr-artifacts/" in markdown
    relative = unquote(re.search(r"!\[figure\]\(([^)]+)\)", markdown).group(1))
    return source, markdown, relative, image_bytes


def _lance(tmp_path, source, markdown):
    return _dataset(tmp_path, [{
        "uris": str(source), "mime": "application/pdf", "blob": b"document", "captions": [markdown],
    }])


@pytest.mark.parametrize("separate_captions", [False, True])
def test_portable_ocr_export_keeps_main_and_paginated_image_links_working(tmp_path, separate_captions):
    source, markdown, relative, image_bytes = _ocr_document(tmp_path)
    (source.parent / "local.png").write_bytes(_png("red"))
    (source.parent / relative).with_name("unreferenced.png").write_bytes(_png("red"))
    output = tmp_path / "exports"
    captions = tmp_path / "captions" if separate_captions else output
    dataset = _lance(tmp_path, source, markdown + "\n![local](local.png)\n")
    options = {"caption_dir": str(captions)} if separate_captions else {}

    for _ in range(2):
        extract_from_lance(dataset, str(output), **options)

    assert (output / "scan.pdf").is_file()
    assert (captions / relative).read_bytes() == image_bytes
    assert not (captions / "local.png").exists()
    assert not (captions / relative).with_name("unreferenced.png").exists()
    for caption in [captions / "scan.md", captions / "scan" / "scan_1.md"]:
        link = unquote(re.search(r"!\[figure\]\(([^)]+)\)", caption.read_text(encoding="utf-8")).group(1))
        assert (caption.parent / link).resolve() == (captions / relative).resolve()


def test_portable_ocr_export_fails_before_publishing_when_asset_is_missing(tmp_path):
    source, markdown, relative, _ = _ocr_document(tmp_path)
    (source.parent / relative).unlink()
    output = tmp_path / "exports"

    with pytest.raises((ValueError, OSError), match="OCR asset"):
        extract_from_lance(_lance(tmp_path, source, markdown), str(output), clip_with_caption=False)

    assert not (output / "scan.pdf").exists()
    assert not (output / "scan.md").exists()


def test_separate_caption_ocr_export_reimports_with_its_managed_figures(tmp_path):
    from module.lanceImport import transform2lance

    source, markdown, relative, image_bytes = _ocr_document(tmp_path)
    (source.parent / relative).with_name("unreferenced.png").write_bytes(_png("red"))
    output, captions = tmp_path / "media", tmp_path / "captions"
    extract_from_lance(_lance(tmp_path, source, markdown), str(output), caption_dir=str(captions),
                       clip_with_caption=False)
    imported = transform2lance(str(output), caption_dir=str(captions))
    assert imported.count_rows() == 1

    second = tmp_path / "second"
    extract_from_lance(imported, str(second), clip_with_caption=False)

    assert (second / "scan.md").read_text(encoding="utf-8") == markdown
    for directory in (output, captions, second):
        assert (directory / relative).read_bytes() == image_bytes
        assert not (directory / relative).with_name("unreferenced.png").exists()


def test_portable_ocr_export_does_not_replace_an_existing_asset(tmp_path):
    source, markdown, relative, _ = _ocr_document(tmp_path)
    dataset = _lance(tmp_path, source, markdown)
    output = tmp_path / "exports"
    extract_from_lance(dataset, str(output), clip_with_caption=False)
    asset = output / relative
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"unrelated destination content")

    with pytest.raises((ValueError, OSError), match="OCR asset"):
        extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert asset.read_bytes() == b"unrelated destination content"


def test_portable_ocr_export_rolls_back_copied_assets_on_index_failure(tmp_path, monkeypatch):
    from module import lanceexport

    source, markdown, relative, image_bytes = _ocr_document(tmp_path)
    dataset = _lance(tmp_path, source, markdown)
    output = tmp_path / "exports"

    def fail_commit(*args):
        raise OSError("simulated caption index failure")

    with monkeypatch.context() as context:
        context.setattr(lanceexport, "write_caption_index", fail_commit)
        with pytest.raises(OSError, match="simulated caption index failure"):
            extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert not (output / relative).exists()
    assert not (output / "scan.md").exists()
    assert (source.parent / relative).read_bytes() == image_bytes
    extract_from_lance(dataset, str(output), clip_with_caption=False)
    assert (output / relative).read_bytes() == image_bytes


def test_portable_ocr_export_leaves_an_unowned_artifact_directory_unchanged(tmp_path):
    source, markdown, relative, _ = _ocr_document(tmp_path)
    output = tmp_path / "exports"
    unowned_root = output / relative.split("/")[0]
    unowned_root.mkdir(parents=True)
    user_file = unowned_root / "user.png"
    user_file.write_bytes(_png("red"))

    with pytest.raises(OSError, match="not owned"):
        extract_from_lance(_lance(tmp_path, source, markdown), str(output), clip_with_caption=False)

    assert list(unowned_root.iterdir()) == [user_file]
    assert user_file.read_bytes() == _png("red")
    assert not (output / "scan.md").exists()


def test_portable_ocr_export_rejects_an_asset_from_another_sources_run(tmp_path):
    source, _, _, _ = _ocr_document(tmp_path)
    other = source.with_name("other.pdf")
    other_asset = reserve_ocr_artifact_dir(other) / "other.png"
    other_asset.write_bytes(_png("red"))
    link = relative_artifact_url(other_asset, source.parent)
    output = tmp_path / "exports"

    with pytest.raises(ValueError, match="does not belong"):
        extract_from_lance(_lance(tmp_path, source, f"![figure]({link})"), str(output), clip_with_caption=False)

    assert not output.exists()


def test_portable_ocr_export_rejects_linked_asset_files(tmp_path):
    source, markdown, relative, _ = _ocr_document(tmp_path)
    user_file = source.parent / "private.png"
    user_file.write_bytes(_png("red"))
    asset = source.parent / relative
    asset.unlink()
    os.link(user_file, asset)
    output = tmp_path / "exports"

    with pytest.raises(ValueError, match="not a physical file"):
        extract_from_lance(_lance(tmp_path, source, markdown), str(output), clip_with_caption=False)

    assert user_file.read_bytes() == _png("red")
    assert not output.exists()


def test_portable_ocr_export_handles_generated_html_image_links(tmp_path):
    source, markdown, relative, image_bytes = _ocr_document(tmp_path)
    asset = source.parent / relative
    url = relative_artifact_url(asset, source.parent)
    markdown = re.sub(r"!\[figure\]\([^)]+\)", f'<img alt="figure" src="{url}">', markdown)
    output = tmp_path / "exports"

    extract_from_lance(_lance(tmp_path, source, markdown), str(output))

    assert (output / relative).read_bytes() == image_bytes
    assert f'src="{url}"' in (output / "scan.md").read_text(encoding="utf-8")
    assert f'src="../{url}"' in (output / "scan" / "scan_1.md").read_text(encoding="utf-8")


def test_in_place_ocr_export_does_not_copy_artifacts(tmp_path):
    source, markdown, relative, image_bytes = _ocr_document(tmp_path)
    output = tmp_path / "exports"

    extract_from_lance(_lance(tmp_path, source, markdown), str(output), clip_with_caption=False, in_place=True)

    assert (source.parent / relative).read_bytes() == image_bytes
    assert source.with_suffix(".md").is_file()
    assert not (output / relative).exists()


def test_disambiguated_ocr_exports_keep_source_binding_across_reimport(tmp_path):
    from module.lanceImport import transform2lance

    inputs = []
    for label in ("red", "blue"):
        parent = tmp_path / label
        parent.mkdir()
        source, markdown, relative, image_bytes = _ocr_document(parent, name="MixedCase.PDF", color=label)
        inputs.append((source, markdown, relative, image_bytes))
    dataset = _dataset(tmp_path, [
        {"uris": str(source), "mime": "application/pdf", "captions": [markdown]}
        for source, markdown, _, _ in inputs
    ])

    copies = [tmp_path / f"copy-{i}" for i in range(3)]
    for output in copies:
        extract_from_lance(dataset, str(output), clip_with_caption=False)
        primaries = sorted(output.glob("*.PDF"))
        assert len(primaries) == 2
        assert all(path.name != "MixedCase.PDF" for path in primaries)
        for _, _, relative, image_bytes in inputs:
            assert (output / relative).read_bytes() == image_bytes
        assert {path.with_suffix(".md").read_text(encoding="utf-8") for path in primaries} == {
            markdown for _, markdown, _, _ in inputs
        }
        dataset = transform2lance(str(output))
        assert dataset.count_rows() == 2

    renamed_sources = sorted(copies[-1].glob("*.PDF"))
    first, second = [path.with_suffix(".md").read_text(encoding="utf-8") for path in renamed_sources]
    mismatched_root = tmp_path / "mismatched"
    mismatched_root.mkdir()
    dataset = _lance(mismatched_root, renamed_sources[0], second)
    assert first != second
    target = tmp_path / "unsafe"
    with pytest.raises(ValueError, match="does not belong"):
        extract_from_lance(dataset, str(target), clip_with_caption=False)
    assert not target.exists()
