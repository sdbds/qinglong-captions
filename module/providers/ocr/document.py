"""Document input, output and generation checks shared by structured OCR adapters."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

from PIL import Image

from utils.ocr_artifacts import relative_artifact_url
from utils.stream_util import iter_pdf_pages_high_quality

_MARKDOWN_IMAGE_LINK = re.compile(r"(!\[[^\]\n]*\]\()(?P<path><[^>\n]+>|[^)\s]+)(?=\s|\))")
_HTML_IMAGE_LINK = re.compile(
    r"(<img\b[^>\n]*?\s+src\s*=\s*(?P<quote>['\"]))(?P<path>[^'\"\n]+)(?=(?P=quote))",
    re.IGNORECASE,
)


def rebase_generated_images(markdown: str, artifact_dir: Path, caption_dir: Path) -> str:
    """Rebase links to existing files inside an OCR result directory."""
    artifact_dir = artifact_dir.resolve()

    def replace_link(match: re.Match[str]) -> str:
        target = unquote(match["path"].removeprefix("<").removesuffix(">"))
        relative = PurePosixPath(target)
        if relative.is_absolute() or ".." in relative.parts or any(char in target for char in ":\\\x00"):
            return match[0]
        try:
            path = artifact_dir.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(artifact_dir) or not path.is_file():
                return match[0]
        except (OSError, ValueError, RuntimeError):
            return match[0]
        return match[1] + relative_artifact_url(path, caption_dir)

    return _HTML_IMAGE_LINK.sub(replace_link, _MARKDOWN_IMAGE_LINK.sub(replace_link, markdown))


def prepare_document_pages(source: Path, output_dir: Path, *, longest_edge: int | None = None, dpi: int = 200) -> list[Path]:
    pages: list[Path] = []

    def save_page(image: Image.Image, number: int) -> None:
        page_dir = output_dir / f"page_{number:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / f"page_{number:04d}.png"
        with image.convert("RGB") as rgb:
            if longest_edge:
                rgb.thumbnail((longest_edge, longest_edge), Image.Resampling.LANCZOS)
            rgb.save(path)
        pages.append(path)

    if source.suffix.lower() == ".pdf":
        for page in iter_pdf_pages_high_quality(source, dpi=dpi):
            try:
                save_page(page.image, page.page_number)
            finally:
                page.image.close()
    else:
        with Image.open(source) as image:
            save_page(image, 1)
    if not pages:
        raise ValueError(f"Document has no pages: {source}")
    return pages


def page_metadata(pages: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    result = []
    for index, path in enumerate(pages):
        with Image.open(path) as image:
            width, height = image.size
        result.append(
            {
                "page_index": index,
                "image": Path(os.path.relpath(path, output_dir)).as_posix(),
                "width": width,
                "height": height,
            }
        )
    return result


def write_layout_json(output_dir: Path, payload: dict, filename: str = "result.json") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def generation_eos_ids(model: Any, processor: Any) -> list[int]:
    ids: set[int] = set()
    config = getattr(model, "config", None)
    for source in (
        getattr(model, "generation_config", None),
        config,
        getattr(config, "text_config", None),
        getattr(processor, "tokenizer", processor),
    ):
        eos = getattr(source, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(int(value) for value in eos)
    return sorted(ids)


def ensure_complete_generation(token_ids, *, max_new_tokens: int, eos_token_ids: list[int]) -> None:
    if len(token_ids) >= max_new_tokens and (not len(token_ids) or int(token_ids[-1]) not in eos_token_ids):
        raise RuntimeError(
            "OCR generation was truncated at max_new_tokens; increase the token budget "
            "or reduce the document window size. Partial output was not published."
        )
