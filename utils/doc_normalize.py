from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Optional

TEXT_SOURCE_EXTENSIONS = frozenset({".txt", ".md"})


class NormalizationError(RuntimeError):
    """Raised when a document cannot be normalized to markdown."""


_DOCLING_CONVERTER: Optional[Any] = None
_MARKITDOWN_CONVERTER: Optional[Any] = None


def normalize_asset(uri: Path, blob: Optional[bytes]) -> str:
    suffix = uri.suffix.lower()
    if suffix in TEXT_SOURCE_EXTENSIONS:
        return normalize_text_asset(uri, blob)
    return normalize_document_asset(uri, blob)


def normalize_text_asset(uri: Path, blob: Optional[bytes]) -> str:
    text = _decode_text(blob if blob is not None else uri.read_bytes())
    suffix = uri.suffix.lower()
    if suffix == ".md":
        return _finalize_markdown(text)
    return _finalize_markdown(_text_to_markdown(text))


def normalize_document_asset(uri: Path, blob: Optional[bytes]) -> str:
    source_path = uri if uri.exists() else None
    if source_path is not None:
        try:
            return _finalize_markdown(_convert_document(source_path))
        except Exception:
            if blob is None:
                raise

    if blob is None:
        raise NormalizationError(f"Missing source blob for {uri}")

    with tempfile.TemporaryDirectory(prefix="qinglong-doc-") as temp_dir:
        temp_path = Path(temp_dir) / f"{uri.stem or 'document'}{uri.suffix}"
        temp_path.write_bytes(blob)
        return _finalize_markdown(_convert_document(temp_path))


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _text_to_markdown(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""
    paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", normalized) if segment.strip()]
    if not paragraphs:
        return ""
    return "\n\n".join(paragraphs)

def _convert_document(source_path: Path) -> str:
    errors: list[str] = []

    try:
        return _convert_with_docling(source_path)
    except Exception as exc:
        errors.append(f"Docling: {exc}")

    try:
        return _convert_with_markitdown(source_path)
    except Exception as exc:
        errors.append(f"MarkItDown: {exc}")

    raise NormalizationError("; ".join(errors) or f"Unsupported document format: {source_path.suffix}")


def _convert_with_docling(source_path: Path) -> str:
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.document_converter import DocumentConverter

        _DOCLING_CONVERTER = DocumentConverter()

    result = _DOCLING_CONVERTER.convert(str(source_path))
    document = getattr(result, "document", result)
    for attr in ("export_to_markdown", "to_markdown"):
        fn = getattr(document, attr, None)
        if callable(fn):
            markdown = fn()
            if markdown:
                return markdown

    markdown = getattr(document, "markdown", None) or getattr(result, "markdown", None)
    if markdown:
        return markdown
    raise NormalizationError(f"Docling returned no markdown for {source_path}")


def _convert_with_markitdown(source_path: Path) -> str:
    global _MARKITDOWN_CONVERTER
    if _MARKITDOWN_CONVERTER is None:
        from markitdown import MarkItDown

        _MARKITDOWN_CONVERTER = MarkItDown()

    result = _MARKITDOWN_CONVERTER.convert(str(source_path))
    if isinstance(result, str):
        return result

    for attr in ("markdown", "text_content", "text"):
        markdown = getattr(result, attr, None)
        if markdown:
            return markdown

    raise NormalizationError(f"MarkItDown returned no markdown for {source_path}")


def _finalize_markdown(markdown: str) -> str:
    cleaned = markdown.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
    return (cleaned + "\n") if cleaned else ""
