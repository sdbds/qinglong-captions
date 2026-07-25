"""Deterministic image/PDF discovery with one-page-at-a-time PDF ownership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from PIL import Image

DEFAULT_PDF_DPI = 144
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
SUPPORTED_PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_INPUT_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


@dataclass(frozen=True)
class InputPage:
    source_path: Path
    source_type: str
    page_index: int | None = None
    page_count: int | None = None
    image: Image.Image | None = None
    rendered_page_size: tuple[int, int] | None = None

    @property
    def page_number(self) -> int | None:
        if self.page_index is None:
            return None
        return self.page_index + 1


def collect_source_inputs(
    input_path: str | Path,
    *,
    recursive: bool = True,
) -> list[Path]:
    candidate = Path(input_path).expanduser()
    if candidate.is_file():
        if candidate.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            raise ValueError(f"Unsupported sheet-music input file: {candidate}")
        return [candidate]
    if not candidate.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    iterator = candidate.rglob("*") if recursive else candidate.glob("*")
    sources = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    ]
    return sorted(
        sources,
        key=lambda path: path.relative_to(candidate).as_posix().casefold(),
    )


def _default_pdf_renderer(
    pdf_path: str | Path,
    *,
    dpi: int,
    image_format: str,
) -> Iterable[Any]:
    try:
        from utils.stream_util import iter_pdf_pages_high_quality
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF input requires PyMuPDF. Install the musvit-onnx extra."
        ) from exc
    yield from iter_pdf_pages_high_quality(
        pdf_path,
        dpi=dpi,
        image_format=image_format,
    )


def iter_source_pages(
    source_path: str | Path,
    *,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    pdf_renderer: Callable[..., Iterable[Any]] | None = None,
) -> Iterator[InputPage]:
    source = Path(source_path).expanduser()
    if source.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        yield InputPage(source_path=source, source_type="image")
        return
    if source.suffix.lower() not in SUPPORTED_PDF_EXTENSIONS:
        raise ValueError(f"Unsupported sheet-music input file: {source}")

    renderer = pdf_renderer or _default_pdf_renderer
    rendered_pages = iter(
        renderer(source, dpi=int(pdf_dpi), image_format="PNG")
    )
    rendered_count = 0
    try:
        for fallback_index, rendered_page in enumerate(rendered_pages):
            rendered_count += 1
            image = getattr(rendered_page, "image", rendered_page)
            if not isinstance(image, Image.Image):
                raise TypeError(
                    f"PDF renderer returned a non-Pillow image for {source}"
                )
            page_index = int(
                getattr(rendered_page, "page_index", fallback_index)
            )
            page_count_value = getattr(rendered_page, "page_count", None)
            page_count = (
                int(page_count_value) if page_count_value is not None else None
            )
            rendered_size = tuple(
                getattr(rendered_page, "size", image.size)
            )
            try:
                yield InputPage(
                    source_path=Path(
                        getattr(rendered_page, "pdf_path", source)
                    ).expanduser(),
                    source_type="pdf_page",
                    page_index=page_index,
                    page_count=page_count,
                    image=image,
                    rendered_page_size=rendered_size,
                )
            finally:
                image.close()
    finally:
        close_renderer = getattr(rendered_pages, "close", None)
        if close_renderer is not None:
            close_renderer()
    if rendered_count == 0:
        raise ValueError(f"PDF contains no renderable pages: {source}")
