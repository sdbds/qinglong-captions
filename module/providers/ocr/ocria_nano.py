"""Local document OCR with LiteParse's OCRia-nano-1 CPU pipeline."""

from functools import lru_cache

from PIL import Image

from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext, RetryConfig
from module.providers.catalog import route_provider_name
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.output_writer import write_markdown_output

_IMAGE_MIMES = frozenset({
    "image/png", "image/jpeg", "image/tiff", "image/bmp", "image/x-ms-bmp",
    "image/gif", "image/webp", "image/svg+xml",
})


@lru_cache(maxsize=1)
def _get_parser(dpi: int, max_pages: int, keep_headers_footers: bool):
    if dpi <= 0 or max_pages <= 0:
        raise ValueError("ocria_nano dpi and max_pages must be positive")
    try:
        from liteparse import LiteParse, models
    except ImportError as exc:
        raise ImportError("Install LiteParse with: uv pip install -r pyproject.toml --extra ocria-nano") from exc

    # The weights have a separate license. Never download them or fall back to Tesseract implicitly.
    if not any(model.name == "ocria-nano" and model.complete for model in models.list()):
        raise RuntimeError("OCRia-nano-1 is not fully installed. Run: uv run --no-sync lit models pull ocria-nano")
    return LiteParse(
        ocr_backend="ocria-nano",
        output_format="markdown",
        image_mode="off",
        quiet=True,
        dpi=dpi,
        max_pages=max_pages,
        keep_headers_footers=keep_headers_footers,
    )


@register_provider("ocria_nano")
class OCRiaNanoProvider(OCRProvider):
    default_model_id = "llamaindex/OCRia-nano-1"

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        if route_provider_name("ocr_model", getattr(args, "ocr_model", "")) != cls.name:
            return False
        return mime == "application/pdf" or (mime in _IMAGE_MIMES and bool(getattr(args, "document_image", False)))

    def prepare_media(self, uri: str, mime: str, args) -> MediaContext:
        return MediaContext(
            uri=uri, mime=mime, sha256hash="",
            modality=MediaModality.DOCUMENT if mime == "application/pdf" else MediaModality.IMAGE,
        )

    def get_retry_config(self) -> RetryConfig:
        return RetryConfig(max_retries=1)

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if getattr(self.ctx.args, "local_runtime_backend", "") == "openai":
            raise ValueError("OCRia-nano-1 requires the direct LiteParse runtime, not an OpenAI-compatible server")
        parser = _get_parser(
            int(self._get_model_config("dpi", 250)),
            int(self._get_model_config("max_pages", 1000)),
            self._get_model_config("keep_headers_footers", False),
        )
        if media.modality == MediaModality.IMAGE and media.mime != "image/svg+xml":
            # LiteParse converts only the first frame and reports a successful one-page document.
            with Image.open(media.uri) as image:
                if getattr(image, "is_animated", False):
                    raise ValueError("OCRia-nano-1 accepts single-frame images; convert all frames to a multi-page PDF")
        result = parser.parse(media.uri)
        if len(result.pages) != result.total_pages:
            raise ValueError(
                f"OCRia-nano-1 parsed {len(result.pages)} of {result.total_pages} pages; "
                "check [ocria_nano].max_pages. Partial output was not published."
            )

        metadata = {"provider": self.name, "model_id": self.default_model_id, "page_count": result.total_pages}
        # Empty fences and separators between stripped page headers are not document content.
        content_pages = [page for page in result.pages if page.text.strip() and page.markdown.strip()]
        if not result.text.strip() or not content_pages:
            return CaptionResult.skipped("OCRia-nano-1 found no text", metadata=metadata)

        output_dir = self.artifact_output_dir(media)
        if media.modality == MediaModality.DOCUMENT:
            for page in content_pages:
                write_markdown_output(output_dir / f"page_{page.page_num:04d}", page.markdown)
        write_markdown_output(output_dir, result.text)
        # Keep the document typed through caption processing and Lance export, even if it contains literal JSON.
        return CaptionResult(
            raw=result.text,
            parsed={"markdown": result.text, "caption_extension": ".md"},
            metadata={**metadata, "output_dir": str(output_dir)},
        )
