"""Shared directory-name prompt context resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.stream_util import split_name_series


DOCUMENT_OCR_PROVIDER_NAMES = frozenset(
    {
        "chandra_ocr",
        "deepseek_ocr",
        "dots_ocr",
        "firered_ocr",
        "glm_ocr",
        "hunyuan_ocr",
        "lighton_ocr",
        "logics_ocr",
        "nanonets_ocr",
        "olmocr",
        "paddle_ocr",
        "qianfan_ocr",
    }
)


@dataclass(frozen=True)
class DirectoryNameContext:
    enabled: bool
    applicable: bool
    source_uri: str = ""
    raw_directory_name: str = ""
    character_name: str = ""
    character_prompt: str = ""
    reason: str = ""

    @property
    def has_prompt(self) -> bool:
        return bool(self.character_prompt)


def _non_empty_string(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _media_extra_source_uri(media: Any) -> str:
    extras = getattr(media, "extras", None)
    if not isinstance(extras, dict):
        return ""
    return _non_empty_string(extras.get("directory_name_source_uri"))


def _resolve_source_uri(*, args: Any, uri: str, media: Any = None, source_uri: str | None = None) -> str:
    for candidate in (
        source_uri,
        getattr(args, "directory_name_source_uri", ""),
        _media_extra_source_uri(media),
        uri,
    ):
        candidate_text = _non_empty_string(candidate)
        if candidate_text:
            return candidate_text
    return ""


def _raw_directory_name(source_uri: str) -> str:
    if not source_uri:
        return ""
    try:
        return Path(source_uri).parent.name or ""
    except (OSError, ValueError):
        return ""


def _is_applicable(*, args: Any, mime: str, provider_name: str) -> tuple[bool, str]:
    mime_text = str(mime or "")
    if not mime_text.startswith(("image", "video")):
        return False, "unsupported_mime"

    if provider_name in DOCUMENT_OCR_PROVIDER_NAMES:
        return False, "document_ocr_provider"

    if getattr(args, "ocr_model", ""):
        return False, "ocr_route"

    if getattr(args, "document_image", False):
        return False, "document_image"

    return True, "visual_caption"


def resolve_directory_name_context(
    *,
    args: Any,
    uri: str,
    mime: str,
    provider_name: str = "",
    media: Any = None,
    source_uri: str | None = None,
) -> DirectoryNameContext:
    """Resolve the user-enabled directory-name context for a provider call."""
    if not getattr(args, "dir_name", False):
        return DirectoryNameContext(enabled=False, applicable=False, reason="disabled")

    resolved_source_uri = _resolve_source_uri(args=args, uri=uri, media=media, source_uri=source_uri)
    raw_name = _raw_directory_name(resolved_source_uri)
    applicable, reason = _is_applicable(args=args, mime=mime, provider_name=provider_name)
    if not applicable:
        return DirectoryNameContext(
            enabled=True,
            applicable=False,
            source_uri=resolved_source_uri,
            raw_directory_name=raw_name,
            reason=reason,
        )

    if not raw_name:
        return DirectoryNameContext(
            enabled=True,
            applicable=True,
            source_uri=resolved_source_uri,
            raw_directory_name=raw_name,
            reason="empty_directory_name",
        )

    character_name = split_name_series(raw_name)
    if not character_name:
        return DirectoryNameContext(
            enabled=True,
            applicable=True,
            source_uri=resolved_source_uri,
            raw_directory_name=raw_name,
            reason="empty_character_name",
        )

    character_prompt = (
        "If there is a person/character or more in the image you must refer to them as "
        f"{character_name}.\n"
    )
    return DirectoryNameContext(
        enabled=True,
        applicable=True,
        source_uri=resolved_source_uri,
        raw_directory_name=raw_name,
        character_name=character_name,
        character_prompt=character_prompt,
        reason="visual_caption",
    )


def apply_directory_name_context(prompts: Any, context: DirectoryNameContext) -> Any:
    """Return prompts with directory-name prompt prepended when applicable."""
    if not context.has_prompt:
        return prompts
    return prompts.with_character(context.character_name, context.character_prompt)
