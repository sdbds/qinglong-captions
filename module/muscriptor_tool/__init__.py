"""MuScriptor batch transcription integration."""

from .options import (
    BatchOptions,
    DecodingMode,
    ModelVariant,
    OutputFormat,
    PreviewContent,
    PreviewFormat,
    PreviewRequest,
    TranscriptionOptions,
)

MUSCRIPTOR_VERSION = "0.4.0"

__all__ = [
    "MUSCRIPTOR_VERSION",
    "BatchOptions",
    "DecodingMode",
    "ModelVariant",
    "OutputFormat",
    "PreviewContent",
    "PreviewFormat",
    "PreviewRequest",
    "TranscriptionOptions",
]
