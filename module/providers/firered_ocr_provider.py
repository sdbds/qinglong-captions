# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.firered
"""
from module.providers.ocr.firered import (
    DEFAULT_OCR_PROMPT,
    _build_messages,
    attempt_firered_ocr,
)

__all__ = ["DEFAULT_OCR_PROMPT", "_build_messages", "attempt_firered_ocr"]
