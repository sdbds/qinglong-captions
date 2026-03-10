# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.nanonets
"""
from module.providers.ocr.nanonets import (
    DEFAULT_OCR_PROMPT,
    _build_messages,
    attempt_nanonets_ocr,
)

__all__ = ["DEFAULT_OCR_PROMPT", "_build_messages", "attempt_nanonets_ocr"]
