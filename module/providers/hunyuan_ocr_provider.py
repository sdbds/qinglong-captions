# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.hunyuan
"""
from module.providers.ocr.hunyuan import attempt_hunyuan_ocr, _build_messages, DEFAULT_OCR_PROMPT

__all__ = ["attempt_hunyuan_ocr", "_build_messages", "DEFAULT_OCR_PROMPT"]
