# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.deepseek
"""
from module.providers.ocr.deepseek import attempt_deepseek_ocr

__all__ = ["attempt_deepseek_ocr"]
