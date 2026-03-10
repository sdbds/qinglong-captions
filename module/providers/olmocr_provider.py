# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.olmocr
"""
from module.providers.ocr.olmocr import attempt_olmocr, _generate_for_image

__all__ = ["attempt_olmocr", "_generate_for_image"]
