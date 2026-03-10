# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.chandra
"""
from module.providers.ocr.chandra import attempt_chandra_ocr

__all__ = ["attempt_chandra_ocr"]
