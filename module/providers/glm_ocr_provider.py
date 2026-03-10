# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.glm
"""
from module.providers.ocr.glm import attempt_glm_ocr

__all__ = ["attempt_glm_ocr"]
