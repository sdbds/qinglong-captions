# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.ocr.paddle
"""
from module.providers.ocr.paddle import (
    _filter_pipeline_kwargs,
    _run_pipeline,
    attempt_paddle_ocr,
)

__all__ = ["_filter_pipeline_kwargs", "_run_pipeline", "attempt_paddle_ocr"]
