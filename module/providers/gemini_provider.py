# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.vision_api.gemini
"""
from module.providers.vision_api.gemini import attempt_gemini, _collect_stream_gemini, _save_binary_file

__all__ = ["attempt_gemini", "_collect_stream_gemini", "_save_binary_file"]
