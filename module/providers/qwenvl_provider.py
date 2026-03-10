# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.cloud_vlm.qwenvl
"""
from module.providers.cloud_vlm.qwenvl import attempt_qwenvl, _collect_stream_qwen

__all__ = ["attempt_qwenvl", "_collect_stream_qwen"]
