# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.cloud_vlm.glm
"""
from module.providers.cloud_vlm.glm import attempt_glm, _collect_stream_glm

__all__ = ["attempt_glm", "_collect_stream_glm"]
