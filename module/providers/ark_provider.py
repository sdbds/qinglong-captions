# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.cloud_vlm.ark
"""
from module.providers.cloud_vlm.ark import attempt_ark, _collect_stream_ark

__all__ = ["attempt_ark", "_collect_stream_ark"]
