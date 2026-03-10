# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.cloud_vlm.stepfun
"""
from module.providers.cloud_vlm.stepfun import attempt_stepfun, _collect_stream_stepfun

__all__ = ["attempt_stepfun", "_collect_stream_stepfun"]
