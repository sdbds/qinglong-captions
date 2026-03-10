# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.local_vlm.moondream
"""
from module.providers.local_vlm.moondream import attempt_moondream

__all__ = ["attempt_moondream"]
