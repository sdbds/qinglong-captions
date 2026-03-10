# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.vision_api.pixtral
"""
from module.providers.vision_api.pixtral import attempt_pixtral

__all__ = ["attempt_pixtral"]
