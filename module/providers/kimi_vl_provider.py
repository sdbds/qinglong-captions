# -*- coding: utf-8 -*-
"""
Backward-compatible re-export.
Actual implementation moved to module.providers.cloud_vlm.kimi_vl
"""
from module.providers.cloud_vlm.kimi_vl import (
    attempt_kimi_vl,
    _collect_stream_kimi,
    _parse_kimi_response,
    _load_tags_from_json,
    _inject_tags_into_messages,
)

__all__ = [
    "attempt_kimi_vl",
    "_collect_stream_kimi",
    "_parse_kimi_response",
    "_load_tags_from_json",
    "_inject_tags_into_messages",
]
