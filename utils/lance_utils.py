from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

_TAG_SANITIZE_RE = re.compile(r"[^0-9A-Za-z._-]+")


def sanitize_tag_component(value: Any) -> str:
    text = _TAG_SANITIZE_RE.sub("_", str(value).strip())
    text = text.strip("._-")
    return text or "default"


def build_version_tag(*parts: Any, timestamp: Optional[str] = None) -> str:
    cleaned = [sanitize_tag_component(part) for part in parts if part not in (None, "")]
    cleaned.append(timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    return ".".join(cleaned)


def get_latest_version_number(dataset: Any) -> int:
    version_attr = getattr(dataset, "version", None)
    if isinstance(version_attr, int):
        return version_attr

    latest_version = 1
    if hasattr(dataset, "versions"):
        try:
            versions = dataset.versions()
        except Exception:
            versions = []
        for version_info in versions:
            if isinstance(version_info, dict):
                candidate = version_info.get("version")
            else:
                candidate = getattr(version_info, "version", None)
            if isinstance(candidate, int):
                latest_version = max(latest_version, candidate)

    return latest_version


def update_or_create_tag(dataset: Any, tag: str, version: Optional[int] = None) -> int:
    version_number = get_latest_version_number(dataset) if version is None else version
    try:
        dataset.tags.create(tag, version_number)
    except Exception:
        dataset.tags.update(tag, version_number)
    return version_number
