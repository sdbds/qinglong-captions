from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def is_progress_event(event: Any, progress_event_type: type | None = None) -> bool:
    if progress_event_type is not None and isinstance(event, progress_event_type):
        return True
    return (
        hasattr(event, "completed")
        and hasattr(event, "total")
        and not hasattr(event, "pitch")
        and not hasattr(event, "end_time")
    )


def event_to_dict(event: Any) -> dict[str, Any]:
    if all(hasattr(event, name) for name in ("pitch", "start_time", "index", "instrument")):
        return {
            "type": "start",
            "pitch": int(event.pitch),
            "start_time": float(event.start_time),
            "index": int(event.index),
            "instrument": str(event.instrument),
        }

    if hasattr(event, "end_time"):
        if hasattr(event, "start_event_index"):
            start_event_index = event.start_event_index
        elif hasattr(event, "start_event") and hasattr(event.start_event, "index"):
            start_event_index = event.start_event.index
        else:
            raise TypeError("Unsupported MuScriptor event: end event has no start reference")
        return {
            "type": "end",
            "end_time": float(event.end_time),
            "start_event_index": int(start_event_index),
        }

    raise TypeError(f"Unsupported MuScriptor event: {type(event).__name__}")


@dataclass
class EventStats:
    note_count: int = 0
    event_count: int = 0
    chunk_count: int = 0
    completed_chunks: int = 0

    def observe_event(self, event: Mapping[str, Any]) -> None:
        self.event_count += 1
        if event.get("type") == "start":
            self.note_count += 1

    def observe_progress(self, *, completed: int, total: int) -> None:
        self.completed_chunks = max(self.completed_chunks, int(completed))
        self.chunk_count = max(self.chunk_count, int(total))
