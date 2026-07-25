from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

ExportWriter = Callable[[Path], object]
ExportValidator = Callable[[Path], object]


@dataclass(frozen=True)
class ExportJob:
    format: str
    target: Path
    writer: ExportWriter
    validator: ExportValidator

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", str(self.format).strip())
        object.__setattr__(self, "target", Path(self.target))
        if not self.format:
            raise ValueError("Export format must not be empty")


@dataclass(frozen=True)
class ExportStatus:
    format: str
    target: Path
    ok: bool
    error: str | None = None
    warning: str | None = None


@contextmanager
def atomic_output_path(target: str | Path) -> Iterator[Path]:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary = target.with_name(
        f"{target.stem}.{os.getpid()}.{nonce}.part{target.suffix}"
    )
    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_export_jobs(jobs: Iterable[ExportJob]) -> tuple[ExportStatus, ...]:
    statuses: list[ExportStatus] = []
    for job in jobs:
        try:
            with atomic_output_path(job.target) as temporary:
                writer_result = job.writer(temporary)
                job.validator(temporary)
        except Exception as exc:
            statuses.append(
                ExportStatus(
                    format=job.format,
                    target=job.target,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            warning = (
                writer_result.strip()
                if isinstance(writer_result, str) and writer_result.strip()
                else None
            )
            statuses.append(
                ExportStatus(
                    format=job.format,
                    target=job.target,
                    ok=True,
                    warning=warning,
                )
            )
    return tuple(statuses)
