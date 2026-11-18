from __future__ import annotations

import os
import shutil
import tempfile
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


def publish_staged_outputs(
    staging_dir: Path,
    item_dir: Path,
    *,
    managed_names: Iterable[str],
    metadata_name: str,
    preserve_previous: bool = False,
) -> Path | None:
    """Publish with rollback; return the retained backup when requested."""
    names = set(managed_names) | {metadata_name}
    if any(Path(name).name != name or name in {"", ".", ".."} for name in names):
        raise ValueError("Output names must be leaf names")
    item_dir = Path(item_dir)
    if not (Path(staging_dir) / metadata_name).is_file():
        raise ValueError("Staged output metadata is required")
    if any((item_dir / name).exists() and not (item_dir / name).is_file() for name in names):
        raise ValueError("An output destination is not a regular file")
    backup = Path(tempfile.mkdtemp(prefix=".previous-", dir=item_dir))
    moved: list[str] = []
    installed: list[str] = []
    try:
        # Invalidate the completion marker before changing any generation's artifacts.
        if (item_dir / metadata_name).exists():
            os.replace(item_dir / metadata_name, backup / metadata_name)
            moved.append(metadata_name)
        for name in sorted(names - {metadata_name}):
            old = item_dir / name
            if old.exists():
                os.replace(old, backup / name)
                moved.append(name)
            new = Path(staging_dir) / name
            if new.is_file():
                os.replace(new, old)
                installed.append(name)
        os.replace(Path(staging_dir) / metadata_name, item_dir / metadata_name)
        installed.append(metadata_name)
    except BaseException:
        try:
            for name in reversed(installed):
                (item_dir / name).unlink(missing_ok=True)
            for name in reversed(moved):
                os.replace(backup / name, item_dir / name)
        except OSError as exc:
            raise RuntimeError(f"Output rollback failed; previous outputs are retained in {backup}") from exc
        shutil.rmtree(backup, ignore_errors=True)
        raise
    if preserve_previous and moved:
        return backup
    shutil.rmtree(backup, ignore_errors=True)
    return None


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
