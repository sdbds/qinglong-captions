from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"})
SEPARATION_METADATA_NAME = "separation_metadata.json"
STEM_MIDI_OUTPUT_DIRNAME = "04_stem_midi"


@dataclass(frozen=True)
class BatchItem:
    source_path: Path
    relative_path: Path


def _is_supported_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def is_separation_output_dir(path: Path) -> bool:
    return (path / SEPARATION_METADATA_NAME).is_file()


def discover_inputs(
    input_path: Path,
    *,
    output_dir: Path | None,
    recursive: bool,
    skip_separation_outputs: bool = False,
) -> tuple[BatchItem, ...]:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if input_path.is_file():
        if not _is_supported_audio(input_path):
            raise ValueError(f"Unsupported audio file: {input_path}")
        return (BatchItem(source_path=input_path, relative_path=Path(input_path.name)),)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path == output_dir:
        raise ValueError("input directory and output directory must be different")

    found: list[BatchItem] = []
    if recursive:
        for directory, directory_names, file_names in os.walk(input_path, followlinks=False):
            directory_path = Path(directory)
            retained_directories: list[str] = []
            for name in sorted(directory_names, key=str.casefold):
                candidate = directory_path / name
                if candidate.is_symlink() or candidate.resolve() == output_dir or (
                    skip_separation_outputs and is_separation_output_dir(candidate)
                ):
                    continue
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in sorted(file_names, key=str.casefold):
                candidate = directory_path / name
                if candidate.is_symlink() or not _is_supported_audio(candidate):
                    continue
                found.append(BatchItem(candidate.resolve(), candidate.resolve().relative_to(input_path)))
    else:
        for candidate in sorted(input_path.iterdir(), key=lambda path: path.name.casefold()):
            if candidate.is_symlink() or not _is_supported_audio(candidate):
                continue
            found.append(BatchItem(candidate.resolve(), Path(candidate.name)))

    return tuple(sorted(found, key=lambda item: item.relative_path.as_posix().casefold()))


def item_output_dir(output_dir: Path, item: BatchItem) -> Path:
    return Path(output_dir) / item.relative_path
