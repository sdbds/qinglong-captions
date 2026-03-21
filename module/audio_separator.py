from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.pretty import Pretty

from module.audio_separator_core import (
    DEFAULT_AUDIO_SEPARATOR_MODEL_DIR,
    DEFAULT_AUDIO_SEPARATOR_REPO_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_OVERLAP,
    DEFAULT_SEGMENT_SIZE,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_OUTPUT_FORMATS,
    AudioSeparator,
)
from utils.console_util import print_exception
from utils.path_safety import safe_child_path, safe_leaf_name

console = Console(color_system="truecolor", force_terminal=True)


def collect_audio_inputs(input_path: Path, *, recursive: bool) -> tuple[list[Path], Path | None]:
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file: {input_path}")
        return [input_path], None

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    return files, input_path


def resolve_output_root(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path
    return input_path.parent


def build_song_output_dir(source_path: Path, *, output_root: Path, input_root: Path | None = None) -> Path:
    relative_parent = Path()
    if input_root is not None:
        try:
            relative_parent = source_path.parent.relative_to(input_root)
        except ValueError:
            relative_parent = Path()

    song_dir_name = safe_leaf_name(source_path.stem, default_name="song")
    return output_root / relative_parent / song_dir_name


def build_stem_output_path(
    song_output_dir: Path,
    *,
    source_path: Path,
    stem_name: str,
    model_tag: str,
    output_format: str,
) -> Path:
    filename = f"{source_path.stem}_({stem_name})_{model_tag}.{output_format.lower()}"
    return safe_child_path(song_output_dir, filename, default_name=f"{safe_leaf_name(stem_name)}.{output_format.lower()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Separate audio into 6 stems using the BS-ROFO-SW-Fixed ONNX model.")
    parser.add_argument("input_path", help="Input audio file or directory")
    parser.add_argument(
        "--repo_id",
        default=DEFAULT_AUDIO_SEPARATOR_REPO_ID,
        help="Hugging Face repository containing model.onnx and model.json",
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_AUDIO_SEPARATOR_MODEL_DIR,
        help="Local cache directory for the audio separator model",
    )
    parser.add_argument(
        "--output_format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=SUPPORTED_OUTPUT_FORMATS,
        help="Output audio format",
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        default=DEFAULT_SEGMENT_SIZE,
        help="Segment size used to derive chunk_size = hop_length * (segment_size - 1)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Overlap factor between adjacent chunks; higher is slower but smoother",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks to infer in one ONNX batch; lower values reduce peak VRAM usage",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for audio files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing song output directories")
    parser.add_argument("--force_download", action="store_true", help="Force re-download model artifacts")
    return parser


def run_audio_separator(args: argparse.Namespace) -> int:
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        console.print(f"[red]Input path does not exist:[/red] {input_path}")
        return 1

    output_root = resolve_output_root(input_path).expanduser()
    audio_files, input_root = collect_audio_inputs(input_path, recursive=bool(args.recursive))
    if not audio_files:
        console.print(f"[yellow]No supported audio files found under:[/yellow] {input_path}")
        return 1

    separator = AudioSeparator(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        force_download=bool(args.force_download),
    )
    console.print("[cyan]Providers:[/cyan]")
    console.print(Pretty(separator.providers, indent_guides=True, expand_all=True))
    console.print(
        f"[cyan]Model:[/cyan] {args.repo_id} | "
        f"[cyan]Format:[/cyan] {args.output_format} | "
        f"[cyan]Segment:[/cyan] {args.segment_size} | "
        f"[cyan]Overlap:[/cyan] {args.overlap} | "
        f"[cyan]Batch:[/cyan] {args.batch_size}"
    )

    failures = 0
    skipped = 0
    processed = 0

    for source_path in audio_files:
        song_output_dir = build_song_output_dir(
            source_path,
            output_root=output_root,
            input_root=input_root,
        )

        if song_output_dir.exists() and not args.overwrite:
            console.print(f"[yellow]Skipping existing song directory:[/yellow] {song_output_dir}")
            skipped += 1
            continue

        console.print(f"[blue]Separating:[/blue] {source_path}")
        try:
            stems = separator.separate_file(
                source_path,
                segment_size=args.segment_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
            song_output_dir.mkdir(parents=True, exist_ok=True)
            for stem_name, waveform in stems.items():
                output_path = build_stem_output_path(
                    song_output_dir,
                    source_path=source_path,
                    stem_name=stem_name,
                    model_tag=separator.model_tag,
                    output_format=args.output_format,
                )
                separator.write_audio(
                    waveform,
                    output_path,
                    output_format=args.output_format,
                )
            console.print(f"[green]Saved stems to:[/green] {song_output_dir}")
            processed += 1
        except Exception as exc:  # pragma: no cover - exercised via CLI smoke or manual runs
            failures += 1
            print_exception(console, exc, prefix=f"Failed to separate {source_path}")

    console.print(
        f"[bold]Finished.[/bold] processed={processed} skipped={skipped} failed={failures}"
    )
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_audio_separator(args)


if __name__ == "__main__":
    sys.exit(main())
