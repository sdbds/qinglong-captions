from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from module.audio_separator_core import (
    DEFAULT_AUDIO_SEPARATOR_MODEL_DIR,
    DEFAULT_AUDIO_SEPARATOR_REPO_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HARMONY_SEPARATOR_REPO_ID,
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
HARMONY_OUTPUT_DIRNAME = "02_harmony_split"
HARMONY_DRY_STEM_NAME = "dry_vocal"
HARMONY_BACKING_STEM_NAME = "harmony"


@dataclass(frozen=True)
class HarmonyCandidate:
    source_path: Path
    song_output_dir: Path
    vocal_path: Path


def select_vocal_stem(stems: Mapping[str, torch.Tensor]) -> tuple[str, torch.Tensor] | None:
    ranked: list[tuple[int, str, torch.Tensor]] = []
    for stem_name, waveform in stems.items():
        lowered = stem_name.lower()
        score = 0
        if "vocals" in lowered:
            score += 5
        elif "vocal" in lowered:
            score += 4
        elif "voice" in lowered:
            score += 3
        elif "lead" in lowered:
            score += 1

        if any(token in lowered for token in ("instrumental", "karaoke", "other", "bass", "drum", "guitar", "piano", "inst")):
            score -= 6

        if score > 0:
            ranked.append((score, stem_name, waveform))

    if not ranked:
        return None

    _, stem_name, waveform = max(ranked, key=lambda item: item[0])
    return stem_name, waveform


def rename_harmony_stems(stems: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    dry_vocal: torch.Tensor | None = None
    harmony: torch.Tensor | None = None

    for stem_name, waveform in stems.items():
        lowered = stem_name.lower()
        if dry_vocal is None and any(token in lowered for token in ("vocals", "vocal", "voice", "lead")):
            dry_vocal = waveform
            continue
        if harmony is None:
            harmony = waveform

    renamed: dict[str, torch.Tensor] = {}
    if dry_vocal is not None:
        renamed[HARMONY_DRY_STEM_NAME] = dry_vocal
    if harmony is not None:
        renamed[HARMONY_BACKING_STEM_NAME] = harmony
    return renamed


def build_harmony_output_dir(song_output_dir: Path) -> Path:
    return safe_child_path(song_output_dir, HARMONY_OUTPUT_DIRNAME, default_name=HARMONY_OUTPUT_DIRNAME)


def collect_audio_inputs(input_path: Path) -> tuple[list[Path], Path | None]:
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file: {input_path}")
        return [input_path], None

    lance_inputs = collect_audio_inputs_from_lance(input_path)
    if lance_inputs is not None:
        return lance_inputs

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*")
    files = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    return files, input_path


def collect_audio_inputs_from_lance(input_path: Path) -> tuple[list[Path], Path | None] | None:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".lance":
        dataset = _open_lance_dataset(input_path)
        return _audio_paths_from_lance_dataset(dataset), None

    if not input_path.is_dir():
        return None

    lance_file = next((file for file in input_path.glob("*.lance") if file.is_dir()), None)
    if lance_file is not None:
        dataset = _open_lance_dataset(lance_file)
        return _audio_paths_from_lance_dataset(dataset), input_path

    console.print("[yellow]Converting dataset to Lance format...[/yellow]")
    dataset = _transform_dir_to_lance(input_path)
    console.print("[green]Dataset converted to Lance format[/green]")
    return _audio_paths_from_lance_dataset(dataset), input_path


def _open_lance_dataset(dataset_path: Path):
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("Lance runtime is required to open .lance datasets") from exc

    return lance.dataset(str(dataset_path))


def _transform_dir_to_lance(input_path: Path):
    from module.lanceImport import transform2lance

    dataset = transform2lance(
        str(input_path),
        output_name="dataset",
        save_binary=False,
        not_save_disk=False,
        tag="AudioSeparator",
        include_text_assets=False,
    )
    if dataset is None:
        raise RuntimeError(f"Failed to convert dataset to Lance format: {input_path}")
    return dataset


def _audio_paths_from_lance_dataset(dataset) -> list[Path]:
    table = dataset.to_table(
        columns=["uris"],
        filter=("mime LIKE 'audio/%'"),
    )
    seen: set[Path] = set()
    files: list[Path] = []
    for uri in table.column("uris").to_pylist():
        if not uri:
            continue
        path = Path(str(uri)).expanduser()
        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return sorted(files)


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
    parser = argparse.ArgumentParser(description="Separate audio into 6 stems and optionally run a second harmony split on vocals.")
    parser.add_argument("input_path", help="Input audio file, directory, or .lance dataset")
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
        type=float,
        default=DEFAULT_OVERLAP,
        help="Chunk step in seconds, matching original mdxc_overlap semantics",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks to infer in one ONNX batch; lower values reduce peak VRAM usage",
    )
    parser.add_argument(
        "--harmony_separation",
        action="store_true",
        help="After the 6-stem pass finishes for all songs, run a second split on non-silent vocals to extract dry vocal and harmony",
    )
    parser.add_argument(
        "--harmony_repo_id",
        default=DEFAULT_HARMONY_SEPARATOR_REPO_ID,
        help="Hugging Face repository containing the harmony-split ONNX model",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing song output directories")
    parser.add_argument("--force_download", action="store_true", help="Force re-download model artifacts")
    return parser


def run_audio_separator(args: argparse.Namespace) -> int:
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        console.print(f"[red]Input path does not exist:[/red] {input_path}")
        return 1

    output_root = resolve_output_root(input_path).expanduser()
    audio_files, input_root = collect_audio_inputs(input_path)
    if not audio_files:
        console.print(f"[yellow]No supported audio files found under:[/yellow] {input_path}")
        return 1

    separator = AudioSeparator(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        force_download=bool(args.force_download),
        logger=console.print,
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
    harmony_candidates: list[HarmonyCandidate] = []

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        MofNCompleteColumn(separator="/"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TaskProgressColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        progress_console = progress.console
        task = progress.add_task("[bold cyan]Separating audio...", total=len(audio_files))

        for source_path in audio_files:
            current_output_root = output_root if input_root is not None else source_path.parent
            song_output_dir = build_song_output_dir(
                source_path,
                output_root=current_output_root,
                input_root=input_root,
            )

            if song_output_dir.exists() and not args.overwrite:
                progress_console.print(f"[yellow]Skipping existing song directory:[/yellow] {song_output_dir}")
                skipped += 1
                progress.update(task, advance=1)
                continue

            progress_console.print(f"[blue]Separating:[/blue] {source_path}")
            try:
                stems = separator.separate_file(
                    source_path,
                    segment_size=args.segment_size,
                    overlap=args.overlap,
                    batch_size=args.batch_size,
                )
                song_output_dir.mkdir(parents=True, exist_ok=True)
                selected_vocal = select_vocal_stem(stems) if args.harmony_separation else None
                vocal_output_path: Path | None = None
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
                    if selected_vocal is not None and stem_name == selected_vocal[0]:
                        vocal_output_path = output_path

                if selected_vocal is None and args.harmony_separation:
                    progress_console.print(f"[yellow]No vocal stem detected for harmony split:[/yellow] {source_path}")
                elif selected_vocal is not None and vocal_output_path is not None:
                    harmony_candidates.append(
                        HarmonyCandidate(
                            source_path=source_path,
                            song_output_dir=song_output_dir,
                            vocal_path=vocal_output_path,
                        )
                    )
                progress_console.print(f"[green]Saved stems to:[/green] {song_output_dir}")
                processed += 1
            except Exception as exc:  # pragma: no cover - exercised via CLI smoke or manual runs
                failures += 1
                print_exception(progress_console, exc, prefix=f"Failed to separate {source_path}")
            finally:
                progress.update(task, advance=1)

    harmony_processed = 0
    harmony_skipped = 0

    if args.harmony_separation:
        if not harmony_candidates:
            console.print("[yellow]Harmony split enabled, but no vocal stems were found.[/yellow]")
        else:
            harmony_separator = AudioSeparator(
                repo_id=args.harmony_repo_id,
                model_dir=args.model_dir,
                force_download=bool(args.force_download),
                logger=console.print,
            )
            console.print("[cyan]Harmony Model:[/cyan] " f"{args.harmony_repo_id}")
            console.print("[cyan]Harmony Providers:[/cyan]")
            console.print(Pretty(harmony_separator.providers, indent_guides=True, expand_all=True))

            with Progress(
                "[progress.description]{task.description}",
                SpinnerColumn(spinner_name="dots"),
                MofNCompleteColumn(separator="/"),
                BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
                TaskProgressColumn(),
                TextColumn("|"),
                TimeElapsedColumn(),
                TextColumn("|"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as progress:
                progress_console = progress.console
                task = progress.add_task("[bold magenta]Separating harmony...", total=len(harmony_candidates))

                for candidate in harmony_candidates:
                    harmony_output_dir = build_harmony_output_dir(candidate.song_output_dir)
                    progress_console.print(f"[magenta]Harmony split:[/magenta] {candidate.vocal_path}")
                    try:
                        harmony_stems = harmony_separator.separate_file(
                            candidate.vocal_path,
                            segment_size=harmony_separator.metadata.default_segment_size,
                            overlap=args.overlap,
                            batch_size=args.batch_size,
                        )
                        renamed_stems = rename_harmony_stems(harmony_stems)
                        if not renamed_stems:
                            harmony_skipped += 1
                            progress_console.print(
                                f"[yellow]Harmony model returned no usable stems:[/yellow] {candidate.vocal_path}"
                            )
                            progress.update(task, advance=1)
                            continue

                        harmony_output_dir.mkdir(parents=True, exist_ok=True)
                        for stem_name, waveform in renamed_stems.items():
                            output_path = build_stem_output_path(
                                harmony_output_dir,
                                source_path=candidate.source_path,
                                stem_name=stem_name,
                                model_tag=harmony_separator.model_tag,
                                output_format=args.output_format,
                            )
                            harmony_separator.write_audio(
                                waveform,
                                output_path,
                                output_format=args.output_format,
                            )
                        harmony_processed += 1
                        progress_console.print(f"[green]Saved harmony stems to:[/green] {harmony_output_dir}")
                    except Exception as exc:  # pragma: no cover - exercised via CLI smoke or manual runs
                        failures += 1
                        print_exception(progress_console, exc, prefix=f"Failed harmony split for {candidate.vocal_path}")
                    finally:
                        progress.update(task, advance=1)

    console.print(
        f"[bold]Finished.[/bold] processed={processed} skipped={skipped} "
        f"harmony_processed={harmony_processed} harmony_skipped={harmony_skipped} failed={failures}"
    )
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_audio_separator(args)


if __name__ == "__main__":
    sys.exit(main())
