"""CLI entry point for MuSViT ONNX full-page OMR."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config
from module.sheet_music_omr.inputs import DEFAULT_PDF_DPI, InputPage
from module.sheet_music_omr.model import (
    CONFIG_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_MAX_LENGTH,
    DEFAULT_MODEL_REPO_ID,
    DEFAULT_MODEL_REVISION,
    MuSViTOnnxRecognizer,
)
from module.sheet_music_omr.pipeline import MuSViTOmrPipeline
from utils.console_util import print_exception

DEFAULT_OUTPUT_DIRNAME = "musvit_omr_output"
DEFAULT_OUTPUT_FORMAT = "musicxml"

console = Console(color_system="truecolor", force_terminal=True)


def _console_safe_text(value: Any, target_console: Console | None = None) -> str:
    text = str(value)
    output_console = target_console or console
    encoding = getattr(output_console.file, "encoding", None)
    if not encoding:
        return text
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def default_output_dir(input_path: str | Path) -> Path:
    source = Path(input_path).expanduser().resolve()
    parent = source if source.is_dir() else source.parent
    return parent / DEFAULT_OUTPUT_DIRNAME


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_defaults(
    config: Mapping[str, Any] | None,
    *,
    config_dir: str | Path = CONFIG_DIR,
) -> Mapping[str, Any]:
    loaded = config if config is not None else load_config(str(config_dir))
    return _as_mapping(loaded.get("musvit"))


def build_parser(
    config: Mapping[str, Any] | None = None,
    *,
    config_dir: str | Path = CONFIG_DIR,
) -> argparse.ArgumentParser:
    defaults = _load_defaults(config, config_dir=config_dir)
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe sheet-music images or PDFs with MuSViT ONNX and "
            "export MusicXML and/or MIDI."
        )
    )
    parser.add_argument("input_path", help="Input image file, PDF file, or directory")
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument(
        "--output_format",
        choices=("musicxml", "midi", "both"),
        default=str(defaults.get("output_format", DEFAULT_OUTPUT_FORMAT)),
    )
    parser.add_argument(
        "--repo_id",
        default=str(defaults.get("repo_id", DEFAULT_MODEL_REPO_ID)),
    )
    parser.add_argument(
        "--revision",
        default=str(defaults.get("revision", DEFAULT_MODEL_REVISION)),
    )
    parser.add_argument(
        "--model_dir",
        default=str(defaults.get("model_dir", DEFAULT_MODEL_DIR)),
    )
    parser.add_argument(
        "--pdf_dpi",
        type=int,
        default=int(defaults.get("pdf_dpi", DEFAULT_PDF_DPI)),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=defaults.get("max_tokens"),
        help="Advanced generation safety limit; defaults to the model maxlen.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("recursive", True)),
    )
    parser.add_argument(
        "--skip_completed",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("skip_completed", True)),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("overwrite", False)),
    )
    parser.add_argument(
        "--force_download",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("force_download", False)),
    )
    return parser


class _ProgressReporter:
    def __init__(self, progress: Progress, task_id: Any) -> None:
        self.progress = progress
        self.task_id = task_id
        self.last_update = 0.0

    def __call__(self, page: InputPage, token_count: int) -> None:
        now = time.monotonic()
        if token_count > 2 and token_count % 50 and now - self.last_update < 0.5:
            return
        self.last_update = now
        source_label = _console_safe_text(
            page.source_path.name,
            self.progress.console,
        )
        if page.page_number is not None:
            source_label += f" page {page.page_number}/{page.page_count or '?'}"
        self.progress.update(
            self.task_id,
            description=(
                f"[bold cyan]Transcribing {source_label} "
                f"({token_count} tokens)"
            ),
        )


def _validate_runtime_args(args: argparse.Namespace) -> None:
    pdf_dpi = int(args.pdf_dpi)
    if pdf_dpi <= 0:
        raise ValueError("pdf_dpi must be positive")

    output_format = str(args.output_format).strip().lower()
    if output_format not in {"musicxml", "midi", "both"}:
        raise ValueError(
            "output_format must be one of: musicxml, midi, both"
        )

    if args.max_tokens is None:
        return
    max_tokens = int(args.max_tokens)
    if max_tokens < 2 or max_tokens > DEFAULT_MODEL_MAX_LENGTH:
        raise ValueError(
            f"max_tokens must be in 2..{DEFAULT_MODEL_MAX_LENGTH}"
        )


def run_sheet_music_musvit(
    args: argparse.Namespace,
    *,
    recognizer_factory: Callable[..., Any] = MuSViTOnnxRecognizer,
    pipeline_factory: Callable[..., Any] = MuSViTOmrPipeline,
) -> int:
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        console.print(
            "[red]Input path does not exist:[/red]",
            Text(_console_safe_text(input_path)),
        )
        return 1

    try:
        _validate_runtime_args(args)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid MuSViT OMR arguments:[/red] {exc}")
        return 1

    raw_output_dir = str(args.output_dir or "").strip()
    output_dir = (
        Path(raw_output_dir).expanduser().resolve()
        if raw_output_dir
        else default_output_dir(input_path)
    )

    try:
        recognizer = recognizer_factory(
            repo_id=args.repo_id,
            revision=args.revision,
            model_dir=args.model_dir,
            force_download=bool(args.force_download),
            logger=console.print,
        )
    except Exception as exc:
        print_exception(console, exc, prefix="Failed to load MuSViT OMR bundle")
        return 1

    console.print(
        "[cyan]MuSViT ONNX providers:[/cyan] "
        + ", ".join(str(provider) for provider in recognizer.providers)
    )
    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("[bold cyan]Preparing OMR...", total=None)
        reporter = _ProgressReporter(progress, task_id)
        try:
            pipeline = pipeline_factory(
                recognizer=recognizer,
                output_dir=output_dir,
                output_format=args.output_format,
                pdf_dpi=int(args.pdf_dpi),
                recursive=bool(args.recursive),
                skip_completed=bool(args.skip_completed),
                overwrite=bool(args.overwrite),
                max_tokens=args.max_tokens,
                progress_callback=reporter,
            )
            result = pipeline.run(input_path)
        except Exception as exc:
            print_exception(progress.console, exc, prefix="MuSViT OMR failed")
            return 1
        progress.update(task_id, description="[bold green]OMR complete")

    console.print(
        f"[bold]Finished.[/bold] processed={result.processed} "
        f"skipped={result.skipped} failed={result.failed}"
    )
    console.print(
        "[green]Manifest:[/green]",
        Text(_console_safe_text(result.manifest_path)),
    )
    return 0 if result.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_sheet_music_musvit(args)


if __name__ == "__main__":
    sys.exit(main())
