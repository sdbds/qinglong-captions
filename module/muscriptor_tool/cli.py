from __future__ import annotations

import inspect
import json
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import List, Optional

import click
import typer
from rich.console import Console
from rich.markup import escape
from typer.core import TyperArgument, TyperOption

from utils.rich_progress import create_caption_progress

from .auralization import preflight_preview
from .batch import default_output_dir, run_batch
from .options import (
    DEFAULT_CFG_COEF,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_PREVIEW_FORMAT,
    DEFAULT_TEMPERATURE,
    BatchOptions,
    DecodingMode,
    ModelVariant,
    OutputFormat,
    PreviewContent,
    PreviewFormat,
    PreviewRequest,
    TranscriptionOptions,
)
from .outputs import OutputTargets, transcribe_once
from .runtime import list_instruments, load_model, muscriptor_version, resolve_instruments


def _patch_click_82_argument_compatibility() -> None:
    if (
        len(inspect.signature(TyperArgument.make_metavar).parameters) != 1
        or len(inspect.signature(click.core.Parameter.make_metavar).parameters) != 2
    ):
        return

    parameter_make_metavar = click.core.Parameter.make_metavar

    def parameter_compatibility(
        parameter: click.Parameter,
        ctx: click.Context | None = None,
    ) -> str:
        return parameter_make_metavar(
            parameter,
            ctx or click.get_current_context(silent=True),
        )

    click.core.Parameter.make_metavar = parameter_compatibility

    def make_metavar(argument: TyperArgument, ctx: click.Context | None = None) -> str:
        if argument.metavar is not None:
            return argument.metavar
        value = (argument.name or "").upper()
        if not argument.required:
            value = f"[{value}]"
        try:
            type_metavar = argument.type.get_metavar(param=argument, ctx=ctx)
        except TypeError:
            type_metavar = argument.type.get_metavar(argument)
        if type_metavar:
            value += f":{type_metavar}"
        if argument.nargs != 1:
            value += "..."
        return value

    TyperArgument.make_metavar = make_metavar

    def option_make_metavar(option: TyperOption, ctx: click.Context | None = None) -> str:
        return parameter_make_metavar(
            option,
            ctx or click.get_current_context(silent=True),
        )

    TyperOption.make_metavar = option_make_metavar


_patch_click_82_argument_compatibility()


class InstrumentListFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class BatchPreviewMode(str, Enum):
    NONE = "none"
    MIDI = "midi"
    COMPARISON = "comparison"


app = typer.Typer(
    add_completion=False,
    help="Transcribe audio with the official MuScriptor models.",
    no_args_is_help=True,
)


def _instrument_values(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _auto_batch_size(raw: int) -> int | None:
    return None if raw == 0 else raw


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _single_output_path(source: Path, output: str | None, output_format: OutputFormat) -> Path | None:
    if output == "-":
        return None
    if output:
        return Path(output).expanduser()
    suffix = {
        OutputFormat.MIDI: ".mid",
        OutputFormat.JSON: ".json",
        OutputFormat.JSONL: ".jsonl",
    }[output_format]
    return source.with_suffix(suffix)


def _validate_single_paths(source: Path, output: Path | None, preview: Path | None) -> None:
    named_paths = [("input", _resolved(source))]
    if output is not None:
        named_paths.append(("output", _resolved(output)))
    if preview is not None:
        named_paths.append(("preview", _resolved(preview)))
    for index, (left_name, left_path) in enumerate(named_paths):
        for right_name, right_path in named_paths[index + 1 :]:
            if left_path == right_path:
                raise ValueError(f"{left_name} and {right_name} paths must be different")


def _preview_request(
    preview_path: Path | None,
    preview_mode: PreviewContent | None,
) -> PreviewRequest | None:
    if preview_path is None:
        if preview_mode is not None:
            raise ValueError("--preview-mode requires --preview")
        return None
    if str(preview_path) == "-":
        raise ValueError("--preview must be a WAV or MP3 file path, not stdout")
    suffix = preview_path.suffix.lower().lstrip(".")
    try:
        preview_format = PreviewFormat(suffix)
    except ValueError as exc:
        raise ValueError("--preview path must end in .wav or .mp3") from exc
    return PreviewRequest(
        content=preview_mode or PreviewContent.COMPARISON,
        format=preview_format,
    )


def _single_targets(output_format: OutputFormat, output_path: Path | None) -> OutputTargets:
    if output_path is not None:
        return OutputTargets(
            midi=output_path if output_format is OutputFormat.MIDI else None,
            json=output_path if output_format is OutputFormat.JSON else None,
            jsonl=output_path if output_format is OutputFormat.JSONL else None,
        )
    if output_format is OutputFormat.MIDI:
        return OutputTargets(midi_stream=click.get_binary_stream("stdout"))
    if output_format is OutputFormat.JSON:
        return OutputTargets(json_stream=click.get_text_stream("stdout"))
    return OutputTargets(jsonl_stream=click.get_text_stream("stdout"))


def _parameter_error(exc: BaseException) -> None:
    raise typer.BadParameter(str(exc)) from exc


def _stderr_console() -> Console:
    return Console(
        file=click.get_text_stream("stderr"),
        color_system="truecolor",
        force_terminal=True,
    )


class _ChunkProgressReporter:
    def __init__(self, console: Console):
        self._progress = create_caption_progress(console, transient=False, expand=True)
        self._task_id: int | None = None
        self._label: str | None = None
        self._started = False

    def __enter__(self) -> "_ChunkProgressReporter":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        if self._started:
            self._progress.stop()
        return False

    def update(self, label: str, completed: int, total: int) -> None:
        if not self._started:
            self._progress.start()
            self._started = True
        description = f"[bold cyan]Transcribing[/bold cyan] {escape(label)}"
        if self._task_id is None:
            self._task_id = self._progress.add_task(
                description,
                total=total,
                completed=completed,
            )
            self._label = label
            return
        self._progress.update(
            self._task_id,
            description=description if label != self._label else None,
            completed=completed,
            total=total,
        )
        self._label = label


def _runtime_failure(exc: BaseException, *, console: Console | None = None) -> None:
    (console or _stderr_console()).print(
        f"Error: {exc}",
        style="red",
        markup=False,
        highlight=False,
    )
    raise typer.Exit(code=1) from exc


@app.command()
def transcribe(
    input_audio: Path = typer.Argument(..., metavar="INPUT_AUDIO"),
    output: Optional[str] = typer.Option(None, "--output", "-o", metavar="PATH|-"),
    output_format: OutputFormat = typer.Option(OutputFormat.MIDI.value, "--format", "-f"),
    notes: bool = typer.Option(False, "--notes"),
    sampling: bool = typer.Option(False, "--sampling"),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, "--temperature", "-t"),
    cfg_coef: float = typer.Option(DEFAULT_CFG_COEF, "--cfg-coef"),
    model: ModelVariant = typer.Option(DEFAULT_MODEL.value, "--model", "-m"),
    device: str = typer.Option(DEFAULT_DEVICE, "--device", "-d"),
    batch_size: int = typer.Option(
        0,
        "--batch-size",
        "-b",
        help="5-second audio chunks per inference batch; 0 selects the runtime default.",
    ),
    strict_eos: bool = typer.Option(False, "--strict-eos"),
    beam_size: int = typer.Option(1, "--beam-size"),
    preview: Optional[Path] = typer.Option(None, "--preview", "--auralize", metavar="PATH"),
    preview_mode: Optional[PreviewContent] = typer.Option(None, "--preview-mode"),
    instruments: str = typer.Option("", "--instruments", metavar="NAME[,NAME...]"),
) -> None:
    source = Path(input_audio).expanduser()
    output_path = _single_output_path(source, output, output_format)
    preview_path = Path(preview).expanduser() if preview is not None else None
    try:
        request = _preview_request(preview_path, preview_mode)
        _validate_single_paths(source, output_path, preview_path)
        raw_instruments = _instrument_values(instruments)
        options = TranscriptionOptions.from_single_cli(
            model=model,
            device=device,
            batch_size=_auto_batch_size(batch_size),
            sampling=sampling,
            temperature=temperature,
            cfg_coef=cfg_coef,
            strict_eos=strict_eos,
            beam_size=beam_size,
            instruments=raw_instruments,
            print_notes=notes,
        )
    except (TypeError, ValueError) as exc:
        _parameter_error(exc)

    if not source.is_file():
        _runtime_failure(FileNotFoundError(f"Input audio file does not exist: {source}"))
    try:
        canonical_instruments = resolve_instruments(options.instruments)
        if canonical_instruments != options.instruments:
            options = replace(options, instruments=canonical_instruments)
    except (TypeError, ValueError) as exc:
        _parameter_error(exc)

    console = _stderr_console()
    try:
        chunk_progress = _ChunkProgressReporter(console)
        with chunk_progress:
            preview_runtime = preflight_preview(request) if request is not None else None
            console.print(
                f"Loading official MuScriptor {options.model.value} model on {options.device}",
                style="cyan",
                markup=False,
                highlight=False,
            )
            loaded = load_model(options, console=console)
            result = transcribe_once(
                loaded,
                source,
                options,
                _single_targets(output_format, output_path),
                stderr=click.get_text_stream("stderr"),
                progress_callback=lambda completed, total: chunk_progress.update(
                    source.name,
                    completed,
                    total,
                ),
                preview_runtime=preview_runtime,
                preview_target=preview_path,
            )
        console.print(
            f"Resolved device: {loaded.resolved_device}",
            style="cyan",
            markup=False,
            highlight=False,
        )
        for warning in result.warnings:
            console.print(
                f"Warning: {warning}",
                style="yellow",
                markup=False,
                highlight=False,
            )
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except Exception as exc:
        _runtime_failure(exc, console=console)


@app.command("batch")
def batch_command(
    input_path: Path = typer.Argument(..., metavar="INPUT_PATH"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Defaults to an input-local muscriptor_output directory.",
    ),
    output_formats: List[OutputFormat] = typer.Option(None, "--format", "-f"),
    model: ModelVariant = typer.Option(DEFAULT_MODEL.value, "--model", "-m"),
    device: str = typer.Option(DEFAULT_DEVICE, "--device", "-d"),
    batch_size: int = typer.Option(
        0,
        "--batch-size",
        "-b",
        help="5-second audio chunks per inference batch; 0 selects the runtime default.",
    ),
    instruments: str = typer.Option("", "--instruments", metavar="NAME[,NAME...]"),
    preview_mode: BatchPreviewMode = typer.Option(BatchPreviewMode.NONE.value, "--preview-mode"),
    preview_format: Optional[PreviewFormat] = typer.Option(None, "--preview-format"),
    decode_mode: DecodingMode = typer.Option(DecodingMode.GREEDY.value, "--decode-mode"),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, "--temperature"),
    cfg_coef: float = typer.Option(DEFAULT_CFG_COEF, "--cfg-coef"),
    beam_size: Optional[int] = typer.Option(None, "--beam-size"),
    strict_eos: bool = typer.Option(False, "--strict-eos"),
    notes: bool = typer.Option(False, "--notes"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
) -> None:
    try:
        if preview_mode is BatchPreviewMode.NONE:
            if preview_format is not None:
                raise ValueError("--preview-format requires --preview-mode midi or comparison")
            preview_request = None
        else:
            preview_request = PreviewRequest(
                content=PreviewContent(preview_mode.value),
                format=preview_format or DEFAULT_PREVIEW_FORMAT,
            )
        canonical_instruments = resolve_instruments(_instrument_values(instruments))
        transcription = TranscriptionOptions.from_batch_cli(
            model=model,
            device=device,
            batch_size=_auto_batch_size(batch_size),
            decode_mode=decode_mode,
            temperature=temperature,
            cfg_coef=cfg_coef,
            strict_eos=strict_eos,
            beam_size=beam_size,
            instruments=canonical_instruments,
            print_notes=notes,
        )
        options = BatchOptions(
            transcription=transcription,
            output_formats=tuple(output_formats or (OutputFormat.MIDI,)),
            preview=preview_request,
            recursive=recursive,
            skip_completed=skip_completed,
            fail_fast=fail_fast,
        )
    except (TypeError, ValueError) as exc:
        _parameter_error(exc)

    console = _stderr_console()
    try:
        with _ChunkProgressReporter(console) as chunk_progress:
            summary = run_batch(
                input_path,
                default_output_dir(input_path) if output_dir is None else output_dir,
                options,
                model_loader=lambda model_options: load_model(model_options, console=console),
                log_callback=lambda message: console.print(message, markup=False, highlight=False),
                chunk_progress_callback=chunk_progress.update,
            )
        console.print(
            " ".join(
                (
                    f"discovered={summary.discovered}",
                    f"processed={summary.processed}",
                    f"skipped={summary.skipped}",
                    f"partial={summary.partial}",
                    f"failed={summary.failed}",
                    f"elapsed={summary.elapsed_seconds:.3f}s",
                )
            ),
            style="bold",
            markup=False,
            highlight=False,
        )
        if summary.exit_code:
            raise typer.Exit(code=summary.exit_code)
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as exc:
        _runtime_failure(exc, console=console)


@app.command("list-instruments")
def list_instruments_command(
    output_format: InstrumentListFormat = typer.Option(InstrumentListFormat.TEXT.value, "--format", "-f"),
) -> None:
    try:
        names = list_instruments()
        if output_format is InstrumentListFormat.JSON:
            typer.echo(
                json.dumps(
                    {
                        "schema_version": 1,
                        "package_version": muscriptor_version(),
                        "instruments": list(names),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            return
        for name in names:
            typer.echo(name)
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except Exception as exc:
        _runtime_failure(exc)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
