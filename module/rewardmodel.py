# /// script
# dependencies = [
#   "setuptools",
#   "pillow>=11.3",
#   "pylance>=2.0.1",
#   "rich>=13.5.0",
#   "imageio>=2.31.1",
#   "imageio-ffmpeg>=0.4.8",
#   "toml",
#   "tomlkit",
#   "huggingface_hub[hf_xet]>=0.35.2",
#   "torch==2.13.0",
#   "transformers[serving]==4.57.6",
#   "torchvision",
#   "qinglong-score>=0.2.2",
# ]
# ///
"""Score image datasets through the public Qinglong Score contract."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from rich.markup import escape

from module.reward_policy import Threshold, assign_threshold, load_reward_policy
from utils.rich_progress import resolve_rich_console
from utils.transformer_loader import (
    hf_download_reporting,
    suppress_library_progress_bars,
)

PromptSource = Literal["override", "caption", "empty"]
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True, slots=True)
class SourceImage:
    path: str
    caption: object = None


@dataclass(frozen=True, slots=True)
class PreparedImage:
    path: str
    tensor: torch.Tensor
    prompt: str | None
    prompt_source: PromptSource | None


@dataclass(frozen=True, slots=True)
class ScoredImage:
    path: str
    prompt: str | None
    prompt_source: PromptSource | None
    score: float


@dataclass(frozen=True, slots=True)
class RunError:
    scope: Literal["item", "batch"]
    stage: Literal["decode", "score"]
    error_type: str
    message: str
    path: str | None = None
    paths: tuple[str, ...] = ()

    @classmethod
    def item(cls, path: str, error: Exception) -> RunError:
        return cls(
            scope="item",
            stage="decode",
            error_type=type(error).__name__,
            message=str(error),
            path=path,
        )

    @classmethod
    def batch(cls, paths: Sequence[str], error: Exception) -> RunError:
        return cls(
            scope="batch",
            stage="score",
            error_type=type(error).__name__,
            message=str(error),
            paths=tuple(paths),
        )


def resolve_dtype(value: str) -> torch.dtype | None:
    try:
        return {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[value]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {value!r}") from error


def resolve_device(value: str) -> torch.device:
    normalized = str(value or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif normalized == "cuda":
        normalized = "cuda:0"

    if normalized == "cpu":
        return torch.device("cpu")
    if not normalized.startswith("cuda:"):
        raise ValueError(
            f"unsupported device {value!r}; use auto, cpu, cuda, or cuda:<index>"
        )
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {normalized}")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid CUDA device: {normalized}") from error
    index = 0 if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index out of range: {normalized}; "
            f"available count={torch.cuda.device_count()}"
        )
    return torch.device(f"cuda:{index}")


def _caption_text(caption: object) -> str:
    if isinstance(caption, str):
        return caption.strip()
    if isinstance(caption, Sequence) and not isinstance(caption, (bytes, bytearray)):
        values = [value.strip() for value in caption if isinstance(value, str) and value.strip()]
        return "\n".join(values)
    return ""


def select_prompt(
    caption: object,
    override: str | None,
    requires_prompts: bool,
) -> tuple[str | None, PromptSource | None]:
    if not requires_prompts:
        return None, None
    if isinstance(override, str) and override.strip():
        return override, "override"
    normalized_caption = _caption_text(caption)
    if normalized_caption:
        return normalized_caption, "caption"
    return "", "empty"


def decode_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _prepare_image(
    record: SourceImage,
    *,
    prompt_override: str | None,
    requires_prompts: bool,
) -> PreparedImage | RunError:
    try:
        tensor = decode_image(record.path)
    except Exception as error:
        return RunError.item(record.path, error)
    prompt, source = select_prompt(
        record.caption,
        prompt_override,
        requires_prompts,
    )
    return PreparedImage(
        path=record.path,
        tensor=tensor,
        prompt=prompt,
        prompt_source=source,
    )


def _validate_scores(scores: Any, *, batch_size: int) -> torch.Tensor:
    if not isinstance(scores, torch.Tensor):
        raise RuntimeError("scorer output must be a torch.Tensor")
    if scores.shape != (batch_size,):
        raise RuntimeError(
            f"scorer output must have shape [{batch_size}], got {list(scores.shape)}"
        )
    if not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError("scorer output must contain only finite values")
    return scores


def score_source_batch(
    records: Sequence[SourceImage],
    *,
    scorer: Any,
    requires_prompts: bool,
    prompt_override: str | None,
    max_workers: int = 16,
) -> tuple[list[ScoredImage], list[RunError]]:
    if not records:
        return [], []

    worker_count = max(1, min(max_workers, len(records)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        prepared_or_errors = list(
            executor.map(
                lambda record: _prepare_image(
                    record,
                    prompt_override=prompt_override,
                    requires_prompts=requires_prompts,
                ),
                records,
            )
        )

    errors = [value for value in prepared_or_errors if isinstance(value, RunError)]
    prepared = [
        value for value in prepared_or_errors if isinstance(value, PreparedImage)
    ]
    groups: dict[tuple[int, int], list[PreparedImage]] = defaultdict(list)
    for image in prepared:
        groups[tuple(image.tensor.shape[1:])].append(image)

    items: list[ScoredImage] = []
    for group in groups.values():
        paths = [image.path for image in group]
        try:
            images = torch.stack([image.tensor for image in group]).to(
                device=scorer.device,
                dtype=scorer.input_dtype,
            )
            prompts = [image.prompt or "" for image in group] if requires_prompts else None
            with torch.inference_mode():
                scores = _validate_scores(
                    scorer.score(images, prompts),
                    batch_size=len(group),
                )
            host_scores = scores.detach().to(device="cpu", dtype=torch.float32).tolist()
        except Exception as error:
            errors.append(RunError.batch(paths, error))
            continue

        items.extend(
            ScoredImage(
                path=image.path,
                prompt=image.prompt,
                prompt_source=image.prompt_source,
                score=float(score),
            )
            for image, score in zip(group, host_scores, strict=True)
        )
    return items, errors


def _log_scored_items(items: Sequence[ScoredImage], *, console: Any) -> None:
    for item in items:
        console.print(
            f"[cyan]{escape(item.path)}[/cyan] "
            f"[bold green]score={item.score:.4f}[/bold green]"
        )


def _failed_image_count(errors: Sequence[RunError]) -> int:
    return sum(1 if error.scope == "item" else len(error.paths) for error in errors)


def _log_run_configuration(
    *,
    console: Any,
    qinglong_score_version: str,
    scorer_name: str,
    selected_checkpoint: Any,
    scorer: Any,
    compute_dtype: torch.dtype,
    thresholds: Sequence[Threshold],
) -> None:
    console.print(f"[cyan]Qinglong Score: {escape(str(qinglong_score_version))}[/cyan]")
    console.print(f"[cyan]Scorer: {escape(scorer_name)}[/cyan]")
    console.print(
        f"[cyan]Checkpoint: {escape(str(selected_checkpoint.identifier))}[/cyan]"
    )
    console.print(
        "[cyan]Runtime: "
        f"device={escape(str(scorer.device))}, "
        f"compute_dtype={escape(str(compute_dtype))}, "
        f"input_dtype={escape(str(scorer.input_dtype))}, "
        f"attention_backend={escape(str(scorer.attention_backend))}[/cyan]"
    )
    if thresholds:
        profile = ", ".join(
            f"{escape(row.name)}<={row.max_score:g}" for row in thresholds
        )
        console.print(f"[cyan]Thresholds: {profile}[/cyan]")
    else:
        console.print("[cyan]Thresholds: disabled[/cyan]")

    checkpoint_identity = scorer.checkpoint_identity
    for artifact in getattr(checkpoint_identity, "artifacts", ()):
        filename = f"/{artifact.filename}" if artifact.filename else ""
        console.print(
            "[dim]Artifact: "
            f"{escape(str(artifact.provider))}:"
            f"{escape(str(artifact.repository))}@"
            f"{escape(str(artifact.revision))}"
            f"{escape(filename)}[/dim]"
        )
    local_path = getattr(checkpoint_identity, "path", None)
    if local_path is not None:
        console.print(f"[dim]Artifact: {escape(str(local_path))}[/dim]")


def _log_run_errors(errors: Sequence[RunError], *, console: Any) -> None:
    for error in errors:
        paths = (error.path,) if error.scope == "item" else error.paths
        for path in paths:
            console.print(
                f"[red]{escape(error.stage)} failed: {escape(str(path))} "
                f"({escape(error.error_type)}: {escape(error.message)})[/red]"
            )


def _log_run_completion(
    items: Sequence[ScoredImage],
    errors: Sequence[RunError],
    *,
    console: Any,
) -> None:
    failed = _failed_image_count(errors)
    empty_prompts = sum(item.prompt_source == "empty" for item in items)
    if not items:
        label = "Scoring failed"
        style = "red"
    elif errors:
        label = "Scoring completed with errors"
        style = "yellow"
    else:
        label = "Scoring completed"
        style = "green"
    console.print(
        f"[{style}]{label}: scored={len(items)}, failed={failed}, "
        f"empty_prompts={empty_prompts}[/{style}]"
    )


def source_images_from_batch(batch: Any) -> list[SourceImage]:
    names = tuple(batch.schema.names)
    if "uris" not in names:
        raise ValueError("Lance image batch is missing the uris column")
    paths = batch["uris"].to_pylist()
    captions = (
        batch["captions"].to_pylist()
        if "captions" in names
        else [None] * len(paths)
    )
    if len(captions) != len(paths):
        raise ValueError("Lance captions cardinality does not match uris")
    return [
        SourceImage(path=str(path), caption=caption)
        for path, caption in zip(paths, captions, strict=True)
    ]


def _relative_path(path: str | Path, source_root: str | Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(source_root).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"source path is outside the report root: {resolved_path}"
        ) from error


def build_report(
    *,
    items: Sequence[ScoredImage],
    source_root: str | Path,
) -> dict[str, Any]:
    rows: list[tuple[str, float]] = []
    for item in items:
        if not math.isfinite(item.score):
            raise ValueError(f"score must be finite for {item.path}")
        rows.append((_relative_path(item.path, source_root), float(item.score)))

    report: dict[str, Any] = {}
    for relative_path, score in sorted(rows, key=lambda row: row[0]):
        parts = PurePosixPath(relative_path).parts
        if not parts:
            raise ValueError(f"source path has no report name: {relative_path}")
        node = report
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"report path conflicts with file: {relative_path}")
            node = child
        if isinstance(node.get(parts[-1]), dict):
            raise ValueError(f"report path conflicts with directory: {relative_path}")
        node[parts[-1]] = score
    return report


def result_path_for_input(input_path: str | Path) -> Path:
    path = Path(input_path)
    if path.suffix.lower() == ".lance":
        return path.with_name(f"{path.stem}.reward_scores.json")
    return path / "reward_scores.json"


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _partition_output_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    parts = pure.parts
    if pure.anchor or len(parts) < 2 or any(part == ".." or ":" in part or "\\" in part for part in parts):
        raise ValueError(f"Invalid partition manifest path: {relative}")
    path = root
    for part in parts[:-1]:
        path = path / part
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Partition output directory is a symlink: {path}")
        if path.exists() and not path.is_dir():
            raise ValueError(f"Partition output parent is not a directory: {path}")
    return path / parts[-1]


def _partition_output_identity(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Partition output is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"kind": "copy", "sha256": digest.hexdigest()}


def apply_thresholds(
    items: Sequence[ScoredImage],
    *,
    thresholds: Sequence[Threshold],
    source_root: str | Path,
    tracking_checkpoint: bool = False,
    warn: Callable[[str], None] = print,
) -> dict[str, str | None]:
    if not thresholds:
        return {item.path: None for item in items}

    root = Path(source_root).resolve()
    if tracking_checkpoint:
        warn(
            "Warning: scorer thresholds are being used with a tracking checkpoint; "
            "future scores may drift."
        )

    manifest = root / ".reward_partition.json"
    if manifest.is_symlink():
        raise ValueError("Partition manifest must not be a symlink")
    owned: dict[str, Any] = {}
    if manifest.exists():
        try:
            owned = json.loads(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise ValueError("Invalid partition manifest") from error
        if not isinstance(owned, dict):
            raise ValueError("Invalid partition manifest")
    previous = {}
    for relative, identity in owned.items():
        path = _partition_output_path(root, relative)
        if not isinstance(identity, dict) or identity.get("kind") not in ("copy", "symlink"):
            raise ValueError("Invalid partition manifest identity")
        actual = _partition_output_identity(path)
        if actual is not None and actual != identity:
            raise ValueError(f"Refusing to remove modified partition output: {path}")
        previous[path] = identity

    # Check every destination before removing any previously generated output.
    buckets: dict[str, str | None] = {}
    planned = []
    for item in items:
        threshold = assign_threshold(item.score, thresholds)
        buckets[item.path] = threshold.name if threshold else None
        if threshold is None:
            continue
        relative = f"{threshold.folder_name}/{_relative_path(item.path, root)}"
        target = _partition_output_path(root, relative)
        if (target.exists() or target.is_symlink()) and target not in previous:
            raise ValueError(f"Refusing to overwrite unowned partition output: {target}")
        planned.append((Path(item.path).resolve(), target, relative))

    for path in previous:
        path.unlink(missing_ok=True)
    current = {}
    try:
        for source, target, relative in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise ValueError(f"Refusing to overwrite unowned partition output: {target}")
            copied = False
            created = False
            try:
                try:
                    target.symlink_to(source)
                    created = True
                except OSError:
                    with source.open("rb") as source_stream, target.open("xb") as destination:
                        created = True
                        shutil.copyfileobj(source_stream, destination)
                    shutil.copystat(source, target)
                    copied = True
                current[relative] = _partition_output_identity(target)
                if copied:
                    warn(f"Symlink unavailable; copied {source} to {target}.")
            except BaseException:
                if created:
                    try:
                        target.unlink()
                    except OSError:
                        current[relative] = _partition_output_identity(target)
                    else:
                        current.pop(relative, None)
                raise
    finally:
        write_json_atomic(manifest, current)
    return buckets


def _resolve_dataset(input_path: str | Path) -> Any:
    import lance

    path = Path(input_path)
    if path.suffix.lower() == ".lance":
        return lance.dataset(str(path))
    if not path.is_dir():
        raise ValueError(f"input must be an image directory or Lance dataset: {path}")

    lance_paths = sorted(
        path.glob("*.lance"),
        key=lambda candidate: candidate.name.casefold(),
    )
    if lance_paths:
        preferred = path / "dataset.lance"
        selected = preferred if preferred in lance_paths else lance_paths[0]
        return lance.dataset(str(selected))

    from module.lanceImport import transform2lance

    dataset = transform2lance(
        str(path),
        output_name="dataset",
        save_binary=False,
        not_save_disk=False,
        tag="RewardScoring",
    )
    if dataset is None:
        raise RuntimeError(f"failed to create Lance dataset from {path}")
    return dataset


def _select_checkpoint_row(checkpoints: Sequence[Any], requested: str | None) -> Any:
    if requested is None:
        defaults = [row for row in checkpoints if row.is_default]
        if len(defaults) != 1:
            raise RuntimeError("scorer must expose exactly one default checkpoint")
        return defaults[0]
    if not requested.strip():
        raise ValueError("checkpoint identifier must be nonempty")
    for row in checkpoints:
        if row.identifier == requested:
            return row
    raise ValueError(f"unregistered checkpoint: {requested!r}")


def _source_root_for_run(
    input_path: str | Path,
    source_paths: Sequence[str],
) -> Path:
    path = Path(input_path)
    if path.suffix.lower() != ".lance":
        return path.resolve()
    if not source_paths:
        return path.resolve().parent
    parents = [str(Path(source).resolve().parent) for source in source_paths]
    try:
        return Path(os.path.commonpath(parents))
    except ValueError as error:
        raise ValueError(
            "direct Lance input contains source images on unrelated filesystem roots"
        ) from error


def run(args: argparse.Namespace) -> int:
    console = resolve_rich_console()
    policy = load_reward_policy(CONFIG_DIR)
    scorer_name = args.scorer or policy.default_scorer
    thresholds = policy.thresholds_for(scorer_name)

    import qinglong_score

    spec = qinglong_score.get_scorer_spec(scorer_name)
    checkpoints = qinglong_score.list_checkpoints(scorer_name)
    selected_checkpoint = _select_checkpoint_row(checkpoints, args.checkpoint)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    console.print(
        f"[cyan]Loading scorer:[/cyan] {escape(scorer_name)} "
        f"([white]{escape(selected_checkpoint.identifier)}[/white])"
    )
    with suppress_library_progress_bars(), hf_download_reporting(console):
        scorer = qinglong_score.load_scorer(
            name=scorer_name,
            checkpoint=args.checkpoint,
            device=str(device),
            dtype=dtype,
            attention_backend="auto",
        )
    console.print(
        f"[green]Scorer ready:[/green] {escape(scorer_name)} "
        f"([white]{escape(selected_checkpoint.identifier)}[/white])"
    )
    compute_dtype = dtype or spec.default_compute_dtype
    _log_run_configuration(
        console=console,
        qinglong_score_version=qinglong_score.__version__,
        scorer_name=scorer_name,
        selected_checkpoint=selected_checkpoint,
        scorer=scorer,
        compute_dtype=compute_dtype,
        thresholds=thresholds,
    )

    dataset = _resolve_dataset(args.train_data_dir)
    schema_names = set(dataset.schema.names)
    missing = {"uris", "mime"} - schema_names
    if missing:
        raise ValueError(
            "Lance image dataset is missing columns: " + ", ".join(sorted(missing))
        )
    columns = ["uris", "mime"]
    if "captions" in schema_names:
        columns.append("captions")
    scanner = dataset.scanner(
        columns=columns,
        filter="mime LIKE 'image/%'",
        scan_in_order=True,
        batch_size=args.batch_size,
        batch_readahead=16,
        fragment_readahead=4,
        io_buffer_size=32 * 1024 * 1024,
        late_materialization=True,
    )

    items: list[ScoredImage] = []
    errors: list[RunError] = []
    source_paths: list[str] = []
    for batch in scanner.to_batches():
        sources = source_images_from_batch(batch)
        source_paths.extend(source.path for source in sources)
        batch_items, batch_errors = score_source_batch(
            sources,
            scorer=scorer,
            requires_prompts=spec.requires_prompts,
            prompt_override=args.prompt,
        )
        items.extend(batch_items)
        errors.extend(batch_errors)
        _log_scored_items(batch_items, console=console)

    source_root = _source_root_for_run(args.train_data_dir, source_paths)
    tracking = bool(selected_checkpoint.tracks_updates)
    apply_thresholds(
        items,
        thresholds=thresholds,
        source_root=source_root,
        tracking_checkpoint=tracking,
        warn=lambda message: console.print(
            f"[yellow]{escape(str(message))}[/yellow]"
        ),
    )
    report = build_report(
        items=items,
        source_root=source_root,
    )
    result_path = result_path_for_input(args.train_data_dir)
    write_json_atomic(result_path, report)

    _log_run_errors(errors, console=console)
    _log_run_completion(items, errors, console=console)
    console.print(f"[green]Results saved to: {escape(str(result_path))}[/green]")
    return 0 if items else 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch_size must be positive")
    return parsed


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score image datasets")
    parser.add_argument("train_data_dir", help="Image directory or Lance dataset")
    parser.add_argument("--scorer", default=None, help="Registered scorer name")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Registered checkpoint identifier; omit for the scorer default",
    )
    parser.add_argument("--batch_size", type=_positive_int, default=1)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(setup_parser().parse_args(argv))


__all__ = [
    "PreparedImage",
    "PromptSource",
    "RunError",
    "ScoredImage",
    "SourceImage",
    "apply_thresholds",
    "build_report",
    "decode_image",
    "result_path_for_input",
    "resolve_device",
    "resolve_dtype",
    "run",
    "score_source_batch",
    "select_prompt",
    "source_images_from_batch",
    "setup_parser",
    "write_json_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
