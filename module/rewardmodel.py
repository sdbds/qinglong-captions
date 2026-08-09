# /// script
# dependencies = [
#   "setuptools",
#   "pillow>=11.3",
#   "pylance>=2.0.1",
#   "rich>=13.5.0",
#   "imageio>=2.31.1",
#   "imageio-ffmpeg>=0.4.8",
#   "toml",
#   "huggingface_hub[hf_xet]>=0.35.2",
#   "torch==2.11.0",
#   "transformers[serving]==4.57.6",
#   "torchvision",
#   "scipy",
#   "imscore",
# ]
# ///
"""Score image datasets through the public Qinglong Score contract."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from module.reward_policy import Threshold, assign_threshold, load_reward_policy

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


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    return value


def serialize_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not is_dataclass(checkpoint):
        raise TypeError("checkpoint identity must be a public dataclass")
    payload = _json_value(asdict(checkpoint))
    if "identifier" in payload and "artifacts" in payload:
        kind = "remote"
    elif "path" in payload:
        kind = "local"
    else:
        raise TypeError("unsupported checkpoint identity")
    return {"kind": kind, **payload}


def _relative_path(path: str | Path, source_root: str | Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(source_root).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"source path is outside the report root: {resolved_path}"
        ) from error


def _serialize_error(error: RunError, source_root: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": error.scope,
        "stage": error.stage,
        "error_type": error.error_type,
        "message": error.message,
    }
    if error.scope == "item":
        if error.path is None:
            raise ValueError("item error must contain one path")
        payload["path"] = _relative_path(error.path, source_root)
    else:
        if not error.paths:
            raise ValueError("batch error must contain at least one path")
        payload["paths"] = [
            _relative_path(path, source_root) for path in error.paths
        ]
    return payload


def build_report(
    *,
    qinglong_score_version: str,
    scorer: str,
    requested_checkpoint: str | None,
    tracking_source: Any | None,
    checkpoint: Any,
    device: str | torch.device,
    compute_dtype: torch.dtype,
    input_dtype: torch.dtype,
    attention_backend: str | None,
    thresholds: Sequence[Threshold],
    items: Sequence[ScoredImage],
    errors: Sequence[RunError],
    source_root: str | Path,
    buckets: Mapping[str, str | None],
) -> dict[str, Any]:
    report_items: list[dict[str, Any]] = []
    for item in items:
        if not math.isfinite(item.score):
            raise ValueError(f"score must be finite for {item.path}")
        report_items.append(
            {
                "path": _relative_path(item.path, source_root),
                "score": item.score,
                "prompt": item.prompt,
                "prompt_source": item.prompt_source,
                "bucket": buckets.get(item.path),
            }
        )
    report_items.sort(key=lambda row: (-row["score"], row["path"]))
    for rank, item in enumerate(report_items, start=1):
        item["rank"] = rank
    report_items = [
        {
            "rank": item["rank"],
            "path": item["path"],
            "score": item["score"],
            "prompt": item["prompt"],
            "prompt_source": item["prompt_source"],
            "bucket": item["bucket"],
        }
        for item in report_items
    ]

    report_errors = [_serialize_error(error, source_root) for error in errors]
    failed = sum(
        1 if error.scope == "item" else len(error.paths) for error in errors
    )
    threshold_rows = [row.as_dict() for row in thresholds]
    return {
        "run": {
            "qinglong_score_version": str(qinglong_score_version),
            "scorer": scorer,
            "requested_checkpoint": requested_checkpoint,
            "tracking_source": (
                serialize_checkpoint(tracking_source)
                if tracking_source is not None
                else None
            ),
            "checkpoint": serialize_checkpoint(checkpoint),
            "device": str(device),
            "compute_dtype": str(compute_dtype),
            "input_dtype": str(input_dtype),
            "attention_backend": attention_backend,
            "thresholds_enabled": bool(threshold_rows),
            "thresholds": threshold_rows,
        },
        "summary": {
            "scored": len(report_items),
            "failed": failed,
            "empty_prompt_count": sum(
                item.prompt_source == "empty" for item in items
            ),
        },
        "items": report_items,
        "errors": report_errors,
    }


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

    quality_roots = {
        threshold.name: root / threshold.folder_name for threshold in thresholds
    }
    for quality_root in quality_roots.values():
        quality_root.mkdir(parents=True, exist_ok=True)
        for existing in quality_root.rglob("*"):
            if existing.is_symlink():
                existing.unlink()

    buckets: dict[str, str | None] = {}
    for item in items:
        threshold = assign_threshold(item.score, thresholds)
        if threshold is None:
            buckets[item.path] = None
            continue
        relative_path = Path(_relative_path(item.path, root))
        source_path = Path(item.path).resolve()
        target_path = quality_roots[threshold.name] / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)
            warn(f"Symlink unavailable; copied {source_path} to {target_path}.")
        buckets[item.path] = threshold.name
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
    policy = load_reward_policy(CONFIG_DIR)
    scorer_name = args.scorer or policy.default_scorer
    thresholds = policy.thresholds_for(scorer_name)

    import qinglong_score

    spec = qinglong_score.get_scorer_spec(scorer_name)
    checkpoints = qinglong_score.list_checkpoints(scorer_name)
    selected_checkpoint = _select_checkpoint_row(checkpoints, args.checkpoint)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    scorer = qinglong_score.load_scorer(
        name=scorer_name,
        checkpoint=args.checkpoint,
        device=str(device),
        dtype=dtype,
        attention_backend="auto",
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

    source_root = _source_root_for_run(args.train_data_dir, source_paths)
    tracking = bool(selected_checkpoint.tracks_updates)
    buckets = apply_thresholds(
        items,
        thresholds=thresholds,
        source_root=source_root,
        tracking_checkpoint=tracking,
    )
    report = build_report(
        qinglong_score_version=qinglong_score.__version__,
        scorer=scorer_name,
        requested_checkpoint=args.checkpoint,
        tracking_source=selected_checkpoint if tracking else None,
        checkpoint=scorer.checkpoint_identity,
        device=scorer.device,
        compute_dtype=dtype or spec.default_compute_dtype,
        input_dtype=scorer.input_dtype,
        attention_backend=scorer.attention_backend,
        thresholds=thresholds,
        items=items,
        errors=errors,
        source_root=source_root,
        buckets=buckets,
    )
    result_path = result_path_for_input(args.train_data_dir)
    write_json_atomic(result_path, report)

    if items and errors:
        print(
            f"Scored {len(items)} image(s); "
            f"{sum(len(error.paths) or 1 for error in errors)} failed."
        )
    print(f"Results saved to: {result_path}")
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
    "serialize_checkpoint",
    "source_images_from_batch",
    "setup_parser",
    "write_json_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
