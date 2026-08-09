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
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from module.reward_policy import load_reward_policy

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


def run(args: argparse.Namespace) -> int:
    policy = load_reward_policy(CONFIG_DIR)
    scorer_name = args.scorer or policy.default_scorer
    policy.thresholds_for(scorer_name)

    import qinglong_score

    spec = qinglong_score.get_scorer_spec(scorer_name)
    checkpoints = qinglong_score.list_checkpoints(scorer_name)
    _select_checkpoint_row(checkpoints, args.checkpoint)
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
    for batch in scanner.to_batches():
        batch_items, batch_errors = score_source_batch(
            source_images_from_batch(batch),
            scorer=scorer,
            requires_prompts=spec.requires_prompts,
            prompt_override=args.prompt,
        )
        items.extend(batch_items)
        errors.extend(batch_errors)

    if items and errors:
        print(
            f"Scored {len(items)} image(s); "
            f"{sum(len(error.paths) or 1 for error in errors)} failed."
        )
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
    "decode_image",
    "resolve_device",
    "resolve_dtype",
    "run",
    "score_source_batch",
    "select_prompt",
    "source_images_from_batch",
    "setup_parser",
]


if __name__ == "__main__":
    raise SystemExit(main())
