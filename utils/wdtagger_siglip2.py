from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from module.onnx_runtime import load_session_bundle

CL_TAGGER_V2_OPTION = "cella110n/cl_tagger_v2"
CL_TAGGER_V2_BACKEND_REPO = "celstk/cl-SigLIP2-lora-onnx"
CL_TAGGER_V2_PROCESSOR_REPO = "google/siglip2-so400m-patch16-naflex"
CL_TAGGER_V2_VERSIONS = ("v1_00", "v1_01", "v1_02", "v1_03", "v1_04", "v1_05")
CL_TAGGER_V2_DEFAULT_VERSION = "v1_05"
CL_TAGGER_V2_FALLBACK_THRESHOLD = 0.5
CL_TAGGER_V2_THRESHOLD_OVERRIDES = {
    "v1_00": 0.6,
    "v1_01": 0.6,
    "v1_02": 0.9,
}
CL_TAGGER_V2_DEFAULT_MAX_NUM_PATCHES = 256
CL_TAGGER_V2_OUTPUT_NAME = "logits"
_KNOWN_CATEGORY_KEYS = (
    "rating",
    "general",
    "character",
    "copyright",
    "artist",
    "meta",
    "quality",
    "model",
)


@dataclass(frozen=True)
class Siglip2Vocabulary:
    names: list[Optional[str]]
    category_indices: dict[str, np.ndarray]
    tag_index_to_category: dict[int, str]


@dataclass(frozen=True)
class Siglip2InferenceContext:
    processor: Any
    max_num_patches: int = CL_TAGGER_V2_DEFAULT_MAX_NUM_PATCHES
    output_name: str = CL_TAGGER_V2_OUTPUT_NAME
    is_naflex: bool = True


@dataclass(frozen=True)
class Siglip2OnnxBundle:
    session: Any
    providers: tuple[Any, ...]
    inference_context: Siglip2InferenceContext
    vocabulary: Siglip2Vocabulary
    model_path: Path
    vocab_path: Path
    metadata_path: Path | None
    processor_repo: str
    resolved_repo_id: str
    version: str
    cache_dir: Path


def _emit_log(logger: Callable[..., Any] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def is_cl_tagger_v2_repo(repo_id: str) -> bool:
    normalized = str(repo_id or "").strip()
    return normalized in {CL_TAGGER_V2_OPTION, CL_TAGGER_V2_BACKEND_REPO}


def resolve_cl_tagger_v2_backend_repo(repo_id: str) -> str:
    normalized = str(repo_id or "").strip()
    if normalized == CL_TAGGER_V2_OPTION:
        return CL_TAGGER_V2_BACKEND_REPO
    return normalized


def resolve_cl_tagger_v2_cache_dir(model_dir: str | Path, repo_id: str) -> Path:
    return Path(model_dir) / str(repo_id).replace("/", "_")


def normalize_cl_tagger_v2_version(version: str | None) -> str:
    value = str(version or CL_TAGGER_V2_DEFAULT_VERSION).strip()
    if not value:
        return CL_TAGGER_V2_DEFAULT_VERSION

    normalized = value.lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    normalized = normalized.replace("_", ".")
    parts = normalized.split(".")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f"v{int(parts[0])}_{int(parts[1]):02d}"
    return value


def default_cl_tagger_v2_threshold(version: str | None = None) -> float:
    normalized = normalize_cl_tagger_v2_version(version)
    return CL_TAGGER_V2_THRESHOLD_OVERRIDES.get(normalized, CL_TAGGER_V2_FALLBACK_THRESHOLD)


def _vocab_get(vocab: dict[str, Any], key: str) -> dict[str, Any]:
    if key in vocab and isinstance(vocab[key], dict):
        return vocab[key]

    suffix = f"/{key}"
    for vocab_key, value in vocab.items():
        if isinstance(vocab_key, str) and vocab_key.endswith(suffix) and isinstance(value, dict):
            return value
    return {}


def load_cl_tagger_v2_vocabulary(vocab_path: str | Path) -> Siglip2Vocabulary:
    with Path(vocab_path).open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    raw_idx_to_tag = _vocab_get(vocab, "idx_to_tag")
    if not raw_idx_to_tag:
        raise ValueError(f"'idx_to_tag' not found in {vocab_path}")

    idx_to_tag = {int(key): value for key, value in raw_idx_to_tag.items()}
    tag_to_category = {str(tag): str(category).strip().lower() for tag, category in _vocab_get(vocab, "tag_to_category").items()}

    max_idx = max(idx_to_tag) if idx_to_tag else -1
    names: list[Optional[str]] = [None] * (max_idx + 1)
    category_buckets: dict[str, list[int]] = {key: [] for key in _KNOWN_CATEGORY_KEYS}
    tag_index_to_category: dict[int, str] = {}

    for idx, tag in idx_to_tag.items():
        names[idx] = tag
        category_name = tag_to_category.get(str(tag), "")
        if not category_name:
            continue
        category_buckets.setdefault(category_name, []).append(idx)
        tag_index_to_category[idx] = category_name

    category_indices = {
        category: np.array(indices, dtype=np.int64)
        for category, indices in category_buckets.items()
    }
    return Siglip2Vocabulary(
        names=names,
        category_indices=category_indices,
        tag_index_to_category=tag_index_to_category,
    )


def load_cl_tagger_v2_metadata(meta_path: str | Path | None) -> tuple[str, bool]:
    processor_repo = CL_TAGGER_V2_PROCESSOR_REPO
    is_naflex = True
    if not meta_path:
        return processor_repo, is_naflex

    path = Path(meta_path)
    if not path.is_file():
        return processor_repo, is_naflex

    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    processor_repo = str(metadata.get("vision_encoder_repo", processor_repo) or processor_repo)
    is_naflex = bool(metadata.get("is_naflex", True))
    return processor_repo, is_naflex


def _load_siglip2_processor(
    *,
    processor_repo: str,
    processor_loader: Any = None,
    token: str | None = None,
) -> Any:
    if processor_loader is None:
        from transformers import AutoProcessor

        processor_loader = AutoProcessor

    def _from_pretrained(*, local_files_only: bool) -> Any:
        kwargs = {"local_files_only": local_files_only}
        if token:
            kwargs["token"] = token
        try:
            return processor_loader.from_pretrained(processor_repo, **kwargs)
        except TypeError:
            kwargs.pop("token", None)
            return processor_loader.from_pretrained(processor_repo, **kwargs)

    try:
        return _from_pretrained(local_files_only=True)
    except Exception:
        from utils.transformer_loader import hf_download_reporting

        with hf_download_reporting():
            return _from_pretrained(local_files_only=False)


def _default_snapshot_download(**kwargs: Any) -> str:
    from utils.transformer_loader import snapshot_download_with_reporting

    repo_id = kwargs.pop("repo_id")
    return snapshot_download_with_reporting(repo_id, **kwargs)


def download_cl_tagger_v2_artifacts(
    *,
    repo_id: str,
    model_dir: str | Path,
    version: str = CL_TAGGER_V2_DEFAULT_VERSION,
    force_download: bool = False,
    logger: Callable[..., Any] | None = None,
    snapshot_downloader: Callable[..., str] | None = None,
) -> tuple[str, Path, Path, Path, Path | None]:
    resolved_repo_id = resolve_cl_tagger_v2_backend_repo(repo_id)
    version = normalize_cl_tagger_v2_version(version)
    local_cache_dir = resolve_cl_tagger_v2_cache_dir(model_dir, repo_id)
    snapshot_downloader = snapshot_downloader or _default_snapshot_download
    token = str(os.environ.get("HF_TOKEN", "")).strip() or None

    try:
        _emit_log(logger, f"[cyan]Downloading cl_tagger v2 snapshot[/cyan] {resolved_repo_id}:{version}")
        snapshot_path = Path(
            snapshot_downloader(
                repo_id=resolved_repo_id,
                allow_patterns=[f"{version}/*"],
                local_dir=str(local_cache_dir),
                force_download=force_download,
                token=token,
            )
        )
    except Exception as exc:
        hint = ""
        message = str(exc).lower()
        if any(marker in message for marker in ("401", "403", "forbidden", "gated", "access")):
            hint = " Set HF_TOKEN after accepting the backend model access terms."
        raise RuntimeError(
            f"Failed to download cl_tagger v2 artifacts from {resolved_repo_id}.{hint}"
        ) from exc

    version_dir_candidates = (
        local_cache_dir / version,
        snapshot_path / version,
    )
    version_dir = next((candidate for candidate in version_dir_candidates if candidate.exists()), version_dir_candidates[0])
    onnx_files = sorted(version_dir.glob("*.onnx"))
    vocab_files = sorted(version_dir.glob("*vocabulary.json"))
    metadata_files = sorted(version_dir.glob("*_metadata.json"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in {version_dir}")
    if not vocab_files:
        raise FileNotFoundError(f"No *vocabulary.json file found in {version_dir}")

    model_path = onnx_files[0]
    vocab_path = vocab_files[0]
    metadata_path = metadata_files[0] if metadata_files else None
    _emit_log(logger, f"[green]Resolved cl_tagger v2 ONNX[/green] {model_path}")
    _emit_log(logger, f"[green]Resolved cl_tagger v2 vocabulary[/green] {vocab_path}")
    if metadata_path is not None:
        _emit_log(logger, f"[green]Resolved cl_tagger v2 metadata[/green] {metadata_path}")
    return resolved_repo_id, local_cache_dir, model_path, vocab_path, metadata_path


def load_cl_tagger_v2_bundle(
    *,
    repo_id: str,
    model_dir: str | Path,
    runtime_config: Any,
    version: str = CL_TAGGER_V2_DEFAULT_VERSION,
    force_download: bool = False,
    logger: Callable[..., Any] | None = None,
    snapshot_downloader: Callable[..., str] | None = None,
    processor_loader: Any = None,
    session_bundle_loader: Callable[..., Any] | None = None,
) -> Siglip2OnnxBundle:
    resolved_repo_id, cache_dir, model_path, vocab_path, metadata_path = download_cl_tagger_v2_artifacts(
        repo_id=repo_id,
        model_dir=model_dir,
        version=version,
        force_download=force_download,
        logger=logger,
        snapshot_downloader=snapshot_downloader,
    )
    version = normalize_cl_tagger_v2_version(version)
    processor_repo, is_naflex = load_cl_tagger_v2_metadata(metadata_path)
    token = str(os.environ.get("HF_TOKEN", "")).strip() or None
    processor = _load_siglip2_processor(
        processor_repo=processor_repo,
        processor_loader=processor_loader,
        token=token,
    )
    vocabulary = load_cl_tagger_v2_vocabulary(vocab_path)
    session_bundle_loader = session_bundle_loader or load_session_bundle
    session_bundle = session_bundle_loader(
        bundle_key=f"wdtagger:{repo_id}",
        session_paths={"model": model_path},
        runtime_config=runtime_config,
    )
    return Siglip2OnnxBundle(
        session=session_bundle.sessions["model"],
        providers=tuple(session_bundle.providers),
        inference_context=Siglip2InferenceContext(processor=processor, is_naflex=is_naflex),
        vocabulary=vocabulary,
        model_path=model_path,
        vocab_path=vocab_path,
        metadata_path=metadata_path,
        processor_repo=processor_repo,
        resolved_repo_id=resolved_repo_id,
        version=version,
        cache_dir=cache_dir,
    )


def process_siglip2_batch(images: list[Any], session: Any, context: Siglip2InferenceContext) -> np.ndarray:
    if not images:
        return np.empty((0, 0), dtype=np.float64)

    if context.is_naflex:
        inputs = context.processor(
            images=images,
            return_tensors="pt",
            max_num_patches=int(context.max_num_patches),
        )
        pixel_values = np.asarray(inputs["pixel_values"].float().numpy(), dtype=np.float32)
        pixel_attention_mask = np.asarray(inputs["pixel_attention_mask"].float().numpy(), dtype=np.float32)
        spatial_shapes = np.asarray(inputs["spatial_shapes"].numpy(), dtype=np.int64)
        feeds = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        }
    else:
        inputs = context.processor(images=images, return_tensors="pt")
        pixel_values = np.asarray(inputs["pixel_values"].float().numpy(), dtype=np.float32)
        feeds = {"pixel_values": pixel_values}

    try:
        outputs = session.run([context.output_name], feeds)
    except Exception:
        outputs = session.run(None, feeds)

    logits = np.asarray(outputs[0], dtype=np.float64)
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
