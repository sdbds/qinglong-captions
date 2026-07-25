"""Pinned MuSViT OMR bundle loading and full-prefix ONNX decoding."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from config.loader import load_config
from module.onnx_runtime import (
    OnnxMultiModelSpec,
    OnnxRuntimeConfig,
    build_local_model_dir,
    load_multi_model_bundle,
    resolve_tool_runtime_config,
)

from .constraints import KernConstraintError, KernGreedyConstraint
from .preprocess import (
    MuSViTPreprocessorConfig,
    load_preprocessor_config,
    preprocess_image,
    preprocess_pil_image,
)

DEFAULT_MODEL_REPO_ID = "bdsqlsz/qinglong-musvit-1.0"
DEFAULT_MODEL_REVISION = "6f47cefe0e736fbdd0eab9e8bc8d4602e81939fe"
DEFAULT_MODEL_DIR = "huggingface"
DEFAULT_MODEL_MAX_LENGTH = 7512
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

_ARTIFACT_FILES = {
    "encoder": "encoder.onnx",
    "decoder": "decoder.onnx",
}
_SUPPORT_FILES = {
    "config": "config.json",
    "preprocessor_config": "preprocessor_config.json",
    "validation": "val_evaluation.json",
}
_PINNED_ARTIFACTS = {
    "encoder": {
        "bytes": 356_197_419,
        "sha256": "1d49c2c15cb91ce6d69f3b4591d0b0fb5e63ce81dcd6089da6d1fd0e53d530f0",
    },
    "decoder": {
        "bytes": 29_469_910,
        "sha256": "748f844cdd8ad2298cbc72579be751e8edca89df4f6fabfd7b0fa88032f37423",
    },
}

_ENCODER_INPUT_CONTRACT = [
    ("pixel_values", "tensor(float)", [1, 3, 1024, 1024]),
]
_ENCODER_OUTPUT_CONTRACT = [
    ("raw_features", "tensor(float)", [1, 4096, 256]),
    ("enhanced_features", "tensor(float)", [1, 4096, 256]),
]
_DECODER_INPUT_CONTRACT = [
    ("raw_features", "tensor(float)", [1, 4096, 256]),
    ("enhanced_features", "tensor(float)", [1, 4096, 256]),
    ("token_ids", "tensor(int64)", [1, "sequence_length"]),
]
_DECODER_OUTPUT_CONTRACT = [
    ("next_token_logits", "tensor(float)", [1, 215]),
]
_SPECIAL_TOKEN_IDS = {
    "<pad>": 0,
    "<bos>": 100,
    "<eos>": 183,
    "<s>": 44,
    "<t>": 29,
    "<b>": 132,
}


@dataclass(frozen=True)
class MuSViTModelConfig:
    i2w: dict[int, str]
    w2i: dict[str, int]
    output_categories: int
    max_length: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int


@dataclass(frozen=True)
class OnnxGenerationResult:
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    terminated_by_eos: bool
    truncated: bool
    elapsed_seconds: float


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {config_path}")
    return payload


def parse_model_config(payload: Mapping[str, Any]) -> MuSViTModelConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("model config must be a JSON object")

    raw_i2w = payload.get("i2w")
    raw_w2i = payload.get("w2i")
    if not isinstance(raw_i2w, Mapping):
        raise ValueError("model config i2w must be an object")
    if not isinstance(raw_w2i, Mapping):
        raise ValueError("model config w2i must be an object")
    for token, expected_id in _SPECIAL_TOKEN_IDS.items():
        if token not in raw_w2i:
            raise KeyError(f"model config w2i is missing {token}")
        if raw_w2i[token] != expected_id:
            raise ValueError(
                f"model special token {token} must have id {expected_id}, "
                f"got {raw_w2i[token]!r}"
            )
    if len(raw_i2w) != 215 or len(raw_w2i) != 215:
        raise ValueError("model vocabulary must contain exactly 215 entries")

    i2w: dict[int, str] = {}
    for raw_token_id, token in raw_i2w.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"i2w token id is not an integer: {raw_token_id!r}") from exc
        if isinstance(token, str):
            i2w[token_id] = token
        else:
            raise ValueError(f"i2w token {token_id} must be a string")
    if set(i2w) != set(range(215)):
        raise ValueError("model i2w ids must be exactly 0..214 (215 entries)")

    w2i: dict[str, int] = {}
    for token, raw_token_id in raw_w2i.items():
        if not isinstance(token, str):
            raise ValueError("w2i token keys must be strings")
        if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
            raise ValueError(f"w2i token id for {token!r} must be an integer")
        w2i[token] = raw_token_id
    expected_inverse = {token: token_id for token_id, token in i2w.items()}
    if w2i != expected_inverse:
        raise ValueError("model w2i must be the exact inverse of i2w")

    output_categories = payload.get("out_categories")
    if type(output_categories) is not int or output_categories != 215:
        raise ValueError(
            f"model out_categories must be 215, got {output_categories!r}"
        )
    max_length = payload.get("maxlen")
    if (
        type(max_length) is not int
        or max_length != DEFAULT_MODEL_MAX_LENGTH
    ):
        raise ValueError(
            f"model maxlen must be {DEFAULT_MODEL_MAX_LENGTH}, got {max_length!r}"
        )

    return MuSViTModelConfig(
        i2w=i2w,
        w2i=w2i,
        output_categories=output_categories,
        max_length=max_length,
        pad_token_id=w2i["<pad>"],
        bos_token_id=w2i["<bos>"],
        eos_token_id=w2i["<eos>"],
    )


def load_model_config(path: str | Path) -> MuSViTModelConfig:
    return parse_model_config(_read_json_object(path, label="model config"))


def _session_contract(session: Any, method_name: str) -> list[tuple[str, str, list[Any]]]:
    return [
        (value.name, value.type, list(value.shape))
        for value in getattr(session, method_name)()
    ]


def _require_session_contract(
    session: Any,
    *,
    label: str,
    input_contract: list[tuple[str, str, list[Any]]],
    output_contract: list[tuple[str, str, list[Any]]],
) -> None:
    actual_inputs = _session_contract(session, "get_inputs")
    actual_outputs = _session_contract(session, "get_outputs")
    if actual_inputs != input_contract:
        raise ValueError(
            f"{label} input contract mismatch: {actual_inputs!r} != {input_contract!r}"
        )
    if actual_outputs != output_contract:
        raise ValueError(
            f"{label} output contract mismatch: {actual_outputs!r} != {output_contract!r}"
        )


def _provider_name(provider: Any) -> str:
    if isinstance(provider, (tuple, list)) and provider:
        return str(provider[0])
    return str(provider)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pinned_artifacts(
    artifact_paths: Mapping[str, Path],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for label, expected in _PINNED_ARTIFACTS.items():
        path = Path(artifact_paths[label])
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"pinned MuSViT {label} artifact size mismatch: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(f"pinned MuSViT {label} artifact SHA-256 mismatch: {path}")
        hashes[label] = actual_hash
    return hashes


def resolve_musvit_model_dir(
    model_dir: str | Path,
    repo_id: str,
    revision: str | None,
) -> Path:
    candidate = Path(model_dir).expanduser()
    required_files = (*_ARTIFACT_FILES.values(), *_SUPPORT_FILES.values())
    if all((candidate / filename).is_file() for filename in required_files):
        return candidate
    return build_local_model_dir(candidate, repo_id, revision=revision)


def resolve_musvit_runtime_config(
    *,
    force_download: bool = False,
    config_dir: str | Path = CONFIG_DIR,
) -> OnnxRuntimeConfig:
    config = load_config(str(config_dir))
    return resolve_tool_runtime_config(
        config,
        tool_name="musvit",
        cli_override={"force_download": force_download},
    )


class MuSViTOnnxRuntime:
    def __init__(
        self,
        *,
        encoder_session: Any,
        decoder_session: Any,
        model_config: MuSViTModelConfig,
        preprocessor_config: MuSViTPreprocessorConfig,
        providers: Sequence[Any] | None = None,
    ) -> None:
        _require_session_contract(
            encoder_session,
            label="encoder",
            input_contract=_ENCODER_INPUT_CONTRACT,
            output_contract=_ENCODER_OUTPUT_CONTRACT,
        )
        _require_session_contract(
            decoder_session,
            label="decoder",
            input_contract=_DECODER_INPUT_CONTRACT,
            output_contract=_DECODER_OUTPUT_CONTRACT,
        )
        self.encoder_session = encoder_session
        self.decoder_session = decoder_session
        self.model_config = model_config
        self.preprocessor_config = preprocessor_config
        raw_providers = providers
        if raw_providers is None and hasattr(encoder_session, "get_providers"):
            raw_providers = encoder_session.get_providers()
        self.providers = tuple(_provider_name(provider) for provider in (raw_providers or ()))

    def encode_pixel_values(self, pixel_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if pixel_values.shape != (1, 3, 1024, 1024):
            raise ValueError(
                "pixel_values shape must be (1, 3, 1024, 1024), "
                f"got {pixel_values.shape!r}"
            )
        if pixel_values.dtype != np.float32:
            raise TypeError("pixel_values must use float32")
        raw, enhanced = self.encoder_session.run(
            ["raw_features", "enhanced_features"],
            {"pixel_values": np.ascontiguousarray(pixel_values)},
        )
        raw = np.asarray(raw)
        enhanced = np.asarray(enhanced)
        expected_shape = (1, 4096, 256)
        if raw.shape != expected_shape or enhanced.shape != expected_shape:
            raise ValueError("encoder output tensors do not match [1, 4096, 256]")
        if raw.dtype != np.float32 or enhanced.dtype != np.float32:
            raise TypeError("encoder output tensors must use float32")
        return np.ascontiguousarray(raw), np.ascontiguousarray(enhanced)

    def next_token_logits(
        self,
        raw_features: np.ndarray,
        enhanced_features: np.ndarray,
        token_ids: np.ndarray,
    ) -> np.ndarray:
        if token_ids.dtype != np.int64 or token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("token_ids must have shape [1, T] and dtype int64")
        (logits,) = self.decoder_session.run(
            ["next_token_logits"],
            {
                "raw_features": np.ascontiguousarray(raw_features, dtype=np.float32),
                "enhanced_features": np.ascontiguousarray(
                    enhanced_features,
                    dtype=np.float32,
                ),
                "token_ids": np.ascontiguousarray(token_ids),
            },
        )
        logits = np.asarray(logits)
        if logits.shape != (1, self.model_config.output_categories):
            raise ValueError(
                "decoder logits shape must be "
                f"(1, {self.model_config.output_categories}), got {logits.shape!r}"
            )
        if logits.dtype != np.float32:
            raise TypeError("decoder logits must use float32")
        return np.ascontiguousarray(logits)

    def generate_pixel_values(
        self,
        pixel_values: np.ndarray,
        *,
        max_tokens: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> OnnxGenerationResult:
        effective_limit = self.model_config.max_length if max_tokens is None else int(max_tokens)
        if effective_limit < 2 or effective_limit > self.model_config.max_length:
            raise ValueError(
                f"max_tokens must be in 2..{self.model_config.max_length}, "
                f"got {effective_limit}"
            )

        started = time.perf_counter()
        raw_features, enhanced_features = self.encode_pixel_values(pixel_values)
        token_ids = [self.model_config.bos_token_id]
        constraint = KernGreedyConstraint(
            self.model_config.i2w,
            eos_token_id=self.model_config.eos_token_id,
        )
        terminated_by_eos = False
        while len(token_ids) < effective_limit:
            prefix = np.asarray([token_ids], dtype=np.int64)
            logits = self.next_token_logits(raw_features, enhanced_features, prefix)
            raw_argmax = int(np.argmax(logits[0]))
            if raw_argmax not in self.model_config.i2w:
                raise KeyError(f"Unknown predicted token id {raw_argmax}")
            allowed_ids = np.asarray(
                constraint.allowed_token_ids(),
                dtype=np.int64,
            )
            allowed_logits = logits[0, allowed_ids]
            finite = np.isfinite(allowed_logits)
            if not np.any(finite):
                raise KernConstraintError(
                    "decoder produced no finite legal token: "
                    f"{constraint.describe()}"
                )
            legal_ids = allowed_ids[finite]
            legal_logits = allowed_logits[finite]
            next_token = int(legal_ids[int(np.argmax(legal_logits))])
            constraint.accept(next_token)
            token_ids.append(next_token)
            if progress_callback is not None:
                progress_callback(len(token_ids))
            if next_token == self.model_config.eos_token_id:
                terminated_by_eos = True
                break

        decoded: list[str] = []
        for token_id in token_ids[1:]:
            if token_id == self.model_config.eos_token_id:
                break
            try:
                decoded.append(self.model_config.i2w[token_id])
            except KeyError:
                raise KeyError(f"Unknown predicted token id {token_id}") from None
        return OnnxGenerationResult(
            token_ids=tuple(token_ids),
            tokens=tuple(decoded),
            terminated_by_eos=terminated_by_eos,
            truncated=not terminated_by_eos,
            elapsed_seconds=time.perf_counter() - started,
        )

    def generate(
        self,
        image: str | Path | Image.Image,
        *,
        max_tokens: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> OnnxGenerationResult:
        if isinstance(image, Image.Image):
            pixels = preprocess_pil_image(image, self.preprocessor_config)
        else:
            pixels = preprocess_image(image, self.preprocessor_config)
        return self.generate_pixel_values(
            pixels,
            max_tokens=max_tokens,
            progress_callback=progress_callback,
        )


class MuSViTOnnxRecognizer:
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_MODEL_REPO_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        force_download: bool = False,
        runtime_config: OnnxRuntimeConfig | None = None,
        config_dir: str | Path = CONFIG_DIR,
        bundle_loader: Callable[..., Any] = load_multi_model_bundle,
        artifact_loader: Callable[..., dict[str, Path]] | None = None,
        support_file_loader: Callable[..., dict[str, Path]] | None = None,
        session_bundle_loader: Callable[..., Any] | None = None,
        logger: Callable[..., Any] | None = None,
        verify_pinned_artifacts: bool = True,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.model_dir = resolve_musvit_model_dir(model_dir, repo_id, revision)
        self.runtime_config = runtime_config or resolve_musvit_runtime_config(
            force_download=force_download,
            config_dir=config_dir,
        )
        self.spec = OnnxMultiModelSpec(
            repo_id=repo_id,
            revision=revision,
            artifacts=_ARTIFACT_FILES,
            support_files=_SUPPORT_FILES,
            local_dir=self.model_dir,
            bundle_key=f"musvit-omr:{repo_id}@{revision}",
        )
        loader_kwargs: dict[str, Any] = {
            "spec": self.spec,
            "runtime_config": self.runtime_config,
            "logger": logger,
        }
        if artifact_loader is not None:
            loader_kwargs["artifact_loader"] = artifact_loader
        if support_file_loader is not None:
            loader_kwargs["support_file_loader"] = support_file_loader
        if session_bundle_loader is not None:
            loader_kwargs["session_bundle_loader"] = session_bundle_loader
        self.bundle = bundle_loader(**loader_kwargs)

        artifact_paths = {
            label: Path(path) for label, path in self.bundle.artifact_paths.items()
        }
        support_paths = {
            label: Path(path) for label, path in self.bundle.support_paths.items()
        }
        missing_artifacts = sorted(set(_ARTIFACT_FILES) - set(artifact_paths))
        missing_support = sorted(set(_SUPPORT_FILES) - set(support_paths))
        if missing_artifacts or missing_support:
            raise RuntimeError(
                "MuSViT bundle is incomplete: "
                f"missing artifacts={missing_artifacts}, support={missing_support}"
            )
        if (
            verify_pinned_artifacts
            and repo_id == DEFAULT_MODEL_REPO_ID
            and revision == DEFAULT_MODEL_REVISION
        ):
            artifact_hashes = _verify_pinned_artifacts(artifact_paths)
        else:
            artifact_hashes = {
                label: _sha256_file(path)
                for label, path in artifact_paths.items()
            }

        model_config = load_model_config(support_paths["config"])
        preprocessor_config = load_preprocessor_config(
            support_paths["preprocessor_config"]
        )
        _read_json_object(support_paths["validation"], label="validation evidence")
        self.runtime = MuSViTOnnxRuntime(
            encoder_session=self.bundle.sessions["encoder"],
            decoder_session=self.bundle.sessions["decoder"],
            model_config=model_config,
            preprocessor_config=preprocessor_config,
            providers=self.bundle.providers,
        )
        self.artifact_paths = artifact_paths
        self.support_paths = support_paths
        self.bundle_hashes = {
            **artifact_hashes,
            "config": _sha256_file(support_paths["config"]),
            "preprocessor_config": _sha256_file(
                support_paths["preprocessor_config"]
            ),
        }
        self.model_config = model_config
        self.preprocessor_config = preprocessor_config
        self.providers = self.runtime.providers

    def generate(
        self,
        image: str | Path | Image.Image,
        *,
        max_tokens: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> OnnxGenerationResult:
        return self.runtime.generate(
            image,
            max_tokens=max_tokens,
            progress_callback=progress_callback,
        )
