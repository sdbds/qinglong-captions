from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from einops import pack, rearrange, repeat, unpack

DEFAULT_MODEL_DIR = Path(r"D:\BS-ROFO-SW-Fixed")
DEFAULT_MUSVIT_REPO_ID = "PRAIG/musvit"
DEFAULT_MUSVIT_OUTPUT_PATH = Path("huggingface") / "PRAIG_musvit_ONNX" / "model.onnx"
TARGET_ROFORMER = "roformer"
TARGET_MUSVIT = "musvit"
MODEL_TYPE_BS_ROFORMER = "bs_roformer"
MODEL_TYPE_MEL_BAND_ROFORMER = "mel_band_roformer"

BS_ROFORMER_LATEST_CONSTRUCTOR_KEYS = {
    "dim",
    "depth",
    "stereo",
    "num_stems",
    "time_transformer_depth",
    "freq_transformer_depth",
    "freqs_per_bands",
    "dim_head",
    "heads",
    "attn_dropout",
    "ff_dropout",
    "flash_attn",
    "dim_freqs_in",
    "stft_n_fft",
    "stft_hop_length",
    "stft_win_length",
    "stft_normalized",
    "zero_dc",
    "mask_estimator_depth",
    "multi_stft_resolution_loss_weight",
    "multi_stft_resolutions_window_sizes",
    "multi_stft_hop_size",
    "multi_stft_normalized",
    "use_pope",
}

MEL_BAND_ROFORMER_LATEST_CONSTRUCTOR_KEYS = {
    "dim",
    "depth",
    "stereo",
    "num_stems",
    "time_transformer_depth",
    "freq_transformer_depth",
    "linear_transformer_depth",
    "num_bands",
    "dim_head",
    "heads",
    "attn_dropout",
    "ff_dropout",
    "flash_attn",
    "linear_flash_attn",
    "dim_freqs_in",
    "sample_rate",
    "stft_n_fft",
    "stft_hop_length",
    "stft_win_length",
    "stft_normalized",
    "zero_dc",
    "mask_estimator_depth",
    "multi_stft_resolution_loss_weight",
    "multi_stft_resolutions_window_sizes",
    "multi_stft_hop_size",
    "multi_stft_normalized",
    "match_input_audio_length",
    "add_value_residual",
    "num_residual_streams",
    "num_residual_fracs",
    "use_pope",
}

IGNORED_BS_ROFORMER_CONFIG_KEYS = {
    "linear_transformer_depth": 0,
    "skip_connection": False,
    "use_torch_checkpoint": False,
    "mlp_expansion_factor": 4,
}

IGNORED_MEL_BAND_ROFORMER_CONFIG_KEYS = {
    "mlp_expansion_factor": 4,
    "sage_attention": False,
    "skip_connection": False,
    "use_torch_checkpoint": False,
}

OLD_TO_LATEST_BRANCH_PATTERN = re.compile(r"^(layers\.\d+\.\d+\.layers\.\d+\.\d+)\.(.+)$")
OLD_TO_LATEST_MEL_BAND_LAYER_PATTERN = re.compile(r"^(layers\.\d+)\.(0|1)\.(.+)$")
LATEST_ONLY_ALLOWED_MISSING_SUFFIXES = (
    "branch.to_value_residual_mix.weight",
    "branch.to_value_residual_mix.bias",
)


class _TupleSafeLoader(yaml.SafeLoader):
    """YAML loader that accepts !!python/tuple without enabling arbitrary objects."""


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


_TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def build_roformer_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a RoFormer checkpoint to ONNX mask format for downstream source-separation inference.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else None,
        help="Directory containing the checkpoint (.ckpt/.pt) and config (.yaml). Defaults to D:\\BS-ROFO-SW-Fixed when it exists.",
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Explicit path to the BS-RoFormer YAML config.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Explicit path to the checkpoint (.ckpt/.pt).")
    parser.add_argument("--output-path", type=Path, default=None, help="Path to the output ONNX model.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help="Device used for checkpoint load and ONNX tracing. Default: cuda.",
    )
    parser.add_argument("--opset-version", type=int, default=20, help="ONNX opset version. Default: 20.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run onnx.checker on the exported graph. This requires the onnx package.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Resolve assets and print the export plan without writing an ONNX file.",
    )
    return parser


def build_musvit_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} {TARGET_MUSVIT}",
        description="Export PRAIG/MuSViT sheet-music encoder embeddings to ONNX.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_MUSVIT_REPO_ID,
        help=f"Hugging Face model repository. Default: {DEFAULT_MUSVIT_REPO_ID}.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision or commit hash.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_MUSVIT_OUTPUT_PATH,
        help=f"Path to the output ONNX model. Default: {DEFAULT_MUSVIT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Path to the output metadata JSON. Defaults to the ONNX path with .json suffix.",
    )
    parser.add_argument(
        "--preprocessor-config-path",
        type=Path,
        default=None,
        help="Path to the output preprocessor_config.json. Defaults to the ONNX output directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Device used for model load and ONNX tracing. Default: cpu.",
    )
    parser.add_argument("--opset-version", type=int, default=20, help="ONNX opset version. Default: 20.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Fixed square input size used for export. Default: 1024.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used for export tracing. Default: 1.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("page_resize", "pad_square"),
        default="page_resize",
        help="Preprocessing contract recorded in metadata. Default: page_resize.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run onnx.checker on the exported graph. This requires the onnx package.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the export plan without downloading or writing an ONNX file.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    target = TARGET_ROFORMER
    if raw_args and raw_args[0] in {TARGET_ROFORMER, TARGET_MUSVIT}:
        target = raw_args.pop(0)

    if target == TARGET_MUSVIT:
        args = build_musvit_arg_parser().parse_args(raw_args)
    else:
        args = build_roformer_arg_parser().parse_args(raw_args)
    args.target = target
    return args


def detect_roformer_model_type(config: Mapping[str, Any]) -> str:
    model_cfg = config.get("model", {})
    if "num_bands" in model_cfg:
        return MODEL_TYPE_MEL_BAND_ROFORMER
    return MODEL_TYPE_BS_ROFORMER


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.load(handle, Loader=_TupleSafeLoader)

    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mapping in config file, got {type(loaded).__name__}: {config_path}")

    return loaded


def resolve_model_assets(
    model_dir: Path | None,
    config_path: Path | None,
    checkpoint_path: Path | None,
) -> tuple[Path, Path]:
    if config_path and checkpoint_path:
        return config_path.resolve(), checkpoint_path.resolve()

    if model_dir is None:
        raise ValueError("Provide either --model-dir or both --config-path and --checkpoint-path.")

    model_dir = model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    config_candidates = sorted(model_dir.glob("*.yaml"))
    checkpoint_candidates = sorted([*model_dir.glob("*.ckpt"), *model_dir.glob("*.pt")])

    if config_path is None:
        if len(config_candidates) != 1:
            raise ValueError(
                f"Expected exactly one YAML config in {model_dir}, found {len(config_candidates)}. Use --config-path explicitly.",
            )
        config_path = config_candidates[0]

    if checkpoint_path is None:
        if len(checkpoint_candidates) != 1:
            raise ValueError(
                f"Expected exactly one checkpoint in {model_dir}, found {len(checkpoint_candidates)}. Use --checkpoint-path explicitly.",
            )
        checkpoint_path = checkpoint_candidates[0]

    return config_path.resolve(), checkpoint_path.resolve()


def choose_output_path(output_path: Path | None, checkpoint_path: Path) -> Path:
    if output_path is not None:
        return output_path.resolve()

    return checkpoint_path.with_suffix(".onnx").resolve()


def determine_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def derive_num_stems(config: Mapping[str, Any]) -> int:
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    target_instrument = training_cfg.get("target_instrument")
    instruments = training_cfg.get("instruments", [])
    return int(model_cfg.get("num_stems", 1 if target_instrument else len(instruments)))


def derive_output_stem_names(config: Mapping[str, Any]) -> list[str]:
    training_cfg = config.get("training", {})
    instruments = [str(item) for item in training_cfg.get("instruments", [])]
    target_instrument = training_cfg.get("target_instrument")
    num_stems = derive_num_stems(config)

    if num_stems == 1 and target_instrument:
        return [str(target_instrument)]

    return instruments[:num_stems]


def derive_secondary_stem_name(config: Mapping[str, Any]) -> str | None:
    training_cfg = config.get("training", {})
    target_instrument = training_cfg.get("target_instrument")
    if not target_instrument or derive_num_stems(config) != 1:
        return None

    for name in training_cfg.get("instruments", []):
        candidate = str(name)
        if candidate != str(target_instrument):
            return candidate
    return None


def build_bs_roformer_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    if "model" not in config:
        raise KeyError("Config is missing the 'model' section.")

    model_cfg = dict(config["model"])

    if "num_stems" not in model_cfg:
        model_cfg["num_stems"] = derive_num_stems(config)

    if "stereo" not in model_cfg:
        model_cfg["stereo"] = int(config.get("audio", {}).get("num_channels", 1)) == 2

    # Export uses the latest bs_roformer package, but this checkpoint matches the single residual stream path.
    model_cfg["flash_attn"] = False
    model_cfg["num_residual_streams"] = 1
    # Reference RoformerLoader defaults zero_dc=True when the YAML omits it.
    model_cfg.setdefault("zero_dc", True)

    kwargs: dict[str, Any] = {}
    unsupported: list[tuple[str, Any]] = []

    for key, value in model_cfg.items():
        if key in BS_ROFORMER_LATEST_CONSTRUCTOR_KEYS or key == "num_residual_streams":
            kwargs[key] = value
            continue

        if key in IGNORED_BS_ROFORMER_CONFIG_KEYS and value == IGNORED_BS_ROFORMER_CONFIG_KEYS[key]:
            continue

        unsupported.append((key, value))

    if unsupported:
        pretty = ", ".join(f"{name}={value!r}" for name, value in unsupported)
        raise ValueError(
            "Unsupported non-default BS-RoFormer config keys for bs_roformer 1.1.0 export: "
            f"{pretty}"
        )

    return kwargs


def build_mel_band_roformer_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    if "model" not in config:
        raise KeyError("Config is missing the 'model' section.")

    model_cfg = dict(config["model"])

    if "num_stems" not in model_cfg:
        model_cfg["num_stems"] = derive_num_stems(config)

    if "stereo" not in model_cfg:
        model_cfg["stereo"] = int(config.get("audio", {}).get("num_channels", 1)) == 2

    model_cfg["flash_attn"] = False
    model_cfg.setdefault("linear_transformer_depth", 0)
    model_cfg.setdefault("linear_flash_attn", False)
    model_cfg["num_residual_streams"] = 1
    model_cfg["add_value_residual"] = False
    model_cfg.setdefault("zero_dc", True)

    kwargs: dict[str, Any] = {}
    unsupported: list[tuple[str, Any]] = []

    for key, value in model_cfg.items():
        if key in MEL_BAND_ROFORMER_LATEST_CONSTRUCTOR_KEYS:
            kwargs[key] = value
            continue

        if key in IGNORED_MEL_BAND_ROFORMER_CONFIG_KEYS and value == IGNORED_MEL_BAND_ROFORMER_CONFIG_KEYS[key]:
            continue

        unsupported.append((key, value))

    if unsupported:
        pretty = ", ".join(f"{name}={value!r}" for name, value in unsupported)
        raise ValueError(
            "Unsupported non-default MelBand-RoFormer config keys for current export path: "
            f"{pretty}"
        )

    return kwargs


def build_model_kwargs(config: Mapping[str, Any], model_type: str) -> dict[str, Any]:
    if model_type == MODEL_TYPE_MEL_BAND_ROFORMER:
        return build_mel_band_roformer_kwargs(config)
    return build_bs_roformer_kwargs(config)


def import_roformer_models():
    try:
        from bs_roformer import BSRoformer, MelBandRoformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'BS-RoFormer'. Install it first, for example:\n"
            "  uv pip install BS-RoFormer onnx"
        ) from exc

    return BSRoformer, MelBandRoformer


def instantiate_separator(config: Mapping[str, Any]) -> torch.nn.Module:
    model_type = detect_roformer_model_type(config)
    BSRoformer, MelBandRoformer = import_roformer_models()
    kwargs = build_model_kwargs(config, model_type)
    if model_type == MODEL_TYPE_MEL_BAND_ROFORMER:
        return MelBandRoformer(**kwargs)
    return BSRoformer(**kwargs)


def load_checkpoint_state_dict(checkpoint_path: Path) -> OrderedDict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "weights", "net"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint did not resolve to a state dict mapping: {type(checkpoint).__name__}")

    state_dict = OrderedDict((str(key), value) for key, value in checkpoint.items())
    return strip_known_prefixes(state_dict)


def strip_known_prefixes(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    prefixes = ("model.", "module.")
    stripped = state_dict

    changed = True
    while changed and stripped:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in stripped):
                stripped = OrderedDict((key[len(prefix) :], value) for key, value in stripped.items())
                changed = True
                break

    return stripped


def rewrite_checkpoint_for_latest_bs_roformer(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    rewritten = OrderedDict()

    for key, value in state_dict.items():
        match = OLD_TO_LATEST_BRANCH_PATTERN.match(key)
        if match:
            key = f"{match.group(1)}.branch.{match.group(2)}"
        rewritten[key] = value

    return rewritten


def rewrite_checkpoint_for_latest_mel_band_roformer(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    rewritten = OrderedDict()
    has_legacy_slots = any(re.match(r"^layers\.\d+\.0\.", key) for key in state_dict) and not any(
        re.match(r"^layers\.\d+\.2\.", key) for key in state_dict
    )

    for key, value in state_dict.items():
        if has_legacy_slots:
            match = OLD_TO_LATEST_MEL_BAND_LAYER_PATTERN.match(key)
            if match:
                key = f"{match.group(1)}.{int(match.group(2)) + 1}.{match.group(3)}"

        match = OLD_TO_LATEST_BRANCH_PATTERN.match(key)
        if match:
            key = f"{match.group(1)}.branch.{match.group(2)}"

        rewritten[key] = value

    return rewritten


def initialize_latest_only_missing_params(separator: torch.nn.Module, missing_keys: list[str]) -> list[str]:
    unresolved: list[str] = []
    named_parameters = dict(separator.named_parameters())

    with torch.no_grad():
        for key in missing_keys:
            if key.endswith("branch.to_value_residual_mix.weight") and key in named_parameters:
                named_parameters[key].zero_()
                continue

            if key.endswith("branch.to_value_residual_mix.bias") and key in named_parameters:
                named_parameters[key].fill_(-20.0)
                continue

            unresolved.append(key)

    return unresolved


def load_separator_weights(separator: torch.nn.Module, checkpoint_path: Path, *, model_type: str) -> None:
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    if model_type == MODEL_TYPE_MEL_BAND_ROFORMER:
        state_dict = rewrite_checkpoint_for_latest_mel_band_roformer(state_dict)
    else:
        state_dict = rewrite_checkpoint_for_latest_bs_roformer(state_dict)
    missing, unexpected = separator.load_state_dict(state_dict, strict=False)
    missing = initialize_latest_only_missing_params(separator, list(missing))

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match export graph.\n"
            f"Missing keys: {missing[:20]}\n"
            f"Unexpected keys: {unexpected[:20]}"
        )


def create_dummy_stft_features(config: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    audio_cfg = config.get("audio", {})
    model_cfg = config.get("model", {})

    chunk_size = int(audio_cfg.get("chunk_size", 0))
    if chunk_size <= 0:
        raise ValueError(f"Invalid audio.chunk_size in config: {chunk_size!r}")

    num_channels = int(audio_cfg.get("num_channels", 2 if model_cfg.get("stereo") else 1))
    n_fft = int(model_cfg.get("stft_n_fft", audio_cfg.get("n_fft", 2048)))
    hop_length = int(model_cfg.get("stft_hop_length", audio_cfg.get("hop_length", n_fft // 4)))
    win_length = int(model_cfg.get("stft_win_length", n_fft))
    normalized = bool(model_cfg.get("stft_normalized", False))

    raw_audio = torch.randn(1, num_channels, chunk_size, device=device)
    flat_audio = raw_audio.reshape(-1, chunk_size)
    window = torch.hann_window(win_length, device=device)

    stft = torch.stft(
        flat_audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        normalized=normalized,
        window=window,
        return_complex=True,
    )
    stft = torch.view_as_real(stft)
    stft = stft.reshape(1, num_channels, stft.shape[1], stft.shape[2], 2)
    return rearrange(stft, "b s f t c -> b t (f s c)")


class BSRoformerMaskWrapper(torch.nn.Module):
    """Expose the ONNX-friendly core graph: STFT features in, complex masks out."""

    def __init__(self, separator: torch.nn.Module):
        super().__init__()
        self.separator = separator

    def forward(self, stft_features: torch.Tensor) -> torch.Tensor:
        x = self.separator.band_split(stft_features)

        time_v_residual = None
        freq_v_residual = None

        x = self.separator.expand_stream(x)

        for time_transformer, freq_transformer in self.separator.layers:
            x = rearrange(x, "b t f d -> b f t d")
            batch_freq_shape = x.shape[:-2]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            x, next_time_v_residual = time_transformer(x, value_residual=time_v_residual)
            if time_v_residual is None:
                time_v_residual = next_time_v_residual
            x = x.reshape(*batch_freq_shape, x.shape[-2], x.shape[-1])

            x = rearrange(x, "b f t d -> b t f d")
            batch_time_shape = x.shape[:-2]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            x, next_freq_v_residual = freq_transformer(x, value_residual=freq_v_residual)
            if freq_v_residual is None:
                freq_v_residual = next_freq_v_residual
            x = x.reshape(*batch_time_shape, x.shape[-2], x.shape[-1])

        x = self.separator.reduce_stream(x)
        x = self.separator.final_norm(x)

        mask = torch.stack([estimator(x) for estimator in self.separator.mask_estimators], dim=1)
        return rearrange(mask, "b n t (f c) -> b n f t c", c=2)


class MelBandRoformerMaskWrapper(torch.nn.Module):
    """Expose an ONNX-friendly graph: full STFT features in, averaged full-band masks out."""

    def __init__(self, separator: torch.nn.Module):
        super().__init__()
        self.separator = separator

    def forward(self, stft_features: torch.Tensor) -> torch.Tensor:
        batch = stft_features.shape[0]
        channels = self.separator.audio_channels

        stft_repr = rearrange(stft_features, "b t (f s c) -> b (f s) t c", s=channels, c=2)
        batch_arange = torch.arange(batch, device=stft_features.device)[..., None]
        x = stft_repr[batch_arange, self.separator.freq_indices]
        x = rearrange(x, "b f t c -> b t (f c)")
        x = self.separator.band_split(x)

        linear_value_residual = None
        time_value_residual = None
        freq_value_residual = None
        x = self.separator.expand_streams(x)

        for linear_transformer, time_transformer, freq_transformer in self.separator.layers:
            if linear_transformer is not None:
                x, ft_ps = pack([x], "b * d")
                x, next_linear_values = linear_transformer(x, value_residual=linear_value_residual)
                if linear_value_residual is None:
                    linear_value_residual = next_linear_values
                (x,) = unpack(x, ft_ps, "b * d")

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x, next_time_values = time_transformer(x, value_residual=time_value_residual)
            if time_value_residual is None:
                time_value_residual = next_time_values
            (x,) = unpack(x, ps, "* t d")

            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x, next_freq_values = freq_transformer(x, value_residual=freq_value_residual)
            if freq_value_residual is None:
                freq_value_residual = next_freq_values
            (x,) = unpack(x, ps, "* f d")

        x = self.separator.reduce_streams(x)
        masks = torch.stack([fn(x) for fn in self.separator.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        scatter_indices = repeat(
            self.separator.freq_indices,
            "f -> b n f t c",
            b=batch,
            n=self.separator.num_stems,
            t=masks.shape[-2],
            c=masks.shape[-1],
        )
        expanded = torch.zeros(
            batch,
            self.separator.num_stems,
            stft_repr.shape[1],
            stft_repr.shape[2],
            masks.shape[-1],
            dtype=masks.dtype,
            device=masks.device,
        )
        summed = expanded.scatter_add_(2, scatter_indices, masks)
        denom = repeat(self.separator.num_bands_per_freq, "f -> 1 1 (f r) 1 1", r=channels).to(
            device=summed.device,
            dtype=summed.dtype,
        )
        averaged = summed / denom.clamp(min=1e-8)
        return averaged


class MuSViTEncoderWrapper(torch.nn.Module):
    """Expose only the MuSViT encoder hidden states, not the MAE decoder/reconstruction head."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values)
        return output.last_hidden_state


def _coerce_positive_int(value: Any, default: int, *, field_name: str) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed!r}")
    return parsed


def _resolve_config_value(config: Any, name: str, default: int) -> int:
    return _coerce_positive_int(getattr(config, name, None), default, field_name=name)


def build_musvit_load_kwargs(revision: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "add_pooling_layer": False,
    }
    if revision:
        kwargs["revision"] = revision
    return kwargs


def load_musvit_model(repo_id: str, revision: str | None = None) -> torch.nn.Module:
    try:
        from transformers import ViTModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers'. Use the project .venv or install the torch/transformers export stack first."
        ) from exc

    kwargs = build_musvit_load_kwargs(revision)
    try:
        return ViTModel.from_pretrained(repo_id, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MuSViT model from {repo_id}. If this is a gated model, log in to Hugging Face "
            "and accept the model conditions before export."
        ) from exc


def create_dummy_musvit_pixels(batch_size: int, image_size: int, device: torch.device) -> torch.Tensor:
    batch_size = _coerce_positive_int(batch_size, 1, field_name="batch_size")
    image_size = _coerce_positive_int(image_size, 1024, field_name="image_size")
    return torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=torch.float32)


def build_musvit_export_metadata(
    *,
    model: Any,
    repo_id: str,
    revision: str | None,
    output_path: Path,
    image_size: int,
    preprocess_mode: str,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    hidden_size = _resolve_config_value(config, "hidden_size", 768)
    patch_size = _resolve_config_value(config, "patch_size", 16)
    num_channels = _resolve_config_value(config, "num_channels", 3)
    resolved_image_size = _resolve_config_value(config, "image_size", image_size)
    if resolved_image_size != image_size:
        resolved_image_size = image_size
    if resolved_image_size % patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size: image_size={resolved_image_size}, patch_size={patch_size}")

    patch_axis = resolved_image_size // patch_size
    resolved_revision = revision or getattr(config, "_commit_hash", None)
    return {
        "model_type": TARGET_MUSVIT,
        "repo_id": repo_id,
        "revision": resolved_revision,
        "onnx_path": str(output_path),
        "input_name": "pixel_values",
        "output_name": "last_hidden_state",
        "input_layout": ["batch", "channels", "height", "width"],
        "output_layout": ["batch", "tokens", "hidden"],
        "input_dtype": "float32",
        "output_dtype": "float32",
        "image_size": [resolved_image_size, resolved_image_size],
        "num_channels": num_channels,
        "patch_size": patch_size,
        "patch_grid": [patch_axis, patch_axis],
        "patch_tokens": patch_axis * patch_axis,
        "output_tokens": patch_axis * patch_axis + 1,
        "hidden_size": hidden_size,
        "contains_cls_token": True,
        "preprocess": {
            "mode": preprocess_mode,
            "resize": [resolved_image_size, resolved_image_size],
            "color": "RGB",
            "scale": "0_to_1",
            "normalize": None,
        },
        "license": "CC BY-NC-SA 4.0",
        "export_notes": {
            "uses_vit_model_encoder": True,
            "does_not_export_mae_decoder": True,
            "fixed_page_size": True,
            "gated_source_model": True,
        },
    }


def write_musvit_export_metadata(metadata: Mapping[str, Any], output_path: Path, metadata_path: Path | None = None) -> Path:
    resolved_path = metadata_path.resolve() if metadata_path is not None else output_path.with_suffix(".json").resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved_path


def build_musvit_preprocessor_config(*, image_size: int, preprocess_mode: str) -> dict[str, Any]:
    image_size = _coerce_positive_int(image_size, 1024, field_name="image_size")
    return {
        "image_processor_type": "ViTImageProcessor",
        "do_convert_rgb": True,
        "do_resize": True,
        "size": {
            "height": image_size,
            "width": image_size,
        },
        "resample": 2,
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": False,
        "preprocess_mode": preprocess_mode,
    }


def write_musvit_preprocessor_config(
    config: Mapping[str, Any],
    output_path: Path,
    preprocessor_config_path: Path | None = None,
) -> Path:
    resolved_path = (
        preprocessor_config_path.resolve()
        if preprocessor_config_path is not None
        else (output_path.parent / "preprocessor_config.json").resolve()
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved_path


def format_musvit_plan(args: argparse.Namespace, output_path: Path, device: torch.device) -> str:
    return json.dumps(
        {
            "target": TARGET_MUSVIT,
            "repo_id": args.repo_id,
            "revision": args.revision,
            "output_path": str(output_path),
            "metadata_path": str(args.metadata_path.resolve()) if args.metadata_path else str(output_path.with_suffix(".json")),
            "preprocessor_config_path": (
                str(args.preprocessor_config_path.resolve())
                if args.preprocessor_config_path
                else str((output_path.parent / "preprocessor_config.json").resolve())
            ),
            "device": str(device),
            "opset_version": int(args.opset_version),
            "image_size": int(args.image_size),
            "batch_size": int(args.batch_size),
            "preprocess_mode": args.preprocess_mode,
            "input_name": "pixel_values",
            "output_name": "last_hidden_state",
        },
        indent=2,
        ensure_ascii=False,
    )


def export_musvit_to_onnx(
    *,
    model: torch.nn.Module,
    output_path: Path,
    device: torch.device,
    opset_version: int,
    image_size: int,
    batch_size: int,
    verify: bool,
) -> Path:
    dummy_input = create_dummy_musvit_pixels(batch_size, image_size, device)
    wrapper = MuSViTEncoderWrapper(model).to(device)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
            },
        )

    if verify:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("The --verify flag requires the 'onnx' package to be installed.") from exc

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    return output_path


def export_to_onnx(
    separator: torch.nn.Module,
    config: Mapping[str, Any],
    model_type: str,
    output_path: Path,
    device: torch.device,
    opset_version: int,
    verify: bool,
) -> Path:
    dummy_input = create_dummy_stft_features(config, device)
    if model_type == MODEL_TYPE_MEL_BAND_ROFORMER:
        wrapper = MelBandRoformerMaskWrapper(separator).to(device)
    else:
        wrapper = BSRoformerMaskWrapper(separator).to(device)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
            input_names=["stft_features"],
            output_names=["mask"],
            dynamic_axes={
                "stft_features": {0: "batch_size", 1: "num_frames"},
                "mask": {0: "batch_size", 3: "num_frames"},
            },
        )

    if verify:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("The --verify flag requires the 'onnx' package to be installed.") from exc

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    return output_path


def build_export_metadata(config: Mapping[str, Any], checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    model_type = detect_roformer_model_type(config)
    model_cfg = config.get("model", {})
    audio_cfg = config.get("audio", {})
    inference_cfg = config.get("inference", {})
    kwargs = build_model_kwargs(config, model_type)
    stem_names = derive_output_stem_names(config)
    hop_length = model_cfg.get("stft_hop_length", audio_cfg.get("hop_length"))
    default_segment_size = inference_cfg.get("dim_t")
    if default_segment_size and hop_length:
        chunk_size = int(hop_length) * (int(default_segment_size) - 1)
    else:
        chunk_size = audio_cfg.get("chunk_size")

    return {
        "model_type": model_type,
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(output_path),
        "input_name": "stft_features",
        "output_name": "mask",
        "input_layout": ["batch", "frames", "freq_channels_complex"],
        "output_layout": ["batch", "stems", "freq_channels", "frames", "complex"],
        "sample_rate": audio_cfg.get("sample_rate"),
        "chunk_size": chunk_size,
        "num_channels": audio_cfg.get("num_channels"),
        "num_stems": int(kwargs.get("num_stems", len(stem_names))),
        "stem_names": stem_names,
        "secondary_stem_name": derive_secondary_stem_name(config),
        "stft": {
            "n_fft": model_cfg.get("stft_n_fft", audio_cfg.get("n_fft")),
            "hop_length": model_cfg.get("stft_hop_length", audio_cfg.get("hop_length")),
            "win_length": model_cfg.get("stft_win_length", model_cfg.get("stft_n_fft", audio_cfg.get("n_fft"))),
            "normalized": model_cfg.get("stft_normalized", False),
            "zero_dc": bool(kwargs.get("zero_dc", False)),
        },
        "export_notes": {
            "flash_attn_disabled_for_export": True,
            "graph_exposes_mask_head_only": True,
            "requires_external_stft_preprocess": True,
            "requires_external_istft_postprocess": True,
        },
    }


def write_export_metadata(metadata: Mapping[str, Any], output_path: Path) -> Path:
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def format_plan(config_path: Path, checkpoint_path: Path, output_path: Path, device: torch.device, config: Mapping[str, Any]) -> str:
    model_type = detect_roformer_model_type(config)
    model_cfg = config.get("model", {})
    audio_cfg = config.get("audio", {})
    inference_cfg = config.get("inference", {})
    hop_length = model_cfg.get("stft_hop_length", audio_cfg.get("hop_length"))
    default_segment_size = inference_cfg.get("dim_t")
    if default_segment_size and hop_length:
        chunk_size = int(hop_length) * (int(default_segment_size) - 1)
    else:
        chunk_size = audio_cfg.get("chunk_size")
    return json.dumps(
        {
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "output_path": str(output_path),
            "device": str(device),
            "model_type": model_type,
            "num_stems": derive_num_stems(config),
            "stems": derive_output_stem_names(config),
            "secondary_stem_name": derive_secondary_stem_name(config),
            "chunk_size": chunk_size,
            "sample_rate": audio_cfg.get("sample_rate"),
            "stft_n_fft": model_cfg.get("stft_n_fft", audio_cfg.get("n_fft")),
            "stft_hop_length": model_cfg.get("stft_hop_length", audio_cfg.get("hop_length")),
        },
        indent=2,
        ensure_ascii=False,
    )


def run_roformer_export(args: argparse.Namespace) -> int:
    config_path, checkpoint_path = resolve_model_assets(args.model_dir, args.config_path, args.checkpoint_path)
    output_path = choose_output_path(args.output_path, checkpoint_path)
    device = determine_device(args.device)
    config = load_yaml_config(config_path)
    model_type = detect_roformer_model_type(config)

    print(format_plan(config_path, checkpoint_path, output_path, device, config), flush=True)

    if args.print_only:
        return 0

    separator = instantiate_separator(config)
    load_separator_weights(separator, checkpoint_path, model_type=model_type)
    separator.to(device)
    separator.eval()

    export_to_onnx(
        separator=separator,
        config=config,
        model_type=model_type,
        output_path=output_path,
        device=device,
        opset_version=args.opset_version,
        verify=args.verify,
    )

    metadata_path = write_export_metadata(build_export_metadata(config, checkpoint_path, output_path), output_path)
    print(f"Exported ONNX model: {output_path}", flush=True)
    print(f"Export metadata: {metadata_path}", flush=True)
    return 0


def run_musvit_export(args: argparse.Namespace) -> int:
    output_path = args.output_path.resolve()
    device = determine_device(args.device)
    print(format_musvit_plan(args, output_path, device), flush=True)

    if args.print_only:
        return 0

    model = load_musvit_model(args.repo_id, revision=args.revision)
    model.to(device)
    model.eval()

    export_musvit_to_onnx(
        model=model,
        output_path=output_path,
        device=device,
        opset_version=int(args.opset_version),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        verify=bool(args.verify),
    )

    metadata = build_musvit_export_metadata(
        model=model,
        repo_id=args.repo_id,
        revision=args.revision,
        output_path=output_path,
        image_size=int(args.image_size),
        preprocess_mode=args.preprocess_mode,
    )
    metadata_path = write_musvit_export_metadata(metadata, output_path, args.metadata_path)
    preprocessor_config = build_musvit_preprocessor_config(
        image_size=int(args.image_size),
        preprocess_mode=args.preprocess_mode,
    )
    preprocessor_config_path = write_musvit_preprocessor_config(
        preprocessor_config,
        output_path,
        args.preprocessor_config_path,
    )
    print(f"Exported MuSViT ONNX model: {output_path}", flush=True)
    print(f"Export metadata: {metadata_path}", flush=True)
    print(f"Preprocessor config: {preprocessor_config_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target == TARGET_MUSVIT:
        return run_musvit_export(args)
    return run_roformer_export(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[onnx_export] {exc}", file=sys.stderr, flush=True)
        raise
