"""Mega-ASR local transcription provider."""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Any

from module.caption_pipeline.postprocess import strip_reasoning_sections
from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.local_alm_base import ALMTaskContract, LocalALMProvider
from module.providers.registry import register_provider
from utils.parse_display import extract_code_block_content


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_project_path(value: Any, default: str) -> Path:
    raw = str(value or default).strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return _project_root() / path


def _str_to_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_torch_dtype(value: Any, *, device_map: Any) -> Any:
    import torch

    if value not in (None, ""):
        normalized = str(value).strip().removeprefix("torch.")
        return getattr(torch, normalized)
    return torch.float32 if str(device_map) == "cpu" else torch.bfloat16


def _extract_transcribe_text(results: Any) -> str:
    if isinstance(results, list):
        texts = [str(getattr(item, "text", item)).strip() for item in results]
        return "\n".join(text for text in texts if text)
    return str(getattr(results, "text", results)).strip()


class _LoRADeltaSwitch:
    def __init__(self, keep_delta_on_gpu: bool = True) -> None:
        self.keep_delta_on_gpu = keep_delta_on_gpu
        self.items: list[dict[str, Any]] = []
        self.active = False

    def _load_adapter_state(self, adapter_dir: str | Path) -> dict[str, Any]:
        import torch
        from safetensors.torch import load_file as safe_load_file

        adapter_dir = str(adapter_dir)
        safetensors_path = Path(adapter_dir) / "adapter_model.safetensors"
        bin_path = Path(adapter_dir) / "adapter_model.bin"
        if safetensors_path.exists():
            return safe_load_file(str(safetensors_path))
        return torch.load(str(bin_path), map_location="cpu")

    @staticmethod
    def _load_adapter_config(adapter_dir: str | Path) -> dict[str, Any]:
        config_path = Path(adapter_dir) / "adapter_config.json"
        return json.loads(config_path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_adapter_blocks(adapter_dir: str | Path) -> dict[str, Any]:
        blocks_path = Path(adapter_dir) / "mega_lora_blocks.json"
        if not blocks_path.exists():
            return {}
        return json.loads(blocks_path.read_text(encoding="utf-8"))

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        for prefix in ("base_model.model.",):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name.startswith("thinker.layers."):
            name = name.replace("thinker.layers.", "thinker.model.layers.", 1)
        return name

    @staticmethod
    def _module_name_candidates(name: str) -> list[str]:
        candidates = [name]
        if name.startswith("model."):
            candidates.append(name[len("model.") :])
        if name.startswith("thinker.layers."):
            candidates.append(name.replace("thinker.layers.", "thinker.model.layers.", 1))
        if name.startswith("thinker.model."):
            candidates.append(name.replace("thinker.model.", "thinker.", 1))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _raw_module_name(key: str, marker: str) -> str:
        name = key.split(marker)[0]
        for prefix in ("base_model.model.", "model."):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _split_lora_key(self, key: str) -> tuple[str | None, str | None, str | None]:
        raw_key = key
        key = self._normalize_module_name(key)
        for marker in (".lora_A.", ".lora_B."):
            if marker in key:
                module_name = key.split(marker)[0]
                raw_module_name = self._raw_module_name(raw_key, marker)
                kind = "A" if marker == ".lora_A." else "B"
                return module_name, raw_module_name, kind
        return None, None, None

    def add_adapter(self, parent_module: Any, adapter_dir: str | Path, name: str) -> None:
        config = self._load_adapter_config(adapter_dir)
        state = self._load_adapter_state(adapter_dir)
        blocks = self._load_adapter_blocks(adapter_dir)

        lora_alpha = config.get("lora_alpha", 1)
        rank = config.get("r")
        alpha_pattern = config.get("alpha_pattern") or {}
        rank_pattern = config.get("rank_pattern") or {}
        fan_in_fan_out = bool(config.get("fan_in_fan_out", False))
        module_dict = dict(parent_module.named_modules())

        grouped: dict[str, dict[str, Any]] = {}
        for key, tensor in state.items():
            module_name, raw_module_name, kind = self._split_lora_key(key)
            if module_name is None or raw_module_name is None or kind is None:
                continue

            matched_name = None
            for candidate in self._module_name_candidates(module_name):
                if candidate in module_dict:
                    matched_name = candidate
                    break

            target_name = matched_name or module_name
            group_key = f"{target_name}\0{raw_module_name}"
            item = grouped.setdefault(
                group_key,
                {
                    "target_module_name": target_name,
                    "raw_module_name": raw_module_name,
                },
            )
            item[kind] = tensor.cpu()

        loaded = 0
        missing = []
        for pair in grouped.values():
            if "A" not in pair or "B" not in pair:
                continue
            module_name = pair["target_module_name"]
            raw_module_name = pair["raw_module_name"]
            if module_name not in module_dict:
                missing.append(module_name)
                continue
            module = module_dict[module_name]
            if not hasattr(module, "weight"):
                missing.append(module_name)
                continue

            weight = module.weight
            a_matrix = pair["A"].to(device=weight.device, dtype=weight.dtype)
            b_matrix = pair["B"].to(device=weight.device, dtype=weight.dtype)
            module_blocks = blocks.get(raw_module_name) or blocks.get(module_name)
            deltas = self._build_lora_deltas(
                a_matrix=a_matrix,
                b_matrix=b_matrix,
                module_blocks=module_blocks,
                raw_module_name=raw_module_name,
                module_name=module_name,
                rank=rank,
                rank_pattern=rank_pattern,
                lora_alpha=lora_alpha,
                alpha_pattern=alpha_pattern,
                fan_in_fan_out=fan_in_fan_out,
            )

            for delta in deltas:
                if delta.shape != weight.shape:
                    try:
                        delta = delta.reshape(weight.shape)
                    except Exception:
                        missing.append(
                            f"{module_name}: delta shape {tuple(delta.shape)} != weight shape {tuple(weight.shape)}"
                        )
                        continue

                if self.keep_delta_on_gpu:
                    delta = delta.to(device=weight.device, dtype=weight.dtype)
                else:
                    delta = delta.to(device="cpu", dtype=weight.dtype)
                self.items.append({"name": name, "module_name": module_name, "weight": weight, "delta": delta})
                loaded += 1

        if missing:
            warnings.warn(
                f"LoRA adapter {name} loaded {loaded} deltas, missing {len(missing)} modules. "
                f"Examples: {missing[:5]}",
                stacklevel=2,
            )

    @staticmethod
    def _build_lora_deltas(
        *,
        a_matrix: Any,
        b_matrix: Any,
        module_blocks: Any,
        raw_module_name: str,
        module_name: str,
        rank: Any,
        rank_pattern: dict[str, Any],
        lora_alpha: Any,
        alpha_pattern: dict[str, Any],
        fan_in_fan_out: bool,
    ) -> list[Any]:
        import torch

        if module_blocks:
            deltas = []
            for block in module_blocks:
                start = int(block["start"])
                end = int(block["end"])
                block_rank = int(block.get("rank", end - start))
                block_alpha = int(block.get("alpha", block_rank))
                delta = torch.matmul(b_matrix[:, start:end], a_matrix[start:end])
                delta = delta * (float(block_alpha) / float(block_rank))
                if fan_in_fan_out:
                    delta = delta.T
                deltas.append(delta)
            return deltas

        adapter_rank = rank_pattern.get(raw_module_name, rank_pattern.get(module_name, rank))
        if adapter_rank is None:
            adapter_rank = a_matrix.shape[0]
        adapter_alpha = alpha_pattern.get(raw_module_name, alpha_pattern.get(module_name, lora_alpha))
        delta = torch.matmul(b_matrix, a_matrix) * (float(adapter_alpha) / float(adapter_rank))
        if fan_in_fan_out:
            delta = delta.T
        return [delta]

    def set_active(self, active: bool) -> float:
        import torch

        if self.active == active:
            return 0.0
        start = time.perf_counter()
        sign = 1.0 if active else -1.0
        with torch.no_grad():
            for item in self.items:
                weight = item["weight"]
                delta = item["delta"]
                if delta.device != weight.device:
                    delta = delta.to(device=weight.device)
                weight.data.add_(delta, alpha=sign)
        self.active = active
        return time.perf_counter() - start


class _AudioQualityRouter:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        import torch

        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model, self.mel_extractor = self._load_model()

    def _load_model(self) -> tuple[Any, Any]:
        import torch
        from safetensors.torch import load_file as safe_load_file
        from safetensors.torch import safe_open

        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            checkpoint_config = json.loads((metadata or {}).get("config", "{}"))
            config = checkpoint_config.get("model", {})
            state_dict = safe_load_file(str(checkpoint_path), device=self.device)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {}).get("model", {})
            state_dict = checkpoint["model_state_dict"]

        model = _create_audio_quality_model(config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        mel_extractor = _create_log_mel_spectrogram(
            sample_rate=self.sample_rate,
            n_mels=config.get("n_mels", 80),
        )
        mel_extractor.to(self.device)
        mel_extractor.eval()
        return model, mel_extractor

    def _load_audio(self, audio_path: str | Path) -> Any:
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly

        audio_np, sr = sf.read(str(audio_path), always_2d=True)
        audio_np = audio_np.mean(axis=1)
        if sr != self.sample_rate:
            gcd = math.gcd(sr, self.sample_rate)
            audio_np = resample_poly(audio_np, self.sample_rate // gcd, sr // gcd)
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        return waveform.to(self.device)

    def infer(self, audio_path: str | Path) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            waveform = self._load_audio(audio_path)
            mel = self.mel_extractor(waveform)
            mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)
            logits = self.model(mel, mask=None)
            probs = torch.softmax(logits, dim=-1)
            degraded_prob = float(probs[0, 1].item())
        is_degraded = degraded_prob >= self.threshold
        return {
            "is_degraded": is_degraded,
            "degraded_prob": degraded_prob,
            "label": int(is_degraded),
        }

    def predict(self, audio_path: str | Path) -> tuple[bool, float]:
        result = self.infer(audio_path)
        return result["is_degraded"], result["degraded_prob"]


def _create_log_mel_spectrogram(*, sample_rate: int, n_mels: int) -> Any:
    import torch
    import torchaudio

    class LogMelSpectrogram(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=160,
                win_length=400,
                n_mels=n_mels,
                norm="slaney",
                mel_scale="slaney",
            )

        def forward(self, waveform: Any) -> Any:
            mel = self.mel_transform(waveform)
            log_mel = torch.clamp(mel, min=1e-10).log10()
            return (log_mel + 4.0) / 4.0

    return LogMelSpectrogram()


def _create_audio_quality_model(config: dict[str, Any]) -> Any:
    import torch
    import torch.nn.functional as F

    nn = torch.nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: Any) -> Any:
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class AttentionPooling(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.query = nn.Linear(d_model, 1)

        def forward(self, x: Any, mask: Any = None) -> Any:
            weights = self.query(x).squeeze(-1)
            if mask is not None:
                weights = weights.masked_fill(~mask, float("-inf"))
            weights = F.softmax(weights, dim=-1)
            return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

    class ConvFrontend(nn.Module):
        def __init__(self, n_mels: int, d_model: int, dropout: float) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_mels, d_model // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x: Any) -> Any:
            x = x.transpose(1, 2)
            x = self.conv(x)
            return x.transpose(1, 2)

    class AudioQualityClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            n_mels = config.get("n_mels", 80)
            d_model = config.get("d_model", 192)
            nhead = config.get("nhead", 4)
            dim_feedforward = config.get("dim_feedforward", 512)
            dropout = config.get("dropout", 0.1)
            max_len = config.get("max_len", 3000)
            num_classes = config.get("num_classes", 2)

            self.downsample_rate = 4
            self.frontend = ConvFrontend(n_mels, d_model, dropout)
            self.pos_encoder = PositionalEncoding(d_model, max_len // 4 + 100, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=nn.LayerNorm(d_model))
            self.pooling = AttentionPooling(d_model)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

        def forward(self, mels: Any, mask: Any = None) -> Any:
            x = self.frontend(mels)
            time_steps = x.shape[1]
            if mask is not None:
                mask = mask[:, :: self.downsample_rate]
                if mask.shape[1] > time_steps:
                    mask = mask[:, :time_steps]
                elif mask.shape[1] < time_steps:
                    pad = torch.ones(
                        mask.shape[0],
                        time_steps - mask.shape[1],
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    mask = torch.cat([mask, pad], dim=1)

            x = self.pos_encoder(x)
            src_key_padding_mask = ~mask if mask is not None else None
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            x = self.pooling(x, mask)
            return self.classifier(x)

    return AudioQualityClassifier()


class _MegaASRDirectModel:
    def __init__(
        self,
        *,
        model_path: Path,
        lora_dir: Path,
        router_checkpoint: Path,
        routing_enabled: bool,
        quality_threshold: float,
        device_map: str | None,
        quality_device: str | None,
        max_inference_batch_size: int,
        max_new_tokens: int,
        keep_delta_on_gpu: bool,
        dtype: Any = None,
    ) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        resolved_device_map = device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
        resolved_dtype = _resolve_torch_dtype(dtype, device_map=resolved_device_map)

        self.model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=resolved_dtype,
            device_map=resolved_device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        self.router = (
            _AudioQualityRouter(
                router_checkpoint,
                device=quality_device,
                threshold=quality_threshold,
            )
            if routing_enabled
            else None
        )
        self.lora_switch = _LoRADeltaSwitch(keep_delta_on_gpu=keep_delta_on_gpu)
        parent_module = getattr(self.model, "model", self.model)
        self.lora_switch.add_adapter(parent_module=parent_module, adapter_dir=lora_dir, name="mega_asr_adapter")
        self._set_lora(True)

    def _set_lora(self, active: bool) -> None:
        self.lora_switch.set_active(active)

    def _route(self, audio: str | Path) -> tuple[bool, float | None, str]:
        if self.router is None:
            return True, None, "default"
        is_degraded, degraded_prob = self.router.predict(audio)
        return is_degraded, degraded_prob, "router"

    def infer(self, audio: str | Path, *, language: str | None = None, return_route: bool = False) -> Any:
        use_lora, degraded_prob, route_source = self._route(audio)
        self._set_lora(use_lora)
        result = self.model.transcribe(
            audio=str(audio),
            language=language,
        )
        text = _extract_transcribe_text(result)
        if return_route:
            return {
                "text": text,
                "use_lora": use_lora,
                "degraded_prob": degraded_prob,
                "route_source": route_source,
            }
        return text


@register_provider("mega_asr_local")
class MegaASRLocalProvider(LocalALMProvider):
    default_model_id = "zhifeixie/Mega-ASR"
    task_contract = ALMTaskContract(
        task_kind="transcribe",
        consumes_prompts=False,
        requires_language=False,
        default_caption_extension=".txt",
    )
    default_ckpt_dir = "huggingface/Mega-ASR"

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "alm_model", "") == "mega_asr_local" and mime.startswith("audio")

    def _load_model(self):
        ckpt_dir = self._ensure_checkpoint_dir()

        routing_enabled = _str_to_bool(self.model_config.get("routing", True), True)
        threshold = float(self.model_config.get("threshold", 0.5))
        max_inference_batch_size = int(self.model_config.get("max_inference_batch_size", 32))
        max_new_tokens = int(self.model_config.get("max_new_tokens", 256))
        keep_delta_on_gpu = _str_to_bool(self.model_config.get("keep_delta_on_gpu", True), True)
        device_map = str(self.model_config.get("device_map", "") or "").strip() or None
        quality_device = str(self.model_config.get("quality_device", "") or "").strip() or None
        dtype = str(self.model_config.get("dtype", "") or "").strip() or None

        self.log(
            f"Loading Mega-ASR model directly with qwen_asr: {self.model_id} "
            f"(ckpt_dir={ckpt_dir}, routing={routing_enabled})",
            "blue",
        )

        try:
            model = _MegaASRDirectModel(
                model_path=ckpt_dir / "Qwen3-ASR-1.7B",
                lora_dir=ckpt_dir / "mega-asr-merged",
                router_checkpoint=ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors",
                routing_enabled=routing_enabled,
                quality_threshold=threshold,
                device_map=device_map,
                quality_device=quality_device,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                keep_delta_on_gpu=keep_delta_on_gpu,
                dtype=dtype,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Mega-ASR dependency import failed: {exc.name or exc}. "
                "Install the local extra with `uv sync --extra mega-asr-local`."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "Mega-ASR failed while importing or initializing PyTorch/CUDA dependencies. "
                "Install `uv sync --extra mega-asr-local` and verify the selected PyTorch build. "
                f"Original error: {exc}"
            ) from exc

        return {"model": model, "ckpt_dir": ckpt_dir}

    def _ckpt_dir(self) -> Path:
        return _resolve_project_path(self.model_config.get("ckpt_dir"), self.default_ckpt_dir)

    def _ensure_checkpoint_dir(self) -> Path:
        ckpt_dir = self._ckpt_dir()
        if self._has_checkpoint_layout(ckpt_dir):
            return ckpt_dir

        from utils.transformer_loader import snapshot_download_with_reporting

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download_with_reporting(
            self.model_id,
            console=self.ctx.console,
            repo_type="model",
            local_dir=str(ckpt_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if not self._has_checkpoint_layout(ckpt_dir):
            raise RuntimeError(
                "Mega-ASR checkpoint snapshot is incomplete. Expected Qwen3-ASR-1.7B, "
                "mega-asr-merged adapter weights, and audio_quality_router under "
                f"{ckpt_dir}."
            )
        return ckpt_dir

    @staticmethod
    def _has_checkpoint_layout(ckpt_dir: Path) -> bool:
        adapter_dir = ckpt_dir / "mega-asr-merged"
        return (
            (ckpt_dir / "Qwen3-ASR-1.7B" / "config.json").is_file()
            and (adapter_dir / "adapter_config.json").is_file()
            and ((adapter_dir / "adapter_model.safetensors").is_file() or (adapter_dir / "adapter_model.bin").is_file())
            and (ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors").is_file()
        )

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        del prompts
        cached = self._get_or_load_model()
        model = cached["model"]

        audio_path = str(Path(media.uri).resolve())
        language = self._resolve_language()
        result = model.infer(
            audio_path,
            language=language,
            return_route=True,
        )

        metadata: dict[str, Any] = {"provider": self.name}
        if isinstance(result, dict):
            metadata.update(
                {
                    "use_lora": result.get("use_lora"),
                    "degraded_prob": result.get("degraded_prob"),
                    "route_source": result.get("route_source"),
                }
            )
            raw = result.get("text", "")
        else:
            raw = result

        return CaptionResult(raw="" if raw is None else str(raw), metadata=metadata)

    def _resolve_language(self) -> str | None:
        runtime_language = getattr(self.ctx.args, "alm_language", None)
        language_value = runtime_language if runtime_language not in (None, "") else self.model_config.get("language", "")
        language = str(language_value or "").strip()
        return language or None

    def _normalize_transcript_text(self, output: str) -> str:
        cleaned = strip_reasoning_sections(output)
        if not cleaned:
            return ""

        if "```" in cleaned:
            extracted = extract_code_block_content(cleaned, console=self.ctx.console)
            if extracted:
                cleaned = extracted

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def post_validate(self, result: CaptionResult, media: MediaContext, args) -> CaptionResult:
        try:
            result.raw = self._normalize_transcript_text(result.raw)
            if not result.raw:
                raise ValueError("EMPTY_TRANSCRIPT_OUTPUT")
            result.parsed = {
                "task_kind": self.task_contract.task_kind,
                "transcript": result.raw,
                "caption_extension": self.task_contract.default_caption_extension,
                "provider": self.name,
            }
        except Exception as exc:
            raise Exception(f"RETRY_INVALID_TRANSCRIPT: {exc}") from exc
        return result
