"""Gemma 4 local multimodal provider.

One provider serves both VLM and ALM routes:
- image / video understanding
- audio ASR
- audio AST
"""

from __future__ import annotations

import base64 as _base64
import json
import re
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import hf_hub_download
from module.providers.backends import OpenAIChatRuntime, find_model_config_section, resolve_runtime_backend
from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext
from module.providers.capabilities import ProviderCapabilities
from module.providers.catalog import provider_config_sections
from module.providers.local_vlm_base import LocalVLMProvider
from module.providers.registry import register_provider
from module.providers.utils import build_vision_messages, encode_image_to_blob
from utils.parse_display import extract_code_block_content
from utils.stream_util import get_video_duration
from utils.transformer_loader import (
    load_pretrained_component,
    move_pretrained_component,
    prepare_multimodal_inputs,
    resolve_device_dtype,
)


_AUDIO_TASK_ALIASES = {
    "asr": "asr",
    "transcribe": "asr",
    "ast": "ast",
}

_IMAGE_SCORE_HEADING_RE = re.compile(r"^\s*\*\*(?:Evaluation\s+)?Scores:\*\*\s*$|^\s*(?:Evaluation\s+)?Scores:\s*$", re.IGNORECASE)
_IMAGE_DESCRIPTION_HEADING_RE = re.compile(
    r"^\s*\*\*Detailed Visual Description:\*\*\s*$|^\s*Detailed Visual Description:\s*$",
    re.IGNORECASE,
)
_IMAGE_AVERAGE_SCORE_RE = re.compile(r"average\s+score[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_IMAGE_TOTAL_SCORE_RE = re.compile(r"total\s+score[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_IMAGE_BOLD_SCORE_LINE_RE = re.compile(r"^\s*\d+\.\s*\*\*(.+?)\*\*\s*([0-9]+(?:\.[0-9]+)?)\s*$")
_IMAGE_PLAIN_SCORE_LINE_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
_IMAGE_CANONICAL_SCORE_ORDER = [
    "Costume & Makeup & Prop Presentation/Accuracy",
    "Character Portrayal & Posing",
    "Setting & Environment Integration",
    "Lighting & Mood",
    "Composition & Framing",
    "Storytelling & Concept",
    "Level of S*e*x*y",
    "Figure",
    "Overall Impact & Uniqueness",
]
_IMAGE_SCORE_LIMITS = {label: 10 for label in _IMAGE_CANONICAL_SCORE_ORDER}
_IMAGE_SCORE_LIMITS["Setting & Environment Integration"] = 5
_IMAGE_SCORE_LIMITS["Storytelling & Concept"] = 5


def _is_cuda_device(device: Any) -> bool:
    return str(device or "").startswith("cuda")


def _resolve_pretrained_device_map(device: Any, device_map: Any) -> Any:
    if device_map != "auto":
        return device_map
    if not _is_cuda_device(device):
        return device_map
    text = str(device)
    if text == "cuda":
        return "auto"
    return {"": text}


def _resolve_model_load_dtype(device: Any, runtime_dtype: Any) -> Any:
    if _is_cuda_device(device):
        return "auto"
    return runtime_dtype


def _looks_like_modelopt_nvfp4_repo(model_id: str | None) -> bool:
    normalized = str(model_id or "").strip().casefold().replace("_", "-")
    return "nvfp4" in normalized or normalized.startswith("nvidia/gemma-4-")


def _image_score_lookup_key(label: str) -> str:
    simplified = str(label or "").replace("\\", " ").replace("*", "").replace("_", " ")
    simplified = re.sub(r"\s+", " ", simplified).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "", simplified)


_IMAGE_SCORE_LABEL_ALIASES = {
    _image_score_lookup_key("Costume & Makeup & Prop Presentation/Accuracy"): "Costume & Makeup & Prop Presentation/Accuracy",
    _image_score_lookup_key("Costume & Makeup & Prop Presentation/Accuracy (in the Photo)"): "Costume & Makeup & Prop Presentation/Accuracy",
    _image_score_lookup_key("Character Portrayal & Posing"): "Character Portrayal & Posing",
    _image_score_lookup_key("Character Portrayal & Posing (Captured by the Photographer)"): "Character Portrayal & Posing",
    _image_score_lookup_key("Setting & Environment Integration"): "Setting & Environment Integration",
    _image_score_lookup_key("Lighting & Mood"): "Lighting & Mood",
    _image_score_lookup_key("Composition & Framing"): "Composition & Framing",
    _image_score_lookup_key("Composition & Framing (Serving the Cosplay)"): "Composition & Framing",
    _image_score_lookup_key("Storytelling & Concept"): "Storytelling & Concept",
    _image_score_lookup_key("Level of S*e*x*y"): "Level of S*e*x*y",
    _image_score_lookup_key("Level of Sexy"): "Level of S*e*x*y",
    _image_score_lookup_key("Figure"): "Figure",
    _image_score_lookup_key("Overall Impact & Uniqueness"): "Overall Impact & Uniqueness",
}


@register_provider("gemma4_local")
class Gemma4LocalProvider(LocalVLMProvider):
    """Route-level provider for Gemma 4 multimodal local runtimes."""

    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_audio=True,
        supports_video=True,
        supports_images=True,
    )
    default_model_id = "google/gemma-4-E2B-it"

    @property
    def model_config(self) -> Dict[str, Any]:
        for section_name in provider_config_sections(self.name):
            section = self.ctx.config.get(section_name, {})
            if section:
                return section
        return {}

    @property
    def model_id(self) -> str:
        config = self.model_config
        args = self.ctx.args

        args_model_id = str(getattr(args, "gemma4_model_id", "") or "").strip()
        if args_model_id:
            return args_model_id

        config_model_id = str(config.get("model_id", "") or "").strip()
        if config_model_id:
            return config_model_id

        return self.default_model_id

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        return str(model_id or "").strip().casefold().replace("_", "-")

    def _model_supports_audio(self, model_id: str | None = None) -> bool:
        normalized = self._normalize_model_id(model_id or self.model_id)
        if not normalized:
            return True
        if "gemma-4-e2b" in normalized or "gemma-4-e4b" in normalized:
            return True
        if "gemma-4-26b" in normalized or "gemma-4-31b" in normalized:
            return False
        return True

    def _chat_template_source_repo(self, model_id: str | None = None) -> str | None:
        normalized = self._normalize_model_id(model_id or self.model_id)
        if "gemma-4-31b" in normalized:
            return "google/gemma-4-31B-it"
        if "gemma-4-26b" in normalized:
            return "google/gemma-4-26B-A4B-it"
        if "gemma-4-e4b" in normalized:
            return "google/gemma-4-E4B-it"
        if "gemma-4-e2b" in normalized:
            return "google/gemma-4-E2B-it"
        return None

    def _ensure_processor_chat_template(self, processor: Any, model_id: str) -> None:
        current_template = getattr(processor, "chat_template", None)
        tokenizer = getattr(processor, "tokenizer", None)
        tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None

        if current_template:
            return
        if tokenizer_template:
            setattr(processor, "chat_template", tokenizer_template)
            return

        template_repo = self._chat_template_source_repo(model_id)
        if not template_repo:
            raise RuntimeError(
                f"GEMMA4_CHAT_TEMPLATE_MISSING: model_id={model_id!r}; no fallback template source is known."
            )

        try:
            template_path = hf_hub_download(template_repo, "chat_template.jinja")
            template_text = Path(template_path).read_text(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(
                f"GEMMA4_CHAT_TEMPLATE_MISSING: model_id={model_id!r}; failed to load chat_template.jinja "
                f"from fallback repo {template_repo!r}: {exc}"
            ) from exc

        setattr(processor, "chat_template", template_text)
        if tokenizer is not None and not getattr(tokenizer, "chat_template", None):
            setattr(tokenizer, "chat_template", template_text)

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        if mime.startswith(("image", "video")):
            return getattr(args, "vlm_image_model", "") == "gemma4_local"
        if mime.startswith("audio"):
            return getattr(args, "alm_model", "") == "gemma4_local"
        return False

    def resolve_prompts(self, uri: str, mime: str, media: MediaContext | None = None) -> PromptContext:
        if not mime.startswith("audio"):
            return super().resolve_prompts(uri, mime, media=media)

        prompts = self.ctx.config.get("prompts", {})
        task = self._resolve_audio_task(required=True)
        char_name, char_prompt = self._get_character_prompt(uri)

        system = self._first_prompt(
            prompts,
            f"gemma4_audio_{task}_system_prompt",
            "gemma4_audio_system_prompt",
            "audio_system_prompt",
        )
        user = self._first_prompt(
            prompts,
            f"gemma4_audio_{task}_prompt",
            "gemma4_audio_prompt",
            "audio_prompt",
        )
        if char_prompt:
            user = char_prompt + user
        return PromptContext(
            system=system,
            user=user,
            character_name=char_name,
            character_prompt=char_prompt,
        )

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        file_path = Path(uri)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        blob = None
        pixels = None
        pair_blob = None
        pair_pixels = None
        extras: Dict[str, Any] = {
            "model_id": self.model_id,
        }
        duration_ms = 0

        if mime.startswith("image"):
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)
            pair_dir = getattr(args, "pair_dir", "")
            if pair_dir:
                pair_path = (Path(pair_dir) / file_path.name).resolve()
                if pair_path.exists():
                    pair_blob, pair_pixels = encode_image_to_blob(str(pair_path), to_rgb=True)
                    extras["pair_uri"] = str(pair_path)
            modality = MediaModality.IMAGE
        elif mime.startswith("video"):
            modality = MediaModality.VIDEO
            duration_ms = self._read_duration_ms(uri)
            self._enforce_duration_limit(duration_ms, kind="video")
        elif mime.startswith("audio"):
            modality = MediaModality.AUDIO
            if not self._model_supports_audio():
                raise RuntimeError(
                    f"GEMMA4_AUDIO_UNSUPPORTED_MODEL: model_id={self.model_id!r}; "
                    "Gemma 4 audio route only supports E2B/E4B variants."
                )
            duration_ms = self._read_duration_ms(uri)
            self._enforce_duration_limit(duration_ms, kind="audio")
            extras["audio_task"] = self._resolve_audio_task(required=True)
        else:
            modality = MediaModality.UNKNOWN

        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=modality,
            file_size=file_size,
            duration_ms=duration_ms,
            blob=blob,
            pixels=pixels,
            pair_blob=pair_blob,
            pair_pixels=pair_pixels,
            extras=extras,
        )

    def get_runtime_backend(self):
        provider_section = self.model_config
        runtime_model_name = (
            getattr(self.ctx.args, "openai_model_name", "")
            or provider_section.get("runtime_model_id", "")
            or self.model_id
        )
        model_section = find_model_config_section(
            self.ctx.config,
            runtime_model_name,
            preferred_sections=tuple(provider_config_sections(self.name)),
        )
        default_temperature = float(model_section.get("temperature", provider_section.get("temperature", 0.0)))
        default_top_p = float(model_section.get("top_p", provider_section.get("top_p", 1.0)))
        default_max_tokens = int(
            model_section.get(
                "out_seq_length",
                model_section.get(
                    "max_new_tokens",
                    provider_section.get("out_seq_length", provider_section.get("max_new_tokens", 2048)),
                ),
            )
        )
        default_model_id = (
            model_section.get("runtime_model_id", "")
            or model_section.get("model_id", "")
            or provider_section.get("runtime_model_id", "")
            or self.model_id
        )
        return resolve_runtime_backend(
            self.ctx.args,
            provider_section,
            arg_prefix="local_runtime",
            shared_prefix="openai",
            default_model_id=default_model_id,
            default_temperature=default_temperature,
            default_top_p=default_top_p,
            default_max_tokens=default_max_tokens,
        )

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        runtime = self.get_runtime_backend()
        if runtime.is_openai:
            backend = OpenAIChatRuntime(runtime)
            result = backend.complete(self._build_runtime_messages(media, prompts))
            normalized = self._normalize_output(media, result)
            return self._build_caption_result(
                media,
                normalized,
                runtime_backend=runtime.mode,
                runtime_model_id=runtime.model_id,
            )

        cached = self._get_or_load_model()
        model = cached["model"]
        processor = cached["processor"]
        torch = cached["torch"]
        device = cached["device"]
        dtype = cached["dtype"]

        chat_template_kwargs: dict[str, Any] = {}
        if media.mime.startswith("video"):
            video_fps = self.model_config.get("video_fps", 1.0)
            if video_fps not in (None, ""):
                chat_template_kwargs["video_fps"] = float(video_fps)
        inputs = prepare_multimodal_inputs(
            processor,
            self._build_messages(media, prompts),
            device=device,
            dtype=dtype,
            chat_template_kwargs=chat_template_kwargs,
        )

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.model_config.get("max_new_tokens", self.model_config.get("out_seq_length", 2048))),
            "do_sample": bool(self.model_config.get("do_sample", False)),
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(self.model_config.get("temperature", 0.2))
            generation_kwargs["top_p"] = float(self.model_config.get("top_p", 0.95))

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        input_len = 0
        input_ids = inputs.get("input_ids")
        if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
            input_len = int(input_ids.shape[1])

        new_tokens = output_ids[0, input_len:] if input_len else output_ids[0]
        response_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        normalized = self._normalize_output(media, response_text)
        return self._build_caption_result(
            media,
            normalized,
            runtime_backend=runtime.mode,
            runtime_model_id=self.model_id,
            runtime_loader=str(cached.get("model_loader") or ""),
        )

    def _resolve_audio_task(self, *, required: bool) -> str:
        raw_value = getattr(self.ctx.args, "audio_task", "")
        if raw_value in (None, ""):
            raw_value = self.model_config.get("audio_task", "")

        normalized = str(raw_value or "").strip().lower()
        if not normalized:
            if required:
                raise RuntimeError(
                    "GEMMA4_AUDIO_TASK_REQUIRED: use --audio_task=asr|ast or set [gemma4_local].audio_task."
                )
            return ""

        task = _AUDIO_TASK_ALIASES.get(normalized, "")
        if not task:
            raise RuntimeError(f"GEMMA4_AUDIO_TASK_INVALID: expected asr|ast, got {normalized!r}")
        return task

    def _build_runtime_messages(self, media: MediaContext, prompts: PromptContext) -> list[dict[str, Any]]:
        if media.mime.startswith("image"):
            if media.blob is None:
                return []
            return build_vision_messages(
                prompts.system,
                prompts.user,
                media.blob,
                pair_blob=media.pair_blob,
            )

        if media.mime.startswith("video"):
            video_base = _base64.b64encode(Path(media.uri).read_bytes()).decode("utf-8")
            return [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": f"data:{media.mime};base64,{video_base}"}},
                        {"type": "text", "text": prompts.user},
                    ],
                },
            ]

        if media.mime.startswith("audio"):
            return [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        self.build_audio_part(str(Path(media.uri).resolve())),
                        self.build_text_part(prompts.user),
                    ],
                },
            ]

        return [
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ]

    def _build_messages(self, media: MediaContext, prompts: PromptContext) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        if media.mime.startswith("image"):
            user_content.append(self.build_image_part(str(Path(media.uri).resolve())))
            pair_uri = media.extras.get("pair_uri")
            if pair_uri:
                user_content.append(self.build_image_part(str(Path(pair_uri).resolve())))
        elif media.mime.startswith("video"):
            user_content.append(self.build_video_part(str(Path(media.uri).resolve())))
        elif media.mime.startswith("audio"):
            user_content.append(self.build_audio_part(str(Path(media.uri).resolve())))
        user_content.append(self.build_text_part(prompts.user))

        messages = [self.build_message("user", user_content)]
        if prompts.system:
            messages.insert(0, self.build_message("system", [self.build_text_part(prompts.system)]))
        return messages

    def _resolve_attention_impl(self, torch_module: Any, device: str, attn_impl: str | None) -> str | None:
        if not _is_cuda_device(device) or attn_impl != "flash_attention_2":
            return attn_impl

        functional = getattr(getattr(torch_module, "nn", None), "functional", None)
        if hasattr(functional, "scaled_dot_product_attention"):
            self.log(
                "Gemma 4 global attention exceeds flash_attention_2 head-dim limits; falling back to sdpa",
                "yellow",
            )
            return "sdpa"

        self.log(
            "Gemma 4 global attention exceeds flash_attention_2 head-dim limits; falling back to eager",
            "yellow",
        )
        return "eager"

    def _load_model(self):
        import transformers
        import torch

        model_id = self.model_id
        if _looks_like_modelopt_nvfp4_repo(model_id):
            raise RuntimeError(
                "GEMMA4_LOCAL_MODEL_LOAD_FAILED: "
                f"model_id={model_id!r} looks like an NVIDIA ModelOpt NVFP4 checkpoint. "
                "The current gemma4_local direct Transformers loader does not support ModelOpt/NVFP4 weights. "
                "Use an OpenAI-compatible vLLM runtime for this model instead of runtime_backend=direct."
            )
        try:
            device, dtype, attn_impl = resolve_device_dtype(
                supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False))
            )
        except TypeError:
            device, dtype, attn_impl = resolve_device_dtype()
        attn_impl = self._resolve_attention_impl(torch, device, attn_impl)
        load_dtype = _resolve_model_load_dtype(device, dtype)
        self.log(
            f"Loading Gemma 4 model: {model_id} (device={device}, dtype={load_dtype}, attn={attn_impl})",
            "blue",
        )

        processor = load_pretrained_component(
            transformers.AutoProcessor,
            model_id,
            console=self.ctx.console,
            component_name="processor",
            trust_remote_code=True,
        )
        self._ensure_processor_chat_template(processor, model_id)

        candidate_classes: list[Any] = []
        for class_name in (
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ):
            cls = getattr(transformers, class_name, None)
            if cls is not None and cls not in candidate_classes:
                candidate_classes.append(cls)

        load_errors: list[str] = []
        for model_cls in candidate_classes:
            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": load_dtype,
            }
            if _is_cuda_device(device):
                load_kwargs["device_map"] = _resolve_pretrained_device_map(device, "auto")
                load_kwargs["attn_implementation"] = attn_impl

            try:
                model = load_pretrained_component(
                    model_cls,
                    model_id,
                    console=self.ctx.console,
                    component_name=f"model via {model_cls.__name__}",
                    **load_kwargs,
                ).eval()
                if not _is_cuda_device(device):
                    model = move_pretrained_component(model, device=device)
                return {
                    "model": model,
                    "processor": processor,
                    "device": device,
                    "dtype": dtype,
                    "torch": torch,
                    "model_loader": model_cls.__name__,
                }
            except Exception as exc:
                load_errors.append(f"{model_cls.__name__}: {exc}")

        detail = "; ".join(load_errors) or "no compatible Transformers auto-model loader found"
        raise RuntimeError(f"GEMMA4_LOCAL_MODEL_LOAD_FAILED: {detail}")

    def _normalize_output(self, media: MediaContext, response_text: str) -> str:
        cleaned = (response_text or "").strip()
        if not media.mime.startswith("audio"):
            return cleaned

        task = media.extras.get("audio_task") or self._resolve_audio_task(required=True)
        srt_payload = extract_code_block_content(cleaned, "srt")
        if srt_payload:
            return srt_payload.strip()
        if task == "ast":
            generic_payload = extract_code_block_content(cleaned)
            if generic_payload:
                return generic_payload.strip()
        return cleaned

    def _maybe_parse_audio_result(self, result: str, media: MediaContext) -> dict[str, Any] | None:
        if not media.mime.startswith("audio"):
            return None

        normalized = (result or "").strip()
        if not normalized:
            return None

        audio_task = str(media.extras.get("audio_task") or self._resolve_audio_task(required=True)).strip().lower()
        if audio_task == "ast":
            return {
                "task_kind": "ast",
                "translation_srt": normalized,
                "caption_extension": ".srt",
                "subtitle_format": "srt",
                "provider": self.name,
            }

        subtitle_format = "srt" if "-->" in normalized else ""
        return {
            "task_kind": "transcribe",
            "transcript": normalized,
            "caption_extension": ".srt" if subtitle_format else ".txt",
            "subtitle_format": subtitle_format,
            "provider": self.name,
        }

    def _maybe_parse_image_result(self, result: str, media: MediaContext) -> dict[str, Any] | None:
        if not media.mime.startswith("image"):
            return None
        if media.pair_pixels is not None:
            return None

        description = str(result or "").strip()
        if not description:
            return None

        structured_payload = self._maybe_load_structured_image_payload(description)
        if structured_payload is not None:
            return structured_payload

        parsed_image = self._parse_freeform_image_caption(description)
        return self._normalize_image_payload(
            {
                "description": parsed_image["description"] or description,
                "scores": parsed_image["scores"],
                "average_score": parsed_image["average_score"],
            }
        )

    def _maybe_load_structured_image_payload(self, text: str) -> dict[str, Any] | None:
        candidates = [text]

        fenced_json = extract_code_block_content(text, "json")
        if fenced_json:
            candidates.append(fenced_json)

        generic_fenced = extract_code_block_content(text)
        if generic_fenced and generic_fenced not in candidates:
            candidates.append(generic_fenced)

        for candidate in candidates:
            payload_text = str(candidate or "").strip()
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and self._looks_like_structured_image_payload(payload):
                return self._normalize_image_payload(payload)
        return None

    def _looks_like_structured_image_payload(self, payload: dict[str, Any]) -> bool:
        return any(key in payload for key in ("description", "scores", "average_score"))

    def _normalize_image_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["description"] = str(payload.get("description", "") or "").strip()
        normalized["scores"] = self._normalize_image_scores(payload.get("scores"))
        try:
            average_score = float(payload.get("average_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            average_score = 0.0
        normalized["average_score"] = max(0.0, average_score)
        normalized["caption_extension"] = ".txt"
        normalized["provider"] = self.name
        return normalized

    def _normalize_image_scores(self, raw_scores: Any) -> dict[str, int | float]:
        if not isinstance(raw_scores, dict):
            return {}

        canonical_scores: dict[str, int | float] = {}
        extra_scores: dict[str, int | float] = {}
        for raw_label, raw_value in raw_scores.items():
            label = self._normalize_image_score_label(raw_label)
            score = self._normalize_image_score_value(label, raw_value)
            if label is None or score is None:
                continue
            if label in _IMAGE_SCORE_LIMITS:
                canonical_scores[label] = score
            else:
                extra_scores[label] = score

        ordered_scores: dict[str, int | float] = {}
        for label in _IMAGE_CANONICAL_SCORE_ORDER:
            if label in canonical_scores:
                ordered_scores[label] = canonical_scores[label]
        for label, score in extra_scores.items():
            ordered_scores[label] = score
        return ordered_scores

    def _normalize_image_score_label(self, label: Any) -> str | None:
        cleaned = re.sub(r"\s+", " ", str(label or "")).strip().rstrip(":").strip()
        if not cleaned:
            return None

        canonical = _IMAGE_SCORE_LABEL_ALIASES.get(_image_score_lookup_key(cleaned))
        if canonical:
            return canonical

        lowered = cleaned.casefold()
        if lowered.startswith("level of s") and lowered.endswith("y"):
            return "Level of S*e*x*y"

        return cleaned

    def _normalize_image_score_value(self, label: str | None, raw_value: Any) -> int | float | None:
        if label is None or isinstance(raw_value, bool):
            return None
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            return None

        max_score = _IMAGE_SCORE_LIMITS.get(label)
        if max_score is not None:
            numeric_value = min(max(numeric_value, 0.0), float(max_score))
        return int(numeric_value) if numeric_value.is_integer() else numeric_value

    def _parse_freeform_image_caption(self, text: str) -> dict[str, Any]:
        lines = text.splitlines()
        description_lines = list(lines)
        score_heading_index: int | None = None
        scores: dict[str, int | float] = {}
        average_score = 0.0

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if score_heading_index is None and _IMAGE_SCORE_HEADING_RE.match(line):
                score_heading_index = index
                continue

            average_match = _IMAGE_AVERAGE_SCORE_RE.search(line)
            if average_match:
                try:
                    average_score = float(average_match.group(1))
                except ValueError:
                    average_score = 0.0
                continue

            if _IMAGE_TOTAL_SCORE_RE.search(line):
                continue

            label = ""
            score_text = ""

            bold_match = _IMAGE_BOLD_SCORE_LINE_RE.match(raw_line)
            if bold_match:
                label = bold_match.group(1).strip().rstrip(":").strip()
                score_text = bold_match.group(2)
            else:
                plain_match = _IMAGE_PLAIN_SCORE_LINE_RE.match(raw_line)
                if plain_match:
                    label = plain_match.group(1).strip()
                    score_text = plain_match.group(2)

            if label and score_text:
                try:
                    numeric_score = float(score_text)
                except ValueError:
                    continue
                scores[label] = int(numeric_score) if numeric_score.is_integer() else numeric_score

        if score_heading_index is not None:
            description_lines = lines[:score_heading_index]

        while description_lines and not description_lines[0].strip():
            description_lines.pop(0)
        while description_lines and not description_lines[-1].strip():
            description_lines.pop()
        if description_lines and _IMAGE_DESCRIPTION_HEADING_RE.match(description_lines[0].strip()):
            description_lines.pop(0)
        while description_lines and not description_lines[0].strip():
            description_lines.pop(0)

        description = "\n".join(description_lines).strip()
        description = re.sub(r"\n{3,}", "\n\n", description)

        return {
            "description": description,
            "scores": scores,
            "average_score": average_score,
        }

    def _build_caption_result(self, media: MediaContext, normalized: str, **metadata_extra: Any) -> CaptionResult:
        parsed = self._maybe_parse_audio_result(normalized, media)
        if parsed is not None:
            return CaptionResult(
                raw=normalized,
                parsed=parsed,
                metadata=self._build_metadata(**metadata_extra),
            )

        parsed = self._maybe_parse_image_result(normalized, media)
        if parsed is None:
            return CaptionResult(
                raw=normalized,
                metadata=self._build_metadata(**metadata_extra),
            )

        return CaptionResult(
            raw=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed,
            metadata=self._build_metadata(structured=True, **metadata_extra),
        )

    def _build_metadata(self, **extra: Any) -> dict[str, Any]:
        metadata = {key: value for key, value in {
            "provider": self.name,
            "model_id": self.model_id,
        }.items() if value not in (None, "")}
        metadata.update({key: value for key, value in extra.items() if value not in (None, "")})
        return metadata

    def _read_duration_ms(self, uri: str) -> int:
        try:
            return int(get_video_duration(uri) or 0)
        except Exception:
            return 0

    def _enforce_duration_limit(self, duration_ms: int, *, kind: str) -> None:
        if duration_ms <= 0:
            return

        if kind == "audio":
            limit_seconds = float(self.model_config.get("audio_max_seconds", 30) or 30)
        else:
            limit_seconds = float(self.model_config.get("video_max_seconds", 60) or 60)

        limit_ms = int(limit_seconds * 1000)
        if duration_ms <= limit_ms:
            return

        raise RuntimeError(
            f"GEMMA4_{kind.upper()}_TOO_LONG: duration={duration_ms / 1000:.2f}s limit={limit_seconds:.2f}s; "
            "Gemma 4 local does not auto-segment or truncate media."
        )

    @staticmethod
    def _first_prompt(prompts: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = prompts.get(key)
            if value is not None:
                return str(value)
        return ""
