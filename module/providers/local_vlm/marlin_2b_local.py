"""Marlin 2B local video provider."""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.capabilities import ProviderCapabilities
from module.providers.local_vlm_base import LocalVLMProvider
from module.providers.registry import register_provider
from utils.stream_util import get_video_duration


def _is_cuda_device(device: Any) -> bool:
    return str(device or "").startswith("cuda")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


def _strip_leading_think(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*<think>\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


@register_provider("marlin_2b_local")
class Marlin2BLocalProvider(LocalVLMProvider):
    """Local provider for NemoStation/Marlin-2B dense video captioning."""

    default_model_id = "NemoStation/Marlin-2B"
    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_images=False,
        supports_video=True,
    )

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "marlin_2b_local" and mime.startswith("video")

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        media = super().prepare_media(uri, mime, args)
        if not mime.startswith("video"):
            return media

        duration_ms = self._read_duration_ms(uri)
        self._enforce_video_duration_limit(duration_ms)
        return replace(media, duration_ms=duration_ms)

    def _read_duration_ms(self, uri: str) -> int:
        try:
            return int(float(get_video_duration(uri)))
        except Exception:
            return 0

    def _video_max_seconds(self) -> float:
        return _coerce_float(self.model_config.get("video_max_seconds"), 120.0)

    def _enforce_video_duration_limit(self, duration_ms: int) -> None:
        max_seconds = self._video_max_seconds()
        if duration_ms <= 0 or max_seconds <= 0:
            return
        if duration_ms > int(max_seconds * 1000):
            raise RuntimeError(
                "MARLIN2B_VIDEO_TOO_LONG: "
                f"duration={duration_ms / 1000:.1f}s exceeds video_max_seconds={max_seconds:.1f}. "
                "Lower --segment_time or raise [marlin_2b_local].video_max_seconds if you accept frame capping."
            )

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM

        from utils.transformer_loader import load_pretrained_component, move_pretrained_component, resolve_device_dtype

        model_id = self.model_id
        device, runtime_dtype, _ = resolve_device_dtype()
        load_dtype = self._select_dtype(device=device, dtype=runtime_dtype, torch_module=torch)
        self.log(f"Loading Marlin 2B model: {model_id} (device={device}, dtype={load_dtype})", "blue")

        load_errors: list[str] = []
        model = None
        for dtype_kwarg in ("dtype", "torch_dtype"):
            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                dtype_kwarg: load_dtype,
            }
            if _is_cuda_device(device):
                load_kwargs["device_map"] = {"": str(device)}

            try:
                model = load_pretrained_component(
                    AutoModelForCausalLM,
                    model_id,
                    console=self.ctx.console,
                    component_name=f"model via AutoModelForCausalLM/{dtype_kwarg}",
                    **load_kwargs,
                ).eval()
                break
            except TypeError as exc:
                load_errors.append(f"{dtype_kwarg}: {exc}")
                if dtype_kwarg == "torch_dtype":
                    raise

        if model is None:
            detail = "; ".join(load_errors) or "unknown loader failure"
            raise RuntimeError(f"MARLIN2B_LOCAL_MODEL_LOAD_FAILED: {detail}")

        if not _is_cuda_device(device):
            model = move_pretrained_component(model, device=device, dtype=load_dtype)

        if _coerce_bool(self.model_config.get("compile"), False):
            compile_model = getattr(model, "compile", None)
            if callable(compile_model):
                compiled = compile_model()
                if compiled is not None:
                    model = compiled

        return {
            "model": model,
            "device": device,
            "dtype": load_dtype,
            "torch": torch,
            "model_loader": "AutoModelForCausalLM",
        }

    def _select_dtype(self, *, device: str, dtype: Any, torch_module: Any):
        requested_dtype = self.model_config.get("dtype")
        if isinstance(requested_dtype, str):
            normalized = requested_dtype.strip().lower()
            dtype_aliases = {
                "float16": torch_module.float16,
                "fp16": torch_module.float16,
                "half": torch_module.float16,
                "bfloat16": torch_module.bfloat16,
                "bf16": torch_module.bfloat16,
                "float32": torch_module.float32,
                "fp32": torch_module.float32,
            }
            if normalized in dtype_aliases:
                return dtype_aliases[normalized]

        if _is_cuda_device(device):
            return torch_module.bfloat16
        return dtype

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        start_time = time.time()

        if self.get_runtime_backend().is_openai:
            result = self.attempt_via_openai_backend(
                media,
                prompts,
                user_prompt=self._runtime_prompt(prompts),
            )
            output = self._parse_runtime_output(result.raw)
            if self._resolve_task() == "find":
                payload = self._normalize_find_payload(output, self._resolve_find_event(prompts))
            else:
                payload = self._normalize_caption_payload(output)
            return CaptionResult(
                raw=str(payload.get("description") or payload.get("caption") or payload.get("raw") or "").strip(),
                parsed=payload,
                metadata={
                    **result.metadata,
                    "provider": self.name,
                    "structured": True,
                },
            )

        cached = self._get_or_load_model()
        model = cached["model"]
        torch = cached["torch"]

        task = self._resolve_task()
        video_path = str(Path(media.uri).resolve())

        with torch.inference_mode():
            if task == "find":
                event = self._resolve_find_event(prompts)
                output = self._call_find(model, video_path, event)
                payload = self._normalize_find_payload(output, event)
            else:
                output = self._call_caption(model, video_path, prompts)
                payload = self._normalize_caption_payload(output)

        elapsed_time = time.time() - start_time
        self.log(f"Marlin 2B video analysis took: {elapsed_time:.2f} seconds", "blue")

        description = str(payload.get("description") or payload.get("caption") or payload.get("raw") or "").strip()
        if description:
            self.ctx.console.print(description)

        return CaptionResult(
            raw=description,
            parsed=payload,
            metadata={
                "provider": self.name,
                "structured": True,
                "runtime_model_id": self.model_id,
                "runtime_loader": str(cached.get("model_loader") or ""),
            },
        )

    def _resolve_task(self) -> str:
        raw_task = str(self.model_config.get("task", self.model_config.get("mode", "caption")) or "").strip().lower()
        if raw_task in {"find", "ground", "grounding", "temporal_grounding"}:
            return "find"
        return "caption"

    def _runtime_prompt(self, prompts: PromptContext) -> str:
        override = self._caption_prompt_override(prompts)
        return override or prompts.user

    def _caption_prompt_override(self, prompts: PromptContext) -> str | None:
        configured_prompt = str(self.model_config.get("prompt") or "").strip()
        if configured_prompt:
            return configured_prompt
        if _coerce_bool(self.model_config.get("use_resolved_prompt"), False):
            return prompts.user
        return None

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.model_config.get("max_new_tokens", 2048)),
            "do_sample": _coerce_bool(self.model_config.get("do_sample"), False),
            "temperature": _coerce_float(self.model_config.get("temperature"), 1.0),
            "top_p": _coerce_float(self.model_config.get("top_p"), 1.0),
        }
        return kwargs

    def _parse_runtime_output(self, output: Any) -> Any:
        if isinstance(output, dict):
            return output
        cleaned = _strip_leading_think(str(output or ""))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned

    def _call_caption(self, model: Any, video_path: str, prompts: PromptContext) -> Any:
        caption = getattr(model, "caption", None)
        if not callable(caption):
            raise RuntimeError(
                "MARLIN2B_CAPTION_METHOD_MISSING: load NemoStation/Marlin-2B with trust_remote_code=True "
                "or use runtime_backend=openai."
            )

        kwargs = self._generation_kwargs()
        prompt_override = self._caption_prompt_override(prompts)
        if prompt_override is not None:
            kwargs["prompt"] = prompt_override
        return caption(video_path, **kwargs)

    def _resolve_find_event(self, prompts: PromptContext) -> str:
        event = str(self.model_config.get("find_event") or "").strip()
        if event:
            return event
        event = str(prompts.user or "").strip()
        if not event:
            raise RuntimeError("MARLIN2B_FIND_EVENT_REQUIRED: set [marlin_2b_local].find_event or provide a prompt.")
        return event

    def _call_find(self, model: Any, video_path: str, event: str) -> Any:
        find = getattr(model, "find", None)
        if not callable(find):
            raise RuntimeError(
                "MARLIN2B_FIND_METHOD_MISSING: load NemoStation/Marlin-2B with trust_remote_code=True "
                "or use runtime_backend=openai."
            )
        return find(video_path, event=event, **self._generation_kwargs())

    def _normalize_caption_payload(self, output: Any) -> dict[str, Any]:
        if isinstance(output, dict):
            payload = dict(output)
        else:
            payload = {"caption": str(output or "")}

        caption = _strip_leading_think(str(payload.get("caption") or ""))
        scene = _strip_leading_think(str(payload.get("scene") or ""))
        events = self._normalize_events(payload.get("events"))
        description = caption or self._format_scene_events(scene, events)
        if not description:
            description = _strip_leading_think(str(output or ""))

        payload["caption"] = caption or description
        payload["scene"] = scene
        payload["events"] = events
        payload["description"] = description
        payload["task_kind"] = "caption"
        payload["caption_extension"] = ".txt"
        payload["provider"] = self.name
        return payload

    def _normalize_find_payload(self, output: Any, event: str) -> dict[str, Any]:
        payload = dict(output) if isinstance(output, dict) else {"raw": str(output or "")}
        raw = _strip_leading_think(str(payload.get("raw") or payload.get("description") or ""))
        span = self._normalize_span(payload.get("span"))
        if not raw and span is not None:
            raw = f"From {span[0]:.1f} to {span[1]:.1f}."

        return {
            "task_kind": "temporal_grounding",
            "event": event,
            "description": raw,
            "raw": raw,
            "span": span,
            "format_ok": bool(payload.get("format_ok", span is not None)),
            "caption_extension": ".txt",
            "provider": self.name,
        }

    @staticmethod
    def _normalize_span(span: Any) -> list[float] | None:
        if span is None:
            return None
        if isinstance(span, (list, tuple)) and len(span) == 2:
            try:
                return [float(span[0]), float(span[1])]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _normalize_events(events: Any) -> list[dict[str, Any]]:
        if not isinstance(events, list):
            return []

        normalized: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            item = dict(event)
            for key in ("start", "end"):
                if key not in item:
                    continue
                try:
                    item[key] = float(item[key])
                except (TypeError, ValueError):
                    item.pop(key, None)
            if "description" in item:
                item["description"] = str(item["description"]).strip()
            normalized.append(item)
        return normalized

    @staticmethod
    def _format_scene_events(scene: str, events: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        if scene:
            lines.append(f"Scene: {scene}")
        if events:
            lines.append("Events:")
            for event in events:
                description = str(event.get("description") or "").strip()
                if not description:
                    continue
                start = event.get("start")
                end = event.get("end")
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    lines.append(f"<{start:.1f} - {end:.1f}> {description}")
                else:
                    lines.append(description)
        return "\n".join(lines).strip()
