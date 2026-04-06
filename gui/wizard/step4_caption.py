"""步骤 4: 字幕生成 - 对应 run.ps1 (captioner.py)"""

import asyncio
import re
from typing import TYPE_CHECKING

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.advanced_inputs import editable_slider, toggle_switch, styled_select, styled_input
from gui.utils.i18n import t
from gui.utils.toml_helpers import assess_current_model_fit as assess_current_model_fit_from_toml
from gui.utils.toml_helpers import load_current_route_model_ids as load_current_route_model_ids_from_toml
from module.providers.catalog import route_choices, route_requires_remote_config

if TYPE_CHECKING:
    from components.execution_panel import ExecutionPanel
    from components.model_config_panel import ModelConfigPanel


def _load_gemini_task_names() -> list[str]:
    import toml
    try:
        cfg = toml.load(Path(__file__).resolve().parent.parent.parent / "config" / "prompts.toml")
        tasks = cfg.get("prompts", {}).get("task", {})
        return [k for k, v in tasks.items() if isinstance(v, str)]
    except Exception:
        return []


def _load_current_route_model_ids() -> dict[str, str]:
    """从 model.toml 读取当前 route -> model_id 映射。"""
    route_names = tuple(
        route_name
        for route_name in (
            *route_choices("ocr_model", include_empty=False),
            *route_choices("vlm_image_model", include_empty=False),
            *route_choices("alm_model", include_empty=False),
        )
        if route_name
    )
    return load_current_route_model_ids_from_toml(route_names)


def get_cached_gpu_probe(*, refresh: bool = False):
    from module.gpu_profile import get_cached_gpu_probe as _get_cached_gpu_probe

    return _get_cached_gpu_probe(refresh=refresh)


def format_gpu_summary(probe) -> str:
    from module.gpu_profile import format_gpu_summary as _format_gpu_summary

    return _format_gpu_summary(probe)


def format_gpu_device_lines(probe) -> tuple[str, ...]:
    from module.gpu_profile import format_gpu_device_lines as _format_gpu_device_lines

    return _format_gpu_device_lines(probe)


def assess_current_model_fit(route_name: str, *, current_model_id: str, probe=None):
    return assess_current_model_fit_from_toml(route_name, current_model_id=current_model_id, probe=probe)


def _load_execution_panel_cls():
    from components.execution_panel import ExecutionPanel

    return ExecutionPanel


def _load_model_config_panel_cls():
    from components.model_config_panel import ModelConfigPanel

    return ModelConfigPanel

class CaptionStep:
    QUANTIZED_RUNTIME_EXTRA = "quantized-runtime"
    _BITSANDBYTES_REPO_PATTERN = re.compile(
        r"(?:^|[-_./])(?:nf4|fp8-block|bnb|bitsandbytes|4bit|8bit|int4|int8)(?:$|[-_./])"
    )

    """字幕生成页面"""

    # API 配置 - 模型列表（与 config.toml / captioner.py 后端保持一致）
    API_CONFIGS = {
        "Gemini": {
            "key_name": "gemini_api_key",
            "models": [
                "gemini-3.1-pro-preview",
                "gemini-3-flash-preview",
                "gemini-3.1-flash-lite-preview",
                "gemini-3.1-flash-image-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
            ],
            "default_model": "gemini-2.5-flash",
            "supports_video": True,
            "supports_task": True,
        },
        "Mistral": {
            "key_name": "mistral_api_key",
            "models": [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
                "mistral-ocr-latest",
                "pixtral-large-latest",
                "pixtral-12b-latest",
            ],
            "default_model": "mistral-large-latest",
            "supports_video": False,
            "supports_task": False,
        },
        "Step": {
            "key_name": "step_api_key",
            "models": [
                "step-3.5-flash",
                "step-3-vl",
                "step-3-vl-mini",
                "step-1o-turbo-vision",
                "step-2o-vision",
                "step-r1-v-mini",
            ],
            "default_model": "step-3-vl-mini",
            "supports_video": True,
            "supports_task": False,
        },
        "Qwen": {
            "key_name": "qwenVL_api_key",
            "models": [
                "qwen3.5-plus",
                "qwen3.5-flash",
                "qwen3-vl-plus",
                "qwen3-vl-plus-latest",
                "qwen3-vl-flash",
                "qwen3-vl-flash-latest",
                "qwen-vl-max",
                "qwen-vl-ocr",
                "qwen-vl-ocr-latest",
            ],
            "default_model": "qwen3.5-plus",
            "supports_video": True,
            "supports_task": False,
        },
        "Kimi": {
            "key_name": "kimi_api_key",
            "models": [
                "kimi-k2.5",
                "kimi-k2",
                "kimi-k2-turbo-preview",
                "kimi-k2-thinking",
                "kimi-k2-thinking-turbo",
            ],
            "default_model": "kimi-k2.5",
            "supports_video": True,
            "supports_task": False,
        },
        "Kimi-Code": {
            "key_name": "kimi_code_api_key",
            "models": [
                "k2p5",
            ],
            "default_model": "k2p5",
            "supports_video": True,
            "supports_task": False,
        },
        "MiniMax": {
            "key_name": "minimax_api_key",
            "models": [
                "MiniMax-M2.7",
                "MiniMax-M2.7-highspeed",
                "MiniMax-M2.5",
                "MiniMax-M2.5-highspeed",
                "MiniMax-M2.1",
                "MiniMax-M2.1-highspeed",
                "MiniMax-M2",
            ],
            "default_model": "MiniMax-M2.7",
            "supports_video": True,
            "supports_task": False,
        },
        "MiniMax-Code": {
            "key_name": "minimax_code_api_key",
            "models": [
                "MiniMax-M2.7",
                "MiniMax-M2.7-highspeed",
                "MiniMax-M2.5",
                "MiniMax-M2.5-highspeed",
                "MiniMax-M2.1",
            ],
            "default_model": "MiniMax-M2.7",
            "supports_video": True,
            "supports_task": False,
        },
        "GLM": {
            "key_name": "glm_api_key",
            "models": [
                "glm-5v-turbo",
                "glm-5",
                "glm-4.5v",
                "glm-4.1v",
                "glm-4v-plus",
            ],
            "default_model": "glm-5v-turbo",
            "supports_video": True,
            "supports_task": False,
        },
        "Ark": {
            "key_name": "ark_api_key",
            "models": [
                "doubao-seed-2-0-pro-260215",
                "doubao-1.5-thinking-vision-pro-250428",
                "doubao-1.5-vision-pro-250328",
                "doubao-1.5-vision-lite-250315",
                "doubao-vision-pro-32k-241028",
                "doubao-vision-lite-32k-241015",
            ],
            "default_model": "doubao-1.5-vision-pro-250328",
            "supports_video": True,
            "supports_task": False,
            "custom_model_input": True,
        },
        "OpenAI-Compatible": {
            "key_name": "openai_api_key",
            "models": [],
            "default_model": "",
            "supports_video": True,
            "supports_task": False,
            "custom_model_input": True,
            "is_openai_compatible": True,
        },
    }

    MODES = ["long", "short", "all"]

    OCR_MODELS = list(route_choices("ocr_model"))

    # OCR 榜单排名 + model_id 显示标签
    _OCR_RANK_SCORES = {
        "chandra_ocr": ("#1", "85.9"),
        "dots_ocr": ("#2", "83.9"),
        "lighton_ocr": ("#3", "83.2"),
        "olmocr": ("#6", "82.4"),
        "paddle_ocr": ("#7", "80.0"),
        "qianfan_ocr": ("#8", "79.8"),
        "deepseek_ocr": ("#10", "76.3"),
        "glm_ocr": ("#14", "75.2"),
        "firered_ocr": ("#15", "70.2"),
        "nanonets_ocr": ("#16", "69.5"),
    }

    @classmethod
    def _build_ocr_labels(cls) -> dict[str, str]:
        labels: dict[str, str] = {}
        for m in cls.OCR_MODELS:
            if not m:
                labels[m] = m
                continue
            if m in cls._OCR_RANK_SCORES:
                rank, score = cls._OCR_RANK_SCORES[m]
                labels[m] = f"{rank} {m} ({score})"
            else:
                labels[m] = m
        return labels

    @classmethod
    def _build_route_labels(cls, models: list[str]) -> dict[str, str]:
        return {m: m for m in models}

    VLM_MODELS = list(route_choices("vlm_image_model"))

    ALM_MODELS = list(route_choices("alm_model"))
    ALM_LANGUAGE_OPTIONS = {
        "cohere_transcribe_local": {
            "en": "English — English",
            "de": "Deutsch — German",
            "fr": "Français — French",
            "it": "Italiano — Italian",
            "es": "Español — Spanish",
            "pt": "Português — Portuguese",
            "el": "Ελληνικά — Greek",
            "nl": "Nederlands — Dutch",
            "pl": "Polski — Polish",
            "ar": "العربية — Arabic",
            "vi": "Tiếng Việt — Vietnamese",
            "zh": "中文 — Chinese",
            "ja": "日本語 — Japanese",
            "ko": "한국어 — Korean",
        }
    }

    OCR_EXTRA_MAP = {
        "paddle_ocr": "paddleocr",
        "deepseek_ocr": "deepseek-ocr",
        "logics_ocr": "logics-ocr",
        "lighton_ocr": "lighton-ocr",
        "dots_ocr": "dots-ocr",
        "qianfan_ocr": "qianfan-ocr",
        "olmocr": "olmocr",
        "hunyuan_ocr": "hunyuan-ocr",
        "moondream": "moondream",
        "glm_ocr": "glm-ocr",
        "nanonets_ocr": "nanonets-ocr",
        "firered_ocr": "firered-ocr",
        "chandra_ocr": "chandra-ocr",
    }

    VLM_EXTRA_MAP = {
        "moondream": "moondream",
        "qwen_vl_local": "qwen-vl-local",
        "step_vl_local": "step-vl-local",
        "penguin_vl_local": "penguin-vl-local",
        "reka_edge_local": "reka-edge-local",
        "gemma4_local": "gemma4-local",
        "lfm_vl_local": "lfm-vl-local",
    }

    ALM_EXTRA_MAP = {
        "gemma4_local": "gemma4-local",
        "music_flamingo_local": "music-flamingo-local",
        "eureka_audio_local": "eureka-audio-local",
        "acestep_transcriber_local": "acestep-transcriber-local",
        "cohere_transcribe_local": "cohere-transcribe-local",
    }

    ALM_AUDIO_TASK_OPTIONS = {
        "gemma4_local": {
            "asr": "ASR",
            "ast": "AST",
        }
    }

    def __init__(self):
        self.config: Dict[str, Any] = {
            "mode": "long",
            "wait_time": 1,
            "max_retries": 100,
            "segment_time": 600,
            "segment_time_explicit": False,
            "audio_task": "asr",
            "tags_highlightrate": 0.38,
            "dir_name": False,
            "not_clip_with_caption": True,
            "document_image": True,
            "mistral_ocr_mode": False,
        }
        self.panel: "ExecutionPanel | None" = None
        self.api_keys = {}
        self._syncing_segment_time = False
        self.gpu_probe = None
        self._gpu_probe_scheduled = False
        self._execution_panel_container = None

    @staticmethod
    def _has_text(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    def _local_model_fit_header(self) -> str:
        if self.gpu_probe is None:
            return f"{t('gpu')}: {t('checking')}"
        return format_gpu_summary(self.gpu_probe)

    def _gpu_detail_lines(self) -> tuple[str, ...]:
        if self.gpu_probe is None:
            return ()
        return format_gpu_device_lines(self.gpu_probe)

    def _render_gpu_details_toggle(self) -> None:
        lines = self._gpu_detail_lines()
        if not lines:
            return

        details_open = {"value": False}

        with ui.row().classes("w-full items-center gap-2 q-mt-xs"):
            ui.icon("dns", size="16px").style(f"color: {COLORS['info']};")
            ui.label(f"{t('detected_gpus')} ({len(lines)})").classes("text-caption").style(
                "color: var(--color-text-secondary);"
            )

            def _toggle_gpu_details() -> None:
                details_open["value"] = not details_open["value"]
                details_container.set_visibility(details_open["value"])

            ui.button(t("toggle"), on_click=_toggle_gpu_details, icon="unfold_more").props('flat dense type="button"')

        details_container = ui.column().classes("w-full gap-1 q-mt-xs")
        details_container.set_visibility(False)
        with details_container:
            for line in lines:
                ui.label(line).classes("text-caption").style("color: var(--color-text-secondary);")

    def _build_local_model_fit_entries(self) -> tuple[dict[str, str], ...]:
        if self.gpu_probe is None:
            return ()
        model_id_map = _load_current_route_model_ids()
        route_specs = (
            ("OCR", "ocr_model", getattr(getattr(self, "ocr_model", None), "value", "") or ""),
            ("VLM", "vlm_image_model", getattr(getattr(self, "vlm_image_model", None), "value", "") or ""),
            ("ALM", "alm_model", self._current_alm_model()),
        )

        entries: list[dict[str, str]] = []
        for family, route_key, route_name in route_specs:
            if not self._has_text(route_name):
                continue
            if route_requires_remote_config(route_key, route_name):
                continue

            current_model_id = model_id_map.get(route_name, "")
            assessment = assess_current_model_fit(
                route_name,
                current_model_id=current_model_id,
                probe=self.gpu_probe,
            )
            if assessment is None:
                continue

            entries.append(
                {
                    "family": family,
                    "route_name": route_name,
                    "current_model_id": assessment.model_id,
                    "status": assessment.status,
                    "status_label": assessment.status_label,
                    "source": assessment.source,
                }
            )

        return tuple(entries)

    def _handle_model_config_saved(self) -> None:
        for attr_name, options in (
            ("ocr_model", self._build_ocr_labels()),
            ("vlm_image_model", self._build_route_labels(self.VLM_MODELS)),
            ("alm_model", self._build_route_labels(self.ALM_MODELS)),
        ):
            control = getattr(self, attr_name, None)
            if control is not None:
                control.set_options(options, value=getattr(control, "value", None))
        self._refresh_local_model_fit_summary()

    async def _load_gpu_probe_async(self) -> None:
        try:
            probe = await asyncio.to_thread(get_cached_gpu_probe, refresh=True)
        except Exception:
            return

        self.gpu_probe = probe
        self._refresh_local_model_fit_summary()

    def _refresh_local_model_fit_summary(self) -> None:
        container = getattr(self, "_local_model_fit_container", None)
        if container is None:
            return

        container.clear()
        entries = self._build_local_model_fit_entries()

        with container:
            warning_entries = [entry for entry in entries if entry["status"] == "warning"]
            unknown_entries = [entry for entry in entries if entry["status"] == "unknown"]
            card_style = (
                """
                background: rgba(245, 158, 11, 0.10);
                border-radius: 10px;
                border: 1px solid rgba(245, 158, 11, 0.20);
                box-shadow: none;
                """
                if warning_entries
                else """
                background: rgba(16, 185, 129, 0.08);
                border-radius: 10px;
                border: 1px solid rgba(16, 185, 129, 0.18);
                box-shadow: none;
                """
            )

            with ui.card().classes("w-full q-pa-sm").style(card_style):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("memory", size="18px").style(f"color: {COLORS['success']};")
                        ui.label(t("local_model_fit")).classes("text-body2 text-weight-bold").style(
                            "color: var(--color-text);"
                        )
                    ui.label(self._local_model_fit_header()).classes("text-caption").style(
                        "color: var(--color-text-secondary);"
                    )

                self._render_gpu_details_toggle()

                if self.gpu_probe is None:
                    ui.label(t("detecting_gpu")).classes(
                        "text-caption q-mt-sm"
                    ).style("color: var(--color-text-secondary);")
                    return

                if not entries:
                    ui.label(t("select_routes_check")).classes(
                        "text-caption q-mt-sm"
                    ).style("color: var(--color-text-secondary);")
                    return

                if warning_entries:
                    ui.label(t("models_exceed_vram")).classes("text-caption q-mt-sm").style(
                        "color: var(--color-text-secondary);"
                    )
                    for entry in warning_entries:
                        with ui.column().classes("w-full gap-0 q-mt-sm").style(
                            """
                            background: rgba(255, 255, 255, 0.55);
                            border-radius: 8px;
                            padding: 8px 10px;
                            """
                        ):
                            ui.label(f"{entry['family']} · {entry['current_model_id']}").classes(
                                "text-caption text-weight-bold"
                            ).style("color: var(--color-text-secondary);")
                            ui.label(entry["status_label"]).classes("text-body2").style("color: var(--color-text);")
                    return

                if unknown_entries:
                    ui.label(t("missing_metadata")).classes(
                        "text-caption q-mt-sm"
                    ).style("color: var(--color-text-secondary);")
                    for entry in unknown_entries:
                        ui.label(f"{entry['family']} · {entry['current_model_id']}").classes("text-caption").style(
                            "color: var(--color-text-secondary);"
                        )
                    return

                ui.label(t("models_fit_vram")).classes("text-caption q-mt-sm").style(
                    "color: var(--color-text-secondary);"
                )

    def _current_alm_model(self) -> str:
        alm_model = getattr(self, "alm_model", None)
        return getattr(alm_model, "value", "") or ""

    def _default_segment_time(self, alm_model: Optional[str] = None) -> int:
        model_name = alm_model if alm_model is not None else self._current_alm_model()
        if model_name == "music_flamingo_local":
            return 1200
        return 600

    def _alm_requires_language(self, alm_model: Optional[str] = None) -> bool:
        model_name = alm_model if alm_model is not None else self._current_alm_model()
        return model_name in self.ALM_LANGUAGE_OPTIONS

    def _alm_language_options(self, alm_model: Optional[str] = None) -> dict[str, str]:
        model_name = alm_model if alm_model is not None else self._current_alm_model()
        return dict(self.ALM_LANGUAGE_OPTIONS.get(model_name, {}))

    def _alm_requires_audio_task(self, alm_model: Optional[str] = None) -> bool:
        model_name = alm_model if alm_model is not None else self._current_alm_model()
        return model_name in self.ALM_AUDIO_TASK_OPTIONS

    def _alm_audio_task_options(self, alm_model: Optional[str] = None) -> dict[str, str]:
        model_name = alm_model if alm_model is not None else self._current_alm_model()
        return dict(self.ALM_AUDIO_TASK_OPTIONS.get(model_name, {}))

    def _initial_alm_audio_task_value(self, alm_model: Optional[str] = None) -> Optional[str]:
        options = self._alm_audio_task_options(alm_model)
        if not options:
            return None
        preferred_value = self.config.get("audio_task", "asr")
        if preferred_value in options:
            return preferred_value
        return next(iter(options), None)

    def _sync_alm_language_options(self, alm_model: Optional[str] = None) -> None:
        alm_language = getattr(self, "alm_language", None)
        if alm_language is None:
            return

        options = self._alm_language_options(alm_model)
        current_value = getattr(alm_language, "value", None)
        next_value = current_value if current_value in options else None
        alm_language.set_options(options, value=next_value)

        # Hide the container when no language options are available
        container = getattr(self, "_alm_language_container", None)
        if container is not None:
            container.set_visibility(bool(options))

    def _handle_audio_task_change(self, selected_task: Optional[str]) -> None:
        if self._has_text(selected_task):
            self.config["audio_task"] = selected_task

    def _sync_alm_audio_task_options(self, alm_model: Optional[str] = None) -> None:
        audio_task = getattr(self, "alm_audio_task", None)
        if audio_task is None:
            return

        options = self._alm_audio_task_options(alm_model)
        current_value = getattr(audio_task, "value", None)
        preferred_value = self.config.get("audio_task", "asr")
        if current_value in options:
            next_value = current_value
        elif preferred_value in options:
            next_value = preferred_value
        else:
            next_value = next(iter(options), None)

        audio_task.set_options(options, value=next_value)
        if next_value is not None:
            self.config["audio_task"] = next_value

        container = getattr(self, "_alm_audio_task_container", None)
        if container is not None:
            container.set_visibility(bool(options))

    @staticmethod
    def _format_args_for_log(args: list[str]) -> str:
        redacted_args: list[str] = []
        for arg in args:
            if not isinstance(arg, str):
                arg = str(arg)
            if arg.startswith("--") and "=" in arg:
                key, _value = arg.split("=", 1)
                if "api_key" in key.lower():
                    redacted_args.append(f"{key}=***")
                    continue
            redacted_args.append(arg)
        return repr(redacted_args)

    def _handle_segment_time_change(self, value: Any) -> None:
        self.config["segment_time"] = int(value)
        if not self._syncing_segment_time:
            self.config["segment_time_explicit"] = True

    def _sync_segment_time_default(self, alm_model: Optional[str] = None) -> None:
        if self.config.get("segment_time_explicit"):
            return

        default_value = self._default_segment_time(alm_model)
        self.config["segment_time"] = default_value

        slider = getattr(self, "segment_time_slider", None)
        if slider is not None and getattr(slider, "value", None) != default_value:
            self._syncing_segment_time = True
            try:
                slider.set_value(default_value)
            finally:
                self._syncing_segment_time = False

    def _handle_alm_model_change(self, selected_model: Optional[str]) -> None:
        self._sync_segment_time_default(selected_model)
        self._sync_alm_language_options(selected_model)
        self._sync_alm_audio_task_options(selected_model)
        self._on_model_select_change("alm", selected_model)

    def _on_model_select_change(self, model_type: str, route_name: Optional[str]) -> None:
        panel_map = {
            "ocr": getattr(self, "_ocr_config_panel", None),
            "vlm": getattr(self, "_vlm_config_panel", None),
            "alm": getattr(self, "_alm_config_panel", None),
        }
        panel: Optional["ModelConfigPanel"] = panel_map.get(model_type)
        if panel is None:
            return
        if route_name and route_name.strip():
            panel.show(route_name)
        else:
            panel.hide()
        self._refresh_local_model_fit_summary()

    def _ensure_execution_panel(self):
        if self.panel is not None:
            return self.panel

        container = self._execution_panel_container
        if container is None:
            raise RuntimeError("Execution panel container is not ready")

        execution_panel_cls = _load_execution_panel_cls()
        with container:
            self.panel = execution_panel_cls(start_label=t("start_caption"))
        self.panel._on_start = self._start_caption
        return self.panel

    def _schedule_gpu_probe_refresh(self) -> None:
        if self._gpu_probe_scheduled:
            return
        self._gpu_probe_scheduled = True
        asyncio.create_task(self._load_gpu_probe_async())

    def _has_remote_provider_config(self) -> bool:
        has_api = any(self._has_text(getattr(key_input, "value", "")) for key_input in self.api_keys.values())
        has_openai_url = hasattr(self, "openai_base_url") and self._has_text(self.openai_base_url.value)
        return has_api or has_openai_url

    def _has_local_route_config(self) -> bool:
        if self._has_text(self.ocr_model.value) and not route_requires_remote_config("ocr_model", self.ocr_model.value):
            return True
        if self._has_text(self.vlm_image_model.value) and not route_requires_remote_config("vlm_image_model", self.vlm_image_model.value):
            return True
        if self._has_text(self._current_alm_model()) and not route_requires_remote_config("alm_model", self._current_alm_model()):
            return True
        return False

    def _has_caption_provider_config(self) -> bool:
        return self._has_remote_provider_config() or self._has_local_route_config()

    @staticmethod
    def _append_extra(extra_args: list[str], seen: set[str], extra_name: Optional[str]) -> None:
        if not extra_name or extra_name in seen:
            return
        seen.add(extra_name)
        extra_args.extend(["--extra", extra_name])

    @classmethod
    def _repo_id_requires_quantized_runtime(cls, repo_id: Optional[str]) -> bool:
        return bool(repo_id and cls._BITSANDBYTES_REPO_PATTERN.search(repo_id.lower()))

    def _selected_local_route_model_ids(self) -> tuple[str, ...]:
        model_id_map = _load_current_route_model_ids()
        route_names = (
            getattr(getattr(self, "ocr_model", None), "value", "") or "",
            getattr(getattr(self, "vlm_image_model", None), "value", "") or "",
            self._current_alm_model(),
        )
        return tuple(model_id_map.get(route_name, "") for route_name in route_names if route_name)

    def _build_local_extra_args(self) -> list[str]:
        extra_args: list[str] = []
        seen: set[str] = set()

        self._append_extra(extra_args, seen, self.OCR_EXTRA_MAP.get(self.ocr_model.value or ""))
        self._append_extra(extra_args, seen, self.VLM_EXTRA_MAP.get(self.vlm_image_model.value or ""))
        self._append_extra(extra_args, seen, self.ALM_EXTRA_MAP.get(self._current_alm_model()))
        if any(
            self._repo_id_requires_quantized_runtime(model_id) for model_id in self._selected_local_route_model_ids()
        ):
            self._append_extra(extra_args, seen, self.QUANTIZED_RUNTIME_EXTRA)

        return extra_args

    def _build_caption_args(self, dataset_path: str) -> list[str]:
        args = [dataset_path]

        for api_name, config in self.API_CONFIGS.items():
            if config.get("is_openai_compatible"):
                continue

            key_input = self.api_keys.get(config["key_name"])
            if key_input and key_input.value:
                args.append(f"--{config['key_name']}={key_input.value}")

                model_select = getattr(self, f"{config['key_name']}_model", None)
                if model_select and model_select.value:
                    args.append(f"--{config['key_name'].replace('api_key', 'model_path')}={model_select.value}")

                if config["supports_task"]:
                    task_input = getattr(self, f"{config['key_name']}_task", None)
                    if task_input and task_input.value:
                        args.append(f"--gemini_task={task_input.value}")

                if api_name == "Mistral" and self.config.get("mistral_ocr_mode"):
                    args.append("--ocr_model=mistral_ocr")
                    if self.config["document_image"]:
                        args.append("--document_image")

        if hasattr(self, "openai_base_url") and self.openai_base_url.value:
            args.append(f"--openai_base_url={self.openai_base_url.value}")
            openai_key = self.api_keys.get("openai_api_key")
            if openai_key and openai_key.value:
                args.append(f"--openai_api_key={openai_key.value}")
            if hasattr(self, "openai_model_name") and self.openai_model_name.value:
                args.append(f"--openai_model_name={self.openai_model_name.value}")

        if self.pair_dir.value:
            args.append(f"--pair_dir={self.pair_dir.value}")

        if self.config["dir_name"]:
            args.append("--dir_name")

        if self.config["not_clip_with_caption"]:
            args.append("--not_clip_with_caption")

        if self.mode.value != "all":
            args.append(f"--mode={self.mode.value}")

        args.append(f"--wait_time={self.config['wait_time']}")
        args.append(f"--max_retries={self.config['max_retries']}")
        if self.config.get("segment_time_explicit"):
            args.append(f"--segment_time={self.config['segment_time']}")

        if self.config["tags_highlightrate"] != 0.38:
            args.append(f"--tags_highlightrate={self.config['tags_highlightrate']}")

        if self.ocr_model.value:
            ocr_arg = f"--ocr_model={self.ocr_model.value}"
            if ocr_arg not in args:
                args.append(ocr_arg)
            if self.config["document_image"] and "--document_image" not in args:
                args.append("--document_image")

        if self.vlm_image_model.value:
            args.append(f"--vlm_image_model={self.vlm_image_model.value}")

        if self._current_alm_model():
            args.append(f"--alm_model={self._current_alm_model()}")
            alm_language = getattr(getattr(self, "alm_language", None), "value", "")
            if self._alm_requires_language() and self._has_text(alm_language):
                args.append(f"--alm_language={alm_language}")
            alm_audio_task = getattr(getattr(self, "alm_audio_task", None), "value", "")
            if self._alm_requires_audio_task() and self._has_text(alm_audio_task):
                args.append(f"--audio_task={alm_audio_task}")

        return args

    def _build_job_name(self) -> str:
        """构建 Job 显示名（包含主要 provider 信息）"""
        for api_name, config in self.API_CONFIGS.items():
            if config.get("is_openai_compatible"):
                continue
            key_input = self.api_keys.get(config["key_name"])
            if key_input and key_input.value:
                return f"Caption ({api_name})"
        if hasattr(self, "openai_base_url") and self.openai_base_url.value:
            return "Caption (OpenAI-Compatible)"
        if self.ocr_model.value:
            return f"Caption (OCR: {self.ocr_model.value})"
        if self.vlm_image_model.value:
            return f"Caption (VLM: {self.vlm_image_model.value})"
        if self._current_alm_model():
            return f"Caption (ALM: {self._current_alm_model()})"
        return "Caption"

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            # 页面标题
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("subtitles", size="32px").style(f"color: {COLORS['primary']};")
                with ui.column().classes("gap-0"):
                    ui.label(t("caption_title")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(t("caption_desc")).classes("text-body2").style("color: var(--color-text-secondary);")

            # 使用标签页组织内容
            with ui.tabs().classes("w-full") as tabs:
                basic_tab = ui.tab(t("basic_settings"), icon="tune")
                api_tab = ui.tab(t("api_configuration"), icon="key")
                ocr_tab = ui.tab(t("local_model_routes"), icon="text_fields")

            with ui.tab_panels(tabs, value=basic_tab).classes("w-full"):
                # 基础设置
                with ui.tab_panel(basic_tab):
                    self._render_basic_settings()

                # API 配置
                with ui.tab_panel(api_tab):
                    self._render_api_settings()

                # OCR/VLM
                with ui.tab_panel(ocr_tab):
                    self._render_ocr_settings()

            # 执行面板懒创建，避免首屏被日志 UI 拖慢。
            self._execution_panel_container = ui.column().classes("w-full")

        ui.timer(0.01, self._ensure_execution_panel, once=True)
        ui.timer(0.01, self._schedule_gpu_probe_refresh, once=True)

    def _render_basic_settings(self):
        """渲染基础设置"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("folder_open", size="22px").style(f"color: {COLORS['info']};")
                ui.label(t("dataset_path")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            # 数据集路径
            self.dataset_path = create_path_selector(
                label=t("dataset_path"), selection_type="dir", placeholder=t("input_path_placeholder")
            )

            # 配对目录
            self.pair_dir = create_path_selector(label=t("pair_dir"), selection_type="dir", placeholder=t("pair_dir_hint"))

            with ui.row().classes("w-full gap-4 q-mt-md"):
                # 模式 - 带图标的现代化下拉框
                self.mode = styled_select(
                    options=dict(zip(self.MODES, self.MODES)),
                    value="long",
                    label=t("mode"),
                    icon="tune",
                    icon_color=COLORS["primary"],
                    flex=1,
                )

            # 开关选项
            with ui.row().classes("w-full gap-4 q-mt-md items-center"):
                toggle_switch("dir_name", self.config, "dir_name")
                toggle_switch("not_clip_with_caption", self.config, "not_clip_with_caption")
                editable_slider(
                    label_key="tags_highlightrate",
                    value_ref=self.config,
                    value_key="tags_highlightrate",
                    min_val=0.0,
                    max_val=1.0,
                    step=0.01,
                    decimals=2,
                )

            # 数值设置 - 使用可编辑滑块
            with ui.row().classes("w-full gap-4 q-mt-md"):
                editable_slider(
                    label_key="wait_time", value_ref=self.config, value_key="wait_time", min_val=0, max_val=60, step=1, decimals=0
                )

                editable_slider(
                    label_key="max_retries",
                    value_ref=self.config,
                    value_key="max_retries",
                    min_val=1,
                    max_val=1000,
                    step=1,
                    decimals=0,
                )

                self.segment_time_slider = editable_slider(
                    label_key="segment_time",
                    value_ref=self.config,
                    value_key="segment_time",
                    min_val=1,
                    max_val=3600,
                    step=10,
                    decimals=0,
                    on_change=self._handle_segment_time_change,
                )

    def _render_api_settings(self):
        """渲染 API 设置"""
        for api_name, config in self.API_CONFIGS.items():
            with ui.expansion(f"{api_name} API").classes("w-full q-mb-md"):
                with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                    # Kimi-Code 提示
                    if api_name == "Kimi-Code":
                        ui.label(t("kimi_code_desc")).classes(
                            "text-caption q-mb-sm"
                        ).style("color: var(--color-text-secondary);")

                    # OpenAI-Compatible 特殊渲染
                    if config.get("is_openai_compatible"):
                        self._render_openai_compatible_settings(api_name, config)
                        continue

                    # API Key
                    key_input = ui.input(label=f"{api_name} {t('api_key')}", placeholder=t("api_key_placeholder"), password=True)
                    key_input.classes("modern-input w-full")
                    self.api_keys[config["key_name"]] = key_input

                    # 模型选择
                    if config.get("custom_model_input"):
                        # Ark 等需要用户手动输入 endpoint model ID
                        model_input = ui.input(
                            label=t("model_path"), placeholder=t("enter_endpoint_model_id"), value=config["default_model"]
                        )
                        model_input.classes("modern-input w-full")
                        setattr(self, f"{config['key_name']}_model", model_input)
                    else:
                        model_select = styled_select(
                            options=dict(zip(config["models"], config["models"])),
                            value=config["default_model"],
                            label=t("model_path"),
                            icon="smart_toy",
                            icon_color=COLORS["primary"],
                            new_value_mode="add-unique",
                        )
                        setattr(self, f"{config['key_name']}_model", model_select)

                    # Gemini 特有：任务名称
                    if config["supports_task"]:
                        task_names = _load_gemini_task_names()
                        task_options = {"": ""} | {n: n for n in task_names}
                        task_select = styled_select(
                            options=task_options,
                            value="",
                            label=t("gemini_task"),
                            icon="task_alt",
                            icon_color=COLORS["info"],
                            new_value_mode="add-unique",
                        )
                        setattr(self, f"{config['key_name']}_task", task_select)

                    # Mistral 专有：OCR 模式开关
                    if api_name == "Mistral":
                        ui.separator().classes("q-my-md")
                        with ui.row().classes("w-full items-center gap-2 q-mb-xs"):
                            ui.icon("document_scanner", size="20px").style(f"color: {COLORS['info']};")
                            ui.label(t("mistral_ocr_mode")).classes(
                                "text-body2 text-weight-medium"
                            ).style("color: var(--color-text);")

                        toggle_switch(
                            "mistral_ocr_toggle", self.config, "mistral_ocr_mode",
                            on_change=self._on_mistral_ocr_toggle,
                        )

                        self._mistral_doc_container = ui.column().classes("w-full q-mt-sm")
                        self._mistral_doc_container.set_visibility(self.config["mistral_ocr_mode"])
                        with self._mistral_doc_container:
                            toggle_switch("document_image", self.config, "document_image")

    def _on_mistral_ocr_toggle(self, enabled: bool) -> None:
        if hasattr(self, "_mistral_doc_container"):
            self._mistral_doc_container.set_visibility(enabled)

    def _render_openai_compatible_settings(self, api_name: str, config: dict):
        """渲染 OpenAI Compatible API 的专用设置"""
        ui.label(t("openai_compatible_desc")).classes("text-caption q-mb-sm").style("color: var(--color-text-secondary);")

        # Base URL（必填）
        self.openai_base_url = styled_input(
            value="",
            label="Base URL",
            icon="dns",
            icon_color=COLORS["info"],
            placeholder="http://localhost:8000/v1",
        )

        with ui.row().classes("w-full gap-4"):
            # API Key（可选）
            key_input = ui.input(
                label=f"{api_name} API Key",
                placeholder=t("openai_no_key_hint"),
                password=True,
            )
            key_input.classes("modern-input w-full")
            key_input.style("flex: 1;")
            self.api_keys[config["key_name"]] = key_input

        # 模型名称
        self.openai_model_name = styled_input(
            value="",
            label=t("model_path"),
            icon="smart_toy",
            icon_color=COLORS["primary"],
            placeholder="Qwen2-VL-7B-Instruct",
            flex=1,
        )

    def _render_ocr_settings(self):
        """渲染 OCR/VLM/ALM 设置"""
        model_config_panel_cls = _load_model_config_panel_cls()
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("text_fields", size="22px").style(f"color: {COLORS['info']};")
                ui.label(t("ocr_settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            self._local_model_fit_container = ui.column().classes("w-full q-mb-md")

            # OCR 模型 - 带图标的现代化下拉框
            self.ocr_model = styled_select(
                options=self._build_ocr_labels(),
                value="",
                label=t("ocr_model"),
                icon="text_fields",
                icon_color=COLORS["info"],
            )
            _ocr_cfg_container = ui.column().classes("w-full")
            self._ocr_config_panel = model_config_panel_cls(_ocr_cfg_container, on_save=self._handle_model_config_saved)
            self.ocr_model.on_value_change(lambda e: self._on_model_select_change("ocr", e.value))

            # 开关选项
            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("document_image", self.config, "document_image")

            with ui.row().classes("w-full items-center gap-2 q-mb-md q-mt-md"):
                ui.icon("visibility", size="22px").style(f"color: {COLORS['secondary']};")
                ui.label(t("vlm_settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            # VLM 图像模型 - 带图标的现代化下拉框
            self.vlm_image_model = styled_select(
                options=self._build_route_labels(self.VLM_MODELS),
                value="",
                label=t("vlm_image_model"),
                icon="visibility",
                icon_color=COLORS["secondary"],
            )
            _vlm_cfg_container = ui.column().classes("w-full")
            self._vlm_config_panel = model_config_panel_cls(_vlm_cfg_container, on_save=self._handle_model_config_saved)
            self.vlm_image_model.on_value_change(lambda e: self._on_model_select_change("vlm", e.value))

            with ui.row().classes("w-full items-center gap-2 q-mb-md q-mt-md"):
                ui.icon("graphic_eq", size="22px").style(f"color: {COLORS['primary']};")
                ui.label(t("alm_settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            self.alm_model = styled_select(
                options=self._build_route_labels(self.ALM_MODELS),
                value="",
                label=t("alm_model"),
                icon="graphic_eq",
                icon_color=COLORS["primary"],
                on_change=self._handle_alm_model_change,
                searchable=False,
            )
            _alm_cfg_container = ui.column().classes("w-full")
            self._alm_config_panel = model_config_panel_cls(_alm_cfg_container, on_save=self._handle_model_config_saved)

            with ui.column().classes("w-full") as self._alm_language_container:
                self.alm_language = styled_select(
                    options=self._alm_language_options(),
                    value=None,
                    label=t("alm_language"),
                    icon="translate",
                    icon_color=COLORS["primary"],
                    placeholder="Select transcription language",
                    searchable=False,
                )
            self._alm_language_container.set_visibility(False)

            with ui.column().classes("w-full") as self._alm_audio_task_container:
                self.alm_audio_task = styled_select(
                    options=self._alm_audio_task_options(),
                    value=self._initial_alm_audio_task_value(),
                    label=t("audio_task"),
                    icon="subtitles",
                    icon_color=COLORS["primary"],
                    placeholder="Select audio task",
                    searchable=False,
                    on_change=self._handle_audio_task_change,
                )
            self._alm_audio_task_container.set_visibility(False)

            self._sync_segment_time_default()
            self._sync_alm_language_options()
            self._sync_alm_audio_task_options()
            self._refresh_local_model_fit_summary()

    async def _start_caption(self):
        """开始字幕生成"""
        dataset_path = self.dataset_path.value
        if not dataset_path or not Path(dataset_path).exists():
            ui.notify(t("select_valid_dataset"), type="warning")
            return

        # 至少要有一个可执行的 provider：远程 API，或本地 OCR/VLM 路由
        if not self._has_caption_provider_config():
            ui.notify(t("at_least_one_api"), type="warning")
            return

        args = self._build_caption_args(dataset_path)
        uv_extra_args = self._build_local_extra_args()
        job_name = self._build_job_name()

        runner_kwargs = {}
        if uv_extra_args:
            runner_kwargs["uv_extra_args"] = uv_extra_args

        def pre_log(lv):
            lv.info(t("log_start_caption"))
            lv.info(f"{t('log_dataset_path')}: {dataset_path}")
            lv.info(f"{t('log_mode')}: {self.mode.value}")
            lv.info(f"{t('log_params')}: {self._format_args_for_log(args)}")
            if uv_extra_args:
                lv.info(f"uv extras: {uv_extra_args}")

        panel = self._ensure_execution_panel()

        await panel.run_job(
            "module.captioner",
            args,
            name=job_name,
            runner_kwargs=runner_kwargs,
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("caption_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("caption_failed"), type="negative"),
        )


def render_caption_step():
    """渲染字幕生成步骤"""
    step = CaptionStep()
    step.render()
