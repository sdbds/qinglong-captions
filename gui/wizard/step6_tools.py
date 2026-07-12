"""步骤 6: 实用工具 - 对应 watermark_detect, preprocess, reward_model 等脚本"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from nicegui import ui

from gui.components.advanced_inputs import editable_slider, styled_input, styled_select, toggle_switch
from gui.components.path_selector import create_path_selector
from gui.theme import COLORS, get_classes
from gui.utils.i18n import t
from module.muscriptor_tool.options import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CFG_COEF,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMATS,
    DEFAULT_PREVIEW_FORMAT,
    DEFAULT_TEMPERATURE,
    DecodingMode,
    ModelVariant,
    OutputFormat,
    PreviewContent,
    PreviewFormat,
    PreviewRequest,
    TranscriptionOptions,
)

if TYPE_CHECKING:
    from gui.components.execution_panel import ExecutionPanel


DEFAULT_GAME_MODEL_REPO_ID = "bdsqlsz/GAME-1.0-large-ONNX"
DEFAULT_VOCAL_MIDI_BATCH_SIZE = 4
DEFAULT_VOCAL_MIDI_SEG_THRESHOLD = 0.2
DEFAULT_VOCAL_MIDI_SEG_RADIUS = 0.02
DEFAULT_VOCAL_MIDI_T0 = 0.0
DEFAULT_VOCAL_MIDI_NSTEPS = 8
DEFAULT_VOCAL_MIDI_EST_THRESHOLD = 0.2
DEFAULT_VOCAL_MIDI_OUTPUT_FORMATS = "mid"
DEFAULT_SHEET_MUSIC_REPO_ID = "bdsqlsz/musvit-onnx"
DEFAULT_SHEET_MUSIC_MODEL_DIR = "huggingface"
DEFAULT_SHEET_MUSIC_OUTPUT_DIR = "workspace/musvit_output"
DEFAULT_SHEET_MUSIC_PDF_DPI = 144
DEFAULT_MUSCRIPTOR_OUTPUT_DIR = "workspace/muscriptor_output"

_MUSCRIPTOR_INSTRUMENT_CACHE: dict[str, tuple[str, ...]] = {}


def _parse_music_instrument_catalog(lines: list[Any]) -> tuple[str, ...]:
    for raw_line in reversed(lines):
        text = raw_line[1] if isinstance(raw_line, tuple) and len(raw_line) == 2 else raw_line
        for line in reversed(str(text).splitlines()):
            try:
                payload = json.loads(line.strip())
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                continue
            version = str(payload.get("package_version") or "").strip()
            raw_names = payload.get("instruments")
            if not version or not isinstance(raw_names, list):
                continue
            names = tuple(dict.fromkeys(str(name).strip() for name in raw_names if str(name).strip()))
            if not names:
                continue
            _MUSCRIPTOR_INSTRUMENT_CACHE[version] = names
            return names
    raise ValueError("MuScriptor instrument probe did not return a valid catalog")

GAME_ONNX_MODEL_LABELS: dict[str, str] = {
    "bdsqlsz/GAME-1.0-small-ONNX": "GAME-1.0-small-ONNX",
    "bdsqlsz/GAME-1.0-medium-ONNX": "GAME-1.0-medium-ONNX",
    "bdsqlsz/GAME-1.0-large-ONNX": "GAME-1.0-large-ONNX",
}

DEFAULT_DEPTH_INFERENCE_STEPS = -1
DEFAULT_SEED = 1026
DEFAULT_DEPTH_RESOLUTION = 720

SEE_THROUGH_REPO_MAP = {
    "none": {
        "layerdiff": "layerdifforg/seethroughv0.0.2_layerdiff3d",
        "depth": "24yearsold/seethroughv0.0.1_marigold",
    },
    "nf4": {
        "layerdiff": "24yearsold/seethroughv0.0.2_layerdiff3d_nf4",
        "depth": "24yearsold/seethroughv0.0.1_marigold_nf4",
    },
}


@dataclass(frozen=True)
class _SeeThroughRecommendation:
    min_vram_gb: float | None
    resolution: int
    resolution_depth: int
    dtype: str
    offload_policy: str
    group_offload: bool
    quant_mode: str
    repo_id_layerdiff: str
    repo_id_depth: str
    note: str | None = None


def _resolve_see_through_repo_ids(
    *,
    quant_mode: str | None,
    repo_id_layerdiff: str | None = None,
    repo_id_depth: str | None = None,
) -> tuple[str, str]:
    normalized_quant_mode = str(quant_mode or "none").strip().lower()
    if normalized_quant_mode not in SEE_THROUGH_REPO_MAP:
        normalized_quant_mode = "none"

    defaults = SEE_THROUGH_REPO_MAP[normalized_quant_mode]
    known_layerdiff_repos = {repos["layerdiff"] for repos in SEE_THROUGH_REPO_MAP.values()}
    known_depth_repos = {repos["depth"] for repos in SEE_THROUGH_REPO_MAP.values()}

    def _resolve_repo(requested_repo: str | None, *, default_repo: str, known_repos: set[str]) -> str:
        normalized_repo = str(requested_repo or "").strip()
        if not normalized_repo:
            return default_repo
        if normalized_repo in known_repos:
            return default_repo
        return normalized_repo

    return (
        _resolve_repo(repo_id_layerdiff, default_repo=defaults["layerdiff"], known_repos=known_layerdiff_repos),
        _resolve_repo(repo_id_depth, default_repo=defaults["depth"], known_repos=known_depth_repos),
    )


def _reward_device_options() -> Dict[str, str]:
    options = {"auto": "Auto", "cpu": "CPU"}
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(index)
                except Exception:
                    name = ""
                label = f"CUDA {index}"
                if name:
                    label = f"{label} - {name}"
                options[f"cuda:{index}"] = label
        else:
            options["cuda"] = "CUDA"
    except Exception:
        options["cuda"] = "CUDA"
    return options


def _default_see_through_recommendation() -> _SeeThroughRecommendation:
    repo_id_layerdiff, repo_id_depth = _resolve_see_through_repo_ids(quant_mode="none")
    return _SeeThroughRecommendation(
        min_vram_gb=None,
        resolution=768,
        resolution_depth=DEFAULT_DEPTH_RESOLUTION,
        dtype="float32",
        offload_policy="delete",
        group_offload=False,
        quant_mode="none",
        repo_id_layerdiff=repo_id_layerdiff,
        repo_id_depth=repo_id_depth,
        note=None,
    )


def _probe_see_through_defaults() -> tuple[Any | None, Any]:
    try:
        from module.gpu_profile import get_cached_gpu_probe
        from module.see_through.see_through_profile import recommend_see_through_config

        probe = get_cached_gpu_probe(refresh=True)
        return probe, recommend_see_through_config(probe)
    except Exception:
        return None, _default_see_through_recommendation()


def _format_gpu_summary(probe: Any | None) -> str:
    if probe is None:
        return "--"

    try:
        from module.gpu_profile import format_gpu_summary

        return format_gpu_summary(probe)
    except Exception:
        return "--"


def _format_gpu_device_lines(probe: Any | None) -> tuple[str, ...]:
    if probe is None:
        return ()

    try:
        from module.gpu_profile import format_gpu_device_lines

        return format_gpu_device_lines(probe)
    except Exception:
        return ()


def _load_execution_panel_cls():
    from gui.components.execution_panel import ExecutionPanel

    return ExecutionPanel


class ToolsStep:
    """工具页面"""

    TOOL_TABS = (
        ("watermark", "watermark_detection", "water_drop"),
        ("preprocess", "preprocess", "image"),
        ("reward", "reward_model", "stars"),
        ("audio_separator", "audio_separator", "graphic_eq"),
        ("music_transcription", "music_transcription", "piano"),
        ("sheet_music", "sheet_music", "library_music"),
        ("translate", "translate", "translate"),
        ("see_through", "see_through", "layers"),
    )

    SEE_THROUGH_LAYERDIFF_DEFAULT = SEE_THROUGH_REPO_MAP["none"]["layerdiff"]
    SEE_THROUGH_LAYERDIFF_NF4 = SEE_THROUGH_REPO_MAP["nf4"]["layerdiff"]
    SEE_THROUGH_DEPTH_DEFAULT = SEE_THROUGH_REPO_MAP["none"]["depth"]
    SEE_THROUGH_DEPTH_NF4 = SEE_THROUGH_REPO_MAP["nf4"]["depth"]
    SEE_THROUGH_LAYERDIFF_REPOS = {
        SEE_THROUGH_LAYERDIFF_DEFAULT: SEE_THROUGH_LAYERDIFF_DEFAULT,
        SEE_THROUGH_LAYERDIFF_NF4: SEE_THROUGH_LAYERDIFF_NF4,
    }
    SEE_THROUGH_DEPTH_REPOS = {
        SEE_THROUGH_DEPTH_DEFAULT: SEE_THROUGH_DEPTH_DEFAULT,
        SEE_THROUGH_DEPTH_NF4: SEE_THROUGH_DEPTH_NF4,
    }
    SEE_THROUGH_DTYPES = {
        "bfloat16": "BF16",
        "float16": "FP16",
        "float32": "FP32",
    }
    SEE_THROUGH_QUANT_MODE_LABEL_KEYS = {
        "none": "see_through_quant_mode_standard",
        "nf4": "see_through_quant_mode_nf4",
    }
    SEE_THROUGH_DEPTH_RESOLUTION_LABELS = {
        "720": "720 px",
        "768": "768 px",
        "1024": "1024 px",
        "1280": "1280 px",
        "-1": "see_through_resolution_depth_match",
    }
    SEE_THROUGH_OFFLOAD_POLICY_LABEL_KEYS = {
        "delete": "see_through_offload_delete",
        "cpu": "see_through_offload_cpu",
    }
    SHEET_MUSIC_PREPROCESS_MODE_LABEL_KEYS = {
        "page_resize": "sheet_music_preprocess_page_resize",
        "pad_square": "sheet_music_preprocess_pad_square",
    }
    MUSCRIPTOR_MODEL_OPTIONS = {item.value: item.value.title() for item in ModelVariant}
    MUSCRIPTOR_BASE_DEVICE_OPTIONS = {
        "auto": "Auto",
        "cpu": "CPU",
        "cuda": "CUDA",
    }

    # 水印检测模型
    WATERMARK_MODELS = [
        "bdsqlsz/joycaption-watermark-detection-onnx",
        "bdsqlsz/Watermark-Detection-SigLIP2-onnx",
    ]

    # 评分模型
    REWARD_MODELS = [
        "RE-N-Y/hpsv3",
        "RE-N-Y/aesthetic-shadow-v2",
        "RE-N-Y/clipscore-vit-large-patch14",
        "RE-N-Y/pickscore",
        "yuvalkirstain/PickScore_v1",
        "RE-N-Y/mpsv1",
        "RE-N-Y/hpsv21",
        "RE-N-Y/ImageReward",
        "RE-N-Y/laion-aesthetic",
    ]

    # 变换类型
    TRANSFORM_TYPES = ["auto", "none"]

    # 翻译模型
    TRANSLATE_MODELS = [
        "tencent/Hy-MT2-7B",
        "tencent/Hy-MT2-1.8B-FP8",
    ]

    # 翻译支持的语言
    TRANSLATE_LANGUAGES = {
        "auto": "Auto Detect",
        "en": "English",
        "zh_cn": "Chinese (Simplified)",
        "zh_tw": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }

    VOCAL_MIDI_MODELS = dict(GAME_ONNX_MODEL_LABELS)
    VOCAL_MIDI_LANGUAGES = {
        "auto": "Auto Detect",
        "en": "English",
        "ja": "Japanese",
        "yue": "Cantonese",
        "zh": "Chinese",
    }
    VOCAL_MIDI_OUTPUT_FORMATS = {
        "mid": "MIDI (.mid)",
        "txt": "Text (.txt)",
        "csv": "CSV (.csv)",
    }

    def __init__(self):
        self.gpu_probe = None
        self.see_through_recommendation = _default_see_through_recommendation()
        self.config: Dict[str, Any] = {
            "watermark_batch_size": 12,
            "watermark_thresh": 1.0,
            "preprocess_workers": 8,
            "max_long_edge": 2048,
            "max_short_edge": 0,
            "max_pixels": 0,
            "crop_transparent": True,
            "preprocess_recursive": True,
            "reward_batch_size": 1,
            "audio_separator_segment_size": 1101,
            "audio_separator_overlap": 8,
            "audio_separator_batch_size": 1,
            "audio_separator_overwrite": False,
            "audio_separator_harmony_separation": False,
            "audio_separator_vocal_midi": False,
            "audio_separator_vocal_midi_batch_size": DEFAULT_VOCAL_MIDI_BATCH_SIZE,
            "audio_separator_vocal_midi_seg_threshold": DEFAULT_VOCAL_MIDI_SEG_THRESHOLD,
            "audio_separator_vocal_midi_seg_radius": DEFAULT_VOCAL_MIDI_SEG_RADIUS,
            "audio_separator_vocal_midi_t0": DEFAULT_VOCAL_MIDI_T0,
            "audio_separator_vocal_midi_nsteps": DEFAULT_VOCAL_MIDI_NSTEPS,
            "audio_separator_vocal_midi_est_threshold": DEFAULT_VOCAL_MIDI_EST_THRESHOLD,
            "music_transcription_input_mode": "directory",
            "music_transcription_model": DEFAULT_MODEL.value,
            "music_transcription_device": DEFAULT_DEVICE,
            "music_transcription_batch_size": 0,
            "music_transcription_instrument_mode": "auto",
            "music_transcription_instruments": [],
            "music_transcription_decode_mode": "greedy",
            "music_transcription_temperature": DEFAULT_TEMPERATURE,
            "music_transcription_cfg_coef": DEFAULT_CFG_COEF,
            "music_transcription_strict_eos": False,
            "music_transcription_beam_size": DEFAULT_BEAM_SIZE,
            "music_transcription_output_formats": [item.value for item in DEFAULT_OUTPUT_FORMATS],
            "music_transcription_preview_mode": "none",
            "music_transcription_preview_format": DEFAULT_PREVIEW_FORMAT.value,
            "music_transcription_recursive": True,
            "music_transcription_skip_completed": True,
            "music_transcription_overwrite": False,
            "music_transcription_fail_fast": False,
            "music_transcription_notes": False,
            "sheet_music_batch_size": 1,
            "sheet_music_pdf_dpi": DEFAULT_SHEET_MUSIC_PDF_DPI,
            "sheet_music_recursive": True,
            "sheet_music_skip_completed": True,
            "sheet_music_overwrite": False,
            "sheet_music_force_download": False,
            "translate_max_chars": 2200,
            "translate_context_chars": 300,
            "translate_max_new_tokens": 4096,
            "translate_temperature": 0.0,
            "translate_skip_normalize": False,
            "translate_normalize_only": False,
            "translate_no_export": False,
            "translate_force_reimport": False,
            "see_through_resolution": self.see_through_recommendation.resolution,
            "see_through_resolution_depth": self.see_through_recommendation.resolution_depth,
            "see_through_inference_steps_depth": DEFAULT_DEPTH_INFERENCE_STEPS,
            "see_through_seed": DEFAULT_SEED,
            "see_through_dtype": self.see_through_recommendation.dtype,
            "see_through_quant_mode": self.see_through_recommendation.quant_mode,
            "see_through_group_offload": self.see_through_recommendation.group_offload,
            "see_through_offload_policy": self.see_through_recommendation.offload_policy,
            "see_through_repo_id_layerdiff": self.see_through_recommendation.repo_id_layerdiff,
            "see_through_repo_id_depth": self.see_through_recommendation.repo_id_depth,
            "see_through_limit_images": 0,
            "see_through_skip_completed": True,
            "see_through_continue_on_error": True,
            "see_through_save_to_psd": True,
            "see_through_tblr_split": False,
            "see_through_force_eager_attention": False,
        }
        self.panel: "ExecutionPanel | None" = None
        self._tool_tab_containers: Dict[str, Any] = {}
        self._rendered_tool_tabs: set[str] = set()
        self._execution_panel_container = None
        self._active_tool_tab = "watermark"
        self._gpu_probe_scheduled = False
        self._see_through_user_edited = False
        self._see_through_summary_label = None
        self._see_through_note_label = None
        self._see_through_summary_meta_container = None
        self._see_through_gpu_details_container = None
        self._see_through_gpu_details_open = False
        self._audio_separator_vocal_midi_container = None
        self._music_transcription_instrument_container = None
        self._music_transcription_sampling_container = None
        self._music_transcription_beam_container = None
        self._music_transcription_preview_container = None
        self._music_transcription_instrument_loading = False

    def _get_tool_renderer(self, tab_key: str):
        renderers = {
            "watermark": self._render_watermark_tool,
            "preprocess": self._render_preprocess_tool,
            "reward": self._render_reward_tool,
            "audio_separator": self._render_audio_separator_tool,
            "music_transcription": self._render_music_transcription_tool,
            "sheet_music": self._render_sheet_music_tool,
            "translate": self._render_translate_tool,
            "see_through": self._render_see_through_tool,
        }
        try:
            return renderers[tab_key]
        except KeyError as exc:
            raise ValueError(f"Unknown tool tab: {tab_key}") from exc

    def _ensure_tool_panel_rendered(self, tab_key: str) -> None:
        if tab_key in self._rendered_tool_tabs:
            return

        container = self._tool_tab_containers.get(tab_key)
        if container is None:
            return

        container.clear()
        with container:
            self._get_tool_renderer(tab_key)()
        self._rendered_tool_tabs.add(tab_key)
        if tab_key == "see_through":
            self._schedule_see_through_recommendation_refresh()
        if tab_key == "music_transcription":
            self._schedule_see_through_recommendation_refresh()

    def _ensure_execution_panel(self):
        if self.panel is not None:
            return self.panel

        container = self._execution_panel_container
        if container is None:
            raise RuntimeError("Execution panel container is not ready")

        execution_panel_cls = _load_execution_panel_cls()
        with container:
            self.panel = execution_panel_cls(
                show_start=True,
                start_label=t("start_watermark"),
                on_start=self._start_watermark,
            )
        self._sync_execution_action()
        return self.panel

    def _tool_action_for_tab(self, tab_key: str) -> tuple[str, Any]:
        actions = {
            "watermark": ("start_watermark", self._start_watermark),
            "preprocess": ("start_preprocess", self._start_preprocess),
            "reward": ("start_scoring", self._start_reward),
            "audio_separator": ("start_audio_separator", self._start_audio_separator),
            "music_transcription": ("start_music_transcription", self._start_music_transcription),
            "sheet_music": ("start_sheet_music", self._start_sheet_music),
            "translate": ("start_translate", self._start_translate),
            "see_through": ("start_see_through", self._start_see_through),
        }
        return actions.get(tab_key, actions["watermark"])

    def _sync_execution_action(self) -> None:
        if self.panel is None:
            return
        label_key, callback = self._tool_action_for_tab(self._active_tool_tab)
        self.panel.set_action(t(label_key), callback, enabled=True)

    def _handle_tool_tab_change(self, tab_key: str) -> None:
        self._active_tool_tab = tab_key
        self._ensure_tool_panel_rendered(tab_key)
        self._sync_execution_action()

    def _build_see_through_summary(self) -> str:
        recommendation = self.see_through_recommendation
        minimum_text = (
            t("cpu_fallback")
            if recommendation.min_vram_gb is None
            else f">= {recommendation.min_vram_gb:.0f} GB"
        )
        quant_mode_options = self._see_through_quant_mode_options()
        return (
            f"{t('gpu')}: {_format_gpu_summary(self.gpu_probe)} | "
            f"{t('recommended_profile')}: {minimum_text} -> "
            f"{recommendation.resolution}px / depth {recommendation.resolution_depth}px / "
            f"{quant_mode_options.get(recommendation.quant_mode, recommendation.quant_mode)} / "
            f"{t('group_offload')}={t('status_on_inline' if recommendation.group_offload else 'status_off_inline')} / "
            f"{self.SEE_THROUGH_DTYPES.get(recommendation.dtype, recommendation.dtype)}"
        )

    def _gpu_detail_lines(self) -> tuple[str, ...]:
        return _format_gpu_device_lines(self.gpu_probe)

    def _see_through_quant_mode_options(self) -> dict[str, str]:
        return {
            option: t(label_key)
            for option, label_key in self.SEE_THROUGH_QUANT_MODE_LABEL_KEYS.items()
        }

    def _see_through_depth_resolution_options(self) -> dict[str, str]:
        options: dict[str, str] = {}
        for value, label in self.SEE_THROUGH_DEPTH_RESOLUTION_LABELS.items():
            options[value] = t(label) if label.startswith("see_through_") else label
        return options

    def _see_through_offload_policy_options(self) -> dict[str, str]:
        return {
            option: t(label_key)
            for option, label_key in self.SEE_THROUGH_OFFLOAD_POLICY_LABEL_KEYS.items()
        }

    def _sheet_music_preprocess_mode_options(self) -> dict[str, str]:
        return {
            option: t(label_key)
            for option, label_key in self.SHEET_MUSIC_PREPROCESS_MODE_LABEL_KEYS.items()
        }

    def _music_transcription_device_options(self) -> dict[str, str]:
        options = dict(self.MUSCRIPTOR_BASE_DEVICE_OPTIONS)
        if self.gpu_probe is not None:
            for device in self.gpu_probe.devices:
                options[f"cuda:{device.index}"] = f"CUDA {device.index} - {device.name}"
        selected = str(self.config.get("music_transcription_device") or DEFAULT_DEVICE)
        options.setdefault(selected, selected.upper())
        return options

    def _refresh_music_transcription_devices(self) -> None:
        selector = getattr(self, "music_transcription_device", None)
        if selector is None:
            return
        selector.options = self._music_transcription_device_options()
        selector.update()

    async def _refresh_music_gpu_probe(self) -> None:
        try:
            from module.gpu_profile import get_cached_gpu_probe

            self.gpu_probe = await asyncio.to_thread(get_cached_gpu_probe, refresh=True)
            self._refresh_music_transcription_devices()
        except Exception as exc:
            ui.notify(f"{t('detecting_gpu')}: {exc}", type="warning")

    @staticmethod
    def _set_config_value(config: Dict[str, Any], key: str, value: Any) -> None:
        config[key] = value

    def _on_music_input_mode_change(self, value: str) -> None:
        mode = str(value or "directory")
        self.config["music_transcription_input_mode"] = mode
        selector = getattr(self, "music_transcription_input", None)
        if selector is not None:
            selector.selection_type = "file" if mode == "file" else "dir"

    def _on_music_instrument_mode_change(self, value: str) -> None:
        mode = str(value or "auto")
        self.config["music_transcription_instrument_mode"] = mode
        if self._music_transcription_instrument_container is not None:
            self._music_transcription_instrument_container.set_visibility(mode == "specify")
        if mode == "specify":
            asyncio.create_task(self._ensure_music_instruments())

    def _on_music_decode_mode_change(self, value: str) -> None:
        mode = str(value or "greedy")
        self.config["music_transcription_decode_mode"] = mode
        if mode == "beam" and int(self.config["music_transcription_beam_size"]) < 2:
            self.config["music_transcription_beam_size"] = 2
            beam_slider = getattr(self, "music_transcription_beam_size", None)
            if beam_slider is not None and hasattr(beam_slider, "update_config"):
                beam_slider.update_config(new_value=2)
        if self._music_transcription_sampling_container is not None:
            self._music_transcription_sampling_container.set_visibility(mode == "sampling")
        if self._music_transcription_beam_container is not None:
            self._music_transcription_beam_container.set_visibility(mode == "beam")

    def _on_music_preview_mode_change(self, value: str) -> None:
        mode = str(value or "none")
        self.config["music_transcription_preview_mode"] = mode
        if self._music_transcription_preview_container is not None:
            self._music_transcription_preview_container.set_visibility(mode != "none")

    def _on_music_overwrite_change(self, enabled: bool) -> None:
        self.config["music_transcription_overwrite"] = bool(enabled)
        skip_toggle = getattr(self, "music_transcription_skip_completed_toggle", None)
        if skip_toggle is not None:
            skip_toggle.set_enabled(not enabled)

    async def _probe_music_instruments(self) -> tuple[str, ...]:
        if _MUSCRIPTOR_INSTRUMENT_CACHE:
            return next(reversed(_MUSCRIPTOR_INSTRUMENT_CACHE.values()))

        panel = self._ensure_execution_panel()
        runner_options: dict[str, Any] = {}
        execution_tabs = getattr(panel, "execution_tabs", None)
        if execution_tabs is not None:
            can_start = getattr(execution_tabs, "active_tab_can_start", None)
            if callable(can_start) and not can_start():
                raise RuntimeError(t("task_already_running"))
            if not await execution_tabs.ensure_active_tab_runtime_ready():
                raise RuntimeError(t("task_tab_not_ready"))
            tab_options = execution_tabs.runner_kwargs()
            if tab_options is None:
                raise RuntimeError(t("task_tab_not_ready"))
            for key in ("python_path", "venv_path"):
                if tab_options.get(key):
                    runner_options[key] = tab_options[key]

        from gui.utils.log_buffer import LogBuffer
        from gui.utils.process_runner import ProcessRunner

        probe_log = LogBuffer(maxlen=500)
        runner = ProcessRunner(log_buffer=probe_log)
        result = await runner.run_python_script(
            "module.muscriptor_tool.cli",
            ["list-instruments", "--format", "json"],
            native_console=False,
            **runner_options,
        )
        if result.return_code != 0:
            raise RuntimeError(result.message)
        return _parse_music_instrument_catalog(probe_log.get_all_lines())

    def _set_music_instrument_options(self, names: tuple[str, ...]) -> None:
        selector = getattr(self, "music_transcription_instruments", None)
        if selector is None:
            return
        selector.options = {name: name.replace("_", " ").title() for name in names}
        selector.update()

    async def _ensure_music_instruments(self) -> None:
        if self._music_transcription_instrument_loading:
            return
        self._music_transcription_instrument_loading = True
        selector = getattr(self, "music_transcription_instruments", None)
        if selector is not None:
            selector.set_enabled(False)
        try:
            names = await self._probe_music_instruments()
            self._set_music_instrument_options(names)
        except Exception as exc:
            ui.notify(
                f"{t('music_transcription_instruments_failed')}: {exc}",
                type="warning",
            )
        finally:
            if selector is not None:
                selector.set_enabled(True)
            self._music_transcription_instrument_loading = False

    def _refresh_see_through_summary(self) -> None:
        if self._see_through_summary_label is not None:
            self._see_through_summary_label.set_text(self._build_see_through_summary())
        if self._see_through_note_label is not None:
            note = self.see_through_recommendation.note or ""
            self._see_through_note_label.set_text(note)
            self._see_through_note_label.set_visibility(bool(note))
        self._refresh_see_through_gpu_details()

    def _toggle_see_through_gpu_details(self) -> None:
        self._see_through_gpu_details_open = not self._see_through_gpu_details_open
        self._refresh_see_through_gpu_details()

    def _refresh_see_through_gpu_details(self) -> None:
        header_container = self._see_through_summary_meta_container
        details_container = self._see_through_gpu_details_container
        if header_container is None or details_container is None:
            return

        lines = self._gpu_detail_lines()
        header_container.clear()
        details_container.clear()

        if not lines:
            header_container.set_visibility(False)
            details_container.set_visibility(False)
            return

        header_container.set_visibility(True)
        with header_container:
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("dns", size="16px").style(f"color: {COLORS['info']};")
                ui.label(f"{t('detected_gpus')} ({len(lines)})").classes("text-caption").style(
                    "color: var(--color-text-secondary);"
                )
                ui.button(
                    t("toggle"),
                    on_click=self._toggle_see_through_gpu_details,
                    icon="unfold_more",
                ).props('flat dense type="button"')

        details_container.set_visibility(self._see_through_gpu_details_open)
        if not self._see_through_gpu_details_open:
            return

        with details_container:
            for line in lines:
                ui.label(line).classes("text-caption").style("color: var(--color-text-secondary);")

    def _set_control_value(self, control: Any, value: Any) -> None:
        if control is None:
            return
        if hasattr(control, "set_value"):
            control.set_value(value)
            return
        control.value = value

    def _mark_see_through_user_edited(self, *_args) -> None:
        self._see_through_user_edited = True

    def _apply_see_through_recommendation(self, recommendation: Any) -> None:
        self.see_through_recommendation = recommendation
        self.config["see_through_resolution"] = recommendation.resolution
        self.config["see_through_resolution_depth"] = recommendation.resolution_depth
        self.config["see_through_dtype"] = recommendation.dtype
        self.config["see_through_quant_mode"] = recommendation.quant_mode
        self.config["see_through_group_offload"] = recommendation.group_offload
        self.config["see_through_offload_policy"] = recommendation.offload_policy
        self.config["see_through_repo_id_layerdiff"] = recommendation.repo_id_layerdiff
        self.config["see_through_repo_id_depth"] = recommendation.repo_id_depth

        if "see_through" in self._rendered_tool_tabs and not self._see_through_user_edited:
            self._set_control_value(getattr(self, "see_through_repo_id_layerdiff", None), recommendation.repo_id_layerdiff)
            self._set_control_value(getattr(self, "see_through_repo_id_depth", None), recommendation.repo_id_depth)
            self._set_control_value(getattr(self, "see_through_resolution_slider", None), recommendation.resolution)
            self._set_control_value(
                getattr(self, "see_through_resolution_depth", None),
                str(recommendation.resolution_depth),
            )
            self._set_control_value(getattr(self, "see_through_quant_mode", None), recommendation.quant_mode)
            self._set_control_value(getattr(self, "see_through_dtype", None), recommendation.dtype)
            self._set_control_value(getattr(self, "see_through_offload_policy", None), recommendation.offload_policy)
            group_offload_toggle = getattr(self, "see_through_group_offload_toggle", None)
            if group_offload_toggle is not None and hasattr(group_offload_toggle, "set_toggle_value"):
                group_offload_toggle.set_toggle_value(recommendation.group_offload)

        self._refresh_see_through_summary()

    async def _load_see_through_recommendation_async(self) -> None:
        try:
            probe, recommendation = await asyncio.to_thread(_probe_see_through_defaults)
        except Exception:
            return

        self.gpu_probe = probe
        self._refresh_music_transcription_devices()
        if not self._see_through_user_edited:
            self._apply_see_through_recommendation(recommendation)
        else:
            self.see_through_recommendation = recommendation
            self._refresh_see_through_summary()

    def _schedule_see_through_recommendation_refresh(self) -> None:
        if self._gpu_probe_scheduled:
            return
        self._gpu_probe_scheduled = True
        asyncio.create_task(self._load_see_through_recommendation_async())

    def _on_see_through_quant_mode_change(self, quant_mode: str) -> None:
        self._mark_see_through_user_edited()
        self.config["see_through_quant_mode"] = quant_mode
        repo_id_layerdiff, repo_id_depth = _resolve_see_through_repo_ids(quant_mode=quant_mode)
        self.config["see_through_repo_id_layerdiff"] = repo_id_layerdiff
        self.config["see_through_repo_id_depth"] = repo_id_depth
        if hasattr(self, "see_through_repo_id_layerdiff"):
            self.see_through_repo_id_layerdiff.value = repo_id_layerdiff
        if hasattr(self, "see_through_repo_id_depth"):
            self.see_through_repo_id_depth.value = repo_id_depth

    def _on_see_through_seed_change(self, value: str) -> None:
        try:
            self.config["see_through_seed"] = int(value)
        except (TypeError, ValueError):
            return

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            # 页面标题
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("construction", size="32px").style(f"color: {COLORS['primary']};")
                with ui.column().classes("gap-0"):
                    ui.label(t("tools_title")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(t("tools_desc")).classes("text-body2").style("color: var(--color-text-secondary);")

            # 使用标签页组织工具
            with ui.tabs().classes("w-full") as tabs:
                for tab_key, label_key, icon in self.TOOL_TABS:
                    ui.tab(tab_key, t(label_key), icon=icon)

            tabs.on_value_change(lambda e: self._handle_tool_tab_change(str(e.value)))

            with ui.tab_panels(tabs, value="watermark").classes("w-full"):
                for tab_key, _label_key, _icon in self.TOOL_TABS:
                    with ui.tab_panel(tab_key):
                        self._tool_tab_containers[tab_key] = ui.column().classes("w-full")

            self._ensure_tool_panel_rendered("watermark")

            # 共享执行面板：工具页的 Start/Stop 统一集中在这里。
            self._execution_panel_container = ui.column().classes("w-full")
            self._ensure_execution_panel()

    def _render_watermark_tool(self):
        """渲染水印检测工具"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("water_drop", size="22px").style(f"color: {COLORS['info']};")
                ui.label(t("watermark_detection")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            ui.label(t("watermark_desc")).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            # 输入路径
            self.watermark_input = create_path_selector(
                label=t("train_data_dir"), selection_type="dir", placeholder=t("input_path_placeholder")
            )

            # 模型选择 - 带图标的现代化下拉框
            self.watermark_model = styled_select(
                options=dict(zip(self.WATERMARK_MODELS, self.WATERMARK_MODELS)),
                value=self.WATERMARK_MODELS[0],
                label=t("watermark_model"),
                icon="water_drop",
                icon_color=COLORS["info"],
            )

            # 使用可编辑滑块
            with ui.row().classes("w-full gap-4 q-mt-md"):
                editable_slider(
                    label_key="batch_size",
                    value_ref=self.config,
                    value_key="watermark_batch_size",
                    min_val=1,
                    max_val=64,
                    step=1,
                    decimals=0,
                )

                editable_slider(
                    label_key="watermark_threshold",
                    value_ref=self.config,
                    value_key="watermark_thresh",
                    min_val=0.0,
                    max_val=1.0,
                    step=0.05,
                    decimals=2,
                )

            # 模型目录
            self.watermark_model_dir = ui.input(label=t("model_dir"), value="watermark_detection")
            self.watermark_model_dir.classes("modern-input w-full")

    def _render_preprocess_tool(self):
        """渲染图像预处理工具"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("image", size="22px").style(f"color: {COLORS['secondary']};")
                ui.label(t("preprocess")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            ui.label(t("preprocess_desc")).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            # 输入目录
            self.preprocess_input = create_path_selector(
                label=t("input_path"), selection_type="dir", placeholder=t("input_path_placeholder")
            )

            # 对齐输入目录
            self.preprocess_align = create_path_selector(
                label=t("align_input_dir"),
                selection_type="dir",
                placeholder=t("optional_pair_align_placeholder"),
            )

            # 使用可编辑滑块
            with ui.row().classes("w-full gap-4 q-mt-md"):
                editable_slider(
                    label_key="max_long_edge",
                    value_ref=self.config,
                    value_key="max_long_edge",
                    min_val=0,
                    max_val=8192,
                    step=64,
                    decimals=0,
                )

                editable_slider(
                    label_key="max_short_edge",
                    value_ref=self.config,
                    value_key="max_short_edge",
                    min_val=0,
                    max_val=8192,
                    step=64,
                    decimals=0,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                editable_slider(
                    label_key="max_pixels",
                    value_ref=self.config,
                    value_key="max_pixels",
                    min_val=0,
                    max_val=67108864,
                    step=262144,
                    decimals=0,
                )

                editable_slider(
                    label_key="workers",
                    value_ref=self.config,
                    value_key="preprocess_workers",
                    min_val=1,
                    max_val=32,
                    step=1,
                    decimals=0,
                )

            # 变换类型 - 带图标的现代化下拉框
            self.transform_type = styled_select(
                options=dict(zip(self.TRANSFORM_TYPES, self.TRANSFORM_TYPES)),
                value="auto",
                label=t("transform_type"),
                icon="transform",
                icon_color=COLORS["primary"],
            )

            # 背景颜色
            self.bg_color = ui.input(label=t("bg_color"), value="255 255 255")
            self.bg_color.classes("modern-input w-full")

            # 开关选项
            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("recursive", self.config, "preprocess_recursive")
                toggle_switch("crop_transparent", self.config, "crop_transparent")

    def _render_reward_tool(self):
        """渲染图像评分工具"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("stars", size="22px").style(f"color: {COLORS['warning']};")
                ui.label(t("reward_model")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            ui.label(t("reward_desc")).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            # 输入路径
            self.reward_input = create_path_selector(
                label=t("train_data_dir"), selection_type="dir", placeholder=t("input_path_placeholder")
            )

            # 模型选择 - 带图标的现代化下拉框
            self.reward_model = styled_select(
                options=dict(zip(self.REWARD_MODELS, self.REWARD_MODELS)),
                value=self.REWARD_MODELS[0],
                label=t("reward_model_select"),
                icon="stars",
                icon_color=COLORS["warning"],
            )

            # 使用可编辑滑块
            editable_slider(
                label_key="batch_size",
                value_ref=self.config,
                value_key="reward_batch_size",
                min_val=1,
                max_val=32,
                step=1,
                decimals=0,
            )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                # 设备 - 带图标的现代化下拉框
                self.reward_device = styled_select(
                    options=_reward_device_options(),
                    value="auto",
                    label=t("device"),
                    icon="memory",
                    icon_color=COLORS["primary"],
                    flex=1,
                )

                # 数据类型 - 带图标的现代化下拉框
                self.reward_dtype = styled_select(
                    options={"auto": "Auto", "float16": "FP16", "float32": "FP32", "bfloat16": "BF16"},
                    value="auto",
                    label=t("dtype"),
                    icon="data_object",
                    icon_color=COLORS["primary"],
                    flex=1,
                )

    def _render_audio_separator_tool(self):
        """渲染音频分轨工具"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("graphic_eq", size="22px").style(f"color: {COLORS['primary']};")
                ui.label(t("audio_separator")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            ui.label(t("audio_separator_desc")).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            self.audio_separator_input = create_path_selector(
                label=t("input_path"),
                selection_type="dir",
            )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.audio_separator_output_format = styled_select(
                    options={"wav": "WAV", "flac": "FLAC", "mp3": "MP3"},
                    value="wav",
                    label=t("output_format"),
                    icon="audiotrack",
                    icon_color=COLORS["primary"],
                    flex=1,
                )

                editable_slider(
                    label_key="segment_size",
                    value_ref=self.config,
                    value_key="audio_separator_segment_size",
                    min_val=2,
                    max_val=2048,
                    step=1,
                    decimals=0,
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                editable_slider(
                    label_key="overlap",
                    value_ref=self.config,
                    value_key="audio_separator_overlap",
                    min_val=1,
                    max_val=50,
                    step=1,
                    decimals=0,
                )

                editable_slider(
                    label_key="batch_size",
                    value_ref=self.config,
                    value_key="audio_separator_batch_size",
                    min_val=1,
                    max_val=32,
                    step=1,
                    decimals=0,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("overwrite", self.config, "audio_separator_overwrite")
                toggle_switch("harmony_separation", self.config, "audio_separator_harmony_separation")
                toggle_switch(
                    "vocal_midi",
                    self.config,
                    "audio_separator_vocal_midi",
                    on_change=self._on_audio_separator_vocal_midi_toggle,
                )

            self._audio_separator_vocal_midi_container = ui.column().classes("w-full q-mt-sm")
            self._audio_separator_vocal_midi_container.set_visibility(self.config["audio_separator_vocal_midi"])
            with self._audio_separator_vocal_midi_container:
                with (
                    ui.card()
                    .classes("w-full q-pa-md")
                    .style("background: var(--ql-inset-bg); border: 1px solid var(--ql-inset-border);")
                ):
                    with ui.row().classes("w-full items-center gap-2 q-mb-sm"):
                        ui.icon("piano", size="18px").style(f"color: {COLORS['secondary']};")
                        (
                            ui.label(t("vocal_midi_settings"))
                            .classes("text-body1 text-weight-medium")
                            .style("color: var(--color-text);")
                        )

                    with ui.row().classes("w-full gap-4 q-mt-sm"):
                        self.audio_separator_vocal_midi_model = styled_select(
                            options=self.VOCAL_MIDI_MODELS,
                            value=DEFAULT_GAME_MODEL_REPO_ID,
                            label=t("vocal_midi_model"),
                            icon="memory",
                            icon_color=COLORS["secondary"],
                            flex=1,
                        )
                        self.audio_separator_vocal_midi_language = styled_select(
                            options=self.VOCAL_MIDI_LANGUAGES,
                            value="auto",
                            label=t("vocal_midi_language"),
                            icon="language",
                            icon_color=COLORS["info"],
                            flex=1,
                        )

                    with ui.column().classes("w-full q-mt-sm"):
                        with ui.row().classes("items-center gap-2 q-mb-xs"):
                            ui.icon("queue_music", size="18px").style(f"color: {COLORS['secondary']};")
                            (
                                ui.label(t("vocal_midi_output_formats"))
                                .classes("text-caption text-weight-medium")
                                .style("color: var(--color-text-secondary);")
                            )
                        self.audio_separator_vocal_midi_output_formats = ui.select(
                            options=self.VOCAL_MIDI_OUTPUT_FORMATS,
                            value=[DEFAULT_VOCAL_MIDI_OUTPUT_FORMATS],
                            multiple=True,
                        ).classes("w-full modern-select force-light-bg")
                        self.audio_separator_vocal_midi_output_formats.props('dense use-chips stack-label input-debounce="0"')

                    with ui.row().classes("w-full gap-4 q-mt-md"):
                        editable_slider(
                            label_key="batch_size",
                            value_ref=self.config,
                            value_key="audio_separator_vocal_midi_batch_size",
                            min_val=1,
                            max_val=32,
                            step=1,
                            decimals=0,
                        )
                        editable_slider(
                            label_key="seg_threshold",
                            value_ref=self.config,
                            value_key="audio_separator_vocal_midi_seg_threshold",
                            min_val=0.05,
                            max_val=0.95,
                            step=0.05,
                            decimals=2,
                        )

                    with ui.row().classes("w-full gap-4 q-mt-md"):
                        editable_slider(
                            label_key="seg_radius",
                            value_ref=self.config,
                            value_key="audio_separator_vocal_midi_seg_radius",
                            min_val=0.01,
                            max_val=0.20,
                            step=0.01,
                            decimals=2,
                        )
                        editable_slider(
                            label_key="t0",
                            value_ref=self.config,
                            value_key="audio_separator_vocal_midi_t0",
                            min_val=0.0,
                            max_val=0.95,
                            step=0.05,
                            decimals=2,
                        )

                    with ui.row().classes("w-full gap-4 q-mt-md"):
                        editable_slider(
                            label_key="nsteps",
                            value_ref=self.config,
                            value_key="audio_separator_vocal_midi_nsteps",
                            min_val=1,
                            max_val=16,
                            step=1,
                            decimals=0,
                        )
                        editable_slider(
                            label_key="est_threshold",
                            value_ref=self.config,
                            value_key="audio_separator_vocal_midi_est_threshold",
                            min_val=0.05,
                            max_val=0.95,
                            step=0.05,
                            decimals=2,
                        )

    def _render_music_transcription_tool(self):
        """Render the official MuScriptor batch transcription tool."""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-sm"):
                ui.icon("piano", size="22px").style(f"color: {COLORS['secondary']};")
                ui.label(t("music_transcription")).classes("text-h6 text-weight-bold").style(
                    "color: var(--color-text);"
                )

            ui.label(t("music_transcription_desc")).classes("text-body2 q-mb-md").style(
                "color: var(--color-text-secondary);"
            )

            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.label(t("music_transcription_input_mode")).classes("text-caption text-weight-medium")
                self.music_transcription_input_mode = ui.toggle(
                    {
                        "directory": t("music_transcription_input_directory"),
                        "file": t("music_transcription_input_file"),
                    },
                    value=self.config["music_transcription_input_mode"],
                    on_change=lambda event: self._on_music_input_mode_change(event.value),
                ).props("dense no-caps")

            self.music_transcription_input = create_path_selector(
                label=t("input_path"),
                selection_type="dir",
                file_filter=".wav .flac .mp3 .m4a .ogg .aac",
                placeholder=t("input_path_placeholder"),
            )
            self.music_transcription_output = create_path_selector(
                label=t("output_dir"),
                default_path=DEFAULT_MUSCRIPTOR_OUTPUT_DIR,
                selection_type="dir",
                placeholder=t("path_placeholder"),
            )

            ui.separator().classes("q-my-md")

            with ui.row().classes("w-full gap-4"):
                self.music_transcription_model = styled_select(
                    options=self.MUSCRIPTOR_MODEL_OPTIONS,
                    value=self.config["music_transcription_model"],
                    label=t("music_transcription_model"),
                    icon="model_training",
                    icon_color=COLORS["primary"],
                    searchable=False,
                    on_change=lambda value: self._set_config_value(
                        self.config,
                        "music_transcription_model",
                        value,
                    ),
                    flex=1,
                )
                self.music_transcription_device = styled_select(
                    options=self._music_transcription_device_options(),
                    value=self.config["music_transcription_device"],
                    label=t("device"),
                    icon="memory",
                    icon_color=COLORS["info"],
                    new_value_mode="add-unique",
                    on_change=lambda value: self._set_config_value(
                        self.config,
                        "music_transcription_device",
                        value,
                    ),
                    flex=1,
                )
                editable_slider(
                    label_key="music_transcription_chunk_batch_size",
                    value_ref=self.config,
                    value_key="music_transcription_batch_size",
                    min_val=0,
                    max_val=16,
                    step=1,
                    decimals=0,
                    flex=1,
                )

            with ui.row().classes("w-full items-center justify-between q-mt-xs"):
                ui.link(
                    t("music_transcription_license"),
                    target="https://huggingface.co/MuScriptor/muscriptor-medium",
                    new_tab=True,
                ).classes("text-caption").style(f"color: {COLORS['info']};")
                ui.button(
                    icon="refresh",
                    on_click=lambda: asyncio.create_task(self._refresh_music_gpu_probe()),
                ).props('flat dense round type="button"').tooltip(t("detected_gpus"))

            with ui.row().classes("w-full items-center gap-3 q-mt-md"):
                ui.label(t("music_transcription_instrument_mode")).classes("text-caption text-weight-medium")
                self.music_transcription_instrument_mode = ui.toggle(
                    {
                        "auto": t("music_transcription_instrument_auto"),
                        "specify": t("music_transcription_instrument_specify"),
                    },
                    value=self.config["music_transcription_instrument_mode"],
                    on_change=lambda event: self._on_music_instrument_mode_change(event.value),
                ).props("dense no-caps")

            cached_names = next(reversed(_MUSCRIPTOR_INSTRUMENT_CACHE.values()), ())
            self._music_transcription_instrument_container = ui.column().classes("w-full q-mt-sm")
            self._music_transcription_instrument_container.set_visibility(
                self.config["music_transcription_instrument_mode"] == "specify"
            )
            with self._music_transcription_instrument_container:
                self.music_transcription_instruments = ui.select(
                    options={name: name.replace("_", " ").title() for name in cached_names},
                    value=self.config["music_transcription_instruments"],
                    multiple=True,
                    label=t("music_transcription_instruments"),
                    on_change=lambda event: self._set_config_value(
                        self.config,
                        "music_transcription_instruments",
                        list(event.value or []),
                    ),
                ).classes("w-full modern-select force-light-bg")
                self.music_transcription_instruments.props(
                    'dense use-input use-chips input-debounce="0" dropdown-icon="search"'
                )

            ui.separator().classes("q-my-md")

            with ui.row().classes("w-full gap-4"):
                self.music_transcription_decode_mode = styled_select(
                    options={
                        DecodingMode.GREEDY.value: t("music_transcription_decode_greedy"),
                        DecodingMode.SAMPLING.value: t("music_transcription_decode_sampling"),
                        DecodingMode.BEAM.value: t("music_transcription_decode_beam"),
                    },
                    value=self.config["music_transcription_decode_mode"],
                    label=t("music_transcription_decode_mode"),
                    icon="account_tree",
                    icon_color=COLORS["secondary"],
                    searchable=False,
                    on_change=self._on_music_decode_mode_change,
                    flex=1,
                )
                editable_slider(
                    label_key="music_transcription_cfg_coef",
                    value_ref=self.config,
                    value_key="music_transcription_cfg_coef",
                    min_val=0.0,
                    max_val=5.0,
                    step=0.1,
                    decimals=1,
                    flex=1,
                )

            self._music_transcription_sampling_container = ui.column().classes("w-full q-mt-sm")
            self._music_transcription_sampling_container.set_visibility(
                self.config["music_transcription_decode_mode"] == "sampling"
            )
            with self._music_transcription_sampling_container:
                editable_slider(
                    label_key="music_transcription_temperature",
                    value_ref=self.config,
                    value_key="music_transcription_temperature",
                    min_val=0.1,
                    max_val=2.0,
                    step=0.1,
                    decimals=1,
                )

            self._music_transcription_beam_container = ui.column().classes("w-full q-mt-sm")
            self._music_transcription_beam_container.set_visibility(
                self.config["music_transcription_decode_mode"] == "beam"
            )
            with self._music_transcription_beam_container:
                self.music_transcription_beam_size = editable_slider(
                    label_key="music_transcription_beam_size",
                    value_ref=self.config,
                    value_key="music_transcription_beam_size",
                    min_val=2,
                    max_val=16,
                    step=1,
                    decimals=0,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch(
                    "music_transcription_strict_eos",
                    self.config,
                    "music_transcription_strict_eos",
                )
                toggle_switch(
                    "music_transcription_notes",
                    self.config,
                    "music_transcription_notes",
                )

            ui.separator().classes("q-my-md")

            self.music_transcription_output_formats = ui.select(
                options={
                    OutputFormat.MIDI.value: "MIDI",
                    OutputFormat.JSON.value: "JSON",
                    OutputFormat.JSONL.value: "JSONL",
                },
                value=self.config["music_transcription_output_formats"],
                multiple=True,
                label=t("music_transcription_symbolic_outputs"),
                on_change=lambda event: self._set_config_value(
                    self.config,
                    "music_transcription_output_formats",
                    list(event.value or []),
                ),
            ).classes("w-full modern-select force-light-bg")
            self.music_transcription_output_formats.props("dense use-chips")

            with ui.row().classes("w-full items-center gap-3 q-mt-md"):
                ui.label(t("music_transcription_preview")).classes("text-caption text-weight-medium")
                self.music_transcription_preview_mode = ui.toggle(
                    {
                        "none": t("music_transcription_preview_off"),
                        PreviewContent.MIDI.value: t("music_transcription_preview_midi"),
                        PreviewContent.COMPARISON.value: t("music_transcription_preview_comparison"),
                    },
                    value=self.config["music_transcription_preview_mode"],
                    on_change=lambda event: self._on_music_preview_mode_change(event.value),
                ).props("dense no-caps")

            self._music_transcription_preview_container = ui.column().classes("w-full q-mt-sm")
            self._music_transcription_preview_container.set_visibility(
                self.config["music_transcription_preview_mode"] != "none"
            )
            with self._music_transcription_preview_container:
                self.music_transcription_preview_format = styled_select(
                    options={PreviewFormat.WAV.value: "WAV", PreviewFormat.MP3.value: "MP3"},
                    value=self.config["music_transcription_preview_format"],
                    label=t("music_transcription_preview_format"),
                    icon="headphones",
                    icon_color=COLORS["info"],
                    searchable=False,
                    on_change=lambda value: self._set_config_value(
                        self.config,
                        "music_transcription_preview_format",
                        value,
                    ),
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch(
                    "recursive",
                    self.config,
                    "music_transcription_recursive",
                )
                self.music_transcription_skip_completed_toggle = toggle_switch(
                    "skip_completed",
                    self.config,
                    "music_transcription_skip_completed",
                )
                self.music_transcription_overwrite_toggle = toggle_switch(
                    "overwrite",
                    self.config,
                    "music_transcription_overwrite",
                    on_change=self._on_music_overwrite_change,
                )
                toggle_switch(
                    "music_transcription_fail_fast",
                    self.config,
                    "music_transcription_fail_fast",
                )
            self.music_transcription_skip_completed_toggle.set_enabled(
                not self.config["music_transcription_overwrite"]
            )

    def _render_sheet_music_tool(self):
        """渲染乐谱扫描 embedding 工具"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("library_music", size="22px").style(f"color: {COLORS['secondary']};")
                ui.label(t("sheet_music")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            ui.label(t("sheet_music_desc")).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            self.sheet_music_input = create_path_selector(
                label=t("input_path"),
                selection_type="file",
                file_filter=".png .jpg .jpeg .webp .bmp .tif .tiff .pdf",
                placeholder=t("input_path_placeholder"),
            )
            self.sheet_music_output = create_path_selector(
                label=t("output_dir"),
                default_path=DEFAULT_SHEET_MUSIC_OUTPUT_DIR,
                selection_type="dir",
                placeholder=t("path_placeholder"),
            )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.sheet_music_repo_id = styled_select(
                    options={DEFAULT_SHEET_MUSIC_REPO_ID: DEFAULT_SHEET_MUSIC_REPO_ID},
                    value=DEFAULT_SHEET_MUSIC_REPO_ID,
                    label=t("sheet_music_repo_id"),
                    icon="cloud_download",
                    icon_color=COLORS["primary"],
                    new_value_mode="add-unique",
                    flex=1,
                )
                self.sheet_music_model_dir = styled_input(
                    value=DEFAULT_SHEET_MUSIC_MODEL_DIR,
                    label=t("sheet_music_model_dir"),
                    icon="folder",
                    icon_color=COLORS["info"],
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.sheet_music_preprocess_mode = styled_select(
                    options=self._sheet_music_preprocess_mode_options(),
                    value="page_resize",
                    label=t("sheet_music_preprocess_mode"),
                    icon="crop",
                    icon_color=COLORS["secondary"],
                    searchable=False,
                    flex=1,
                )
                editable_slider(
                    label_key="batch_size",
                    value_ref=self.config,
                    value_key="sheet_music_batch_size",
                    min_val=1,
                    max_val=16,
                    step=1,
                    decimals=0,
                    flex=1,
                )
                editable_slider(
                    label_key="sheet_music_pdf_dpi",
                    label_default="PDF DPI",
                    value_ref=self.config,
                    value_key="sheet_music_pdf_dpi",
                    min_val=72,
                    max_val=300,
                    step=12,
                    decimals=0,
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("recursive", self.config, "sheet_music_recursive")
                toggle_switch("skip_completed", self.config, "sheet_music_skip_completed")

            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("overwrite", self.config, "sheet_music_overwrite")
                toggle_switch("sheet_music_force_download", self.config, "sheet_music_force_download")

    def _render_translate_tool(self):
        """渲染文本/文档翻译工具"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("translate", size="22px").style(f"color: {COLORS['primary']};")
                ui.label(t("translate")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            ui.label(t("translate_desc")).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            # 输入路径（目录或 .lance 文件）
            self.translate_input = create_path_selector(
                label=t("translate_input_path"), selection_type="dir", placeholder=t("translate_input_placeholder")
            )

            # 输出数据集名称
            self.translate_output_name = ui.input(label=t("output_name"), value="dataset")
            self.translate_output_name.classes("modern-input w-full")

            with ui.row().classes("w-full gap-4 q-mt-md"):
                # 翻译模型
                self.translate_model = styled_select(
                    options=dict(zip(self.TRANSLATE_MODELS, self.TRANSLATE_MODELS)),
                    value=self.TRANSLATE_MODELS[0],
                    label=t("translate_model"),
                    icon="smart_toy",
                    icon_color=COLORS["primary"],
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                # 源语言
                self.translate_source_lang = styled_select(
                    options=self.TRANSLATE_LANGUAGES,
                    value="auto",
                    label=t("translate_source_lang"),
                    icon="language",
                    icon_color=COLORS["info"],
                    flex=1,
                )

                # 目标语言
                self.translate_target_lang = styled_select(
                    options={k: v for k, v in self.TRANSLATE_LANGUAGES.items() if k != "auto"},
                    value="zh_cn",
                    label=t("translate_target_lang"),
                    icon="language",
                    icon_color=COLORS["secondary"],
                    flex=1,
                )

            # 高级参数
            with ui.expansion(t("translate_advanced"), icon="tune").classes("w-full q-mt-md"):
                with ui.row().classes("w-full gap-4 q-mt-sm"):
                    editable_slider(
                        label_key="translate_max_chars",
                        value_ref=self.config,
                        value_key="translate_max_chars",
                        min_val=500,
                        max_val=8000,
                        step=100,
                        decimals=0,
                    )

                    editable_slider(
                        label_key="translate_context_chars",
                        value_ref=self.config,
                        value_key="translate_context_chars",
                        min_val=0,
                        max_val=1000,
                        step=50,
                        decimals=0,
                    )

                with ui.row().classes("w-full gap-4 q-mt-md"):
                    editable_slider(
                        label_key="translate_max_new_tokens",
                        value_ref=self.config,
                        value_key="translate_max_new_tokens",
                        min_val=256,
                        max_val=8192,
                        step=256,
                        decimals=0,
                    )

                    editable_slider(
                        label_key="translate_temperature",
                        value_ref=self.config,
                        value_key="translate_temperature",
                        min_val=0.0,
                        max_val=1.0,
                        step=0.05,
                        decimals=2,
                    )

                # 术语表文件
                self.translate_glossary = create_path_selector(
                    label=t("translate_glossary_file"),
                    selection_type="file",
                    placeholder=t("translate_glossary_placeholder"),
                )

                # 开关选项
                with ui.row().classes("w-full gap-4 q-mt-md"):
                    toggle_switch("translate_skip_normalize", self.config, "translate_skip_normalize")
                    toggle_switch("translate_normalize_only", self.config, "translate_normalize_only")

                with ui.row().classes("w-full gap-4 q-mt-md"):
                    toggle_switch("translate_no_export", self.config, "translate_no_export")
                    toggle_switch("translate_force_reimport", self.config, "translate_force_reimport")

    def _render_see_through_tool(self):
        """渲染 see-through 工具"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("layers", size="22px").style(f"color: {COLORS['secondary']};")
                ui.label(t("see_through")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            ui.label(t("see_through_desc")).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            self.see_through_input = create_path_selector(
                label=t("input_path"),
                selection_type="dir",
                placeholder=t("input_path_placeholder"),
            )
            self.see_through_output = create_path_selector(
                label=t("output_dir"),
                selection_type="dir",
                placeholder=t("path_placeholder"),
            )

            with ui.row().classes("w-full items-center gap-2 q-mt-sm q-mb-sm"):
                ui.icon("memory", size="18px").style(f"color: {COLORS['info']};")
                self._see_through_summary_label = (
                    ui.label(self._build_see_through_summary())
                    .classes("text-caption")
                    .style("color: var(--color-text-secondary);")
                )

            self._see_through_note_label = (
                ui.label(self.see_through_recommendation.note or "")
                .classes("text-caption")
                .style(f"color: {COLORS['warning']};")
            )
            self._see_through_note_label.set_visibility(bool(self.see_through_recommendation.note))
            self._see_through_summary_meta_container = ui.column().classes("w-full gap-1")
            self._see_through_gpu_details_container = ui.column().classes("w-full gap-1")
            self._refresh_see_through_gpu_details()

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.see_through_repo_id_layerdiff = styled_select(
                    options=self.SEE_THROUGH_LAYERDIFF_REPOS,
                    value=self.config["see_through_repo_id_layerdiff"],
                    label=t("repo_id_layerdiff"),
                    icon="layers",
                    icon_color=COLORS["secondary"],
                    new_value_mode="add-unique",
                    on_change=lambda value: (
                        self._mark_see_through_user_edited(),
                        self.config.__setitem__("see_through_repo_id_layerdiff", value),
                    ),
                    flex=1,
                )
                self.see_through_repo_id_depth = styled_select(
                    options=self.SEE_THROUGH_DEPTH_REPOS,
                    value=self.config["see_through_repo_id_depth"],
                    label=t("repo_id_depth"),
                    icon="blur_on",
                    icon_color=COLORS["info"],
                    new_value_mode="add-unique",
                    on_change=lambda value: (
                        self._mark_see_through_user_edited(),
                        self.config.__setitem__("see_through_repo_id_depth", value),
                    ),
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.see_through_resolution_slider = editable_slider(
                    label_key="resolution",
                    value_ref=self.config,
                    value_key="see_through_resolution",
                    min_val=768,
                    max_val=1280,
                    step=64,
                    decimals=0,
                    on_change=self._mark_see_through_user_edited,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.see_through_resolution_depth = styled_select(
                    options=self._see_through_depth_resolution_options(),
                    value=str(self.config["see_through_resolution_depth"]),
                    label=t("resolution_depth"),
                    icon="straighten",
                    icon_color=COLORS["info"],
                    new_value_mode="add-unique",
                    on_change=lambda value: (
                        self._mark_see_through_user_edited(),
                        self.config.__setitem__("see_through_resolution_depth", int(value)),
                    ),
                    flex=1,
                )
                self.see_through_quant_mode = styled_select(
                    options=self._see_through_quant_mode_options(),
                    value=self.config["see_through_quant_mode"],
                    label=t("quant_mode"),
                    icon="tune",
                    icon_color=COLORS["secondary"],
                    on_change=self._on_see_through_quant_mode_change,
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                editable_slider(
                    label_key="inference_steps_depth",
                    label_default="Depth Steps",
                    value_ref=self.config,
                    value_key="see_through_inference_steps_depth",
                    min_val=-1,
                    max_val=20,
                    step=1,
                    decimals=0,
                )
                self.see_through_seed = styled_input(
                    value=str(self.config["see_through_seed"]),
                    label=t("seed", "Seed"),
                    icon="casino",
                    icon_color=COLORS["warning"],
                    on_change=self._on_see_through_seed_change,
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.see_through_dtype = styled_select(
                    options=self.SEE_THROUGH_DTYPES,
                    value=self.config["see_through_dtype"],
                    label=t("dtype"),
                    icon="data_object",
                    icon_color=COLORS["primary"],
                    on_change=lambda value: (
                        self._mark_see_through_user_edited(),
                        self.config.__setitem__("see_through_dtype", value),
                    ),
                    flex=1,
                )
                self.see_through_offload_policy = styled_select(
                    options=self._see_through_offload_policy_options(),
                    value=self.config["see_through_offload_policy"],
                    label=t("offload_policy"),
                    icon="memory",
                    icon_color=COLORS["primary"],
                    on_change=lambda value: (
                        self._mark_see_through_user_edited(),
                        self.config.__setitem__("see_through_offload_policy", value),
                    ),
                    flex=1,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.see_through_group_offload_toggle = toggle_switch(
                    "group_offload",
                    self.config,
                    "see_through_group_offload",
                    on_change=self._mark_see_through_user_edited,
                )

            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("skip_completed", self.config, "see_through_skip_completed")

            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("save_to_psd", self.config, "see_through_save_to_psd")
                toggle_switch("tblr_split", self.config, "see_through_tblr_split")

    def _on_audio_separator_vocal_midi_toggle(self, enabled: bool) -> None:
        if hasattr(self, "_audio_separator_vocal_midi_container"):
            self._audio_separator_vocal_midi_container.set_visibility(enabled)

    def _build_music_transcription_args(self) -> list[str]:
        input_path = str(getattr(getattr(self, "music_transcription_input", None), "value", "") or "").strip()
        source = Path(input_path).expanduser() if input_path else None
        if source is None or not source.exists():
            raise ValueError(t("select_valid_input"))
        input_mode = str(self.config["music_transcription_input_mode"])
        if (input_mode == "file" and not source.is_file()) or (input_mode == "directory" and not source.is_dir()):
            raise ValueError(t("music_transcription_input_mode_mismatch"))

        output_dir = str(getattr(getattr(self, "music_transcription_output", None), "value", "") or "").strip()
        if not output_dir:
            raise ValueError(t("music_transcription_output_required"))
        output_path = Path(output_dir).expanduser()
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"{t('music_transcription_output_required')}: {exc}") from exc
        if not output_path.is_dir():
            raise ValueError(t("music_transcription_output_required"))

        formats = tuple(OutputFormat(value) for value in self.config["music_transcription_output_formats"])
        if not formats:
            raise ValueError(t("music_transcription_output_required"))

        instrument_mode = str(self.config["music_transcription_instrument_mode"])
        instruments = tuple(
            str(value).strip()
            for value in self.config["music_transcription_instruments"]
            if str(value).strip()
        )
        if instrument_mode == "auto":
            instruments = ()
        elif instrument_mode == "specify" and not instruments:
            raise ValueError(t("music_transcription_instruments_required"))

        decode_mode = DecodingMode(self.config["music_transcription_decode_mode"])
        temperature = (
            float(self.config["music_transcription_temperature"])
            if decode_mode is DecodingMode.SAMPLING
            else 1.0
        )
        beam_size = int(self.config["music_transcription_beam_size"]) if decode_mode is DecodingMode.BEAM else None
        raw_batch_size = int(self.config["music_transcription_batch_size"])
        if raw_batch_size < 0:
            raise ValueError(t("music_transcription_invalid_config"))

        transcription = TranscriptionOptions.from_batch_cli(
            model=ModelVariant(self.config["music_transcription_model"]),
            device=str(self.config["music_transcription_device"]),
            batch_size=raw_batch_size or None,
            decode_mode=decode_mode,
            temperature=temperature,
            cfg_coef=float(self.config["music_transcription_cfg_coef"]),
            strict_eos=bool(self.config["music_transcription_strict_eos"]),
            beam_size=beam_size,
            instruments=instruments,
            print_notes=bool(self.config["music_transcription_notes"]),
        )

        preview_mode = str(self.config["music_transcription_preview_mode"])
        preview = None
        if preview_mode != "none":
            preview = PreviewRequest(
                content=PreviewContent(preview_mode),
                format=PreviewFormat(self.config["music_transcription_preview_format"]),
            )

        args = [
            "batch",
            str(source),
            f"--output-dir={output_dir}",
            f"--model={transcription.model.value}",
            f"--device={transcription.device}",
        ]
        if transcription.batch_size is not None:
            args.append(f"--batch-size={transcription.batch_size}")
        if transcription.instruments:
            args.append(f"--instruments={','.join(transcription.instruments)}")
        args.append(f"--decode-mode={transcription.decode_mode.value}")
        if transcription.decode_mode is DecodingMode.SAMPLING:
            args.append(f"--temperature={transcription.temperature}")
        if transcription.decode_mode is DecodingMode.BEAM:
            args.append(f"--beam-size={transcription.beam_size}")
        args.append(f"--cfg-coef={transcription.cfg_coef}")
        if transcription.strict_eos:
            args.append("--strict-eos")
        if transcription.print_notes:
            args.append("--notes")
        args.extend(f"--format={output_format.value}" for output_format in formats)
        if preview is not None:
            args.extend(
                (
                    f"--preview-mode={preview.content.value}",
                    f"--preview-format={preview.format.value}",
                )
            )
        args.append("--recursive" if self.config["music_transcription_recursive"] else "--no-recursive")
        args.append(
            "--skip-completed"
            if self.config["music_transcription_skip_completed"]
            else "--no-skip-completed"
        )
        if self.config["music_transcription_overwrite"]:
            args.append("--overwrite")
        if self.config["music_transcription_fail_fast"]:
            args.append("--fail-fast")
        return args

    async def _start_music_transcription(self):
        try:
            args = self._build_music_transcription_args()
        except (TypeError, ValueError) as exc:
            ui.notify(str(exc) or t("music_transcription_invalid_config"), type="warning")
            return

        if (
            self.config["music_transcription_model"] == ModelVariant.LARGE.value
            and self.config["music_transcription_device"] == "cpu"
        ):
            ui.notify(t("music_transcription_large_cpu_warning"), type="warning")

        def pre_log(lv):
            lv.info(t("log_start_music_transcription"))
            lv.info(f"{t('log_input_path')}: {args[1]}")
            lv.info(f"{t('log_params')}: {args}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "module.muscriptor_tool.cli",
            args,
            name=t("job_name_music_transcription"),
            pre_log=pre_log,
            on_success=lambda result: ui.notify(t("music_transcription_success"), type="positive"),
            on_failure=lambda result: ui.notify(t("music_transcription_failed"), type="negative"),
        )

    async def _start_sheet_music(self):
        """开始乐谱扫描 embedding 提取"""
        input_path = getattr(getattr(self, "sheet_music_input", None), "value", "")
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        output_path = str(getattr(getattr(self, "sheet_music_output", None), "value", "") or "").strip()
        repo_id = str(
            getattr(getattr(self, "sheet_music_repo_id", None), "value", DEFAULT_SHEET_MUSIC_REPO_ID)
            or DEFAULT_SHEET_MUSIC_REPO_ID
        ).strip()
        model_dir = str(
            getattr(getattr(self, "sheet_music_model_dir", None), "value", DEFAULT_SHEET_MUSIC_MODEL_DIR)
            or DEFAULT_SHEET_MUSIC_MODEL_DIR
        ).strip()
        preprocess_mode = str(
            getattr(getattr(self, "sheet_music_preprocess_mode", None), "value", "page_resize") or "page_resize"
        ).strip()

        args = [input_path]
        if output_path:
            args.append(f"--output_dir={output_path}")
        args.append(f"--repo_id={repo_id}")
        args.append(f"--model_dir={model_dir}")
        args.append(f"--batch_size={int(self.config['sheet_music_batch_size'])}")
        args.append(f"--pdf_dpi={int(self.config['sheet_music_pdf_dpi'])}")
        args.append(f"--preprocess_mode={preprocess_mode}")
        args.append("--recursive" if self.config["sheet_music_recursive"] else "--no-recursive")
        args.append("--skip_completed" if self.config["sheet_music_skip_completed"] else "--no-skip_completed")
        if self.config["sheet_music_overwrite"]:
            args.append("--overwrite")
        if self.config["sheet_music_force_download"]:
            args.append("--force_download")

        def pre_log(lv):
            lv.info(t("log_start_sheet_music"))
            lv.info(f"{t('log_input_path')}: {input_path}")
            if output_path:
                lv.info(f"{t('log_output_dir')}: {output_path}")
            lv.info(f"{t('log_model')}: {repo_id}")
            lv.info(f"{t('log_params')}: {args}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "module.sheet_music_musvit",
            args,
            name=t("job_name_sheet_music"),
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("sheet_music_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("sheet_music_failed"), type="negative"),
        )

    async def _start_see_through(self):
        """开始 see-through 批处理"""
        input_path = self.see_through_input.value
        output_path = self.see_through_output.value
        resolution_depth_value = getattr(
            getattr(self, "see_through_resolution_depth", None),
            "value",
            str(self.config["see_through_resolution_depth"]),
        )
        quant_mode_value = getattr(
            getattr(self, "see_through_quant_mode", None),
            "value",
            self.config["see_through_quant_mode"],
        )
        dtype_value = getattr(getattr(self, "see_through_dtype", None), "value", self.config["see_through_dtype"])
        seed_value = getattr(getattr(self, "see_through_seed", None), "value", str(self.config["see_through_seed"]))
        offload_policy_value = getattr(
            getattr(self, "see_through_offload_policy", None),
            "value",
            self.config["see_through_offload_policy"],
        )
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return
        if not output_path:
            output_path = input_path

        repo_id_layerdiff_value = getattr(
            getattr(self, "see_through_repo_id_layerdiff", None),
            "value",
            self.config["see_through_repo_id_layerdiff"],
        )
        repo_id_depth_value = getattr(
            getattr(self, "see_through_repo_id_depth", None),
            "value",
            self.config["see_through_repo_id_depth"],
        )
        args = [
            f"--input_dir={input_path}",
            f"--output_dir={output_path}",
            f"--repo_id_layerdiff={repo_id_layerdiff_value}",
            f"--repo_id_depth={repo_id_depth_value}",
            f"--resolution={int(self.config['see_through_resolution'])}",
            f"--resolution_depth={int(resolution_depth_value)}",
            f"--inference_steps_depth={int(self.config['see_through_inference_steps_depth'])}",
            f"--seed={int(seed_value)}",
            f"--dtype={dtype_value}",
            f"--quant_mode={quant_mode_value}",
            f"--offload_policy={offload_policy_value}",
        ]

        args.append("--group_offload" if self.config["see_through_group_offload"] else "--no-group_offload")
        args.append("--skip_completed" if self.config["see_through_skip_completed"] else "--no-skip_completed")
        args.append("--save_to_psd" if self.config["see_through_save_to_psd"] else "--no-save_to_psd")
        args.append("--tblr_split" if self.config["see_through_tblr_split"] else "--no-tblr_split")

        def pre_log(lv):
            lv.info(t("log_start_see_through"))
            lv.info(f"{t('log_input_path')}: {input_path}")
            lv.info(f"{t('log_output_dir')}: {output_path}")
            lv.info(f"{t('log_params')}: {args}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "module.see_through.cli",
            args,
            name=t("job_name_see_through"),
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("see_through_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("see_through_failed"), type="negative"),
        )

    async def _start_translate(self):
        """开始文本翻译"""
        input_path = self.translate_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        # 构建参数
        args = [input_path]
        args.append(f"--output_name={self.translate_output_name.value}")
        args.append(f"--model_id={self.translate_model.value}")
        args.append(f"--source_lang={self.translate_source_lang.value}")
        args.append(f"--target_lang={self.translate_target_lang.value}")
        args.append(f"--max_chars={int(self.config['translate_max_chars'])}")
        args.append(f"--context_chars={int(self.config['translate_context_chars'])}")
        args.append(f"--max_new_tokens={int(self.config['translate_max_new_tokens'])}")
        args.append(f"--temperature={self.config['translate_temperature']}")

        if self.translate_glossary.value:
            args.append(f"--glossary_file={self.translate_glossary.value}")

        if self.config["translate_skip_normalize"]:
            args.append("--skip_normalize")

        if self.config["translate_normalize_only"]:
            args.append("--normalize_only")

        if self.config["translate_no_export"]:
            args.append("--no_export")

        if self.config["translate_force_reimport"]:
            args.append("--force_reimport")

        def pre_log(lv):
            lv.info(t("log_start_translate"))
            lv.info(f"{t('log_input_path')}: {input_path}")
            lv.info(f"{t('log_model')}: {self.translate_model.value}")
            lv.info(f"{t('log_params')}: {args}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "module.texttranslate",
            args,
            name=t("job_name_translate"),
            runner_kwargs={"uv_extra": "translate"},
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("translate_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("translate_failed"), type="negative"),
        )

    async def _start_watermark(self):
        """开始水印检测"""
        input_path = self.watermark_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        # 构建参数
        args = [input_path]
        args.append(f"--repo_id={self.watermark_model.value}")
        args.append(f"--model_dir={self.watermark_model_dir.value}")
        args.append(f"--batch_size={int(self.config['watermark_batch_size'])}")

        if self.config["watermark_thresh"] != 1.0:
            args.append(f"--thresh={self.config['watermark_thresh']}")

        def pre_log(lv):
            lv.info(t("log_start_watermark"))
            lv.info(f"{t('log_input_path')}: {input_path}")
            lv.info(f"{t('log_model')}: {self.watermark_model.value}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "module.waterdetect",
            args,
            name=t("job_name_watermark"),
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("watermark_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("watermark_failed"), type="negative"),
        )

    async def _start_preprocess(self):
        """开始图像预处理"""
        input_path = self.preprocess_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        # 构建参数
        args = [f"--input={input_path}"]

        if self.preprocess_align.value:
            args.append(f"--align-input={self.preprocess_align.value}")

        if self.config["max_long_edge"]:
            args.append(f"--max-long-edge={int(self.config['max_long_edge'])}")

        if self.config["max_short_edge"]:
            args.append(f"--max-short-edge={int(self.config['max_short_edge'])}")

        if self.config["max_pixels"]:
            args.append(f"--max-pixels={int(self.config['max_pixels'])}")

        if self.config["preprocess_workers"]:
            args.append(f"--workers={int(self.config['preprocess_workers'])}")

        if self.transform_type.value:
            args.append(f"--transform-type={self.transform_type.value}")

        if self.bg_color.value:
            args.append("--bg-color")
            for component in self.bg_color.value.split():
                args.append(component)

        if self.config["preprocess_recursive"]:
            args.append("--recursive")

        if self.config["crop_transparent"]:
            args.append("--crop-transparent")

        def pre_log(lv):
            lv.info(t("log_start_preprocess"))
            lv.info(f"{t('log_input_path')}: {input_path}")
            lv.info(f"{t('log_params')}: {args}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "utils.preprocess_datasets",
            args,
            name=t("job_name_preprocess"),
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("preprocess_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("preprocess_failed"), type="negative"),
        )

    async def _start_reward(self):
        """开始图像评分"""
        input_path = self.reward_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        # 构建参数
        args = [input_path]
        args.append(f"--repo_id={self.reward_model.value}")
        args.append(f"--batch_size={int(self.config['reward_batch_size'])}")
        args.append(f"--device={self.reward_device.value}")
        args.append(f"--dtype={self.reward_dtype.value}")

        def pre_log(lv):
            lv.info(t("log_start_scoring"))
            lv.info(f"{t('log_input_path')}: {input_path}")
            lv.info(f"{t('log_model')}: {self.reward_model.value}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "module.rewardmodel",
            args,
            name=t("job_name_reward"),
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("scoring_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("scoring_failed"), type="negative"),
        )

    async def _start_audio_separator(self):
        """开始音频分轨"""
        input_path = self.audio_separator_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        args = [input_path]

        args.append(f"--output_format={self.audio_separator_output_format.value}")
        args.append(f"--segment_size={int(self.config['audio_separator_segment_size'])}")
        args.append(f"--overlap={int(self.config['audio_separator_overlap'])}")
        args.append(f"--batch_size={int(self.config['audio_separator_batch_size'])}")

        if self.config["audio_separator_overwrite"]:
            args.append("--overwrite")
        if self.config["audio_separator_harmony_separation"]:
            args.append("--harmony_separation")
        if self.config["audio_separator_vocal_midi"]:
            selected_formats = self.audio_separator_vocal_midi_output_formats.value or ["mid"]
            args.append("--vocal_midi")
            args.append(f"--vocal_midi_repo_id={self.audio_separator_vocal_midi_model.value}")
            if self.audio_separator_vocal_midi_language.value != "auto":
                args.append(f"--vocal_midi_language={self.audio_separator_vocal_midi_language.value}")
            args.append(f"--vocal_midi_output_formats={','.join(selected_formats)}")
            args.append(f"--vocal_midi_batch_size={int(self.config['audio_separator_vocal_midi_batch_size'])}")
            args.append(f"--vocal_midi_seg_threshold={self.config['audio_separator_vocal_midi_seg_threshold']}")
            args.append(f"--vocal_midi_seg_radius={self.config['audio_separator_vocal_midi_seg_radius']}")
            args.append(f"--vocal_midi_t0={self.config['audio_separator_vocal_midi_t0']}")
            args.append(f"--vocal_midi_nsteps={int(self.config['audio_separator_vocal_midi_nsteps'])}")
            args.append(f"--vocal_midi_est_threshold={self.config['audio_separator_vocal_midi_est_threshold']}")

        def pre_log(lv):
            lv.info(t("log_start_audio_separator"))
            lv.info(f"{t('log_input_path')}: {input_path}")
            lv.info(f"{t('log_params')}: {args}")

        panel = self._ensure_execution_panel()
        await panel.run_job(
            "module.audio_separator",
            args,
            name=t("job_name_audio_separator"),
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("audio_separator_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("audio_separator_failed"), type="negative"),
        )


def render_tools_step():
    """渲染工具步骤"""
    step = ToolsStep()
    step.render()
