"""步骤 4: 字幕生成 - 对应 run.ps1 (captioner.py)"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.advanced_inputs import editable_slider, toggle_switch, styled_select, styled_input
from components.execution_panel import ExecutionPanel
from gui.utils.i18n import t
from module.providers.catalog import route_choices, route_requires_remote_config


class CaptionStep:
    """字幕生成页面"""

    # API 配置 - 模型列表（与 config.toml / captioner.py 后端保持一致）
    API_CONFIGS = {
        "Gemini": {
            "key_name": "gemini_api_key",
            "models": [
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-image-preview",
                "gemini-3.1-flash-lite-preview",
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
                "pixtral-large-latest",
                "pixtral-12b-latest",
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
            ],
            "default_model": "mistral-large-latest",
            "supports_video": False,
            "supports_task": False,
        },
        "Step": {
            "key_name": "step_api_key",
            "models": [
                "step-3-vl",
                "step-3-vl-mini",
                "step-2o-vision",
                "step-1.5v-max",
                "step-1.5v-mini",
                "step-r1-v-mini",
            ],
            "default_model": "step-3-vl-mini",
            "supports_video": True,
            "supports_task": False,
        },
        "Qwen": {
            "key_name": "qwenVL_api_key",
            "models": [
                "qwen3-max-lastest",
                "qwen3-vl-plus",
                "qwen3-vl-plus-latest",
                "qwen3-vl-flash",
                "qwen3-vl-flash-latest",
                "qwen-vl-ocr",
                "qwen-vl-ocr-latest",
            ],
            "default_model": "qwen3-vl-plus",
            "supports_video": True,
            "supports_task": False,
        },
        "Kimi": {
            "key_name": "kimi_api_key",
            "models": [
                "kimi-k2.5",
                "moonshot-v1-8k",
                "moonshot-v1-32k",
                "moonshot-v1-128k",
            ],
            "default_model": "kimi-k2.5",
            "supports_video": True,
            "supports_task": False,
            "note": "kimi-latest 已于 2026年1月28日停止新用户使用",
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
                "MiniMax-M2.5",
                "MiniMax-M2.5-highspeed",
                "MiniMax-M2.1",
                "MiniMax-M2.1-highspeed",
                "MiniMax-M2",
            ],
            "default_model": "MiniMax-M2.5",
            "supports_video": True,
            "supports_task": False,
        },
        "MiniMax-Code": {
            "key_name": "minimax_code_api_key",
            "models": [
                "MiniMax-M2.5",
                "MiniMax-M2.5-highspeed",
                "MiniMax-M2.1",
            ],
            "default_model": "MiniMax-M2.5",
            "supports_video": True,
            "supports_task": False,
        },
        "GLM": {
            "key_name": "glm_api_key",
            "models": [
                "glm-5",
                "glm-4.5v",
                "glm-4.1v",
                "glm-4v-plus",
                "glm-4v",
            ],
            "default_model": "glm-5",
            "supports_video": True,
            "supports_task": False,
        },
        "Ark": {
            "key_name": "ark_api_key",
            "models": [],
            "default_model": "",
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

    SCENE_DETECTORS = [
        "AdaptiveDetector",
        "ContentDetector",
        "HashDetector",
        "HistogramDetector",
        "ThresholdDetector",
    ]

    OCR_MODELS = list(route_choices("ocr_model"))

    VLM_MODELS = list(route_choices("vlm_image_model"))

    ALM_MODELS = list(route_choices("alm_model"))

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
        "lfm_vl_local": "lfm-vl-local",
    }

    ALM_EXTRA_MAP = {
        "music_flamingo_local": "music-flamingo-local",
    }

    def __init__(self):
        self.config: Dict[str, Any] = {
            "mode": "long",
            "wait_time": 1,
            "max_retries": 100,
            "segment_time": 600,
            "segment_time_explicit": False,
            "scene_detector": "AdaptiveDetector",
            "scene_threshold": 0.0,
            "scene_min_len": 15,
            "tags_highlightrate": 0.38,
            "dir_name": False,
            "not_clip_with_caption": True,
            "scene_luma_only": False,
            "document_image": True,
            "mistral_ocr_mode": False,
        }
        self.panel: ExecutionPanel = None
        self.api_keys = {}
        self._syncing_segment_time = False

    @staticmethod
    def _has_text(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    def _current_alm_model(self) -> str:
        alm_model = getattr(self, "alm_model", None)
        return getattr(alm_model, "value", "") or ""

    def _default_segment_time(self) -> int:
        if self._current_alm_model() == "music_flamingo_local":
            return 1200
        return 600

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

    def _sync_segment_time_default(self) -> None:
        if self.config.get("segment_time_explicit"):
            return

        default_value = self._default_segment_time()
        self.config["segment_time"] = default_value

        slider = getattr(self, "segment_time_slider", None)
        if slider is not None and getattr(slider, "value", None) != default_value:
            self._syncing_segment_time = True
            try:
                slider.set_value(default_value)
            finally:
                self._syncing_segment_time = False

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

    def _build_local_extra_args(self) -> list[str]:
        extra_args: list[str] = []
        seen: set[str] = set()

        self._append_extra(extra_args, seen, self.OCR_EXTRA_MAP.get(self.ocr_model.value or ""))
        self._append_extra(extra_args, seen, self.VLM_EXTRA_MAP.get(self.vlm_image_model.value or ""))
        self._append_extra(extra_args, seen, self.ALM_EXTRA_MAP.get(self._current_alm_model()))

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

        if self.scene_detector.value != "AdaptiveDetector":
            args.append(f"--scene_detector={self.scene_detector.value}")

        if self.config["scene_threshold"] != 0.0:
            args.append(f"--scene_threshold={self.config['scene_threshold']}")

        if self.config["scene_min_len"] != 15:
            args.append(f"--scene_min_len={self.config['scene_min_len']}")

        if self.config["scene_luma_only"]:
            args.append("--scene_luma_only")

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
                scene_tab = ui.tab(t("scene_detector"), icon="radar")
                ocr_tab = ui.tab(t("local_model_routes"), icon="text_fields")

            with ui.tab_panels(tabs, value=basic_tab).classes("w-full"):
                # 基础设置
                with ui.tab_panel(basic_tab):
                    self._render_basic_settings()

                # API 配置
                with ui.tab_panel(api_tab):
                    self._render_api_settings()

                # 场景检测
                with ui.tab_panel(scene_tab):
                    self._render_scene_settings()

                # OCR/VLM
                with ui.tab_panel(ocr_tab):
                    self._render_ocr_settings()

            # 执行面板 (Start/Stop + LogViewer)
            self.panel = ExecutionPanel(start_label=t("start_caption"))
            self.panel._on_start = self._start_caption

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
                        task_input = ui.input(label=t("gemini_task"), placeholder=t("gemini_task_placeholder"))
                        task_input.classes("modern-input w-full")
                        setattr(self, f"{config['key_name']}_task", task_input)

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

    def _render_scene_settings(self):
        """渲染场景检测设置"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("radar", size="22px").style(f"color: {COLORS['warning']};")
                ui.label(t("scene_detector")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            # 场景检测器 - 带图标的现代化下拉框
            self.scene_detector = styled_select(
                options=dict(zip(self.SCENE_DETECTORS, self.SCENE_DETECTORS)),
                value="AdaptiveDetector",
                label=t("scene_detector"),
                icon="radar",
                icon_color=COLORS["warning"],
            )

            # 数值设置 - 使用可编辑滑块
            with ui.row().classes("w-full gap-4 q-mt-md"):
                editable_slider(
                    label_key="scene_threshold",
                    value_ref=self.config,
                    value_key="scene_threshold",
                    min_val=0.0,
                    max_val=100.0,
                    step=0.1,
                    decimals=1,
                )

                editable_slider(
                    label_key="scene_min_len",
                    value_ref=self.config,
                    value_key="scene_min_len",
                    min_val=1,
                    max_val=1000,
                    step=1,
                    decimals=0,
                )

            # 开关选项
            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("scene_luma_only", self.config, "scene_luma_only")

    def _render_ocr_settings(self):
        """渲染 OCR/VLM/ALM 设置"""
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("text_fields", size="22px").style(f"color: {COLORS['info']};")
                ui.label("OCR " + t("settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            # OCR 模型 - 带图标的现代化下拉框
            self.ocr_model = styled_select(
                options=dict(zip(self.OCR_MODELS, self.OCR_MODELS)),
                value="",
                label=t("ocr_model"),
                icon="text_fields",
                icon_color=COLORS["info"],
            )

            # 开关选项
            with ui.row().classes("w-full gap-4 q-mt-md"):
                toggle_switch("document_image", self.config, "document_image")

            with ui.row().classes("w-full items-center gap-2 q-mb-md q-mt-md"):
                ui.icon("visibility", size="22px").style(f"color: {COLORS['secondary']};")
                ui.label("VLM " + t("settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            # VLM 图像模型 - 带图标的现代化下拉框
            self.vlm_image_model = styled_select(
                options=dict(zip(self.VLM_MODELS, self.VLM_MODELS)),
                value="",
                label=t("vlm_image_model"),
                icon="visibility",
                icon_color=COLORS["secondary"],
            )

            with ui.row().classes("w-full items-center gap-2 q-mb-md q-mt-md"):
                ui.icon("graphic_eq", size="22px").style(f"color: {COLORS['primary']};")
                ui.label("ALM " + t("settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            self.alm_model = styled_select(
                options=dict(zip(self.ALM_MODELS, self.ALM_MODELS)),
                value="",
                label=t("alm_model"),
                icon="graphic_eq",
                icon_color=COLORS["primary"],
                on_change=lambda _value: self._sync_segment_time_default(),
            )

            self._sync_segment_time_default()

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

        await self.panel.run_job(
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
