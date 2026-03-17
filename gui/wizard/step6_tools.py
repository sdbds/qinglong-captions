"""步骤 6: 实用工具 - 对应 watermark_detect, preprocess, reward_model 等脚本"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.log_viewer import create_log_viewer
from components.advanced_inputs import editable_slider, toggle_switch, styled_select
from gui.utils.job_manager import job_manager
from gui.utils.process_runner import ProcessStatus
from gui.utils.i18n import t


class ToolsStep:
    """工具页面"""

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
        "tencent/HY-MT1.5-7B",
        "tencent/HY-MT1.5-1.8B",
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

    def __init__(self):
        self.config: Dict[str, Any] = {
            "watermark_batch_size": 12,
            "watermark_thresh": 1.0,
            "preprocess_workers": 8,
            "max_long_edge": 2048,
            "crop_transparent": True,
            "preprocess_recursive": True,
            "reward_batch_size": 1,
            "translate_max_chars": 2200,
            "translate_context_chars": 300,
            "translate_max_new_tokens": 2048,
            "translate_temperature": 0.0,
            "translate_skip_normalize": False,
            "translate_normalize_only": False,
            "translate_no_export": False,
            "translate_force_reimport": False,
        }
        self.log_viewer = None
        self.is_running = False
        self.current_tool = None
        self.current_job = None

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
                watermark_tab = ui.tab(t("watermark_detection"), icon="water_drop")
                preprocess_tab = ui.tab(t("preprocess"), icon="image")
                reward_tab = ui.tab(t("reward_model"), icon="stars")
                translate_tab = ui.tab(t("translate"), icon="translate")

            with ui.tab_panels(tabs, value=watermark_tab).classes("w-full"):
                # 水印检测
                with ui.tab_panel(watermark_tab):
                    self._render_watermark_tool()

                # 图像预处理
                with ui.tab_panel(preprocess_tab):
                    self._render_preprocess_tool()

                # 图像评分
                with ui.tab_panel(reward_tab):
                    self._render_reward_tool()

                # 文本翻译
                with ui.tab_panel(translate_tab):
                    self._render_translate_tool()

            # 日志查看器（共享）
            with ui.card().classes(get_classes("card") + " w-full q-pa-md q-mt-md"):
                with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                    ui.icon("article", size="22px").style(f"color: {COLORS['primary']};")
                    ui.label(t("log_output")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                with ui.row().classes("w-full items-center justify-end q-mb-md"):
                    self.stop_btn = ui.button(t("stop"), on_click=self._stop_tool, icon="stop")
                    self.stop_btn.classes("modern-btn-danger").props('type="button"')
                    self.stop_btn.set_enabled(False)

                self.log_viewer = create_log_viewer()

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

            # 开始按钮
            with ui.row().classes("w-full justify-end q-mt-md"):
                start_btn = ui.button(t("start_watermark"), on_click=self._start_watermark, icon="play_arrow")
                start_btn.classes("modern-btn-success").props('type="button"')

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
                    min_val=64,
                    max_val=8192,
                    step=64,
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

            # 开始按钮
            with ui.row().classes("w-full justify-end q-mt-md"):
                start_btn = ui.button(t("start_preprocess"), on_click=self._start_preprocess, icon="play_arrow")
                start_btn.classes("modern-btn-success").props('type="button"')

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
                    options={"cuda": "CUDA", "cpu": "CPU"},
                    value="cuda",
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

            # 开始按钮
            with ui.row().classes("w-full justify-end q-mt-md"):
                start_btn = ui.button(t("start_scoring"), on_click=self._start_reward, icon="play_arrow")
                start_btn.classes("modern-btn-success").props('type="button"')

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

            # 开始按钮
            with ui.row().classes("w-full justify-end q-mt-md"):
                start_btn = ui.button(t("start_translate"), on_click=self._start_translate, icon="play_arrow")
                start_btn.classes("modern-btn-success").props('type="button"')

    async def _start_translate(self):
        """开始文本翻译"""
        input_path = self.translate_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        self.current_tool = "translate"
        self.is_running = True
        self.stop_btn.set_enabled(True)
        self.log_viewer.clear()

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

        # 提交 Job
        job = await job_manager.submit(
            "module.texttranslate",
            args,
            name="Translate",
            uv_extra="translate",
        )
        self.current_job = job
        self.log_viewer.attach_job(job)

        self.log_viewer.info(t("log_start_translate"))
        self.log_viewer.info(f"{t('log_input_path')}: {input_path}")
        self.log_viewer.info(f"{t('log_model')}: {self.translate_model.value}")
        self.log_viewer.info(f"{t('log_params')}: {args}")

        result = await job.wait()

        try:
            if result.status == ProcessStatus.SUCCESS:
                self.log_viewer.success(t("translate_success"))
                ui.notify(t("translate_success"), type="positive")
            else:
                self.log_viewer.error(t("translate_failed"))
                ui.notify(t("translate_failed"), type="negative")

            self.is_running = False
            self.current_job = None
            self.stop_btn.set_enabled(False)
        except RuntimeError:
            pass

    async def _start_watermark(self):
        """开始水印检测"""
        input_path = self.watermark_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        self.current_tool = "watermark"
        self.is_running = True
        self.stop_btn.set_enabled(True)
        self.log_viewer.clear()

        # 构建参数
        args = [input_path]
        args.append(f"--repo_id={self.watermark_model.value}")
        args.append(f"--model_dir={self.watermark_model_dir.value}")
        args.append(f"--batch_size={int(self.config['watermark_batch_size'])}")

        if self.config["watermark_thresh"] != 1.0:
            args.append(f"--thresh={self.config['watermark_thresh']}")

        # 提交 Job
        job = await job_manager.submit("module.waterdetect", args, name="Watermark")
        self.current_job = job
        self.log_viewer.attach_job(job)

        self.log_viewer.info(t("log_start_watermark"))
        self.log_viewer.info(f"{t('log_input_path')}: {input_path}")
        self.log_viewer.info(f"{t('log_model')}: {self.watermark_model.value}")

        result = await job.wait()

        try:
            if result.status == ProcessStatus.SUCCESS:
                self.log_viewer.success(t("watermark_success"))
                ui.notify(t("watermark_success"), type="positive")
            else:
                self.log_viewer.error(t("watermark_failed"))
                ui.notify(t("watermark_failed"), type="negative")

            self.is_running = False
            self.current_job = None
            self.stop_btn.set_enabled(False)
        except RuntimeError:
            pass

    async def _start_preprocess(self):
        """开始图像预处理"""
        input_path = self.preprocess_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        self.current_tool = "preprocess"
        self.is_running = True
        self.stop_btn.set_enabled(True)
        self.log_viewer.clear()

        # 构建参数
        args = [f"--input={input_path}"]

        if self.preprocess_align.value:
            args.append(f"--align-input={self.preprocess_align.value}")

        if self.config["max_long_edge"]:
            args.append(f"--max-long-edge={int(self.config['max_long_edge'])}")

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

        # 提交 Job
        job = await job_manager.submit("utils.preprocess_datasets", args, name="Preprocess")
        self.current_job = job
        self.log_viewer.attach_job(job)

        self.log_viewer.info(t("log_start_preprocess"))
        self.log_viewer.info(f"{t('log_input_path')}: {input_path}")
        self.log_viewer.info(f"{t('log_params')}: {args}")

        result = await job.wait()

        try:
            if result.status == ProcessStatus.SUCCESS:
                self.log_viewer.success(t("preprocess_success"))
                ui.notify(t("preprocess_success"), type="positive")
            else:
                self.log_viewer.error(t("preprocess_failed"))
                ui.notify(t("preprocess_failed"), type="negative")

            self.is_running = False
            self.current_job = None
            self.stop_btn.set_enabled(False)
        except RuntimeError:
            pass

    async def _start_reward(self):
        """开始图像评分"""
        input_path = self.reward_input.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        self.current_tool = "reward"
        self.is_running = True
        self.stop_btn.set_enabled(True)
        self.log_viewer.clear()

        # 构建参数
        args = [input_path]
        args.append(f"--repo_id={self.reward_model.value}")
        args.append(f"--batch_size={int(self.config['reward_batch_size'])}")
        args.append(f"--device={self.reward_device.value}")
        args.append(f"--dtype={self.reward_dtype.value}")

        # 提交 Job
        job = await job_manager.submit("module.rewardmodel", args, name="Reward")
        self.current_job = job
        self.log_viewer.attach_job(job)

        self.log_viewer.info(t("log_start_scoring"))
        self.log_viewer.info(f"{t('log_input_path')}: {input_path}")
        self.log_viewer.info(f"{t('log_model')}: {self.reward_model.value}")

        result = await job.wait()

        try:
            if result.status == ProcessStatus.SUCCESS:
                self.log_viewer.success(t("scoring_success"))
                ui.notify(t("scoring_success"), type="positive")
            else:
                self.log_viewer.error(t("scoring_failed"))
                ui.notify(t("scoring_failed"), type="negative")

            self.is_running = False
            self.current_job = None
            self.stop_btn.set_enabled(False)
        except RuntimeError:
            pass

    def _stop_tool(self):
        """停止当前工具"""
        if self.current_job:
            job_manager.cancel(self.current_job.id)
            self.current_job = None
        self.is_running = False
        self.stop_btn.set_enabled(False)
        self.log_viewer.info(t("task_stopped"))
        ui.notify(t("task_stopped"), type="info")


def render_tools_step():
    """渲染工具步骤"""
    step = ToolsStep()
    step.render()
