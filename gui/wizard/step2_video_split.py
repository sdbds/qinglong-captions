"""步骤 2: 视频场景分割 - 对应 video_spliter.ps1"""

from nicegui import ui
from pathlib import Path
from typing import Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.advanced_inputs import editable_slider, toggle_switch, styled_select
from components.execution_panel import ExecutionPanel
from gui.utils.i18n import t


class VideoSplitStep:
    """视频分割页面"""

    DETECTORS = [
        "AdaptiveDetector",
        "ContentDetector",
        "HashDetector",
        "HistogramDetector",
        "ThresholdDetector",
    ]

    # 默认阈值
    DEFAULT_THRESHOLDS = {
        "ContentDetector": 27.0,
        "AdaptiveDetector": 3.0,
        "HashDetector": 0.395,
        "HistogramDetector": 0.05,
        "ThresholdDetector": 12.0,
    }

    def __init__(self):
        self.config: Dict[str, Any] = {
            "threshold": 0.0,
            "min_scene_len": 16,
            "images_per_scene": 1,
            "luma_only": False,
            "save_html": True,
            "recursive": False,
        }
        self.panel: ExecutionPanel = None

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            # 页面标题
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("movie", size="32px").style(f"color: {COLORS['primary']};")
                with ui.column().classes("gap-0"):
                    ui.label(t("split_title")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(t("split_desc")).classes("text-body2").style("color: var(--color-text-secondary);")

            with ui.stepper().props("vertical").classes("w-full") as stepper:
                # 步骤 2.1: 配置路径
                with ui.step(t("config_paths"), icon="folder_open"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("folder_open", size="22px").style(f"color: {COLORS['info']};")
                            ui.label(t("dataset_path")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 输入视频目录
                        self.input_video_dir = create_path_selector(
                            label=t("input_video_dir"), selection_type="dir", placeholder=t("input_path_placeholder")
                        )

                        # 输出目录
                        self.output_dir = create_path_selector(
                            label=t("output_dir"), selection_type="dir", placeholder=t("output_dir_placeholder")
                        )

                    with ui.row().classes("w-full justify-end q-mt-md"):
                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 2.2: 配置检测器
                with ui.step(t("detector"), icon="radar"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("radar", size="22px").style(f"color: {COLORS['warning']};")
                            ui.label(t("detector")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 检测器选择 - 带图标的现代化下拉框
                        self.detector = styled_select(
                            options=dict(zip(self.DETECTORS, self.DETECTORS)),
                            value="AdaptiveDetector",
                            label=t("detector"),
                            icon="radar",
                            icon_color=COLORS["warning"],
                            on_change=self._on_detector_change,
                        )

                        # 阈值 - 使用可编辑滑块
                        with ui.row().classes("w-full items-center gap-2"):
                            editable_slider(
                                label_key="threshold",
                                value_ref=self.config,
                                value_key="threshold",
                                min_val=0.0,
                                max_val=100.0,
                                step=0.1,
                                decimals=1,
                            )
                            ui.label(t("threshold_hint")).classes("text-caption").style("color: var(--color-text-secondary);")

                        # 最小场景长度 - 使用可编辑滑块
                        editable_slider(
                            label_key="min_scene_len",
                            value_ref=self.config,
                            value_key="min_scene_len",
                            min_val=1,
                            max_val=1000,
                            step=1,
                            decimals=0,
                        )

                        # 每场景图片数 - 使用可编辑滑块
                        editable_slider(
                            label_key="images_per_scene",
                            value_ref=self.config,
                            value_key="images_per_scene",
                            min_val=0,
                            max_val=10,
                            step=1,
                            decimals=0,
                        )
                        ui.label(t("images_per_scene_hint")).classes("text-caption").style("color: var(--color-text-secondary);")

                        # 选项 - 使用按钮式开关
                        with ui.row().classes("w-full gap-4 q-mt-md"):
                            toggle_switch("luma_only", self.config, "luma_only")
                            toggle_switch("save_html", self.config, "save_html")
                            toggle_switch("recursive", self.config, "recursive")

                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                        prev_btn.classes("modern-btn-ghost").props('type="button"')

                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 2.3: 开始分割
                with ui.step(t("start_split"), icon="play_circle"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("play_circle", size="22px").style(f"color: {COLORS['success']};")
                            ui.label(t("start_split")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                    # 导航按钮
                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                        prev_btn.classes("modern-btn-ghost").props('type="button"')

                    # 执行面板 (Start/Stop + LogViewer)
                    self.panel = ExecutionPanel(start_label=t("start_split"))
                    self.panel._on_start = self._start_split

    def _on_detector_change(self, e):
        """检测器改变时更新默认阈值"""
        detector = e.value
        default_threshold = self.DEFAULT_THRESHOLDS.get(detector, 0.0)
        self.config["threshold"] = default_threshold

    async def _start_split(self):
        """开始分割"""
        input_dir = self.input_video_dir.value
        if not input_dir or not Path(input_dir).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        detector = self.detector.value
        threshold = self.config["threshold"]
        min_scene_len = int(self.config["min_scene_len"])

        # 构建参数
        args = [input_dir]

        if self.output_dir.value:
            args.append(f"--output_dir={self.output_dir.value}")

        if detector != "AdaptiveDetector":
            args.append(f"--detector={detector}")

        if threshold != 0.0:
            args.append(f"--threshold={threshold}")

        if min_scene_len != 16:
            args.append(f"--min_scene_len={min_scene_len}")

        if self.config["luma_only"]:
            args.append("--luma_only")

        if self.config["save_html"]:
            args.append("--save_html")

        if self.config["recursive"]:
            args.append("--recursive")

        images_per_scene = int(self.config["images_per_scene"])
        if images_per_scene > 0:
            args.append(f"--video2images_min_number={images_per_scene}")

        def pre_log(lv):
            lv.info(t("log_start_split"))
            lv.info(f"{t('log_input_path')}: {input_dir}")
            lv.info(f"{t('log_detector')}: {detector}")
            lv.info(f"{t('log_threshold')}: {threshold}")
            lv.info(f"{t('log_min_scene_len')}: {min_scene_len}")
            lv.info(f"{t('log_params')}: {args}")

        await self.panel.run_job(
            "module.videospilter",
            args,
            name="Video Split",
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("split_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("split_failed"), type="negative"),
        )


def render_video_split_step():
    """渲染视频分割步骤"""
    step = VideoSplitStep()
    step.render()
