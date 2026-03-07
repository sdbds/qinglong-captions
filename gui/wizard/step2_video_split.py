"""步骤 2: 视频场景分割 - 对应 video_spliter.ps1"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.log_viewer import create_log_viewer
from components.advanced_inputs import editable_slider, toggle_switch, styled_select
from gui.utils.process_runner import process_runner, ProcessStatus
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
        self.log_viewer = None
        self.is_running = False

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
                with ui.step(t("config_paths", "Configure Paths")):
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

                    with ui.row().classes("q-mt-md"):
                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 2.2: 配置检测器
                with ui.step(t("detector")):
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
                        ui.label("设为0则不保存图片").classes("text-caption").style("color: var(--color-text-secondary);")

                        # 选项 - 使用按钮式开关
                        with ui.row().classes("w-full gap-4 q-mt-md"):
                            toggle_switch("luma_only", self.config, "luma_only")
                            toggle_switch("save_html", self.config, "save_html")
                            toggle_switch("recursive", self.config, "recursive")

                    with ui.row().classes("q-mt-md gap-2"):
                        prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                        prev_btn.classes("modern-btn-ghost").props('type="button"')

                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 2.3: 开始分割
                with ui.step(t("start_split")):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("play_circle", size="22px").style(f"color: {COLORS['success']};")
                            ui.label(t("start_split")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                    # 控制按钮
                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        with ui.row().classes("gap-2"):
                            prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                            prev_btn.classes("modern-btn-ghost").props('type="button"')

                        with ui.row().classes("gap-2"):
                            self.stop_btn = ui.button(t("stop"), on_click=self._stop_split, icon="stop")
                            self.stop_btn.classes("modern-btn-danger").props('type="button"')
                            self.stop_btn.set_enabled(False)

                            self.start_btn = ui.button(t("start_split"), on_click=self._start_split, icon="play_arrow")
                            self.start_btn.classes("modern-btn-success").props('type="button"')

                    # 日志查看器
                    self.log_viewer = create_log_viewer()

    def _on_detector_change(self, e):
        """检测器改变时更新默认阈值"""
        detector = e.value
        default_threshold = self.DEFAULT_THRESHOLDS.get(detector, 0.0)
        self.config["threshold"] = default_threshold

    async def _start_split(self):
        """开始分割"""
        input_dir = self.input_video_dir.value
        if not input_dir or not Path(input_dir).exists():
            ui.notify("请选择有效的输入目录", type="warning")
            return

        self.is_running = True
        self.start_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)

        detector = self.detector.value
        threshold = self.config["threshold"]
        min_scene_len = int(self.config["min_scene_len"])

        self.log_viewer.info(f"开始视频场景分割...")
        self.log_viewer.info(f"输入目录: {input_dir}")
        self.log_viewer.info(f"检测器: {detector}")
        self.log_viewer.info(f"阈值: {threshold}")
        self.log_viewer.info(f"最小场景长度: {min_scene_len}")

        # 将日志回调连接到 log_viewer
        process_runner.set_callbacks(log_callback=self.log_viewer.info)

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

        self.log_viewer.info(f"参数: {args}")

        # 运行分割
        result = await process_runner.run_python_script("module.videospilter", args)

        if result.status == ProcessStatus.SUCCESS:
            self.log_viewer.success("视频分割完成")
            ui.notify("视频分割完成", type="positive")
        else:
            self.log_viewer.error("视频分割失败")
            ui.notify("视频分割失败", type="negative")

        process_runner.set_callbacks(log_callback=None)
        self.is_running = False
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)

    def _stop_split(self):
        """停止分割"""
        process_runner.terminate()
        self.is_running = False
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        self.log_viewer.info(t("task_stopped"))
        ui.notify(t("task_stopped"), type="info")


def render_video_split_step():
    """渲染视频分割步骤"""
    step = VideoSplitStep()
    step.render()
