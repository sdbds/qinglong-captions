"""步骤 2: 视频场景分割 - 对应 video_spliter.ps1"""

from pathlib import Path
from typing import Any, Dict

from components.advanced_inputs import editable_slider, styled_select, toggle_switch
from components.execution_panel import ExecutionPanel
from components.path_selector import create_path_selector
from nicegui import ui

from gui.theme import COLORS, get_classes
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
        "AdaptiveDetector": 3.5,
        "ContentDetector": 31.0,
        "HashDetector": 0.35,
        "HistogramDetector": 0.2,
        "ThresholdDetector": 12.0,
    }

    # Per-detector slider config: (min, max, step, decimals)
    SLIDER_CONFIG = {
        "ContentDetector": (0.0, 100.0, 0.5, 1),
        "AdaptiveDetector": (0.0, 100.0, 0.1, 1),
        "HashDetector": (0.0, 1.0, 0.001, 3),
        "HistogramDetector": (0.0, 1.0, 0.001, 3),
        "ThresholdDetector": (0.0, 255.0, 1.0, 0),
    }

    def __init__(self):
        self.config: Dict[str, Any] = {
            "threshold": 3.5,
            "backend": "pyav",
            "min_scene_len_seconds": 0.6,
            "adaptive_window_width": 3,
            "hash_size": 8,
            "histogram_bins": 128,
            "images_per_scene": 1,
            "luma_only": False,
            "save_html": True,
            "recursive": False,
        }
        self.panel: ExecutionPanel = None
        self.threshold_slider = None
        self.advanced_rows = {}

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

                        styled_select(
                            options={"pyav": "PyAV", "opencv": "OpenCV"},
                            value=self.config["backend"],
                            label=t("video_backend"),
                            icon="video_settings",
                            icon_color=COLORS["info"],
                            on_change=lambda value: self.config.__setitem__("backend", value),
                            searchable=False,
                        )

                        # 阈值 - 使用可编辑滑块
                        with ui.row().classes("w-full items-center gap-2"):
                            self.threshold_slider = editable_slider(
                                label_key="threshold",
                                value_ref=self.config,
                                value_key="threshold",
                                min_val=0.0,
                                max_val=100.0,
                                step=0.1,
                                decimals=1,
                            )
                            ui.label(t("threshold_hint")).classes("text-caption").style("color: var(--color-text-secondary);")

                        # 最小场景长度（秒）
                        editable_slider(
                            label_key="min_scene_len_seconds",
                            value_ref=self.config,
                            value_key="min_scene_len_seconds",
                            min_val=0.0,
                            max_val=60.0,
                            step=0.1,
                            decimals=1,
                        )

                        with ui.row().classes("w-full items-center gap-2") as adaptive_row:
                            editable_slider(
                                label_key="adaptive_window_width",
                                value_ref=self.config,
                                value_key="adaptive_window_width",
                                min_val=1,
                                max_val=10,
                                step=1,
                                decimals=0,
                            )
                        self.advanced_rows["AdaptiveDetector"] = adaptive_row

                        with ui.row().classes("w-full items-center gap-2") as hash_row:
                            editable_slider(
                                label_key="hash_size",
                                value_ref=self.config,
                                value_key="hash_size",
                                min_val=1,
                                max_val=32,
                                step=1,
                                decimals=0,
                            )
                        self.advanced_rows["HashDetector"] = hash_row

                        with ui.row().classes("w-full items-center gap-2") as histogram_row:
                            editable_slider(
                                label_key="histogram_bins",
                                value_ref=self.config,
                                value_key="histogram_bins",
                                min_val=16,
                                max_val=256,
                                step=16,
                                decimals=0,
                            )
                        self.advanced_rows["HistogramDetector"] = histogram_row
                        self._update_advanced_visibility("AdaptiveDetector")

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

    def _on_detector_change(self, detector):
        """检测器改变时更新默认阈值和滑块配置"""
        default_threshold = self.DEFAULT_THRESHOLDS.get(detector, 0.0)

        if self.threshold_slider is not None:
            smin, smax, sstep, sdec = self.SLIDER_CONFIG.get(detector, (0.0, 100.0, 0.1, 1))
            self.threshold_slider.update_config(
                new_min=smin, new_max=smax, new_step=sstep,
                new_decimals=sdec, new_value=default_threshold,
            )
        else:
            self.config["threshold"] = default_threshold

        self._update_advanced_visibility(detector)

    def _update_advanced_visibility(self, detector):
        for detector_name, row in self.advanced_rows.items():
            row.set_visibility(detector_name == detector)

    def _build_args(self, input_dir, output_dir=None):
        detector = self.detector.value
        threshold = float(self.config["threshold"])
        min_scene_len_seconds = float(self.config["min_scene_len_seconds"])

        args = [input_dir]
        if output_dir:
            args.append(f"--output_dir={output_dir}")
        if detector != "AdaptiveDetector":
            args.append(f"--detector={detector}")

        args.extend(
            [
                f"--backend={self.config['backend']}",
                f"--threshold={threshold}",
                f"--min_scene_len_seconds={min_scene_len_seconds}",
            ]
        )
        if detector == "AdaptiveDetector":
            args.append(f"--adaptive_window_width={int(self.config['adaptive_window_width'])}")
        elif detector == "HashDetector":
            args.append(f"--hash_size={int(self.config['hash_size'])}")
        elif detector == "HistogramDetector":
            args.append(f"--histogram_bins={int(self.config['histogram_bins'])}")

        if self.config["luma_only"]:
            args.append("--luma_only")
        if self.config["save_html"]:
            args.append("--save_html")
        if self.config["recursive"]:
            args.append("--recursive")

        images_per_scene = int(self.config["images_per_scene"])
        if images_per_scene > 0:
            args.append(f"--video2images_min_number={images_per_scene}")
        return args

    async def _start_split(self):
        """开始分割"""
        input_dir = self.input_video_dir.value
        if not input_dir or not Path(input_dir).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        detector = self.detector.value
        threshold = self.config["threshold"]
        min_scene_len_seconds = float(self.config["min_scene_len_seconds"])
        args = self._build_args(input_dir, self.output_dir.value)

        def pre_log(lv):
            lv.info(t("log_start_split"))
            lv.info(f"{t('log_input_path')}: {input_dir}")
            lv.info(f"{t('log_detector')}: {detector}")
            lv.info(f"{t('log_video_backend')}: {self.config['backend']}")
            lv.info(f"{t('log_threshold')}: {threshold}")
            lv.info(f"{t('log_min_scene_len_seconds')}: {min_scene_len_seconds}")
            lv.info(f"{t('log_params')}: {args}")

        await self.panel.run_job(
            "module.videospilter",
            args,
            name=t("job_name_video_split"),
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("split_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("split_failed"), type="negative"),
        )


def render_video_split_step():
    """渲染视频分割步骤"""
    step = VideoSplitStep()
    step.render()
