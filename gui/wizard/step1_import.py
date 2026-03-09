"""步骤 1: 数据集导入 - 对应 lanceImport.ps1"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.log_viewer import create_log_viewer
from components.advanced_inputs import toggle_switch, styled_select, styled_input
from gui.utils.process_runner import process_runner, ProcessStatus
from gui.utils.i18n import t


class ImportStep:
    """数据集导入页面"""

    IMPORT_MODES = {
        "All Files": 0,
        "Video Only": 1,
        "Audio Only": 2,
        "Split Mode": 3,
    }

    def __init__(self):
        self.config: Dict[str, Any] = {
            "import_mode": 0,
            "tag": "gemini",
            "no_save_binary": False,
            "not_save_disk": False,
        }
        self.log_viewer = None
        self.is_running = False

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            # 页面标题
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("download", size="32px").style(f"color: {COLORS['primary']};")
                with ui.column().classes("gap-0"):
                    ui.label(t("import_title")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(t("import_desc")).classes("text-body2").style("color: var(--color-text-secondary);")

            with ui.stepper().props("vertical").classes("w-full") as stepper:
                # 步骤 1.1: 选择输入路径
                with ui.step(t("input_path")):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("folder_open", size="22px").style(f"color: {COLORS['info']};")
                            ui.label(t("dataset_path")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 输入路径
                        self.input_path = create_path_selector(
                            label=t("input_path"), selection_type="dir", placeholder=t("input_path_placeholder")
                        )

                        # 输出名称
                        self.output_name = ui.input(
                            label=t("output_name"), placeholder=t("output_name_placeholder"), value="dataset"
                        )
                        self.output_name.classes("modern-input w-full")

                    with ui.row().classes("q-mt-md"):
                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 1.2: 配置导入选项
                with ui.step(t("import_mode")):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("settings", size="22px").style(f"color: {COLORS['warning']};")
                            ui.label(t("import_mode")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 导入模式 - 带图标的现代化下拉框
                        self.import_mode = styled_select(
                            options=dict(zip(self.IMPORT_MODES.keys(), self.IMPORT_MODES.keys())),
                            value="All Files",
                            label=t("import_mode"),
                            icon="settings",
                            icon_color=COLORS["warning"],
                        )

                        # 标签
                        self.tag = styled_input(value="gemini", label=t("tag"), icon="label", icon_color=COLORS["primary"])

                        # 选项开关
                        with ui.row().classes("w-full gap-4 q-mt-md"):
                            toggle_switch("no_save_binary", self.config, "no_save_binary")
                            toggle_switch("not_save_disk", self.config, "not_save_disk")

                    with ui.row().classes("q-mt-md gap-2"):
                        prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                        prev_btn.classes("modern-btn-ghost").props('type="button"')

                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 1.3: 开始导入
                with ui.step(t("start_import")):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("play_circle", size="22px").style(f"color: {COLORS['success']};")
                            ui.label(t("start_import")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 参数预览
                        with ui.row().classes("w-full gap-4"):
                            with ui.column().classes("gap-1"):
                                ui.label(t("input_path")).classes("text-caption").style("color: var(--color-text-secondary);")
                                ui.label("").bind_text_from(self.input_path, "value").classes("text-body2")

                            with ui.column().classes("gap-1"):
                                ui.label(t("output_name")).classes("text-caption").style("color: var(--color-text-secondary);")
                                ui.label("").bind_text_from(self.output_name, "value").classes("text-body2")

                    # 控制按钮
                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        with ui.row().classes("gap-2"):
                            prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                            prev_btn.classes("modern-btn-ghost").props('type="button"')

                        with ui.row().classes("gap-2"):
                            self.stop_btn = ui.button(t("stop"), on_click=self._stop_import, icon="stop")
                            self.stop_btn.classes("modern-btn-danger").props('type="button"')
                            self.stop_btn.set_enabled(False)

                            self.start_btn = ui.button(t("start_import"), on_click=self._start_import, icon="play_arrow")
                            self.start_btn.classes("modern-btn-success").props('type="button"')

                    # 日志查看器
                    self.log_viewer = create_log_viewer()

    async def _start_import(self):
        """开始导入"""
        input_path = self.input_path.value
        if not input_path or not Path(input_path).exists():
            ui.notify(t("select_valid_input"), type="warning")
            return

        output_name = self.output_name.value or "dataset"
        import_mode = self.IMPORT_MODES.get(self.import_mode.value, 0)
        tag = self.tag.value or "gemini"

        self.is_running = True
        self.start_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)

        self.log_viewer.info(t("log_start_import"))
        self.log_viewer.info(f"{t('log_input_path')}: {input_path}")
        self.log_viewer.info(f"{t('log_output_name')}: {output_name}")
        self.log_viewer.info(f"{t('log_import_mode')}: {self.import_mode.value}")
        self.log_viewer.info(f"{t('log_tag')}: {tag}")

        # 将日志回调连接到 log_viewer
        process_runner.set_callbacks(log_callback=self.log_viewer.info)

        # 构建参数
        args = [input_path]
        args.append(f"--output_name={output_name}")
        args.append(f"--import_mode={import_mode}")
        args.append(f"--tag={tag}")

        if self.config["no_save_binary"]:
            args.append("--no_save_binary")
        if self.config["not_save_disk"]:
            args.append("--not_save_disk")

        self.log_viewer.info(f"{t('log_params')}: {args}")

        # 运行导入
        result = await process_runner.run_python_script("module.lanceImport", args)

        if result.status == ProcessStatus.SUCCESS:
            self.log_viewer.success(t("import_success"))
            ui.notify(t("import_success"), type="positive")
        else:
            self.log_viewer.error(t("import_failed"))
            ui.notify(t("import_failed"), type="negative")

        process_runner.set_callbacks(log_callback=None)
        self.is_running = False
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)

    def _stop_import(self):
        """停止导入"""
        process_runner.terminate()
        self.is_running = False
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        self.log_viewer.info(t("task_stopped"))
        ui.notify(t("task_stopped"), type="info")


def render_import_step():
    """渲染导入步骤"""
    step = ImportStep()
    step.render()
