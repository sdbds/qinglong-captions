"""步骤 5: 数据集导出 - 对应 lanceExport.ps1"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.log_viewer import create_log_viewer
from components.advanced_inputs import toggle_switch, styled_select
from gui.utils.job_manager import job_manager
from gui.utils.process_runner import ProcessStatus
from gui.utils.i18n import t


class ExportStep:
    """数据集导出页面"""

    VERSIONS = ["gemini", "WDtagger", "pixtral"]

    def __init__(self):
        self.config: Dict[str, Any] = {
            "not_clip_with_caption": False,
        }
        self.log_viewer = None
        self.is_running = False
        self.current_job = None

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            # 页面标题
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("upload", size="32px").style(f"color: {COLORS['primary']};")
                with ui.column().classes("gap-0"):
                    ui.label(t("export_title")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(t("export_desc")).classes("text-body2").style("color: var(--color-text-secondary);")

            with ui.stepper().props("vertical").classes("w-full") as stepper:
                # 步骤 5.1: 选择 Lance 文件
                with ui.step(t("lance_file"), icon="insert_drive_file"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("insert_drive_file", size="22px").style(f"color: {COLORS['info']};")
                            ui.label(t("lance_file")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # Lance 文件路径
                        self.lance_file = create_path_selector(
                            label=t("lance_file"),
                            selection_type="file",
                            placeholder=t("lance_file_placeholder"),
                            file_filter="*.lance",
                        )

                        # 如果默认的 dataset.lance 存在，设置默认值
                        default_lance = Path("./datasets/dataset.lance")
                        if default_lance.exists():
                            self.lance_file.value = str(default_lance)

                    with ui.row().classes("w-full justify-end q-mt-md"):
                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 5.2: 配置导出选项
                with ui.step(t("export_settings"), icon="settings"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("settings", size="22px").style(f"color: {COLORS['warning']};")
                            ui.label(t("export_settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 输出目录
                        self.output_dir = create_path_selector(
                            label=t("output_dir"), selection_type="dir", placeholder=t("output_dir_placeholder")
                        )

                        # 默认使用 datasets 目录
                        default_output = Path("./datasets")
                        if default_output.exists():
                            self.output_dir.value = str(default_output)

                        # 版本选择 - 带图标的现代化下拉框
                        self.version = styled_select(
                            options=dict(zip(self.VERSIONS, self.VERSIONS)),
                            value="gemini",
                            label=t("version"),
                            icon="tag",
                            icon_color=COLORS["primary"],
                        )

                        # 不根据字幕裁剪
                        toggle_switch("not_clip_with_caption", self.config, "not_clip_with_caption")

                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                        prev_btn.classes("modern-btn-ghost").props('type="button"')

                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 5.3: 开始导出
                with ui.step(t("start_export"), icon="play_circle"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("play_circle", size="22px").style(f"color: {COLORS['success']};")
                            ui.label(t("start_export")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                    # 控制按钮
                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        with ui.row().classes("gap-2"):
                            prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                            prev_btn.classes("modern-btn-ghost").props('type="button"')

                        with ui.row().classes("gap-2"):
                            self.stop_btn = ui.button(t("stop"), on_click=self._stop_export, icon="stop")
                            self.stop_btn.classes("modern-btn-danger").props('type="button"')
                            self.stop_btn.set_enabled(False)

                            self.start_btn = ui.button(t("start_export"), on_click=self._start_export, icon="play_arrow")
                            self.start_btn.classes("modern-btn-success").props('type="button"')

                    # 日志查看器
                    self.log_viewer = create_log_viewer()

    async def _start_export(self):
        """开始导出"""
        lance_file = self.lance_file.value
        if not lance_file or not Path(lance_file).exists():
            ui.notify(t("select_valid_lance"), type="warning")
            return

        self.is_running = True
        self.start_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)

        output_dir = self.output_dir.value or "./datasets"
        version = self.version.value

        # 构建参数
        args = [lance_file]
        args.append(f"--output_dir={output_dir}")
        args.append(f"--version={version}")

        if self.config["not_clip_with_caption"]:
            args.append("--not_clip_with_caption")

        # 提交 Job
        job = await job_manager.submit("module.lanceexport", args, name="Export")
        self.current_job = job
        self.log_viewer.attach_job(job)

        self.log_viewer.info(t("log_start_export"))
        self.log_viewer.info(f"{t('log_lance_file')}: {lance_file}")
        self.log_viewer.info(f"{t('log_output_dir')}: {output_dir}")
        self.log_viewer.info(f"{t('log_version')}: {version}")
        self.log_viewer.info(f"{t('log_params')}: {args}")

        result = await job.wait()

        if result.status == ProcessStatus.SUCCESS:
            self.log_viewer.success(t("export_success"))
            ui.notify(t("export_success"), type="positive")
        else:
            self.log_viewer.error(t("export_failed"))
            ui.notify(t("export_failed"), type="negative")

        self.is_running = False
        self.current_job = None
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)

    def _stop_export(self):
        """停止导出"""
        if self.current_job:
            job_manager.cancel(self.current_job.id)
            self.current_job = None
        self.is_running = False
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        self.log_viewer.info(t("task_stopped"))
        ui.notify(t("task_stopped"), type="info")


def render_export_step():
    """渲染导出步骤"""
    step = ExportStep()
    step.render()
