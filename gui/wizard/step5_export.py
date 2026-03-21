"""步骤 5: 数据集导出 - 对应 lanceExport.ps1"""

from nicegui import ui
from pathlib import Path
from typing import Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.advanced_inputs import toggle_switch, styled_select
from components.execution_panel import ExecutionPanel
from gui.utils.i18n import t


class ExportStep:
    """数据集导出页面"""

    VERSIONS = ["gemini", "WDtagger", "pixtral"]

    def __init__(self):
        self.config: Dict[str, Any] = {
            "not_clip_with_caption": False,
        }
        self.panel: ExecutionPanel = None

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

                    # 导航按钮
                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                        prev_btn.classes("modern-btn-ghost").props('type="button"')

                    # 执行面板 (Start/Stop + LogViewer)
                    self.panel = ExecutionPanel(start_label=t("start_export"))
                    self.panel._on_start = self._start_export

    async def _start_export(self):
        """开始导出"""
        lance_file = self.lance_file.value
        if not lance_file or not Path(lance_file).exists():
            ui.notify(t("select_valid_lance"), type="warning")
            return

        output_dir = self.output_dir.value or "./datasets"
        version = self.version.value

        # 构建参数
        args = [lance_file]
        args.append(f"--output_dir={output_dir}")
        args.append(f"--version={version}")

        if self.config["not_clip_with_caption"]:
            args.append("--not_clip_with_caption")

        def pre_log(lv):
            lv.info(t("log_start_export"))
            lv.info(f"{t('log_lance_file')}: {lance_file}")
            lv.info(f"{t('log_output_dir')}: {output_dir}")
            lv.info(f"{t('log_version')}: {version}")
            lv.info(f"{t('log_params')}: {args}")

        await self.panel.run_job(
            "module.lanceexport",
            args,
            name="Export",
            pre_log=pre_log,
            on_success=lambda r: ui.notify(t("export_success"), type="positive"),
            on_failure=lambda r: ui.notify(t("export_failed"), type="negative"),
        )


def render_export_step():
    """渲染导出步骤"""
    step = ExportStep()
    step.render()
