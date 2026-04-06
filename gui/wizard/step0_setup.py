"""步骤 0: 环境检查和初始设置 - 现代化样式"""

from importlib import metadata
from nicegui import ui
from pathlib import Path
import sys
import platform
from typing import Optional
from theme import get_classes, COLORS
from gui.utils.i18n import t
from module.gpu_profile import format_gpu_device_lines, format_gpu_summary, get_cached_gpu_probe


class SetupStep:
    """环境检查与设置页面 - 现代化样式"""

    def __init__(self, on_complete: Optional[callable] = None):
        self.on_complete = on_complete
        self.check_results = {}
        self.gpu_probe = get_cached_gpu_probe()
        self.gpu_summary_label = None
        self.gpu_detail_section = None

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            # 页面标题
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("settings", size="32px").style(f"color: {COLORS['primary']};")
                with ui.column().classes("gap-0"):
                    ui.label(t("env_check")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(t("env_check_desc")).classes("text-body2").style("color: var(--color-text-secondary);")

            # 系统信息
            with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                    ui.icon("computer", size="22px").style(f"color: {COLORS['info']};")
                    ui.label(t("system_info")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                with ui.grid(columns=2).classes("w-full gap-4"):
                    with ui.column().classes("gap-1"):
                        ui.label(t("os")).classes("text-caption").style("color: var(--color-text-secondary);")
                        ui.label(f"{platform.system()} {platform.release()}").classes("text-body2").style(
                            "color: var(--color-text);"
                        )

                    with ui.column().classes("gap-1"):
                        ui.label(t("python_version")).classes("text-caption").style("color: var(--color-text-secondary);")
                        ui.label(f"{platform.python_version()}").classes("text-body2").style("color: var(--color-text);")

                    with ui.column().classes("gap-1"):
                        ui.label(t("working_dir")).classes("text-caption").style("color: var(--color-text-secondary);")
                        ui.label(f"{Path.cwd().absolute()}").classes("text-body2").style("color: var(--color-text);")

                    with ui.column().classes("gap-1"):
                        ui.label(t("gpu")).classes("text-caption").style("color: var(--color-text-secondary);")
                        self.gpu_summary_label = (
                            ui.label(format_gpu_summary(self.gpu_probe))
                            .classes("text-body2")
                            .style("color: var(--color-text);")
                        )
                self.gpu_detail_section = ui.column().classes("w-full gap-1 q-mt-md")
                self._refresh_gpu_probe_ui()

            # 环境检查列表
            with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                    ui.icon("checklist", size="22px").style(f"color: {COLORS['success']};")
                    ui.label(t("dependency_check")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                self.check_items = []

                # Python 检查
                with (
                    ui.row()
                    .classes("w-full items-center justify-between q-pa-sm")
                    .style(f"""
                    background: rgba(5, 150, 105, 0.08);
                    border-radius: 10px;
                    border: 1px solid rgba(5, 150, 105, 0.2);
                """) as python_row
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("code", size="20px").style(f"color: {COLORS['primary']};")
                        ui.label("Python 3.8+").classes("text-body2").style("color: var(--color-text);")
                    self.python_status = (
                        ui.label(t("checking")).classes("text-caption").style("color: var(--color-text-secondary);")
                    )
                    self.check_items.append(("python", python_row))

                # PyTorch 检查
                with (
                    ui.row()
                    .classes("w-full items-center justify-between q-pa-sm q-mt-sm")
                    .style(f"""
                    background: rgba(251, 191, 36, 0.08);
                    border-radius: 10px;
                    border: 1px solid rgba(251, 191, 36, 0.2);
                """) as torch_row
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("local_fire_department", size="20px").style(f"color: {COLORS['secondary']};")
                        ui.label("PyTorch").classes("text-body2").style("color: var(--color-text);")
                    self.torch_label = ui.label(t("checking")).classes("text-caption").style("color: var(--color-text-secondary);")
                    self.check_items.append(("torch", torch_row))

                # GPU 检查
                with (
                    ui.row()
                    .classes("w-full items-center justify-between q-pa-sm q-mt-sm")
                    .style(f"""
                    background: rgba(16, 185, 129, 0.1);
                    border-radius: 10px;
                    border: 1px solid rgba(16, 185, 129, 0.2);
                """) as cuda_row
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("memory", size="20px").style(f"color: {COLORS['success']};")
                        ui.label(t("gpu")).classes("text-body2").style("color: var(--color-text);")
                    self.cuda_label = ui.label(t("checking")).classes("text-caption").style("color: var(--color-text-secondary);")
                    self.check_items.append(("cuda", cuda_row))

            # 操作按钮
            with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(t("ready_to_start")).classes("text-body2").style("color: var(--color-text-secondary);")

                    with ui.row().classes("gap-2"):
                        recheck_btn = ui.button(t("recheck"), on_click=self._run_checks, icon="refresh")
                        recheck_btn.classes("modern-btn-ghost").props('type="button"')

                        continue_btn = ui.button(t("continue"), on_click=self._on_continue, icon="arrow_forward")
                        continue_btn.classes("modern-btn-primary").props('type="button"')

        # 运行检查
        ui.timer(0.5, self._run_checks, once=True)

    def _run_checks(self):
        """运行环境检查"""
        self.gpu_probe = get_cached_gpu_probe(refresh=True)
        self._refresh_gpu_probe_ui()

        # 检查 Python 版本
        py_version = sys.version_info
        if py_version >= (3, 8):
            self.python_status.text = f"✅ {py_version.major}.{py_version.minor}.{py_version.micro}"
            self.python_status.style(f"color: {COLORS['success']};")
            self.check_results["python"] = True
        else:
            self.python_status.text = f"❌ {py_version.major}.{py_version.minor} ({t('need_python_38')})"
            self.python_status.style(f"color: {COLORS['error']};")
            self.check_results["python"] = False

        try:
            self.torch_label.text = f"✅ {metadata.version('torch')}"
            self.torch_label.style(f"color: {COLORS['success']};")
            self.check_results["torch"] = True
        except metadata.PackageNotFoundError:
            self.torch_label.text = f"❌ {t('not_installed')}"
            self.torch_label.style(f"color: {COLORS['error']};")
            self.check_results["torch"] = False

        if self.gpu_probe.cuda_available:
            self.cuda_label.text = f"✅ {format_gpu_summary(self.gpu_probe)}"
            self.cuda_label.style(f"color: {COLORS['success']};")
            self.check_results["cuda"] = True
        else:
            self.cuda_label.text = f"⚠️ {t('not_detected_cpu_mode')}"
            self.cuda_label.style(f"color: {COLORS['warning']};")
            self.check_results["cuda"] = False

    def _on_continue(self):
        """继续到主页面"""
        if self.on_complete:
            self.on_complete()
        else:
            ui.navigate.to("/")

    def _refresh_gpu_probe_ui(self) -> None:
        if self.gpu_summary_label is not None:
            self.gpu_summary_label.text = format_gpu_summary(self.gpu_probe)
        if self.gpu_detail_section is None:
            return

        self.gpu_detail_section.clear()
        if self.gpu_probe.device_count <= 1:
            return

        with self.gpu_detail_section:
            ui.label(t("detected_gpus")).classes("text-caption").style("color: var(--color-text-secondary);")
            for line in format_gpu_device_lines(self.gpu_probe):
                ui.label(line).classes("text-body2").style("color: var(--color-text);")


def render_setup_step():
    """渲染设置步骤"""
    step = SetupStep()
    step.render()
