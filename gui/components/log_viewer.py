"""日志查看器组件 - 现代化样式"""

from nicegui import ui
from typing import Optional
from datetime import datetime
from gui.theme import get_classes, COLORS
from gui.utils.i18n import t
from gui.components.advanced_inputs import toggle_switch_simple


class LogViewer:
    """日志查看器，支持实时滚动和导出 - 现代化样式"""

    def __init__(self, max_lines: int = 1000, height: str = "50vh"):
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.auto_scroll = True

        with ui.card().classes(get_classes("card")).style("width: 66vw; max-width: 100%; box-sizing: border-box;"):
            # 工具栏
            with ui.row().classes("w-full items-center justify-between q-pa-md"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("article", size="22px").style(f"color: {COLORS['primary']};")
                    ui.label(t("log_output")).classes("text-subtitle1 text-weight-bold").style("color: var(--color-text);")

                with ui.row().classes("items-center gap-2"):
                    # 自动滚动开关 - 使用按钮式开关
                    def on_auto_scroll_change(new_value):
                        self.auto_scroll = new_value

                    self.scroll_toggle, self.get_scroll_value = toggle_switch_simple(
                        label=t("auto_scroll", "Auto Scroll"), value=True, on_change=on_auto_scroll_change
                    )

                    ui.separator().props("vertical")

                    # 操作按钮
                    clear_btn = ui.button(icon="clear", on_click=self.clear)
                    clear_btn.classes("modern-btn-ghost")
                    clear_btn.props('dense type="button"').tooltip(t("clear_log"))

                    save_btn = ui.button(icon="save", on_click=self._save_log)
                    save_btn.classes("modern-btn-secondary")
                    save_btn.props('dense type="button"').tooltip(t("save_log"))

                    copy_btn = ui.button(icon="content_copy", on_click=self._copy_all)
                    copy_btn.classes("modern-btn-secondary")
                    copy_btn.props('dense type="button"').tooltip(t("copy_log"))

            # 日志文本区域 - 现代化样式
            with (
                ui.element("div")
                .classes("w-full q-px-md q-mb-md")
                .style(f"""
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(99, 102, 241, 0.2);
                border-radius: 12px;
                overflow: hidden;
            """)
            ):
                self.log_area = ui.log(max_lines=max_lines).classes("w-full font-mono text-sm")
                self.log_area.style(f"height: {height}; padding: 12px;")

    def append(self, message: str, level: str = "info"):
        """添加日志行"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 根据级别着色
        color_map = {
            "info": ("💙", COLORS["info"]),
            "success": ("💚", COLORS["success"]),
            "warning": ("💛", COLORS["warning"]),
            "error": ("❌", COLORS["error"]),
        }
        emoji, color = color_map.get(level, ("📝", COLORS["text_secondary"]))

        formatted = f"[{timestamp}] {emoji} {message}"
        self.lines.append(formatted)

        # 限制行数
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines :]

        self.log_area.push(formatted)

    def info(self, message: str):
        """添加信息日志"""
        self.append(message, "info")

    def success(self, message: str):
        """添加成功日志"""
        self.append(message, "success")

    def warning(self, message: str):
        """添加警告日志"""
        self.append(message, "warning")

    def error(self, message: str):
        """添加错误日志"""
        self.append(message, "error")

    def clear(self):
        """清空日志"""
        self.lines.clear()
        self.log_area.clear()
        self.info("日志已清空")

    def _save_log(self):
        """保存日志到文件"""
        from tkinter import filedialog, Tk
        from datetime import datetime

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filename = filedialog.asksaveasfilename(
            defaultextension=".log",
            initialfile=f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        root.destroy()

        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(self.lines))
                self.success(f"日志已保存: {filename}")
            except Exception as e:
                self.error(f"保存失败: {e}")

    def _copy_all(self):
        """复制所有日志到剪贴板"""
        text = "\n".join(self.lines)
        escaped_text = text.replace("`", "\\`").replace("\\", "\\\\")
        ui.run_javascript(f"navigator.clipboard.writeText(`{escaped_text}`)")
        ui.notify("✅ 日志已复制到剪贴板", type="positive")

    def get_text(self) -> str:
        """获取所有日志文本"""
        return "\n".join(self.lines)


def create_log_viewer(**kwargs) -> LogViewer:
    """创建日志查看器的便捷函数"""
    return LogViewer(**kwargs)
