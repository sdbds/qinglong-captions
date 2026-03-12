"""日志查看器组件 - 支持 ANSI 彩色渲染"""

from nicegui import ui
from typing import Optional
from datetime import datetime
from gui.theme import get_classes, COLORS
from gui.utils.i18n import t
from gui.components.advanced_inputs import toggle_switch_simple
from gui.utils.ansi_to_html import AnsiToHtmlConverter, strip_ansi


class LogViewer:
    """日志查看器，支持 ANSI 彩色渲染、实时滚动和导出

    使用缓冲区 + 定时刷新机制，将日志批量推送到前端，
    避免高速输出时大量 WebSocket 消息导致浏览器断连。
    """

    def __init__(self, max_lines: int = 1000, height: str = "50vh"):
        self.max_lines = max_lines
        self.lines: list[str] = []  # 原始文本（可能含 ANSI）
        self.auto_scroll = True
        self._buffer: list[str] = []
        self._line_count = 0  # 已渲染到 DOM 的行数
        self._converter = AnsiToHtmlConverter()

        with ui.card().classes(get_classes("card")).style("width: 66vw; max-width: 100%; box-sizing: border-box;"):
            # 工具栏
            with ui.row().classes("w-full items-center justify-between q-pa-md"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("article", size="22px").style(f"color: {COLORS['primary']};")
                    ui.label(t("log_output")).classes("text-subtitle1 text-weight-bold").style("color: var(--color-text);")

                with ui.row().classes("items-center gap-2"):
                    # 自动滚动开关
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

                    ui.separator().props("vertical")

                    # 弹出控制台按钮
                    console_btn = ui.button(
                        icon="open_in_new",
                        on_click=lambda: ui.run_javascript("window.open('/console', '_blank')")
                    )
                    console_btn.classes("modern-btn-secondary")
                    console_btn.props('dense type="button"').tooltip(t("open_console", "Open Console"))

            # 日志 HTML 渲染区域
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
                self.scroll_area = ui.scroll_area().classes("w-full").style(f"height: {height};")
                with self.scroll_area:
                    self.log_container = ui.element("div").style(
                        "font-family: 'Cascadia Code', 'Consolas', 'Monaco', monospace; "
                        "font-size: 13px; line-height: 1.5; padding: 12px; "
                        "white-space: pre-wrap; word-break: break-all; "
                        "color: #e5e5e5;"
                    )

        # 定时刷新缓冲区（150ms 间隔）
        ui.timer(0.15, self._flush_buffer)

    def _flush_buffer(self):
        """将缓冲区中的日志批量渲染为 HTML 推送到前端"""
        if not self._buffer:
            return

        batch = self._buffer
        self._buffer = []

        self.lines.extend(batch)
        self._line_count += len(batch)

        # 将批量行转为 HTML
        html_parts = []
        for line in batch:
            html_line = self._converter.convert_line(line)
            html_parts.append(html_line)

        html_block = "<br>".join(html_parts) + "<br>"

        # 追加到容器（sanitize=False 关闭 DOMPurify，保留 <span style> 颜色）
        with self.log_container:
            ui.html(html_block, sanitize=False).style("display: inline;")

        # DOM 超过 max_lines 时清理最早的子元素
        if self._line_count > self.max_lines:
            overflow = self._line_count - self.max_lines
            ui.run_javascript(f'''
                (() => {{
                    const c = document.getElementById("c" + {self.log_container.id});
                    if (!c) return;
                    let n = Math.min({overflow}, c.children.length - 1);
                    while (n-- > 0 && c.firstChild) c.removeChild(c.firstChild);
                }})();
            ''')
            self._line_count = self.max_lines
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]

        # 自动滚动
        if self.auto_scroll:
            self.scroll_area.scroll_to(percent=1.0)

    def append(self, message: str, level: str = "info"):
        """添加日志行（写入缓冲区，由定时器批量推送）"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # 根据级别添加 ANSI 颜色
        level_colors = {
            "success": "\x1b[32m",   # 绿色
            "warning": "\x1b[33m",   # 黄色
            "error":   "\x1b[1;31m", # 粗体红色
        }
        color = level_colors.get(level, "")
        reset = "\x1b[0m" if color else ""
        formatted = f"{color}[{timestamp}] {message}{reset}"
        self._buffer.append(formatted)

    def info(self, message: str):
        """添加信息日志（直接缓冲原始文本，可能含 ANSI）"""
        self._buffer.append(message)

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
        self._buffer.clear()
        self._line_count = 0
        self._converter.reset()
        self.log_container.clear()
        self.info("日志已清空")

    def _save_log(self):
        """保存日志到文件（纯文本，无 ANSI）"""
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
                clean_lines = [strip_ansi(line) for line in self.lines]
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(clean_lines))
                self.success(f"日志已保存: {filename}")
            except Exception as e:
                self.error(f"保存失败: {e}")

    def _copy_all(self):
        """复制所有日志到剪贴板（纯文本，无 ANSI）"""
        clean_lines = [strip_ansi(line) for line in self.lines]
        text = "\n".join(clean_lines)
        escaped_text = text.replace("`", "\\`").replace("\\", "\\\\")
        ui.run_javascript(f"navigator.clipboard.writeText(`{escaped_text}`)")
        ui.notify("日志已复制到剪贴板", type="positive")

    def get_text(self) -> str:
        """获取所有日志纯文本"""
        return "\n".join(strip_ansi(line) for line in self.lines)


def create_log_viewer(**kwargs) -> LogViewer:
    """创建日志查看器的便捷函数"""
    return LogViewer(**kwargs)
