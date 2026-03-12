"""/console 全屏终端页面 - 新标签页打开，不带导航栏

GitHub 暗色终端风格，实时 ANSI → HTML 渲染。
"""

from nicegui import ui, app
from gui.utils.ansi_to_html import AnsiToHtmlConverter
from gui.utils.log_buffer import log_buffer
from gui.utils.i18n import t


def render_console_page():
    """渲染 /console 全屏终端页面"""

    converter = AnsiToHtmlConverter()
    auto_scroll = {"value": True}
    sub_id = {"value": None}
    buffer = {"lines": []}
    flush_timer = {"ref": None}

    # 页面样式
    ui.add_head_html("""
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #0d1117;
            overflow: hidden;
        }
        .console-toolbar {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 8px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 48px;
            box-sizing: border-box;
        }
        .console-title {
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 14px;
            font-weight: 600;
        }
        .console-title .dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .console-title .dot-green { background: #3fb950; }
        .console-body {
            font-family: 'Cascadia Code', 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: #c9d1d9;
            padding: 12px 16px;
            white-space: pre-wrap;
            word-break: break-all;
        }
    </style>
    """)

    # 工具栏
    with ui.element("div").classes("console-toolbar"):
        with ui.row().classes("items-center gap-2"):
            ui.html('<span class="console-title"><span class="dot dot-green"></span>Console</span>')
        with ui.row().classes("items-center gap-3"):
            # 自动滚动
            scroll_switch = ui.switch("Auto Scroll", value=True).props("dense dark color=green")
            scroll_switch.style("color: #c9d1d9; font-size: 12px;")

            def on_scroll_change(e):
                auto_scroll["value"] = e.value
            scroll_switch.on_value_change(on_scroll_change)

            # 清屏
            clear_btn = ui.button(icon="delete_sweep", on_click=lambda: log_container.clear())
            clear_btn.props("flat dense dark").style("color: #8b949e;")
            clear_btn.tooltip("Clear Screen")

    # 终端区域
    scroll_area = ui.scroll_area().classes("w-full").style(
        "height: calc(100vh - 48px); background: #0d1117;"
    )
    with scroll_area:
        log_container = ui.element("div").classes("console-body")

    # -- 回放已有日志 --
    history = log_buffer.get_all_lines()
    if history:
        html_parts = []
        for _seq, line in history:
            html_parts.append(converter.convert_line(line))
        with log_container:
            ui.html("<br>".join(html_parts) + "<br>", sanitize=False)
        scroll_area.scroll_to(percent=1.0)

    # -- 定时批量刷新 --
    def flush():
        lines = buffer["lines"]
        if not lines:
            return
        buffer["lines"] = []

        html_parts = []
        for line in lines:
            html_parts.append(converter.convert_line(line))

        with log_container:
            ui.html("<br>".join(html_parts) + "<br>", sanitize=False)

        if auto_scroll["value"]:
            scroll_area.scroll_to(percent=1.0)

    flush_timer["ref"] = ui.timer(0.1, flush)

    # -- 订阅实时日志 --
    def on_new_line(seq: int, line: str):
        buffer["lines"].append(line)

    sub_id["value"] = log_buffer.subscribe(on_new_line)

    # -- 断开时清理订阅 --
    def cleanup():
        sid = sub_id.get("value") if isinstance(sub_id, dict) else None
        if sid is not None:
            log_buffer.unsubscribe(sid)
            sub_id["value"] = None

    app.on_disconnect(cleanup)
