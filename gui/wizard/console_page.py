"""/console 全屏终端页面 - 新标签页打开，不带导航栏

全屏控制台，使用主题 CSS 变量，实时 ANSI → HTML 渲染。
"""

from nicegui import ui, context
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

    # 页面样式 — 使用主题 CSS 变量，不再硬编码 GitHub-dark 色板
    ui.add_head_html("""
    <style>
        body {
            margin: 0;
            padding: 0;
            background: var(--color-bg, rgba(15, 23, 42, 1));
            overflow: hidden;
        }
        .console-toolbar {
            background: var(--color-surface, rgba(22, 27, 34, 0.95));
            border-bottom: 1px solid var(--color-border, rgba(48, 54, 61, 0.8));
            padding: 8px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 48px;
            box-sizing: border-box;
            gap: 12px;
        }
        .console-title {
            color: var(--color-text, #e5e5e5);
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
        .console-title .dot-green { background: var(--color-success, #10b981); }
        .console-body {
            font-family: 'Cascadia Code', 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: #e5e5e5;
            padding: 12px 16px;
            white-space: pre-wrap;
            word-break: break-all;
        }
    </style>
    """)

    # 工具栏
    with ui.element("div").classes("console-toolbar"):
        # 左侧: 返回/关闭按钮 + 标题
        with ui.row().classes("items-center gap-2"):
            back_btn = ui.button(
                icon="arrow_back",
                on_click=lambda: ui.run_javascript("""
                    (() => {
                        if (window.opener && !window.opener.closed) {
                            window.close();
                            return;
                        }
                        if (window.history.length > 1) {
                            window.history.back();
                            return;
                        }
                        window.location.href = '/';
                    })();
                """),
            )
            back_btn.props("flat dense").style("color: var(--color-text-secondary, #94a3b8);")
            back_btn.tooltip(t("close", "Close"))

            ui.html('<span class="console-title"><span class="dot dot-green"></span>Console</span>')

        # 右侧: 自动滚动开关 + 清屏
        with ui.row().classes("items-center gap-3"):
            scroll_switch = ui.switch("Auto Scroll", value=True).props("dense")
            scroll_switch.style("color: var(--color-text, #e5e5e5); font-size: 12px;")

            def on_scroll_change(e):
                auto_scroll["value"] = e.value
            scroll_switch.on_value_change(on_scroll_change)

            clear_btn = ui.button(icon="delete_sweep", on_click=lambda: log_container.clear())
            clear_btn.props("flat dense").style("color: var(--color-text-secondary, #94a3b8);")
            clear_btn.tooltip("Clear Screen")

    # 终端区域 — 背景色与 log_viewer 保持一致
    scroll_area = ui.scroll_area().classes("w-full").style(
        "height: calc(100vh - 48px); background: rgba(15, 23, 42, 0.8);"
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

    # -- 定时批量刷新（0.1s，全屏控制台可更激进）--
    def flush():
        lines = buffer["lines"]
        if not lines:
            return
        buffer["lines"] = []

        html_parts = []
        for line in lines:
            html_parts.append(converter.convert_line(line))

        try:
            with log_container:
                ui.html("<br>".join(html_parts) + "<br>", sanitize=False)

            if auto_scroll["value"]:
                scroll_area.scroll_to(percent=1.0)
        except RuntimeError:
            pass

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
        if flush_timer["ref"] is not None:
            flush_timer["ref"].active = False

    context.client.on_disconnect(cleanup)
