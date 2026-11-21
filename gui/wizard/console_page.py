"""/console 全屏终端页面 - 新标签页打开，不带导航栏

全屏控制台，使用主题 CSS 变量，实时 ANSI → HTML 渲染。
"""

from collections import deque
from collections.abc import Iterable

from nicegui import ui, context
from gui.theme import apply_theme
from gui.utils.ansi_to_html import AnsiToHtmlConverter
from gui.utils.log_buffer import log_buffer
from gui.utils.i18n import t

_MAX_CONSOLE_LINES = 2000
_LINES_PER_BLOCK = 64


def render_console_page():
    """渲染 /console 全屏终端页面"""
    apply_theme()

    converter = AnsiToHtmlConverter()
    auto_scroll = {"value": True}
    sub_id = {"value": None}
    buffer = {"lines": []}
    blocks: deque[tuple[ui.html, list[str]]] = deque()
    visible_count = 0
    last_seq = {"value": 0}
    flush_timer = {"ref": None}

    # 页面样式 — 使用主题 CSS 变量，不再硬编码 GitHub-dark 色板
    ui.add_head_html("""
    <style>
        body {
            margin: 0;
            padding: 0;
            background: var(--color-bg);
            overflow: hidden;
        }
        .console-toolbar {
            background: var(--color-surface);
            border-bottom: 1px solid var(--color-border);
            padding: 8px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 48px;
            box-sizing: border-box;
            gap: 12px;
        }
        .console-title {
            color: var(--color-text);
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
        .console-title .dot-green { background: var(--color-success); }
        .console-body {
            font-family: 'Cascadia Code', 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: var(--ql-console-text);
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
            back_btn.props("flat dense").style("color: var(--color-text-secondary);")
            back_btn.tooltip(t("close", "Close"))

            ui.html(f'<span class="console-title"><span class="dot dot-green"></span>{t("console")}</span>')

        # 右侧: 自动滚动开关 + 清屏
        with ui.row().classes("items-center gap-3"):
            scroll_switch = ui.switch(t("auto_scroll"), value=True).props("dense")
            scroll_switch.style("color: var(--color-text); font-size: 12px;")

            def on_scroll_change(e):
                auto_scroll["value"] = e.value

            scroll_switch.on_value_change(on_scroll_change)

            def clear_display():
                nonlocal visible_count
                blocks.clear()
                visible_count = 0
                buffer["lines"].clear()
                log_container.clear()

            clear_btn = ui.button(icon="delete_sweep", on_click=clear_display)
            clear_btn.props("flat dense").style("color: var(--color-text-secondary);")
            clear_btn.tooltip(t("clear_screen"))

    # 终端区域 — 背景色与 log_viewer 保持一致
    scroll_area = ui.scroll_area().classes("w-full").style(
        "height: calc(100vh - 48px); background: var(--ql-console-bg);"
    )
    with scroll_area:
        log_container = ui.element("div").classes("console-body")

    def append_lines(lines: Iterable[str]) -> None:
        nonlocal visible_count
        # Convert in stream order, including discarded lines that can change the ANSI style.
        fresh = list(deque((converter.convert_line(line) + "<br>" for line in lines), maxlen=_MAX_CONSOLE_LINES))
        if not fresh:
            return
        visible_count += len(fresh)
        while visible_count > _MAX_CONSOLE_LINES:
            node, rows = blocks[0]
            drop = min(len(rows), visible_count - _MAX_CONSOLE_LINES)
            visible_count -= drop
            if drop == len(rows):
                blocks.popleft()
                node.delete()
            else:
                del rows[:drop]
                node.set_content("".join(rows))

        # Only the two boundary blocks can change; all interior HTML stays on the client.
        if blocks and len(blocks[-1][1]) < _LINES_PER_BLOCK:
            node, rows = blocks[-1]
            room = _LINES_PER_BLOCK - len(rows)
            rows.extend(fresh[:room])
            node.set_content("".join(rows))
            fresh = fresh[room:]
        with log_container:
            for i in range(0, len(fresh), _LINES_PER_BLOCK):
                rows = fresh[i:i + _LINES_PER_BLOCK]
                blocks.append((ui.html("".join(rows), sanitize=False), rows))

    # 先订阅再回放历史，避免回放与订阅间隙丢行
    def on_new_line(seq: int, line: str):
        buffer["lines"].append((seq, line))

    sub_id["value"] = log_buffer.subscribe(on_new_line)

    # -- 回放已有日志（仅保留最近 _MAX_CONSOLE_LINES 行） --
    history = log_buffer.get_all_lines()
    if history:
        last_seq["value"] = history[-1][0]
        append_lines(line for _, line in history)
        scroll_area.scroll_to(percent=1.0)

    # -- 定时批量刷新（0.1s，全屏控制台可更激进，受行数与块数上限约束）--
    def flush():
        pending = buffer["lines"]
        if not pending:
            return
        buffer["lines"] = []

        fresh_lines = []
        for seq, line in pending:
            if seq <= last_seq["value"]:
                continue
            last_seq["value"] = seq
            fresh_lines.append(line)

        if not fresh_lines:
            return

        try:
            append_lines(fresh_lines)
            if auto_scroll["value"]:
                scroll_area.scroll_to(percent=1.0)
        except RuntimeError:
            pass

    flush_timer["ref"] = ui.timer(0.1, flush)

    # -- 断开时清理订阅 --
    def cleanup():
        sid = sub_id.get("value") if isinstance(sub_id, dict) else None
        if sid is not None:
            log_buffer.unsubscribe(sid)
            sub_id["value"] = None
        if flush_timer["ref"] is not None:
            flush_timer["ref"].active = False

    context.client.on_disconnect(cleanup)
