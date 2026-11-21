from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_apply_theme_is_idempotent(monkeypatch):
    import gui.theme as theme_mod

    head_calls = []
    css_calls = []
    monkeypatch.setattr(theme_mod, "_theme_applied", False)
    monkeypatch.setattr(theme_mod.ui, "add_head_html", lambda html, **kw: head_calls.append((html, kw)))
    monkeypatch.setattr(theme_mod.ui, "add_css", lambda css, **kw: css_calls.append((css, kw)))

    for _ in range(5):
        theme_mod.apply_theme()

    assert len(head_calls) == 2
    assert len(css_calls) == 1
    assert all(kw.get("shared") is True for _, kw in head_calls)
    assert css_calls[0][1].get("shared") is True


def test_first_page_includes_lazy_execution_panel_styles_before_client_connects(monkeypatch):
    from nicegui import Client, ui
    from gui import theme

    monkeypatch.setattr(Client, "shared_head_html", "")
    monkeypatch.setattr(theme, "_theme_applied", False)
    for path in ["/caption", "/import"]:
        client = Client(ui.page(path))
        try:
            with client:
                theme.apply_theme()
            assert client.head_html.count(".task-tab-btn--active") == 1
            assert client.head_html.count(".log-scroll-area .q-scrollarea__thumb--v") == 1
        finally:
            client.delete()


@pytest.fixture
def live_console(monkeypatch):
    from nicegui import Client, ui
    from gui.utils.log_buffer import LogBuffer
    from gui.wizard import console_page

    source = LogBuffer()
    timers = []
    monkeypatch.setattr(console_page, "log_buffer", source)
    monkeypatch.setattr(console_page, "apply_theme", lambda: None)
    monkeypatch.setattr(ui, "timer", lambda _, cb: timers.append(cb) or SimpleNamespace(active=True))
    monkeypatch.setattr(ui.scroll_area, "scroll_to", lambda *_a, **_kw: None)
    client = Client(ui.page("/console"))
    def open_console(history=()):
        for line in history:
            source.push(line)
        with client:
            console_page.render_console_page()
        body = next(el for el in client.elements.values() if "console-body" in el.classes)
        return source, timers[0], client, body

    try:
        yield open_console
    finally:
        client.delete()


def _console_rows(body):
    return [row for block in body.default_slot.children for row in block.content.split("<br>")[:-1]]


def test_full_console_appends_without_resending_the_entire_window(live_console):
    from nicegui import ui

    source, flush, client, body = live_console()
    for i in range(2000):
        source.push(str(i))
    flush()
    for i in [2000, 2001, 2002]:
        client.outbox.updates.clear()
        source.push(str(i))
        flush()
        transmitted = [el.content for el in client.outbox.updates.values() if isinstance(el, ui.html)]
        assert 0 < sum(block.count("<br>") for block in transmitted) < 200
    assert _console_rows(body) == [str(i) for i in range(3, 2003)]


def test_eviction_preserves_ansi_history_and_continues_the_live_color(live_console):
    source, flush, _, body = live_console()
    for i in range(1999):
        source.push(str(i))
    source.push("\x1b[31mred")
    flush()
    source.push("still red")
    flush()
    rows = _console_rows(body)
    assert rows[0] == "1"
    assert rows[-2:] == [
        '<span style="color:#cd0000 !important">red</span>',
        '<span style="color:#cd0000 !important">still red</span>',
    ]


def test_console_limits_history_and_large_batches_without_losing_ansi_state(live_console):
    history = ["\x1b[31mstart red", *map(str, range(2000))]
    source, flush, _, body = live_console(history)
    rows = _console_rows(body)
    assert len(rows) == 2000
    assert rows[0] == '<span style="color:#cd0000 !important">0</span>'

    source.push("\x1b[0mreset")
    for i in range(3000):
        source.push(str(i))
    flush()
    assert _console_rows(body) == [str(i) for i in range(1000, 3000)]


def test_clipboard_copy_helpers_safely_encode_js_payloads(monkeypatch):
    from gui.components import log_viewer as lv_mod
    from gui.components import path_selector as ps_mod

    js_calls = []
    monkeypatch.setattr(lv_mod.ui, "run_javascript", lambda code: js_calls.append(code))
    monkeypatch.setattr(lv_mod.ui, "notify", lambda *_a, **_kw: None)
    monkeypatch.setattr(ps_mod.ui, "run_javascript", lambda code: js_calls.append(code))
    monkeypatch.setattr(ps_mod.ui, "notify", lambda *_a, **_kw: None)

    payload = r"C:\dir\it`s ${alert(1)}\\"
    viewer = lv_mod.LogViewer.__new__(lv_mod.LogViewer)
    viewer.lines = [payload]
    viewer._copy_all()

    selector = ps_mod.PathSelector.__new__(ps_mod.PathSelector)
    selector.input = SimpleNamespace(value=payload)
    selector._copy_path()

    expected = f"navigator.clipboard.writeText({json.dumps(payload)})"
    assert js_calls == [expected, expected]


def test_ps_escape_escapes_unicode_single_quotes():
    from gui.utils.process_runner import _ps_escape

    raw = "a'b\u2018c\u2019d\u201ae\u201bf"
    assert _ps_escape(raw) == "a''b\u2018\u2018c\u2019\u2019d\u201a\u201ae\u201b\u201bf"


def test_run_python_script_rejects_path_traversal_script_key(tmp_path):
    from gui.utils.process_runner import ProcessRunner, ProcessStatus

    runner = ProcessRunner()
    result = asyncio.run(
        runner.run_python_script("../outside_script", [], cwd=str(tmp_path), native_console=False)
    )
    assert result.status is ProcessStatus.ERROR
    assert "Invalid unregistered script key" in result.message


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native console path")
def test_run_native_bypasses_powershell_and_preserves_smart_quote_args(monkeypatch, tmp_path):
    import gui.utils.process_runner as pr_mod
    from gui.utils.log_buffer import LogBuffer

    launched_cmds = []

    class FakeProc:
        pid = 9999
        returncode = None

        def poll(self):
            return self.returncode

    def fake_popen(wrapper_cmd, **_kwargs):
        launched_cmds.append(list(wrapper_cmd))
        exit_file = Path(wrapper_cmd[2])
        log_file = Path(wrapper_cmd[3])

        async def finish():
            log_file.write_text("ok\n", encoding="utf-8")
            exit_file.write_text("0", encoding="utf-8")

        asyncio.create_task(finish())
        return FakeProc()

    monkeypatch.setattr(pr_mod.subprocess, "Popen", fake_popen)

    async def scenario():
        runner = pr_mod.ProcessRunner(log_buffer=LogBuffer())
        runner._running = True
        tricky_arg = "dir\u2019; calc.exe #"
        rc = await runner._run_native(["python", "worker.py", tricky_arg], tmp_path, {})
        assert rc == 0
        assert launched_cmds[0][0] == sys.executable
        assert launched_cmds[0][1] == pr_mod._WRAPPER_PATH
        assert "pwsh" not in launched_cmds[0][0].lower()
        assert "powershell" not in launched_cmds[0][0].lower()
        assert launched_cmds[0][-1] == tricky_arg

    asyncio.run(scenario())
