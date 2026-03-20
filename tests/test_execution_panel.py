import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gui.components.execution_panel import ExecutionPanel
from gui.utils.process_runner import ProcessResult, ProcessStatus


class _DummyControl:
    def __init__(self):
        self.enabled = None
        self.visible = None

    def set_enabled(self, value):
        self.enabled = value

    def set_visibility(self, value):
        self.visible = value


class _NoClearLogViewer:
    def __init__(self):
        self.reset_display_calls = 0
        self.attached_jobs = []
        self.success_messages = []
        self.error_messages = []

    def reset_display(self):
        self.reset_display_calls += 1

    def clear(self):
        raise AssertionError("run_job should not clear the backing log source")

    def attach_job(self, job):
        self.attached_jobs.append(job)

    def success(self, message):
        self.success_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)


def _make_panel():
    panel = ExecutionPanel.__new__(ExecutionPanel)
    panel.is_running = False
    panel.current_job = None
    panel._on_start = None
    panel.stop_btn = _DummyControl()
    panel.start_btn = None
    panel.progress = _DummyControl()
    panel._external_start_buttons = []
    panel.log_viewer = _NoClearLogViewer()
    return panel


def test_set_running_disables_registered_external_start_buttons():
    panel = _make_panel()
    extra_one = _DummyControl()
    extra_two = _DummyControl()
    panel._external_start_buttons = [extra_one, extra_two]

    ExecutionPanel._set_running(panel, True)

    assert extra_one.enabled is False
    assert extra_two.enabled is False


def test_run_job_rejects_reentry(monkeypatch):
    panel = _make_panel()
    panel.is_running = True
    submit_calls = []
    notifications = []

    async def fake_submit(*args, **kwargs):
        submit_calls.append((args, kwargs))
        raise AssertionError("submit should not be called while a job is already running")

    monkeypatch.setattr("gui.components.execution_panel.job_manager.submit", fake_submit)
    monkeypatch.setattr("gui.components.execution_panel.ui.notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    result = asyncio.run(panel.run_job("module.captioner", ["--demo"], name="Caption"))

    assert result.status == ProcessStatus.ERROR
    assert submit_calls == []
    assert notifications
    assert notifications[-1][1]["type"] == "warning"


def test_run_job_resets_display_without_clearing_log_source(monkeypatch):
    panel = _make_panel()
    result = ProcessResult(ProcessStatus.SUCCESS, 0, "ok")

    class _FakeJob:
        def __init__(self, result):
            self.result = result
            self.id = "job-1"

        async def wait(self):
            return self.result

    async def fake_submit(*args, **kwargs):
        return _FakeJob(result)

    monkeypatch.setattr("gui.components.execution_panel.job_manager.submit", fake_submit)
    monkeypatch.setattr("gui.components.execution_panel.ui.notify", lambda *args, **kwargs: None)

    returned = asyncio.run(panel.run_job("module.captioner", ["--demo"], name="Caption"))

    assert returned is result
    assert panel.log_viewer.reset_display_calls == 1
    assert len(panel.log_viewer.attached_jobs) == 1
    assert panel.log_viewer.success_messages
