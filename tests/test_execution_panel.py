import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gui.components.execution_panel import ExecutionPanel
from gui.utils.i18n import t
from gui.utils.job_manager import JobStatus
from gui.utils.log_buffer import LogBuffer
from gui.utils.process_runner import ProcessResult, ProcessStatus


class _DummyControl:
    def __init__(self):
        self.enabled = None
        self.visible = None
        self.text = None
        self.styles = []

    def set_enabled(self, value):
        self.enabled = value

    def set_visibility(self, value):
        self.visible = value

    def set_text(self, value):
        self.text = value

    def style(self, value):
        self.styles.append(value)


class _DummyExecutionTabs:
    def __init__(self):
        self.active_tab = SimpleNamespace(
            id="tab-0001",
            name="默认任务",
            status="ready",
            venv_path=".venv",
            error_message=None,
            current_job_id=None,
        )
        self.marked_jobs = []
        self.cleared_jobs = []
        self.ensure_calls = 0
        self.ensure_ready = True

    def runner_kwargs(self):
        return {"tab_id": self.active_tab.id, "tab_name": self.active_tab.name}

    async def ensure_active_tab_runtime_ready(self):
        self.ensure_calls += 1
        return self.ensure_ready

    def active_tab_can_start(self):
        return self.active_tab.status == "ready" and self.active_tab.current_job_id is None

    def mark_job(self, job):
        self.active_tab.current_job_id = job.id
        self.marked_jobs.append(job.id)

    def clear_job(self, job_id):
        if self.active_tab.current_job_id == job_id:
            self.active_tab.current_job_id = None
        self.cleared_jobs.append(job_id)

    async def retry_active_tab(self):
        return None


class _NoClearLogViewer:
    def __init__(self):
        self.reset_display_calls = 0
        self.attached_jobs = []
        self.attached_sources = []

    def reset_display(self):
        self.reset_display_calls += 1

    def clear(self):
        raise AssertionError("run_job should not clear the backing log source")

    def attach_job(self, job):
        self.attached_jobs.append(job)

    def attach_log_source(self, log_source):
        self.attached_sources.append(log_source)


def _make_panel():
    panel = ExecutionPanel.__new__(ExecutionPanel)
    panel.is_running = False
    panel.current_job = None
    panel._on_start = None
    panel._action_enabled = True
    panel._tab_current_jobs = {}
    panel._tab_last_jobs = {}
    panel._tab_log_buffers = {}
    panel._external_start_buttons = []
    panel._active_jobs = set()
    panel.execution_tabs = _DummyExecutionTabs()
    panel.stop_btn = _DummyControl()
    panel.start_btn = _DummyControl()
    panel.retry_btn = _DummyControl()
    panel.progress = _DummyControl()
    panel.action_label = _DummyControl()
    panel.status_label = _DummyControl()
    panel.log_viewer = _NoClearLogViewer()
    return panel


def test_set_running_keeps_registered_external_start_buttons_enabled():
    panel = _make_panel()
    extra_one = _DummyControl()
    extra_two = _DummyControl()
    panel._external_start_buttons = [extra_one, extra_two]

    ExecutionPanel._set_running(panel, True)

    assert extra_one.enabled is None
    assert extra_two.enabled is None


def test_set_action_updates_start_button_and_disabled_action_does_not_run(monkeypatch):
    panel = _make_panel()
    calls = []
    notifications = []

    monkeypatch.setattr("gui.components.execution_panel.ui.notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    panel.set_action("Run Tool", lambda: calls.append("called"), enabled=False)
    asyncio.run(panel._handle_start())

    assert panel.action_label.text == "Run Tool"
    assert panel.start_btn.text == "Run Tool"
    assert panel.start_btn.enabled is False
    assert calls == []
    assert notifications


def test_legacy_on_start_assignment_enables_default_active_tab_start_button():
    panel = _make_panel()
    panel._controls_ready = True
    panel.start_btn.enabled = False

    panel._on_start = lambda: None

    assert panel.start_btn.enabled is True


def test_run_job_rejects_same_active_tab_reentry(monkeypatch):
    panel = _make_panel()
    running_job = SimpleNamespace(id="job-running", status=JobStatus.RUNNING)
    panel._tab_current_jobs["tab-0001"] = running_job.id
    submit_calls = []
    notifications = []

    async def fake_submit(*args, **kwargs):
        submit_calls.append((args, kwargs))
        raise AssertionError("submit should not be called while active tab is running")

    monkeypatch.setattr("gui.components.execution_panel.job_manager.get_job", lambda job_id: running_job)
    monkeypatch.setattr("gui.components.execution_panel.job_manager.submit", fake_submit)
    monkeypatch.setattr("gui.components.execution_panel.ui.notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    result = asyncio.run(panel.run_job("module.captioner", ["--demo"], name="Caption"))

    assert result.status == ProcessStatus.ERROR
    assert submit_calls == []
    assert notifications
    assert notifications[-1][1]["type"] == "warning"


def test_run_job_refreshes_state_when_runtime_prepare_fails(monkeypatch):
    panel = _make_panel()
    panel.execution_tabs.ensure_ready = False
    panel.execution_tabs.active_tab.status = "error"
    submit_calls = []

    async def fake_submit(*args, **kwargs):
        submit_calls.append((args, kwargs))
        raise AssertionError("submit should not run when runtime prepare fails")

    monkeypatch.setattr("gui.components.execution_panel.job_manager.submit", fake_submit)

    result = asyncio.run(panel.run_job("module.captioner", ["--demo"], name="Caption"))

    assert result.status == ProcessStatus.ERROR
    assert submit_calls == []
    assert panel.retry_btn.visible is True
    assert panel.status_label.text.startswith("默认任务 | error")


def test_run_job_attaches_job_log_without_clearing_source(monkeypatch):
    panel = _make_panel()
    result = ProcessResult(ProcessStatus.SUCCESS, 0, "ok")

    class _FakeJob:
        def __init__(self, result):
            self.result = result
            self.id = "job-1"
            self.name = "Caption"
            self.tab_id = "tab-0001"
            self.status = JobStatus.RUNNING
            self.log_buffer = LogBuffer()

        async def wait(self):
            self.status = JobStatus.SUCCESS
            return self.result

    fake_job = _FakeJob(result)

    async def fake_submit(*args, **kwargs):
        return fake_job

    monkeypatch.setattr("gui.components.execution_panel.job_manager.submit", fake_submit)
    monkeypatch.setattr("gui.components.execution_panel.job_manager.get_job", lambda job_id: fake_job if job_id == fake_job.id else None)
    monkeypatch.setattr("gui.components.execution_panel.ui.notify", lambda *args, **kwargs: None)

    returned = asyncio.run(
        panel.run_job(
            "module.captioner",
            ["--demo"],
            name="Caption",
            pre_log=lambda lv: lv.info("pre-log"),
        )
    )

    log_lines = [line for _seq, line in fake_job.log_buffer.get_all_lines()]

    assert returned is result
    assert panel.log_viewer.reset_display_calls == 0
    assert fake_job in panel.log_viewer.attached_jobs
    assert "pre-log" in log_lines
    assert any(t("task_finished") in line for line in log_lines)
