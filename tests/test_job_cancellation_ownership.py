import asyncio
import sys
from datetime import datetime

import pytest

from gui.components import execution_tabs as tabs_module
from gui.utils.i18n import t
from gui.utils.job_manager import JobManager, JobStatus
from gui.utils.log_buffer import log_buffer as global_log_buffer
from gui.utils.process_runner import ProcessRunner, ProcessStatus
from tests.test_execution_panel import _make_panel


def _make_shared_panel(store, tab_id):
    panel = _make_panel()
    panel._on_start = lambda: None
    tabs = tabs_module.ExecutionTabs.__new__(tabs_module.ExecutionTabs)
    tabs.tabs = store.tabs
    tabs._store = store
    tabs.active_tab_id = tab_id
    tabs._tab_bar = None
    tabs._on_tab_change = panel._handle_tab_change
    tabs._on_tab_log = None
    store.views.add(tabs)
    panel.execution_tabs = tabs
    return panel


def test_stop_keeps_tab_owned_until_spawned_process_is_reaped(monkeypatch, tmp_path):
    psutil = pytest.importorskip("psutil")
    (tmp_path / "ownership_worker.py").write_text(
        "import sys, time\n"
        "print('started', flush=True)\n"
        "if '--hold' in sys.argv: time.sleep(30)\n",
        encoding="utf-8",
    )
    manager = JobManager()
    notifications = []
    monkeypatch.setattr("gui.components.execution_panel.job_manager", manager)
    monkeypatch.setattr(tabs_module, "job_manager", manager)
    monkeypatch.setattr(tabs_module, "save_task_tabs", lambda tabs: None)
    monkeypatch.setattr(tabs_module, "_base_install_needed", lambda venv: False)
    monkeypatch.setattr(tabs_module.ui, "notify", lambda message, **kwargs: notifications.append(message))
    monkeypatch.setattr(ProcessRunner, "_find_uv", staticmethod(lambda: None))
    monkeypatch.setattr(ProcessRunner, "_requires_threaded_subprocess", staticmethod(lambda: False))
    store = tabs_module.TaskTabStore([
        tabs_module.TaskTab("tab-0001", "First", 1, ".", ".venv", None, datetime.now(), status="ready"),
        tabs_module.TaskTab("tab-0002", "Other", 2, ".", sys.prefix, sys.executable, datetime.now(), status="ready"),
    ])
    panel = _make_shared_panel(store, "tab-0001")
    other_panel = _make_shared_panel(store, "tab-0002")
    original_spawn = asyncio.create_subprocess_exec

    async def scenario():
        spawned = []
        first_spawned = asyncio.Event()
        other_spawned = asyncio.Event()
        release_first_handle = asyncio.Event()

        async def delayed_handoff(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            spawned.append(process)
            if len(spawned) == 1:
                first_spawned.set()
                await release_first_handle.wait()
            else:
                other_spawned.set()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_handoff)
        kwargs = {"cwd": str(tmp_path), "native_console": False, "python_path": sys.executable}
        first = asyncio.create_task(panel.run_job("ownership_worker", ["--hold"], "first", runner_kwargs=kwargs))
        other = None
        try:
            await asyncio.wait_for(first_spawned.wait(), 5)
            old_job = manager.get_all_jobs()[0]
            other = asyncio.create_task(other_panel.run_job("ownership_worker", ["--hold"], "other", runner_kwargs=kwargs))
            await asyncio.wait_for(other_spawned.wait(), 5)
            assert old_job.runner.process is None
            panel.cancel()
            panel.cancel()
            assert old_job in manager.get_active_jobs()
            assert old_job.finished_at is None
            assert not old_job._task.done()
            assert panel.execution_tabs.active_tab.current_job_id == old_job.id
            assert panel.start_btn.enabled is False
            assert notifications[-1] == t("task_stopping")
            rejected = await panel.run_job("ownership_worker", [], "rejected", runner_kwargs=kwargs)
            assert rejected.status is ProcessStatus.ERROR
            assert len(spawned) == 2
            assert all(psutil.pid_exists(process.pid) for process in spawned)

            release_first_handle.set()
            await asyncio.wait_for(first, 5)
            assert old_job.status is JobStatus.CANCELLED
            assert old_job.finished_at is not None
            assert old_job not in manager.get_active_jobs()
            assert not psutil.pid_exists(spawned[0].pid)
            assert psutil.pid_exists(spawned[1].pid)
            assert panel.start_btn.enabled is True
            assert other_panel.start_btn.enabled is False
            retried = await panel.run_job("ownership_worker", [], "retry", runner_kwargs=kwargs)
            assert retried.status is ProcessStatus.SUCCESS
        finally:
            release_first_handle.set()
            first.cancel()
            if other is not None:
                other.cancel()
            await asyncio.gather(first, *([other] if other is not None else []), return_exceptions=True)
            for process in spawned:
                if process.returncode is None:
                    process.kill()
                await process.wait()

    asyncio.run(scenario())


def test_cancel_before_first_task_step_finishes_and_unsubscribes_logs(monkeypatch):
    manager = JobManager()
    started = []

    async def run(self, *args, **kwargs):
        started.append(True)
        await asyncio.Event().wait()

    monkeypatch.setattr(ProcessRunner, "run_python_script", run)

    async def scenario():
        job = await manager.submit("module.captioner", [], "not-started", tab_id="tab-0001")
        assert manager.cancel(job.id)
        assert manager.cancel(job.id)
        assert job in manager.get_active_jobs()
        assert job.finished_at is None
        manager.remove_job(job.id)
        assert manager.get_job(job.id) is job
        with pytest.raises(asyncio.CancelledError):
            await job.wait()
        assert started == []
        assert job.status is JobStatus.CANCELLED
        assert job.finished_at is not None
        assert job.result.status is ProcessStatus.ERROR
        assert manager.get_active_jobs() == []
        marker = "after-cancel-forwarding-must-be-unsubscribed"
        job.log_buffer.push(marker)
        assert all(marker not in line for _, line in global_log_buffer.get_all_lines())
        manager.remove_job(job.id)
        assert manager.get_job(job.id) is None

    asyncio.run(scenario())


def test_repeated_job_cancel_does_not_interrupt_inflight_cleanup(monkeypatch):
    manager = JobManager()

    async def scenario():
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()

        async def run(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()
                raise

        monkeypatch.setattr(ProcessRunner, "run_python_script", run)
        job = await manager.submit("module.captioner", [], "cleanup", tab_id="tab-0001")
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert manager.cancel(job.id)
            await asyncio.wait_for(cleanup_started.wait(), 2)
            assert manager.cancel(job.id)
            assert job in manager.get_active_jobs()
            assert job.finished_at is None
            with pytest.raises(RuntimeError, match="Tab already has an active job"):
                await manager.submit("module.captioner", [], "too-early", tab_id=job.tab_id)
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await job.wait()
            assert cleanup_finished.is_set()
            assert job.status is JobStatus.CANCELLED
            assert manager.get_active_jobs() == []
        finally:
            release_cleanup.set()
            job._task.cancel()
            await asyncio.gather(job._task, return_exceptions=True)

    asyncio.run(scenario())
