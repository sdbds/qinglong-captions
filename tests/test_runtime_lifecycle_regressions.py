import asyncio
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gui.components import execution_tabs as tabs_module
from gui.utils.job_manager import JobManager, JobStatus, job_manager
from gui.utils.log_buffer import LogBuffer
from gui.utils.process_runner import ProcessResult, ProcessRunner, ProcessStatus
from module.onnx_runtime import session as session_module
from module.onnx_runtime.config import OnnxRuntimeConfig
from module.onnx_runtime.multi_model import OnnxMultiModelSpec, load_multi_model_bundle
from module.onnx_runtime.single_model import OnnxModelSpec, load_single_model_bundle
from tests.test_execution_panel import _make_panel


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("refresh", ["force_download", "revision"])
def test_refreshed_artifacts_do_not_reuse_old_sessions(tmp_path, multi, refresh):
    session_module.clear_session_bundle_cache()
    created = []
    closed = []
    version = "v1"

    def artifact_loader(*args, **kwargs):
        path = tmp_path / "model.onnx"
        path.write_text(version, encoding="utf-8")
        return {"model": path} if multi else path

    def session_factory(path, **kwargs):
        session = SimpleNamespace(
            version=Path(path).read_text(encoding="utf-8"),
            get_inputs=lambda: [],
            close=lambda: closed.append(path),
        )
        created.append(session)
        return session

    def session_loader(**kwargs):
        return session_module.load_session_bundle(
            **kwargs, available_providers=["CPUExecutionProvider"],
            session_factory=session_factory, session_options_factory=lambda: None,
        )

    def load(revision=None, force=False):
        common = dict(repo_id="audit/model", local_dir=tmp_path, bundle_key="audit", revision=revision)
        if multi:
            spec = OnnxMultiModelSpec(artifacts={"model": "model.onnx"}, **common)
            loader = load_multi_model_bundle
        else:
            spec = OnnxModelSpec(onnx_filename="model.onnx", **common)
            loader = load_single_model_bundle
        return loader(spec=spec, runtime_config=OnnxRuntimeConfig(force_download=force),
                      artifact_loader=artifact_loader, session_bundle_loader=session_loader)

    first = load("v1" if refresh == "revision" else None)
    version = "v2"
    second = load("v2" if refresh == "revision" else None, refresh == "force_download")
    assert second.session_bundle.sessions["model"].version == "v2"
    assert first.session_bundle.sessions["model"].version == "v1"
    assert len(created) == 2
    assert closed == []
    # The cache still owns entries; discard them without exercising explicit close.
    session_module._SESSION_BUNDLE_CACHE.clear()


def test_observer_keeps_completed_job_log(monkeypatch):
    panel = _make_panel()
    job = SimpleNamespace(id="observed", tab_id="tab-0001", status=JobStatus.RUNNING,
                          log_buffer=LogBuffer())
    job.log_buffer.push("failure diagnostic")
    monkeypatch.setattr(job_manager, "get_active_jobs", lambda: [job] if job.status == JobStatus.RUNNING else [])
    monkeypatch.setattr(job_manager, "get_job", lambda job_id: job if job_id == job.id else None)
    panel._attach_log_for_active_tab()
    job.status = JobStatus.ERROR
    panel._attach_log_for_active_tab()
    assert panel._job_for_tab(job.tab_id) is job
    assert panel.log_viewer.attached_sources == []


def test_failed_force_refresh_does_not_leave_stale_session_cached():
    session_module.clear_session_bundle_cache()
    version = "v1"
    fail = False

    def factory(*args, **kwargs):
        if fail:
            raise RuntimeError("refresh failed")
        return SimpleNamespace(version=version)

    def load(force=False):
        return session_module.load_session_bundle(
            bundle_key="failed-refresh", session_paths={"model": "model.onnx"},
            runtime_config=OnnxRuntimeConfig(force_download=force),
            available_providers=["CPUExecutionProvider"], session_factory=factory,
            session_options_factory=lambda: None,
        )

    first = load()
    version = "v2"
    fail = True
    with pytest.raises(RuntimeError, match="refresh failed"):
        load(force=True)
    fail = False
    assert load().sessions["model"].version == "v2"
    assert first.sessions["model"].version == "v1"


def test_inflight_old_load_cannot_replace_force_refreshed_cache():
    session_module.clear_session_bundle_cache()
    started, release = threading.Event(), threading.Event()

    def old_factory(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(version="v1")

    def load(factory, force=False):
        return session_module.load_session_bundle(
            bundle_key="concurrent-refresh", session_paths={"model": "model.onnx"},
            runtime_config=OnnxRuntimeConfig(force_download=force),
            available_providers=["CPUExecutionProvider"], session_factory=factory,
            session_options_factory=lambda: None,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(load, old_factory)
        try:
            assert started.wait(5)
            refreshed = load(lambda *a, **k: SimpleNamespace(version="v2"), force=True)
        finally:
            release.set()
        assert pending.result(timeout=5).sessions["model"].version == "v1"
    assert load(lambda *a, **k: pytest.fail("cache should contain v2")) is refreshed


def test_another_page_can_stop_preparation_and_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(tabs_module, "save_task_tabs", lambda tabs: None)
    monkeypatch.setattr(tabs_module.ui, "notify", lambda *a, **k: None)
    monkeypatch.setattr(job_manager, "get_active_jobs", lambda: [])
    venv = tmp_path / "runtime"
    python = tabs_module._python_for_venv(venv)
    python.parent.mkdir(parents=True)
    python.touch()
    tab = tabs_module.TaskTab("tab-0002", "Second", 2, ".", str(venv), str(python), datetime.now())
    store = tabs_module.TaskTabStore([tab])

    def make_panel():
        panel = _make_panel()
        tabs = tabs_module.ExecutionTabs.__new__(tabs_module.ExecutionTabs)
        tabs.tabs = store.tabs
        tabs._store = store
        tabs.active_tab_id = tab.id
        tabs._tab_bar = None
        tabs._on_tab_change = panel._handle_tab_change
        tabs._on_tab_log = None
        store.views.add(tabs)
        panel.execution_tabs = tabs
        return panel

    owner, observer = make_panel(), make_panel()

    async def scenario():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def install(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        monkeypatch.setattr(owner.execution_tabs, "_install_base_dependencies_for_tab", install)
        task = asyncio.create_task(owner.run_job("module.captioner", [], "audit"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert observer.stop_btn.enabled is True
            observer.cancel()
            await asyncio.wait_for(task, 2)
            assert stopped.is_set()
            assert tab.status == "missing"
            assert observer.execution_tabs.active_tab_can_start()

            async def install_ok(*args, **kwargs):
                return None

            monkeypatch.setattr(owner.execution_tabs, "_install_base_dependencies_for_tab", install_ok)
            assert await owner.execution_tabs.ensure_active_tab_runtime_ready(tab)
            assert tab.status == "ready"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_cancel_setup_kills_only_its_subprocess_tree(monkeypatch):
    psutil = pytest.importorskip("psutil")

    async def scenario():
        started = asyncio.Event()
        child_pids = []
        processes = []
        original_spawn = asyncio.create_subprocess_exec

        async def spawn(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            processes.append(process)
            return process

        def log(tab_id, text):
            child_pids.append(int(text))
            started.set()

        owner = SimpleNamespace(_log=log)
        tab = SimpleNamespace(id="setup-tab", env_vars={})
        sleeper = "import time; time.sleep(60)"
        script = f"import subprocess, sys, time; p=subprocess.Popen([sys.executable, '-c', {sleeper!r}]); print(p.pid, flush=True); time.sleep(60)"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        unrelated = await original_spawn(sys.executable, "-c", sleeper, creationflags=flags)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(tabs_module.ExecutionTabs._run_logged_command(
            owner, tab, [sys.executable, "-c", script], creationflags=flags, failure_label="setup",
        ))
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert processes[0].returncode is not None
            assert not psutil.pid_exists(child_pids[0])
            assert unrelated.returncode is None
        finally:
            task.cancel()
            for process in processes:
                try:
                    tree = psutil.Process(process.pid)
                    for item in tree.children(recursive=True) + [tree]:
                        try:
                            item.kill()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                await process.wait()
            unrelated.kill()
            await unrelated.wait()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_stop_at_preparation_completion_boundary_prevents_job_submit(monkeypatch, tmp_path):
    monkeypatch.setattr(tabs_module, "save_task_tabs", lambda tabs: None)
    monkeypatch.setattr(job_manager, "get_active_jobs", lambda: [])
    panel = _make_panel()
    venv = tmp_path / "runtime"
    python = tabs_module._python_for_venv(venv)
    python.parent.mkdir(parents=True)
    python.touch()
    tab = tabs_module.TaskTab(
        "tab-0002", "Second", 2, ".", str(venv), str(python), datetime.now(), status="missing",
    )
    store = tabs_module.TaskTabStore([tab])
    tabs = tabs_module.ExecutionTabs.__new__(tabs_module.ExecutionTabs)
    tabs.tabs = store.tabs
    tabs._store = store
    tabs.active_tab_id = tab.id
    tabs._tab_bar = None
    tabs._on_tab_change = None
    tabs._on_tab_log = None
    store.views.add(tabs)
    panel.execution_tabs = tabs
    submitted = []
    prepare_calls = 0

    async def prepare(value):
        nonlocal prepare_calls
        prepare_calls += 1
        value.status = "ready"
        if prepare_calls == 1:
            asyncio.get_running_loop().call_soon(panel.cancel)

    async def submit(*args, **kwargs):
        submitted.append(kwargs)
        raise AssertionError("Stop must prevent submit after preparation")

    tabs._create_venv_for_tab = prepare
    monkeypatch.setattr("gui.components.execution_panel.job_manager.submit", submit)

    result = asyncio.run(panel.run_job("module.captioner", [], "Race"))

    assert result.status == ProcessStatus.ERROR
    assert submitted == []
    assert store.preparation_tasks == {}
    assert store.cancelled_preparations == set()
    assert tabs.active_tab_can_start()
    assert asyncio.run(tabs.ensure_active_tab_runtime_ready(tab)) is True


def _write_process_tree_fixture(tmp_path: Path) -> None:
    (tmp_path / "tree_worker.py").write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )


async def _wait_for_process_tree(runner: ProcessRunner, psutil):
    for _ in range(500):
        process = runner.process
        if process is not None:
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                if children:
                    return parent, children
            except psutil.NoSuchProcess:
                pass
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"process tree did not start: process={runner.process!r}, "
        f"running={runner.is_running}, logs={runner._log_buffer.get_all_lines()!r}"
    )


@pytest.mark.parametrize("threaded", [False, True])
def test_cancelling_panel_wait_terminates_only_its_process_tree(monkeypatch, tmp_path, threaded):
    psutil = pytest.importorskip("psutil")
    _write_process_tree_fixture(tmp_path)
    manager = JobManager()
    panel = _make_panel()
    monkeypatch.setattr("gui.components.execution_panel.job_manager", manager)
    monkeypatch.setattr(ProcessRunner, "_find_uv", staticmethod(lambda: None))
    monkeypatch.setattr(ProcessRunner, "_requires_threaded_subprocess", staticmethod(lambda: threaded))

    async def scenario():
        unrelated = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        task = asyncio.create_task(panel.run_job(
            "tree_worker", [], "Worker",
            runner_kwargs={"cwd": str(tmp_path), "native_console": False, "python_path": sys.executable},
        ))
        tracked = []
        try:
            for _ in range(500):
                jobs = manager.get_all_jobs()
                if jobs:
                    break
                await asyncio.sleep(0.01)
            job = manager.get_all_jobs()[0]
            parent, children = await _wait_for_process_tree(job.runner, psutil)
            tracked = [parent, *children]
            task.cancel()
            result = await task
            _, alive = psutil.wait_procs(tracked, timeout=5)
            assert result.status == ProcessStatus.ERROR
            assert job.status == JobStatus.CANCELLED
            assert alive == []
            assert job.runner.process is None
            assert job.runner.is_running is False
            assert unrelated.returncode is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for process in tracked:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            if unrelated.returncode is None:
                unrelated.kill()
            await unrelated.wait()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_twice", [False, True])
def test_cancelling_during_async_spawn_handoff_terminates_spawned_tree(monkeypatch, tmp_path, cancel_twice):
    psutil = pytest.importorskip("psutil")
    _write_process_tree_fixture(tmp_path)
    runner = ProcessRunner()
    original_spawn = asyncio.create_subprocess_exec
    spawned = []

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_return(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            spawned.append(process)
            started.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_return)
        monkeypatch.setattr(ProcessRunner, "_find_uv", staticmethod(lambda: None))
        monkeypatch.setattr(ProcessRunner, "_requires_threaded_subprocess", staticmethod(lambda: False))
        task = asyncio.create_task(runner.run_python_script(
            "tree_worker", [], cwd=str(tmp_path), native_console=False, python_path=sys.executable,
        ))
        tracked = []
        try:
            await asyncio.wait_for(started.wait(), 5)
            parent = psutil.Process(spawned[0].pid)
            for _ in range(500):
                children = parent.children(recursive=True)
                if children:
                    break
                await asyncio.sleep(0.01)
            tracked = [parent, *children]
            task.cancel()
            if cancel_twice:
                await asyncio.sleep(0)
                task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            _, alive = psutil.wait_procs(tracked, timeout=5)
            assert alive == []
            assert runner.process is None
            assert runner.is_running is False
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for process in tracked:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            for process in spawned:
                try:
                    await process.wait()
                except ProcessLookupError:
                    pass

    asyncio.run(scenario())


def test_repeated_cancellation_waits_for_process_tree_reaping(monkeypatch, tmp_path):
    psutil = pytest.importorskip("psutil")
    _write_process_tree_fixture(tmp_path)
    runner = ProcessRunner()
    original_terminate = ProcessRunner.terminate_process_tree

    async def scenario():
        cleanup_started = asyncio.Event()
        release_cleanup = threading.Event()
        loop = asyncio.get_running_loop()

        def delayed_terminate(process, *, process_group=False):
            loop.call_soon_threadsafe(cleanup_started.set)
            assert release_cleanup.wait(5)
            original_terminate(process, process_group=process_group)

        monkeypatch.setattr(ProcessRunner, "terminate_process_tree", staticmethod(delayed_terminate))
        monkeypatch.setattr(ProcessRunner, "_find_uv", staticmethod(lambda: None))
        monkeypatch.setattr(ProcessRunner, "_requires_threaded_subprocess", staticmethod(lambda: False))
        task = asyncio.create_task(runner.run_python_script(
            "tree_worker", [], cwd=str(tmp_path), native_console=False, python_path=sys.executable,
        ))
        tracked = []
        try:
            parent, children = await _wait_for_process_tree(runner, psutil)
            tracked = [parent, *children]
            task.cancel()
            await asyncio.wait_for(cleanup_started.wait(), 5)
            task.cancel()
            loop.call_later(0.1, release_cleanup.set)
            with pytest.raises(asyncio.CancelledError):
                await task
            assert release_cleanup.is_set(), "cancelled task returned before process cleanup completed"
            _, alive = psutil.wait_procs(tracked, timeout=5)
            assert alive == []
        finally:
            release_cleanup.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for process in tracked:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass

    asyncio.run(scenario())


def test_cannot_close_prepared_tab_before_its_start_continuation(monkeypatch, tmp_path):
    manager = JobManager()
    monkeypatch.setattr("gui.components.execution_panel.job_manager", manager)
    monkeypatch.setattr(tabs_module, "job_manager", manager)
    monkeypatch.setattr(tabs_module, "save_task_tabs", lambda tabs: None)
    monkeypatch.setattr(tabs_module.ui, "notify", lambda *args, **kwargs: None)
    panel = _make_panel()
    venv = tmp_path / "runtime"
    python = tabs_module._python_for_venv(venv)
    python.parent.mkdir(parents=True)
    python.touch()
    original = tabs_module.TaskTab("tab-0001", "First", 1, ".", ".venv", None, datetime.now())
    tab = tabs_module.TaskTab(
        "tab-0002", "Second", 2, ".", str(venv), str(python), datetime.now(), status="missing",
    )
    store = tabs_module.TaskTabStore([original, tab])
    tabs = tabs_module.ExecutionTabs.__new__(tabs_module.ExecutionTabs)
    tabs.tabs = store.tabs
    tabs._store = store
    tabs.active_tab_id = tab.id
    tabs._tab_bar = None
    tabs._on_tab_change = None
    tabs._on_tab_log = None
    store.views.add(tabs)
    panel.execution_tabs = tabs

    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()

        async def prepare(value):
            value.status = "ready"
            asyncio.get_running_loop().call_soon(tabs._close_tab, value.id)

        async def run(_self, *_args, **_kwargs):
            started.set()
            await finish.wait()
            return ProcessResult(ProcessStatus.SUCCESS, 0, "ok")

        monkeypatch.setattr(tabs, "_create_venv_for_tab", prepare)
        monkeypatch.setattr(ProcessRunner, "run_python_script", run)
        task = asyncio.create_task(panel.run_job("module.captioner", [], "close-boundary"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            job = manager.get_all_jobs()[0]
            assert tabs.find_tab(job.tab_id) is tab
            assert original.current_job_id is None
            assert panel.stop_btn.enabled is True
            assert panel._active_tab_current_job() is job
        finally:
            finish.set()
            await task

    asyncio.run(scenario())


def test_second_run_job_cannot_overtake_preparation_owner_at_ready_handoff(monkeypatch, tmp_path):
    manager = JobManager()
    monkeypatch.setattr("gui.components.execution_panel.job_manager", manager)
    monkeypatch.setattr(tabs_module, "job_manager", manager)
    monkeypatch.setattr(tabs_module, "save_task_tabs", lambda tabs: None)
    monkeypatch.setattr(tabs_module, "_base_install_needed", lambda _venv: False)
    monkeypatch.setattr(tabs_module.ui, "notify", lambda *args, **kwargs: None)
    panel = _make_panel()
    venv = tmp_path / "runtime"
    python = tabs_module._python_for_venv(venv)
    python.parent.mkdir(parents=True)
    python.touch()
    tab = tabs_module.TaskTab(
        "tab-0002", "Second", 2, ".", str(venv), str(python), datetime.now(), status="missing",
    )
    store = tabs_module.TaskTabStore([tab])
    tabs = tabs_module.ExecutionTabs.__new__(tabs_module.ExecutionTabs)
    tabs.tabs = store.tabs
    tabs._store = store
    tabs.active_tab_id = tab.id
    tabs._tab_bar = None
    tabs._on_tab_change = None
    tabs._on_tab_log = None
    store.views.add(tabs)
    panel.execution_tabs = tabs

    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()
        second_tasks = []

        async def prepare(value):
            value.status = "ready"
            second_tasks.append(asyncio.create_task(panel.run_job("module.captioner", [], "second")))

        async def run(_self, *_args, **_kwargs):
            started.set()
            await finish.wait()
            return ProcessResult(ProcessStatus.SUCCESS, 0, "ok")

        monkeypatch.setattr(tabs, "_create_venv_for_tab", prepare)
        monkeypatch.setattr(ProcessRunner, "run_python_script", run)
        first_task = asyncio.create_task(panel.run_job("module.captioner", [], "first"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert manager.get_all_jobs()[0].name == "first"
        finally:
            finish.set()
            await asyncio.gather(first_task, *second_tasks)

    asyncio.run(scenario())
