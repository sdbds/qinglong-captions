import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from gui.components import execution_tabs as tabs_module
from gui.components.execution_tabs import ExecutionTabs, TaskTab, TaskTabStore


pytestmark = pytest.mark.compat
def test_consecutive_preparation_cancels_are_python310_compatible_and_do_not_interrupt_cleanup():
    async def scenario():
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()

        async def prepare():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()
                raise

        task = asyncio.create_task(prepare())
        owner = SimpleNamespace(
            active_tab_id="tab-0002",
            _store=SimpleNamespace(
                preparation_tasks={"tab-0002": task},
                cancelled_preparations=set(),
            ),
        )
        try:
            await asyncio.sleep(0)
            assert ExecutionTabs.cancel_active_preparation(owner) is True
            await asyncio.wait_for(cleanup_started.wait(), 1)
            assert ExecutionTabs.cancel_active_preparation(owner) is True
            release_cleanup.set()
            await asyncio.gather(task, return_exceptions=True)
            assert cleanup_finished.is_set()
        finally:
            release_cleanup.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_outer_cancel_then_stop_does_not_interrupt_preparation_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(tabs_module, "save_task_tabs", lambda tabs: None)
    monkeypatch.setattr(tabs_module.job_manager, "get_active_jobs", lambda: [])
    venv = tmp_path / "runtime"
    python = tabs_module._python_for_venv(venv)
    python.parent.mkdir(parents=True)
    python.touch()
    tab = TaskTab(
        "tab-0002", "Second", 2, ".", str(venv), str(python), datetime.now(), status="missing",
    )
    tabs = ExecutionTabs.__new__(ExecutionTabs)
    tabs.tabs = [tab]
    tabs._store = TaskTabStore(tabs.tabs)
    tabs.active_tab_id = tab.id
    tabs._tab_bar = None
    tabs._on_tab_change = None
    tabs._on_tab_log = None

    async def scenario():
        prepare_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()

        async def prepare(_tab):
            prepare_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()
                raise

        tabs._create_venv_for_tab = prepare
        outer = asyncio.create_task(tabs.ensure_active_tab_runtime_ready(tab))
        try:
            await asyncio.wait_for(prepare_started.wait(), 1)
            outer.cancel()
            await asyncio.wait_for(cleanup_started.wait(), 1)
            assert tabs.cancel_active_preparation() is True
            release_cleanup.set()
            await asyncio.gather(outer, return_exceptions=True)
            assert cleanup_finished.is_set()
        finally:
            release_cleanup.set()
            outer.cancel()
            await asyncio.gather(outer, return_exceptions=True)

    asyncio.run(scenario())
