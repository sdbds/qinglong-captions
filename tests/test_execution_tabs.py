from datetime import datetime

import gui.components.execution_tabs as execution_tabs_module
from gui.components.execution_tabs import ExecutionTabs, TaskTab


def test_task_tab_dataclass_accepts_defaults_after_created_at():
    tab = TaskTab(
        id="tab-0002",
        name="任务 2",
        index=2,
        work_dir=".",
        venv_path=".runtime_venvs/tab-0002",
        python_path=".runtime_venvs/tab-0002/Scripts/python.exe",
        created_at=datetime.now(),
    )

    assert tab.env_vars == {}
    assert tab.status == "missing"
    assert tab.error_message is None


def test_default_tab_runner_kwargs_keeps_legacy_python_resolution_but_includes_tab_metadata():
    tabs = ExecutionTabs.__new__(ExecutionTabs)
    tabs.tabs = [
        TaskTab(
            id="tab-0001",
            name="默认任务",
            index=1,
            work_dir=".",
            venv_path=".venv",
            python_path=None,
            created_at=datetime.now(),
            status="ready",
        )
    ]
    tabs.active_tab_id = "tab-0001"

    kwargs = ExecutionTabs.runner_kwargs(tabs)

    assert kwargs == {"tab_id": "tab-0001", "tab_name": "默认任务"}


def test_load_task_tabs_resets_interrupted_venv_creation_to_missing(monkeypatch, tmp_path):
    config_path = tmp_path / "task_tabs.toml"
    config_path.write_text(
        """
[[tab]]
id = "tab-0002"
name = "任务 2"
index = 2
work_dir = "."
venv_path = ".runtime_venvs/tab-0002"
python_path = ".runtime_venvs/tab-0002/Scripts/python.exe"
created_at = "2026-06-02T00:00:00"
selected_task_key = ""
current_job_id = ""
status = "creating_venv"
error_message = ""

[tab.env_vars]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(execution_tabs_module, "TABS_CONFIG_PATH", config_path)

    tabs = execution_tabs_module.load_task_tabs()

    assert tabs[0].status == "missing"


def test_active_tab_can_start_rejects_busy_tab():
    tabs = ExecutionTabs.__new__(ExecutionTabs)
    tabs.tabs = [
        TaskTab(
            id="tab-0002",
            name="任务 2",
            index=2,
            work_dir=".",
            venv_path=".runtime_venvs/tab-0002",
            python_path=".runtime_venvs/tab-0002/Scripts/python.exe",
            created_at=datetime.now(),
            current_job_id="job-1",
            status="busy",
        )
    ]
    tabs.active_tab_id = "tab-0002"

    assert ExecutionTabs.active_tab_can_start(tabs) is False


def test_missing_non_default_tab_can_start_so_runtime_is_created_on_start():
    tabs = ExecutionTabs.__new__(ExecutionTabs)
    tabs.tabs = [
        TaskTab(
            id="tab-0002",
            name="任务 2",
            index=2,
            work_dir=".",
            venv_path=".runtime_venvs/tab-0002",
            python_path=".runtime_venvs/tab-0002/Scripts/python.exe",
            created_at=datetime.now(),
            status="missing",
        )
    ]
    tabs.active_tab_id = "tab-0002"

    assert ExecutionTabs.active_tab_can_start(tabs) is True


def test_add_tab_does_not_create_venv_until_start(monkeypatch):
    tabs = ExecutionTabs.__new__(ExecutionTabs)
    tabs.tabs = [
        TaskTab(
            id="tab-0001",
            name="默认任务",
            index=1,
            work_dir=".",
            venv_path=".venv",
            python_path=None,
            created_at=datetime.now(),
            status="ready",
        )
    ]
    tabs.active_tab_id = "tab-0001"
    tabs._next_tab_index = 2
    tabs._render_tabs = lambda: None
    tabs._notify_tab_change = lambda: None
    calls = []

    async def fake_create(tab):
        calls.append(tab.id)

    monkeypatch.setattr(tabs, "_create_venv_for_tab", fake_create)
    monkeypatch.setattr(execution_tabs_module, "save_task_tabs", lambda value: None)

    import asyncio

    asyncio.run(ExecutionTabs._add_tab(tabs))

    assert calls == []
    assert tabs.active_tab_id == "tab-0002"
    assert tabs.tabs[-1].status == "missing"


def test_ensure_active_tab_runtime_ready_creates_missing_runtime(monkeypatch):
    tabs = ExecutionTabs.__new__(ExecutionTabs)
    tab = TaskTab(
        id="tab-0002",
        name="任务 2",
        index=2,
        work_dir=".",
        venv_path=".runtime_venvs/tab-0002",
        python_path=".runtime_venvs/tab-0002/Scripts/python.exe",
        created_at=datetime.now(),
        status="missing",
    )
    tabs.tabs = [tab]
    tabs.active_tab_id = tab.id
    calls = []

    async def fake_create(value):
        calls.append(value.id)
        value.status = "ready"

    monkeypatch.setattr(tabs, "_create_venv_for_tab", fake_create)

    import asyncio

    assert asyncio.run(ExecutionTabs.ensure_active_tab_runtime_ready(tabs)) is True
    assert calls == ["tab-0002"]


def test_close_non_default_tab_switches_back_to_default(monkeypatch):
    tabs = ExecutionTabs.__new__(ExecutionTabs)
    tabs.tabs = [
        TaskTab(
            id="tab-0001",
            name="默认任务",
            index=1,
            work_dir=".",
            venv_path=".venv",
            python_path=None,
            created_at=datetime.now(),
            status="ready",
        ),
        TaskTab(
            id="tab-0002",
            name="任务 2",
            index=2,
            work_dir=".",
            venv_path=".runtime_venvs/tab-0002",
            python_path=".runtime_venvs/tab-0002/Scripts/python.exe",
            created_at=datetime.now(),
            status="missing",
        ),
    ]
    tabs.active_tab_id = "tab-0002"
    tabs._render_tabs = lambda: None
    changed = []
    tabs._notify_tab_change = lambda: changed.append(tabs.active_tab_id)
    monkeypatch.setattr(execution_tabs_module, "save_task_tabs", lambda value: None)

    ExecutionTabs._close_tab(tabs, "tab-0002")

    assert [tab.id for tab in tabs.tabs] == ["tab-0001"]
    assert tabs.active_tab_id == "tab-0001"
    assert changed == ["tab-0001"]
