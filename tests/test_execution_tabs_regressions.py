import asyncio
from types import SimpleNamespace

import pytest
import toml

from gui.components import execution_tabs as module
from gui.utils.job_manager import job_manager


class _Element:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


@pytest.fixture
def pages(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "TABS_CONFIG_PATH", tmp_path / "tabs.toml")
    monkeypatch.setattr(module.ExecutionTabs, "_ensure_styles", lambda self: None)
    monkeypatch.setattr(module.ExecutionTabs, "_render_tabs", lambda self: None)
    for name in ("row", "element", "icon"):
        monkeypatch.setattr(module.ui, name, lambda *args, **kwargs: _Element())
    monkeypatch.setattr(module.ui, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(job_manager, "get_active_jobs", lambda: [])
    return module.ExecutionTabs


def _stored_ids():
    return [tab.id for tab in module.load_task_tabs()]


def test_old_page_completion_preserves_new_page_added_tab(pages):
    old = pages()
    old.mark_job(SimpleNamespace(id="old-job", tab_id="tab-0001"))
    new = pages()
    asyncio.run(new._add_tab())
    old.clear_job("old-job")
    assert _stored_ids() == ["tab-0001", "tab-0002"]
    assert [tab.id for tab in old.tabs] == _stored_ids()


def test_old_page_completion_does_not_resurrect_closed_tab(pages):
    old = pages()
    asyncio.run(old._add_tab())
    old.mark_job(SimpleNamespace(id="old-job", tab_id="tab-0001"))
    new = pages()
    new._close_tab("tab-0002")
    old.clear_job("old-job")
    assert _stored_ids() == ["tab-0001"]
    assert [tab.id for tab in old.tabs] == ["tab-0001"]


def test_pages_allocate_distinct_ids_even_when_created_before_add(pages):
    first, second = pages(), pages()
    asyncio.run(first._add_tab())
    asyncio.run(second._add_tab())
    assert first.active_tab_id != second.active_tab_id
    assert _stored_ids() == ["tab-0001", "tab-0002", "tab-0003"]


def test_busy_ownership_comes_from_job_manager(pages, monkeypatch):
    first = pages()
    asyncio.run(first._add_tab())
    job = SimpleNamespace(id="running-job", tab_id="tab-0002")
    monkeypatch.setattr(job_manager, "get_active_jobs", lambda: [job])
    second = pages()
    second.active_tab_id = "tab-0002"
    assert second.active_tab_can_start() is False
    second._close_tab("tab-0002")
    assert "tab-0002" in _stored_ids()


def test_pip_fallback_passes_only_project_base_requirements(monkeypatch, tmp_path):
    requirements = ["rich>=13", "toml", "colorama; sys_platform == 'win32'"]
    (tmp_path / "pyproject.toml").write_text(toml.dumps({
        "project": {"dependencies": requirements, "optional-dependencies": {"gpu": ["torch"]}},
    }), encoding="utf-8")
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_find_uv_executable", lambda: None)
    command = module._base_dependency_install_command(tmp_path / "python.exe")
    assert command[5:] == requirements
    assert "pyproject.toml" not in command


def test_bootstrap_subprocess_inherits_gui_environment(pages, monkeypatch):
    seen = []
    configured = {"HTTPS_PROXY": "http://gui-proxy.invalid:1234", "UV_INDEX_URL": "https://gui-index.invalid/simple"}
    monkeypatch.setattr("utils.runtime_env._load_gui_env_overrides", lambda: configured)

    async def spawn(*args, **kwargs):
        seen.append(kwargs)

        async def wait():
            return 0

        return SimpleNamespace(stdout=None, wait=wait)

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    page = pages()
    asyncio.run(page._run_logged_command(page.active_tab, ["fake-installer"],
                                        creationflags=0, failure_label="install"))
    actual_proxy = seen[0].get("env", {}).get("HTTPS_PROXY")
    actual_index = seen[0].get("env", {}).get("UV_INDEX_URL")
    assert actual_proxy == configured["HTTPS_PROXY"]
    assert actual_index == configured["UV_INDEX_URL"]


def test_uv_install_strategy_uses_gui_override(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "_find_uv_executable", lambda: "uv")
    monkeypatch.setenv("UV_INDEX_STRATEGY", "first-index")
    monkeypatch.setattr("utils.runtime_env._load_gui_env_overrides", lambda: {"UV_INDEX_STRATEGY": "unsafe-best-match"})
    command = module._base_dependency_install_command(tmp_path / "python.exe")
    assert command[command.index("--index-strategy") + 1] == "unsafe-best-match"


def test_pip_bootstrap_translates_gui_indexes_to_pip_environment(pages, monkeypatch):
    seen = []
    configured = {"UV_INDEX_URL": "https://gui-index.invalid/simple",
                  "UV_EXTRA_INDEX_URL": "https://gui-extra.invalid/simple"}
    monkeypatch.setattr("utils.runtime_env._load_gui_env_overrides", lambda: configured)

    async def spawn(*args, **kwargs):
        seen.append(kwargs)

        async def wait():
            return 0

        return SimpleNamespace(stdout=None, wait=wait)

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    page = pages()
    asyncio.run(page._run_logged_command(page.active_tab, ["python", "-m", "pip", "install", "rich"],
                                        creationflags=0, failure_label="install"))
    actual_index = seen[0]["env"].get("PIP_INDEX_URL")
    actual_extra = seen[0]["env"].get("PIP_EXTRA_INDEX_URL")
    assert actual_index == configured["UV_INDEX_URL"]
    assert actual_extra == configured["UV_EXTRA_INDEX_URL"]


def test_cannot_close_tab_while_its_runtime_is_being_prepared(pages):
    page = pages()
    asyncio.run(page._add_tab())
    page.active_tab.status = "creating_venv"
    page._close_tab(page.active_tab.id)
    assert len(page.tabs) == 2


def test_old_job_cleanup_keeps_new_job_busy_state(pages):
    page = pages()
    page.mark_job(SimpleNamespace(id="new-job", tab_id="tab-0001"))
    page.clear_job("old-job")
    assert page.active_tab.current_job_id == "new-job"
    assert page.active_tab.status == "busy"


def test_closing_tab_in_another_page_updates_selection_before_render(pages):
    first = pages()
    asyncio.run(first._add_tab())
    second = pages()
    rendered_selection = []
    first._render_tabs = lambda: rendered_selection.append(first.active_tab_id)
    second._close_tab("tab-0002")
    assert rendered_selection == ["tab-0001"]


def test_disconnected_page_does_not_interrupt_other_page_changes(pages):
    first, second = pages(), pages()

    def disconnected_render():
        raise RuntimeError("parent slot has been deleted")

    first._render_tabs = disconnected_render
    asyncio.run(second._add_tab())
    assert _stored_ids() == ["tab-0001", "tab-0002"]
