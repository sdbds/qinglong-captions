"""Execution tabs with per-tab runtime venv selection."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional

import toml
from nicegui import ui

from gui.theme import COLORS
from gui.utils.i18n import t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABS_CONFIG_PATH = PROJECT_ROOT / "config" / "task_tabs.toml"
RUNTIME_VENVS_DIR = PROJECT_ROOT / ".runtime_venvs"


@dataclass
class TaskTab:
    id: str
    name: str
    index: int
    work_dir: str
    venv_path: Optional[str]
    python_path: Optional[str]
    created_at: datetime
    env_vars: dict[str, str] = field(default_factory=dict)
    selected_task_key: Optional[str] = None
    current_job_id: Optional[str] = None
    status: Literal["missing", "creating_venv", "ready", "busy", "error"] = "missing"
    error_message: Optional[str] = None


def _relative_or_absolute(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_stored_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _python_for_venv(venv_path: Path) -> Path:
    return venv_path / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")


def _default_tab() -> TaskTab:
    return TaskTab(
        id="tab-0001",
        name=t("default_task_tab", "默认任务"),
        index=1,
        work_dir=".",
        venv_path=".venv",
        python_path=None,
        created_at=datetime.now(),
        status="ready",
    )


def load_task_tabs() -> list[TaskTab]:
    if not TABS_CONFIG_PATH.exists():
        return [_default_tab()]

    try:
        payload = toml.load(TABS_CONFIG_PATH)
    except Exception:
        return [_default_tab()]

    tabs: list[TaskTab] = []
    for item in payload.get("tab", []) or []:
        try:
            created_raw = item.get("created_at")
            created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.now()
            status = item.get("status") or "missing"
            if status == "busy":
                status = "ready"
            if status == "creating_venv":
                status = "missing"
            tabs.append(
                TaskTab(
                    id=str(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    index=int(item.get("index") or len(tabs) + 1),
                    work_dir=str(item.get("work_dir") or "."),
                    venv_path=item.get("venv_path") or None,
                    python_path=item.get("python_path") or None,
                    created_at=created_at,
                    env_vars=dict(item.get("env_vars") or {}),
                    selected_task_key=item.get("selected_task_key") or None,
                    current_job_id=None,
                    status=status,
                    error_message=item.get("error_message") or None,
                )
            )
        except Exception:
            continue

    return tabs or [_default_tab()]


def save_task_tabs(tabs: list[TaskTab]) -> None:
    TABS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tab": [
            {
                "id": tab.id,
                "name": tab.name,
                "index": tab.index,
                "work_dir": tab.work_dir,
                "venv_path": tab.venv_path or "",
                "python_path": tab.python_path or "",
                "created_at": tab.created_at.isoformat(),
                "selected_task_key": tab.selected_task_key or "",
                "current_job_id": "",
                "status": "ready" if tab.status == "busy" else tab.status,
                "error_message": tab.error_message or "",
                "env_vars": tab.env_vars,
            }
            for tab in tabs
        ]
    }
    with open(TABS_CONFIG_PATH, "w", encoding="utf-8") as f:
        toml.dump(payload, f)


class ExecutionTabs:
    """Browser-like task tab bar that owns per-tab runtime venv selection."""

    _styles_injected = False

    def __init__(
        self,
        *,
        on_tab_change: Optional[Callable[[TaskTab], None]] = None,
        on_tab_log: Optional[Callable[[str, str, str], None]] = None,
    ):
        self._ensure_styles()
        self.tabs = load_task_tabs()
        self.active_tab_id = self.tabs[0].id
        self._next_tab_index = max((tab.index for tab in self.tabs), default=1) + 1
        self._on_tab_change = on_tab_change
        self._on_tab_log = on_tab_log
        self._tab_bar = None

        with ui.row().classes("w-full items-end justify-between gap-2"):
            self._tab_bar = ui.row().classes("items-end gap-1").style("min-width: 0; overflow-x: auto;")
            add_btn = ui.element("button").classes("task-tab-add").props('type="button"')
            add_btn.on("click", self._handle_add_tab_click)
            add_btn.tooltip(t("new_task_tab", "新任务 tab"))
            with add_btn:
                ui.icon("add", size="18px")

        self._render_tabs()

    @classmethod
    def _ensure_styles(cls) -> None:
        if cls._styles_injected:
            return
        ui.add_head_html("""
        <style>
            .task-tab-btn,
            .task-tab-close,
            .task-tab-add {
                appearance: none;
                border: 1px solid var(--ql-border);
                background: var(--ql-surface);
                color: var(--ql-text-secondary);
                font: inherit;
                height: 32px;
                box-shadow: none;
                outline: none;
                cursor: pointer;
                transition: none !important;
                animation: none !important;
            }

            .task-tab-btn {
                padding: 0 12px;
                max-width: 180px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .task-tab-btn--active {
                background: var(--ql-surface-raised);
                color: var(--ql-text);
                border-color: var(--ql-border-hover);
                font-weight: 600;
            }

            .task-tab-btn--inactive {
                opacity: 0.86;
            }

            .task-tab-close,
            .task-tab-add {
                width: 32px;
                min-width: 32px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0;
            }

            .task-tab-close:hover,
            .task-tab-add:hover,
            .task-tab-btn:hover {
                background: var(--ql-surface-raised);
                color: var(--ql-text);
            }
        </style>
        """)
        cls._styles_injected = True

    @property
    def active_tab(self) -> TaskTab:
        for tab in self.tabs:
            if tab.id == self.active_tab_id:
                return tab
        self.active_tab_id = self.tabs[0].id
        return self.tabs[0]

    def runner_kwargs(self) -> Optional[dict]:
        tab = self.active_tab
        kwargs = {
            "tab_id": tab.id,
            "tab_name": tab.name,
        }
        if tab.index == 1:
            return kwargs
        if tab.status != "ready" or not tab.python_path or not tab.venv_path:
            ui.notify(t("task_tab_not_ready", "当前任务 tab 的 venv 尚未就绪"), type="warning")
            return None
        kwargs.update({
            "python_path": str(_resolve_stored_path(tab.python_path)),
            "venv_path": str(_resolve_stored_path(tab.venv_path)),
        })
        return kwargs

    async def ensure_active_tab_runtime_ready(self) -> bool:
        tab = self.active_tab
        if tab.index == 1:
            return True
        if tab.status == "ready" and tab.python_path and tab.venv_path:
            if _resolve_stored_path(tab.python_path).exists():
                return True
            tab.status = "missing"
            tab.error_message = None
        if tab.status == "creating_venv":
            ui.notify(t("task_tab_not_ready", "当前任务 tab 的 venv 尚未就绪"), type="warning")
            return False
        await self._create_venv_for_tab(tab)
        return tab.status == "ready" and bool(tab.python_path and tab.venv_path)

    def active_tab_can_start(self) -> bool:
        tab = self.active_tab
        if tab.status == "creating_venv":
            return False
        return tab.current_job_id is None

    def find_tab(self, tab_id: Optional[str]) -> Optional[TaskTab]:
        if tab_id is None:
            return None
        for tab in self.tabs:
            if tab.id == tab_id:
                return tab
        return None

    async def retry_active_tab(self) -> None:
        tab = self.active_tab
        if tab.index > 1:
            await self._create_venv_for_tab(tab)

    def mark_job(self, job) -> None:
        tab = self.find_tab(getattr(job, "tab_id", None)) or self.active_tab
        tab.current_job_id = getattr(job, "id", None)
        tab.status = "busy"
        save_task_tabs(self.tabs)
        self._render_tabs()
        self._notify_tab_change()

    def clear_job(self, job_id: str) -> None:
        changed = False
        for tab in self.tabs:
            if tab.current_job_id == job_id:
                tab.current_job_id = None
                if tab.status == "busy":
                    tab.status = "ready"
                changed = True
        if changed:
            save_task_tabs(self.tabs)
            self._render_tabs()
            self._notify_tab_change()

    def _status_color(self, tab: TaskTab) -> str:
        return {
            "ready": COLORS.get("success", "var(--ql-success)"),
            "creating_venv": COLORS.get("info", "var(--ql-info)"),
            "busy": COLORS.get("warning", "var(--ql-warning)"),
            "error": COLORS.get("error", "var(--ql-error)"),
            "missing": COLORS.get("text_muted", "var(--ql-text-muted)"),
        }.get(tab.status, "var(--ql-text-muted)")

    def _notify_tab_change(self) -> None:
        if self._on_tab_change is not None:
            self._on_tab_change(self.active_tab)

    def _log(self, tab_id: str, message: str, level: str = "info") -> None:
        if self._on_tab_log is not None:
            self._on_tab_log(tab_id, message, level)

    def _render_tabs(self) -> None:
        if self._tab_bar is None:
            return
        self._tab_bar.clear()
        with self._tab_bar:
            for tab in self.tabs:
                active = tab.id == self.active_tab_id
                has_close = tab.index > 1
                tab_radius = "8px 0 0 0" if has_close else "8px 8px 0 0"
                with ui.row().classes("items-center gap-0"):
                    btn = ui.element("button").props('type="button"')
                    btn.on("click", lambda _e, tab_id=tab.id: self._select_tab(tab_id))
                    if active:
                        btn.classes("task-tab-btn task-tab-btn--active")
                        btn.style(
                            f"border-radius: {tab_radius}; "
                            f"border-bottom: 3px solid {self._status_color(tab)};"
                        )
                    else:
                        btn.classes("task-tab-btn task-tab-btn--inactive")
                        btn.style(f"border-radius: {tab_radius}; opacity: 0.82;")
                    with btn:
                        ui.label(tab.name)
                    if has_close:
                        close_btn = ui.element("button").classes("task-tab-close").props('type="button"')
                        close_btn.on("click", lambda e, tab_id=tab.id: self._close_tab(tab_id))
                        close_btn.style(
                            "border-radius: 0 8px 0 0; "
                            f"color: {COLORS.get('text_muted', 'var(--ql-text-muted)')};"
                        )
                        close_btn.tooltip(t("close", "关闭"))
                        with close_btn:
                            ui.icon("close", size="16px")

    def _select_tab(self, tab_id: str) -> None:
        self.active_tab_id = tab_id
        self._render_tabs()
        self._notify_tab_change()

    async def _handle_add_tab_click(self, _event=None) -> None:
        await self._add_tab()

    def _close_tab(self, tab_id: str) -> None:
        tab = self.find_tab(tab_id)
        if tab is None or tab.index == 1:
            return
        if tab.current_job_id:
            ui.notify(t("task_already_running", "已有任务正在运行"), type="warning")
            return

        removed_index = self.tabs.index(tab)
        self.tabs = [item for item in self.tabs if item.id != tab_id]
        if not self.tabs:
            self.tabs = [_default_tab()]
        if self.active_tab_id == tab_id:
            next_index = min(max(removed_index - 1, 0), len(self.tabs) - 1)
            self.active_tab_id = self.tabs[next_index].id
        save_task_tabs(self.tabs)
        self._render_tabs()
        self._notify_tab_change()

    async def _add_tab(self) -> None:
        next_index = getattr(self, "_next_tab_index", max((tab.index for tab in self.tabs), default=1) + 1)
        self._next_tab_index = next_index + 1
        tab_id = f"tab-{next_index:04d}"
        venv_path = RUNTIME_VENVS_DIR / tab_id
        tab = TaskTab(
            id=tab_id,
            name=t("task_tab_name", "任务 {index}").format(index=next_index),
            index=next_index,
            work_dir=".",
            venv_path=_relative_or_absolute(venv_path),
            python_path=_relative_or_absolute(_python_for_venv(venv_path)),
            created_at=datetime.now(),
            status="missing",
        )
        self.tabs.append(tab)
        self.active_tab_id = tab.id
        save_task_tabs(self.tabs)
        self._render_tabs()
        self._notify_tab_change()

    async def _create_venv_for_tab(self, tab: TaskTab) -> None:
        if tab.index == 1 or not tab.venv_path:
            return
        tab.status = "creating_venv"
        tab.error_message = None
        save_task_tabs(self.tabs)
        self._render_tabs()
        self._notify_tab_change()

        venv_path = _resolve_stored_path(tab.venv_path)
        python_path = _python_for_venv(venv_path)
        if python_path.exists():
            tab.python_path = _relative_or_absolute(python_path)
            tab.status = "ready"
            save_task_tabs(self.tabs)
            self._render_tabs()
            self._notify_tab_change()
            self._log(tab.id, f"{tab.name} venv ready: {tab.python_path}", "success")
            return

        cmd = [shutil.which("uv") or sys.executable]
        if shutil.which("uv"):
            cmd.extend(["venv", str(venv_path)])
        else:
            cmd.extend(["-m", "venv", str(venv_path)])

        self._log(tab.id, f"creating venv for {tab.name}: {' '.join(cmd)}")
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=creationflags,
            )
            if process.stdout is not None:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    self._log(tab.id, line.decode("utf-8", errors="replace").rstrip())
            return_code = await process.wait()
            if return_code != 0:
                raise RuntimeError(f"venv creation failed with code {return_code}")
            if not python_path.exists():
                raise RuntimeError(f"venv python not found: {python_path}")
            tab.python_path = _relative_or_absolute(python_path)
            tab.status = "ready"
            self._log(tab.id, f"{tab.name} venv ready: {tab.python_path}", "success")
        except Exception as exc:
            tab.status = "error"
            tab.error_message = str(exc)
            self._log(tab.id, f"{tab.name} venv error: {exc}", "error")
        finally:
            save_task_tabs(self.tabs)
            self._render_tabs()
            self._notify_tab_change()


def create_execution_tabs(**kwargs) -> ExecutionTabs:
    return ExecutionTabs(**kwargs)
