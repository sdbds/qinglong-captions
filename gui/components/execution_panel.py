"""共享执行面板 - 封装任务 tab、Start/Stop、progress 和 LogViewer。"""

from __future__ import annotations

import inspect
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from nicegui import ui

from gui.theme import COLORS, get_classes
from gui.utils.i18n import t
from gui.utils.job_manager import JobStatus, job_manager
from gui.utils.log_buffer import LogBuffer
from gui.utils.process_runner import ProcessResult, ProcessStatus

if TYPE_CHECKING:
    from gui.components.execution_tabs import TaskTab
    from gui.components.log_viewer import LogViewer
    from gui.utils.job_manager import Job


def _create_execution_tabs(*, on_tab_change: Callable, on_tab_log: Callable):
    from gui.components.execution_tabs import create_execution_tabs

    return create_execution_tabs(on_tab_change=on_tab_change, on_tab_log=on_tab_log)


def _create_log_viewer(*, height: str, embedded: bool):
    from gui.components.log_viewer import create_log_viewer

    return create_log_viewer(height=height, embedded=embedded)


class _LogSink:
    """Small LogViewer-compatible writer for a single LogBuffer."""

    def __init__(self, log_buffer: LogBuffer):
        self._log_buffer = log_buffer

    def append(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_colors = {
            "success": "\x1b[32m",
            "warning": "\x1b[33m",
            "error": "\x1b[1;31m",
        }
        color = level_colors.get(level, "")
        reset = "\x1b[0m" if color else ""
        self._log_buffer.push(f"{color}[{timestamp}] {message}{reset}")

    def info(self, message: str) -> None:
        self._log_buffer.push(message)

    def success(self, message: str) -> None:
        self.append(message, "success")

    def warning(self, message: str) -> None:
        self.append(message, "warning")

    def error(self, message: str) -> None:
        self.append(message, "error")


class ExecutionPanel:
    """统一任务控制区：任务 tab、当前动作、Stop 和当前 tab 日志。"""

    def __init__(
        self,
        *,
        start_label: Optional[str] = None,
        height: str = "50vh",
        show_start: bool = True,
        on_start: Optional[Callable] = None,
    ):
        self.is_running = False
        self.current_job: Optional["Job"] = None
        self._active_jobs: set[str] = set()
        self._tab_current_jobs: dict[str, str] = {}
        self._tab_last_jobs: dict[str, str] = {}
        self._tab_log_buffers: dict[str, LogBuffer] = {}
        self._controls_ready = False
        self._on_start = on_start
        self._action_enabled = show_start
        self._external_start_buttons = []

        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            self.execution_tabs = _create_execution_tabs(
                on_tab_change=self._handle_tab_change,
                on_tab_log=self._write_tab_log,
            )

            self.progress = (
                ui.linear_progress(value=None)
                .props("indeterminate color=primary")
                .classes("w-full q-mt-sm")
                .style("height: 4px;")
            )
            self.progress.set_visibility(False)

            with ui.row().classes("w-full items-center justify-between gap-3 q-mt-sm q-mb-sm"):
                with ui.column().classes("gap-0").style("min-width: 0;"):
                    self.action_label = ui.label(start_label or t("start")).classes("text-subtitle2 text-weight-bold")
                    self.action_label.style("color: var(--color-text);")
                    self.status_label = ui.label("").classes("text-caption")
                    self.status_label.style("color: var(--color-text-secondary);")

                with ui.row().classes("items-center gap-2"):
                    self.retry_btn = ui.button(t("retry", "重试"), on_click=self._retry_active_tab, icon="refresh")
                    self.retry_btn.classes("modern-btn-secondary").props('type="button"')
                    self.retry_btn.set_visibility(False)

                    self.stop_btn = ui.button(t("stop"), on_click=self.cancel, icon="stop")
                    self.stop_btn.classes("modern-btn-danger").props('type="button"')
                    self.stop_btn.set_enabled(False)

                    if show_start:
                        self.start_btn = ui.button(start_label or t("start"), on_click=self._handle_start, icon="play_arrow")
                        self.start_btn.classes("modern-btn-success").props('type="button"')
                    else:
                        self.start_btn = None

            self.log_viewer: "LogViewer" = _create_log_viewer(height=height, embedded=True)

        self._controls_ready = True
        self._attach_log_for_active_tab()
        self._sync_active_tab_state()

    @property
    def _on_start(self) -> Optional[Callable]:
        return getattr(self, "_action_callback", None)

    @_on_start.setter
    def _on_start(self, callback: Optional[Callable]) -> None:
        self._action_callback = callback
        if getattr(self, "_controls_ready", False):
            self._safe_sync_active_tab_state()

    def set_action(self, label: str, callback: Optional[Callable], *, enabled: bool = True) -> None:
        """更新统一 Start 按钮当前绑定的动作。"""
        self._on_start = callback
        self._action_enabled = enabled
        try:
            self.action_label.set_text(label)
            if self.start_btn is not None:
                self.start_btn.set_text(label)
            self._sync_active_tab_state()
        except RuntimeError:
            pass

    async def _handle_start(self):
        """统一 Start 按钮点击。"""
        if not self._can_start_active_tab():
            ui.notify(t("task_already_running", "已有任务正在运行"), type="warning")
            return
        if self._on_start is None:
            return
        result = self._on_start()
        if inspect.isawaitable(result):
            await result

    def register_external_start_button(self, button):
        """兼容旧调用方；外部 Start 按钮不再由面板统一禁用。"""
        if button is None or button in self._external_start_buttons:
            return
        self._external_start_buttons.append(button)

    async def run_job(
        self,
        script_key: str,
        args: list,
        name: str,
        *,
        runner_kwargs: Optional[dict] = None,
        pre_log: Optional[Callable] = None,
        on_success: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
    ):
        """执行完整的 submit -> wait -> cleanup 流程。"""
        tab = self.execution_tabs.active_tab
        tab_id = tab.id
        if self._tab_has_running_job(tab_id):
            message = t("task_already_running", "已有任务正在运行")
            ui.notify(message, type="warning")
            return ProcessResult(ProcessStatus.ERROR, -1, message)

        merged_runner_kwargs = dict(runner_kwargs or {})
        if not await self.execution_tabs.ensure_active_tab_runtime_ready():
            self._safe_sync_active_tab_state()
            return ProcessResult(ProcessStatus.ERROR, -1, t("task_tab_not_ready", "当前任务 tab 的 venv 尚未就绪"))

        tab_kwargs = self.execution_tabs.runner_kwargs()
        if tab_kwargs is None:
            self._safe_sync_active_tab_state()
            return ProcessResult(ProcessStatus.ERROR, -1, t("task_tab_not_ready", "当前任务 tab 的 venv 尚未就绪"))
        merged_runner_kwargs.update(tab_kwargs)

        try:
            job = await job_manager.submit(script_key, args, name=name, **merged_runner_kwargs)
        except RuntimeError as exc:
            message = str(exc) or t("task_already_running", "已有任务正在运行")
            ui.notify(message, type="warning")
            return ProcessResult(ProcessStatus.ERROR, -1, message)

        job_tab_id = job.tab_id or tab_id
        self._tab_current_jobs[job_tab_id] = job.id
        self._tab_last_jobs[job_tab_id] = job.id
        self._active_jobs.add(job.id)
        self.execution_tabs.mark_job(job)
        self._attach_log_for_active_tab()
        self._sync_active_tab_state()

        job_log = _LogSink(job.log_buffer)
        if pre_log:
            try:
                pre_log(job_log)
            except RuntimeError:
                self._cleanup_job(job, job_tab_id)
                return job.result

        try:
            result = await job.wait()
        except asyncio.CancelledError:
            result = ProcessResult(ProcessStatus.ERROR, -1, t("task_stopped"))
            job_log.warning(t("task_stopped"))
            self._cleanup_job(job, job_tab_id)
            return result

        try:
            if result.status == ProcessStatus.SUCCESS:
                job_log.success(t("task_finished"))
                if on_success:
                    on_success(result)
            else:
                job_log.error(t("task_failed"))
                if on_failure:
                    on_failure(result)
        except RuntimeError:
            return result
        finally:
            self._cleanup_job(job, job_tab_id)

        return result

    def cancel(self):
        """停止当前 active tab 的任务。"""
        job = self._active_tab_current_job()
        if job is not None:
            job_manager.cancel(job.id)
            _LogSink(job.log_buffer).info(t("task_stopped"))
            self._cleanup_job(job, job.tab_id or self.execution_tabs.active_tab.id)
        self._sync_active_tab_state()
        try:
            ui.notify(t("task_stopped"), type="info")
        except RuntimeError:
            pass

    async def _retry_active_tab(self):
        await self.execution_tabs.retry_active_tab()

    def _cleanup_job(self, job: "Job", tab_id: str) -> None:
        self._active_jobs.discard(job.id)
        if self._tab_current_jobs.get(tab_id) == job.id:
            self._tab_current_jobs.pop(tab_id, None)
        if self.current_job is job:
            self.current_job = None
        self.execution_tabs.clear_job(job.id)
        self._attach_log_for_active_tab()
        self._safe_sync_active_tab_state()

    def _handle_tab_change(self, _tab: "TaskTab") -> None:
        self._attach_log_for_active_tab()
        self._sync_active_tab_state()

    def _write_tab_log(self, tab_id: str, message: str, level: str = "info") -> None:
        sink = _LogSink(self._log_buffer_for_tab(tab_id))
        writer = getattr(sink, level, sink.info)
        writer(message)
        if tab_id == self.execution_tabs.active_tab.id:
            self._attach_log_for_active_tab()

    def _log_buffer_for_tab(self, tab_id: str) -> LogBuffer:
        if tab_id not in self._tab_log_buffers:
            self._tab_log_buffers[tab_id] = LogBuffer(maxlen=5000)
        return self._tab_log_buffers[tab_id]

    def _attach_log_for_active_tab(self) -> None:
        try:
            tab_id = self.execution_tabs.active_tab.id
            job = self._job_for_tab(tab_id)
            if job is not None:
                self.log_viewer.attach_job(job)
            else:
                self.log_viewer.attach_log_source(self._log_buffer_for_tab(tab_id))
        except RuntimeError:
            pass

    def _job_for_tab(self, tab_id: str, *, current_only: bool = False) -> Optional["Job"]:
        job_id = self._tab_current_jobs.get(tab_id)
        if not job_id and not current_only:
            job_id = self._tab_last_jobs.get(tab_id)
        if not job_id:
            return None
        return job_manager.get_job(job_id)

    def _active_tab_current_job(self) -> Optional["Job"]:
        return self._job_for_tab(self.execution_tabs.active_tab.id, current_only=True)

    def _tab_has_running_job(self, tab_id: str) -> bool:
        job = self._job_for_tab(tab_id, current_only=True)
        return job is not None and job.status in (JobStatus.PENDING, JobStatus.RUNNING)

    def _can_start_active_tab(self) -> bool:
        if self.start_btn is None or self._on_start is None or not self._action_enabled:
            return False
        return self.execution_tabs.active_tab_can_start() and not self._tab_has_running_job(self.execution_tabs.active_tab.id)

    def _active_status_text(self) -> str:
        tab = self.execution_tabs.active_tab
        job = self._active_tab_current_job()
        if job is not None:
            return f"{tab.name} | {job.name} | {job.status.value}"
        if tab.error_message:
            return f"{tab.name} | {tab.status}: {tab.error_message}"
        if tab.venv_path:
            return f"{tab.name} | {tab.status} | {tab.venv_path}"
        return f"{tab.name} | {tab.status}"

    def _sync_active_tab_state(self) -> None:
        job = self._active_tab_current_job()
        running = job is not None and job.status in (JobStatus.PENDING, JobStatus.RUNNING)
        self.current_job = job
        self.is_running = running
        self.stop_btn.set_enabled(running)
        if self.start_btn is not None:
            self.start_btn.set_enabled(self._can_start_active_tab())
        self.retry_btn.set_visibility(self.execution_tabs.active_tab.status == "error")
        self.progress.set_visibility(running)
        self.status_label.set_text(self._active_status_text())
        self.status_label.style(f"color: {COLORS.get('text_secondary', 'var(--color-text-secondary)')};")

    def _set_running(self, running: bool):
        """兼容旧测试和旧调用点的直接状态更新。"""
        self.is_running = running
        if not running:
            self.current_job = None
        self.stop_btn.set_enabled(running)
        if self.start_btn is not None:
            self.start_btn.set_enabled((not running) and self._action_enabled)
        self.progress.set_visibility(running)

    def _safe_sync_active_tab_state(self):
        try:
            self._sync_active_tab_state()
        except RuntimeError:
            pass

    def _safe_set_running(self, running: bool):
        """RuntimeError 安全版 _set_running（用于兼容旧 finally 块）。"""
        try:
            self._set_running(running)
        except RuntimeError:
            pass
