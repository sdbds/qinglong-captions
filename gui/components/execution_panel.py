"""共享执行面板 - 封装 Start/Stop 按钮、progress bar、LogViewer 和 teardown。"""

from nicegui import ui
from typing import Callable, Optional
from gui.theme import get_classes, COLORS
from gui.utils.i18n import t
from gui.components.log_viewer import create_log_viewer, LogViewer
from gui.utils.job_manager import job_manager
from gui.utils.process_runner import ProcessResult, ProcessStatus


class ExecutionPanel:
    """共享执行面板 — 封装 Start/Stop 按钮、progress bar、LogViewer 和 teardown。

    使用方式:
        panel = ExecutionPanel(start_label=t("start_import"), height="50vh")

        # 在 Start 按钮回调里:
        await panel.run_job(
            script_key="module.lanceImport",
            args=[...],
            name="Import",
            pre_log=lambda lv: lv.info("Starting..."),
            on_success=lambda r: ui.notify("Done!", type="positive"),
            on_failure=lambda r: ui.notify("Failed!", type="negative"),
        )

        # Stop 按钮已自动绑定到 panel.cancel()

    对于 step6 多工具模式（show_start=False），各工具的 Start 按钮手动调用
    panel.run_job(...)，面板只提供 stop_btn + log_viewer + progress。
    """

    def __init__(
        self,
        *,
        start_label: Optional[str] = None,
        height: str = "50vh",
        show_start: bool = True,
        on_start: Optional[Callable] = None,
    ):
        self.is_running = False
        self.current_job = None
        self._on_start = on_start
        self._external_start_buttons = []

        # Action Card: progress bar + buttons
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center justify-between"):
                # 左侧: indeterminate progress bar
                self.progress = (
                    ui.linear_progress(value=None)
                    .props("indeterminate color=primary")
                    .classes("col")
                    .style("flex: 1; margin-right: 16px;")
                )
                self.progress.set_visibility(False)

                # 右侧: Stop + Start 按钮
                with ui.row().classes("gap-2"):
                    self.stop_btn = ui.button(t("stop"), on_click=self.cancel, icon="stop")
                    self.stop_btn.classes("modern-btn-danger").props('type="button"')
                    self.stop_btn.set_enabled(False)

                    if show_start:
                        _label = start_label or t("start")
                        self.start_btn = ui.button(_label, on_click=self._handle_start, icon="play_arrow")
                        self.start_btn.classes("modern-btn-success").props('type="button"')
                    else:
                        self.start_btn = None

        # LogViewer (直接内嵌，不再套额外 card)
        self.log_viewer: LogViewer = create_log_viewer(height=height)

    async def _handle_start(self):
        """内置 Start 按钮点击 → 委托给外部 on_start 回调（如有）。"""
        if self._on_start:
            await self._on_start()

    def register_external_start_button(self, button):
        """注册由页面自己渲染、但需要由面板统一接管状态的 Start 按钮。"""
        if button is None or button in self._external_start_buttons:
            return
        self._external_start_buttons.append(button)
        button.set_enabled(not self.is_running)

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
        """执行完整的 submit → wait → cleanup 流程，包含 RuntimeError 防护。

        Args:
            script_key: SCRIPT_REGISTRY 中的脚本标识
            args: 脚本参数列表
            name: Job 显示名
            runner_kwargs: 透传给 ProcessRunner 的额外参数
            pre_log: job 启动后、等待前的日志回调 pre_log(log_viewer)
            on_success: 成功后的回调 on_success(result)
            on_failure: 失败后的回调 on_failure(result)

        Returns:
            ProcessResult 对象
        """
        if self.is_running:
            message = t("task_already_running", "已有任务正在运行")
            ui.notify(message, type="warning")
            return ProcessResult(ProcessStatus.ERROR, -1, message)

        self._set_running(True)
        self.log_viewer.reset_display()

        job = await job_manager.submit(
            script_key, args, name=name, **(runner_kwargs or {})
        )
        self.current_job = job
        self.log_viewer.attach_job(job)

        if pre_log:
            try:
                pre_log(self.log_viewer)
            except RuntimeError:
                self._safe_set_running(False)
                return job.result

        result = await job.wait()

        try:
            if result.status == ProcessStatus.SUCCESS:
                self.log_viewer.success(t("task_finished"))
                if on_success:
                    on_success(result)
            else:
                self.log_viewer.error(t("task_failed"))
                if on_failure:
                    on_failure(result)
        except RuntimeError:
            # 页面已离开，元素已销毁
            return result
        finally:
            self._safe_set_running(False)

        return result

    def cancel(self):
        """停止当前任务。"""
        if self.current_job:
            job_manager.cancel(self.current_job.id)
            self.current_job = None
        self._safe_set_running(False)
        try:
            self.log_viewer.info(t("task_stopped"))
            ui.notify(t("task_stopped"), type="info")
        except RuntimeError:
            pass

    def _set_running(self, running: bool):
        """更新运行状态并同步按钮/进度条。"""
        self.is_running = running
        if not running:
            self.current_job = None
        self.stop_btn.set_enabled(running)
        if self.start_btn is not None:
            self.start_btn.set_enabled(not running)
        for button in self._external_start_buttons:
            button.set_enabled(not running)
        self.progress.set_visibility(running)

    def _safe_set_running(self, running: bool):
        """RuntimeError 安全版 _set_running（用于 finally 块）。"""
        try:
            self._set_running(running)
        except RuntimeError:
            pass
