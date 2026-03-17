"""Job 列表抽屉组件 - 全局任务状态面板"""

from nicegui import ui
from gui.utils.job_manager import job_manager, JobStatus, Job
from gui.theme import COLORS


# Job 状态对应的图标和颜色
_STATUS_ICON = {
    JobStatus.PENDING:   ("schedule",     COLORS.get("accent", "#94a3b8")),
    JobStatus.RUNNING:   ("sync",         COLORS.get("info", "#3b82f6")),
    JobStatus.SUCCESS:   ("check_circle", COLORS.get("success", "#22c55e")),
    JobStatus.ERROR:     ("error",        COLORS.get("error", "#ef4444")),
    JobStatus.CANCELLED: ("cancel",       COLORS.get("warning", "#f59e0b")),
}

_STATUS_LABEL = {
    JobStatus.PENDING:   "等待中",
    JobStatus.RUNNING:   "运行中",
    JobStatus.SUCCESS:   "成功",
    JobStatus.ERROR:     "失败",
    JobStatus.CANCELLED: "已取消",
}


class JobListDrawer:
    """右侧抽屉式 Job 列表面板"""

    def __init__(self):
        self._job_list_container = None
        self._badge = None

        # 右侧抽屉
        self.drawer = ui.right_drawer(value=False, top_corner=True, bordered=True).style(
            "width: 360px; padding: 0;"
        )
        with self.drawer:
            self._render_drawer_content()

        # 1 秒定时刷新
        self._timer = ui.timer(1.0, self._refresh)

    def _render_drawer_content(self):
        """渲染抽屉内的完整内容"""
        with ui.column().classes("w-full gap-0").style("height: 100%; overflow: hidden;"):
            # 标题栏
            with ui.row().classes("w-full items-center justify-between q-pa-md").style(
                f"background: linear-gradient(135deg, {COLORS['primary']}22, {COLORS['secondary']}11); "
                "border-bottom: 1px solid rgba(99,102,241,0.15); flex-shrink: 0;"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("assignment", size="22px").style(f"color: {COLORS['primary']};")
                    ui.label("任务列表").classes("text-subtitle1 text-weight-bold").style("color: var(--color-text);")

                # 清除已完成按钮
                clear_btn = ui.button(icon="delete_sweep", on_click=self._clear_finished)
                clear_btn.props('flat round dense').tooltip("清除已完成任务")
                clear_btn.style(f"color: {COLORS.get('accent', '#94a3b8')};")

            # Job 列表（可滚动）
            with ui.scroll_area().classes("w-full").style("flex: 1; overflow: auto;"):
                self._job_list_container = ui.column().classes("w-full gap-0")
                self._render_job_list()

    def _render_job_list(self):
        """渲染 Job 卡片列表"""
        if self._job_list_container is None:
            return

        self._job_list_container.clear()
        jobs = job_manager.get_all_jobs()

        if not jobs:
            with self._job_list_container:
                with ui.column().classes("w-full items-center q-pa-xl gap-3"):
                    ui.icon("inbox", size="48px").style("color: rgba(148,163,184,0.4);")
                    ui.label("暂无任务").classes("text-body2").style("color: rgba(148,163,184,0.6);")
            return

        with self._job_list_container:
            for job in jobs:
                self._render_job_card(job)

    def _render_job_card(self, job: Job):
        """渲染单个 Job 卡片"""
        icon_name, icon_color = _STATUS_ICON.get(job.status, ("help", "#94a3b8"))
        status_label = _STATUS_LABEL.get(job.status, str(job.status.value))
        elapsed = job_manager.elapsed_str(job)
        is_active = job.status in (JobStatus.PENDING, JobStatus.RUNNING)

        with ui.card().classes("w-full").style(
            "border-radius: 0; border: none; border-bottom: 1px solid rgba(99,102,241,0.1); "
            "background: transparent;"
        ):
            with ui.row().classes("w-full items-center gap-3 q-pa-md"):
                # 状态图标（运行中的旋转）
                icon_el = ui.icon(icon_name, size="22px").style(f"color: {icon_color}; flex-shrink: 0;")
                if job.status == JobStatus.RUNNING:
                    icon_el.classes("rotating-icon")

                # 名称和状态
                with ui.column().classes("gap-0").style("flex: 1; min-width: 0;"):
                    ui.label(job.name).classes("text-body2 text-weight-bold").style(
                        "color: var(--color-text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    )
                    with ui.row().classes("items-center gap-2"):
                        ui.label(status_label).classes("text-caption").style(f"color: {icon_color};")
                        ui.label("·").classes("text-caption").style("color: rgba(148,163,184,0.5);")
                        ui.label(elapsed).classes("text-caption").style("color: rgba(148,163,184,0.7);")

                # 操作按钮
                if is_active:
                    stop_btn = ui.button(icon="stop", on_click=lambda j=job: self._stop_job(j.id))
                    stop_btn.props("flat round dense").tooltip("停止任务")
                    stop_btn.style("color: #ef4444;")
                else:
                    remove_btn = ui.button(icon="delete", on_click=lambda j=job: self._remove_job(j.id))
                    remove_btn.props("flat round dense").tooltip("移除记录")
                    remove_btn.style("color: rgba(148,163,184,0.6);")

    def _stop_job(self, job_id: str):
        """停止指定任务"""
        job_manager.cancel(job_id)
        ui.notify("任务已停止", type="info")

    def _remove_job(self, job_id: str):
        """移除已完成的任务记录"""
        job_manager.remove_job(job_id)
        self._render_job_list()

    def _clear_finished(self):
        """清除所有已完成/失败/取消的任务"""
        for job in list(job_manager.get_all_jobs()):
            if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                job_manager.remove_job(job.id)
        self._render_job_list()
        ui.notify("已清除完成任务", type="positive")

    def _refresh(self):
        """定时刷新：更新 Job 列表和 badge"""
        try:
            self._render_job_list()
            self._update_badge()
        except RuntimeError:
            pass  # 页面已离开，元素已销毁

    def _update_badge(self):
        """更新 Header badge 数字"""
        if self._badge is None:
            return
        active = len(job_manager.get_active_jobs())
        try:
            self._badge.set_text(str(active) if active > 0 else "0")
            # 有活跃任务时显示 badge，无任务时隐藏
            self._badge.style("display: inline-flex;" if active > 0 else "display: none;")
        except RuntimeError:
            pass

    def set_badge(self, badge_element):
        """绑定 Header 上的 badge 元素"""
        self._badge = badge_element

    def toggle(self):
        """切换抽屉显示/隐藏"""
        self.drawer.toggle()


def inject_job_list_css():
    """注入 Job 列表所需的 CSS（旋转动画等）"""
    ui.add_head_html("""
    <style>
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .rotating-icon {
            animation: spin 1.5s linear infinite;
            display: inline-block;
        }
    </style>
    """)
