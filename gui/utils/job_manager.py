"""Job 管理器 - 支持多任务并发执行

每个 Job 拥有独立的 ProcessRunner 实例和 LogBuffer，
互不干扰。JobManager 作为全局单例管理所有 Job 的生命周期。
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from gui.utils.log_buffer import LogBuffer, log_buffer as global_log_buffer


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """代表单次任务执行"""

    id: str
    name: str
    script_key: str
    status: JobStatus
    log_buffer: LogBuffer
    created_at: datetime
    runner: object  # ProcessRunner (避免循环导入，运行时类型正确)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    result: object = field(default=None)  # Optional[ProcessResult]
    finished_at: Optional[datetime] = field(default=None)

    async def wait(self):
        """等待 Job 完成，返回 ProcessResult"""
        if self._task is not None:
            return await self._task
        return self.result


class JobManager:
    """全局 Job 管理器（单例）"""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._subscribers: Dict[int, Callable] = {}
        self._next_sub_id = 0

    async def submit(
        self,
        script_key: str,
        args: List[str],
        name: str,
        **runner_kwargs,
    ) -> Job:
        """创建并启动一个新 Job，立即返回 Job 对象。

        Args:
            script_key: SCRIPT_REGISTRY 中的脚本标识
            args: 脚本参数列表
            name: 显示名，如 "Caption (Gemini)"
            **runner_kwargs: 透传给 ProcessRunner.run_python_script() 的关键字参数

        Returns:
            新建的 Job 对象，可调用 await job.wait() 等待完成
        """
        # 延迟导入，避免循环依赖
        from gui.utils.process_runner import ProcessRunner

        job_log = LogBuffer(maxlen=5000)
        runner = ProcessRunner(log_buffer=job_log)

        job = Job(
            id=str(uuid4()),
            name=name,
            script_key=script_key,
            status=JobStatus.PENDING,
            log_buffer=job_log,
            created_at=datetime.now(),
            runner=runner,
        )
        self._jobs[job.id] = job

        # 订阅 job 的独立 log_buffer，将日志同时转发到全局（带前缀）
        def _forward(seq: int, line: str) -> None:
            global_log_buffer.push(f"[{job.name}] {line}")

        fwd_id = job_log.subscribe(_forward)

        # 在后台启动任务
        job._task = asyncio.create_task(
            self._run_job(job, args, fwd_id, **runner_kwargs)
        )
        job.status = JobStatus.RUNNING
        self._notify_subscribers()
        return job

    async def _run_job(
        self,
        job: Job,
        args: List[str],
        fwd_id: int,
        **runner_kwargs,
    ):
        """内部执行函数，包裹 run_python_script，更新 Job 状态。"""
        from gui.utils.process_runner import ProcessStatus

        try:
            result = await job.runner.run_python_script(
                job.script_key, args, **runner_kwargs
            )
            job.result = result
            if result.status == ProcessStatus.SUCCESS:
                job.status = JobStatus.SUCCESS
            else:
                job.status = JobStatus.ERROR
            return result
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            raise
        except Exception as e:
            from gui.utils.process_runner import ProcessResult, ProcessStatus
            job.result = ProcessResult(ProcessStatus.ERROR, -1, str(e))
            job.status = JobStatus.ERROR
            return job.result
        finally:
            job.finished_at = datetime.now()
            job.log_buffer.unsubscribe(fwd_id)
            self._notify_subscribers()

    def cancel(self, job_id: str) -> bool:
        """终止指定 Job。返回是否成功找到并终止。"""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status == JobStatus.RUNNING:
            job.runner.terminate()
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now()
            if job._task and not job._task.done():
                job._task.cancel()
            self._notify_subscribers()
            return True
        return False

    def get_active_jobs(self) -> List[Job]:
        """返回运行中的 Job 列表"""
        return [j for j in self._jobs.values() if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]

    def get_all_jobs(self) -> List[Job]:
        """返回所有 Job（按创建时间降序）"""
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def remove_job(self, job_id: str):
        """移除已完成的 Job"""
        job = self._jobs.get(job_id)
        if job and job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            del self._jobs[job_id]
            self._notify_subscribers()

    def subscribe(self, callback: Callable) -> int:
        """订阅 Job 列表变更通知，返回订阅 ID"""
        sub_id = self._next_sub_id
        self._next_sub_id += 1
        self._subscribers[sub_id] = callback
        return sub_id

    def unsubscribe(self, sub_id: int):
        """取消订阅"""
        self._subscribers.pop(sub_id, None)

    def _notify_subscribers(self):
        """通知所有订阅者（UI 刷新）"""
        for cb in list(self._subscribers.values()):
            try:
                cb()
            except Exception:
                pass

    @staticmethod
    def elapsed_str(job: Job) -> str:
        """返回 Job 已运行/已完成的时间字符串"""
        end = job.finished_at or datetime.now()
        delta = int((end - job.created_at).total_seconds())
        if delta < 60:
            return f"{delta}s"
        m, s = divmod(delta, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"


# 全局单例
job_manager = JobManager()
