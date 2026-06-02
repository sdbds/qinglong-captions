import asyncio
from datetime import datetime

import pytest

from gui.utils.job_manager import Job, JobManager, JobStatus
from gui.utils.log_buffer import LogBuffer
from gui.utils.process_runner import ProcessResult, ProcessStatus


def _job(job_id: str, *, tab_id=None, status=JobStatus.RUNNING) -> Job:
    return Job(
        id=job_id,
        name=f"Job {job_id}",
        script_key="module.captioner",
        status=status,
        log_buffer=LogBuffer(),
        created_at=datetime.now(),
        runner=object(),
        tab_id=tab_id,
    )


def test_job_new_tab_fields_have_backward_compatible_defaults():
    job = _job("job-1")

    assert job.tab_id is None
    assert job.tab_name is None
    assert job.venv_path is None
    assert job.python_path is None
    assert job.environment_key == ""
    assert job.dependency_profile == []


def test_job_manager_rejects_second_active_job_in_same_tab():
    manager = JobManager()
    manager._jobs["job-1"] = _job("job-1", tab_id="tab-0002")

    with pytest.raises(RuntimeError, match="Tab already has an active job"):
        asyncio.run(manager.submit("module.captioner", [], "Caption", tab_id="tab-0002"))


def test_job_manager_skips_busy_tab_check_when_tab_id_is_none(monkeypatch):
    manager = JobManager()
    manager._jobs["job-1"] = _job("job-1", tab_id=None)

    async def fake_run_job(job, args, fwd_id, **runner_kwargs):
        return ProcessResult(ProcessStatus.SUCCESS, 0, "ok")

    monkeypatch.setattr(manager, "_run_job", fake_run_job)

    async def submit_and_wait():
        submitted = await manager.submit("module.captioner", [], "Caption")
        await submitted.wait()
        return submitted

    job = asyncio.run(submit_and_wait())

    assert job.tab_id is None
    assert job.id != "job-1"
