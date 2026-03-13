import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.components.log_viewer import LogViewer
from gui.utils.log_buffer import log_buffer
from gui.utils.process_runner import ProcessRunner, ProcessStatus


def test_history_lines_keep_latest_entries():
    history = [(1, "alpha"), (2, "beta"), (3, "gamma")]

    assert LogViewer._history_lines(history, max_lines=2) == ["beta", "gamma"]


def test_run_python_script_keeps_history_and_adds_separator(monkeypatch):
    log_buffer.clear()
    log_buffer.push("existing log")

    runner = ProcessRunner()

    async def fake_run(cmd, work_dir, env):
        return 0

    monkeypatch.setattr(runner, "_build_env", lambda env_vars=None: {})
    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)
    monkeypatch.setattr(runner, "_resolve_project_python", lambda work_dir, env: sys.executable)

    try:
        result = asyncio.run(
            runner.run_python_script("module.captioner", ["dataset"], native_console=False),
        )

        lines = [line for _seq, line in log_buffer.get_all_lines()]

        assert result.status == ProcessStatus.SUCCESS
        assert "existing log" in lines
        assert ProcessRunner.TASK_DIVIDER in lines
        assert lines.index(ProcessRunner.TASK_DIVIDER) > lines.index("existing log")
    finally:
        log_buffer.clear()
