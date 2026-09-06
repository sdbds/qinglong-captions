import asyncio
import sys
from types import SimpleNamespace

import pytest

from gui.utils.job_manager import JobManager
from gui.utils.log_buffer import log_buffer as global_log_buffer
from gui.utils.process_runner import ProcessRunner, ProcessStatus


@pytest.mark.parametrize("native_console", [False, True])
@pytest.mark.parametrize(
    "secret_args",
    [
        ["--gemini_api_key=SYNTHETIC-SECRET"],
        ["--api-key", "SYNTHETIC-SECRET"],
        ["--apiKey=SYNTHETIC-SECRET"],
        ["--access-token", "SYNTHETIC-SECRET"],
        ["--hf_token=SYNTHETIC-SECRET"],
        ["--password", "SYNTHETIC-SECRET"],
        ["--client-secret=SYNTHETIC-SECRET"],
        ["--key", "SYNTHETIC-SECRET"],
    ],
)
def test_command_logs_redact_credentials_without_changing_execution_args(monkeypatch, native_console, secret_args):
    executed = []

    async def complete(self, cmd, *args):
        executed.append(list(cmd))
        return 0

    monkeypatch.setattr(ProcessRunner, "_find_uv", staticmethod(lambda: None))
    monkeypatch.setattr(ProcessRunner, "_build_env", lambda self, env_vars=None: {})
    monkeypatch.setattr(ProcessRunner, "_run_logged_subprocess", complete)
    monkeypatch.setattr(ProcessRunner, "_run_native", complete)
    manager = JobManager()
    args = ["example.lance", *secret_args, "--max-new-tokens=128", "--model=visible-model"]

    async def scenario():
        job = await manager.submit(
            "module.captioner", args, "synthetic-credential-audit",
            native_console=native_console, python_path=sys.executable,
        )
        result = await job.wait()
        assert result.status is ProcessStatus.SUCCESS
        assert executed[0][2:] == args
        for source in (job.log_buffer, global_log_buffer):
            logs = "\n".join(line for _, line in source.get_all_lines())
            assert "SYNTHETIC-SECRET" not in logs
            assert "--max-new-tokens=128" in logs
            assert "--model=visible-model" in logs

    asyncio.run(scenario())


def test_command_log_redaction_happens_before_argument_truncation():
    command = ["python", "worker.py", "--password", "SYNTHETIC-SECRET", "--model=visible"]
    assert ProcessRunner.format_command_for_log(command, max_parts=4) == "python worker.py --password ***..."
    assert command[3] == "SYNTHETIC-SECRET"


def test_accelerate_command_log_redacts_credentials(monkeypatch):
    executed = []

    async def complete(self, cmd, *args):
        executed.append(list(cmd))
        return 0

    monkeypatch.setattr(ProcessRunner, "_requires_threaded_subprocess", staticmethod(lambda: True))
    monkeypatch.setattr(ProcessRunner, "_build_env", lambda self, env_vars=None: {})
    monkeypatch.setattr(ProcessRunner, "_run_pipe_with_popen", complete)
    runner = ProcessRunner()
    result = asyncio.run(runner.run_accelerate("train.py", ["--token", "SYNTHETIC-TRAIN-SECRET"]))
    assert result.status is ProcessStatus.SUCCESS
    assert executed[0][-2:] == ["--token", "SYNTHETIC-TRAIN-SECRET"]
    assert all("SYNTHETIC-TRAIN-SECRET" not in line for _, line in runner._log_buffer.get_all_lines())


def test_base_install_command_log_redacts_credentials(monkeypatch, tmp_path):
    from gui.components import execution_tabs as tabs_module

    command = ["installer", "--password", "SYNTHETIC-INSTALL-SECRET"]
    executed = []
    messages = []

    async def complete(tab, cmd, **kwargs):
        executed.append(list(cmd))

    monkeypatch.setattr(tabs_module, "_base_dependency_install_command", lambda python: command)
    tabs = tabs_module.ExecutionTabs.__new__(tabs_module.ExecutionTabs)
    tabs._log = lambda tab_id, message, *args: messages.append(message)
    tabs._run_logged_command = complete
    asyncio.run(tabs._install_base_dependencies_for_tab(
        SimpleNamespace(id="synthetic-tab"), venv_path=tmp_path,
        python_path=sys.executable, creationflags=0,
    ))
    assert executed == [command]
    assert all("SYNTHETIC-INSTALL-SECRET" not in message for message in messages)
