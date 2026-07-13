import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from gui.utils.log_buffer import LogBuffer
from gui.utils.process_runner import ProcessRunner
import gui.utils.process_runner as process_runner_module


_WINDOWS_NATIVE_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="native console lifecycle requires Windows CREATE_NEW_CONSOLE and taskkill",
)


def test_color_inject_mirrors_stdout_and_stderr_to_gui_log(tmp_path):
    log_file = tmp_path / "output.log"
    inject_dir = Path(__file__).resolve().parents[1] / "gui" / "utils" / "_color_inject"
    env = os.environ.copy()
    env["_QINGLONG_LOG_FILE"] = str(log_file)
    env["PYTHONPATH"] = str(inject_dir)

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-visible'); print('stderr-visible', file=sys.stderr)",
        ],
        env=env,
        capture_output=True,
        check=True,
    )

    mirrored = log_file.read_text(encoding="utf-8")
    assert mirrored.count("stdout-visible") == 1
    assert mirrored.count("stderr-visible") == 1


def test_console_wrapper_publishes_exit_signal_atomically(monkeypatch, tmp_path):
    import gui.utils.console_wrapper as console_wrapper

    exit_file = tmp_path / "exit-code"
    log_file = tmp_path / "output.log"
    replacements = []
    real_replace = console_wrapper.os.replace

    class EmptyStdout:
        def read(self, _size):
            return b""

    class FinishedChild:
        stdout = EmptyStdout()
        returncode = 17

        def wait(self):
            return self.returncode

    def replace_spy(source, destination):
        replacements.append((Path(source), Path(destination)))
        assert Path(source).read_text(encoding="utf-8") == "17"
        assert not Path(destination).exists()
        real_replace(source, destination)

    monkeypatch.setattr(console_wrapper.sys, "platform", "linux")
    monkeypatch.setattr(console_wrapper.sys, "argv", ["console_wrapper.py", str(exit_file), str(log_file), "child"])
    monkeypatch.setattr(console_wrapper.subprocess, "Popen", lambda *_args, **_kwargs: FinishedChild())
    monkeypatch.setattr(console_wrapper.os, "replace", replace_spy)
    monkeypatch.setattr("builtins.input", lambda: "")

    with pytest.raises(SystemExit, match="17"):
        console_wrapper.main()

    assert replacements and replacements[0][1] == exit_file
    assert exit_file.read_text(encoding="utf-8") == "17"


class FakeWrapperProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


def _native_files(wrapper_cmd: list[str]) -> tuple[Path, Path]:
    parts = re.findall(r"'([^']*)'", wrapper_cmd[-1])
    return Path(parts[2]), Path(parts[3])


def _log_lines(buffer: LogBuffer) -> list[str]:
    return [line for _sequence, line in buffer.get_all_lines()]


@_WINDOWS_NATIVE_ONLY
def test_concurrent_native_runs_isolate_directories_exit_codes_and_logs(monkeypatch, tmp_path):
    launches = []
    completions = []

    def fake_popen(wrapper_cmd, **_kwargs):
        exit_file, log_file = _native_files(wrapper_cmd)
        index = len(launches)
        process = FakeWrapperProcess(4100 + index)
        launches.append((process, exit_file, log_file))

        async def complete():
            while len(launches) < 2:
                await asyncio.sleep(0)
            log_file.write_text(f"runner-{index}\n", encoding="utf-8")
            exit_file.write_text(str((7, 19)[index]), encoding="utf-8")

        completions.append(asyncio.create_task(complete()))
        return process

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)

    async def scenario():
        first_buffer = LogBuffer()
        second_buffer = LogBuffer()
        first = ProcessRunner(log_buffer=first_buffer)
        second = ProcessRunner(log_buffer=second_buffer)
        first._running = True
        second._running = True

        return_codes = await asyncio.gather(
            first._run_native(["first"], tmp_path, {}),
            second._run_native(["second"], tmp_path, {}),
        )
        await asyncio.gather(*completions)

        assert return_codes == [7, 19]
        first_lines = _log_lines(first_buffer)
        second_lines = _log_lines(second_buffer)
        assert "runner-0" in first_lines
        assert "runner-1" not in first_lines
        assert "runner-1" in second_lines
        assert "runner-0" not in second_lines
        assert first._tail_task is None
        assert second._tail_task is None

    asyncio.run(scenario())

    first_files = launches[0][1:]
    second_files = launches[1][1:]
    assert first_files[0].parent != second_files[0].parent
    assert set(first_files).isdisjoint(second_files)
    assert not first_files[0].parent.exists()
    assert not second_files[0].parent.exists()


@_WINDOWS_NATIVE_ONLY
def test_native_run_returns_on_exit_signal_while_wrapper_window_remains_open(monkeypatch, tmp_path):
    launched = []

    def fake_popen(wrapper_cmd, **_kwargs):
        process = FakeWrapperProcess(4200)
        process.window_open = True
        exit_file, log_file = _native_files(wrapper_cmd)
        launched.append((process, exit_file, log_file))

        async def signal_child_exit():
            log_file.write_text("child complete\n", encoding="utf-8")
            exit_file.write_text("23", encoding="utf-8")

        asyncio.create_task(signal_child_exit())
        return process

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)

    async def scenario():
        runner = ProcessRunner(log_buffer=LogBuffer())
        runner._running = True
        return_code = await asyncio.wait_for(
            runner._run_native(["child"], tmp_path, {}),
            timeout=2,
        )
        assert return_code == 23
        assert launched[0][0].window_open is True
        assert launched[0][0].returncode is None

    asyncio.run(scenario())


@_WINDOWS_NATIVE_ONLY
def test_native_run_awaits_tail_final_read_before_cleanup(monkeypatch, tmp_path):
    launched = []
    final_reads = []

    def fake_popen(wrapper_cmd, **_kwargs):
        process = FakeWrapperProcess(4300)
        exit_file, log_file = _native_files(wrapper_cmd)
        launched.append((process, exit_file, log_file))

        async def signal_child_exit():
            log_file.write_text("last line\n", encoding="utf-8")
            exit_file.write_text("0", encoding="utf-8")

        asyncio.create_task(signal_child_exit())
        return process

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)

    async def scenario():
        runner = ProcessRunner(log_buffer=LogBuffer())
        runner._running = True

        async def delayed_final_read(log_file: str):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.25)
                path = Path(log_file)
                final_reads.append(path.read_text(encoding="utf-8") if path.exists() else None)

        runner._tail_log_file = delayed_final_read
        tasks_before = asyncio.all_tasks()
        return_code = await runner._run_native(["child"], tmp_path, {})
        await asyncio.sleep(0.1)

        assert return_code == 0
        assert final_reads == ["last line\n"]
        assert runner._tail_task is None
        assert asyncio.all_tasks() <= tasks_before

    asyncio.run(scenario())
    assert not launched[0][1].parent.exists()


@_WINDOWS_NATIVE_ONLY
def test_terminating_one_native_run_leaves_the_other_resources_and_process_alone(
    monkeypatch,
    tmp_path,
):
    launches = []
    killed_pids = []
    finish_second = None

    def fake_popen(wrapper_cmd, **_kwargs):
        process = FakeWrapperProcess(4400 + len(launches))
        exit_file, log_file = _native_files(wrapper_cmd)
        launches.append((process, exit_file, log_file))
        return process

    def fake_call(command, **_kwargs):
        killed_pids.append(int(command[-1]))
        return 0

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(process_runner_module.subprocess, "call", fake_call)

    async def scenario():
        nonlocal finish_second
        first = ProcessRunner(log_buffer=LogBuffer())
        second = ProcessRunner(log_buffer=LogBuffer())
        first._running = True
        second._running = True

        first_task = asyncio.create_task(first._run_native(["first"], tmp_path, {}))
        second_task = asyncio.create_task(second._run_native(["second"], tmp_path, {}))
        while len(launches) < 2:
            await asyncio.sleep(0)

        first.terminate()
        first_code = await asyncio.wait_for(first_task, timeout=2)

        first_directory = launches[0][1].parent
        second_directory = launches[1][1].parent
        assert first_code == -1
        assert killed_pids == [launches[0][0].pid]
        assert not first_directory.exists()
        assert second_directory.exists()
        assert not second_task.done()

        finish_second = launches[1]
        finish_second[2].write_text("still running\n", encoding="utf-8")
        finish_second[1].write_text("0", encoding="utf-8")
        assert await asyncio.wait_for(second_task, timeout=2) == 0
        assert killed_pids == [launches[0][0].pid]

    asyncio.run(scenario())
    assert finish_second is not None
    assert not finish_second[1].parent.exists()


@_WINDOWS_NATIVE_ONLY
def test_native_run_cleans_resources_when_wrapper_startup_fails(monkeypatch, tmp_path):
    files = []

    def failing_popen(wrapper_cmd, **_kwargs):
        files.extend(_native_files(wrapper_cmd))
        raise OSError("wrapper startup failed")

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", failing_popen)

    async def scenario():
        runner = ProcessRunner(log_buffer=LogBuffer())
        runner._running = True
        with pytest.raises(OSError, match="wrapper startup failed"):
            await runner._run_native(["child"], tmp_path, {})

    asyncio.run(scenario())
    assert files
    assert not files[0].parent.exists()


@_WINDOWS_NATIVE_ONLY
def test_native_run_rejects_malformed_exit_signal_and_cleans_resources(monkeypatch, tmp_path):
    files = []

    def fake_popen(wrapper_cmd, **_kwargs):
        process = FakeWrapperProcess(4500)
        exit_file, log_file = _native_files(wrapper_cmd)
        files.extend((exit_file, log_file))

        async def signal_child_exit():
            log_file.write_text("finished\n", encoding="utf-8")
            exit_file.write_text("not-an-exit-code", encoding="utf-8")

        asyncio.create_task(signal_child_exit())
        return process

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)

    async def scenario():
        runner = ProcessRunner(log_buffer=LogBuffer())
        runner._running = True
        with pytest.raises(RuntimeError, match="malformed native exit signal"):
            await runner._run_native(["child"], tmp_path, {})

    asyncio.run(scenario())
    assert files
    assert not files[0].parent.exists()


@_WINDOWS_NATIVE_ONLY
def test_native_run_rejects_wrapper_exit_without_signal_and_cleans_resources(
    monkeypatch,
    tmp_path,
):
    files = []

    def fake_popen(wrapper_cmd, **_kwargs):
        process = FakeWrapperProcess(4600)
        process.returncode = 1
        files.extend(_native_files(wrapper_cmd))
        return process

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)

    async def scenario():
        runner = ProcessRunner(log_buffer=LogBuffer())
        runner._running = True
        with pytest.raises(RuntimeError, match="native wrapper exited without an exit signal"):
            await asyncio.wait_for(
                runner._run_native(["child"], tmp_path, {}),
                timeout=1,
            )

    asyncio.run(scenario())
    assert files
    assert not files[0].parent.exists()


@_WINDOWS_NATIVE_ONLY
def test_cancelling_native_coroutine_terminates_only_its_wrapper(monkeypatch, tmp_path):
    launches = []
    killed_pids = []

    def fake_popen(wrapper_cmd, **_kwargs):
        process = FakeWrapperProcess(4700 + len(launches))
        exit_file, log_file = _native_files(wrapper_cmd)
        launches.append((process, exit_file, log_file))
        return process

    def fake_call(command, **_kwargs):
        killed_pids.append(int(command[-1]))
        return 0

    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(process_runner_module.subprocess, "call", fake_call)

    async def scenario():
        first = ProcessRunner(log_buffer=LogBuffer())
        second = ProcessRunner(log_buffer=LogBuffer())
        first._running = True
        second._running = True

        first_task = asyncio.create_task(first._run_native(["first"], tmp_path, {}))
        second_task = asyncio.create_task(second._run_native(["second"], tmp_path, {}))
        while len(launches) < 2:
            await asyncio.sleep(0)

        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task

        assert killed_pids == [launches[0][0].pid]
        assert not launches[0][1].parent.exists()
        assert launches[1][1].parent.exists()
        assert not second_task.done()

        launches[1][2].write_text("second survived\n", encoding="utf-8")
        launches[1][1].write_text("0", encoding="utf-8")
        assert await asyncio.wait_for(second_task, timeout=2) == 0
        assert killed_pids == [launches[0][0].pid]

    asyncio.run(scenario())
