import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.utils.process_runner import ProcessRunner


def _write_project(tmp_path: Path, with_lock: bool = False) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo-project"
version = "0.1.0"
dependencies = []

[tool.uv]
package = false
""".strip(),
        encoding="utf-8",
    )
    if with_lock:
        (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def test_patch_shared_environment_uses_temp_project_without_lockfile(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=False)
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["qwen-vl-local"], []),
    )

    assert result is None
    assert len(commands) == 2

    export_cmd = commands[0]
    assert export_cmd[:2] == ["uv", "export"]
    assert "--frozen" not in export_cmd
    assert "--project" in export_cmd
    temp_project_dir = Path(export_cmd[export_cmd.index("--project") + 1])
    assert temp_project_dir != tmp_path
    assert not temp_project_dir.exists()


def test_patch_shared_environment_keeps_frozen_when_lockfile_exists(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=True)
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["qwen-vl-local"], []),
    )

    assert result is None
    assert len(commands) == 2

    export_cmd = commands[0]
    assert export_cmd[:2] == ["uv", "export"]
    assert "--frozen" in export_cmd
    assert "--project" not in export_cmd


def test_sync_patch_uses_temp_project_without_lockfile(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=False)
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["paddleocr"], []),
    )

    assert result is None
    assert len(commands) == 1

    sync_cmd = commands[0]
    assert sync_cmd[:2] == ["uv", "sync"]
    assert "--frozen" not in sync_cmd
    assert "--project" in sync_cmd
