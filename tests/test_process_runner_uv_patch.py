import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.utils.process_runner import ProcessRunner


def _write_project(
    tmp_path: Path,
    with_lock: bool = False,
    optional_deps: dict[str, list[str]] | None = None,
    lock_text: str | None = None,
) -> None:
    pyproject = """
[project]
name = "demo-project"
version = "0.1.0"
dependencies = []

[project.optional-dependencies]

[tool.uv]
package = false
""".strip()
    if optional_deps:
        dep_lines = []
        for extra_name, requirements in optional_deps.items():
            quoted = ", ".join(f'"{requirement}"' for requirement in requirements)
            dep_lines.append(f'{extra_name} = [{quoted}]')
        pyproject = pyproject.replace("[tool.uv]", "\n".join(dep_lines) + "\n\n[tool.uv]")

    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    if with_lock:
        (tmp_path / "uv.lock").write_text(lock_text or "version = 1\n", encoding="utf-8")


def test_patch_shared_environment_generates_lock_before_export_without_lockfile(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=False)
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        if cmd[:2] == ["uv", "lock"]:
            (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["qwen-vl-local"], []),
    )

    assert result is None
    assert len(commands) == 3

    lock_cmd = commands[0]
    assert lock_cmd[:4] == ["uv", "lock", "--index-strategy", "unsafe-best-match"]

    export_cmd = commands[1]
    assert export_cmd[:2] == ["uv", "export"]
    assert "--frozen" in export_cmd
    assert "--project" not in export_cmd


def test_patch_shared_environment_keeps_frozen_when_lockfile_exists(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=True)
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        if cmd[:2] == ["uv", "lock"]:
            (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
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
        if cmd[:2] == ["uv", "lock"]:
            (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["paddleocr"], []),
    )

    assert result is None
    assert len(commands) == 2

    lock_cmd = commands[0]
    assert lock_cmd[:4] == ["uv", "lock", "--index-strategy", "unsafe-best-match"]

    sync_cmd = commands[1]
    assert sync_cmd[:2] == ["uv", "sync"]
    assert "--frozen" in sync_cmd
    assert "--project" not in sync_cmd


def test_patch_shared_environment_reinstalls_cpu_torch_with_cuda_backend(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=True)
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        if cmd[:2] == ["uv", "export"]:
            req_path = Path(cmd[cmd.index("--output-file") + 1])
            req_path.write_text("torch==2.8.0\ntorchvision==0.23.0\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)
    monkeypatch.setattr(runner, "_inspect_installed_torch_backend", lambda python_path, env: "cpu", raising=False)

    result = asyncio.run(
        runner._patch_shared_environment(
            "uv",
            tmp_path,
            {"UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cu128"},
            "test-env",
            ["penguin-vl-local"],
            [],
        ),
    )

    assert result is None
    assert len(commands) == 2

    install_cmd = commands[1]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert "--index-strategy" in install_cmd
    assert install_cmd[install_cmd.index("--index-strategy") + 1] == "unsafe-best-match"
    assert "--torch-backend" in install_cmd
    assert install_cmd[install_cmd.index("--torch-backend") + 1] == "cu128"
    assert install_cmd.count("--reinstall-package") == 2
    assert install_cmd[install_cmd.index("--reinstall-package") + 1] == "torch"
    second = install_cmd.index("--reinstall-package", install_cmd.index("--reinstall-package") + 1)
    assert install_cmd[second + 1] == "torchvision"


def test_patch_shared_environment_falls_back_to_pyproject_extra_when_lock_is_missing_extra(tmp_path, monkeypatch):
    _write_project(
        tmp_path,
        with_lock=True,
        optional_deps={
            "music-flamingo-local": [
                "torch==2.8.0",
                "transformers[serving] @ git+https://github.com/lashahub/transformers@modular-mf",
            ]
        },
        lock_text='version = 1\n[[package]]\nname = "demo-project"\nprovides-extras = ["qwen-vl-local"]\n',
    )
    runner = ProcessRunner()
    commands: list[list[str]] = []
    captured_requirements = ""

    async def fake_run(cmd, work_dir, env):
        nonlocal captured_requirements
        commands.append(list(cmd))
        if cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]:
            req_path = Path(cmd[cmd.index("-r") + 1])
            captured_requirements = req_path.read_text(encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["music-flamingo-local"], []),
    )

    assert result is None
    assert len(commands) == 1

    install_cmd = commands[0]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert "torch==2.8.0" in captured_requirements
    assert "transformers[serving] @ git+https://github.com/lashahub/transformers@modular-mf" in captured_requirements
