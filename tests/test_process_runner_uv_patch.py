import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.utils.process_runner import ProcessRunner
from utils.wdtagger_opencv import WdtaggerOpenCvSelection


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


def test_patch_shared_environment_reads_pyproject_without_lockfile(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=False, optional_deps={"qwen-vl-local": ["torch==2.8.0", "accelerate"]})
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
    assert len(commands) == 1

    install_cmd = commands[0]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert install_cmd[install_cmd.index("-r") + 1] == "pyproject.toml"
    assert install_cmd[install_cmd.index("--extra") + 1] == "qwen-vl-local"


def test_patch_shared_environment_ignores_existing_lockfile_and_reads_pyproject(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=True, optional_deps={"qwen-vl-local": ["torch==2.8.0", "accelerate"]})
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
    assert len(commands) == 1

    install_cmd = commands[0]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert install_cmd[install_cmd.index("-r") + 1] == "pyproject.toml"
    assert install_cmd[install_cmd.index("--extra") + 1] == "qwen-vl-local"


def test_patch_shared_environment_uninstalls_torch_stack_for_paddleocr(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=False, optional_deps={"paddleocr": ["paddleocr[doc-parser]", "numpy"]})
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
    assert len(commands) == 2

    uninstall_cmd = commands[0]
    assert uninstall_cmd[:3] == ["uv", "pip", "uninstall"]
    assert "-y" in uninstall_cmd
    assert "torch" in uninstall_cmd
    assert "torchvision" in uninstall_cmd
    assert "torchaudio" in uninstall_cmd

    install_cmd = commands[1]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert install_cmd[install_cmd.index("-r") + 1] == "pyproject.toml"
    assert install_cmd[install_cmd.index("--extra") + 1] == "paddleocr"


def test_patch_shared_environment_reinstalls_cpu_torch_with_cuda_backend(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=True, optional_deps={"penguin-vl-local": ["torch==2.8.0", "torchvision==0.23.0"]})
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
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
    assert len(commands) == 1

    install_cmd = commands[0]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert "--index-strategy" in install_cmd
    assert install_cmd[install_cmd.index("--index-strategy") + 1] == "unsafe-best-match"
    assert "--torch-backend" in install_cmd
    assert install_cmd[install_cmd.index("--torch-backend") + 1] == "cu128"
    assert install_cmd.count("--reinstall-package") == 2
    assert install_cmd[install_cmd.index("--reinstall-package") + 1] == "torch"
    second = install_cmd.index("--reinstall-package", install_cmd.index("--reinstall-package") + 1)
    assert install_cmd[second + 1] == "torchvision"
    assert install_cmd[install_cmd.index("-r") + 1] == "pyproject.toml"
    assert install_cmd[install_cmd.index("--extra") + 1] == "penguin-vl-local"

def test_patch_shared_environment_reads_requirements_directly_from_pyproject(tmp_path, monkeypatch):
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

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["music-flamingo-local"], []),
    )

    assert result is None
    assert len(commands) == 1

    install_cmd = commands[0]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert install_cmd[install_cmd.index("-r") + 1] == "pyproject.toml"
    assert install_cmd[install_cmd.index("--extra") + 1] == "music-flamingo-local"


def test_patch_shared_environment_overrides_wdtagger_opencv_on_windows(tmp_path, monkeypatch):
    _write_project(tmp_path, with_lock=False, optional_deps={"wdtagger": ["opencv-contrib-python", "torch==2.8.0"]})
    runner = ProcessRunner()
    commands: list[list[str]] = []

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)
    monkeypatch.setattr(runner, "_resolve_project_python", lambda work_dir, env: "python", raising=False)
    monkeypatch.setattr("gui.utils.process_runner.sys.platform", "win32")
    monkeypatch.setattr(
        "gui.utils.process_runner.resolve_wdtagger_windows_opencv_requirement",
        lambda env=None, platform=None: WdtaggerOpenCvSelection(
            package_spec="opencv-contrib-python @ https://example.invalid/opencv-cu129.whl",
            cuda_tag="cu129",
            source="cuda-wheel",
            detail="detected CUDA 12.9",
        ),
    )

    result = asyncio.run(
        runner._patch_shared_environment("uv", tmp_path, {}, "test-env", ["wdtagger"], []),
    )

    assert result is None
    assert len(commands) == 2

    install_cmd = commands[0]
    assert install_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert install_cmd[install_cmd.index("-r") + 1] == "pyproject.toml"
    assert install_cmd[install_cmd.index("--extra") + 1] == "wdtagger"

    opencv_cmd = commands[1]
    assert opencv_cmd[:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert "--python" in opencv_cmd
    assert opencv_cmd[opencv_cmd.index("--python") + 1] == "python"
    assert "--reinstall-package" in opencv_cmd
    assert opencv_cmd[opencv_cmd.index("--reinstall-package") + 1] == "opencv-contrib-python"
    assert opencv_cmd[-1] == "opencv-contrib-python @ https://example.invalid/opencv-cu129.whl"


def test_video_split_uses_video_split_extra_by_default(monkeypatch):
    runner = ProcessRunner()
    captured: dict[str, list[str]] = {}
    commands: list[list[str]] = []

    monkeypatch.setattr(runner, "_find_uv", staticmethod(lambda: "uv"))
    monkeypatch.setattr(runner, "_resolve_project_python", staticmethod(lambda work_dir, env: "python"))

    async def fake_patch(uv, work_dir, env, env_name, extras, groups):
        captured["extras"] = list(extras)
        captured["groups"] = list(groups)
        return None

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(runner, "_patch_shared_environment", fake_patch)
    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner.run_python_script("module.videospilter", ["./datasets"], native_console=False),
    )

    assert result.status.value == "成功"
    assert captured == {"extras": ["video-split"], "groups": []}
    assert commands == [["python", "./module/videospilter.py", "./datasets"]]


def test_audio_separator_uses_vocal_midi_extra_by_default(monkeypatch):
    runner = ProcessRunner()
    captured: dict[str, list[str]] = {}
    commands: list[list[str]] = []

    monkeypatch.setattr(runner, "_find_uv", staticmethod(lambda: "uv"))
    monkeypatch.setattr(runner, "_resolve_project_python", staticmethod(lambda work_dir, env: "python"))

    async def fake_patch(uv, work_dir, env, env_name, extras, groups):
        captured["extras"] = list(extras)
        captured["groups"] = list(groups)
        return None

    async def fake_run(cmd, work_dir, env):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(runner, "_patch_shared_environment", fake_patch)
    monkeypatch.setattr(runner, "_run_logged_subprocess", fake_run)

    result = asyncio.run(
        runner.run_python_script("module.audio_separator", ["./datasets"], native_console=False),
    )

    assert result.status.value == "成功"
    assert captured == {"extras": ["vocal-midi"], "groups": []}
    assert commands == [["python", "./module/audio_separator.py", "./datasets"]]
