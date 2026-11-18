import asyncio
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _python(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def test_named_runtimes_are_created_once_per_profile_and_never_in_project_venv(tmp_path, monkeypatch):
    from utils import deepseek_runtime as runtime

    calls = []
    read_bytes = Path.read_bytes

    def read_without_lock_byte(path):
        if path.name.endswith(".create.lock"):
            raise PermissionError("Windows may deny reads on a byte held by another creator")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_without_lock_byte)

    def create(cmd, **kwargs):
        destination = Path(cmd[-1])
        assert cmd[0:2] == ["uv", "venv"] and "--clear" not in cmd
        assert "--seed" in cmd and cmd[cmd.index("--python") + 1] == "3.11"
        calls.append(destination)
        python = _python(destination)
        python.parent.mkdir(parents=True)
        python.touch()
        (destination / "pyvenv.cfg").write_text("version_info = 3.11\ninclude-system-site-packages = false\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runtime.subprocess, "run", create)
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(
            pool.map(
                lambda extra: runtime.ensure_runtime(tmp_path, extra, uv="uv"),
                ["deepseek-ocr", "deepseek-ocr-native", "deepseek-ocr-legacy"],
            )
        )
    assert paths[0] == paths[1] == _python(tmp_path / ".runtime_venvs" / "deepseek-ocr-native")
    assert paths[2] == _python(tmp_path / ".runtime_venvs" / "deepseek-ocr-legacy")
    assert len(calls) == 2 and not (tmp_path / ".venv").exists()


def test_existing_nonempty_lock_is_not_used_or_truncated(tmp_path, monkeypatch):
    from utils import deepseek_runtime as runtime

    state = tmp_path / ".runtime_venvs"
    state.mkdir()
    lock = state / ".deepseek-ocr-native.create.lock"
    lock.write_bytes(b"private file")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_a, **_kw: pytest.fail("unexpected provisioning"))
    with pytest.raises(OSError, match="Unowned"):
        runtime.ensure_runtime(tmp_path, "deepseek-ocr")
    assert lock.read_bytes() == b"private file"


def test_existing_unowned_runtime_is_never_passed_to_uv(tmp_path, monkeypatch):
    from utils import deepseek_runtime as runtime

    unowned = tmp_path / ".runtime_venvs" / "deepseek-ocr-legacy"
    unowned.mkdir(parents=True)
    note = unowned / "important.txt"
    note.write_text("mine")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: pytest.fail("unowned runtime was reused"))
    with pytest.raises((OSError, ValueError), match="unowned|not owned"):
        runtime.ensure_runtime(tmp_path, "deepseek-ocr-legacy", uv="uv")
    assert list(unowned.iterdir()) == [note] and note.read_text() == "mine"


@pytest.mark.parametrize(
    "extra, supplied, expected",
    [
        ("deepseek-ocr", None, ".runtime_venvs/deepseek-ocr-native"),
        ("deepseek-ocr-native", ".venv", ".runtime_venvs/deepseek-ocr-native"),
        ("deepseek-ocr-legacy", ".venv", ".runtime_venvs/deepseek-ocr-legacy"),
        ("deepseek-ocr", "custom-tab", "custom-tab"),
        ("moondream", ".venv", ".venv"),
    ],
)
def test_gui_install_and_caption_use_same_selected_environment(tmp_path, monkeypatch, extra, supplied, expected):
    from gui.utils.process_runner import ProcessRunner, ProcessStatus

    runner = ProcessRunner()
    target = tmp_path / expected
    selected = tmp_path / supplied if supplied else None
    for folder in (target, tmp_path / ".venv"):
        python = _python(folder)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
    installs, captions, provisions = [], [], []

    async def patch(uv, work_dir, env, env_name, extras, groups, target_python, environment_key):
        installs.append(target_python)

    async def run(cmd, work_dir, env):
        if "utils.deepseek_runtime" in cmd:
            provisions.append(cmd)
        else:
            captions.append((cmd[0], env.get("VIRTUAL_ENV")))
        return 0

    monkeypatch.setattr(runner, "_find_uv", lambda: "uv")
    monkeypatch.setattr(runner, "_patch_shared_environment", patch)
    monkeypatch.setattr(runner, "_run_logged_subprocess", run)
    result = asyncio.run(
        runner.run_python_script(
            "module.captioner",
            ["sample"],
            cwd=str(tmp_path),
            uv_extra=extra,
            python_path=str(_python(selected)) if selected else None,
            venv_path=str(selected) if selected else None,
            native_console=False,
        )
    )
    assert result.status == ProcessStatus.SUCCESS
    assert installs == [str(_python(target))] and captions == [(str(_python(target)), str(target))]
    assert len(provisions) == int(expected.startswith(".runtime_venvs/"))


@pytest.mark.parametrize(
    "caption_exit, fail_install, caller_has_venv",
    [(0, False, True), (3, False, False), (0, True, True)],
    ids=["success", "partial", "install-error"],
)
def test_powershell_isolates_install_and_caption_from_the_callers_environment(
    tmp_path, caption_exit, fail_install, caller_has_venv,
):
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell unavailable")
    python = _python(tmp_path / "user-test-venv")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(python.parent.parent)], check=True)
    entry = tmp_path / "module/captioner.py"
    entry.parent.mkdir()
    entry.write_text(
        "import os, sys\nprint('caption=' + sys.executable)\nprint('venv=' + os.environ['VIRTUAL_ENV'])\n"
        "sys.exit(int(sys.argv[1]))\n",
        encoding="utf-8",
    )
    runner = tmp_path / "invoke.ps1"
    runner.write_text(
        """
param($Source, $Python, $Root, [int]$CaptionExit, [int]$FailInstall, [int]$CallerHasVenv)
$ErrorActionPreference = 'Stop'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$null, [ref]$null)
foreach ($name in @('Invoke-CaptionWithRuntime','Install-UvDependencyPatch')) {
    $fn = $ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -eq $name}, $true)
    if (-not $fn) { throw "Caption launcher function missing: $name" }
    Invoke-Expression $fn.Extent.Text
}
function Get-DeepSeekRuntimePython { return $Python }
function Get-UvEnvName { return 'project' }
function Get-UvProfile { return 'test' }
function Get-ProjectPython { throw 'install redirected to the shared interpreter' }
function uv {
    $i = [array]::IndexOf($args, '--python')
    if ($i -lt 0 -or $args -notcontains 'deepseek-ocr-legacy') { throw 'Wrong uv install command' }
    $scriptsDir = Split-Path -Parent $Python
    if ($env:VIRTUAL_ENV -ne (Split-Path -Parent $scriptsDir) -or
        ($env:PATH -split [IO.Path]::PathSeparator)[0] -ne $scriptsDir) { throw 'Wrong install environment' }
    if ($FailInstall) { throw 'Test install failure' }
    Write-Output ("install=" + $args[$i+1])
}
Set-Location $Root
$originalPath = $env:PATH
if ($CallerHasVenv) { $env:VIRTUAL_ENV = Join-Path $Root 'caller' } else { $env:VIRTUAL_ENV = $null }
$originalVenv = $env:VIRTUAL_ENV
try {
    Invoke-CaptionWithRuntime -ArgsList @('--extra=deepseek-ocr-legacy') -DatasetPath "$CaptionExit" -CaptionArgs @()
    $code = $LASTEXITCODE
    if ($FailInstall) { throw 'Lost install error' }
} catch {
    if (-not $FailInstall -or $_.Exception.Message -ne 'Test install failure') { throw }
    $code = 17
}
if ($env:VIRTUAL_ENV -cne $originalVenv -or $env:PATH -cne $originalPath) { throw 'DeepSeek environment leaked' }
exit $code
""",
        encoding="utf-8",
    )
    run = subprocess.run(
        [
            pwsh, "-NoProfile", "-File", str(runner), str(ROOT / "4.captioner.ps1"), str(python), str(tmp_path),
            str(caption_exit), str(int(fail_install)), str(int(caller_has_venv)),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert run.returncode == (17 if fail_install else caption_exit), run.stderr
    if fail_install:
        assert "caption=" not in run.stdout
    else:
        assert f"install={python}" in run.stdout and f"caption={python}" in run.stdout
        assert f"venv={python.parent.parent}" in run.stdout
