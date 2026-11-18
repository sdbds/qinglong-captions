from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gui.wizard import step5_export


def test_export_wizard_can_browse_lance_dataset_directories():
    step = step5_export.ExportStep()
    step.render()

    assert step.lance_file.selection_type in {"dir", "file_or_dir"}
    assert Path(step.output_dir.value) == Path("exports")


@pytest.mark.parametrize(("selection", "destination"), [("", "./exports"), ("custom path", "custom path")])
def test_export_wizard_uses_selected_destination_or_dedicated_default(tmp_path, selection, destination):
    lance_file = tmp_path / "dataset.lance"
    lance_file.mkdir()
    calls = []

    async def run_job(module, args, **kwargs):
        calls.append((module, args))

    step = step5_export.ExportStep()
    step.lance_file = SimpleNamespace(value=str(lance_file))
    step.output_dir = SimpleNamespace(value=selection)
    step.version = SimpleNamespace(value="gemini")
    step.panel = SimpleNamespace(run_job=run_job)

    asyncio.run(step._start_export())

    assert calls == [("module.lanceexport", [str(lance_file), f"--output_dir={destination}", "--version=gemini"])]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_public_export_wrapper_preserves_exit_code_without_prompt(tmp_path, exit_code):
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell is not installed")
    wrapper = tmp_path / "lanceExport.ps1"
    shutil.copyfile(Path(__file__).resolve().parents[1] / wrapper.name, wrapper)
    activate = tmp_path / ".venv/bin/activate.ps1"
    activate.parent.mkdir(parents=True)
    activate.touch()

    script = """
    function global:python {
        Write-Output ($args -join '|')
        $global:LASTEXITCODE = [int]$env:EXPORT_TEST_CODE
    }
    function global:Read-Host { throw 'Unexpected interactive prompt' }
    & $env:EXPORT_TEST_WRAPPER
    exit $LASTEXITCODE
    """
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, "EXPORT_TEST_WRAPPER": str(wrapper), "EXPORT_TEST_CODE": str(exit_code)},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == exit_code, result.stderr
    assert "module.lanceexport|./datasets/dataset.lance|--output_dir=./exports" in result.stdout
    assert ("Export finished" in result.stdout) == (exit_code == 0)
