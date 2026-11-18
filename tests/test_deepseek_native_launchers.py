import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("deepseek_ocr", "deepseek-ai/DeepSeek-OCR-2", "", "deepseek-ocr"),
    ("deepseek_ocr", "local/custom", "", "deepseek-ocr-legacy"),
    ("deepseek_ocr", "deepseek-ai/DeepSeek-OCR-2", "crop_mode = false", "deepseek-ocr-legacy"),
    ("deepseek_ocr", "deepseek-ai/DeepSeek-OCR-2", "image_size = 512", "deepseek-ocr-legacy"),
    ("deepseek_ocr_native", "local/custom", "", "deepseek-ocr-native"),
    ("deepseek_ocr_legacy", "deepseek-ai/DeepSeek-OCR-2", "", "deepseek-ocr-legacy"),
]


def _config(tmp_path, model, settings):
    path = tmp_path / "config" / "model.toml"
    path.parent.mkdir()
    path.write_text(f'[deepseek_ocr]\nmodel_id = "{model}"\n{settings}\n', encoding="utf-8")
    return path


@pytest.mark.parametrize("route,model,settings,expected", CASES)
def test_caption_gui_installs_the_selected_deepseek_stack(tmp_path, monkeypatch, route, model, settings, expected):
    from gui.utils import toml_helpers

    monkeypatch.setattr(toml_helpers, "_MODEL_CONFIG_PATH", _config(tmp_path, model, settings))
    monkeypatch.setattr(sys, "path", [*sys.path, str(ROOT / "gui")])
    from gui.wizard.step4_caption import CaptionStep

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value=route)
    step.vlm_image_model = step.alm_model = SimpleNamespace(value="")
    assert step._build_local_extra_args() == ["--extra", expected]


@pytest.mark.parametrize("route,model,settings,expected", CASES)
def test_powershell_dispatch_installs_the_selected_deepseek_stack(tmp_path, route, model, settings, expected):
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell is unavailable")
    _config(tmp_path, model, settings)
    # Execute the real OCR selection block and its helper, not a package install.
    runner = tmp_path / "launcher test.ps1"
    runner.write_text(
        """
param($Source, $ocr_model, $Python)
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw $errors[0] }
function Get-ProjectPython { return $Python }
function Add-UvExtra($Name) { Write-Output $Name }
$fn = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Get-DeepSeekOcrExtra' }, $true)
if ($fn) {
    # A file-backed definition retains $PSScriptRoot for the real default config lookup.
    $helper = Join-Path $PSScriptRoot 'selector.ps1'
    [IO.File]::WriteAllText($helper, $fn.Extent.Text)
    . $helper
}
$ext_args = [System.Collections.ArrayList]::new()
$dispatch = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.IfStatementAst] -and
    $node.Clauses[0].Item1.Extent.Text -eq '$ocr_model' }, $true)
if (-not $dispatch) { throw 'OCR selection block missing' }
Invoke-Expression $dispatch.Extent.Text
""",
        encoding="utf-8",
    )
    run = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(runner), str(ROOT / "4.captioner.ps1"), route, sys.executable],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == expected
