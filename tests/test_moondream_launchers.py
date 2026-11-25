import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = "moondream/moondream3-preview"
PHOTON = "moondream/moondream3.1-9B-A2B"


@pytest.mark.parametrize(
    "model,tasks,ocr,url,expected",
    [
        (PREVIEW, "caption", False, "", "moondream-legacy"),
        (PHOTON, "caption", False, "", "moondream"),
        (PHOTON, "query", False, "http://localhost:8000", None),
        (PHOTON, "all", True, "http://localhost:8000", None),
        (PHOTON, "query,detect", False, "http://localhost:8000", "moondream"),
    ],
)
@pytest.mark.parametrize("filename", ["model.toml", "config.toml"])
def test_gui_keeps_other_routes_on_their_own_stack(tmp_path, monkeypatch, model, tasks, ocr, url, expected, filename):
    monkeypatch.syspath_prepend(str(ROOT / "gui"))
    from config import loader
    from gui.utils import toml_helpers
    from gui.wizard import step4_caption as caption

    (tmp_path / filename).write_text(f'[moondream]\nmodel_id = "{model}"\ntasks = "{tasks}"\n', encoding="utf-8")
    monkeypatch.setattr(toml_helpers, "_MODEL_CONFIG_PATH", tmp_path / "model.toml")
    load = loader.load_config
    monkeypatch.setattr(loader, "load_config", lambda _: load(str(tmp_path)))
    step = caption.CaptionStep()
    step.ocr_model = SimpleNamespace(value="moondream" if ocr else "")
    step.vlm_image_model = SimpleNamespace(value="" if ocr else "moondream")
    step.openai_base_url = SimpleNamespace(value=url)
    step.alm_model = SimpleNamespace(value="cohere_transcribe_local" if model == PREVIEW else "")
    wanted = ["--extra", expected] if expected else []
    if model == PREVIEW:
        wanted += ["--extra", "cohere-transcribe-local"]
    assert step._build_local_extra_args() == wanted


@pytest.mark.parametrize(
    "route,model,backend,url,expected",
    [
        ("VLM", PREVIEW, "", False, "moondream-legacy"),
        ("VLM", PHOTON, "direct", True, "moondream"),
        ("OCR", PHOTON, "", True, ""),
    ],
)
@pytest.mark.parametrize("shell", ["pwsh", "powershell.exe"])
def test_powershell_selects_actual_moondream_dependency(tmp_path, route, model, backend, url, expected, shell):
    pwsh = shutil.which(shell)
    if not pwsh:
        pytest.skip(f"{shell} unavailable")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "model.toml").write_text(f'[moondream]\nmodel_id = "{model}"\ntasks = "all"\n', encoding="utf-8")
    runner = tmp_path / "run.ps1"
    runner.write_text(
        r"""
param($Source, $Python, $Route, $Backend, [int]$HasUrl)
$ErrorActionPreference = 'Stop'
$local_runtime_backend = $Backend.Substring(5)
$openai_base_url = if ($HasUrl) { 'http://localhost:8000' } else { '' }
$ocr_model = if ($Route -eq 'OCR') { 'moondream' } else { '' }
$vlm_image_model = if ($Route -eq 'VLM') { 'moondream' } else { '' }
$uv_args = [Collections.ArrayList]::new()
$ext_args = [Collections.ArrayList]::new()
function Get-ProjectPython { return $Python }
$ast = [Management.Automation.Language.Parser]::ParseFile($Source, [ref]$null, [ref]$null)
foreach ($name in @('Add-UvExtra', 'Get-MoondreamExtra')) {
    $fn = $ast.Find({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name }, $true)
    if ($fn) {
        $helper = Join-Path $PSScriptRoot "$name.ps1"
        [IO.File]::WriteAllText($helper, $fn.Extent.Text)
        . $helper
    }
}
foreach ($block in $ast.EndBlock.Statements) {
    if ($block.Extent.Text -match '^if \(\$(ocr_model|vlm_image_model)\) \{') { Invoke-Expression $block.Extent.Text }
}
$uv_args
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-File",
            str(runner),
            str(ROOT / "4.captioner.ps1"),
            sys.executable,
            route,
            f"mode:{backend}",
            str(int(url)),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (f"--extra={expected}" if expected else "")
