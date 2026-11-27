import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["config/model.toml"])
def test_shipped_configs_select_new_versions_without_forcing_legacy_protocols(filename):
    config = tomllib.loads((ROOT / filename).read_text(encoding="utf-8"))
    assert config["unlimited_ocr"]["model_id"] == "baidu/Unlimited-OCR-2"
    assert config["logics_ocr"]["model_id"] == "Logics-MLLM/Logics-Parsing-V3"
    assert config["lighton_ocr"]["model_id"] == "lightonai/LightOnOCR-3-1B"
    assert config["lighton_ocr"]["mode"] == "auto"
    assert config["logics_ocr"]["window_size"] == 2
    for name in ("unlimited_ocr", "logics_ocr", "lighton_ocr"):
        assert config[name]["model_version"] == "auto"


def test_catalog_preserves_legacy_choices_and_includes_all_lighton_sizes():
    config = tomllib.loads((ROOT / "config/model.toml").read_text(encoding="utf-8"))
    model_ids = {
        entry["model_id"]
        for name in ("unlimited_ocr", "logics_ocr", "lighton_ocr")
        for entry in config[name]["model_list"].values()
    }
    assert {
        "baidu/Unlimited-OCR",
        "baidu/Unlimited-OCR-2",
        "Logics-MLLM/Logics-Parsing-v2",
        "Logics-MLLM/Logics-Parsing-V3",
        "lightonai/LightOnOCR-2-1B",
        "lightonai/LightOnOCR-3-0.8B",
        "lightonai/LightOnOCR-3-1B",
        "lightonai/LightOnOCR-3-4B",
    } <= model_ids


def test_dependency_profiles_isolate_unlimited_versions_and_include_qwen35():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["optional-dependencies"]
    requirements = {
        name: next(Requirement(dep) for dep in deps[name] if Requirement(dep).name == "transformers")
        for name in ("unlimited-ocr", "unlimited-ocr2", "logics-ocr", "lighton-ocr")
    }
    assert "4.57.1" in requirements["unlimited-ocr"].specifier
    assert "4.52.1" in requirements["unlimited-ocr2"].specifier
    for name in ("logics-ocr", "lighton-ocr"):
        assert "5.10.1" in requirements[name].specifier
        assert all(version not in requirements[name].specifier for version in ("4.57.1", "5.6.0", "6.0.0"))
    conflicts = [{item.get("extra") for item in group} for group in project["tool"]["uv"]["conflicts"]]
    for other in ("unlimited-ocr", "logics-ocr", "lighton-ocr", "translate", "paddleocr-native"):
        assert any({"unlimited-ocr2", other} <= group for group in conflicts)


def test_unlimited_ocr2_is_isolated_from_incompatible_test_transformers():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(
        {"extra": "unlimited-ocr2"} in group and {"group": "test"} in group
        for group in project["tool"]["uv"]["conflicts"]
    )


@pytest.mark.parametrize(
    "model_id,version,expected",
    [
        ("baidu/Unlimited-OCR", "auto", "unlimited-ocr"),
        ("baidu/Unlimited-OCR-2", "auto", "unlimited-ocr2"),
        ("local/renamed", "2", "unlimited-ocr2"),
    ],
)
def test_launcher_extra_selector_reads_toml_and_version_override(tmp_path, model_id, version, expected):
    from utils.ocr_model_config import unlimited_ocr_extra_from_config

    config = tmp_path / "models.toml"
    config.write_text(f'[unlimited_ocr]\nmodel_id = "{model_id}"\nmodel_version = "{version}"\n')
    assert unlimited_ocr_extra_from_config(config) == expected


def test_powershell_uses_structured_extra_selector_without_running_captioner(tmp_path):
    import shutil

    powershell = shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is unavailable")
    config = tmp_path / "model with spaces.toml"
    config.write_text('[unlimited_ocr]\nmodel_id = "custom/model"\nmodel_version = "2"\n')
    script = """
param($Root, $Config, $Python)
$tokens = $null
$errors = $null
$source = Join-Path $Root '4.captioner.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw $errors[0] }
$fn = $ast.Find({param($node)
  $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-CaptionRuntimeExtras'
}, $true)
if (-not $fn) { throw 'Get-CaptionRuntimeExtras is missing' }
Invoke-Expression $fn.Extent.Text
$ocr_model = "unlimited_ocr"
Get-CaptionRuntimeExtras -ModelConfigPath $Config -PythonExe $Python
"""
    # PowerShell passes remaining -Command arguments as code; a temporary script
    # keeps paths as data, including spaces/non-ASCII, without running the launcher.
    runner = tmp_path / "select-extra.ps1"
    runner.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [powershell, "-NoProfile", "-File", str(runner), str(ROOT), str(config), sys.executable],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unlimited-ocr2"


def test_powershell_get_project_python_prefers_dot_venv_over_venv_and_path(tmp_path):
    import shutil

    powershell = shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is unavailable")
    dot_venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_py = tmp_path / "venv" / "Scripts" / "python.exe"
    dot_venv_py.parent.mkdir(parents=True)
    venv_py.parent.mkdir(parents=True)
    dot_venv_py.write_bytes(b"")
    venv_py.write_bytes(b"")

    runner = tmp_path / "check-project-python.ps1"
    runner.write_text(
        """
param($Source, $WorkDir)
$ErrorActionPreference = 'Stop'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$null, [ref]$null)
$fn = $ast.Find({param($n)
  $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-ProjectPython'
}, $true)
if (-not $fn) { throw 'Get-ProjectPython is missing' }
Invoke-Expression $fn.Extent.Text
Set-Location $WorkDir
$first = Get-ProjectPython
Remove-Item -LiteralPath (Join-Path $WorkDir '.venv/Scripts/python.exe')
$second = Get-ProjectPython
Write-Output $first
Write-Output $second
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-File", str(runner), str(ROOT / "4.captioner.ps1"), str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert [Path(line).resolve() for line in lines] == [dot_venv_py.resolve(), venv_py.resolve()]


def test_powershell_extra_selector_falls_back_to_uv_isolated_python_on_cold_start(tmp_path):
    import shutil

    powershell = shutil.which("pwsh")
    if not powershell or not shutil.which("uv"):
        pytest.skip("PowerShell or uv is unavailable")
    config = tmp_path / "model.toml"
    config.write_text(
        '[unlimited_ocr]\nmodel_id = "baidu/Unlimited-OCR-2"\n'
        '[moondream]\nmodel_id = "moondream/moondream3.1-9B-A2B"\n',
        encoding="utf-8",
    )
    runner = tmp_path / "cold-start-extras.ps1"
    runner.write_text(
        """
param($Root, $ConfigDir, $ModelConfig)
$ErrorActionPreference = 'Stop'
$source = Join-Path $Root '4.captioner.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($source, [ref]$null, [ref]$null)
$fn = $ast.Find({param($n)
  $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-CaptionRuntimeExtras'
}, $true)
if (-not $fn) { throw 'Get-CaptionRuntimeExtras is missing' }
Invoke-Expression $fn.Extent.Text
$ocr_model = "unlimited_ocr"
$vlm_image_model = "moondream"
$alm_model = "gemma4_local"
Get-CaptionRuntimeExtras -ConfigDir $ConfigDir -ModelConfigPath $ModelConfig -PythonExe ""
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-File", str(runner), str(ROOT), str(tmp_path), str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert [line.strip() for line in result.stdout.splitlines() if line.strip()] == [
        "unlimited-ocr2",
        "moondream",
        "gemma4-local",
    ]
