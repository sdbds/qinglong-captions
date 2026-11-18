import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["config/model.toml", "config/config.toml"])
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
    assert "5.6.0" in requirements["logics-ocr"].specifier
    assert "4.57.1" not in requirements["logics-ocr"].specifier
    assert "5.5.4" in requirements["lighton-ocr"].specifier
    assert "5.0.0" not in requirements["lighton-ocr"].specifier
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
  $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-UnlimitedOcrExtra'
}, $true)
if (-not $fn) { throw 'Get-UnlimitedOcrExtra is missing' }
Invoke-Expression $fn.Extent.Text
Get-UnlimitedOcrExtra -ModelConfigPath $Config -PythonExe $Python
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
