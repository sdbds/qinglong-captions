from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_image2psd_wrapper_uses_see_through_cli_profile():
    content = (ROOT / "2.6.image2psd.ps1").read_text(encoding="utf-8")

    assert 'Install-UvExtraPatch @("see-through")' in content
    assert 'Write-Output "runtime dependency profile: extra:see-through"' in content
    assert "python -m module.see_through.cli" in content
    assert "function Add-BooleanOptionalArg" in content
    assert "--inference_steps_depth=" in content
    assert "--seed=" in content
    assert "--input_dir=" in content
    assert "--output_dir=" in content
    assert "workspace/image2psd_output" in content
