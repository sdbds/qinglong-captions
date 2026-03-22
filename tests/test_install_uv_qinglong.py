from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_install_uv_qinglong_bootstraps_base_without_lock_generation():
    content = (ROOT / "1.install-uv-qinglong.ps1").read_text(encoding="utf-8")

    assert 'Write-Output "uv pip install dependency profile: base-only"' in content
    assert 'Write-Output "基础安装直接使用 uv pip install -r pyproject.toml，不启用任何 extra"' in content
    assert "Export-BaseRequirementsFromPyproject" not in content
    assert "Ensure-UvLockFile" not in content
    assert "uv lock --index-strategy $IndexStrategy" not in content
    assert "uv export --frozen" not in content
