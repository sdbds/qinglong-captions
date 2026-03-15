from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"


def test_test_workflow_bootstraps_uv_lock_before_frozen_sync():
    content = WORKFLOW.read_text(encoding="utf-8")

    lock_index = content.index("uv lock --index-strategy")
    sync_index = content.index("uv sync --frozen --group test")

    assert lock_index < sync_index


def test_test_workflow_uses_script_aligned_env_vars():
    content = WORKFLOW.read_text(encoding="utf-8")

    expected_lines = [
        "HF_HOME: huggingface",
        "HF_ENDPOINT: https://hf-mirror.com",
        'XFORMERS_FORCE_DISABLE_TRITON: "1"',
        "UV_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cu128",
        'UV_NO_BUILD_ISOLATION: "1"',
        'UV_NO_CACHE: "0"',
        "UV_LINK_MODE: symlink",
        "UV_INDEX_STRATEGY: unsafe-best-match",
        "PYTHONIOENCODING: utf-8",
        'PYTHONUTF8: "1"',
        "PYTHONPATH: ${{ github.workspace }}",
    ]

    for expected in expected_lines:
        assert expected in content
