import subprocess
import sys
from pathlib import Path

import module.auto_rig as auto_rig

ROOT = Path(__file__).resolve().parent.parent


def test_auto_rig_public_api_exports_resolve():
    assert auto_rig.__all__
    missing = [name for name in auto_rig.__all__ if not hasattr(auto_rig, name)]
    assert missing == []
    assert len(auto_rig.__all__) == len(set(auto_rig.__all__))


def test_auto_rig_imports_in_fresh_python_process():
    result = subprocess.run(
        [sys.executable, "-c", "import module.auto_rig; assert module.auto_rig.__all__"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
