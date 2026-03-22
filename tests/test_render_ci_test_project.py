import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".github" / "scripts" / "render_ci_test_project.py"
PYPROJECT = ROOT / "pyproject.toml"


def test_render_ci_test_project_preserves_base_and_test_dependencies_only(tmp_path):
    output = tmp_path / "pyproject.toml"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(PYPROJECT), "--output", str(output)],
        check=True,
    )

    source = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    rendered = tomllib.loads(output.read_text(encoding="utf-8"))

    assert rendered["project"]["name"] == "qinglong-captions-ci-test"
    assert rendered["project"]["version"] == source["project"]["version"]
    assert rendered["project"]["requires-python"] == source["project"]["requires-python"]
    assert rendered["project"]["dependencies"] == source["project"]["dependencies"]
    assert rendered["dependency-groups"] == {"test": source["dependency-groups"]["test"]}
    assert "optional-dependencies" not in rendered["project"]
    assert rendered["tool"]["uv"] == {"package": False}
