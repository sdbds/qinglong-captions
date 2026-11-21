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


def test_offline_runtime_can_write_numpy_values_to_lance(tmp_path):
    import lance
    import numpy as np
    import pyarrow as pa

    values = np.array([1, 2], dtype=np.int32)
    dataset = lance.write_dataset(pa.table({"value": values}), tmp_path / "smoke.lance")
    assert dataset.to_table().to_pydict() == {"value": [1, 2]}


def test_render_ci_test_project_includes_offline_geometry_without_gpu_extras(tmp_path):
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
    geometry = source["project"]["optional-dependencies"]["auto-rig"]
    assert rendered["dependency-groups"] == {"test": [*source["dependency-groups"]["test"], *geometry]}
    assert "optional-dependencies" not in rendered["project"]
    assert rendered["tool"]["uv"] == {"package": False}
