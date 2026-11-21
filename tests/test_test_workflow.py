import os
import shlex
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version


pytestmark = pytest.mark.compat


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
SETUP_SCRIPT = ROOT / ".github" / "scripts" / "setup_ci_test_env.ps1"


def _workflow_and_setup() -> tuple[str, str]:
    return (
        WORKFLOW.read_text(encoding="utf-8"),
        SETUP_SCRIPT.read_text(encoding="utf-8"),
    )


def test_test_workflow_renders_minimal_ci_project_before_exporting_test_requirements():
    workflow, setup = _workflow_and_setup()

    assert ".github/scripts/setup_ci_test_env.ps1" in workflow
    render_index = setup.index("python .github/scripts/render_ci_test_project.py --source pyproject.toml --output $CiPyproject")
    export_index = setup.index(
        "uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy "
        "--group test --no-emit-project --format requirements-txt --output-file $RequirementsFile"
    )

    assert render_index < export_index


def test_test_workflow_uses_cpu_pytorch_index_and_uv_cache():
    content = WORKFLOW.read_text(encoding="utf-8")

    expected_lines = [
        "HF_HOME: huggingface",
        'XFORMERS_FORCE_DISABLE_TRITON: "1"',
        "UV_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu",
        'UV_NO_CACHE: "0"',
        "UV_LINK_MODE: symlink",
        "UV_INDEX_STRATEGY: unsafe-best-match",
        "PYTHONIOENCODING: utf-8",
        'PYTHONUTF8: "1"',
        "PYTHONPATH: ${{ github.workspace }}",
        "uses: astral-sh/setup-uv@v5",
        "enable-cache: true",
    ]

    for expected in expected_lines:
        assert expected in content

    assert "download.pytorch.org/whl/cu130" not in content
    assert "HF_ENDPOINT: https://hf-mirror.com" not in content


def test_test_workflow_fails_fast_when_ci_test_dependency_export_fails():
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert 'throw "render ci test project failed"' in setup
    assert 'throw "uv export failed"' in setup
    assert 'throw "uv pip install failed"' in setup


def test_cold_ci_export_builds_source_metadata_in_isolation(tmp_path):
    # Source-only dependency with a local PEP 517 backend; never fetch packages.
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n',
        encoding="utf-8",
    )
    (source / "backend.py").write_text(
        "from pathlib import Path\nimport os, sys\n"
        "def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):\n"
        "    if Path(sys.prefix).resolve() == Path(os.environ['CI_RUNNER_PREFIX']).resolve():\n"
        "        raise RuntimeError('build ran in the bare matrix interpreter')\n"
        "    info = Path(metadata_directory) / 'source_only-1.dist-info'\n"
        "    info.mkdir()\n"
        "    (info / 'METADATA').write_text('Metadata-Version: 2.1\\nName: source-only\\nVersion: 1\\n')\n"
        "    return info.name\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ci-probe"\nversion = "1"\nrequires-python = ">=3.11"\n'
        f'[dependency-groups]\ntest = ["source-only @ {source.as_uri()}"]\n',
        encoding="utf-8",
    )
    runner = tmp_path / "runner"
    venv.create(runner, with_pip=False)
    python = runner / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    output = tmp_path / "requirements.txt"
    variables = {
        "$UvExportProjectDir": str(tmp_path), "$RunnerPython": str(python),
        "$IndexStrategy": "unsafe-best-match", "$RequirementsFile": str(output),
    }
    workflow, setup = _workflow_and_setup()
    command = next(line for line in setup.splitlines() if line.startswith("uv export "))
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_") and k not in {"PYTHONPATH", "VIRTUAL_ENV"}}
    env.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_CACHE_DIR=str(tmp_path / "empty-cache"), CI_RUNNER_PREFIX=str(runner))
    build_isolation = yaml.safe_load(workflow)["env"].get("UV_NO_BUILD_ISOLATION")
    if build_isolation is not None:
        env["UV_NO_BUILD_ISOLATION"] = str(build_isolation)
    result = subprocess.run(
        [variables.get(token, token) for token in shlex.split(command)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert any(line and not line.startswith("#") for line in output.read_text(encoding="utf-8").splitlines())


def test_test_workflow_preinstalls_build_bootstrap_packages_before_ci_test_export():
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")

    bootstrap_index = setup.index("uv pip install --python $PythonExe setuptools wheel wheel_stub")
    export_index = setup.index(
        "uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy "
        "--group test --no-emit-project --format requirements-txt --output-file $RequirementsFile"
    )

    assert bootstrap_index < export_index


def test_test_workflow_exports_test_requirements_with_the_matrix_python_not_a_hardcoded_version():
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert (
        "uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy "
        "--group test --no-emit-project --format requirements-txt --output-file $RequirementsFile"
    ) in setup
    assert "uv export --project $UvExportProjectDir --python 3.11 --index-strategy" not in setup


def test_test_workflow_bootstraps_project_virtualenv_before_ci_test_install():
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")

    venv_index = setup.index("uv venv")
    project_python_index = setup.index("$PythonExe = if ($IsWindows)")
    bootstrap_index = setup.index("uv pip install --python $PythonExe setuptools wheel wheel_stub")
    export_index = setup.index(
        "uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy "
        "--group test --no-emit-project --format requirements-txt --output-file $RequirementsFile"
    )
    install_index = setup.index("uv pip install --python $PythonExe --index-strategy $IndexStrategy -r $RequirementsFile")

    assert venv_index < project_python_index < bootstrap_index < export_index < install_index
    assert "./.venv/Scripts/python.exe" in setup
    assert "./.venv/bin/python" in setup
    assert '"VENV_PYTHON=$PythonExe" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append' in setup


def test_test_workflow_runs_suites_with_bootstrapped_venv_python():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "& $env:VENV_PYTHON -m pytest" in content
    assert '& $env:VENV_PYTHON -c "from module.providers.registry import get_registry; reg = get_registry(); reg.discover(strict=True)"' in content
    assert "uv run --group test" not in content


def test_test_workflow_runs_strict_provider_discovery_before_provider_v2_suite():
    content = WORKFLOW.read_text(encoding="utf-8")

    strict_discovery_index = content.index("reg.discover(strict=True)")
    provider_suite_index = content.index(
        "& $env:VENV_PYTHON -m pytest tests/test_provider_v2.py tests/test_provider_registry.py "
        "tests/test_provider_routes.py tests/test_api_handler_v2.py -q --durations=20"
    )

    assert strict_discovery_index < provider_suite_index


def test_test_workflow_strict_provider_discovery_bootstraps_module_import_root():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "from module.providers.registry import get_registry" in content
    assert "sys.path.insert(0, str(ROOT / 'module'))" not in content


def test_test_workflow_runs_authoritative_offline_unit_suite_on_python_311():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert 'python-version: ["3.11"]' in content
    assert (
        '& $env:VENV_PYTHON -m pytest tests -q --strict-markers '
        '-m "not optional_runtime and not gpu and not network" --durations=25'
    ) in content


def test_test_workflow_runs_marker_based_compatibility_matrix():
    job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["compatibility"]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    versions = sorted(Version(v) for v in job["strategy"]["matrix"]["python-version"])
    assert versions, "The compatibility job must test at least one Python minor version"
    first, last = versions[0], versions[-1]
    # The project supports a contiguous range of Python minor releases.
    assert versions == [Version(f"{first.major}.{minor}") for minor in range(first.minor, last.minor + 1)]
    covered = SpecifierSet(f">={first},<{last.major}.{last.minor + 1}")
    assert covered == SpecifierSet(project["requires-python"])

    command = "& $env:VENV_PYTHON -m pytest tests -q --strict-markers -m compat"
    assert any(command in step.get("run", "").splitlines() for step in job["steps"])


def test_test_workflow_has_lightweight_static_correctness_gate():
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = parsed["jobs"]["static-correctness"]
    steps_run = "\n".join(step.get("run", "") for step in job["steps"])

    assert ".github/scripts/setup_ci_test_env.ps1" not in steps_run
    assert "['lint']" in steps_run
    assert "-m compileall -q module gui utils config tests tools .github/scripts" in steps_run
    assert "-m ruff check module gui utils config tests tools .github/scripts --select F821,F823" in steps_run
    assert (
        "-m pytest tests/test_test_workflow.py "
        "tests/test_test_isolation_contract.py::test_ordinary_tests_do_not_mutate_import_state_at_module_scope "
        "-q --strict-markers"
    ) in steps_run


def test_static_pytest_command_collects_without_runtime_dependencies():
    job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["static-correctness"]
    command = next(step["run"] for step in job["steps"] if "-m pytest " in step.get("run", ""))
    args = shlex.split(command.split("-m pytest ", 1)[1])
    # A full development venv must not mask imports missing from the lint job.
    probe = (
        "import sys; sys.modules['torch'] = sys.modules['rich'] = None; "
        "import pytest; sys.exit(pytest.main(['--collect-only', *sys.argv[1:]]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, *args], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_test_workflow_covers_all_runtime_and_ci_trigger_roots():
    content = WORKFLOW.read_text(encoding="utf-8")
    trigger_paths = [
        '"module/**"',
        '"gui/**"',
        '"utils/**"',
        '"config/**"',
        '"tests/**"',
        '"tools/**"',
        '"third_party/**"',
        '"*.ps1"',
        '"*.sh"',
        '".github/scripts/**"',
        '".github/workflows/test.yml"',
        '"pyproject.toml"',
    ]

    for path in trigger_paths:
        assert content.count(path) >= 2, f"missing push/pull_request trigger for {path}"
