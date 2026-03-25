from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"


def test_test_workflow_renders_minimal_ci_project_before_exporting_test_requirements():
    content = WORKFLOW.read_text(encoding="utf-8")

    render_index = content.index("python .github/scripts/render_ci_test_project.py --source pyproject.toml --output $CiPyproject")
    export_index = content.index("uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy --group test --no-emit-project --format requirements-txt --output-file $RequirementsFile")

    assert render_index < export_index


def test_test_workflow_uses_script_aligned_env_vars():
    content = WORKFLOW.read_text(encoding="utf-8")

    expected_lines = [
        "HF_HOME: huggingface",
        "HF_ENDPOINT: https://hf-mirror.com",
        'XFORMERS_FORCE_DISABLE_TRITON: "1"',
        (
            "UV_EXTRA_INDEX_URL: "
            "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/ "
            "https://download.pytorch.org/whl/cu130"
        ),
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


def test_test_workflow_fails_fast_when_ci_test_dependency_export_fails():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert 'throw "render ci test project failed"' in content
    assert 'throw "uv export failed"' in content
    assert 'throw "uv pip install failed"' in content


def test_test_workflow_preinstalls_build_bootstrap_packages_before_ci_test_export():
    content = WORKFLOW.read_text(encoding="utf-8")

    install_index = content.index("python -m pip install --upgrade pip uv wheel_stub setuptools wheel tomli")
    export_index = content.index("uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy --group test --no-emit-project --format requirements-txt --output-file $RequirementsFile")

    assert install_index < export_index


def test_test_workflow_exports_test_requirements_with_the_matrix_python_not_a_hardcoded_version():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy --group test --no-emit-project --format requirements-txt --output-file $RequirementsFile" in content
    assert "uv export --project $UvExportProjectDir --python 3.11 --index-strategy" not in content


def test_test_workflow_bootstraps_project_virtualenv_before_ci_test_install():
    content = WORKFLOW.read_text(encoding="utf-8")

    venv_index = content.index("uv venv")
    project_python_index = content.index("$PythonExe = if ($IsWindows)")
    bootstrap_index = content.index("uv pip install --python $PythonExe setuptools wheel wheel_stub")
    export_index = content.index("uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy --group test --no-emit-project --format requirements-txt --output-file $RequirementsFile")
    install_index = content.index("uv pip install --python $PythonExe --index-strategy $IndexStrategy -r $RequirementsFile")

    assert venv_index < project_python_index < bootstrap_index < export_index < install_index
    assert "./.venv/Scripts/python.exe" in content
    assert "./.venv/bin/python" in content
    assert '"VENV_PYTHON=$PythonExe" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append' in content


def test_test_workflow_runs_suites_with_bootstrapped_venv_python():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "& $env:VENV_PYTHON -m pytest" in content
    assert '& $env:VENV_PYTHON -c "from providers.registry import get_registry; reg = get_registry(); reg.discover(strict=True)"' in content
    assert "uv run --group test" not in content


def test_test_workflow_runs_strict_provider_discovery_before_provider_v2_suite():
    content = WORKFLOW.read_text(encoding="utf-8")

    strict_discovery_index = content.index("reg.discover(strict=True)")
    provider_suite_index = content.index("& $env:VENV_PYTHON -m pytest tests/test_provider_v2.py tests/test_provider_registry.py tests/test_provider_routes.py tests/test_api_handler_v2.py -q --durations=20")

    assert strict_discovery_index < provider_suite_index


def test_test_workflow_strict_provider_discovery_bootstraps_module_import_root():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "from providers.registry import get_registry" in content
    assert "sys.path.insert(0, str(ROOT / 'module'))" not in content
