import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run_python(code: str, *, pythonpath: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_api_handler_v2_imports_from_project_root_only():
    result = _run_python("import module.api_handler_v2; print('ok')", pythonpath=str(ROOT))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_captioner_ignores_removed_v1_toggle():
    result = _run_python(
        (
            "import os, sys; "
            "os.environ['QINGLONG_API_V2'] = '0'; "
            "import module.captioner as captioner; "
            "print('module.api_handler' in sys.modules); "
            "print(hasattr(captioner, '_api_process_batch_v2'))"
        ),
        pythonpath=str(ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["False", "True"]


def test_top_level_providers_namespace_is_not_importable():
    result = _run_python(
        (
            "try:\n"
            "    import providers\n"
            "except ModuleNotFoundError:\n"
            "    print('missing')\n"
            "else:\n"
            "    print(getattr(providers, '__file__', 'loaded'))\n"
        ),
        pythonpath=str(ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "missing"


def test_code_uses_only_canonical_provider_namespace():
    legacy_provider_patterns = (
        re.compile(r"(^|\s)from providers(\.|\s+import)", re.MULTILINE),
        re.compile(r"(^|\s)import providers(\.|\s|$)", re.MULTILINE),
        re.compile(r"['\"]providers\."),
    )
    violations: list[str] = []
    for base_dir in (ROOT / "module", ROOT / "tests", ROOT / "gui", ROOT / "utils"):
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("*.py"):
            if path == Path(__file__):
                continue
            rel = path.relative_to(ROOT).as_posix()
            content = path.read_text(encoding="utf-8")
            if any(pattern.search(content) for pattern in legacy_provider_patterns):
                violations.append(rel)

    assert not violations, f"code still imports top-level providers namespace: {violations}"


def test_provider_modules_have_single_loaded_identity():
    result = _run_python(
        (
            "import sys; "
            "import module.providers.base; "
            "import module.providers.registry; "
            "import module.providers.ocr.dots; "
            "bad = sorted(name for name in sys.modules if name == 'providers' or name.startswith('providers.')); "
            "print(bad)"
        ),
        pythonpath=str(ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_tests_do_not_add_module_directory_to_sys_path_for_provider_imports():
    violations: list[str] = []
    module_path_insert_pattern = re.compile(
        r"^\s*sys\.path\.insert\(0,\s*str\(ROOT / ['\"]module['\"]\)\)",
        re.MULTILINE,
    )
    for path in (ROOT / "tests").rglob("*.py"):
        if path == Path(__file__):
            continue
        rel = path.relative_to(ROOT).as_posix()
        content = path.read_text(encoding="utf-8")
        inserts_module_dir = bool(module_path_insert_pattern.search(content))
        imports_providers = "module.providers" in content
        if inserts_module_dir and imports_providers:
            violations.append(rel)

    assert not violations, f"provider tests still place ROOT/module on sys.path: {violations}"


def test_gui_env_config_does_not_expose_v1_toggle():
    from gui.utils.env_config import ENV_VAR_DEFINITIONS

    keys = {entry["key"] for entry in ENV_VAR_DEFINITIONS}

    assert "QINGLONG_API_V2" not in keys
