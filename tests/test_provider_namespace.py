import os
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


def test_providers_alias_matches_module_providers_identity():
    result = _run_python(
        (
            "import module.providers.base as mpb; "
            "import providers.base as pb; "
            "print(mpb is pb); "
            "print(mpb.CaptionResult is pb.CaptionResult); "
            "print(mpb.Provider is pb.Provider)"
        ),
        pythonpath=str(ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["True", "True", "True"]


def test_module_package_uses_single_provider_namespace():
    violations: list[str] = []
    for path in (ROOT / "module").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("module/providers/") or rel == "module/api_handler_v2.py":
            content = path.read_text(encoding="utf-8")
            if "from providers." in content or "import providers." in content or "from providers import" in content:
                violations.append(rel)

    assert not violations, f"module package still imports top-level providers namespace: {violations}"


def test_gui_env_config_does_not_expose_v1_toggle():
    from gui.utils.env_config import ENV_VAR_DEFINITIONS

    keys = {entry["key"] for entry in ENV_VAR_DEFINITIONS}

    assert "QINGLONG_API_V2" not in keys
