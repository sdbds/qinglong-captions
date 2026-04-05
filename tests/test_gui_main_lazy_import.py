import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clear_modules(*names: str) -> None:
    for module_name in list(sys.modules):
        if module_name in names:
            sys.modules.pop(module_name, None)
            continue
        for name in names:
            if module_name.startswith(f"{name}."):
                sys.modules.pop(module_name, None)
                break


def test_main_import_defers_wizard_step_modules(monkeypatch):
    from gui.path_setup import build_pythonpath

    monkeypatch.setattr(sys, "path", build_pythonpath(sys.path, ROOT))
    _clear_modules("main", "wizard", "gui.wizard")

    importlib.import_module("main")

    assert "wizard" not in sys.modules
    assert "wizard.step0_setup" not in sys.modules
    assert "wizard.step6_tools" not in sys.modules


def test_tools_page_imports_tools_step_on_demand(monkeypatch):
    from gui.path_setup import build_pythonpath

    monkeypatch.setattr(sys, "path", build_pythonpath(sys.path, ROOT))
    _clear_modules("main", "wizard", "gui.wizard")

    main = importlib.import_module("main")
    calls: list[str] = []
    monkeypatch.setattr(main, "page_base", lambda render_func: calls.append(render_func.__name__))

    main.tools_page()

    assert calls == ["render_tools_step"]
    assert "wizard.step6_tools" in sys.modules


def test_step6_tools_import_defers_backend_modules(monkeypatch):
    from gui.path_setup import build_pythonpath

    monkeypatch.setattr(sys, "path", build_pythonpath(sys.path, ROOT))
    _clear_modules(
        "wizard",
        "gui.wizard",
        "module.vocal_midi",
        "module.gpu_profile",
        "module.see_through",
        "components.execution_panel",
        "gui.components.execution_panel",
        "components.log_viewer",
        "gui.components.log_viewer",
    )

    importlib.import_module("wizard.step6_tools")

    assert "module.vocal_midi" not in sys.modules
    assert "module.gpu_profile" not in sys.modules
    assert "module.see_through.see_through_profile" not in sys.modules
    assert "components.execution_panel" not in sys.modules
    assert "gui.components.execution_panel" not in sys.modules
    assert "components.log_viewer" not in sys.modules
    assert "gui.components.log_viewer" not in sys.modules
