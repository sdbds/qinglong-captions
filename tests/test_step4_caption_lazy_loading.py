import importlib
import importlib.util
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


def _load_step4_module(module_name: str):
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    step4_caption = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    gui_path = str(ROOT / "gui")
    original_sys_path = list(sys.path)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if gui_path not in sys.path:
        sys.path.insert(1, gui_path)
    try:
        spec.loader.exec_module(step4_caption)
    finally:
        sys.path[:] = original_sys_path

    return step4_caption


class _DummyContainer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyExecutionPanel:
    def __init__(self, *, start_label=None, show_start=True, on_start=None, height="50vh"):
        self.start_label = start_label
        self.show_start = show_start
        self.on_start = on_start
        self.height = height


def test_step4_caption_import_defers_gpu_and_execution_panel_modules(monkeypatch):
    from gui.path_setup import build_pythonpath

    monkeypatch.setattr(sys, "path", build_pythonpath(sys.path, ROOT))
    _clear_modules(
        "wizard",
        "gui.wizard",
        "module.gpu_profile",
        "components.execution_panel",
        "gui.components.execution_panel",
    )

    importlib.import_module("wizard.step4_caption")

    assert "module.gpu_profile" not in sys.modules
    assert "components.execution_panel" not in sys.modules
    assert "gui.components.execution_panel" not in sys.modules


def test_caption_step_init_keeps_gpu_probe_pending(monkeypatch):
    step4 = _load_step4_module("test_step4_caption_no_probe_in_init")
    monkeypatch.setattr(
        step4,
        "get_cached_gpu_probe",
        lambda: (_ for _ in ()).throw(AssertionError("GPU probe should not run in __init__")),
    )

    step = step4.CaptionStep()

    assert step.gpu_probe is None


def test_caption_step_defers_execution_panel_until_requested(monkeypatch):
    step4 = _load_step4_module("test_step4_caption_deferred_execution_panel")
    step = step4.CaptionStep()
    step._execution_panel_container = _DummyContainer()

    loads = []

    def _load_dummy_panel():
        loads.append("loaded")
        return _DummyExecutionPanel

    monkeypatch.setattr(step4, "_load_execution_panel_cls", _load_dummy_panel)

    assert step.panel is None

    panel = step._ensure_execution_panel()

    assert panel.show_start is True
    assert panel.start_label
    assert step._ensure_execution_panel() is panel
    assert loads == ["loaded"]
