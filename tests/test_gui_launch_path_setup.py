import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clear_utils_modules() -> None:
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            sys.modules.pop(name, None)


def test_gui_launch_path_setup_keeps_project_utils_importable(monkeypatch):
    from gui.path_setup import build_pythonpath

    pythonpath = build_pythonpath([str(ROOT / "gui"), "sentinel"], ROOT)

    assert pythonpath[0] == str(ROOT)
    assert pythonpath.index(str(ROOT)) < pythonpath.index(str(ROOT / "gui"))

    monkeypatch.setattr(sys, "path", pythonpath[:])
    _clear_utils_modules()

    output_writer = importlib.import_module("utils.output_writer")

    assert Path(output_writer.__file__).resolve() == ROOT / "utils" / "output_writer.py"
