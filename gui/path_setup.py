from pathlib import Path
import sys
from typing import Iterable


def build_pythonpath(existing_paths: Iterable[str], project_root: Path) -> list[str]:
    root = str(project_root.resolve())
    gui_dir = str((project_root / "gui").resolve())
    ordered: list[str] = []
    seen: set[str] = set()

    for entry in (root, gui_dir, *existing_paths):
        path = str(entry)
        if not path or path in seen:
            continue
        ordered.append(path)
        seen.add(path)

    return ordered


def configure_sys_path(project_root: Path) -> None:
    sys.path[:] = build_pythonpath(sys.path, project_root)
