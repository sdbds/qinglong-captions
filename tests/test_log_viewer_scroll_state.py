import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.components.log_viewer import LogViewer


def test_near_bottom_when_content_fits_container():
    assert LogViewer._is_near_bottom(0, 300, 320)


def test_near_bottom_when_within_threshold():
    assert LogViewer._is_near_bottom(380, 500, 100)


def test_not_near_bottom_when_user_scrolled_up():
    assert not LogViewer._is_near_bottom(320, 500, 100)
