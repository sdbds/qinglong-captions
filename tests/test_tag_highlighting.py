# -*- coding: utf-8 -*-

import importlib
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


class FakeClassifier:
    def classify(self, tags):
        return {"0": [f"[cyan]{tag}[/cyan]" for tag in tags]}

    def get_colored_tag(self, tag):
        return f"[cyan]{tag}[/cyan]"


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_format_description_does_not_import_wdtagger():
    sys.modules.pop("utils.wdtagger", None)
    stream_util = _reload_module("utils.stream_util")

    with patch.object(stream_util, "get_tag_classifier", return_value=FakeClassifier()):
        formatted, highlight_rate = stream_util.format_description(
            "cat girl in a school uniform",
            "cat girl, school uniform",
        )

    assert "[cyan]cat girl[/cyan]" in formatted
    assert "[cyan]school uniform[/cyan]" in formatted
    assert highlight_rate
    assert "utils.wdtagger" not in sys.modules


def test_caption_layout_does_not_import_wdtagger():
    sys.modules.pop("utils.wdtagger", None)
    console_util = _reload_module("utils.console_util")

    with patch.object(console_util, "get_tag_classifier", return_value=FakeClassifier()):
        layout = console_util.CaptionLayout(
            tag_description="<cat girl>, school uniform",
            short_description="short",
            long_description="long",
            pixels="pixels",
        )

    assert "[cyan]cat girl[/cyan]" in layout.tag_description
    assert "[cyan]school uniform[/cyan]" in layout.tag_description
    assert "utils.wdtagger" not in sys.modules
