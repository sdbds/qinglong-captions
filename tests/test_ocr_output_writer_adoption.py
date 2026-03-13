import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


OCR_PROVIDER_FILES = [
    "module/providers/ocr/deepseek.py",
    "module/providers/ocr/firered.py",
    "module/providers/ocr/hunyuan.py",
    "module/providers/ocr/nanonets.py",
    "module/providers/ocr/olmocr.py",
    "module/providers/ocr/glm.py",
    "module/providers/ocr/chandra.py",
]


@pytest.mark.parametrize("relative_path", OCR_PROVIDER_FILES)
def test_ocr_providers_route_markdown_sidecars_through_shared_writer(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "write_markdown_output" in source
    assert ".write_text(" not in source
