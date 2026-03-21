import io
import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_logics_html_postprocess_handles_structured_blocks():
    from providers.ocr.logics import _logics_html_to_markdown

    raw = """
<div class="chart">flowchart LR
A-->B
</div>
<div class="code"><pre>print('hello')</pre></div>
<div class="music">X:1
Z:meta
C D E</div>
<div class="formula">E = mc^2</div>
"""
    rendered = _logics_html_to_markdown(raw)

    assert "```mermaid" in rendered
    assert "flowchart LR" in rendered
    assert "```code" in rendered
    assert "print('hello')" in rendered
    assert "```abc" in rendered
    assert "Z:meta" not in rendered
    assert "E = mc^2" in rendered


def test_logics_ocr_uses_official_default_prompt():
    from providers.base import ProviderContext
    from providers.ocr.logics import LogicsOCRProvider

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"prompts": {"logics_ocr_prompt": ""}},
        args=SimpleNamespace(ocr_model="logics_ocr"),
    )

    provider = LogicsOCRProvider(ctx)
    system_prompt, user_prompt = provider.get_prompts("image/png")

    assert system_prompt == ""
    assert user_prompt == "QwenVL HTML"
