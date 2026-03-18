import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_write_caption_output_writes_structured_text_and_json(tmp_path):
    from utils.output_writer import write_caption_output

    source = tmp_path / "image.png"
    source.write_bytes(b"image")

    text_path, json_path = write_caption_output(
        source,
        {"long_description": "long text", "short_description": "short text"},
        mime="image/png",
    )

    assert text_path.read_text(encoding="utf-8") == "long text"
    assert json.loads(json_path.read_text(encoding="utf-8"))["short_description"] == "short text"


def test_write_caption_output_writes_markdown_for_documents(tmp_path):
    from utils.output_writer import write_caption_output

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4")

    text_path, json_path = write_caption_output(source, "# Title\n", mime="application/pdf")

    assert text_path.suffix == ".md"
    assert text_path.read_text(encoding="utf-8") == "# Title\n"
    assert json_path is None


def test_write_markdown_output_skips_empty_markdown(tmp_path):
    from utils.output_writer import write_markdown_output

    output_dir = tmp_path / "doc"

    result = write_markdown_output(output_dir, "   \n\t  ")

    assert result is None
    assert not (output_dir / "result.md").exists()
