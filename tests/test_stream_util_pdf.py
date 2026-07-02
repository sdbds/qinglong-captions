import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _png_bytes(size=(8, 6), color="white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_iter_pdf_pages_high_quality_is_lazy_and_closes_document(monkeypatch, tmp_path):
    from utils.stream_util import iter_pdf_pages_high_quality

    events = []

    class FakePixmap:
        def __init__(self, page_index: int):
            self.page_index = page_index

        def tobytes(self, image_format: str) -> bytes:
            events.append(("tobytes", self.page_index, image_format))
            return _png_bytes(size=(8 + self.page_index, 6 + self.page_index))

    class FakePage:
        def __init__(self, page_index: int):
            self.page_index = page_index

        def get_pixmap(self, *, matrix, alpha):
            events.append(("render", self.page_index, matrix, alpha))
            return FakePixmap(self.page_index)

    class FakeDocument:
        page_count = 2

        def __init__(self):
            self.closed = False

        def __getitem__(self, page_index: int):
            return FakePage(page_index)

        def close(self):
            events.append(("close",))
            self.closed = True

    fake_doc = FakeDocument()

    def fake_open(path):
        events.append(("open", path))
        return fake_doc

    fake_fitz = SimpleNamespace(open=fake_open, Matrix=lambda x, y: ("matrix", x, y))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    pdf_path = tmp_path / "score.pdf"
    iterator = iter_pdf_pages_high_quality(pdf_path, dpi=72, image_format="PNG")

    assert events == []

    first = next(iterator)
    assert first.pdf_path == pdf_path
    assert first.page_index == 0
    assert first.page_number == 1
    assert first.page_count == 2
    assert first.size == (8, 6)
    assert fake_doc.closed is False

    second = next(iterator)
    assert second.page_index == 1
    assert second.page_number == 2
    assert second.page_count == 2
    assert second.size == (9, 7)

    with pytest.raises(StopIteration):
        next(iterator)

    assert fake_doc.closed is True
    assert events[0] == ("open", str(pdf_path))
    assert events[-1] == ("close",)


def test_iter_pdf_pages_high_quality_closes_document_on_render_error(monkeypatch, tmp_path):
    from utils.stream_util import iter_pdf_pages_high_quality

    class FakePage:
        @staticmethod
        def get_pixmap(*, matrix, alpha):
            raise RuntimeError("render failed")

    class FakeDocument:
        page_count = 1

        def __init__(self):
            self.closed = False

        def __getitem__(self, page_index: int):
            return FakePage()

        def close(self):
            self.closed = True

    fake_doc = FakeDocument()
    fake_fitz = SimpleNamespace(open=lambda path: fake_doc, Matrix=lambda x, y: ("matrix", x, y))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    with pytest.raises(RuntimeError, match="page 1"):
        next(iter_pdf_pages_high_quality(tmp_path / "broken.pdf"))

    assert fake_doc.closed is True


def test_iter_pdf_pages_high_quality_rejects_zero_page_pdf(monkeypatch, tmp_path):
    from utils.stream_util import iter_pdf_pages_high_quality

    class FakeDocument:
        page_count = 0

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_doc = FakeDocument()
    fake_fitz = SimpleNamespace(open=lambda path: fake_doc, Matrix=lambda x, y: ("matrix", x, y))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    with pytest.raises(ValueError, match="no renderable pages"):
        next(iter_pdf_pages_high_quality(tmp_path / "empty.pdf"))

    assert fake_doc.closed is True


def test_eager_pdf_helper_is_removed_from_active_code():
    import utils.stream_util as stream_util

    deleted_name = "pdf_to_images_" + "high_quality"
    assert not hasattr(stream_util, deleted_name)

    offenders = []
    for root_name in ("module", "utils", "tests"):
        for path in (ROOT / root_name).rglob("*.py"):
            if path.name == "test_stream_util_pdf.py":
                continue
            if deleted_name in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
