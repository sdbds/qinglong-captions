from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def test_source_discovery_is_deterministic_and_filters_extensions(tmp_path: Path):
    from module.sheet_music_omr.inputs import collect_source_inputs

    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    Image.new("RGB", (2, 2)).save(tmp_path / "b" / "score.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "a" / "score.JPG")
    (tmp_path / "a" / "book.pdf").write_bytes(b"%PDF")
    (tmp_path / "ignore.txt").write_text("no", encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in collect_source_inputs(tmp_path)] == [
        "a/book.pdf",
        "a/score.JPG",
        "b/score.png",
    ]
    assert [path.name for path in collect_source_inputs(tmp_path, recursive=False)] == []


def test_pdf_page_iterator_closes_each_image_before_requesting_next(tmp_path: Path):
    from module.sheet_music_omr.inputs import iter_source_pages

    pdf_path = tmp_path / "score.pdf"
    pdf_path.write_bytes(b"%PDF")
    events = []

    def renderer(path, *, dpi, image_format):
        assert Path(path) == pdf_path
        assert dpi == 123
        assert image_format == "PNG"
        for page_number in (1, 2):
            events.append(("render", page_number))
            image = Image.new("RGB", (page_number + 2, 4))
            original_close = image.close

            def close(number=page_number, close_image=original_close):
                events.append(("close", number))
                close_image()

            image.close = close
            yield SimpleNamespace(
                pdf_path=pdf_path,
                page_index=page_number - 1,
                page_number=page_number,
                page_count=2,
                image=image,
                size=image.size,
            )

    pages = iter_source_pages(pdf_path, pdf_dpi=123, pdf_renderer=renderer)
    first = next(pages)
    assert first.page_number == 1
    assert events == [("render", 1)]

    second = next(pages)
    assert second.page_number == 2
    assert events == [("render", 1), ("close", 1), ("render", 2)]

    pages.close()
    assert events == [
        ("render", 1),
        ("close", 1),
        ("render", 2),
        ("close", 2),
    ]
