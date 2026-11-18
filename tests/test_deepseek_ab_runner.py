import pytest

from tools import deepseek_ocr2_run as runner


def test_rejects_snapshot_with_different_code_before_loading_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot"):
        runner.verify_snapshot(tmp_path, "legacy")


def test_pdf_manifest_checks_returned_and_published_order(tmp_path):
    for i, word in enumerate(["first", "second"], start=1):
        page = tmp_path / f"page_{i:04}"
        page.mkdir()
        (page / "result.md").write_text(word, encoding="utf-8")
    result = "first\n<--- Page Split --->\nsecond"
    (tmp_path / "result.md").write_text(result, encoding="utf-8")
    assert runner.pdf_output_pages(tmp_path, result, 2) == [tmp_path / "page_0001", tmp_path / "page_0002"]
    swapped = "second\n<--- Page Split --->\nfirst"
    (tmp_path / "result.md").write_text(swapped, encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        runner.pdf_output_pages(tmp_path, swapped, 2)
