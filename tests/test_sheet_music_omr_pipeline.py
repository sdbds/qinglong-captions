import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def _valid_tokens() -> tuple[str, ...]:
    return (
        "*clefG2",
        "<t>",
        "*clefG2",
        "<b>",
        "*M4/4",
        "<t>",
        "*M4/4",
        "<b>",
        "=1",
        "<t>",
        "=1",
        "<b>",
        "4",
        "c",
        "<t>",
        "4",
        "e",
        "<b>",
        "*-",
        "<t>",
    )


class _FakeRecognizer:
    repo_id = "test/repo"
    revision = "test-revision"
    providers = ("CPUExecutionProvider",)
    model_config = SimpleNamespace(max_length=7512)
    preprocessor_config = SimpleNamespace(
        image_size=(1024, 1024),
        interpolation="bilinear",
        rescale_factor=1 / 255,
    )
    bundle_hashes = {
        "encoder": "encoder-hash",
        "decoder": "decoder-hash",
        "config": "config-hash",
        "preprocessor_config": "preprocessor-hash",
    }

    def __init__(self, *, tokens=None, truncated=False, error=None):
        self.tokens = tuple(tokens or _valid_tokens())
        self.truncated = truncated
        self.error = error
        self.calls = []

    def generate(self, image, *, max_tokens=None, progress_callback=None):
        self.calls.append(image)
        if self.error is not None:
            raise self.error
        if progress_callback is not None:
            progress_callback(8)
        token_ids = (100, *range(1, len(self.tokens) + 1))
        if not self.truncated:
            token_ids = (*token_ids, 183)
        return SimpleNamespace(
            token_ids=token_ids,
            tokens=self.tokens,
            terminated_by_eos=not self.truncated,
            truncated=self.truncated,
            elapsed_seconds=0.25,
        )


def test_image_pipeline_writes_diagnostics_and_both_symbolic_formats(tmp_path: Path):
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    output_dir = tmp_path / "out"

    result = MuSViTOmrPipeline(
        recognizer=_FakeRecognizer(),
        output_dir=output_dir,
        output_format="both",
    ).run(image_path)

    page_dir = output_dir / "score.png"
    assert result.ok is True
    assert (page_dir / "tokens.json").is_file()
    assert (page_dir / "score.krn").read_text(encoding="utf-8").startswith("**kern\t**kern\n")
    assert (page_dir / "score.musicxml").is_file()
    assert (page_dir / "score.mid").is_file()
    metadata = json.loads((page_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["kern_status"] == "ok"
    assert metadata["outputs"] == {
        "midi": "score.mid",
        "musicxml": "score.musicxml",
    }
    assert metadata["warnings"] == {}
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"] == {"failed": 0, "processed": 1, "skipped": 0}


def test_page_musicxml_export_preserves_parsed_notation(monkeypatch, tmp_path: Path):
    import module.sheet_music_omr.pipeline as pipeline_module

    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    make_notation_values = []
    original_writer = pipeline_module.write_musicxml_score

    def recording_writer(score, target, *, make_notation=None):
        make_notation_values.append(make_notation)
        return original_writer(
            score,
            target,
            make_notation=make_notation,
        )

    monkeypatch.setattr(
        pipeline_module,
        "write_musicxml_score",
        recording_writer,
    )

    result = pipeline_module.MuSViTOmrPipeline(
        recognizer=_FakeRecognizer(),
        output_dir=tmp_path / "out",
        output_format="musicxml",
    ).run(image_path)

    assert result.ok is True
    assert make_notation_values == [False]


def test_page_export_normalizes_null_only_kern_slots_as_hidden_silence(
    tmp_path: Path,
):
    from music21 import converter, note, stream

    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    tokens = (
        "*clefF4",
        "<t>",
        "*clefG2",
        "<b>",
        "*M4/4",
        "<t>",
        "*M4/4",
        "<b>",
        ".",
        "<t>",
        "8cc",
        "<b>",
        "=",
        "<t>",
        "=",
        "<b>",
        "1C",
        "<t>",
        "1ee",
        "<b>",
        "=",
        "<t>",
        "=",
        "<b>",
        "*^",
        "<t>",
        "*",
        "<b>",
        "1D",
        "<t>",
        "1F",
        "<t>",
        ".",
        "<b>",
        "*v",
        "<t>",
        "*v",
        "<t>",
        "*",
        "<b>",
        "=",
        "<t>",
        "=",
        "<b>",
        "1E",
        "<t>",
        "1gg",
        "<b>",
        "=",
        "<t>",
        "=",
        "<b>",
        "*-",
        "<t>",
    )

    result = MuSViTOmrPipeline(
        recognizer=_FakeRecognizer(tokens=tokens),
        output_dir=tmp_path / "out",
        output_format="musicxml",
    ).run(image_path)

    assert result.ok is True
    page_dir = tmp_path / "out" / "score.png"
    score = converter.parse(page_dir / "score.musicxml")
    assert [len(tuple(part.getElementsByClass(stream.Measure))) for part in score.parts] == [4, 4]
    hidden_rests = [
        item for part in score.parts for item in part.recurse().getElementsByClass(note.Rest) if item.style.hideObjectOnPrint
    ]
    assert [rest.quarterLength for rest in hidden_rests] == [4, 0.5]
    metadata = json.loads((page_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["padding"] == {
        "count": 2,
        "quarter_length": 4.5,
    }


def test_truncated_page_keeps_diagnostics_but_writes_no_final_formats(tmp_path: Path):
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    page_dir = tmp_path / "out" / "score.png"
    page_dir.mkdir(parents=True)
    (page_dir / "score.musicxml").write_text("stale", encoding="utf-8")
    (page_dir / "score.mid").write_bytes(b"stale")

    result = MuSViTOmrPipeline(
        recognizer=_FakeRecognizer(truncated=True),
        output_dir=tmp_path / "out",
        output_format="both",
        max_tokens=64,
    ).run(image_path)

    assert result.ok is False
    assert (page_dir / "tokens.json").is_file()
    assert (page_dir / "score.krn").is_file()
    assert not (page_dir / "score.musicxml").exists()
    assert not (page_dir / "score.mid").exists()
    metadata = json.loads((page_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["truncated"] is True
    assert "generation" in metadata["failures"]


def test_invalid_kern_keeps_exact_candidate_and_skips_exports(tmp_path: Path):
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    recognizer = _FakeRecognizer(tokens=("4", "c", "<t>", "4", "e"))
    page_dir = tmp_path / "out" / "score.png"

    result = MuSViTOmrPipeline(
        recognizer=recognizer,
        output_dir=tmp_path / "out",
        output_format="musicxml",
    ).run(image_path)

    assert result.ok is False
    assert (page_dir / "score.krn").read_text(encoding="utf-8") == "4c\t4e\n"
    assert not (page_dir / "score.musicxml").exists()
    metadata = json.loads((page_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["kern_status"] == "invalid"
    assert "kern" in metadata["failures"]


def test_matching_resume_signature_skips_valid_requested_outputs(tmp_path: Path):
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    recognizer = _FakeRecognizer()
    pipeline = MuSViTOmrPipeline(
        recognizer=recognizer,
        output_dir=tmp_path / "out",
        output_format="musicxml",
        skip_completed=True,
    )

    first = pipeline.run(image_path)
    second = pipeline.run(image_path)

    assert first.ok is True
    assert second.ok is True
    assert len(recognizer.calls) == 1
    assert second.pages[0].skipped is True


def test_notation_reconstruction_invalidates_previous_exports():
    from module.sheet_music_omr.pipeline import (
        AGGREGATE_SCHEMA_VERSION,
        EXPORT_SCHEMA_VERSION,
    )

    assert EXPORT_SCHEMA_VERSION == 6
    assert AGGREGATE_SCHEMA_VERSION == 6


def test_streamed_pdf_runs_and_releases_one_page_at_a_time(tmp_path: Path):
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF")
    live_pages = []
    closed_pages = []

    def renderer(path, *, dpi, image_format):
        for page_number in (1, 2):
            image = Image.new("RGB", (8, 8), color="white")
            live_pages.append(page_number)
            original_close = image.close

            def close(number=page_number, close_image=original_close):
                live_pages.remove(number)
                closed_pages.append(number)
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

    class StreamingRecognizer(_FakeRecognizer):
        def generate(self, image, **kwargs):
            assert len(live_pages) == 1
            return super().generate(image, **kwargs)

    result = MuSViTOmrPipeline(
        recognizer=StreamingRecognizer(),
        output_dir=tmp_path / "out",
        output_format="both",
        pdf_dpi=123,
        pdf_renderer=renderer,
    ).run(pdf_path)

    assert result.ok is True
    assert closed_pages == [1, 2]
    assert live_pages == []
    assert (tmp_path / "out" / "book.pdf" / "page_0001" / "score.musicxml").is_file()
    assert (tmp_path / "out" / "book.pdf" / "page_0002" / "score.musicxml").is_file()
    document_dir = tmp_path / "out" / "book.pdf"
    assert (document_dir / "score.musicxml").is_file()
    assert (document_dir / "score.mid").is_file()
    document_metadata = json.loads((document_dir / "metadata.json").read_text(encoding="utf-8"))
    assert document_metadata["page_count"] == 2
    assert document_metadata["outputs"] == {
        "midi": "score.mid",
        "musicxml": "score.musicxml",
    }
    assert result.documents[0].ok is True
    assert result.processed == 2
    assert result.skipped == 0
    assert result.failed == 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["documents"][0]["ok"] is True


def test_failed_pdf_page_removes_stale_aggregate_and_keeps_page_diagnostics(
    tmp_path: Path,
):
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF")
    document_dir = tmp_path / "out" / "book.pdf"
    document_dir.mkdir(parents=True)
    stale_xml = document_dir / "score.musicxml"
    stale_midi = document_dir / "score.mid"
    stale_xml.write_text("stale", encoding="utf-8")
    stale_midi.write_bytes(b"stale")

    def renderer(path, *, dpi, image_format):
        for page_index in range(2):
            image = Image.new("RGB", (8, 8), color="white")
            yield SimpleNamespace(
                pdf_path=pdf_path,
                page_index=page_index,
                page_count=2,
                image=image,
                size=image.size,
            )

    class SecondPageFailsRecognizer(_FakeRecognizer):
        def generate(self, image, **kwargs):
            if len(self.calls) == 1:
                self.tokens = ("4", "c")
            return super().generate(image, **kwargs)

    result = MuSViTOmrPipeline(
        recognizer=SecondPageFailsRecognizer(),
        output_dir=tmp_path / "out",
        output_format="both",
        pdf_renderer=renderer,
    ).run(pdf_path)

    assert result.ok is False
    assert result.documents[0].ok is False
    assert result.failed == 1
    assert "pages" in result.documents[0].failures
    assert not stale_xml.exists()
    assert not stale_midi.exists()
    assert (document_dir / "page_0002" / "tokens.json").is_file()


def test_pdf_aggregate_runtime_error_becomes_document_failure(
    monkeypatch,
    tmp_path: Path,
):
    import module.sheet_music_omr.pipeline as pipeline_module

    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF")

    def renderer(path, *, dpi, image_format):
        image = Image.new("RGB", (8, 8), color="white")
        yield SimpleNamespace(
            pdf_path=pdf_path,
            page_index=0,
            page_number=1,
            page_count=1,
            image=image,
            size=image.size,
        )

    def fail_aggregate(scores):
        raise RuntimeError("music21 aggregation failed")

    monkeypatch.setattr(
        pipeline_module,
        "combine_page_scores",
        fail_aggregate,
    )

    result = pipeline_module.MuSViTOmrPipeline(
        recognizer=_FakeRecognizer(),
        output_dir=tmp_path / "out",
        output_format="both",
        pdf_renderer=renderer,
    ).run(pdf_path)

    assert result.ok is False
    assert result.failed == 1
    assert result.documents[0].ok is False
    assert "RuntimeError" in result.documents[0].failures["aggregate"]


def test_directory_layout_preserves_relative_parents_for_equal_names(tmp_path: Path):
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    source_root = tmp_path / "scores"
    for parent in ("left", "right"):
        path = source_root / parent
        path.mkdir(parents=True)
        Image.new("RGB", (8, 8), color="white").save(path / "page.png")

    result = MuSViTOmrPipeline(
        recognizer=_FakeRecognizer(),
        output_dir=tmp_path / "out",
        output_format="musicxml",
    ).run(source_root)

    assert result.ok is True
    assert (tmp_path / "out" / "left" / "page.png" / "score.musicxml").is_file()
    assert (tmp_path / "out" / "right" / "page.png" / "score.musicxml").is_file()
