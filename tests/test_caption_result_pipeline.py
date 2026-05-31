import io
import json
from types import SimpleNamespace

from rich.console import Console


def _quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def test_caption_result_exposes_text_payload_extension_and_dataset_serialization():
    from module.providers.base import CaptionResult

    raw = CaptionResult(raw="plain caption")
    structured = CaptionResult(raw="{}", parsed={"description": "structured caption", "caption_extension": "txt"})
    ast = CaptionResult(
        raw="{}",
        parsed={
            "task_kind": "ast",
            "translation_srt": "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
            "caption_extension": ".srt",
            "subtitle_format": "srt",
        },
    )
    document = CaptionResult(raw="{}", parsed={"markdown": "# Title\n", "caption_extension": ".md"})

    assert raw.payload == "plain caption"
    assert raw.text == "plain caption"
    assert raw.to_dataset_caption() == "plain caption"
    assert structured.payload == {"description": "structured caption", "caption_extension": "txt"}
    assert structured.text == "structured caption"
    assert structured.caption_extension == ".txt"
    assert json.loads(structured.to_dataset_caption())["description"] == "structured caption"
    assert ast.text == "1\n00:00:00,000 --> 00:00:01,000\n你好\n"
    assert ast.caption_extension == ".srt"
    assert document.text == "# Title\n"
    assert document.caption_extension == ".md"


def test_postprocess_wraps_json_object_as_caption_result(tmp_path):
    from module.caption_pipeline.postprocess import postprocess_caption_content
    from module.providers.base import CaptionResult

    path = tmp_path / "image.png"
    result = postprocess_caption_content(
        '{"description": "json caption"}',
        path,
        SimpleNamespace(mode="all", document_image=False, ocr_model=""),
        _quiet_console(),
    )

    assert isinstance(result, CaptionResult)
    assert result.is_structured
    assert result.description == "json caption"


def test_postprocess_audio_subtitle_remains_caption_result(tmp_path):
    from module.caption_pipeline.postprocess import postprocess_caption_content
    from module.providers.base import CaptionResult

    path = tmp_path / "audio.wav"
    result = postprocess_caption_content(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        path,
        SimpleNamespace(mode="all", document_image=False, ocr_model=""),
        _quiet_console(),
    )

    assert isinstance(result, CaptionResult)
    assert not result.is_structured
    assert "hello" in result.raw


def test_write_caption_output_accepts_structured_caption_result(tmp_path):
    from module.providers.base import CaptionResult
    from utils.output_writer import write_caption_output

    source = tmp_path / "sample.wav"
    source.write_bytes(b"audio")
    result = CaptionResult(
        raw="{}",
        parsed={
            "task_kind": "transcribe",
            "transcript": "spoken words",
            "caption_extension": ".txt",
        },
    )

    text_path, json_path = write_caption_output(source, result, "audio/wav")

    assert text_path.read_text(encoding="utf-8") == "spoken words"
    assert json_path is not None
    assert json.loads(json_path.read_text(encoding="utf-8"))["transcript"] == "spoken words"


def test_update_dataset_captions_serializes_caption_result():
    from module.caption_pipeline.dataset_sync import update_dataset_captions
    from module.providers.base import CaptionResult

    executed_tables = []

    class MergeBuilder:
        def when_matched_update_all(self):
            return self

        def execute(self, table):
            executed_tables.append(table)

    class FakeTags:
        def create(self, name, value):
            self.created = (name, value)

        def update(self, name, value):
            self.updated = (name, value)

    class FakeDataset:
        def __init__(self):
            self.tags = FakeTags()

        def merge_insert(self, on):
            assert on == "uris"
            return MergeBuilder()

    update_dataset_captions(
        FakeDataset(),
        ["a.png"],
        [CaptionResult(raw="{}", parsed={"description": "stored"})],
        merge_batch_size=10,
        console=_quiet_console(),
    )

    rows = executed_tables[0].to_pylist()
    assert json.loads(rows[0]["captions"][0])["description"] == "stored"
