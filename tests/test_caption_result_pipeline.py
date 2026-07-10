import io
import json
from types import SimpleNamespace

from rich.console import Console
import pytest


pytestmark = pytest.mark.compat


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


@pytest.mark.parametrize(
    ("result", "has_content", "is_persistable"),
    [
        (lambda CaptionResult: CaptionResult(raw="caption"), True, True),
        (lambda CaptionResult: CaptionResult(raw="  "), False, False),
        (lambda CaptionResult: CaptionResult(raw="{}", parsed={"scores": {"quality": 9}}), False, False),
        (lambda CaptionResult: CaptionResult(raw="{}", parsed={"description": "  "}), False, False),
        (lambda CaptionResult: CaptionResult(raw="{}", parsed={"markdown": "# title"}), True, True),
        (lambda CaptionResult: CaptionResult.skipped("policy", raw="diagnostic"), True, False),
        (lambda CaptionResult: CaptionResult.failed("backend", raw="diagnostic"), True, False),
    ],
)
def test_caption_result_persistence_is_explicit_and_semantic(result, has_content, is_persistable):
    from module.providers.base import CaptionResult

    value = result(CaptionResult)

    assert value.has_content is has_content
    assert value.is_persistable is is_persistable
    assert bool(value) is is_persistable


def test_caption_result_named_outcomes_keep_diagnostics_without_becoming_persistable():
    from module.providers.base import CaptionResult, CaptionStatus

    skipped = CaptionResult.skipped("missing pair", metadata={"provider": "test"})
    failed = CaptionResult.failed("request failed", raw="response body")

    assert skipped.status is CaptionStatus.SKIPPED
    assert skipped.metadata["skip_reason"] == "missing pair"
    assert skipped.error is None
    assert failed.status is CaptionStatus.FAILED
    assert failed.error == "request failed"
    assert failed.raw == "response body"
    assert not skipped
    assert not failed


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


@pytest.mark.parametrize("factory", ["skipped", "failed"])
def test_postprocess_preserves_non_success_outcome_and_diagnostic_text(tmp_path, factory):
    from module.caption_pipeline.postprocess import postprocess_caption_content
    from module.providers.base import CaptionResult

    source = CaptionResult.skipped("policy", raw="diagnostic") if factory == "skipped" else CaptionResult.failed("backend", raw="diagnostic")
    result = postprocess_caption_content(
        source,
        tmp_path / "image.png",
        SimpleNamespace(mode="all", document_image=False, ocr_model=""),
        _quiet_console(),
    )

    assert result is source
    assert not result.is_persistable


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


def test_update_dataset_captions_uses_paired_updates_and_moves_tag_only_after_merge(monkeypatch):
    import module.caption_pipeline.dataset_sync as dataset_sync
    from module.caption_pipeline.dataset_sync import CaptionUpdate, update_dataset_captions

    calls = []
    dataset = object()
    monkeypatch.setattr(
        dataset_sync,
        "merge_rows_preserving_schema",
        lambda target, updates, **kwargs: calls.append((target, updates, kwargs)),
    )
    monkeypatch.setattr(dataset_sync, "update_or_create_tag", lambda target, tag: calls.append(("tag", target, tag)))

    update_dataset_captions(
        dataset,
        [CaptionUpdate(uri="a.png", caption=json.dumps({"description": "stored"}))],
        merge_batch_size=10,
        console=_quiet_console(),
    )

    _, row_updates, kwargs = calls[0]
    assert row_updates[0].uri == "a.png"
    assert json.loads(row_updates[0].values["captions"][0])["description"] == "stored"
    assert kwargs == {"batch_size": 10}
    assert calls[1] == ("tag", dataset, "gemini")


def test_update_dataset_captions_with_no_updates_does_not_merge_or_move_tag(monkeypatch):
    import module.caption_pipeline.dataset_sync as dataset_sync
    from module.caption_pipeline.dataset_sync import update_dataset_captions

    monkeypatch.setattr(
        dataset_sync,
        "merge_rows_preserving_schema",
        lambda *_args, **_kwargs: pytest.fail("empty updates must not merge"),
    )
    monkeypatch.setattr(
        dataset_sync,
        "update_or_create_tag",
        lambda *_args, **_kwargs: pytest.fail("empty updates must not move the success tag"),
    )

    dataset = object()
    assert update_dataset_captions(dataset, [], merge_batch_size=10, console=_quiet_console()) is dataset


def test_orchestrator_builds_only_persistable_uri_caption_pairs():
    from module.caption_pipeline.dataset_sync import CaptionUpdate
    from module.caption_pipeline.orchestrator import CaptionJobResult, _build_caption_updates
    from module.providers.base import CaptionResult

    results = [
        CaptionJobResult(0, "a.png", "image/png", CaptionResult(raw="first")),
        CaptionJobResult(1, "b.png", "image/png", CaptionResult.skipped("policy", raw="diagnostic")),
        CaptionJobResult(2, "c.png", "image/png", CaptionResult.failed("backend", raw="diagnostic")),
        CaptionJobResult(3, "d.png", "image/png", CaptionResult(raw="  ")),
        CaptionJobResult(4, "e.png", "image/png", CaptionResult(raw="{}", parsed={"description": "fifth"})),
    ]

    assert _build_caption_updates(results) == [
        CaptionUpdate(uri="a.png", caption="first"),
        CaptionUpdate(uri="e.png", caption=json.dumps({"description": "fifth"}, ensure_ascii=False)),
    ]
