"""Provider output contracts exercised through execution and publication."""

import io
import json

import pytest
from PIL import Image
from rich.console import Console

from module.caption_pipeline.orchestrator import CaptionJob, _build_caption_updates, _process_single_caption_job
from module.providers.base import CaptionResult, CaptionStatus, ProviderContext
from tests.provider_v2_helpers import make_provider_args
from utils.output_writer import write_caption_output

BBOX_PAYLOAD = {
    "high_level_description": "A red square on a white background.",
    "style_description": {"aesthetics": "clean geometric shapes"},
    "compositional_deconstruction": {
        "background": "white",
        "elements": [{"type": "object", "bbox": [0, 0, 100, 100], "desc": "red square"}],
    },
}


def _image(tmp_path):
    source = tmp_path / "input.jpg"
    Image.new("RGB", (32, 32), "red").save(source)
    return source


def _publish(source, response, *, template="bbox_json"):
    args = make_provider_args(
        image_prompt_template=template, mode="long", segment_time=None,
        scene_threshold=0, scene_min_len=0,
    )
    return _process_single_caption_job(
        CaptionJob(0, str(source), "image/jpeg", 0, "offline-hash"),
        args,
        {},
        api_process_batch_fn=lambda **kwargs: response,
        console_obj=Console(file=io.StringIO()),
    )


@pytest.mark.parametrize("already_parsed", [False, True])
def test_bbox_contract_publishes_complete_json_sidecar_and_lance_update(tmp_path, already_parsed):
    source = _image(tmp_path)
    response = CaptionResult(
        raw=json.dumps(BBOX_PAYLOAD),
        parsed=BBOX_PAYLOAD if already_parsed else None,
    )

    result = _publish(source, response)

    assert result.output.is_persistable
    assert json.loads(source.with_suffix(".json").read_text(encoding="utf-8")) == BBOX_PAYLOAD
    assert json.loads(source.with_suffix(".txt").read_text(encoding="utf-8")) == BBOX_PAYLOAD
    updates = _build_caption_updates([result])
    assert len(updates) == 1
    assert updates[0].uri == str(source)
    assert json.loads(updates[0].caption) == BBOX_PAYLOAD


@pytest.mark.parametrize("payload", [
    {},
    {"provider": "gemini", "scores": {"quality": 10}, "average_score": 10},
    {"error": "upstream unavailable"},
    {"high_level_description": "", "style_description": {}},
    {"high_level_description": " \n\t"},
    {"high_level_description": None},
    {"high_level_description": []},
    {"high_level_description": {}},
    {"high_level_description": {"error": "upstream unavailable"}},
    {"high_level_description": 42},
    {"high_level_description": True},
])
def test_nonsemantic_structured_result_keeps_existing_sidecars_and_has_no_lance_update(tmp_path, payload):
    source = _image(tmp_path)
    write_caption_output(source, {"description": "Existing caption."}, "image/jpeg")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}

    result = _publish(source, CaptionResult(raw=json.dumps(payload), parsed=payload))

    assert not result.output.is_persistable
    assert _build_caption_updates([result]) == []
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


def _gemini_provider(**overrides):
    from module.providers.vision_api.gemini import GeminiProvider

    args = make_provider_args(
        gemini_api_key="offline-placeholder", gemini_model_path="gemini-2.0-flash",
        max_retries=2, wait_time=0, mode="all",
    )
    vars(args).update(overrides)
    return GeminiProvider(ProviderContext(
        console=Console(file=io.StringIO()),
        args=args,
        config={
            "prompts": {
                "image_system_prompt": "Describe the image and score its quality.",
                "image_prompt": "Describe this image.",
                "pair_image_system_prompt": "Compare the two images.",
                "pair_image_prompt": "Describe the change.",
                "tag_system": "Return comma-separated tags.",
                "rating_system": "Return the quality assessment as text.",
                "bbox_system": "Return a JSON object describing the image and its bounding boxes.",
                "image_templates": {
                    "danbooru_tags": {"system_key": "tag_system", "output": "text"},
                    "rating": {"system_key": "rating_system", "output": "text"},
                    "bbox_json": {"system_key": "bbox_system", "output": "json"},
                },
            },
            "generation_config": {"default": {"response_mime_type": "text/plain"}},
        },
    ))


def _gemini_endpoint(monkeypatch, response_text, *, filtered=False):
    from google import genai
    from google.genai import types

    calls = []

    def generate_content_stream(**kwargs):
        calls.append(kwargs)
        candidate = (
            types.Candidate(finish_reason="SAFETY") if filtered else
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=response_text)]), finish_reason="STOP")
        )
        return iter([types.GenerateContentResponse(candidates=[candidate])])

    from types import SimpleNamespace
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=generate_content_stream),
    ))
    return calls


@pytest.mark.parametrize(("template", "response", "expected"), [
    ("danbooru_tags", "red square, geometric shape", "red square, geometric shape"),
    ("rating", "**Scores:** 8/10\n**Description:** A red square.", "Scores: 8/10\nDescription: A red square."),
])
def test_gemini_text_template_preserves_valid_plaintext(monkeypatch, tmp_path, template, response, expected):
    calls = _gemini_endpoint(monkeypatch, response)
    provider = _gemini_provider(image_prompt_template=template)

    result = provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert result.status is CaptionStatus.SUCCESS
    assert result.raw == expected
    assert result.parsed is None
    assert result.is_persistable
    assert len(calls) == 1
    assert calls[0]["config"].response_schema is None
    assert calls[0]["config"].response_mime_type == "text/plain"
    assert "filtered" not in provider.ctx.console.file.getvalue().lower()


def test_gemini_text_template_overrides_default_json_mime(monkeypatch, tmp_path):
    calls = _gemini_endpoint(monkeypatch, "red square, geometric shape")
    provider = _gemini_provider(image_prompt_template="danbooru_tags")
    provider.ctx.config["generation_config"]["default"]["response_mime_type"] = "application/json"

    result = provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert result.raw == "red square, geometric shape"
    assert result.parsed is None
    assert calls[0]["config"].response_mime_type == "text/plain"


@pytest.mark.parametrize("template", ["", "custom", "unknown-template"])
def test_gemini_default_json_caption_preserves_structured_output(monkeypatch, tmp_path, template):
    payload = {"description": "A red square.", "scores": {}, "average_score": 8}
    calls = _gemini_endpoint(monkeypatch, json.dumps(payload))
    provider = _gemini_provider(image_prompt_template=template)

    result = provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert result.parsed == payload
    assert result.is_persistable
    assert calls[0]["config"].response_mime_type == "application/json"
    assert calls[0]["config"].response_schema is not None


def test_gemini_bbox_json_template_keeps_json_without_rating_schema(monkeypatch, tmp_path):
    calls = _gemini_endpoint(monkeypatch, json.dumps(BBOX_PAYLOAD))
    provider = _gemini_provider(image_prompt_template="bbox_json")

    result = provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert result.parsed == BBOX_PAYLOAD
    assert result.is_persistable
    assert calls[0]["config"].response_mime_type == "application/json"
    assert calls[0]["config"].response_schema is None


@pytest.mark.parametrize("payload", [
    pytest.param([{"label": "red square", "bbox": [0, 0, 100, 100]}], id="object-array"),
    pytest.param(["red square", "white background"], id="string-array"),
    pytest.param([], id="empty-array"),
    pytest.param("red square", id="string"),
    pytest.param("", id="empty-string"),
    pytest.param(42, id="number"),
    pytest.param(0, id="zero"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(None, id="null"),
])
def test_gemini_custom_nonobject_json_preserves_sidecar_and_lance_update(monkeypatch, tmp_path, payload):
    response_text = json.dumps(payload)
    calls = _gemini_endpoint(monkeypatch, response_text)
    provider = _gemini_provider(image_prompt_template="custom_json")
    provider.ctx.config["prompts"]["json_system"] = "Return valid JSON."
    provider.ctx.config["prompts"]["image_templates"]["custom_json"] = {
        "system_key": "json_system", "output": "json",
    }
    source = _image(tmp_path)

    response = provider.execute(str(source), "image/jpeg", "offline-hash")
    result = _publish(source, response, template="custom_json")

    assert response.raw == response_text
    assert response.parsed is None
    assert result.output.is_persistable
    assert json.loads(source.with_suffix(".txt").read_text(encoding="utf-8")) == payload
    updates = _build_caption_updates([result])
    assert len(updates) == 1
    assert updates[0].uri == str(source)
    assert json.loads(updates[0].caption) == payload
    assert len(calls) == 1
    assert calls[0]["config"].response_mime_type == "application/json"
    assert calls[0]["config"].response_schema is None


@pytest.mark.parametrize("template", ["", "bbox_json"])
def test_gemini_malformed_json_contract_retries_instead_of_reporting_filtering(monkeypatch, tmp_path, template):
    calls = _gemini_endpoint(monkeypatch, "not valid JSON")
    provider = _gemini_provider(image_prompt_template=template)

    with pytest.raises(json.JSONDecodeError):
        provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert len(calls) == 2
    assert "filtered" not in provider.ctx.console.file.getvalue().lower()


@pytest.mark.parametrize("filtered", [False, True])
def test_gemini_empty_or_filtered_response_does_not_become_persistable(monkeypatch, tmp_path, filtered):
    calls = _gemini_endpoint(monkeypatch, "", filtered=filtered)
    provider = _gemini_provider()

    result = provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert not result.is_persistable
    assert result.raw == ""
    assert len(calls) == 1


def test_gemini_pair_prompt_takes_precedence_over_text_template(monkeypatch, tmp_path):
    calls = _gemini_endpoint(monkeypatch, '{"prompt": "The square changed from white to red."}')
    source = _image(tmp_path)
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    Image.new("RGB", (32, 32), "white").save(pair_dir / source.name)
    provider = _gemini_provider(pair_dir=str(pair_dir), image_prompt_template="danbooru_tags")

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.raw == "The square changed from white to red."
    assert result.parsed is None
    assert result.is_persistable
    assert calls[0]["config"].response_schema is not None
    assert len(calls[0]["contents"]) == 3


def test_gemini_missing_pair_keeps_the_effective_text_template(monkeypatch, tmp_path):
    calls = _gemini_endpoint(monkeypatch, "red square, geometric shape")
    provider = _gemini_provider(pair_dir=str(tmp_path / "missing-pairs"), image_prompt_template="danbooru_tags")

    result = provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert result.raw == "red square, geometric shape"
    assert result.is_persistable
    assert calls[0]["config"].response_schema is None
    assert calls[0]["config"].response_mime_type == "text/plain"


def test_gemini_image_task_preserves_plaintext(monkeypatch, tmp_path):
    calls = _gemini_endpoint(monkeypatch, "The requested image is ready.")
    provider = _gemini_provider(gemini_task="edit", image_prompt_template="bbox_json")

    result = provider.execute(str(_image(tmp_path)), "image/jpeg", "offline-hash")

    assert result.raw == "The requested image is ready."
    assert result.parsed is None
    assert calls[0]["model"] == "gemini-3-pro-image"
    assert calls[0]["config"].response_schema is None


@pytest.mark.parametrize("mime", ["audio/wav", "video/mp4"])
def test_gemini_media_keeps_srt_contract(monkeypatch, tmp_path, mime):
    from types import SimpleNamespace

    import module.providers.gemini_utils as gemini_utils

    srt = "1\n00:00:00,000 --> 00:00:01,000\nA red square.\n"
    calls = _gemini_endpoint(monkeypatch, f"```srt\n{srt}```")
    source = tmp_path / ("input.wav" if mime.startswith("audio") else "input.mp4")
    source.write_bytes(b"offline media content")
    monkeypatch.setattr(gemini_utils, "upload_or_get", lambda **kwargs: (
        True, [SimpleNamespace(uri="https://offline.invalid/files/video")],
    ))
    provider = _gemini_provider(image_prompt_template="bbox_json")

    result = provider.execute(str(source), mime, "offline-hash")

    assert result.raw.strip() == srt.strip()
    assert result.parsed is None
    assert result.is_persistable
    assert calls[0]["config"].response_schema is None
