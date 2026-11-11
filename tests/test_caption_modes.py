"""Structured short/long captions must agree at every persistence boundary."""

import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from rich.console import Console

from module.caption_pipeline.postprocess import postprocess_caption_content
from module.providers.base import CaptionResult, ProviderContext
from module.providers.cloud_vlm.openai_compatible import OpenAICompatibleProvider
from utils.output_writer import write_caption_output


SHORT = "A pink bottle."
LONG = "A glossy pink glass bottle with a flower-shaped stopper."
SECTIONED = f"###Short: {SHORT}\n\n###Long: {LONG}"
EXPECTED = {
    "short": {"short_description": SHORT},
    "long": {"long_description": LONG},
    "all": {"short_description": SHORT, "long_description": LONG},
}


def _args(mode):
    return SimpleNamespace(mode=mode, pair_dir="", image_prompt_template="", ocr_model="", document_image=False)


def _console():
    return Console(file=io.StringIO())


@pytest.mark.parametrize("mode", ["short", "long", "all"])
@pytest.mark.parametrize("field", ["description", "long_description", "short_description", "long", "short"])
def test_provider_splits_embedded_sections_before_mode_filtering(mode, field):
    provider = OpenAICompatibleProvider(ProviderContext(console=_console(), args=_args(mode)))
    payload = {field: SECTIONED, "tags": ["glass"], "scores": {"quality": 9}}
    original = deepcopy(payload)

    result = provider._filter_by_mode(payload, mode)

    assert result == {**EXPECTED[mode], "tags": ["glass"], "scores": {"quality": 9}}
    assert payload == original


@pytest.mark.parametrize("mode", ["short", "long", "all"])
@pytest.mark.parametrize("representation", ["dict", "json", "parsed_result", "raw_result", "fenced", "template_fenced"])
def test_postprocess_normalizes_structured_modes_for_all_output_boundaries(tmp_path, mode, representation):
    payload = {"description": SECTIONED, "tags": ["glass"], "scores": {"quality": 9}}
    original = deepcopy(payload)
    encoded = json.dumps(payload)
    metadata = {"provider": "test", "structured": True}
    args = _args(mode)
    if representation == "dict":
        output = payload
    elif representation == "json":
        output = encoded
    elif representation == "parsed_result":
        output = CaptionResult(raw=encoded, parsed=payload, metadata=metadata)
    elif representation == "raw_result":
        output = CaptionResult(raw=encoded, metadata=metadata)
    else:
        output = f"```json\n{encoded}\n```"
        if representation == "template_fenced":
            args.image_prompt_template = "bbox_json"
    expected = {**EXPECTED[mode], "tags": ["glass"], "scores": {"quality": 9}}
    expected_text = SHORT if mode == "short" else LONG
    source = tmp_path / "example.PNG"

    result = postprocess_caption_content(output, source, args, _console())

    assert result.parsed == expected
    assert result.text == expected_text
    assert json.loads(result.raw) == expected
    assert json.loads(result.to_dataset_caption()) == expected
    text_path, json_path = write_caption_output(source, result, "image/png")
    assert text_path.read_text(encoding="utf-8") == expected_text
    assert json.loads(json_path.read_text(encoding="utf-8")) == expected
    assert payload == original
    if isinstance(output, CaptionResult):
        assert result.metadata == metadata
        assert output.raw == encoded


@pytest.mark.parametrize("text", [
    f"### SHORT: {SHORT}\n### long: {LONG}",
    f"### Short Description:\n{SHORT}\n\n### Long Description:\n{LONG}",
    f"### short_description\r\n{SHORT}\r\n### long_description\r\n{LONG}",
    f"###Long: {LONG}\n###Short: {SHORT}",
    f"### Short\uff1a {SHORT}\n### Long\uff1a {LONG}",
])
def test_postprocess_recognizes_explicit_heading_variants(text):
    result = postprocess_caption_content({"description": text}, "example.png", _args("all"), _console())

    assert result.parsed == EXPECTED["all"]


@pytest.mark.parametrize("mode", ["short", "long", "all"])
def test_explicit_fields_win_over_embedded_sections_and_aliases(mode):
    payload = {
        "description": SECTIONED,
        "short": "alias short",
        "short_description": "explicit short",
        "long": "alias long",
        "long_description": "explicit long",
        "nested": {"description": SECTIONED},
    }
    original = deepcopy(payload)
    expected = {"nested": {"description": SECTIONED}}
    if mode != "long":
        expected["short_description"] = "explicit short"
    if mode != "short":
        expected["long_description"] = "explicit long"

    result = postprocess_caption_content(payload, "example.png", _args(mode), _console())

    assert result.parsed == expected
    assert payload == original


def test_split_canonical_fields_take_precedence_over_aliases_and_other_sections():
    payload = {
        "short_description": f"###Short: {SHORT}\n###Long: Incidental long.",
        "long_description": f"###Short: Incidental short.\n###Long: {LONG}",
        "short": "Alias short.",
        "long": "Alias long.",
    }

    result = postprocess_caption_content(payload, "example.png", _args("all"), _console())

    assert result.parsed == EXPECTED["all"]


@pytest.mark.parametrize("mode", ["short", "long", "all"])
def test_embedded_sections_replace_empty_caption_fields_not_custom_metadata(mode):
    payload = {"description": SECTIONED, "short_description": "  ", "long_description": None, "short": {"pixels": 800}}

    result = postprocess_caption_content(payload, "example.png", _args(mode), _console())

    assert result.parsed == {**EXPECTED[mode], "short": {"pixels": 800}}
    assert result.text == (SHORT if mode == "short" else LONG)


def test_short_mode_does_not_write_legacy_description_over_split_short_caption():
    payload = {"long_description": SECTIONED, "description": "An alternate caption.", "tags": ["glass"]}

    result = postprocess_caption_content(payload, "example.png", _args("short"), _console())

    assert result.parsed == {**EXPECTED["short"], "tags": ["glass"]}
    assert result.text == SHORT


def test_postprocess_keeps_custom_schema_with_a_structured_description():
    payload = {"description": {"language": "en"}, "long_description": SECTIONED}

    result = postprocess_caption_content(payload, "example.png", _args("short"), _console())

    assert result.parsed == payload


@pytest.mark.parametrize("mode", ["short", "long", "all"])
@pytest.mark.parametrize("payload", [
    {"description": "An ordinary caption.", "scores": {"quality": 9}},
    {"description": "An ordinary caption.", "short": {"pixels": 800}, "long": {"pixels": 1024}},
    {"custom": {"description": SECTIONED}, "tags": ["glass"]},
    {"description": {"short": "nested", "long": "custom structure"}},
    {"description": ["custom", "values"]},
    {"description": "### Short: Only one section."},
    {"description": f"### Short:\n### Long: {LONG}"},
    {"description": f"Preamble that must not be lost.\n{SECTIONED}"},
    {"description": f"{SECTIONED}\n### Short: Repeated section."},
    {"description": "A sign says ###Short: small and ###Long: tall."},
    {"description": "### Short story\nOrdinary prose.\n### Long road\nMore prose."},
])
def test_postprocess_preserves_plain_nested_or_ambiguous_captions(payload, mode):
    original = deepcopy(payload)

    result = postprocess_caption_content(payload, "example.png", _args(mode), _console())

    assert result.parsed == original
    assert payload == original


def test_incomplete_section_in_explicit_field_is_not_discarded():
    payload = {"long_description": "###Short: An incomplete caption.", "tags": ["glass"]}

    result = postprocess_caption_content(payload, "example.png", _args("short"), _console())

    assert result.parsed == payload
    assert result.is_persistable


@pytest.mark.parametrize("extension", ["mp4", "wav", "pdf", "txt"])
def test_postprocess_does_not_normalize_non_image_payloads(extension):
    payload = {"description": SECTIONED, "short": "custom short", "long": "custom long"}

    result = postprocess_caption_content(payload, f"example.{extension}", _args("long"), _console())

    assert result.parsed == payload


@pytest.mark.parametrize("payload", [
    {"custom": {"description": SECTIONED}},
    {"description": "An ordinary caption.", "short": {"pixels": 800}},
])
def test_fenced_custom_json_without_template_remains_raw(payload):
    encoded = json.dumps(payload)
    output = f"```json\n{encoded}\n```"

    result = postprocess_caption_content(output, "example.png", _args("long"), _console())

    assert result.parsed is None
    assert result.raw == output


@pytest.mark.parametrize("template_id", ["", "bbox_json"])
def test_sectioned_fenced_json_preserves_backticks_inside_caption(template_id):
    long = f"{LONG} The label reads ```code```."
    encoded = json.dumps({"description": f"###Short: {SHORT}\n###Long: {long}"})
    args = _args("long")
    args.image_prompt_template = template_id

    result = postprocess_caption_content(f"```json\n{encoded}\n```", "example.png", args, _console())

    assert result.parsed == {"long_description": long}
    assert result.text == long


def test_provider_preserves_legacy_plain_description_alias_for_all_mode():
    provider = OpenAICompatibleProvider(ProviderContext(console=_console(), args=_args("all")))

    assert provider._filter_by_mode({"description": "Plain caption."}, "all") == {"long_description": "Plain caption."}


@pytest.mark.parametrize("split_sections", [False, True])
def test_provider_preserves_legacy_fields_when_not_splitting_sections(split_sections):
    provider = OpenAICompatibleProvider(ProviderContext(console=_console(), args=_args("all")))
    payload = {"description": "An ordinary caption.", "long_description": LONG, "long": {"frames": 5}}

    assert provider._filter_by_mode(payload, "all", split_sections=split_sections) == payload
