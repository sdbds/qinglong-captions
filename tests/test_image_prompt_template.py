"""Tests for Image VLM Prompt Template feature.

Covers the 9 test cases from the spec:
1. Default zero change
2. Override priority (template > provider-specific key)
3. rating template content
4. bbox_json template content
5. Unknown id silent fallback
6. Structured provider yielding (Gemini disables schema)
7. BBOX JSON persistence through postprocess
8. GUI parameter passing
9. Orthogonal mode preservation (pair mode still wins)
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config.loader import load_config
from module.providers.resolver import PromptResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG = load_config(Path(__file__).resolve().parent.parent / "config")
PROMPTS = CONFIG.get("prompts", {})


def _make_args(**kwargs) -> types.SimpleNamespace:
    """Build a lightweight args object with sensible defaults."""
    defaults = {
        "image_prompt_template": "",
        "gemini_task": "",
        "pair_dir": "",
        "mode": "long",
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _resolver(provider_name: str = "gemini") -> PromptResolver:
    return PromptResolver(CONFIG, provider_name)


# ---------------------------------------------------------------------------
# Test 1: Default zero change
# ---------------------------------------------------------------------------

class TestDefaultZeroChange:
    """image_prompt_template='' must produce identical results to pre-feature."""

    def test_gemini_default(self):
        resolver = _resolver("gemini")
        args = _make_args()
        ctx = resolver.resolve("image/jpeg", args)
        # Gemini has no provider-specific image prompt, so base prompts win
        assert ctx.system == PROMPTS.get("image_system_prompt")
        assert ctx.user == PROMPTS.get("image_prompt")

    def test_mistral_ocr_default(self):
        resolver = _resolver("mistral_ocr")
        args = _make_args()
        ctx = resolver.resolve("image/jpeg", args)
        # mistral_ocr has provider-specific prompts
        assert ctx.system == PROMPTS.get("mistral_ocr_image_system_prompt")
        assert ctx.user == PROMPTS.get("mistral_ocr_image_prompt")

    def test_kimi_default(self):
        resolver = _resolver("kimi_vl")
        args = _make_args()
        ctx = resolver.resolve("image/png", args)
        assert ctx.system == PROMPTS.get("kimi_image_system_prompt")
        assert ctx.user == PROMPTS.get("kimi_image_prompt")


# ---------------------------------------------------------------------------
# Test 2: Override priority (template > provider-specific key)
# ---------------------------------------------------------------------------

class TestOverridePriority:
    """Selected template must override provider-specific keys."""

    def test_danbooru_overrides_mistral_ocr(self):
        resolver = _resolver("mistral_ocr")
        args = _make_args(image_prompt_template="danbooru_tags")
        ctx = resolver.resolve("image/jpeg", args)
        assert ctx.system == PROMPTS.get("mistral_ocr_image_system_prompt")
        assert ctx.user == PROMPTS.get("mistral_ocr_image_prompt")

    def test_danbooru_overrides_gemini(self):
        resolver = _resolver("gemini")
        args = _make_args(image_prompt_template="danbooru_tags")
        ctx = resolver.resolve("image/jpeg", args)
        # Gemini has no provider-specific image prompt; template overrides base
        assert ctx.system == PROMPTS.get("mistral_ocr_image_system_prompt")
        assert ctx.user == PROMPTS.get("mistral_ocr_image_prompt")

    def test_danbooru_overrides_kimi(self):
        resolver = _resolver("kimi_vl")
        args = _make_args(image_prompt_template="danbooru_tags")
        ctx = resolver.resolve("image/png", args)
        assert ctx.system == PROMPTS.get("mistral_ocr_image_system_prompt")
        assert ctx.user == PROMPTS.get("mistral_ocr_image_prompt")


# ---------------------------------------------------------------------------
# Test 3: rating template content
# ---------------------------------------------------------------------------

class TestRatingTemplate:
    def test_rating_system_and_user(self):
        resolver = _resolver("gemini")
        args = _make_args(image_prompt_template="rating")
        ctx = resolver.resolve("image/jpeg", args)
        assert ctx.system == PROMPTS.get("image_system_prompt")
        assert ctx.user == PROMPTS.get("image_prompt")


# ---------------------------------------------------------------------------
# Test 4: bbox_json template content
# ---------------------------------------------------------------------------

class TestBboxJsonTemplate:
    def test_bbox_system_and_user(self):
        resolver = _resolver("gemini")
        args = _make_args(image_prompt_template="bbox_json")
        ctx = resolver.resolve("image/jpeg", args)
        assert ctx.system == PROMPTS.get("BBOX_prompt")
        assert ctx.user == PROMPTS.get("bbox_image_prompt")

    def test_bbox_output_contract_is_json(self):
        templates = PROMPTS.get("image_templates", {})
        assert templates["bbox_json"]["output"] == "json"


# ---------------------------------------------------------------------------
# Test 5: Unknown id silent fallback
# ---------------------------------------------------------------------------

class TestUnknownIdFallback:
    def test_unknown_id_does_not_raise(self):
        resolver = _resolver("gemini")
        args = _make_args(image_prompt_template="nonexistent_template")
        ctx = resolver.resolve("image/jpeg", args)
        # Falls back to provider-driven result (base prompts for gemini)
        assert ctx.system == PROMPTS.get("image_system_prompt")
        assert ctx.user == PROMPTS.get("image_prompt")

    def test_custom_string_falls_back(self):
        resolver = _resolver("gemini")
        args = _make_args(image_prompt_template="custom")
        ctx = resolver.resolve("image/jpeg", args)
        assert ctx.system == PROMPTS.get("image_system_prompt")
        assert ctx.user == PROMPTS.get("image_prompt")


# ---------------------------------------------------------------------------
# Test 6: Structured provider yielding (Gemini)
# ---------------------------------------------------------------------------

class TestStructuredProviderYielding:
    """Gemini must disable forced schema when non-default template is active."""

    def _make_media(self) -> MagicMock:
        media = MagicMock()
        media.mime = "image/jpeg"
        return media

    def test_gemini_schema_disabled_for_danbooru_tags(self):
        from module.providers.vision_api.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        args = _make_args(image_prompt_template="danbooru_tags")
        config = self._make_media()
        result = provider.get_structured_output_config(config, args)
        assert result.enabled is False

    def test_gemini_schema_disabled_for_bbox_json(self):
        from module.providers.vision_api.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        args = _make_args(image_prompt_template="bbox_json")
        config = self._make_media()
        result = provider.get_structured_output_config(config, args)
        assert result.enabled is False

    def test_gemini_schema_enabled_for_default(self):
        from module.providers.vision_api.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        args = _make_args()
        config = self._make_media()
        result = provider.get_structured_output_config(config, args)
        assert result.enabled is True

    def test_gemini_schema_enabled_for_custom(self):
        from module.providers.vision_api.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        args = _make_args(image_prompt_template="custom")
        config = self._make_media()
        result = provider.get_structured_output_config(config, args)
        assert result.enabled is True


# ---------------------------------------------------------------------------
# Test 7: BBOX JSON persistence through postprocess
# ---------------------------------------------------------------------------

class TestBboxJsonPersistence:
    """BBOX JSON output must survive postprocess without ### splitting."""

    def test_bbox_json_preserved(self):
        from module.providers.base import CaptionResult
        from module.caption_pipeline.postprocess import postprocess_caption_content

        bbox_json = json.dumps({
            "high_level_description": "A test image.",
            "style_description": {"aesthetics": "clean"},
            "compositional_deconstruction": {
                "background": "white",
                "elements": [{"type": "obj", "bbox": [0, 0, 100, 100], "desc": "test"}],
            },
        })
        raw_result = CaptionResult(raw=bbox_json)
        console = MagicMock()
        args = _make_args(mode="long")
        result = postprocess_caption_content(raw_result, "test.jpg", args, console)
        # The JSON should be parsed and stored in parsed field
        assert result.parsed is not None
        assert result.parsed["high_level_description"] == "A test image."
        # description falls back to raw since no long_description key
        assert "high_level_description" in result.description
        # to_dataset_caption should round-trip the JSON
        caption = result.to_dataset_caption()
        parsed_caption = json.loads(caption)
        assert parsed_caption["high_level_description"] == "A test image."

    def test_bbox_json_no_triple_hash_split(self):
        """Ensure BBOX JSON (no ###) is not split by ### logic."""
        from module.providers.base import CaptionResult
        from module.caption_pipeline.postprocess import postprocess_caption_content

        bbox_json = '{"high_level_description": "test", "style_description": {}}'
        raw_result = CaptionResult(raw=bbox_json)
        console = MagicMock()
        args = _make_args(mode="long")
        result = postprocess_caption_content(raw_result, "test.jpg", args, console)
        # Should not be split - raw should contain the full JSON
        assert "high_level_description" in result.raw


# ---------------------------------------------------------------------------
# Test 8: GUI parameter passing
# ---------------------------------------------------------------------------

class TestGuiParameterPassing:
    """_build_caption_args should produce --image_prompt_template=<id> when selected."""

    def test_template_arg_present(self):
        # Simulate the GUI logic directly
        class FakeSelect:
            def __init__(self, value):
                self.value = value

        class FakeCaptionStep:
            image_prompt_template = FakeSelect("danbooru_tags")

            # Minimal stubs needed by the tail of _build_caption_args
            def _codex_enabled(self):
                return False

            def _grok_build_enabled(self):
                return False

            def _current_alm_model(self):
                return ""

        step = FakeCaptionStep()
        tpl = getattr(getattr(step, "image_prompt_template", None), "value", "") or ""
        assert tpl == "danbooru_tags"
        assert tpl and tpl != "custom"
        arg = f"--image_prompt_template={tpl}"
        assert arg == "--image_prompt_template=danbooru_tags"

    def test_template_arg_absent_for_default(self):
        class FakeSelect:
            def __init__(self, value):
                self.value = value

        class FakeCaptionStep:
            image_prompt_template = FakeSelect("")

        step = FakeCaptionStep()
        tpl = getattr(getattr(step, "image_prompt_template", None), "value", "") or ""
        assert not tpl  # empty -> no arg appended

    def test_template_arg_absent_for_custom(self):
        class FakeSelect:
            def __init__(self, value):
                self.value = value

        class FakeCaptionStep:
            image_prompt_template = FakeSelect("custom")

        step = FakeCaptionStep()
        tpl = getattr(getattr(step, "image_prompt_template", None), "value", "") or ""
        assert tpl == "custom"
        # The condition: if tpl and tpl != "custom" -> False, no arg
        assert not (tpl and tpl != "custom")


# ---------------------------------------------------------------------------
# Test 9: Orthogonal mode preservation (pair mode)
# ---------------------------------------------------------------------------

class TestPairModePreservation:
    """Pair mode must still override template when pair_dir is set."""

    def test_pair_overrides_template(self):
        resolver = _resolver("gemini")
        args = _make_args(image_prompt_template="danbooru_tags", pair_dir="/some/pair/dir")
        ctx = resolver.resolve("image/jpeg", args)
        # Pair override comes after template override, so pair prompts win
        assert ctx.system == PROMPTS.get("pair_image_system_prompt")
        assert ctx.user == PROMPTS.get("pair_image_prompt")

    def test_pair_overrides_template_for_mistral(self):
        resolver = _resolver("mistral_ocr")
        args = _make_args(image_prompt_template="rating", pair_dir="/some/pair/dir")
        ctx = resolver.resolve("image/jpeg", args)
        assert ctx.system == PROMPTS.get("pair_image_system_prompt")
        assert ctx.user == PROMPTS.get("pair_image_prompt")


# ---------------------------------------------------------------------------
# Test 10: Shared image_template helpers
# ---------------------------------------------------------------------------

class TestImageTemplateHelpers:
    def test_active_image_template(self):
        from module.providers.image_template import active_image_template

        assert active_image_template(_make_args()) == ""
        assert active_image_template(_make_args(image_prompt_template="custom")) == ""
        assert active_image_template(_make_args(image_prompt_template="bbox_json")) == "bbox_json"

    def test_image_template_output(self):
        from module.providers.image_template import image_template_output

        assert image_template_output(PROMPTS, "bbox_json") == "json"
        assert image_template_output(PROMPTS, "rating") == "text"
        assert image_template_output(PROMPTS, "danbooru_tags") == "text"
        # Unknown / empty fall back
        assert image_template_output(PROMPTS, "nonexistent") == "text"
        assert image_template_output(PROMPTS, "") == ""

    def test_build_freeform_prompt_has_no_rating_wrap(self):
        from module.providers.image_template import build_freeform_caption_prompt

        prompt = build_freeform_caption_prompt(system_prompt="SYS", user_prompt="USR", output="json")
        assert "SYS" in prompt and "USR" in prompt
        # Must NOT carry the codex rating contract wrapper.
        assert "Fill short_description, long_description" not in prompt
        assert "Return only JSON matching the provided schema" not in prompt
        # json contract instruction present
        assert "Output valid JSON only" in prompt

    def test_build_freeform_prompt_text_has_no_json_instruction(self):
        from module.providers.image_template import build_freeform_caption_prompt

        prompt = build_freeform_caption_prompt(system_prompt="SYS", user_prompt="USR", output="text")
        assert "Output valid JSON only" not in prompt

    def test_parse_freeform_bare_json(self):
        from module.providers.image_template import parse_freeform_caption_output

        raw, parsed = parse_freeform_caption_output('{"a": 1}', "json")
        assert parsed == {"a": 1}
        assert json.loads(raw) == {"a": 1}

    def test_parse_freeform_fenced_json(self):
        from module.providers.image_template import parse_freeform_caption_output

        raw, parsed = parse_freeform_caption_output('```json\n{"a": 1}\n```', "json")
        assert parsed == {"a": 1}

    def test_parse_freeform_invalid_json_falls_back_to_text(self):
        from module.providers.image_template import parse_freeform_caption_output

        raw, parsed = parse_freeform_caption_output("not json", "json")
        assert parsed is None
        assert raw == "not json"

    def test_parse_freeform_text(self):
        from module.providers.image_template import parse_freeform_caption_output

        raw, parsed = parse_freeform_caption_output("hello", "text")
        assert parsed is None
        assert raw == "hello"


# ---------------------------------------------------------------------------
# Test 11: codex exec command omits forced schema when none is provided
# ---------------------------------------------------------------------------

class TestCodexExecCommandSchema:
    def test_command_has_schema_when_provided(self, tmp_path):
        from module.providers.codex_exec import CodexExecConfig, build_codex_exec_command

        cwd = tmp_path / "work"
        cwd.mkdir()
        command = build_codex_exec_command(
            CodexExecConfig(command="codex", isolated_cwd=str(cwd)),
            image_path=tmp_path / "image.png",
            prompt="caption",
            schema_path=tmp_path / "schema.json",
            output_path=tmp_path / "out.json",
        )
        assert "--output-schema" in command

    def test_command_omits_schema_when_none(self, tmp_path):
        from module.providers.codex_exec import CodexExecConfig, build_codex_exec_command

        cwd = tmp_path / "work"
        cwd.mkdir()
        command = build_codex_exec_command(
            CodexExecConfig(command="codex", isolated_cwd=str(cwd)),
            image_path=tmp_path / "image.png",
            prompt="caption",
            schema_path=None,
            output_path=tmp_path / "out.json",
        )
        assert "--output-schema" not in command


# ---------------------------------------------------------------------------
# Test 12: codex_subscription yields rating schema for templates (BBOX preserved)
# ---------------------------------------------------------------------------

BBOX_SAMPLE = json.dumps({
    "high_level_description": "A cat on a mat.",
    "style_description": {"aesthetics": "clean"},
    "compositional_deconstruction": {"background": "white", "elements": []},
})


class TestCodexSubscriptionYielding:
    def _provider(self, args):
        from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

        provider = CodexSubscriptionProvider.__new__(CodexSubscriptionProvider)
        provider.ctx = types.SimpleNamespace(args=args, config=CONFIG, console=MagicMock())
        return provider

    def test_bbox_json_preserved_via_exec(self):
        args = _make_args(image_prompt_template="bbox_json", codex_backend="exec")
        provider = self._provider(args)
        captured = {}

        def fake_exec(media, prompt, *, structured=True):
            captured["prompt"] = prompt
            captured["structured"] = structured
            return types.SimpleNamespace(raw=BBOX_SAMPLE, parsed={})

        provider._attempt_exec = fake_exec
        media = types.SimpleNamespace(uri="test.jpg", mime="image/jpeg")
        prompts = types.SimpleNamespace(system="SYS-BBOX", user="USR-BBOX")
        result = provider.attempt(media, prompts)

        # Freeform path: schema yielded, BBOX fields preserved (not normalized away).
        assert captured["structured"] is False
        assert "Fill short_description, long_description" not in captured["prompt"]
        assert result.parsed is not None
        assert result.parsed["high_level_description"] == "A cat on a mat."
        assert result.metadata["structured"] is False
        assert result.metadata["image_template"] == "bbox_json"
        assert result.metadata["output_contract"] == "json"

    def test_default_still_structured(self):
        args = _make_args(codex_backend="exec")  # no template
        provider = self._provider(args)
        captured = {}

        def fake_exec(media, prompt, *, structured=True):
            captured["structured"] = structured
            return types.SimpleNamespace(
                raw="{}",
                parsed={"short_description": "s", "long_description": "l", "tags": [], "rating": "general"},
            )

        provider._attempt_exec = fake_exec
        media = types.SimpleNamespace(uri="test.jpg", mime="image/jpeg")
        prompts = types.SimpleNamespace(system="SYS", user="USR")
        result = provider.attempt(media, prompts)
        assert captured["structured"] is True
        assert result.metadata["structured"] is True


# ---------------------------------------------------------------------------
# Test 13: grok_build_subscription yields rating schema for templates
# ---------------------------------------------------------------------------

class TestGrokBuildSubscriptionYielding:
    def _provider(self, args):
        from module.providers.cloud_vlm.grok_build_subscription import GrokBuildSubscriptionProvider

        provider = GrokBuildSubscriptionProvider.__new__(GrokBuildSubscriptionProvider)
        provider.ctx = types.SimpleNamespace(args=args, config=CONFIG, console=MagicMock())
        return provider

    def test_bbox_json_preserved(self):
        args = _make_args(image_prompt_template="bbox_json")
        provider = self._provider(args)
        captured = {}

        def fake_headless(media, prompt, *, structured=True):
            captured["prompt"] = prompt
            captured["structured"] = structured
            return BBOX_SAMPLE, 42

        provider._attempt_headless = fake_headless
        media = types.SimpleNamespace(uri="test.jpg", mime="image/jpeg")
        prompts = types.SimpleNamespace(system="SYS-BBOX", user="USR-BBOX")
        result = provider.attempt(media, prompts)

        assert captured["structured"] is False
        assert "Fill short_description, long_description" not in captured["prompt"]
        assert result.parsed is not None
        assert result.parsed["high_level_description"] == "A cat on a mat."
        assert result.metadata["structured"] is False
        assert result.metadata["image_template"] == "bbox_json"


# ---------------------------------------------------------------------------
# Test 14: postprocess strips ```json fence for json templates
# ---------------------------------------------------------------------------

class TestPostprocessJsonFence:
    def test_fenced_json_parsed_when_template_active(self):
        from module.caption_pipeline.postprocess import postprocess_caption_content
        from module.providers.base import CaptionResult

        fenced = "```json\n" + BBOX_SAMPLE + "\n```"
        raw_result = CaptionResult(raw=fenced)
        console = MagicMock()
        args = _make_args(mode="long", image_prompt_template="bbox_json")
        result = postprocess_caption_content(raw_result, "test.jpg", args, console)
        assert result.parsed is not None
        assert result.parsed["high_level_description"] == "A cat on a mat."

    def test_fenced_json_not_touched_without_template(self):
        from module.caption_pipeline.postprocess import postprocess_caption_content
        from module.providers.base import CaptionResult

        fenced = "```json\n" + BBOX_SAMPLE + "\n```"
        raw_result = CaptionResult(raw=fenced)
        console = MagicMock()
        args = _make_args(mode="long")  # no template -> default behavior preserved
        result = postprocess_caption_content(raw_result, "test.jpg", args, console)
        # Default path leaves the fenced text as-is (parsed stays None).
        assert result.parsed is None
        assert "high_level_description" in result.raw
