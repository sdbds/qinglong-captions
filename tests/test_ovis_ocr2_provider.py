import base64
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from rich.console import Console

from module.providers.backends import RuntimeBackendConfig
from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
from module.providers.ocr import ovis_ocr2 as ovis_module
from module.providers.ocr.ovis_ocr2 import (
    OvisOCR2Provider,
    _clean_truncated_repeats,
    _DirectPageInferencer,
    _OpenAIPageInferencer,
    _prefix_visual_region_paths,
    _process_visual_regions,
)
from module.providers.ocr.ovis_ocr2_contract import OVIS_OCR2_DEFAULT_PROMPT


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def _provider(tmp_path: Path, section=None, *, args=None, prompts=None) -> OvisOCR2Provider:
    config = {"ovis_ocr2": dict(section or {})}
    if prompts is not None:
        config["prompts"] = prompts
    return OvisOCR2Provider(
        ProviderContext(
            console=_console(),
            config=config,
            args=args or SimpleNamespace(),
        )
    )


def _media(uri: Path, mime: str, output_dir: Path) -> MediaContext:
    return MediaContext(
        uri=str(uri),
        mime=mime,
        sha256hash="",
        modality=MediaModality.DOCUMENT if mime == "application/pdf" else MediaModality.IMAGE,
        extras={"output_dir": output_dir},
    )


class _SequenceInferencer:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def infer_page(self, image, prompt):
        self.calls.append((image.mode, image.size, prompt))
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return output


class _CharacterTokenProcessor:
    def __init__(self):
        self.decoded_widths = []

    def batch_decode(self, token_ids, **kwargs):
        values = token_ids[0].tolist()
        self.decoded_widths.append(len(values))
        return ["".join(chr(value) for value in values)]


def test_provider_uses_shared_official_prompt_contract():
    assert OvisOCR2Provider.default_prompt == OVIS_OCR2_DEFAULT_PROMPT
    assert OVIS_OCR2_DEFAULT_PROMPT.startswith("\nExtract all readable content")
    assert '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />' in OVIS_OCR2_DEFAULT_PROMPT


def test_clean_truncated_repeats_keeps_text_shorter_than_official_threshold():
    text = "x" * 7899 + "0123456789ABCDEFGHIJ" * 5
    assert len(text) == 7999
    assert _clean_truncated_repeats(text) == text


def test_clean_truncated_repeats_collapses_official_boundary_to_one_unit():
    unit = "0123456789ABCDEFGHIJ"
    text = "x" * 7900 + unit * 5

    assert len(text) == 8000
    assert _clean_truncated_repeats(text) == "x" * 7900 + unit


@pytest.mark.parametrize(
    "unit,repeats",
    [
        ("0123456789ABCDEFGHIJKLMNO", 4),
        ("0123456789ABCDEFGHI", 5),
        ("".join(chr(0x100 + index) for index in range(201)), 5),
    ],
)
def test_clean_truncated_repeats_respects_repeat_count_character_and_period_limits(unit, repeats):
    repeated = unit * repeats
    text = "#" * (8000 - len(repeated)) + repeated
    assert _clean_truncated_repeats(text) == text


def test_thinking_marker_truncation_discards_the_generated_control_tail():
    text = "valid markdown\n\n<think>\n\n</think>\n\n## 1\n## 2"

    assert ovis_module._truncate_at_thinking_marker(text) == "valid markdown"
    assert ovis_module._truncate_at_thinking_marker("inline <think> text") == "inline <think> text"
    assert ovis_module._truncate_at_thinking_marker("literal line\n<think>\nbody") == "literal line\n<think>\nbody"
    assert (
        ovis_module._truncate_at_thinking_marker("literal block\n<think>\nreasoning\n</think>")
        == "literal block\n<think>\nreasoning\n</think>"
    )


def test_find_repeated_tail_selects_shortest_qualifying_period():
    match = ovis_module._find_repeated_tail(
        "header#" + "ab" * 100,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )

    assert match is not None
    assert match.period_len == 2
    assert match.matched_chars == 200
    assert match.repeat_times == 100
    assert match.trailing_chars == 0


def test_collapse_repeated_tail_preserves_partial_period():
    text = "header#" + "abc" * 70 + "ab"
    match = ovis_module._find_repeated_tail(
        text,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )

    assert match is not None
    assert match.period_len == 3
    assert match.trailing_chars == 2
    assert ovis_module._collapse_repeated_tail(text, match) == "header#abcab"


def test_early_match_requires_both_repeat_and_character_thresholds():
    seven_long_units = "".join(chr(0x400 + index) for index in range(30)) * 7
    eight_units = "".join(chr(0x500 + index) for index in range(25)) * 8

    assert (
        ovis_module._find_repeated_tail(
            "header#" + seven_long_units,
            min_text_len=0,
            max_period=200,
            min_period=1,
            min_repeat_chars=200,
            min_repeat_times=8,
        )
        is None
    )
    assert (
        ovis_module._find_repeated_tail(
            "header#" + eight_units,
            min_text_len=0,
            max_period=200,
            min_period=1,
            min_repeat_chars=200,
            min_repeat_times=8,
        )
        is not None
    )


def test_early_match_accepts_period_200_and_rejects_period_201():
    boundary_unit = "".join(chr(0x600 + index) for index in range(200))
    oversized_unit = "".join(chr(0x800 + index) for index in range(201))

    boundary_match = ovis_module._find_repeated_tail(
        "header#" + boundary_unit * 8,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )
    oversized_match = ovis_module._find_repeated_tail(
        "header#" + oversized_unit * 8,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )

    assert boundary_match is not None
    assert boundary_match.period_len == 200
    assert oversized_match is None


def test_repeat_stopping_excludes_prompt_and_checks_on_schedule():
    processor = _CharacterTokenProcessor()
    prompt = torch.full((1, 250), ord("1"), dtype=torch.long)
    criterion = ovis_module._RepeatedTailStoppingCriteria(processor, prompt_length=250)

    ids = torch.cat([prompt, torch.full((1, 127), ord("1"), dtype=torch.long)], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == []

    ids = torch.cat([prompt, torch.full((1, 128), ord("1"), dtype=torch.long)], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == [128]

    ids = torch.cat([prompt, torch.full((1, 159), ord("1"), dtype=torch.long)], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == [128]

    ids = torch.cat([prompt, torch.full((1, 160), ord("1"), dtype=torch.long)], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == [128, 160]


def test_repeat_stopping_returns_device_bool_and_bounds_tail():
    processor = _CharacterTokenProcessor()
    prompt = torch.tensor([[10, 11]], dtype=torch.long)
    generated = torch.full((1, 900), ord("1"), dtype=torch.long)
    criterion = ovis_module._RepeatedTailStoppingCriteria(processor, prompt_length=2)

    stopped = criterion(torch.cat([prompt, generated], dim=1), None)

    assert stopped.shape == (1,)
    assert stopped.dtype == torch.bool
    assert stopped.device == generated.device
    assert stopped.item()
    assert processor.decoded_widths == [768]
    assert criterion.triggered_match is not None
    assert criterion.triggered_at_tokens == 900


def test_repeat_stopping_treats_generated_thinking_marker_as_terminal():
    processor = _CharacterTokenProcessor()
    prompt = torch.tensor([[10, 11]], dtype=torch.long)
    valid = "# Contents\n" + "".join(f"item {index}: page {index * 7}\n" for index in range(20))
    generated_text = valid + "\n<think>\n\n</think>\n\n" + "".join(f"## {index}\n" for index in range(1, 20))
    generated = torch.tensor([[ord(char) for char in generated_text]], dtype=torch.long)
    criterion = ovis_module._RepeatedTailStoppingCriteria(processor, prompt_length=2)

    stopped = criterion(torch.cat([prompt, generated], dim=1), None)

    assert stopped.item()
    assert criterion.stop_reason == "thinking_marker"
    assert criterion.triggered_match is None
    assert criterion.triggered_at_tokens == len(generated_text)


def test_repeat_stopping_rejects_batching():
    criterion = ovis_module._RepeatedTailStoppingCriteria(_CharacterTokenProcessor(), 2)

    with pytest.raises(ValueError, match="batch size 1"):
        criterion(torch.ones((2, 130), dtype=torch.long), None)


def test_repeat_stopping_keeps_normal_markdown_table_and_page_state_isolated():
    markdown = "| index | value |\n| --- | --- |\n" + "".join(f"| {index} | item-{index} |\n" for index in range(20))
    prompt = torch.tensor([[10, 11]], dtype=torch.long)
    output = torch.tensor([[ord(char) for char in markdown]], dtype=torch.long)
    repeated = torch.full((1, 250), ord("1"), dtype=torch.long)
    first = ovis_module._RepeatedTailStoppingCriteria(_CharacterTokenProcessor(), 2)
    second = ovis_module._RepeatedTailStoppingCriteria(_CharacterTokenProcessor(), 2)

    assert first(torch.cat([prompt, repeated], dim=1), None).item()
    assert not second(torch.cat([prompt, output], dim=1), None).item()
    assert first.triggered_match is not None
    assert second.triggered_match is None


def test_normalize_triggered_repeat_requires_the_recorded_fingerprint():
    repeated = "1\n\n" * 80
    trigger = ovis_module._find_repeated_tail(
        repeated,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )

    assert trigger is not None
    assert ovis_module._normalize_triggered_repeat("header\n" + repeated, trigger) == "header\n1\n\n"
    assert ovis_module._normalize_triggered_repeat("header\n" + repeated[:-1] + "x", trigger) is None


def test_crop_visual_regions_uses_official_rounding_and_keeps_renderable_tag(tmp_path):
    image = Image.new("RGB", (101, 203), "white")
    tag = '<img src="images/bbox_100_200_700_800.jpg" />'

    rendered = _process_visual_regions(tag, image, tmp_path, mode="crop", warn=MagicMock())

    assert rendered == tag
    crop_path = tmp_path / "images" / "bbox_100_200_700_800.jpg"
    assert crop_path.exists()
    with Image.open(crop_path) as crop:
        assert crop.size == (61, 121)


def test_drop_visual_regions_removes_only_matching_tags_and_does_not_create_assets(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    tag = '<img src="images/bbox_0_0_500_500.jpg" />'
    markdown = f"before {tag} after ![kept](images/other.png)"

    rendered = _process_visual_regions(markdown, image, tmp_path, mode="drop", warn=MagicMock())

    assert rendered == "before  after ![kept](images/other.png)"
    assert not (tmp_path / "images").exists()


def test_crop_visual_regions_removes_invalid_bbox_and_warns(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    tag = '<img src="images/bbox_800_100_200_900.jpg" />'
    warn = MagicMock()

    rendered = _process_visual_regions(f"left {tag} right", image, tmp_path, mode="crop", warn=warn)

    assert rendered == "left  right"
    warn.assert_called_once()
    assert not (tmp_path / "images" / "bbox_800_100_200_900.jpg").exists()


def test_crop_visual_regions_handles_repeated_bbox_deterministically(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    tag = '<img src="images/bbox_0_0_500_500.jpg" />'

    rendered = _process_visual_regions(f"{tag}\n{tag}", image, tmp_path, mode="crop", warn=MagicMock())

    assert rendered == f"{tag}\n{tag}"
    assert [path.name for path in (tmp_path / "images").iterdir()] == ["bbox_0_0_500_500.jpg"]


def test_crop_visual_regions_removes_tag_when_crop_save_fails(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    tag = '<img src="images/bbox_0_0_500_500.jpg" />'
    warn = MagicMock()

    with patch.object(Image.Image, "save", side_effect=OSError("disk full")):
        rendered = _process_visual_regions(tag, image, tmp_path, mode="crop", warn=warn)

    assert rendered == ""
    warn.assert_called_once()
    assert not (tmp_path / "images" / "bbox_0_0_500_500.jpg").exists()


def test_prefix_visual_region_paths_only_rewrites_ovis_tags():
    tag = '<img src="images/bbox_0_10_500_600.jpg" />'
    markdown = f'{tag}\n<img src="images/chart.jpg" />\n![other](images/other.png)'

    rendered = _prefix_visual_region_paths(markdown, "page_0001")

    assert '<img src="page_0001/images/bbox_0_10_500_600.jpg" />' in rendered
    assert '<img src="images/chart.jpg" />' in rendered
    assert "![other](images/other.png)" in rendered


def _run_direct_character_generation(monkeypatch, *, final_text=None):
    captured = {}
    console = _console()
    trigger_text = "header\n" + "1\n\n" * 80

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return FakeBatch(input_ids=torch.tensor([[10, 11]]), pixel_values=torch.tensor([1.0]))

        def batch_decode(self, token_ids, **kwargs):
            return ["".join(chr(int(value)) for value in row.tolist()) for row in token_ids]

    class FakeRepeatingModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            trigger_ids = torch.tensor(
                [[10, 11, *[ord(char) for char in trigger_text]]],
                dtype=torch.long,
            )
            assert kwargs["stopping_criteria"][0](trigger_ids, None).item()
            output_text = final_text if final_text is not None else trigger_text
            captured["trigger"] = kwargs["stopping_criteria"][0]
            return torch.tensor(
                [[10, 11, *[ord(char) for char in output_text]]],
                dtype=torch.long,
            )

    processor = FakeProcessor()
    model = FakeRepeatingModel()

    class FakeLoader:
        def get_or_load_processor(self, *args, **kwargs):
            return processor

        def get_or_load_model(self, *args, **kwargs):
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = type("FakeAutoProcessor", (), {})
    fake_transformers.Qwen3_5ForConditionalGeneration = type("FakeQwen35", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(ovis_module, "_TRANS_LOADER", None)
    monkeypatch.setattr(ovis_module, "transformerLoader", lambda *args, **kwargs: FakeLoader())
    monkeypatch.setattr(ovis_module, "resolve_device_dtype", lambda: ("cpu", torch.float32, "eager"))

    result = _DirectPageInferencer(
        model_id="ATH-MaaS/OvisOCR2",
        max_new_tokens=16384,
        min_pixels=448 * 448,
        max_pixels=2880 * 2880,
        console=console,
    ).infer_page(Image.new("RGB", (1600, 1200), "white"), "prompt")
    return result, console.file.getvalue(), captured["trigger"]


def test_direct_inferencer_normalizes_and_logs_triggered_repeat(monkeypatch):
    result, log, trigger = _run_direct_character_generation(monkeypatch)

    assert result == "header\n1\n\n"
    assert trigger.triggered_at_tokens == 247
    assert "generated_tokens=247" in log
    assert "period_chars=3" in log
    assert "repeat_times=80" in log


def test_direct_inferencer_preserves_output_when_trigger_cannot_be_revalidated(monkeypatch):
    result, log, _ = _run_direct_character_generation(
        monkeypatch,
        final_text="header\nnot repeated",
    )

    assert result == "header\nnot repeated"
    assert "could not revalidate repeated tail" in log


def test_direct_inferencer_uses_native_qwen35_contract(monkeypatch):
    from module.providers.ocr import ovis_ocr2 as ovis_module

    captured = {}

    class FakeBatch(dict):
        def to(self, device):
            captured["batch_device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["processor_kwargs"] = kwargs
            return FakeBatch(input_ids=torch.tensor([[10, 11]]), pixel_values=torch.tensor([1.0]))

        def batch_decode(self, token_ids, **kwargs):
            captured["decoded_ids"] = [item.tolist() for item in token_ids]
            captured["decode_kwargs"] = kwargs
            return ["raw markdown"]

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[10, 11, 20, 21]])

    fake_processor = FakeProcessor()
    fake_model = FakeModel()

    class FakeLoader:
        def get_or_load_processor(self, *args, **kwargs):
            captured["processor_load"] = (args, kwargs)
            return fake_processor

        def get_or_load_model(self, *args, **kwargs):
            captured["model_load"] = (args, kwargs)
            return fake_model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = type("FakeAutoProcessor", (), {})
    fake_transformers.Qwen3_5ForConditionalGeneration = type("FakeQwen35", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(ovis_module, "_TRANS_LOADER", None)
    monkeypatch.setattr(ovis_module, "transformerLoader", lambda *args, **kwargs: FakeLoader())
    monkeypatch.setattr(ovis_module, "resolve_device_dtype", lambda: ("cpu", torch.float32, "eager"))

    image = Image.new("RGB", (1600, 1200), "white")
    result = _DirectPageInferencer(
        model_id="ATH-MaaS/OvisOCR2",
        max_new_tokens=16384,
        min_pixels=448 * 448,
        max_pixels=2880 * 2880,
        console=_console(),
    ).infer_page(image, "prompt")

    assert result == "raw markdown"
    assert captured["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "prompt"},
            ],
        }
    ]
    assert captured["processor_kwargs"] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
        "processor_kwargs": {
            "images_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880},
        },
    }
    assert captured["processor_load"][1]["trust_remote_code"] is False
    assert captured["model_load"][0][1] is fake_transformers.Qwen3_5ForConditionalGeneration
    assert captured["model_load"][1]["trust_remote_code"] is False
    assert captured["generate_kwargs"]["do_sample"] is False
    assert captured["generate_kwargs"]["max_new_tokens"] == 16384
    assert len(captured["generate_kwargs"]["stopping_criteria"]) == 1
    assert isinstance(
        captured["generate_kwargs"]["stopping_criteria"][0],
        ovis_module._RepeatedTailStoppingCriteria,
    )
    assert captured["decoded_ids"] == [[20, 21]]


def test_openai_inferencer_uses_full_resolution_png_and_vllm_extra_body():
    runtime = RuntimeBackendConfig(
        mode="openai",
        base_url="http://127.0.0.1:8000/v1",
        api_key="",
        model_id="ATH-MaaS/OvisOCR2",
        temperature=0.0,
        top_p=1.0,
        max_tokens=16384,
    )
    image = Image.new("RGB", (1601, 1203), "white")

    with patch(
        "module.providers.ocr.ovis_ocr2.OpenAIChatRuntime.complete",
        return_value="raw markdown",
    ) as complete:
        result = _OpenAIPageInferencer(
            runtime=runtime,
            min_pixels=448 * 448,
            max_pixels=2880 * 2880,
        ).infer_page(image, "prompt")

    assert result == "raw markdown"
    messages = complete.call_args.args[0]
    assert [message["role"] for message in messages] == ["user"]
    assert [part["type"] for part in messages[0]["content"]] == ["image_url", "text"]
    data_url = messages[0]["content"][0]["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as encoded:
        assert encoded.size == (1601, 1203)
        assert encoded.format == "PNG"
    assert messages[0]["content"][1]["text"] == "prompt"
    assert complete.call_args.kwargs["extra_body"] == {
        "mm_processor_kwargs": {"images_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880}},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_provider_prompt_precedence_ignores_blank_values(tmp_path):
    provider = _provider(
        tmp_path,
        {"prompt": "  "},
        prompts={"ovis_ocr2_prompt": "legacy prompt"},
    )
    assert provider.get_prompts("image/png") == ("", "legacy prompt")

    provider = _provider(
        tmp_path,
        {"prompt": "provider prompt"},
        prompts={"ovis_ocr2_prompt": "legacy prompt"},
    )
    assert provider.get_prompts("image/png") == ("", "provider prompt")

    provider = _provider(tmp_path, {"prompt": ""}, prompts={"ovis_ocr2_prompt": ""})
    assert provider.get_prompts("image/png") == ("", OvisOCR2Provider.default_prompt)


def test_provider_rejects_invalid_visual_region_mode_immediately(tmp_path):
    provider = _provider(tmp_path, {"visual_region_mode": "keep"})
    with pytest.raises(ValueError, match="visual_region_mode"):
        provider._get_visual_region_mode()


def test_provider_runtime_uses_top_p_config_and_preserves_cli_override(tmp_path):
    provider = _provider(tmp_path, {"top_p": 0.42})
    assert provider.get_runtime_backend().top_p == 0.42

    provider = _provider(
        tmp_path,
        {"top_p": 0.42},
        args=SimpleNamespace(local_runtime_top_p=0.75),
    )
    assert provider.get_runtime_backend().top_p == 0.75


def test_provider_handles_only_pdf_and_opted_in_images():
    args = SimpleNamespace(ocr_model="ovis_ocr2", document_image=True)
    assert OvisOCR2Provider.can_handle(args, "application/pdf") is True
    assert OvisOCR2Provider.can_handle(args, "application/zip") is False
    assert OvisOCR2Provider.can_handle(args, "image/png") is True
    args.document_image = False
    assert OvisOCR2Provider.can_handle(args, "image/png") is False


@pytest.mark.parametrize("backend", ["direct", "openai"])
def test_single_image_pipeline_is_backend_independent(tmp_path, backend):
    image_path = tmp_path / f"source-{backend}.png"
    Image.new("RGBA", (100, 80), (255, 255, 255, 128)).save(image_path)
    output_dir = tmp_path / f"output-{backend}"
    provider = _provider(tmp_path, {"runtime_backend": backend, "visual_region_mode": "crop"})
    inferencer = _SequenceInferencer(['heading\n<img src="images/bbox_0_0_500_500.jpg" />'])
    provider._create_inferencer = MagicMock(return_value=inferencer)

    result = provider.attempt(
        _media(image_path, "image/png", output_dir),
        PromptContext(system="ignored", user="prompt"),
    )

    assert result.raw == 'heading\n<img src="images/bbox_0_0_500_500.jpg" />'
    assert result.metadata["runtime_backend"] == backend
    assert result.metadata["failed_pages"] == []
    assert inferencer.calls == [("RGB", (100, 80), "prompt")]
    assert (output_dir / "page.png").exists()
    assert (output_dir / "result.md").read_text(encoding="utf-8") == result.raw
    assert (output_dir / "images" / "bbox_0_0_500_500.jpg").exists()


@pytest.mark.parametrize("backend", ["direct", "openai"])
def test_single_image_pipeline_truncates_generated_thinking_and_counter_tail(tmp_path, backend):
    image_path = tmp_path / f"thinking-{backend}.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    output_dir = tmp_path / f"thinking-output-{backend}"
    provider = _provider(tmp_path, {"runtime_backend": backend})
    provider._create_inferencer = MagicMock(
        return_value=_SequenceInferencer(["valid markdown\n\n<think>\n\n</think>\n\n## 1\n## 2"])
    )

    result = provider.attempt(
        _media(image_path, "image/png", output_dir),
        PromptContext(system="", user="prompt"),
    )

    assert result.raw == "valid markdown"
    assert (output_dir / "result.md").read_text(encoding="utf-8") == "valid markdown"


def test_backend_variants_produce_identical_markdown_and_assets(tmp_path):
    outputs = []
    for backend in ("direct", "openai"):
        image_path = tmp_path / f"same-{backend}.png"
        Image.new("RGB", (100, 80), "white").save(image_path)
        output_dir = tmp_path / backend
        provider = _provider(tmp_path, {"runtime_backend": backend, "visual_region_mode": "crop"})
        provider._create_inferencer = MagicMock(
            return_value=_SequenceInferencer(['same\n<img src="images/bbox_0_0_500_500.jpg" />'])
        )
        result = provider.attempt(
            _media(image_path, "image/png", output_dir),
            PromptContext(system="", user="prompt"),
        )
        outputs.append(
            (
                result.raw,
                (output_dir / "result.md").read_bytes(),
                (output_dir / "images" / "bbox_0_0_500_500.jpg").read_bytes(),
            )
        )

    assert outputs[0] == outputs[1]


def test_pdf_pipeline_writes_page_results_and_rewrites_only_root_paths(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf")
    output_dir = tmp_path / "source"
    provider = _provider(tmp_path, {"visual_region_mode": "crop"})
    provider._create_inferencer = MagicMock(
        return_value=_SequenceInferencer(
            [
                'page one\n<img src="images/bbox_0_0_500_500.jpg" />',
                'page two\n<img src="images/bbox_500_500_1000_1000.jpg" />',
            ]
        )
    )
    pages = [
        SimpleNamespace(page_number=1, image=Image.new("RGB", (100, 100), "white")),
        SimpleNamespace(page_number=2, image=Image.new("RGB", (100, 100), "white")),
    ]

    with patch("module.providers.ocr.ovis_ocr2.iter_pdf_pages_high_quality", return_value=iter(pages)):
        result = provider.attempt(
            _media(pdf_path, "application/pdf", output_dir),
            PromptContext(system="", user="prompt"),
        )

    assert '<img src="page_0001/images/bbox_0_0_500_500.jpg" />' in result.raw
    assert '<img src="page_0002/images/bbox_500_500_1000_1000.jpg" />' in result.raw
    assert "\n<--- Page Split --->\n" in result.raw
    assert '<img src="images/bbox_0_0_500_500.jpg" />' in (output_dir / "page_0001" / "result.md").read_text(encoding="utf-8")
    assert (output_dir / "result.md").read_text(encoding="utf-8") == result.raw
    assert result.metadata["failed_pages"] == []


def test_pdf_pipeline_continues_after_failed_page_and_records_metadata(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf")
    output_dir = tmp_path / "source"
    provider = _provider(tmp_path)
    provider._create_inferencer = MagicMock(return_value=_SequenceInferencer(["page one", RuntimeError("bad page")]))
    pages = [
        SimpleNamespace(page_number=1, image=Image.new("RGB", (20, 20), "white")),
        SimpleNamespace(page_number=2, image=Image.new("RGB", (20, 20), "white")),
    ]

    with patch("module.providers.ocr.ovis_ocr2.iter_pdf_pages_high_quality", return_value=iter(pages)):
        result = provider.attempt(
            _media(pdf_path, "application/pdf", output_dir),
            PromptContext(system="", user="prompt"),
        )

    assert result.raw == "page one"
    assert result.metadata["failed_pages"] == [2]
    assert not (output_dir / "page_0002" / "result.md").exists()


def test_pdf_pipeline_treats_rgb_conversion_as_a_page_local_failure(tmp_path):
    class BrokenPageImage:
        def convert(self, _mode):
            raise OSError("bad pixels")

        def close(self):
            pass

    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf")
    output_dir = tmp_path / "source"
    provider = _provider(tmp_path)
    provider._create_inferencer = MagicMock(return_value=_SequenceInferencer(["page two"]))
    pages = [
        SimpleNamespace(page_number=1, image=BrokenPageImage()),
        SimpleNamespace(page_number=2, image=Image.new("RGB", (20, 20), "white")),
    ]

    with patch("module.providers.ocr.ovis_ocr2.iter_pdf_pages_high_quality", return_value=iter(pages)):
        result = provider.attempt(
            _media(pdf_path, "application/pdf", output_dir),
            PromptContext(system="", user="prompt"),
        )

    assert result.raw == "page two"
    assert result.metadata["failed_pages"] == [1]


@pytest.mark.parametrize("outputs", [[RuntimeError("one"), RuntimeError("two")], ["", "   "]])
def test_pdf_pipeline_raises_when_every_page_inference_fails(tmp_path, outputs):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf")
    output_dir = tmp_path / "source"
    provider = _provider(tmp_path)
    provider._create_inferencer = MagicMock(return_value=_SequenceInferencer(outputs))
    pages = [
        SimpleNamespace(page_number=1, image=Image.new("RGB", (20, 20), "white")),
        SimpleNamespace(page_number=2, image=Image.new("RGB", (20, 20), "white")),
    ]

    with (
        patch("module.providers.ocr.ovis_ocr2.iter_pdf_pages_high_quality", return_value=iter(pages)),
        pytest.raises(RuntimeError, match="all 2 PDF pages"),
    ):
        provider.attempt(
            _media(pdf_path, "application/pdf", output_dir),
            PromptContext(system="", user="prompt"),
        )

    assert not (output_dir / "result.md").exists()


def test_single_image_empty_model_output_is_an_error(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    provider = _provider(tmp_path)
    provider._create_inferencer = MagicMock(return_value=_SequenceInferencer(["  "]))

    with pytest.raises(ValueError, match="empty output"):
        provider.attempt(
            _media(image_path, "image/png", tmp_path / "output"),
            PromptContext(system="", user="prompt"),
        )


def test_page_snapshot_failure_does_not_block_inference_or_crop(tmp_path, monkeypatch):
    from module.providers.ocr import ovis_ocr2 as ovis_module

    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    output_dir = tmp_path / "output"
    provider = _provider(tmp_path)
    inferencer = _SequenceInferencer(['ok <img src="images/bbox_0_0_500_500.jpg" />'])
    provider._create_inferencer = MagicMock(return_value=inferencer)
    monkeypatch.setattr(ovis_module, "_save_page_snapshot", lambda *args, **kwargs: False)

    result = provider.attempt(
        _media(image_path, "image/png", output_dir),
        PromptContext(system="", user="prompt"),
    )

    assert result.raw.startswith("ok")
    assert inferencer.calls == [("RGB", (20, 20), "prompt")]
    assert (output_dir / "images" / "bbox_0_0_500_500.jpg").exists()
