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
from module.providers.ocr.ovis_ocr2 import (
    OvisOCR2Provider,
    _clean_truncated_repeats,
    _DirectPageInferencer,
    _OpenAIPageInferencer,
    _prefix_visual_region_paths,
    _process_visual_regions,
)


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
        "images_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880},
    }
    assert captured["processor_load"][1]["trust_remote_code"] is False
    assert captured["model_load"][0][1] is fake_transformers.Qwen3_5ForConditionalGeneration
    assert captured["model_load"][1]["trust_remote_code"] is False
    assert captured["generate_kwargs"]["do_sample"] is False
    assert captured["generate_kwargs"]["max_new_tokens"] == 16384
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
