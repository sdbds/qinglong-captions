import base64
import importlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console

from config.runtime_config import coerce_runtime_config
from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext


def _context(config=None, **args):
    return ProviderContext(
        console=Console(file=io.StringIO()),
        config=coerce_runtime_config(config or {}),
        args=SimpleNamespace(**args),
    )


@pytest.mark.parametrize("ocr", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_moondream_direct_opens_rgb_image_and_closes_it(tmp_path, monkeypatch, ocr, raises):
    from module.providers.local_vlm import moondream

    path = tmp_path / "input.png"
    Image.new("RGBA", (32, 16), (1, 2, 3, 255)).save(path)
    provider = moondream.MoondreamProvider(_context(ocr_model="moondream" if ocr else ""))
    seen = []

    def infer(**kwargs):
        image = kwargs["image"]
        assert image is not None
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (1, 2, 3)
        assert kwargs["ocr"] is ocr
        seen.append(image)
        if raises:
            raise RuntimeError("inference failed")
        return "readable content"

    monkeypatch.setattr(moondream, "attempt_moondream", infer)
    media = MediaContext(str(path), "image/png", "", MediaModality.IMAGE)
    if raises:
        with pytest.raises(RuntimeError, match="inference failed"):
            provider.attempt(media, PromptContext("", "Describe"))
    else:
        assert provider.attempt(media, PromptContext("", "Describe")).raw == "readable content"
    with pytest.raises(ValueError, match="closed image"):
        seen[0].getpixel((0, 0))


def test_mimo_honors_frozen_runtime_settings(monkeypatch):
    from module.providers.cloud_vlm import mimo

    provider = mimo.MimoProvider(_context(
        {"mimo": {"thinking": "enabled", "max_completion_tokens": 12345}},
        mimo_api_key="test-key", mode="all",
    ))
    captured = {}

    def infer(**kwargs):
        captured.update(kwargs)
        return "caption"

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(mimo, "attempt_kimi_vl", infer)
    media = MediaContext("input.png", "image/png", "", MediaModality.IMAGE, blob="encoded")
    provider.attempt(media, PromptContext("", "Describe"))
    assert captured["thinking"] == "enabled"
    assert captured["max_tokens"] == 12345


def test_qianfan_honors_frozen_prompt_and_thinking_settings():
    from module.providers.ocr.qianfan import QianfanOCRProvider

    provider = QianfanOCRProvider(_context({"qianfan_ocr": {
        "prompt": "Read exactly", "prompt_strategy": "replace", "think_enabled": False,
    }}))
    assert provider.get_prompts("image/png") == ("", "Read exactly")


@pytest.mark.parametrize("legacy", [False, True])
def test_ovis_honors_frozen_prompt_and_visual_settings(legacy):
    from module.providers.ocr.ovis_ocr2 import OvisOCR2Provider

    config = {"ovis_ocr2": {"visual_region_mode": "drop", "top_p": 0.7}}
    if legacy:
        config["prompts"] = {"ovis_ocr2_prompt": "Read exactly"}
    else:
        config["ovis_ocr2"]["prompt"] = "Read exactly"
    provider = OvisOCR2Provider(_context(config))
    assert provider.get_prompts("image/png") == ("", "Read exactly")
    assert provider._get_visual_region_mode() == "drop"
    assert provider.get_runtime_backend().top_p == 0.7


@pytest.mark.parametrize("module_name,class_name,section", [
    ("acestep_transcriber_local", "AceStepTranscriberLocalProvider", "acestep_transcriber_local"),
    ("eureka_audio_local", "EurekaAudioLocalProvider", "eureka_audio_local"),
    ("music_flamingo_local", "MusicFlamingoLocalProvider", "music_flamingo_local"),
])
def test_local_alm_honors_frozen_nested_generate_kwargs(module_name, class_name, section):
    module = importlib.import_module(f"module.providers.local_alm.{module_name}")
    provider = getattr(module, class_name)(_context({section: {"generate_kwargs": {"max_new_tokens": 123}}}))
    assert provider._resolve_generate_kwargs()["max_new_tokens"] == 123


@pytest.mark.parametrize("size", [(1, 1), (8, 8), (4096, 32), (32, 4096), (256, 256)])
def test_image_encoder_keeps_positive_dimensions(size):
    from module.providers.utils import encode_image_to_blob

    source = io.BytesIO()
    Image.new("RGB", size).save(source, format="PNG")
    source.seek(0)
    blob, _ = encode_image_to_blob(source, to_rgb=True)
    assert blob
    with Image.open(io.BytesIO(base64.b64decode(blob))) as encoded:
        assert 0 < encoded.width <= 1024
        assert 0 < encoded.height <= 1024


@pytest.mark.parametrize("task,key,expected", [
    ("combine red and blue", "combine_a_and_b", "red / blue"),
    ("combine red with blue", "combine_a_and_b", "red / blue"),
    ("transform style photo to ink", "transform_style_a_to_b", "photo / ink"),
    ("change red to blue", "change_a_to_b", "red / blue"),
    ("add red to blue", "add_a_to_b", "red / blue"),
])
@pytest.mark.parametrize("template", ["{a} / {b}", "<a> / <b>", "a / b"])
def test_gemini_task_template_only_substitutes_semantic_arguments(task, key, expected, template):
    from module.providers.resolver import PromptResolver

    resolver = PromptResolver({"prompts": {"task": {key: template}}}, "gemini")
    assert resolver.resolve("image/png", SimpleNamespace(gemini_task=task)).user == expected
