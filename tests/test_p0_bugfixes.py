from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image
from rich.console import Console


def test_save_caption_preserves_bbox_json_instead_of_truncating_txt(tmp_path):
    import json

    from module.lanceexport import save_caption

    payload = {"high_level_description": "A red apple on a wooden desk.", "objects": []}
    raw = json.dumps(payload, ensure_ascii=False)
    out_txt = tmp_path / "sample.txt"
    assert save_caption(str(out_txt), [raw], "image/png") is True
    assert json.loads(out_txt.read_text(encoding="utf-8")) == payload
    assert json.loads((tmp_path / "sample.json").read_text(encoding="utf-8")) == payload


def test_normalize_subtitle_timestamps_preserves_single_digit_hour():
    from module.caption_pipeline.postprocess import _normalize_subtitle_timestamps

    srt = "1\n1:02:03,456 --> 1:02:08,900\nhello\n\n2\n0:05,100 --> 10:12,200\nworld\n"
    normalized = _normalize_subtitle_timestamps(srt)
    assert "01:02:03,456 --> 01:02:08,900" in normalized
    assert "00:00:05,100 --> 00:10:12,200" in normalized
    assert "1:000:" not in normalized


def test_gemini_task_template_does_not_corrupt_english_article_a():
    from module.providers.resolver import PromptResolver

    config = {
        "prompts": {
            "task": {
                "combine_a_and_b": "Create a new image by combining the elements.",
                "change_a_to_b": "change a to b",
            }
        }
    }
    resolver = PromptResolver(config, "gemini")
    combined = resolver._apply_gemini_task_template("combine cat and dog", "")
    assert combined == "Create a new image by combining the elements."

    legacy = resolver._apply_gemini_task_template("change cat to dog", "")
    assert legacy == "change cat to dog"


def test_normalize_runtime_args_is_idempotent_for_segment_time_explicit():
    from module.providers.catalog import normalize_runtime_args

    args = SimpleNamespace(ocr_model="", vlm_image_model="", alm_model="", segment_time=None)
    normalize_runtime_args(args)
    assert args.segment_time == 600
    assert args.segment_time_explicit is False

    normalize_runtime_args(args)
    assert args.segment_time == 600
    assert args.segment_time_explicit is False


def test_openai_compatible_keyless_default_and_video_skips_json_mode(tmp_path):
    from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from module.providers.cloud_vlm.openai_compatible import OpenAICompatibleProvider

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video-bytes")
    args = SimpleNamespace(openai_api_key="", openai_base_url="http://localhost:8000/v1", openai_model_name="local-vl")
    ctx = ProviderContext(console=Console(file=io.StringIO()), config={}, args=args)
    provider = OpenAICompatibleProvider(ctx)

    captured_client = {}
    captured_req = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured_client.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured_req.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="1\n00:00:00,000 --> 00:00:02,000\nok"))])

        def close(self):
            pass

    media = MediaContext(uri=str(clip), mime="video/mp4", sha256hash="abc", modality=MediaModality.VIDEO, blob="AAAA")
    prompts = PromptContext(system="sys", user="usr")
    with patch("openai.OpenAI", FakeClient):
        result = provider.attempt(media, prompts)

    assert captured_client["api_key"] == "sk-no-key-required"
    assert "response_format" not in captured_req
    assert "00:00:00,000" in result.raw


def test_encode_image_to_blob_works_without_rich_pixels(tmp_path, monkeypatch):
    from module.providers import utils as provider_utils

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), "blue").save(img_path)
    monkeypatch.setattr(provider_utils, "Pixels", None)

    blob, pixels = provider_utils.encode_image_to_blob(str(img_path))
    assert isinstance(blob, str) and len(blob) > 0
    assert pixels is None


def test_classify_remote_api_error_avoids_path_false_positives_and_fast_fails_oom():
    from module.providers.utils import classify_remote_api_error

    assert classify_remote_api_error(FileNotFoundError("missing.png"), base_wait=1.0) is None
    assert classify_remote_api_error(RuntimeError("CUDA out of memory"), base_wait=1.0) is None
    # Path/dimension numbers like img_400.png or 429x429 must not trigger HTTP 400/429 handling
    assert classify_remote_api_error(RuntimeError("failed on /tmp/img_400.png"), base_wait=1.0) == 1.0
    assert classify_remote_api_error(RuntimeError("resize 429x429 failed"), base_wait=1.0) == 1.0
    # Real HTTP status codes still match
    assert classify_remote_api_error(RuntimeError("HTTP 429 Too Many Requests"), base_wait=1.0) == 59.0
    assert classify_remote_api_error(RuntimeError("401 Unauthorized"), base_wait=1.0) is None


def test_process_runner_derives_torch_and_torchvision_extras_from_pyproject():
    from gui.utils.process_runner import _UV_TORCH_EXTRAS, _UV_TORCHVISION_EXTRAS

    for extra in ("infinity-parser2-ocr", "ovis-ocr2", "mega-asr-local", "vocal-midi", "image-align", "auto-rig-pose"):
        assert extra in _UV_TORCH_EXTRAS
        assert extra in _UV_TORCHVISION_EXTRAS
    assert "dettato-local" not in _UV_TORCH_EXTRAS


def test_log_buffer_clear_keeps_sequence_monotonic():
    from gui.utils.log_buffer import LogBuffer

    buf = LogBuffer()
    seen = []
    buf.subscribe(lambda seq, line: seen.append((seq, line)))
    buf.push("first")
    first_seq = seen[-1][0]
    buf.clear()
    buf.push("second")
    assert seen[-1][0] > first_seq


def test_settings_dialog_save_preserves_toml_comments(tmp_path, monkeypatch):
    from gui.wizard import step7_settings

    cfg_file = tmp_path / "general.toml"
    cfg_file.write_text("# important comment\n[media]\nimage_quality = 85 # inline note\n", encoding="utf-8")
    monkeypatch.setattr(step7_settings, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(step7_settings.ui, "notify", MagicMock())

    dialog = step7_settings.SettingsDialog.__new__(step7_settings.SettingsDialog)
    dialog._active_key = "general"
    dialog.data = {"general": {"media": {"image_quality": 92}}}
    dialog._save_current()

    saved = cfg_file.read_text(encoding="utf-8")
    assert "# important comment" in saved
    assert "# inline note" in saved
    assert "image_quality = 92" in saved
