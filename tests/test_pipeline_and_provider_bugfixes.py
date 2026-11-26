from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lance
import numpy as np
import pyarrow as pa
import pytest
from PIL import Image
from rich.console import Console
from rich.style import Style


# ---------------------------------------------------------------------------
# 1. WDTagger: WD14 no double sigmoid, full-schema Lance merge, no force_filename
# ---------------------------------------------------------------------------


def test_wd14_process_batch_does_not_apply_double_sigmoid():
    from module.wdtagger.preprocess import process_batch

    raw_probs = np.array([[0.0, 0.2, 0.85]], dtype=np.float32)

    class FakeSession:
        def run(self, _outputs, _inputs):
            return [raw_probs]

    images = [np.zeros((4, 4, 3), dtype=np.float32)]

    # SmilingWolf WD14 default: outputs are already probabilities
    wd14_out = process_batch(images, FakeSession(), "input_1")
    np.testing.assert_allclose(wd14_out, raw_probs)

    # cl_tagger v1: logits -> sigmoid applied once
    cl_out = process_batch(images, FakeSession(), "input_1", True)
    assert cl_out[0, 0] == pytest.approx(0.5, abs=1e-6)
    assert cl_out[0, 2] > 0.7


def test_wdtagger_merge_caption_updates_preserves_full_schema(tmp_path):
    from module.wdtagger.lance_io import merge_caption_updates

    ds_path = tmp_path / "full.lance"
    table = pa.table(
        {
            "uris": pa.array(["img_a.png", "img_b.png"], type=pa.string()),
            "mime": pa.array(["image/png", "image/png"], type=pa.string()),
            "width": pa.array([640, 800], type=pa.int32()),
            "height": pa.array([480, 600], type=pa.int32()),
            "captions": pa.array([[], []], type=pa.list_(pa.string())),
        }
    )
    dataset = lance.write_dataset(table, str(ds_path))

    merge_caption_updates(dataset, [("img_a.png", ["1girl", "solo"])])

    reloaded = lance.dataset(str(ds_path)).to_table().to_pylist()
    by_uri = {row["uris"]: row for row in reloaded}
    assert by_uri["img_a.png"]["captions"] == ["1girl", "solo"]
    assert by_uri["img_a.png"]["width"] == 640
    assert by_uri["img_a.png"]["height"] == 480
    assert by_uri["img_b.png"]["captions"] == []


def test_wdtagger_load_parent_map_does_not_pass_force_filename(tmp_path):
    from module.wdtagger.model_loader import _load_parent_map

    csv_file = tmp_path / "tag_implications.csv"
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        csv_file.write_text(
            "id,created_at,updated_at,antecedent_name,consequent_name\n"
            "1,2024-01-01,2024-01-01,long_hair,hair\n",
            encoding="utf-8",
        )
        return str(csv_file)

    args = SimpleNamespace(
        remove_parents_tag=True,
        model_dir=str(tmp_path),
        force_download=True,
        remove_underscore=False,
    )
    result = _load_parent_map(args, hf_hub_download_fn=fake_download)

    assert "force_filename" not in captured
    assert "hair" in result


# ---------------------------------------------------------------------------
# 2. calculate_dimensions max_short_edge & small-image floor
# ---------------------------------------------------------------------------


def test_calculate_dimensions_respects_max_short_edge_only_and_clamps_small_images():
    from utils.stream_util import calculate_dimensions

    # Previously ignored max_short_edge when max_pixels & max_long_edge were None
    w, h = calculate_dimensions(2000, 1000, max_short_edge=512, img_scale_num=16)
    assert (w, h) == (1024, 512)

    # Tiny images must never round down to 0
    w_small, h_small = calculate_dimensions(4, 4, img_scale_num=16)
    assert w_small == 16 and h_small == 16

    w_scaled, h_scaled = calculate_dimensions(1000, 10, max_long_edge=32, img_scale_num=16)
    assert w_scaled >= 16 and h_scaled >= 16


# ---------------------------------------------------------------------------
# 3. Providers deterministic crash & security fixes
# ---------------------------------------------------------------------------


def test_minimax_code_parses_markdown_fenced_json_without_unbound_local():
    from module.providers.cloud_vlm.minimax_code import attempt_minimax_code

    fenced = '```json\n{"short_description": "a cat", "long_description": "a fluffy cat"}\n```'
    client = MagicMock()
    client.chat.completions.create.return_value = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning_content=None, content=fenced),
                    finish_reason="stop",
                )
            ]
        )
    ]

    with patch("utils.parse_display.display_caption_and_rate"):
        out = attempt_minimax_code(
            client=client,
            model_path="MiniMax-M3.1-Flash",
            messages=[{"role": "user", "content": "describe"}],
            console=Console(file=io.StringIO()),
            uri="cat.jpg",
            mode="short",
        )

    assert not out.startswith("```")
    assert json.loads(out) == {"short_description": "a cat", "long_description": "a fluffy cat"}


def test_moondream_caption_calls_display_caption_layout_with_required_args_and_raises_on_error(tmp_path):
    from module.providers.local_vlm import moondream as md_mod

    img_path = tmp_path / "sample.png"
    pil_img = Image.new("RGB", (32, 32), (10, 20, 30))
    pil_img.save(img_path)

    class GoodModel:
        def caption(self, _img, length="short"):
            return {"caption": f"{length} desc"}

    class BrokenModel:
        def caption(self, _img, length="short"):
            raise RuntimeError("cuda oom")

    display_calls = []

    def fake_display(**kwargs):
        assert "tag_description" in kwargs
        assert "panel_height" in kwargs
        display_calls.append(kwargs)

    with patch.object(md_mod, "display_caption_layout", side_effect=fake_display):
        res = md_mod.attempt_moondream(
            uri=str(img_path),
            mime="image/png",
            get_model=GoodModel,
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            image=pil_img,
            task="caption",
            captions=["1girl"],
        )

    assert "short desc" in res and "long desc" in res
    assert len(display_calls) == 1
    assert display_calls[0]["tag_description"] == "1girl"
    assert display_calls[0]["panel_height"] == 32

    with pytest.raises(RuntimeError, match="Moondream task\\(s\\) failed"):
        md_mod.attempt_moondream(
            uri=str(img_path),
            mime="image/png",
            get_model=BrokenModel,
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            image=pil_img,
            task="caption",
        )


def test_mistral_large_4_caption_keeps_only_final_text_chunks():
    from mistralai.client.models import TextChunk, ThinkChunk
    from module.providers.vision_api import pixtral

    client = MagicMock()
    client.chat.complete.return_value.choices = [SimpleNamespace(message=SimpleNamespace(content=[
        ThinkChunk(thinking=[TextChunk(text="Not part of the caption. ###")]),
        TextChunk(text="Short: red light"),
        TextChunk(text=" ### Long: A red signal above the crossing."),
    ]))]
    with patch.object(pixtral, "display_caption_layout"):
        result = pixtral.attempt_pixtral(
            client=client, model_path="mistral-large-4-0", mime="image/png",
            console=Console(file=io.StringIO()), progress=None, task_id=None, uri="signal.png",
            messages=[{"role": "user", "content": "Describe the image."}],
        )

    assert result == "Short: red light ### Long: A red signal above the crossing."
    assert client.chat.complete.call_args.kwargs["model"] == "mistral-large-4-0"


def test_pixtral_pdf_ocr_returns_joined_markdown_and_upload_failure_marks_failed():
    from module.providers.base import CaptionStatus, MediaContext, MediaModality, PromptContext, ProviderContext
    from module.providers.vision_api import pixtral as pixtral_mod

    pages = [
        SimpleNamespace(index=0, markdown="# Page 1\nHello", images=[]),
        SimpleNamespace(index=1, markdown="# Page 2\nWorld", images=[]),
    ]
    client = MagicMock()
    client.ocr.process.return_value = SimpleNamespace(pages=pages)

    with patch.object(pixtral_mod, "display_markdown"):
        md_text = pixtral_mod.attempt_pixtral(
            client=client,
            model_path="mistral-ocr-latest",
            mime="application/pdf",
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            uri="doc.pdf",
            document_image=False,
            signed_url_url="https://example.invalid/doc.pdf",
        )

    assert md_text == "# Page 1\nHello\n\n# Page 2\nWorld"

    ctx = ProviderContext(
        console=Console(file=io.StringIO()),
        config={},
        args=SimpleNamespace(mistral_api_key="mk-test", ocr_model="mistral_ocr", document_image=False),
    )
    provider = pixtral_mod.MistralOCRProvider(ctx)
    media = MediaContext(uri="doc.pdf", mime="application/pdf", sha256hash="", modality=MediaModality.DOCUMENT)
    with patch.object(pixtral_mod, "_create_mistral_client", return_value=MagicMock()):
        failed = provider.attempt(media, PromptContext(system="", user=""))

    assert failed.status is CaptionStatus.FAILED
    assert not failed.is_persistable


def test_grok_build_freeform_unwraps_cli_json_envelope(tmp_path):
    from module.providers.grok_build_headless import (
        GrokBuildHeadlessConfig,
        _extract_grok_build_envelope_text,
        _unwrap_npm_cmd_shim,
        run_grok_build_headless_caption,
    )

    assert _extract_grok_build_envelope_text(json.dumps({"text": "clean freeform caption"})) == "clean freeform caption"
    assert _extract_grok_build_envelope_text("plain output") == "plain output"

    script = tmp_path / "fake_grok.py"
    script.write_text('import json; print(json.dumps({"text": "unwrapped caption"}))', encoding="utf-8")
    if os.name == "nt":
        wrapper = tmp_path / "grok.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = tmp_path / "grok"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)

    img = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (0, 255, 0)).save(img)

    res = run_grok_build_headless_caption(
        GrokBuildHeadlessConfig(command=str(wrapper), timeout=10),
        image_path=img,
        prompt="describe",
        mime="image/png",
        structured=False,
    )
    assert res.raw == "unwrapped caption"
    assert json.loads(res.stdout)["text"] == "unwrapped caption"

    # Verify npm .cmd shim unwrapping skips non-npm wrappers
    assert _unwrap_npm_cmd_shim([str(wrapper), "--version"]) == [str(wrapper), "--version"]


def test_codex_app_server_pool_recovers_after_replacement_failure_and_scopes_reset(monkeypatch, tmp_path):
    from module.providers import codex_app_server as cas

    image = tmp_path / "img.png"
    image.write_bytes(b"fake")
    create_calls = 0

    class FakeSdkClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def flaky_create(_config):
        nonlocal create_calls
        create_calls += 1
        if create_calls == 2:
            raise RuntimeError("transient spawn failure")
        return FakeSdkClient()

    caption_calls = 0

    def fake_caption(self, **_kwargs):
        nonlocal caption_calls
        caption_calls += 1
        if caption_calls == 1:
            raise cas.CodexAppServerError("transport failed", kind="transport")
        return cas.CodexAppServerResult(
            raw='{"long_description": "recovered"}',
            parsed={"long_description": "recovered"},
            thread_id="t",
            turn_id="u",
            metadata={},
        )

    monkeypatch.setattr(cas, "_create_sdk_client", flaky_create)
    monkeypatch.setattr(cas.CodexAppServerCaptionClient, "caption_image", fake_caption)

    pool = cas.CodexAppServerClientPool(
        cas.CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        size=1,
    )
    with pytest.raises(cas.CodexAppServerError):
        pool.caption_image(image_path=image, prompt="caption", timeout=1.0)

    # Replacement failed on attempt #2, but slot placeholder kept the pool alive for attempt #3
    recovered = pool.caption_image(image_path=image, prompt="caption", timeout=1.0)
    assert recovered.parsed["long_description"] == "recovered"
    assert create_calls == 3
    pool.close()


def test_pool_returns_recovered_client_when_startup_exhausts_request_deadline(monkeypatch):
    from module.providers import codex_app_server as cas

    clock = [20.0]
    created = []
    attempts = 0

    def create_client(_config):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("replacement unavailable")
        if attempts == 3:
            clock[0] += 2.0
        client = MagicMock()
        created.append(client)
        return client

    def caption(self, **kwargs):
        if self.client is created[0]:
            raise cas.CodexAppServerError("connection closed", kind="transport")
        return "caption"

    monkeypatch.setattr(cas, "_create_sdk_client", create_client)
    monkeypatch.setattr(cas.CodexAppServerCaptionClient, "caption_image", caption)
    pool = cas.CodexAppServerClientPool(cas.CodexAppServerConfig(), 1)
    request = dict(image_path="unused.png", prompt="describe", timeout=0.1)
    try:
        with monkeypatch.context() as patch_clock:
            patch_clock.setattr(cas, "time", SimpleNamespace(monotonic=lambda: clock[0]))
            with pytest.raises(cas.CodexAppServerError, match="connection closed"):
                pool.caption_image(**request)
            with pytest.raises(cas.CodexAppServerError, match="exhausting the request timeout"):
                pool.caption_image(**request)
        # The real deadline bounds this assertion even if the slot was lost.
        assert pool.caption_image(**request) == "caption"
    finally:
        pool.close()
    assert attempts == 3
    for client in created:
        client.close.assert_called_once()


# ---------------------------------------------------------------------------
# 4. lanceexport.py & config/general.toml fixes
# ---------------------------------------------------------------------------


def test_lanceexport_no_headers_uses_stem_dir_and_dedupes_keeping_first_image(tmp_path):
    from module.lanceexport import save_caption_by_pages

    base_path = tmp_path / "manual.txt"
    assert save_caption_by_pages(base_path, ["Plain text without headers"]) is True
    assert (tmp_path / "manual" / "manual.md").is_file()
    assert not (tmp_path / "manual.md").is_dir()

    paged_path = tmp_path / "paged.txt"
    markdown = (
        '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>\n'
        "Intro ![first](sub/img1.png) middle ![dup](sub/img1.png) end ![second](sub/img2.png)\n"
    )
    assert save_caption_by_pages(paged_path, [markdown]) is True
    saved = (tmp_path / "paged" / "paged_1.md").read_text(encoding="utf-8")
    assert "![first](img1.png)" in saved
    assert "![dup](img1.png)" not in saved
    assert "![second](img2.png)" in saved


def test_general_toml_console_colors_are_valid_rich_styles():
    import tomllib

    data = tomllib.loads((Path("config") / "general.toml").read_text(encoding="utf-8"))
    for media_type, style_str in data["colors"].items():
        parsed = Style.parse(style_str)
        assert parsed is not None, f"Invalid Rich style for {media_type}: {style_str!r}"
