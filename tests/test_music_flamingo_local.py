# -*- coding: utf-8 -*-

import io
import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_prompt_resolver_prefers_music_flamingo_audio_prompts():
    from providers.resolver import PromptResolver

    config = {
        "prompts": {
            "audio_system_prompt": "base-system",
            "audio_prompt": "base-user",
            "music_flamingo_audio_system_prompt": "mf-system",
            "music_flamingo_audio_prompt": "mf-user",
        }
    }

    prompts = PromptResolver(config, "music_flamingo_local").resolve(
        "audio/wav",
        SimpleNamespace(pair_dir=""),
    )

    assert prompts.system == "mf-system"
    assert prompts.user == "mf-user"


def test_postprocess_audio_extracts_srt_code_block(tmp_path):
    from module.caption_pipeline.postprocess import postprocess_caption_content

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    console = Console(file=io.StringIO(), force_terminal=False)
    args = SimpleNamespace(mode="all", ocr_model="")
    output = """<think>hidden</think>
```srt
1
00:00,000 --> 00:01,000
gentle piano intro
```"""

    processed = postprocess_caption_content(output, audio_path, args, console)

    assert processed == "1\n00:00:00,000 --> 00:00:01,000\ngentle piano intro"


def test_music_flamingo_post_validate_strips_reasoning_and_extracts_srt(tmp_path):
    from providers.base import CaptionResult, ProviderContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.post_validate(
        CaptionResult(
            raw="""<think>private reasoning</think>
```srt
1
00:00,000 --> 00:01,000
gentle piano intro
```"""
        ),
        media,
        ctx.args,
    )

    assert result.raw == "1\n00:00:00,000 --> 00:00:01,000\ngentle piano intro"


def test_music_flamingo_post_validate_raises_retry_on_invalid_srt(tmp_path):
    from providers.base import CaptionResult, ProviderContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)

    try:
        provider.post_validate(CaptionResult(raw="plain text, not subtitles"), media, ctx.args)
    except Exception as exc:
        assert "RETRY_INVALID_SRT" in str(exc)
    else:
        raise AssertionError("expected post_validate to raise retryable subtitle validation error")


def test_music_flamingo_attempt_builds_audio_conversation_and_decodes_new_tokens(monkeypatch, tmp_path):
    import torch

    from providers.base import ProviderContext, PromptContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, conversation, **kwargs):
            captured["conversation"] = conversation
            captured["template_kwargs"] = kwargs
            return FakeInputs(
                {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
                    "input_features": torch.randn(2, 4, dtype=torch.float32),
                }
            )

        def batch_decode(self, token_ids, **kwargs):
            captured["decoded_ids"] = token_ids.tolist()
            captured["decode_kwargs"] = kwargs
            return ["decoded subtitle text"]

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.bfloat16

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 3, 7, 8]], dtype=torch.int64)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "music_flamingo_local": {
                "max_new_tokens": 123,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.8,
            }
        },
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": FakeProcessor()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(
        media,
        PromptContext(system="system prompt", user="describe this audio"),
    )

    assert captured["conversation"][0]["role"] == "system"
    assert captured["conversation"][1]["content"][0]["type"] == "text"
    assert captured["conversation"][1]["content"][1] == {"type": "audio", "path": str(audio_path.resolve())}
    assert captured["generate_kwargs"]["input_features"].dtype == torch.bfloat16
    assert captured["generate_kwargs"]["max_new_tokens"] == 123
    assert captured["generate_kwargs"]["do_sample"] is True
    assert captured["generate_kwargs"]["temperature"] == 0.6
    assert captured["generate_kwargs"]["top_p"] == 0.8
    assert captured["decoded_ids"] == [[7, 8]]
    assert result.raw == "decoded subtitle text"
