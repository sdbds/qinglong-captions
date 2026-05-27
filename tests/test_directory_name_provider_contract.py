import io
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from tests.provider_v2_helpers import make_provider_args


def _quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def _media_for(uri: Path, mime: str):
    from module.providers.base import MediaContext, MediaModality

    if mime.startswith("image"):
        modality = MediaModality.IMAGE
        return MediaContext(uri=str(uri), mime=mime, sha256hash="", modality=modality, blob="base64data")
    if mime.startswith("video"):
        modality = MediaModality.VIDEO
        return MediaContext(uri=str(uri), mime=mime, sha256hash="", modality=modality)
    if mime.startswith("audio"):
        modality = MediaModality.AUDIO
        return MediaContext(uri=str(uri), mime=mime, sha256hash="", modality=modality)
    return MediaContext(uri=str(uri), mime=mime, sha256hash="", modality=MediaModality.DOCUMENT)


VISUAL_PROVIDER_CASES = [
    ("kimi_code", "image/jpeg", {"kimi_code_api_key": "test-key"}),
    ("kimi_vl", "image/jpeg", {"kimi_api_key": "test-key"}),
    ("mimo", "image/jpeg", {"mimo_api_key": "test-key"}),
    ("minimax_api", "image/jpeg", {}),
    ("minimax_code", "image/jpeg", {}),
    (
        "openai_compatible",
        "image/jpeg",
        {
            "openai_api_key": "test-key",
            "openai_base_url": "http://localhost:1234/v1",
            "openai_model_name": "test-model",
        },
    ),
    ("stepfun", "video/mp4", {"step_api_key": "test-key"}),
    ("qwenvl", "video/mp4", {"qwenVL_api_key": "test-key"}),
    ("ark", "video/mp4", {"ark_api_key": "test-key"}),
    ("glm", "video/mp4", {"glm_api_key": "test-key"}),
    ("codex_subscription", "image/jpeg", {"codex_subscription": True}),
    ("gemini", "image/jpeg", {"gemini_api_key": "test-key"}),
    ("mistral_ocr", "image/jpeg", {"mistral_api_key": "test-key"}),
    ("moondream", "image/jpeg", {"vlm_image_model": "moondream"}),
    ("qwen_vl_local", "image/jpeg", {"vlm_image_model": "qwen_vl_local"}),
    ("step_vl_local", "video/mp4", {"vlm_image_model": "step_vl_local"}),
    ("penguin_vl_local", "image/jpeg", {"vlm_image_model": "penguin_vl_local"}),
    ("reka_edge_local", "image/jpeg", {"vlm_image_model": "reka_edge_local"}),
    ("lfm_vl_local", "image/jpeg", {"vlm_image_model": "lfm_vl_local"}),
    ("gemma4_local", "video/mp4", {"vlm_image_model": "gemma4_local"}),
    ("marlin_2b_local", "video/mp4", {"vlm_image_model": "marlin_2b_local"}),
]


@pytest.mark.parametrize(("provider_name", "mime", "arg_overrides"), VISUAL_PROVIDER_CASES)
def test_visual_caption_providers_receive_global_directory_name_prompt(
    tmp_path,
    provider_name,
    mime,
    arg_overrides,
):
    from module.providers.base import CaptionResult, ProviderContext
    from module.providers.registry import get_registry

    suffix = ".mp4" if mime.startswith("video") else ".jpg"
    media_path = tmp_path / "Alice (Wonderland)" / f"sample{suffix}"
    captured = {}
    args = make_provider_args(
        dir_name=True,
        pair_dir="",
        max_retries=1,
        wait_time=0.01,
        gemini_task="",
        mode="long",
        **arg_overrides,
    )
    provider_cls = get_registry().get_provider(provider_name)
    provider = provider_cls(
        ProviderContext(
            console=_quiet_console(),
            config={
                "prompts": {
                    "image_prompt": "Describe this image.",
                    "video_prompt": "Describe this video.",
                }
            },
            args=args,
        )
    )
    media = _media_for(media_path, mime)

    def fake_attempt(_media, prompts):
        captured["prompts"] = prompts
        return CaptionResult(raw="ok")

    with (
        patch.object(provider, "prepare_media", return_value=media),
        patch.object(provider, "attempt", side_effect=fake_attempt),
    ):
        result = provider.execute(str(media_path), mime, "hash")

    assert result.raw == "ok"
    prompts = captured["prompts"]
    assert prompts.character_name == "<Alice> from (Wonderland)"
    assert prompts.character_prompt == (
        "If there is a person/character or more in the image you must refer to them as "
        "<Alice> from (Wonderland).\n"
    )
    assert prompts.user.startswith(prompts.character_prompt)


NO_POLLUTION_CASES = [
    ("deepseek_ocr", "application/pdf", {"ocr_model": "deepseek_ocr"}),
    ("deepseek_ocr", "image/png", {"ocr_model": "deepseek_ocr", "document_image": True}),
    ("dots_ocr", "image/png", {"ocr_model": "dots_ocr", "document_image": True}),
    ("mistral_ocr", "image/png", {"ocr_model": "mistral_ocr", "document_image": True, "mistral_api_key": "test-key"}),
    ("moondream", "image/png", {"ocr_model": "moondream", "document_image": True}),
    ("gemma4_local", "audio/wav", {"alm_model": "gemma4_local", "audio_task": "asr"}),
]


@pytest.mark.parametrize(("provider_name", "mime", "arg_overrides"), NO_POLLUTION_CASES)
def test_non_caption_routes_do_not_receive_directory_name_prompt(
    tmp_path,
    provider_name,
    mime,
    arg_overrides,
):
    from module.providers.base import CaptionResult, ProviderContext
    from module.providers.registry import get_registry

    suffix = ".pdf" if mime.startswith("application") else ".wav" if mime.startswith("audio") else ".png"
    media_path = tmp_path / "Alice (Wonderland)" / f"sample{suffix}"
    captured = {}
    args = make_provider_args(
        dir_name=True,
        pair_dir="",
        max_retries=1,
        wait_time=0.01,
        gemini_task="",
        mode="long",
        **arg_overrides,
    )
    provider_cls = get_registry().get_provider(provider_name)
    provider = provider_cls(
        ProviderContext(
            console=_quiet_console(),
            config={
                "prompts": {
                    "audio_prompt": "Transcribe this audio.",
                    "image_prompt": "Describe this image.",
                }
            },
            args=args,
        )
    )
    media = _media_for(media_path, mime)

    def fake_attempt(_media, prompts):
        captured["prompts"] = prompts
        return CaptionResult(raw="ok")

    patches = [
        patch.object(provider, "prepare_media", return_value=media),
        patch.object(provider, "attempt", side_effect=fake_attempt),
    ]
    if provider_name == "dots_ocr":
        patches.append(patch.object(provider, "_resolve_prompt_mode_and_prompt", return_value=("prompt_ocr", "OCR prompt.")))

    with patches[0], patches[1]:
        if len(patches) == 3:
            with patches[2]:
                result = provider.execute(str(media_path), mime, "hash")
        else:
            result = provider.execute(str(media_path), mime, "hash")

    assert result.raw == "ok"
    prompts = captured["prompts"]
    assert prompts.character_name == ""
    assert prompts.character_prompt == ""
    assert not prompts.user.startswith("If there is a person/character")
