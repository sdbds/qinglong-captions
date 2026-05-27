from pathlib import Path
from types import SimpleNamespace


def _args(**overrides):
    values = {
        "dir_name": True,
        "directory_name_source_uri": "",
        "ocr_model": "",
        "document_image": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_disabled_returns_empty_context(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    image_path = tmp_path / "Alice (Wonderland)" / "sample.jpg"

    context = resolve_directory_name_context(
        args=_args(dir_name=False),
        uri=str(image_path),
        mime="image/jpeg",
    )

    assert context.enabled is False
    assert context.has_prompt is False
    assert context.character_name == ""


def test_image_directory_name_builds_character_prompt(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    image_path = tmp_path / "Alice (Wonderland)" / "sample.jpg"

    context = resolve_directory_name_context(
        args=_args(),
        uri=str(image_path),
        mime="image/jpeg",
    )

    assert context.enabled is True
    assert context.applicable is True
    assert context.raw_directory_name == "Alice (Wonderland)"
    assert context.character_name == "<Alice> from (Wonderland)"
    assert context.character_prompt == (
        "If there is a person/character or more in the image you must refer to them as "
        "<Alice> from (Wonderland).\n"
    )


def test_video_uses_same_directory_name_context(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    video_path = tmp_path / "Alice (Wonderland)" / "sample.mp4"

    context = resolve_directory_name_context(
        args=_args(),
        uri=str(video_path),
        mime="video/mp4",
    )

    assert context.applicable is True
    assert context.character_name == "<Alice> from (Wonderland)"
    assert context.has_prompt is True


def test_audio_does_not_inject_directory_name(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    audio_path = tmp_path / "Alice (Wonderland)" / "sample.wav"

    context = resolve_directory_name_context(
        args=_args(),
        uri=str(audio_path),
        mime="audio/wav",
    )

    assert context.enabled is True
    assert context.applicable is False
    assert context.raw_directory_name == "Alice (Wonderland)"
    assert context.character_name == ""
    assert context.character_prompt == ""
    assert context.reason == "unsupported_mime"


def test_pdf_does_not_inject_directory_name(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    pdf_path = tmp_path / "Alice (Wonderland)" / "sample.pdf"

    context = resolve_directory_name_context(
        args=_args(),
        uri=str(pdf_path),
        mime="application/pdf",
    )

    assert context.applicable is False
    assert context.character_prompt == ""


def test_ocr_image_route_does_not_inject_directory_name(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    image_path = tmp_path / "Alice (Wonderland)" / "page.png"

    context = resolve_directory_name_context(
        args=_args(ocr_model="deepseek_ocr", document_image=True),
        uri=str(image_path),
        mime="image/png",
    )

    assert context.applicable is False
    assert context.reason == "ocr_route"
    assert context.character_prompt == ""


def test_document_ocr_provider_does_not_inject_even_without_route_arg(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    image_path = tmp_path / "Alice (Wonderland)" / "page.png"

    context = resolve_directory_name_context(
        args=_args(ocr_model="", document_image=False),
        uri=str(image_path),
        mime="image/png",
        provider_name="deepseek_ocr",
    )

    assert context.applicable is False
    assert context.reason == "document_ocr_provider"
    assert context.character_prompt == ""


def test_source_uri_overrides_clip_directory(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    original_path = tmp_path / "Alice (Wonderland)" / "movie.mp4"
    clip_path = tmp_path / "Alice (Wonderland)" / "movie_clip" / "movie_000.mp4"

    context = resolve_directory_name_context(
        args=_args(directory_name_source_uri=str(original_path)),
        uri=str(clip_path),
        mime="video/mp4",
    )

    assert context.source_uri == str(original_path)
    assert context.raw_directory_name == "Alice (Wonderland)"
    assert context.character_name == "<Alice> from (Wonderland)"


def test_explicit_source_uri_has_highest_priority(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    explicit_source = tmp_path / "Alice (Wonderland)" / "sample.jpg"
    args_source = tmp_path / "Bob (Builder)" / "sample.jpg"
    current_uri = tmp_path / "Current" / "sample.jpg"

    context = resolve_directory_name_context(
        args=_args(directory_name_source_uri=str(args_source)),
        uri=str(current_uri),
        mime="image/jpeg",
        source_uri=str(explicit_source),
    )

    assert context.source_uri == str(explicit_source)
    assert context.character_name == "<Alice> from (Wonderland)"


def test_pair_directory_is_not_used_as_source(tmp_path):
    from module.providers.directory_name_context import resolve_directory_name_context

    image_path = tmp_path / "Alice (Wonderland)" / "sample.jpg"
    pair_path = tmp_path / "Bob (Builder)" / "sample.jpg"
    media = SimpleNamespace(extras={"pair_uri": str(pair_path)})

    context = resolve_directory_name_context(
        args=_args(pair_dir=str(pair_path.parent)),
        uri=str(image_path),
        mime="image/jpeg",
        media=media,
    )

    assert context.character_name == "<Alice> from (Wonderland)"


def test_invalid_or_root_like_path_does_not_raise():
    from module.providers.directory_name_context import resolve_directory_name_context

    context = resolve_directory_name_context(
        args=_args(),
        uri=str(Path("/sample.jpg")),
        mime="image/jpeg",
    )

    assert context.has_prompt is False
