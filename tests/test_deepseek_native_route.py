import io
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from rich.console import Console

from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
from module.providers.catalog import get_provider_declaration, route_choices

ROOT = Path(__file__).resolve().parents[1]


def test_native_is_an_explicit_local_route_separate_from_legacy():
    routes = route_choices("ocr_model")
    assert "deepseek_ocr_native" in routes and "deepseek_ocr" in routes
    native = get_provider_declaration("deepseek_ocr_native")
    legacy = get_provider_declaration("deepseek_ocr")
    assert native.module_path != legacy.module_path
    assert not native.routes[0].requires_remote_config
    assert native.capabilities.supports_images and native.capabilities.supports_documents
    assert "deepseek_ocr_legacy" in routes


def test_standard_extra_installs_native_but_legacy_remains_isolated():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = cfg["project"]["optional-dependencies"]
    native, = (req for req in map(Requirement, deps["deepseek-ocr-native"]) if req.name == "transformers")
    assert "5.23.0" in native.specifier
    assert all(v not in native.specifier for v in ("4.46.3", "5.17.0", "5.23.1", "6.0.0"))
    assert any(Requirement(item).name.lower() == "pymupdf" for item in deps["deepseek-ocr-native"])
    groups = [{i.get("extra") for i in g} for g in cfg["tool"]["uv"]["conflicts"]]
    assert "qinglong-captions[deepseek-ocr-native]" in deps["deepseek-ocr"]
    assert "transformers[serving]==4.46.3" in deps["deepseek-ocr-legacy"]
    for old in ("deepseek-ocr-legacy", "unlimited-ocr", "unlimited-ocr2", "dots-ocr", "penguin-vl-local",
                "reward-model", "cohere-transcribe-local", "eureka-audio-local", "paddleocr-native"):
        assert any({"deepseek-ocr-native", old} <= group for group in groups), old


def test_selection_preserves_custom_checkpoints_and_geometry(tmp_path):
    from utils.ocr_model_config import deepseek_native_model_id, deepseek_ocr_extra, deepseek_ocr_extra_from_config

    official = "deepseek-ai/DeepSeek-OCR-2"
    assert deepseek_native_model_id(official) == "deepseek-community/DeepSeek-OCR-2"
    assert deepseek_ocr_extra(official) == "deepseek-ocr"
    assert deepseek_ocr_extra(official, crop_mode=False) == "deepseek-ocr-legacy"
    assert deepseek_ocr_extra("deepseek-community/DeepSeek-OCR-2", image_size=640) == "deepseek-ocr-legacy"
    assert deepseek_ocr_extra("local/older-checkpoint") == "deepseek-ocr-legacy"
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_ocr2"}', encoding="utf-8")
    assert deepseek_native_model_id(str(tmp_path)) == str(tmp_path.resolve())
    config = tmp_path / "model.toml"
    config.write_text(f'[deepseek_ocr]\nmodel_id = "{official}"\nimage_size = 640\n', encoding="utf-8")
    assert deepseek_ocr_extra_from_config(config) == "deepseek-ocr-legacy"


def test_existing_route_switches_to_native_without_silent_fallback(tmp_path, monkeypatch):
    from module.providers.ocr import deepseek, deepseek_native

    old_prompt = "<image>\ncustom prompt \n"
    ctx = ProviderContext(console=Console(file=io.StringIO()), args=SimpleNamespace(),
                          config={"prompts": {"deepseek_ocr_prompt": old_prompt}, "deepseek_ocr": {"max_new_tokens": 491}})
    provider = deepseek.DeepSeekOCRProvider(ctx)
    media = MediaContext(uri=str(tmp_path / "page.pdf"), mime="application/pdf", modality=MediaModality.DOCUMENT,
                         sha256hash="", blob=None, extras={"output_dir": tmp_path})
    prompts = provider.resolve_prompts(media.uri, media.mime, media)
    seen = []

    def native(**kwargs):
        seen.append(kwargs)
        return "ok"

    monkeypatch.setattr(deepseek_native, "attempt_deepseek_native", native)
    monkeypatch.setattr(deepseek, "attempt_deepseek_ocr", lambda **_: pytest.fail("unexpected legacy inference"))
    result = provider.attempt(media, prompts)
    assert result.raw == "ok" and result.metadata["provider"] == "deepseek_ocr"
    assert seen[0]["model_id"] == "deepseek-community/DeepSeek-OCR-2"
    assert seen[0]["prompt_text"] == old_prompt and seen[0]["max_new_tokens"] == 491

    def fail(**_):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(deepseek_native, "attempt_deepseek_native", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        provider.attempt(media, prompts)


def test_existing_remote_route_and_explicit_legacy_are_preserved(tmp_path, monkeypatch):
    from module.providers.ocr import deepseek, deepseek_native

    settings = {"prompts": {"deepseek_ocr_prompt": "legacy prompt"}, "deepseek_ocr": {"model_id": "custom/old"}}
    ctx = ProviderContext(console=Console(file=io.StringIO()), args=SimpleNamespace(), config=settings)
    media = MediaContext(uri=str(tmp_path / "page.pdf"), mime="application/pdf", modality=MediaModality.DOCUMENT,
                         sha256hash="", extras={"output_dir": tmp_path})
    prompts = PromptContext(system="", user="legacy prompt")
    seen = []
    monkeypatch.setattr(deepseek, "attempt_deepseek_ocr", lambda **kw: seen.append(kw["model_id"]) or "old")
    monkeypatch.setattr(deepseek_native, "attempt_deepseek_native", lambda **_: pytest.fail("unexpected native inference"))
    provider = deepseek.DeepSeekOCRProvider(ctx)
    assert provider.attempt(media, prompts).raw == "old"
    settings["deepseek_ocr"]["model_id"] = "deepseek-ai/DeepSeek-OCR-2"
    legacy = deepseek.DeepSeekLegacyOCRProvider(ctx)
    assert legacy.get_prompts(media.mime)[1] == "legacy prompt"
    assert legacy.attempt(media, prompts).raw == "old"
    settings["deepseek_ocr"]["model_id"] = "deepseek-community/DeepSeek-OCR-2"
    assert legacy.attempt(media, prompts).raw == "old"
    assert seen == ["custom/old", "deepseek-ai/DeepSeek-OCR-2", "deepseek-ai/DeepSeek-OCR-2"]
    ctx.args.local_runtime_backend = "openai"
    monkeypatch.setattr(provider, "attempt_via_openai_backend", lambda *_: "remote")
    assert provider.attempt(media, prompts) == "remote"


def test_legacy_official_weights_and_code_are_pinned_but_local_paths_are_not_replaced(tmp_path, monkeypatch):
    from module.providers.ocr import deepseek

    calls = []
    monkeypatch.setattr(deepseek, "snapshot_download_with_reporting", lambda repo, **kw: calls.append((repo, kw)) or str(tmp_path))
    deepseek._official_legacy_snapshot.cache_clear()
    try:
        assert deepseek._legacy_source(str(tmp_path)) == str(tmp_path)
        assert deepseek._legacy_source("private/old") == "private/old"
        assert deepseek._legacy_source("deepseek-ai/DeepSeek-OCR-2") == str(tmp_path)
        assert deepseek._legacy_source("deepseek-ai/DeepSeek-OCR-2") == str(tmp_path)
        assert len(calls) == 1 and calls[0][1]["revision"] == "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
    finally:
        deepseek._official_legacy_snapshot.cache_clear()
