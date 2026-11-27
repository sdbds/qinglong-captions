import io
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from module.providers.base import PromptContext, ProviderContext
from module.providers.local_vlm.moondream import MoondreamProvider

MODEL = "moondream3.1-9B-A2B"
REPO = "moondream/" + MODEL
CONFIG = Path(__file__).resolve().parents[1] / "config"


def provider(section, **args):
    return MoondreamProvider(
        ProviderContext(
            console=Console(file=io.StringIO()),
            config={"moondream": section},
            args=SimpleNamespace(**args),
        )
    )


def test_catalog_and_both_config_formats_select_31():
    for filename in ("model.toml",):
        with (CONFIG / filename).open("rb") as file:
            section = tomllib.load(file)["moondream"]
        assert provider(section).model_id == REPO
        if filename == "model.toml":
            assert REPO in {item["model_id"] for item in section["model_list"].values()}
    assert provider({}).model_id == REPO


@pytest.fixture
def image_media(tmp_path):
    path = tmp_path / "sign.png"
    Image.new("RGB", (32, 32), "white").save(path)
    return provider({}).prepare_media(str(path), "image/png", SimpleNamespace())


@pytest.fixture
def photon(monkeypatch):
    import safetensors

    monkeypatch.setattr("torch.version.cuda", "12.8")
    weights = SimpleNamespace(safetensors=safetensors)
    monkeypatch.setitem(sys.modules, "kestrel.models.moondream", SimpleNamespace(weights=weights))
    model = Mock(spec=["caption", "query", "point", "detect", "close"])
    model.caption.side_effect = lambda image, length: {"caption": f"{length} description"}
    model.query.return_value = {"answer": "OPEN"}
    model.point.return_value = {"points": [{"x": 0.25, "y": 0.5}]}
    model.detect.return_value = {"objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]}
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "moondream", SimpleNamespace(photon=factory))
    # Never let a regression in model routing download a checkpoint in a unit test.
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: pytest.fail("3.1 used the Transformers loader")
    )
    yield factory, model
    MoondreamProvider.unload_model()


@pytest.mark.parametrize("model_id", [REPO, MODEL])
def test_31_native_tasks_share_one_local_engine(model_id, image_media, photon):
    factory, model = photon
    vlm = provider({"model_id": model_id, "tasks": "all", "reasoning": True})
    prompt = PromptContext("", "sign")
    result = vlm.attempt(image_media, prompt).raw
    for expected in (
        "short description",
        "long description",
        "OPEN",
        "x=0.250, y=0.500",
        "x_min=0.100, y_min=0.200, x_max=0.300, y_max=0.400",
    ):
        assert expected in result
    assert model.query.call_args.kwargs["reasoning"] is True

    ocr = provider({"model_id": model_id, "tasks": "all", "reasoning": True}, ocr_model="moondream")
    assert ocr.attempt(image_media, prompt).raw == "OPEN"
    assert "Markdown" in model.query.call_args.kwargs["question"]
    assert model.query.call_args.kwargs["reasoning"] is False
    assert model.caption.call_count == 2
    factory.assert_called_once_with(MODEL)
    MoondreamProvider.unload_model(model_id)
    model.close.assert_called_once_with()


def test_31_missing_extra_reports_how_to_install_it(monkeypatch, image_media):
    monkeypatch.setitem(sys.modules, "moondream", None)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: pytest.fail("3.1 tried to fall back to Transformers")
    )
    with pytest.raises(ImportError, match="--extra moondream"):
        provider({"model_id": REPO}).attempt(image_media, PromptContext("", "sign"))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Photon wheels ship CUDA 12 kernels")
def test_31_wrong_cuda_build_fails_before_loading_weights(photon, monkeypatch, image_media):
    factory, _ = photon
    monkeypatch.setattr("torch.version.cuda", "13.0")
    with pytest.raises(RuntimeError, match="CUDA 12"):
        provider({}).attempt(image_media, PromptContext("", "sign"))
    factory.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows weight loading workaround")
def test_31_windows_weight_reader_is_local_to_model_startup(photon, image_media):
    from kestrel.models.moondream import weights

    original = weights.safetensors
    open_weights = Mock()
    weights.safetensors = reader = SimpleNamespace(safe_open=open_weights)
    factory, _ = photon

    def failed_start(model):
        weights.safetensors.safe_open("weights", framework="pt")
        raise RuntimeError("startup failed")

    factory.side_effect = failed_start
    with pytest.raises(RuntimeError, match="startup failed"):
        provider({}).attempt(image_media, PromptContext("", "sign"))
    open_weights.assert_called_once_with("weights", framework="pt", backend="pread")
    assert weights.safetensors is reader
    assert original is sys.modules["safetensors"]


def test_remote_image_ocr_does_not_load_photon_for_unused_spatial_tasks(monkeypatch, image_media):
    vlm = provider({"tasks": "all"}, ocr_model="moondream", local_runtime_backend="openai")
    monkeypatch.setattr(vlm, "_get_or_load_model", lambda: pytest.fail("unexpected local model load"))
    monkeypatch.setattr("module.providers.backends.OpenAIChatRuntime.complete", lambda self, messages, **kw: "OPEN")
    assert vlm.attempt(image_media, PromptContext("", "Find the sign")).raw == "OPEN"


def test_preview_still_uses_transformers_without_photon(monkeypatch, image_media):
    model_id = "moondream/moondream3-preview"
    model = Mock(spec=["query", "compile", "eval"])
    model.eval.return_value = model
    model.query.return_value = {"answer": "OPEN"}
    loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "moondream", None)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", loader)
    vlm = provider({"model_id": model_id, "tasks": "query"})
    try:
        for _ in range(2):
            assert "OPEN" in vlm.attempt(image_media, PromptContext("", "Read")).raw
        assert loader.call_args.args == (model_id,)
        assert loader.call_args.kwargs["trust_remote_code"] is True
        model.compile.assert_called_once_with()
    finally:
        vlm.unload_model(model_id)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("RUN_MOONDREAM_31_SMOKE") != "1", reason="requires Photon and 10.5 GB of weights")
def test_31_local_gpu_reads_known_sign(tmp_path):
    path = tmp_path / "sign.png"
    with Image.new("RGB", (512, 320), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.ellipse((32, 40, 160, 168), fill="red")
        draw.text((256, 240), "OPEN 42", font=ImageFont.load_default(size=60), anchor="mm", fill="black")
        image.save(path)
    vlm = provider({"tasks": "all"})
    media = vlm.prepare_media(str(path), "image/png", vlm.ctx.args)
    try:
        result = vlm.attempt(media, PromptContext("", "red circle")).raw
        for expected in ("### Short Description", "### Long Description", "## Query Result", "Point 1:", "Object 1:"):
            assert expected in result, result
        answer = provider({"tasks": "query"}).attempt(media, PromptContext("", "What color is the circle?")).raw
        assert "red" in answer.lower(), answer
        ocr = provider({}, ocr_model="moondream").attempt(media, PromptContext("", "")).raw
        assert ocr.strip("#* \n").upper() == "OPEN 42", ocr
        print(result, answer, ocr, sep="\n")
    finally:
        vlm.unload_model()
