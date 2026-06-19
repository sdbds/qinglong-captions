import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import module.providers.ocr.paddle as paddle_module
from module.providers.base import MediaContext, MediaModality, ProviderContext
from module.providers.ocr.paddle import PaddleOCRProvider


def make_ctx(config):
    return ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config=config,
        args=SimpleNamespace(
            ocr_model="paddle_ocr",
            document_image=True,
            dir_name=False,
            openai_model_name="",
            local_runtime_backend="",
        ),
    )


def make_media(path: Path, output_dir: Path | None = None):
    return MediaContext(
        uri=str(path),
        mime="image/png",
        sha256hash="",
        modality=MediaModality.IMAGE,
        extras={"output_dir": output_dir or path.with_suffix("")},
    )


def test_model_tier_maps_to_ppocrv6_det_and_rec_names():
    assert paddle_module._resolve_ppocrv6_model_names("tiny") == (
        "PP-OCRv6_tiny_det",
        "PP-OCRv6_tiny_rec",
    )
    assert paddle_module._resolve_ppocrv6_model_names("small") == (
        "PP-OCRv6_small_det",
        "PP-OCRv6_small_rec",
    )
    assert paddle_module._resolve_ppocrv6_model_names("medium") == (
        "PP-OCRv6_medium_det",
        "PP-OCRv6_medium_rec",
    )


def test_default_backend_uses_ppocrv6_onnx_without_importing_paddle_vl(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    captured = {}

    class FakeResult:
        json = {
            "rec_texts": ["", "Hello", "世界", "  "],
            "rec_scores": [0.9, 0.8],
            "rec_boxes": [[1, 2, 3, 4]],
            "rec_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
        }

        def print(self):
            pass

        def save_to_json(self, save_path):
            captured["json_path"] = save_path

        def save_to_img(self, save_path):
            captured["img_path"] = save_path

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def predict(self, input):
            captured["input"] = input
            return [FakeResult()]

    monkeypatch.setattr(paddle_module, "_import_paddle_ocr", lambda: FakePaddleOCR)
    monkeypatch.setattr(
        paddle_module,
        "_import_paddle_ocr_vl",
        lambda: (_ for _ in ()).throw(AssertionError("PaddleOCRVL must not be imported by the default backend")),
    )
    monkeypatch.setattr(paddle_module, "display_markdown", lambda **_kwargs: None)

    provider = PaddleOCRProvider(make_ctx({"paddle_ocr": {}}))
    result = provider.attempt(make_media(image_path, tmp_path / "out"), SimpleNamespace())

    assert captured["kwargs"]["engine"] == "onnxruntime"
    assert captured["kwargs"]["ocr_version"] == "PP-OCRv6"
    assert captured["kwargs"]["text_detection_model_name"] == "PP-OCRv6_medium_det"
    assert captured["kwargs"]["text_recognition_model_name"] == "PP-OCRv6_medium_rec"
    assert captured["input"] == str(image_path)
    assert result.raw == "Hello\n世界"
    assert result.metadata["backend"] == "ppocrv6_onnx"
    assert result.metadata["engine"] == "onnxruntime"
    assert result.metadata["model_tier"] == "medium"
    assert result.metadata["rec_texts"] == ["", "Hello", "世界", "  "]
    assert result.metadata["rec_scores"] == [0.9, 0.8]
    assert result.metadata["rec_boxes"] == [[1, 2, 3, 4]]
    assert result.metadata["rec_polys"] == [[[1, 2], [3, 2], [3, 4], [1, 4]]]


def test_configured_model_tier_is_passed_to_ppocrv6_onnx(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    captured = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def predict(self, input):
            return [SimpleNamespace(json={"rec_texts": ["tiny"]}, print=lambda: None)]

    monkeypatch.setattr(paddle_module, "_import_paddle_ocr", lambda: FakePaddleOCR)
    monkeypatch.setattr(paddle_module, "display_markdown", lambda **_kwargs: None)

    provider = PaddleOCRProvider(make_ctx({"paddle_ocr": {"model_tier": "tiny"}}))
    result = provider.attempt(make_media(image_path, tmp_path / "out"), SimpleNamespace())

    assert captured["kwargs"]["text_detection_model_name"] == "PP-OCRv6_tiny_det"
    assert captured["kwargs"]["text_recognition_model_name"] == "PP-OCRv6_tiny_rec"
    assert result.metadata["model_tier"] == "tiny"


def test_default_config_separates_ppocrv6_ocr_and_vl_pipeline_kwargs():
    layout_keys = {
        "format_block_content",
        "layout_merge_bboxes_mode",
        "layout_nms",
        "layout_threshold",
        "layout_unclip_ratio",
        "markdown_ignore_labels",
        "merge_layout_blocks",
        "use_chart_recognition",
        "use_layout_detection",
        "use_ocr_for_image_block",
        "use_queues",
        "use_seal_recognition",
    }

    for config_path in (ROOT / "config" / "model.toml", ROOT / "config" / "config.toml"):
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        paddle_config = data["paddle_ocr"]
        pipeline = paddle_config["pipeline"]
        vl_pipeline = paddle_config["vl_pipeline"]

        assert pipeline["use_doc_orientation_classify"] is False
        assert pipeline["use_doc_unwarping"] is False
        assert pipeline["use_textline_orientation"] is False
        assert not (layout_keys & set(pipeline))
        assert vl_pipeline["use_layout_detection"] is False
        assert vl_pipeline["format_block_content"] is True
        assert vl_pipeline["merge_layout_blocks"] is True
        assert vl_pipeline["markdown_ignore_labels"] == []
        assert "use_queues" in vl_pipeline


def test_ppocrv6_onnx_does_not_pass_layout_only_kwargs(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    captured = {}

    class FakePaddleOCR:
        allowed_kwargs = {
            "cpu_threads",
            "device",
            "enable_hpi",
            "engine",
            "lang",
            "ocr_version",
            "text_det_limit_side_len",
            "text_detection_model_name",
            "text_recognition_model_name",
            "use_doc_unwarping",
        }

        def __init__(self, **kwargs):
            unknown = set(kwargs) - self.allowed_kwargs
            if unknown:
                raise ValueError(f"Unknown argument: {sorted(unknown)[0]}")
            captured["kwargs"] = kwargs

        def predict(self, input):
            return [SimpleNamespace(json={"rec_texts": ["ok"]}, print=lambda: None)]

    monkeypatch.setattr(paddle_module, "_import_paddle_ocr", lambda: FakePaddleOCR)
    monkeypatch.setattr(paddle_module, "display_markdown", lambda **_kwargs: None)

    provider = PaddleOCRProvider(
        make_ctx(
            {
                "paddle_ocr": {
                    "pipeline": {
                        "cpu_threads": "4",
                        "device": "cpu",
                        "lang": "ch",
                        "text_det_limit_side_len": "960",
                        "use_doc_unwarping": "false",
                        "enable_hpi": "false",
                        "use_layout_detection": False,
                        "use_chart_recognition": True,
                        "use_seal_recognition": True,
                        "use_ocr_for_image_block": True,
                        "format_block_content": True,
                        "merge_layout_blocks": True,
                        "markdown_ignore_labels": ["formula"],
                    }
                }
            }
        )
    )

    result = provider.attempt(make_media(image_path, tmp_path / "out"), SimpleNamespace())

    assert result.raw == "ok"
    assert captured["kwargs"]["cpu_threads"] == 4
    assert captured["kwargs"]["text_det_limit_side_len"] == 960
    assert captured["kwargs"]["use_doc_unwarping"] is False
    assert "use_layout_detection" not in captured["kwargs"]
    assert "use_chart_recognition" not in captured["kwargs"]
    assert "use_seal_recognition" not in captured["kwargs"]
    assert "use_ocr_for_image_block" not in captured["kwargs"]
    assert "format_block_content" not in captured["kwargs"]
    assert "merge_layout_blocks" not in captured["kwargs"]
    assert "markdown_ignore_labels" not in captured["kwargs"]


def test_direct_onnx_backend_loads_det_and_rec_bundles_from_shared_runtime(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    captured = []

    def fake_load_single_model_bundle(*, spec, runtime_config, logger=None):
        captured.append((spec, runtime_config, logger))
        return SimpleNamespace()

    monkeypatch.setattr(paddle_module, "load_single_model_bundle", fake_load_single_model_bundle)

    provider = PaddleOCRProvider(
        make_ctx(
            {
                "paddle_ocr": {"backend": "ppocrv6_direct_onnx", "model_tier": "small"},
                "onnx_runtime": {
                    "defaults": {"execution_provider": "auto"},
                    "paddle_ocr": {"execution_provider": "cpu"},
                },
            }
        )
    )

    with pytest.raises(NotImplementedError, match="experimental"):
        provider.attempt(make_media(image_path, tmp_path / "out"), SimpleNamespace())

    assert [item[0].repo_id for item in captured] == [
        "PaddlePaddle/PP-OCRv6_small_det_onnx",
        "PaddlePaddle/PP-OCRv6_small_rec_onnx",
    ]
    assert [item[0].onnx_filename for item in captured] == ["inference.onnx", "inference.onnx"]
    assert [item[0].support_files for item in captured] == [
        {"inference_config": "inference.yml", "inference_program": "inference.json"},
        {"inference_config": "inference.yml", "inference_program": "inference.json"},
    ]
    assert all(item[1].execution_provider == "cpu" for item in captured)


def test_paddle_vl_native_backend_reads_vl_pipeline_kwargs(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    captured = {}

    class FakeLegacyResult:
        def print(self):
            pass

        def save_to_markdown(self, save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)
            (Path(save_path) / "legacy.md").write_text("legacy document parse", encoding="utf-8")

    class FakePaddleOCRVL:
        allowed_kwargs = {
            "cpu_threads",
            "device",
            "format_block_content",
            "layout_merge_bboxes_mode",
            "layout_nms",
            "layout_threshold",
            "markdown_ignore_labels",
            "merge_layout_blocks",
            "use_chart_recognition",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_layout_detection",
            "use_ocr_for_image_block",
            "use_queues",
            "use_seal_recognition",
        }

        def __init__(self, **kwargs):
            unknown = set(kwargs) - self.allowed_kwargs
            if unknown:
                raise ValueError(f"Unknown argument: {sorted(unknown)[0]}")
            captured["kwargs"] = kwargs

        def predict(self, input):
            return [FakeLegacyResult()]

    monkeypatch.setattr(paddle_module, "_import_paddle_ocr_vl", lambda: FakePaddleOCRVL)
    monkeypatch.setattr(paddle_module, "display_markdown", lambda **_kwargs: None)

    provider = PaddleOCRProvider(
        make_ctx(
            {
                "paddle_ocr": {
                    "backend": "paddle_vl_native",
                    "pipeline": {
                        "cpu_threads": "4",
                        "device": "cpu",
                        "use_doc_orientation_classify": "false",
                        "use_doc_unwarping": "false",
                        "use_textline_orientation": "false",
                    },
                    "vl_pipeline": {
                        "format_block_content": "true",
                        "layout_merge_bboxes_mode": "large",
                        "layout_nms": "0.5",
                        "layout_threshold": "0.3",
                        "markdown_ignore_labels": ["footer"],
                        "merge_layout_blocks": "true",
                        "use_chart_recognition": "false",
                        "use_layout_detection": "false",
                        "use_ocr_for_image_block": "true",
                        "use_queues": "false",
                        "use_seal_recognition": "false",
                    },
                }
            }
        )
    )

    result = provider.attempt(make_media(image_path, tmp_path / "out"), SimpleNamespace())

    assert result.raw == "legacy document parse"
    assert captured["kwargs"]["cpu_threads"] == 4
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["layout_threshold"] == 0.3
    assert captured["kwargs"]["layout_nms"] == 0.5
    assert captured["kwargs"]["use_layout_detection"] is False
    assert captured["kwargs"]["use_ocr_for_image_block"] is True
    assert captured["kwargs"]["use_queues"] is False
    assert captured["kwargs"]["markdown_ignore_labels"] == ["footer"]
    assert "use_textline_orientation" not in captured["kwargs"]


def test_paddle_vl_pdf_kwargs_use_latest_restructure_names():
    provider = PaddleOCRProvider(
        make_ctx(
            {
                "paddle_ocr": {
                    "backend": "paddle_vl_native",
                    "pdf": {
                        "merge_tables": "true",
                        "relevel_titles": "false",
                        "concatenate_pages": "true",
                    },
                }
            }
        )
    )

    config = provider._get_paddle_config()

    assert config["pdf_kwargs"] == {
        "merge_tables": True,
        "relevel_titles": False,
        "concatenate_pages": True,
    }


def test_paddle_vl_native_backend_remains_explicit_and_deprecated(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    captured = {}

    class FakeLegacyResult:
        def print(self):
            pass

        def save_to_markdown(self, save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)
            (Path(save_path) / "legacy.md").write_text("legacy document parse", encoding="utf-8")

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def predict(self, input):
            captured["input"] = input
            return [FakeLegacyResult()]

    monkeypatch.setattr(paddle_module, "_import_paddle_ocr_vl", lambda: FakePaddleOCRVL)
    monkeypatch.setattr(
        paddle_module,
        "_import_paddle_ocr",
        lambda: (_ for _ in ()).throw(AssertionError("PaddleOCR should not be used by paddle_vl_native")),
    )
    monkeypatch.setattr(paddle_module, "display_markdown", lambda **_kwargs: None)

    provider = PaddleOCRProvider(make_ctx({"paddle_ocr": {"backend": "paddle_vl_native"}}))
    result = provider.attempt(make_media(image_path, tmp_path / "out"), SimpleNamespace())

    assert result.raw == "legacy document parse"
    assert result.metadata["backend"] == "paddle_vl_native"
    assert result.metadata["deprecated"] is True
