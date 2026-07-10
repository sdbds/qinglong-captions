import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _write_preprocessor_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "image_processor_type": "ViTImageProcessor",
                "do_resize": True,
                "size": {"height": 1024, "width": 1024},
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_default_spec_uses_uploaded_musvit_onnx_repo(tmp_path):
    from module.sheet_music_musvit import DEFAULT_MUSVIT_ONNX_REPO_ID, MuSViTOnnxEmbedder

    config_path = _write_preprocessor_config(tmp_path / "preprocessor_config.json")
    captured = {}

    def fake_bundle_loader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model_path=tmp_path / "model.onnx",
            support_paths={"preprocessor_config": config_path},
            session=SimpleNamespace(get_outputs=lambda: [SimpleNamespace(name="last_hidden_state")]),
            providers=("CPUExecutionProvider",),
            input_metas=(SimpleNamespace(name="pixel_values"),),
        )

    MuSViTOnnxEmbedder(model_dir=tmp_path, bundle_loader=fake_bundle_loader)

    spec = captured["spec"]
    assert spec.repo_id == DEFAULT_MUSVIT_ONNX_REPO_ID == "bdsqlsz/musvit-onnx"
    assert spec.onnx_filename == "model.onnx"
    assert spec.support_files == {"preprocessor_config": "preprocessor_config.json"}


def test_preprocess_image_resizes_to_float_chw(tmp_path):
    from module.sheet_music_musvit import load_musvit_preprocessor_config, preprocess_image

    config_path = _write_preprocessor_config(tmp_path / "preprocessor_config.json")
    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 4), color=(255, 128, 0)).save(image_path)

    config = load_musvit_preprocessor_config(config_path)
    tensor = preprocess_image(image_path, config, preprocess_mode="page_resize")

    assert tensor.shape == (3, 1024, 1024)
    assert tensor.dtype == np.float32
    assert float(tensor.max()) <= 1.0
    assert float(tensor.min()) >= 0.0


def test_embed_file_writes_npz_and_metadata(tmp_path):
    from module.sheet_music_musvit import MuSViTOnnxEmbedder

    config_path = _write_preprocessor_config(tmp_path / "preprocessor_config.json")
    image_path = tmp_path / "score.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)

    class FakeSession:
        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="last_hidden_state")]

        @staticmethod
        def run(output_names, feed):
            assert output_names == ["last_hidden_state"]
            assert feed["pixel_values"].shape == (1, 3, 1024, 1024)
            return [np.zeros((1, 4097, 768), dtype=np.float32)]

    def fake_bundle_loader(**kwargs):
        return SimpleNamespace(
            model_path=tmp_path / "model.onnx",
            support_paths={"preprocessor_config": config_path},
            session=FakeSession(),
            providers=("CPUExecutionProvider",),
            input_metas=(SimpleNamespace(name="pixel_values"),),
        )

    embedder = MuSViTOnnxEmbedder(model_dir=tmp_path, bundle_loader=fake_bundle_loader)
    result = embedder.embed_file(image_path, output_dir=tmp_path / "out", overwrite=True)

    assert result.embedding_path.exists()
    assert result.metadata_path.exists()
    arrays = np.load(result.embedding_path)
    assert arrays["last_hidden_state"].shape == (4097, 768)
    assert arrays["cls_embedding"].shape == (768,)
    assert arrays["patch_embeddings"].shape == (4096, 768)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_path"] == str(image_path)
    assert metadata["providers"] == ["CPUExecutionProvider"]


def test_embed_inputs_runs_session_in_batches(tmp_path):
    from module.sheet_music_musvit import MuSViTOnnxEmbedder

    config_path = _write_preprocessor_config(tmp_path / "preprocessor_config.json")
    input_dir = tmp_path / "scores"
    input_dir.mkdir()
    for index in range(3):
        Image.new("RGB", (8, 8), color=(255, 255, 255)).save(input_dir / f"score_{index}.png")

    run_shapes = []

    class FakeSession:
        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="last_hidden_state")]

        @staticmethod
        def run(output_names, feed):
            batch_shape = feed["pixel_values"].shape
            run_shapes.append(batch_shape)
            return [np.zeros((batch_shape[0], 4097, 768), dtype=np.float32)]

    def fake_bundle_loader(**kwargs):
        return SimpleNamespace(
            model_path=tmp_path / "model.onnx",
            support_paths={"preprocessor_config": config_path},
            session=FakeSession(),
            providers=("CPUExecutionProvider",),
            input_metas=(SimpleNamespace(name="pixel_values"),),
        )

    embedder = MuSViTOnnxEmbedder(model_dir=tmp_path, bundle_loader=fake_bundle_loader)
    results = embedder.embed_inputs(input_dir, output_dir=tmp_path / "out", batch_size=2, overwrite=True)

    assert run_shapes == [(2, 3, 1024, 1024), (1, 3, 1024, 1024)]
    assert len(results) == 3
    assert all(result.embedding_path.exists() for result in results)


def test_embed_inputs_expands_pdf_pages_and_writes_page_metadata(tmp_path):
    from module.sheet_music_musvit import MuSViTOnnxEmbedder

    config_path = _write_preprocessor_config(tmp_path / "preprocessor_config.json")
    pdf_path = tmp_path / "score.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    live_pages = []
    closed_pages = []
    captured_dpi = []
    live_pages_during_inference = []

    def tracked_page(page_number: int, size: tuple[int, int]):
        image = Image.new("RGB", size, color=(255, 255, 255))
        live_pages.append(page_number)
        original_close = image.close

        def close():
            closed_pages.append(page_number)
            if page_number in live_pages:
                live_pages.remove(page_number)
            original_close()

        image.close = close
        return SimpleNamespace(
            pdf_path=pdf_path,
            page_index=page_number - 1,
            page_number=page_number,
            page_count=2,
            image=image,
            size=size,
            dpi=123,
            image_format="PNG",
        )

    def fake_pdf_renderer(path, *, dpi, image_format):
        captured_dpi.append((Path(path), dpi, image_format))
        yield tracked_page(1, (12, 8))
        yield tracked_page(2, (10, 6))

    class FakeSession:
        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="last_hidden_state")]

        @staticmethod
        def run(output_names, feed):
            live_pages_during_inference.append(tuple(live_pages))
            assert feed["pixel_values"].shape == (1, 3, 1024, 1024)
            return [np.zeros((1, 4097, 768), dtype=np.float32)]

    def fake_bundle_loader(**kwargs):
        return SimpleNamespace(
            model_path=tmp_path / "model.onnx",
            support_paths={"preprocessor_config": config_path},
            session=FakeSession(),
            providers=("CPUExecutionProvider",),
            input_metas=(SimpleNamespace(name="pixel_values"),),
        )

    embedder = MuSViTOnnxEmbedder(model_dir=tmp_path, bundle_loader=fake_bundle_loader)
    results = embedder.embed_inputs(
        pdf_path,
        output_dir=tmp_path / "out",
        batch_size=1,
        pdf_dpi=123,
        pdf_renderer=fake_pdf_renderer,
        overwrite=True,
    )

    assert captured_dpi == [(pdf_path, 123, "PNG")]
    assert live_pages_during_inference == [(1,), (2,)]
    assert closed_pages == [1, 2]
    assert live_pages == []
    assert [result.output_dir.name for result in results] == ["page_0001", "page_0002"]
    assert (tmp_path / "out" / "score.pdf" / "page_0001" / "embedding.npz").exists()
    metadata = json.loads((tmp_path / "out" / "score.pdf" / "page_0002" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_path"] == str(pdf_path)
    assert metadata["source_type"] == "pdf_page"
    assert metadata["pdf_page_index"] == 1
    assert metadata["pdf_page_number"] == 2
    assert metadata["pdf_page_count"] == 2
    assert metadata["rendered_page_size"] == [10, 6]


def test_build_parser_defaults_to_uploaded_repo():
    from module.sheet_music_musvit import DEFAULT_MUSVIT_ONNX_REPO_ID, build_parser

    args = build_parser().parse_args(["input.png"])

    assert args.repo_id == DEFAULT_MUSVIT_ONNX_REPO_ID
    assert args.model_dir == "huggingface"
    assert args.batch_size == 1
    assert args.pdf_dpi == 144
