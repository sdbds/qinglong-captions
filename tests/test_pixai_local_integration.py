"""Opt-in checks against a downloaded official model and its local ONNX export."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.integration, pytest.mark.optional_runtime]


@pytest.fixture(scope="module")
def local_bundle():
    directory = os.environ.get("PIXAI_TEST_MODEL_DIR")
    image_path = os.environ.get("PIXAI_TEST_IMAGE")
    if not directory or not image_path:
        pytest.skip("Set PIXAI_TEST_MODEL_DIR and PIXAI_TEST_IMAGE for local model verification")
    import torch
    from module.onnx_runtime import OnnxRuntimeConfig
    from module.wdtagger.pixai import PIXAI_REPO_ID, PIXAI_SOURCE_REPO_ID, load_pixai_bundle

    torch.set_num_threads(8)
    directory = Path(directory).resolve()
    runtime = OnnxRuntimeConfig.from_mapping(
        {"execution_provider": "cuda", "intra_op_num_threads": 8, "cuda": {"use_tf32": False, "tunable_op_tuning_enable": False}}
    )
    repo_id = PIXAI_SOURCE_REPO_ID if directory.name == PIXAI_SOURCE_REPO_ID.replace("/", "_") else PIXAI_REPO_ID
    bundle = load_pixai_bundle(repo_id=repo_id, model_dir=directory.parent, runtime_config=runtime)
    with Image.open(image_path) as image:
        sample = image.copy()
    return directory, bundle, sample


def test_real_image_matches_original_pytorch(local_bundle):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from module.wdtagger.pixai import preprocess_pixai_images
    from utils.onnx_export_pixai import compare_logits

    directory, bundle, image = local_bundle
    source_directory = Path(os.environ.get("PIXAI_TEST_SOURCE_DIR", str(directory)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoImageProcessor.from_pretrained(source_directory, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(source_directory, trust_remote_code=True, local_files_only=True).float().cuda().eval()
    images = [image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
    pixels = processor(images)["pixel_values"]
    np.testing.assert_array_equal(preprocess_pixai_images(images), pixels.numpy())
    with torch.inference_mode():
        expected = model(pixels.cuda()).float().cpu().numpy()
    actual = bundle.session.run(["logits"], {"pixel_values": pixels.numpy()})[0]
    report = compare_logits(expected, actual)
    print("REAL_IMAGE_PARITY " + json.dumps(report))
    assert report["max_probability_error"] < 1e-4
    assert actual.shape == (2, 30877)
    del model
    torch.cuda.empty_cache()


def test_runner_writes_real_tags_and_lance_without_touching_source(local_bundle, tmp_path, monkeypatch):
    from module.wdtagger import model_loader, runner
    from module.wdtagger.cli import finalize_args, setup_parser
    from module.wdtagger.outputs import read_sidecar_caption

    directory, bundle, image = local_bundle
    image.save(tmp_path / "sample.png")
    monkeypatch.setattr(model_loader, "load_pixai_bundle", lambda **kwargs: bundle)
    monkeypatch.setattr(runner, "_print_tag_frequencies", lambda _freq: None)
    args = finalize_args(
        setup_parser().parse_args(
            [
                str(tmp_path),
                "--repo_id=bdsqlsz/pixai-tagger-v1.0-ONNX",
                f"--model_dir={directory.parent}",
                "--batch_size=1",
                "--use_rating_tags",
            ]
        )
    )
    runner.main(args)
    tags = read_sidecar_caption(str(tmp_path / "sample.png"), ".txt")
    assert tags and len(tags[0].split(", ")) > 3
    assert (tmp_path / "tags.json").is_file()
    import lance

    rows = lance.dataset(str(tmp_path / "dataset.lance")).to_table(columns=["captions"]).to_pylist()
    assert rows[0]["captions"] == tags
    print(f"REAL_RUNNER_TAGS {len(tags[0].split(', '))}")
