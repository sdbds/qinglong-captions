import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def pixai_module():
    assert importlib.util.find_spec("module.wdtagger.pixai") is not None, "PixAI ONNX adapter is missing"
    return importlib.import_module("module.wdtagger.pixai")


def test_ci_declares_pixai_preprocessing_dependencies():
    import toml
    from packaging.requirements import Requirement

    project = toml.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = project["project"]["dependencies"] + project["dependency-groups"]["test"]
    assert {"torch", "torchvision"} <= {Requirement(requirement).name for requirement in requirements}


def model_config():
    return {
        "model_type": "cls_vitdet",
        "img_size": 4,
        "num_classes": 7,
        "tags": ["1girl", "solo", "alice", "series", "watercolor", "highres", "rating:g"],
        "tags_split": [["general", 2], ["character", 1], ["copyright", 1], ["style", 1], ["meta", 1], ["rating", 1]],
        "category_best_threshold": {
            "general": 0.17,
            "character": 0.27,
            "copyright": 0.24,
            "style": 0.15,
            "meta": 0.17,
            "rating": 0.41,
        },
    }


def test_pixai_config_preserves_official_tag_order_and_categories(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(model_config()), encoding="utf-8")
    labels, context = pixai_module().load_pixai_config(path)
    assert labels.names == ["1girl", "solo", "alice", "series", "watercolor", "highres", "rating:g"]
    assert labels.category_indices["copyright"].tolist() == [3]
    assert labels.category_indices["style"].tolist() == [4]
    assert context.image_size == 4


def test_pixai_config_rejects_mismatched_vocabulary(tmp_path):
    data = model_config()
    data["tags"].pop()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="tags_split"):
        pixai_module().load_pixai_config(path)


def test_pixai_preprocess_preserves_rgb_and_black_padding():
    result = pixai_module().preprocess_pixai_images([Image.new("RGB", (4, 2), (255, 0, 0))], 4)
    assert result.shape == (1, 3, 4, 4)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, :, 0], -np.ones((3, 4)))
    np.testing.assert_array_equal(result[0, :, 1, 0], [1, -1, -1])


def test_pixai_preprocess_composites_transparency_on_white():
    result = pixai_module().preprocess_pixai_images([Image.new("RGBA", (4, 4), (255, 0, 0, 0))], 4)
    np.testing.assert_array_equal(result, np.ones((1, 3, 4, 4)))


def test_pixai_category_thresholds_allow_zero_overrides():
    from module.wdtagger.cli import finalize_args, setup_parser

    defaults = finalize_args(setup_parser().parse_args(["images", "--repo_id=bdsqlsz/pixai-tagger-v1.0-ONNX"]))
    assert defaults.general_threshold == 0.17
    assert defaults.character_threshold == 0.27
    assert defaults.style_threshold == 0.15
    assert defaults.copyright_threshold == 0.24
    assert defaults.meta_threshold == 0.17
    assert defaults.rating_threshold == 0.41
    overrides = finalize_args(
        setup_parser().parse_args(["images", "--repo_id=bdsqlsz/pixai-tagger-v1.0-ONNX", "--thresh=0.2", "--style_threshold=0"])
    )
    assert overrides.general_threshold == 0.2
    assert overrides.character_threshold == 0.2
    assert overrides.style_threshold == 0


def test_pixai_selection_uses_all_six_thresholds_and_keeps_meta_without_rating(tmp_path):
    from module.wdtagger.cli import finalize_args, setup_parser
    from module.wdtagger.tag_assembly import assemble_final_tags, get_tags_official

    path = tmp_path / "config.json"
    path.write_text(json.dumps(model_config()), encoding="utf-8")
    labels, _ = pixai_module().load_pixai_config(path)
    args = finalize_args(setup_parser().parse_args(["images", "--repo_id=bdsqlsz/pixai-tagger-v1.0-ONNX"]))
    thresholds = {category: getattr(args, f"{category}_threshold") for category in model_config()["category_best_threshold"]}
    result = get_tags_official(
        np.array([0.18, 0.1, 0.28, 0.25, 0.16, 0.18, 0.40]),
        labels,
        args.general_threshold,
        args.character_threshold,
        True,
        False,
        False,
        category_thresholds=thresholds,
    )
    assert result["rating"] == []
    assert [tag for tag, _ in result["style"]] == ["watercolor"]
    assert [tag for tag, _ in result["copyright"]] == ["series"]
    assert assemble_final_tags(result, args, {}) == ["1girl", "alice", "series", "highres", "watercolor"]


def test_pixai_image_loading_keeps_alpha_and_skips_broken_images(tmp_path):
    from module.wdtagger import preprocess

    path = tmp_path / "transparent.png"
    Image.new("RGBA", (4, 4), (0, 255, 0, 0)).save(path)
    assert hasattr(preprocess, "load_pil_batch"), "Image loader must preserve transparency for PixAI"
    uris, images = preprocess.load_pil_batch([str(tmp_path / "missing.png"), str(path)], preserve_alpha=True)
    assert uris == [str(path)]
    assert images[0].mode == "RGBA"


def test_pixai_process_batch_applies_sigmoid_once():
    from module.wdtagger.preprocess import process_batch

    class Session:
        def run(self, outputs, inputs):
            assert outputs == ["logits"]
            assert inputs["pixel_values"].shape == (1, 3, 4, 4)
            return [np.array([[0.0, np.log(3.0)]], dtype=np.float32)]

    result = process_batch([Image.new("RGB", (4, 4))], Session(), pixai_module().PixaiInferenceContext(image_size=4))
    np.testing.assert_allclose(result, [[0.5, 0.75]])


@pytest.fixture
def stub_download_bundle(tmp_path, monkeypatch):
    import module.onnx_runtime as runtime

    directory = tmp_path / "bdsqlsz_pixai-tagger-v1.0-ONNX"
    directory.mkdir()
    (directory / "model.onnx").write_bytes(b"first graph")
    (directory / "config.json").write_text(json.dumps(model_config()), encoding="utf-8")
    captured = {}
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="pixel_values", shape=["batch", 3, 4, 4])],
        get_outputs=lambda: [SimpleNamespace(name="logits", shape=["batch", 7])],
    )

    def load_bundle(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            session=session,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            model_path=directory / "model.onnx",
            support_paths={"config": directory / "config.json"},
        )

    monkeypatch.setattr(runtime, "load_single_model_bundle", load_bundle)
    return directory, captured, session


def test_pixai_downloads_onnx_and_config_without_revision_pin(tmp_path, stub_download_bundle):
    directory, captured, session = stub_download_bundle
    bundle = pixai_module().load_pixai_bundle(model_dir=tmp_path, runtime_config=None)
    assert captured["spec"].repo_id == "bdsqlsz/pixai-tagger-v1.0-ONNX"
    assert captured["spec"].onnx_filename == "model.onnx"
    assert captured["spec"].support_files == {"config": "config.json"}
    assert captured["spec"].revision is None
    assert captured["spec"].local_dir == directory
    assert captured["runtime_config"].execution_provider == "cuda"
    assert captured["runtime_config"].provider_options["cuda"]["use_tf32"] is False
    assert bundle.session is session
    assert bundle.label_data.names[4] == "watercolor"
    assert not (directory / "model.json").exists()


def test_pixai_refresh_accepts_replaced_weights_without_hash_checks(tmp_path, stub_download_bundle, monkeypatch):
    import module.onnx_runtime as runtime
    from utils import file_hash

    directory, captured, session = stub_download_bundle
    (directory / "model.onnx").write_bytes(b"updated weights with a different size")
    (directory / "model.json").write_text('{"onnx_sha256": "outdated", "onnx_size_bytes": 1}', encoding="utf-8")

    def reject_hash(*args, **kwargs):
        pytest.fail("User-owned PixAI model must not be hash-checked")

    monkeypatch.setattr(file_hash, "sha256_file", reject_hash)
    monkeypatch.setattr(pixai_module(), "sha256_file", reject_hash, raising=False)
    bundle = pixai_module().load_pixai_bundle(model_dir=tmp_path, runtime_config=runtime.OnnxRuntimeConfig(force_download=True))
    assert captured["runtime_config"].force_download is True
    assert bundle.session is session


@pytest.mark.parametrize(
    "repo,want", [("bdsqlsz/pixai-tagger-v1.0-ONNX", 1), ("pixai-labs/pixai-tagger-v1.0", 1), ("cella110n/cl_tagger_v2", 4)]
)
def test_cli_batch_default_is_model_aware_without_overriding_user(repo, want):
    from module.wdtagger.cli import finalize_args, setup_parser

    assert finalize_args(setup_parser().parse_args(["images", f"--repo_id={repo}"])).batch_size == want
    assert finalize_args(setup_parser().parse_args(["images", f"--repo_id={repo}", "--batch_size=2"])).batch_size == 2


@pytest.mark.parametrize("options,expected", [([], 0.8), (["--thresh=0.4"], 0.4), (["--thresh=0.4", "--general_threshold=0"], 0.0)])
def test_downloaded_config_controls_defaults_but_explicit_cli_wins(tmp_path, options, expected):
    from module.wdtagger.cli import finalize_args, setup_parser
    from module.wdtagger.tag_assembly import get_tags_official

    data = model_config()
    data["category_best_threshold"]["general"] = 0.8
    config = tmp_path / "config.json"
    config.write_text(json.dumps(data), encoding="utf-8")
    pixai = pixai_module()
    labels, context = pixai.load_pixai_config(config)
    args = finalize_args(setup_parser().parse_args(["images", "--repo_id=bdsqlsz/pixai-tagger-v1.0-ONNX", *options]))
    thresholds = pixai.resolve_pixai_thresholds(args, context)
    assert thresholds["general"] == expected
    tags = get_tags_official(
        np.array([0.6, 0.0, 0, 0, 0, 0, 0]), labels, 0.17, 0.27, False, False, False, category_thresholds=thresholds
    )
    assert [tag for tag, _ in tags["general"]] == ([] if expected == 0.8 else ["1girl"])
