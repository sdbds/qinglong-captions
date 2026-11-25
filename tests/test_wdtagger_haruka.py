import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

REPO = "KaraKaraWitch/haruka-tagger-large"


def cli(*options):
    from module.wdtagger.cli import finalize_args, setup_parser

    return finalize_args(setup_parser().parse_args(["images", f"--repo_id={REPO}", *options]))


def metadata():
    return {
        "img_size": 4,
        "total_tags": 7,
        "idx_to_tag": {
            "6": "series",
            "1": "era_2023_2025",
            "4": "rating_g",
            "2": "an_artist",
            "0": "solo",
            "5": "alice",
            "3": "rating_s",
        },
        "tag_to_category": {
            "solo": "general",
            "era_2023_2025": "year",
            "alice": "character",
            "an_artist": "artist",
            "series": "copyright",
            "rating_s": "rating",
            "rating_g": "rating",
        },
    }


class Session:
    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=["batch_size", 3, 4, 4], type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits", shape=["batch_size", 7], type="tensor(float)")]

    def run(self, outputs, feed):
        assert outputs == ["logits"]
        self.pixels = feed["input"]
        return [np.tile([1, -1, -1, 0, -2, -1, -1], (len(self.pixels), 1)).astype(np.float32)]


def load_cached(tmp_path, args, session, *, data=None):
    from module.wdtagger.model_loader import load_model_and_tags

    args.model_dir = str(tmp_path)
    directory = tmp_path / "KaraKaraWitch_haruka-tagger-large"
    directory.mkdir(exist_ok=True)
    meta_path = directory / "haruka-tagger_metadata.json"
    meta_path.write_text(json.dumps(metadata() if data is None else data), encoding="utf-8")

    def load_bundle(*, spec, runtime_config, **kwargs):
        assert spec.repo_id == REPO and spec.local_dir == directory
        assert spec.onnx_filename == "haruka-tagger.onnx"
        assert set(spec.support_files.values()) == {meta_path.name}
        assert runtime_config.force_download == args.force_download
        return SimpleNamespace(
            session=session,
            providers=("CPUExecutionProvider",),
            support_paths={key: directory / name for key, name in spec.support_files.items()},
        )

    return load_model_and_tags(
        args,
        load_single_model_bundle_fn=load_bundle,
        hf_hub_download_fn=lambda **kw: pytest.fail(f"Unexpected legacy download: {kw['filename']}"),
    )


def test_haruka_batch_uses_graph_input_rgb_letterbox_and_one_sigmoid(tmp_path):
    from module.wdtagger.preprocess import process_batch

    session, context, labels, _ = load_cached(tmp_path, cli("--force_download"), Session())
    assert labels.names == ["solo", "era_2023_2025", "an_artist", "rating_s", "rating_g", "alice", "series"]
    assert labels.category_indices["rating"].tolist() == [3, 4]
    images = [Image.new("RGB", (8, 4), "red"), Image.new("RGBA", (4, 4), (0, 0, 0, 0))]
    scores = process_batch(images, session, context)
    assert session.pixels.shape == (2, 3, 4, 4)
    assert session.pixels.dtype == np.float32 and session.pixels.flags.c_contiguous
    white = [2.2489083, 2.4285714, 2.64]
    np.testing.assert_allclose(session.pixels[0, :, 0, 0], white, rtol=1e-6)
    np.testing.assert_allclose(session.pixels[0, :, 1, 0], [2.2489083, -2.0357143, -1.8044444], rtol=1e-6)
    np.testing.assert_allclose(session.pixels[1, :, 2, 2], white, rtol=1e-6)
    assert scores.shape == (2, 7)
    np.testing.assert_allclose(scores[:, :4], [[0.7310586, 0.2689414, 0.2689414, 0.5]] * 2, rtol=1e-6)


@pytest.mark.parametrize("invalid", ["missing_index", "missing_category", "wrong_size", "wrong_width"])
def test_haruka_rejects_graph_vocabulary_mismatch(tmp_path, invalid):
    data = metadata()
    if invalid == "missing_index":
        data["idx_to_tag"]["8"] = data["idx_to_tag"].pop("1")
    elif invalid == "missing_category":
        del data["tag_to_category"]["series"]
    elif invalid == "wrong_size":
        data["img_size"] = 8
    else:
        data["idx_to_tag"]["7"] = "extra"
        data["tag_to_category"]["extra"] = "general"
        data["total_tags"] = 8
    with pytest.raises(ValueError, match="Haruka"):
        load_cached(tmp_path, cli(), Session(), data=data)


def test_haruka_cli_uses_category_defaults_and_preserves_zero_overrides():
    args = cli()
    assert args.batch_size == 1
    assert [
        args.general_threshold,
        args.character_threshold,
        args.copyright_threshold,
        args.artist_threshold,
        args.year_threshold,
        args.rating_threshold,
    ] == [0.35, 0.35, 0.35, 0.25, 0.25, 0.50]
    args = cli("--thresh=0.8", "--year_threshold=0", "--batch_size=2")
    assert (args.batch_size, args.artist_threshold, args.year_threshold, args.rating_threshold) == (2, 0.8, 0.0, 0.8)


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], ["solo", "an_artist", "era_2023_2025"]),
        (["--use_rating_tags"], ["rating_s", "solo", "an_artist", "era_2023_2025"]),
        (["--use_rating_tags_as_last_tag"], ["solo", "an_artist", "era_2023_2025", "rating_s"]),
        (["--thresh=0.8", "--character_threshold=0"], ["alice"]),
    ],
)
def test_haruka_runner_writes_thresholded_tags_to_sidecar_and_json(tmp_path, options, expected):
    import lance
    import pyarrow as pa

    from module.wdtagger.runner import main

    path = tmp_path / "sample.png"
    Image.new("RGBA", (6, 4), (0, 0, 0, 0)).save(path)
    lance.write_dataset(pa.table({"uris": [str(path)], "mime": ["image/png"]}), tmp_path / "dataset.lance")
    args = cli("--overwrite", "--lance_update_mode=none", *options)
    args.train_data_dir = str(tmp_path)
    session = Session()
    main(args, load_model_and_tags_fn=lambda args: load_cached(tmp_path, args, session))
    assert path.with_suffix(".txt").read_text(encoding="utf-8").split(", ") == expected
    categories = json.loads((tmp_path / "tags.json").read_text(encoding="utf-8"))[str(path)]
    assert categories["year"] == [tag for tag in expected if tag.startswith("era_")]
    assert categories["artist"] == [tag for tag in expected if tag == "an_artist"]
    np.testing.assert_allclose(session.pixels[0, :, 2, 2], [2.2489083, 2.4285714, 2.64], rtol=1e-6)
