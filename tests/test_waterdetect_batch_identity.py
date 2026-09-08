import importlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image


@pytest.fixture
def waterdetect(monkeypatch):
    parent = importlib.import_module("module")
    monkeypatch.setattr(parent, "waterdetect", None, raising=False)
    monkeypatch.setitem(sys.modules, "module.waterdetect", None)
    sys.modules.pop("module.waterdetect")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = object
    fake_lance = types.ModuleType("lance")
    fake_lance.LanceDataset = type("LanceDataset", (), {})
    fake_importer = types.ModuleType("module.lanceImport")
    fake_importer.transform2lance = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_importer)
    return importlib.import_module("module.waterdetect")


def _configure_inference(waterdetect, monkeypatch, uris, *, result_rows=None):
    table = pa.table({"uris": uris, "mime": ["image/png"] * len(uris)})
    dataset = SimpleNamespace(
        to_table=lambda **_kwargs: table,
        scanner=lambda **_kwargs: SimpleNamespace(to_batches=lambda: table.to_batches()),
    )
    monkeypatch.setattr(waterdetect, "transform2lance", lambda *_args, **_kwargs: dataset)

    class PixelTensor:
        def __init__(self, image):
            self.image = image

        def numpy(self):
            return np.asarray(self.image, dtype=np.float32)

    class Session:
        def run(self, _outputs, inputs):
            probabilities = [0.9 if image.mean() > 128 else 0.1 for image in inputs["pixels"]]
            if result_rows is not None:
                probabilities = probabilities[:result_rows]
            return [np.array([[1 - probability, probability] for probability in probabilities])]

    monkeypatch.setattr(
        waterdetect,
        "processor",
        lambda *, images, **_kwargs: {"pixel_values": [PixelTensor(images)]},
        raising=False,
    )
    monkeypatch.setattr(waterdetect, "load_model", lambda _args: (Session(), "pixels"))


@pytest.mark.parametrize("broken_index", [0, 1, 2])
def test_decode_failure_keeps_watermark_scores_with_their_sources(waterdetect, monkeypatch, tmp_path, broken_index):
    sources = [tmp_path / f"image-{index}.png" for index in range(3)]
    for index, source in enumerate(sources):
        if index == broken_index:
            source.write_bytes(b"invalid image fixture")
        else:
            Image.new("RGB", (4, 4), "white" if index == 1 else "black").save(source)
    _configure_inference(waterdetect, monkeypatch, [str(source) for source in sources])

    waterdetect.main(SimpleNamespace(train_data_dir=str(tmp_path), batch_size=3, thresh=0.5))

    results = json.loads((tmp_path / "watermark_detection_results.json").read_text(encoding="utf-8"))
    expected = {source.name: "0.9000" if index == 1 else "0.1000"
                for index, source in enumerate(sources) if index != broken_index}
    assert {name: value[:6] for name, value in results.items()} == expected
    for index, source in enumerate(sources):
        if index == broken_index:
            assert not (tmp_path / "watermarked" / source.name).exists()
            assert not (tmp_path / "no_watermark" / source.name).exists()
        else:
            partition = "watermarked" if index == 1 else "no_watermark"
            assert (tmp_path / partition / source.name).read_bytes() == source.read_bytes()


def test_probability_count_mismatch_is_rejected_before_any_classification(waterdetect, monkeypatch, tmp_path):
    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    for source in sources:
        Image.new("RGB", (4, 4), "white").save(source)
    _configure_inference(waterdetect, monkeypatch, [str(source) for source in sources], result_rows=1)

    with pytest.raises(ValueError, match="probability row count"):
        waterdetect.main(SimpleNamespace(train_data_dir=str(tmp_path), batch_size=2, thresh=0.5))

    assert not (tmp_path / "watermarked").exists()
    assert not (tmp_path / "no_watermark").exists()
    assert not (tmp_path / "watermark_detection_results.json").exists()


def test_all_decode_failures_produce_no_classification(waterdetect, monkeypatch, tmp_path):
    source = tmp_path / "broken.png"
    source.write_bytes(b"invalid image fixture")
    _configure_inference(waterdetect, monkeypatch, [str(source)])

    waterdetect.main(SimpleNamespace(train_data_dir=str(tmp_path), batch_size=1, thresh=0.5))

    assert json.loads((tmp_path / "watermark_detection_results.json").read_text(encoding="utf-8")) == {}
