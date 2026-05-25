import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def _import_wdtagger_with_stubbed_runtime(monkeypatch, tmp_path):
    sys.modules.pop("utils.wdtagger", None)

    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_lance_import = types.ModuleType("module.lanceImport")

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_lance_import.load_data = lambda *args, **kwargs: []
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    return importlib.import_module("utils.wdtagger")


def test_wdtagger_scans_caption_state_in_python(monkeypatch, tmp_path):
    wdtagger = _import_wdtagger_with_stubbed_runtime(monkeypatch, tmp_path)
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_c = tmp_path / "c.jpg"

    batch = pa.record_batch(
        {
            "uris": pa.array([str(image_a), str(image_b), str(image_c)]),
            "mime": pa.array(["image/jpeg", "image/jpeg", "image/png"]),
            "captions": pa.array([[], ["already tagged"], None], type=pa.list_(pa.string())),
        }
    )
    scanner_kwargs = []

    class FakeScanner:
        def to_batches(self):
            return [batch]

    class FakeDataset:
        def scanner(self, **kwargs):
            scanner_kwargs.append(kwargs)
            return FakeScanner()

    args = SimpleNamespace(batch_size=32, overwrite=False)

    batches = list(wdtagger._scan_wdtagger_candidate_batches(FakeDataset(), args))

    assert batches[0]["uris"].to_pylist() == [str(image_a), str(image_c)]
    assert scanner_kwargs[0]["columns"] == ["uris", "mime", "captions"]
    assert scanner_kwargs[0]["filter"] == "mime LIKE 'image/%'"
    assert "array_length" not in scanner_kwargs[0]["filter"]
    assert scanner_kwargs[0]["late_materialization"] is False


def test_wdtagger_scan_skips_existing_sidecar_captions(monkeypatch, tmp_path):
    wdtagger = _import_wdtagger_with_stubbed_runtime(monkeypatch, tmp_path)
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.with_suffix(".txt").write_text("already tagged", encoding="utf-8")

    batch = pa.record_batch(
        {
            "uris": pa.array([str(image_a), str(image_b)]),
            "mime": pa.array(["image/jpeg", "image/png"]),
            "captions": pa.array([[], []], type=pa.list_(pa.string())),
        }
    )

    class FakeScanner:
        def to_batches(self):
            return [batch]

    class FakeDataset:
        def scanner(self, **_kwargs):
            return FakeScanner()

    args = SimpleNamespace(batch_size=32, overwrite=False, caption_extension=".txt")

    batches = list(wdtagger._scan_wdtagger_candidate_batches(FakeDataset(), args))

    assert batches[0]["uris"].to_pylist() == [str(image_b)]


def test_wdtagger_overwrite_scan_does_not_read_captions(monkeypatch, tmp_path):
    wdtagger = _import_wdtagger_with_stubbed_runtime(monkeypatch, tmp_path)

    batch = pa.record_batch(
        {
            "uris": pa.array(["a.jpg", "b.jpg"]),
            "mime": pa.array(["image/jpeg", "image/png"]),
        }
    )
    scanner_kwargs = []

    class FakeScanner:
        def to_batches(self):
            return [batch]

    class FakeDataset:
        def scanner(self, **kwargs):
            scanner_kwargs.append(kwargs)
            return FakeScanner()

    args = SimpleNamespace(batch_size=32, overwrite=True)

    assert wdtagger._count_wdtagger_candidate_rows(FakeDataset(), args) == 2
    assert scanner_kwargs[0]["columns"] == ["uris", "mime"]
    assert scanner_kwargs[0]["filter"] == "mime LIKE 'image/%'"
    assert scanner_kwargs[0]["late_materialization"] is False


def test_wdtagger_load_model_and_tags_uses_single_model_bundle(monkeypatch, tmp_path):
    sys.modules.pop("utils.wdtagger", None)

    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_lance_import = types.ModuleType("module.lanceImport")

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_lance_import.load_data = lambda *args, **kwargs: []
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    wdtagger = importlib.import_module("utils.wdtagger")

    captured = {}
    repo_dir = tmp_path / "SmilingWolf_wd-v1-4-moat-tagger-v2"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "selected_tags.csv").write_text(
        "tag_id,name,category,count\n0,safe,9,1\n1,1girl,0,1\n2,character_a,4,1\n",
        encoding="utf-8",
    )

    fake_session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input")])

    def fake_loader(*, spec, runtime_config, logger=None):
        captured["spec"] = spec
        captured["runtime"] = runtime_config
        captured["logger"] = logger
        return SimpleNamespace(
            session=fake_session,
            providers=("CPUExecutionProvider",),
            input_metas=(SimpleNamespace(name="input"),),
        )

    monkeypatch.setattr(wdtagger, "load_single_model_bundle", fake_loader, raising=False)

    args = SimpleNamespace(
        repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2",
        model_dir=str(tmp_path),
        force_download=False,
        remove_parents_tag=False,
    )

    session, input_name, label_data, parent_to_child_map = wdtagger.load_model_and_tags(args)

    assert session is fake_session
    assert input_name == "input"
    assert label_data.names[0] == "safe"
    assert parent_to_child_map == {}
    assert captured["spec"].bundle_key == "wdtagger:SmilingWolf/wd-v1-4-moat-tagger-v2"
    assert callable(captured["logger"])


def test_wdtagger_load_model_and_tags_uses_siglip2_bundle_for_explicit_cl_tagger_v2_repo(monkeypatch, tmp_path):
    sys.modules.pop("utils.wdtagger", None)

    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_lance_import = types.ModuleType("module.lanceImport")

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_lance_import.load_data = lambda *args, **kwargs: []
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    wdtagger = importlib.import_module("utils.wdtagger")

    captured = {}
    fake_session = SimpleNamespace()
    fake_context = wdtagger.Siglip2InferenceContext(processor="processor")

    def fake_siglip2_bundle(*, repo_id, model_dir, runtime_config, version, force_download, logger=None):
        captured["bundle"] = {
            "repo_id": repo_id,
            "model_dir": model_dir,
            "runtime_config": runtime_config,
            "version": version,
            "force_download": force_download,
            "logger": logger,
        }
        return SimpleNamespace(
            session=fake_session,
            providers=("CPUExecutionProvider",),
            inference_context=fake_context,
            vocabulary=SimpleNamespace(
                names=["masterpiece", "1girl"],
                category_indices={
                    "quality": np.array([0], dtype=np.int64),
                    "general": np.array([1], dtype=np.int64),
                    "rating": np.array([], dtype=np.int64),
                    "character": np.array([], dtype=np.int64),
                    "copyright": np.array([], dtype=np.int64),
                    "artist": np.array([], dtype=np.int64),
                    "meta": np.array([], dtype=np.int64),
                    "model": np.array([], dtype=np.int64),
                },
                tag_index_to_category={0: "quality", 1: "general"},
            ),
        )

    monkeypatch.setattr(wdtagger, "load_cl_tagger_v2_bundle", fake_siglip2_bundle, raising=False)
    monkeypatch.setattr(
        wdtagger,
        "load_single_model_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy ONNX bundle should not be used for cl_tagger alias")),
        raising=False,
    )

    args = SimpleNamespace(
        repo_id="cella110n/cl_tagger_v2",
        model_dir=str(tmp_path),
        force_download=False,
        remove_parents_tag=False,
    )

    session, input_name, label_data, parent_to_child_map = wdtagger.load_model_and_tags(args)

    assert session is fake_session
    assert input_name is fake_context
    assert label_data.names == ["masterpiece", "1girl"]
    assert label_data.category_indices["quality"].tolist() == [0]
    assert parent_to_child_map == {}
    assert captured["bundle"]["repo_id"] == "cella110n/cl_tagger_v2"
    assert captured["bundle"]["version"] == "v1_04"
    assert callable(captured["bundle"]["logger"])


def test_wdtagger_finalize_args_infers_cl_tagger_v2_threshold(monkeypatch, tmp_path):
    sys.modules.pop("utils.wdtagger", None)

    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_lance_import = types.ModuleType("module.lanceImport")

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_lance_import.load_data = lambda *args, **kwargs: []
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    wdtagger = importlib.import_module("utils.wdtagger")
    parser = wdtagger.setup_parser()

    current_siglip2_args = wdtagger.finalize_args(parser.parse_args(["./datasets", "--repo_id=cella110n/cl_tagger_v2"]))
    assert current_siglip2_args.cl_tagger_v2_version == "v1_04"
    assert current_siglip2_args.thresh == 0.5
    assert current_siglip2_args.general_threshold == 0.5
    assert current_siglip2_args.character_threshold == 0.5

    v102_siglip2_args = wdtagger.finalize_args(
        parser.parse_args(["./datasets", "--repo_id=cella110n/cl_tagger_v2", "--cl_tagger_v2_version=1.02"])
    )
    assert v102_siglip2_args.cl_tagger_v2_version == "v1_02"
    assert v102_siglip2_args.thresh == 0.9
    assert v102_siglip2_args.general_threshold == 0.9
    assert v102_siglip2_args.character_threshold == 0.9

    legacy_args = wdtagger.finalize_args(parser.parse_args(["./datasets"]))
    assert legacy_args.thresh == 0.35


def test_wdtagger_get_tags_official_preserves_dynamic_categories(monkeypatch, tmp_path):
    sys.modules.pop("utils.wdtagger", None)

    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_lance_import = types.ModuleType("module.lanceImport")

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_lance_import.load_data = lambda *args, **kwargs: []
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    wdtagger = importlib.import_module("utils.wdtagger")
    labels = wdtagger.LabelData(
        names=["1girl", "unclassified_tag"],
        category_indices={
            "general": np.array([0], dtype=np.int64),
            "unknown": np.array([1], dtype=np.int64),
        },
        tag_index_to_category={0: "general", 1: "unknown"},
    )

    result = wdtagger.get_tags_official(
        np.array([0.9, 0.95], dtype=np.float32),
        labels,
        gen_threshold=0.5,
        char_threshold=0.5,
        use_rating_tags=False,
        use_quality_tags=False,
        use_model_tags=False,
    )

    assert result["general"] == [("1girl", float(np.float32(0.9)))]
    assert result["unknown"] == [("unclassified_tag", float(np.float32(0.95)))]

    tags_json = wdtagger.assemble_tags_json(
        result,
        add_tags_threshold=False,
        remove_parents_tag=False,
    )
    assert tags_json["unknown"] == ["unclassified_tag"]

    final_tags = wdtagger.assemble_final_tags(
        result,
        SimpleNamespace(
            add_tags_threshold=False,
            use_quality_tags=False,
            use_rating_tags=False,
            use_model_tags=False,
            use_rating_tags_as_last_tag=False,
            character_tags_first=False,
            frequency_tags=False,
            always_first_tags="",
            remove_parents_tag=False,
        ),
        parent_to_child_map={},
        tag_freq={},
    )
    assert final_tags == ["1girl"]
