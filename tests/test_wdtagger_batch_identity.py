import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest


_MISSING = object()
_WDTAGGER_MODULES = (
    "module.wdtagger.preprocess",
    "module.wdtagger.lance_io",
    "module.wdtagger.runner",
)


@pytest.fixture(autouse=True)
def _isolate_wdtagger_modules():
    saved_modules = {name: sys.modules.get(name, _MISSING) for name in _WDTAGGER_MODULES}
    parent = importlib.import_module("module.wdtagger")
    saved_attrs = {name.rpartition(".")[2]: getattr(parent, name.rpartition(".")[2], _MISSING) for name in _WDTAGGER_MODULES}
    yield

    for module_name, saved in saved_modules.items():
        sys.modules.pop(module_name, None)
        if saved is not _MISSING:
            sys.modules[module_name] = saved
    for attr_name, saved in saved_attrs.items():
        if saved is _MISSING:
            if hasattr(parent, attr_name):
                delattr(parent, attr_name)
        else:
            setattr(parent, attr_name, saved)


def _stub_wdtagger_runtime(monkeypatch, tmp_path):
    fake_cv2 = types.ModuleType("cv2")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_lance_import = types.ModuleType("module.lanceImport")

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_lance_import.load_data = lambda *args, **kwargs: []
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    sys.modules.pop("module.wdtagger.runner", None)
    sys.modules.pop("module.wdtagger.preprocess", None)


@pytest.mark.parametrize("rejected_index", [0, 1, 2])
def test_load_and_preprocess_batch_returns_only_valid_uri_image_pairs(monkeypatch, tmp_path, rejected_index):
    _stub_wdtagger_runtime(monkeypatch, tmp_path)
    preprocess = importlib.import_module("module.wdtagger.preprocess")
    uris = [str(tmp_path / f"image-{index}.png") for index in range(3)]

    class FakeImage:
        def __init__(self, uri):
            self.uri = uri

        def convert(self, _mode):
            return self

    def fake_open(uri):
        if uri == uris[rejected_index]:
            raise OSError("corrupt image")
        return FakeImage(uri)

    monkeypatch.setattr(preprocess.Image, "open", fake_open)
    monkeypatch.setattr(preprocess, "preprocess_image", lambda image, _is_cl_tagger: f"pixels:{image.uri}")

    valid_uris, images = preprocess.load_and_preprocess_batch(uris)

    expected_uris = [uri for index, uri in enumerate(uris) if index != rejected_index]
    assert valid_uris == expected_uris
    assert images == [f"pixels:{uri}" for uri in expected_uris]


class _FakeProgress:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.console = SimpleNamespace(print=lambda *_args, **_kwargs: None)
        self.advances = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def add_task(self, *_args, **_kwargs):
        return 1

    def update(self, _task, *, advance):
        self.advances.append(advance)


def _runner_args(tmp_path):
    return SimpleNamespace(
        train_data_dir=str(tmp_path),
        lance_update_mode="merge",
        repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2",
        merge_batch_size=100,
        general_threshold=None,
        character_threshold=None,
        thresh=0.5,
        use_rating_tags=False,
        use_quality_tags=False,
        use_model_tags=False,
        append_tags=False,
        caption_extension=".txt",
        caption_separator=", ",
        add_tags_threshold=False,
        remove_parents_tag=False,
    )


def _configure_runner(monkeypatch, tmp_path, *, uris, valid_uris, images, probabilities):
    _stub_wdtagger_runtime(monkeypatch, tmp_path)
    runner = importlib.import_module("module.wdtagger.runner")
    dataset = object()
    batch = pa.record_batch({"uris": pa.array(uris)})
    sidecar_calls = []
    json_calls = []
    lance_calls = []

    _FakeProgress.instances.clear()
    monkeypatch.setattr(runner, "Progress", _FakeProgress)
    monkeypatch.setattr(runner, "resolve_dataset", lambda _path: SimpleNamespace(dataset=dataset, dataset_path=None))
    monkeypatch.setattr(runner, "resolve_lance_rebuild_source", lambda *_args: None)
    monkeypatch.setattr(runner, "count_wdtagger_candidate_rows", lambda *_args: len(uris))
    monkeypatch.setattr(runner, "scan_wdtagger_candidate_batches", lambda *_args: [batch])
    monkeypatch.setattr(runner, "process_tags", lambda *_args: {})
    monkeypatch.setattr(runner, "is_cl_tagger_v2_repo", lambda _repo_id: False)
    monkeypatch.setattr(runner, "load_and_preprocess_batch", lambda *_args: (valid_uris, images))
    monkeypatch.setattr(runner, "process_batch", lambda *_args: probabilities)
    monkeypatch.setattr(runner, "get_tags_official", lambda probability, *_args: probability)
    monkeypatch.setattr(
        runner,
        "assemble_final_tags",
        lambda probability, *_args: [f"score-{int(probability[0])}"],
    )
    monkeypatch.setattr(
        runner,
        "assemble_tags_json",
        lambda probability, **_kwargs: {"general": [f"score-{int(probability[0])}"]},
    )
    monkeypatch.setattr(
        runner,
        "write_sidecar_caption",
        lambda path, tags, **_kwargs: sidecar_calls.append((path, tags)),
    )
    monkeypatch.setattr(runner, "write_tags_json", lambda _path, records: json_calls.append(records))
    monkeypatch.setattr(runner, "merge_caption_updates", lambda _dataset, updates: lance_calls.extend(updates))
    monkeypatch.setattr(runner, "update_wdtagger_tag", lambda _dataset: None)
    monkeypatch.setattr(runner, "_print_tag_frequencies", lambda _frequencies: None)

    return runner, sidecar_calls, json_calls, lance_calls


@pytest.mark.parametrize("rejected_index", [0, 1, 2])
def test_runner_keeps_uri_probability_identity_when_decode_fails(monkeypatch, tmp_path, rejected_index):
    uris = [str(tmp_path / f"image-{index}.png") for index in range(3)]
    valid_uris = [uri for index, uri in enumerate(uris) if index != rejected_index]
    images = [f"pixels:{uri}" for uri in valid_uris]
    probabilities = np.array([[index + 10] for index in range(len(valid_uris))])
    runner, sidecar_calls, json_calls, lance_calls = _configure_runner(
        monkeypatch,
        tmp_path,
        uris=uris,
        valid_uris=valid_uris,
        images=images,
        probabilities=probabilities,
    )

    runner.main(_runner_args(tmp_path), load_model_and_tags_fn=lambda _args: (object(), "input", object(), {}))

    expected = [(uri, [f"score-{index + 10}"]) for index, uri in enumerate(valid_uris)]
    assert sidecar_calls == expected
    assert lance_calls == expected
    assert json_calls == [
        {uri: {"general": [f"score-{index + 10}"]} for index, uri in enumerate(valid_uris)}
    ]
    assert uris[rejected_index] not in json_calls[0]
    assert _FakeProgress.instances[0].advances == [len(uris)]


def test_runner_rejects_uri_image_count_mismatch(monkeypatch, tmp_path):
    uris = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    runner, *_ = _configure_runner(
        monkeypatch,
        tmp_path,
        uris=uris,
        valid_uris=uris,
        images=["one-image"],
        probabilities=np.array([[0.9]]),
    )

    with pytest.raises(ValueError, match="valid URI count .* image count"):
        runner.main(_runner_args(tmp_path), load_model_and_tags_fn=lambda _args: (object(), "input", object(), {}))


def test_runner_rejects_uri_probability_count_mismatch(monkeypatch, tmp_path):
    uris = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    runner, *_ = _configure_runner(
        monkeypatch,
        tmp_path,
        uris=uris,
        valid_uris=uris,
        images=["image-a", "image-b"],
        probabilities=np.array([[0.9]]),
    )

    with pytest.raises(ValueError, match="valid URI count .* probability row count"):
        runner.main(_runner_args(tmp_path), load_model_and_tags_fn=lambda _args: (object(), "input", object(), {}))
