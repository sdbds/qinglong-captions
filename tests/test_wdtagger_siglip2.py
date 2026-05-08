import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from utils.wdtagger_siglip2 import (
    CL_TAGGER_V2_BACKEND_REPO,
    CL_TAGGER_V2_DEFAULT_VERSION,
    CL_TAGGER_V2_OPTION,
    CL_TAGGER_V2_PROCESSOR_REPO,
    Siglip2InferenceContext,
    _default_snapshot_download,
    _load_siglip2_processor,
    default_cl_tagger_v2_threshold,
    download_cl_tagger_v2_artifacts,
    load_cl_tagger_v2_bundle,
    load_cl_tagger_v2_metadata,
    normalize_cl_tagger_v2_version,
    process_siglip2_batch,
)


def _write_siglip2_snapshot(
    root: Path,
    *,
    version: str = CL_TAGGER_V2_DEFAULT_VERSION,
    stem: str | None = None,
    metadata: dict | None = None,
) -> tuple[Path, Path, Path | None]:
    if stem is None:
        stem = "step_486342" if version == "v1_02" else "step_270385"

    version_dir = root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / f"{stem}.onnx"
    vocab_path = version_dir / f"{stem}_vocabulary.json"
    model_path.write_text("onnx", encoding="utf-8")
    (version_dir / f"{stem}.onnx.data").write_text("external", encoding="utf-8")
    vocab_path.write_text(
        json.dumps(
            {
                "idx_to_tag": {"0": "masterpiece", "1": "1girl", "2": "character_a", "3": "safe"},
                "tag_to_category": {
                    "masterpiece": "Quality",
                    "1girl": "General",
                    "character_a": "Character",
                    "safe": "Rating",
                },
            }
        ),
        encoding="utf-8",
    )
    metadata_path = None
    if metadata is not None:
        metadata_path = version_dir / f"{stem}_metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, vocab_path, metadata_path


def _install_fake_transformer_loader(monkeypatch, **attrs):
    fake_module = ModuleType("utils.transformer_loader")
    for name, value in attrs.items():
        setattr(fake_module, name, value)
    monkeypatch.setitem(sys.modules, "utils.transformer_loader", fake_module)
    return fake_module


def test_cl_tagger_v2_defaults_track_current_space_version():
    assert CL_TAGGER_V2_DEFAULT_VERSION == "v1_02"
    assert normalize_cl_tagger_v2_version("1.02") == "v1_02"
    assert normalize_cl_tagger_v2_version("v1_2") == "v1_02"
    assert default_cl_tagger_v2_threshold("v1_01") == 0.6
    assert default_cl_tagger_v2_threshold("v1_02") == 0.9


def test_default_snapshot_download_uses_rich_reporting(monkeypatch):
    captured = {}

    def fake_snapshot_download_with_reporting(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["kwargs"] = dict(kwargs)
        return "C:/cache/demo"

    _install_fake_transformer_loader(
        monkeypatch,
        snapshot_download_with_reporting=fake_snapshot_download_with_reporting,
    )

    result = _default_snapshot_download(
        repo_id="demo/model",
        allow_patterns=["v1_01/*"],
        local_dir="C:/cache",
        force_download=True,
        token="hf_test",
    )

    assert result == "C:/cache/demo"
    assert captured == {
        "repo_id": "demo/model",
        "kwargs": {
            "allow_patterns": ["v1_01/*"],
            "local_dir": "C:/cache",
            "force_download": True,
            "token": "hf_test",
        },
    }


def test_download_cl_tagger_v2_artifacts_uses_explicit_v2_cache_dir(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    def fake_snapshot_download(*, repo_id, allow_patterns, local_dir, force_download, token=None):
        captured["snapshot"] = {
            "repo_id": repo_id,
            "allow_patterns": list(allow_patterns),
            "local_dir": local_dir,
            "force_download": force_download,
            "token": token,
        }
        _write_siglip2_snapshot(Path(local_dir))
        return local_dir

    resolved_repo_id, cache_dir, model_path, vocab_path, metadata_path = download_cl_tagger_v2_artifacts(
        repo_id=CL_TAGGER_V2_OPTION,
        model_dir=tmp_path,
        snapshot_downloader=fake_snapshot_download,
    )

    assert resolved_repo_id == CL_TAGGER_V2_BACKEND_REPO
    assert cache_dir == tmp_path / "cella110n_cl_tagger_v2"
    assert model_path == cache_dir / CL_TAGGER_V2_DEFAULT_VERSION / "step_486342.onnx"
    assert vocab_path == cache_dir / CL_TAGGER_V2_DEFAULT_VERSION / "step_486342_vocabulary.json"
    assert metadata_path is None
    assert captured["snapshot"]["repo_id"] == CL_TAGGER_V2_BACKEND_REPO
    assert captured["snapshot"]["allow_patterns"] == [f"{CL_TAGGER_V2_DEFAULT_VERSION}/*"]
    assert captured["snapshot"]["token"] == "hf_test"


def test_load_siglip2_processor_local_hit_skips_reporting(monkeypatch):
    calls = []

    @contextmanager
    def fail_if_reporting_starts():
        raise AssertionError("hf_download_reporting should not start for local cache hits")
        yield

    _install_fake_transformer_loader(monkeypatch, hf_download_reporting=fail_if_reporting_starts)

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(repo_id, local_files_only=False, token=None):
            calls.append((repo_id, local_files_only, token))
            return "processor"

    result = _load_siglip2_processor(
        processor_repo="demo/processor",
        processor_loader=FakeProcessorLoader,
        token="hf_test",
    )

    assert result == "processor"
    assert calls == [("demo/processor", True, "hf_test")]


def test_load_siglip2_processor_online_fallback_uses_rich_reporting(monkeypatch):
    calls = []
    state = {"active": False, "entries": 0}

    @contextmanager
    def fake_hf_download_reporting():
        state["entries"] += 1
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False

    _install_fake_transformer_loader(monkeypatch, hf_download_reporting=fake_hf_download_reporting)

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(repo_id, local_files_only=False, token=None):
            calls.append((repo_id, local_files_only, token, state["active"]))
            if local_files_only:
                raise FileNotFoundError("cache miss")
            return "processor"

    result = _load_siglip2_processor(
        processor_repo="demo/processor",
        processor_loader=FakeProcessorLoader,
        token="hf_test",
    )

    assert result == "processor"
    assert state["entries"] == 1
    assert calls == [
        ("demo/processor", True, "hf_test", False),
        ("demo/processor", False, "hf_test", True),
    ]


def test_load_siglip2_processor_token_typeerror_retry_stays_in_reporting(monkeypatch):
    calls = []
    state = {"active": False, "entries": 0}

    @contextmanager
    def fake_hf_download_reporting():
        state["entries"] += 1
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False

    _install_fake_transformer_loader(monkeypatch, hf_download_reporting=fake_hf_download_reporting)

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(repo_id, local_files_only=False, **kwargs):
            calls.append((repo_id, local_files_only, dict(kwargs), state["active"]))
            if "token" in kwargs:
                raise TypeError("unexpected keyword argument 'token'")
            if local_files_only:
                raise FileNotFoundError("cache miss")
            return "processor"

    result = _load_siglip2_processor(
        processor_repo="demo/processor",
        processor_loader=FakeProcessorLoader,
        token="hf_test",
    )

    assert result == "processor"
    assert state["entries"] == 1
    assert calls == [
        ("demo/processor", True, {"token": "hf_test"}, False),
        ("demo/processor", True, {}, False),
        ("demo/processor", False, {"token": "hf_test"}, True),
        ("demo/processor", False, {}, True),
    ]


def test_load_cl_tagger_v2_bundle_loads_processor_vocab_and_session(tmp_path):
    captured = {}

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(repo_id, local_files_only=False, token=None):
            captured.setdefault("processor_calls", []).append((repo_id, local_files_only, token))
            return "processor"

    def fake_snapshot_download(*, repo_id, allow_patterns, local_dir, force_download, token=None):
        _write_siglip2_snapshot(
            Path(local_dir),
            metadata={
                "vision_encoder_repo": "google/siglip2-base-patch16-224",
                "is_naflex": False,
            },
        )
        return local_dir

    def fake_session_bundle_loader(**kwargs):
        captured["session_bundle"] = kwargs
        return SimpleNamespace(
            sessions={"model": "session"},
            providers=("CPUExecutionProvider",),
        )

    bundle = load_cl_tagger_v2_bundle(
        repo_id=CL_TAGGER_V2_OPTION,
        model_dir=tmp_path,
        runtime_config="runtime-config",
        snapshot_downloader=fake_snapshot_download,
        processor_loader=FakeProcessorLoader,
        session_bundle_loader=fake_session_bundle_loader,
    )

    assert bundle.session == "session"
    assert bundle.providers == ("CPUExecutionProvider",)
    assert bundle.inference_context == Siglip2InferenceContext(processor="processor", is_naflex=False)
    assert bundle.cache_dir == tmp_path / "cella110n_cl_tagger_v2"
    assert bundle.vocabulary.names[0] == "masterpiece"
    assert bundle.vocabulary.category_indices["general"].tolist() == [1]
    assert bundle.vocabulary.category_indices["character"].tolist() == [2]
    assert bundle.vocabulary.category_indices["rating"].tolist() == [3]
    assert bundle.metadata_path == bundle.cache_dir / CL_TAGGER_V2_DEFAULT_VERSION / "step_486342_metadata.json"
    assert bundle.processor_repo == "google/siglip2-base-patch16-224"
    assert captured["processor_calls"] == [("google/siglip2-base-patch16-224", True, None)]
    assert captured["session_bundle"]["bundle_key"] == f"wdtagger:{CL_TAGGER_V2_OPTION}"
    assert captured["session_bundle"]["runtime_config"] == "runtime-config"
    assert captured["session_bundle"]["session_paths"] == {"model": bundle.model_path}
    assert bundle.version == CL_TAGGER_V2_DEFAULT_VERSION


def test_load_cl_tagger_v2_metadata_defaults_to_naflex_when_missing(tmp_path):
    metadata_path = tmp_path / "missing_metadata.json"

    processor_repo, is_naflex = load_cl_tagger_v2_metadata(metadata_path)

    assert processor_repo == CL_TAGGER_V2_PROCESSOR_REPO
    assert is_naflex is True


def test_process_siglip2_batch_uses_named_inputs_and_sigmoid():
    captured = {}

    class FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array)

        def float(self):
            return FakeTensor(self._array.astype(np.float32))

        def numpy(self):
            return self._array

    class FakeProcessor:
        def __call__(self, *, images, return_tensors, max_num_patches):
            captured["processor_call"] = {
                "count": len(images),
                "return_tensors": return_tensors,
                "max_num_patches": max_num_patches,
            }
            return {
                "pixel_values": FakeTensor(np.ones((1, 3, 2, 2), dtype=np.float32)),
                "pixel_attention_mask": FakeTensor(np.ones((1, 2, 2), dtype=np.float32)),
                "spatial_shapes": FakeTensor(np.array([[2, 2]], dtype=np.int64)),
            }

    class FakeSession:
        def run(self, output_names, feeds):
            captured["session_call"] = {
                "output_names": output_names,
                "feeds": feeds,
            }
            return [np.array([[0.0, 2.0]], dtype=np.float32)]

    probs = process_siglip2_batch(
        [object()],
        FakeSession(),
        Siglip2InferenceContext(processor=FakeProcessor(), max_num_patches=128, is_naflex=True),
    )

    assert captured["processor_call"] == {
        "count": 1,
        "return_tensors": "pt",
        "max_num_patches": 128,
    }
    assert captured["session_call"]["output_names"] == ["logits"]
    assert set(captured["session_call"]["feeds"]) == {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
    assert np.allclose(probs, np.array([[0.5, 1.0 / (1.0 + np.exp(-2.0))]]), atol=1e-6)


def test_process_siglip2_batch_falls_back_to_standard_pixel_values_only():
    captured = {}

    class FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array)

        def float(self):
            return FakeTensor(self._array.astype(np.float32))

        def numpy(self):
            return self._array

    class FakeProcessor:
        def __call__(self, *, images, return_tensors):
            captured["processor_call"] = {
                "count": len(images),
                "return_tensors": return_tensors,
            }
            return {
                "pixel_values": FakeTensor(np.ones((1, 3, 2, 2), dtype=np.float32)),
            }

    class FakeSession:
        def run(self, output_names, feeds):
            captured["session_call"] = {
                "output_names": output_names,
                "feeds": feeds,
            }
            return [np.array([[1.0]], dtype=np.float32)]

    probs = process_siglip2_batch(
        [object()],
        FakeSession(),
        Siglip2InferenceContext(processor=FakeProcessor(), is_naflex=False),
    )

    assert captured["processor_call"] == {
        "count": 1,
        "return_tensors": "pt",
    }
    assert captured["session_call"]["output_names"] == ["logits"]
    assert set(captured["session_call"]["feeds"]) == {"pixel_values"}
    assert np.allclose(probs, np.array([[1.0 / (1.0 + np.exp(-1.0))]]), atol=1e-6)
