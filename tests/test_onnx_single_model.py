import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_load_single_model_bundle_downloads_artifact_and_builds_session(tmp_path):
    from module.onnx_runtime.config import OnnxRuntimeConfig
    from module.onnx_runtime.single_model import OnnxModelSpec, load_single_model_bundle

    captured = {}

    class FakeSession:
        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="pixel_values", shape=[1, 3, 224, 224])]

    def fake_download(repo_id, onnx_filename, **kwargs):
        captured["download"] = (repo_id, onnx_filename, kwargs)
        model_path = tmp_path / onnx_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("onnx", encoding="utf-8")
        return model_path

    def fake_load_session_bundle(**kwargs):
        captured["bundle"] = kwargs
        return SimpleNamespace(
            sessions={"model": FakeSession()},
            providers=("CPUExecutionProvider",),
        )

    spec = OnnxModelSpec(
        repo_id="repo/model",
        onnx_filename="model.onnx",
        local_dir=tmp_path / "cache",
        bundle_key="single:model",
    )
    runtime = OnnxRuntimeConfig(execution_provider="cpu")

    bundle = load_single_model_bundle(
        spec=spec,
        runtime_config=runtime,
        artifact_loader=fake_download,
        session_bundle_loader=fake_load_session_bundle,
    )

    assert bundle.model_path.name == "model.onnx"
    assert bundle.session.__class__ is FakeSession
    assert bundle.providers == ("CPUExecutionProvider",)
    assert bundle.input_metas[0].name == "pixel_values"
    assert captured["download"][0:2] == ("repo/model", "model.onnx")
    assert "logger" not in captured["download"][2]
    assert captured["bundle"]["bundle_key"] == "single:model"
    assert captured["bundle"]["session_paths"] == {"model": tmp_path / "model.onnx"}


def test_load_single_model_bundle_forwards_logger_when_loader_supports_it(tmp_path):
    from module.onnx_runtime.config import OnnxRuntimeConfig
    from module.onnx_runtime.single_model import OnnxModelSpec, load_single_model_bundle

    captured = {}

    class FakeSession:
        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="pixel_values", shape=[1, 3, 224, 224])]

    def fake_download(repo_id, onnx_filename, **kwargs):
        captured["download"] = (repo_id, onnx_filename, kwargs)
        model_path = tmp_path / onnx_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("onnx", encoding="utf-8")
        return model_path

    def fake_load_session_bundle(**kwargs):
        return SimpleNamespace(
            sessions={"model": FakeSession()},
            providers=("CPUExecutionProvider",),
        )

    spec = OnnxModelSpec(
        repo_id="repo/model",
        onnx_filename="model.onnx",
        local_dir=tmp_path / "cache",
        bundle_key="single:model",
    )
    runtime = OnnxRuntimeConfig(execution_provider="cpu")

    load_single_model_bundle(
        spec=spec,
        runtime_config=runtime,
        artifact_loader=fake_download,
        session_bundle_loader=fake_load_session_bundle,
        logger=lambda message: None,
    )

    assert callable(captured["download"][2]["logger"])


def test_load_multi_model_bundle_downloads_artifacts_and_support_files(tmp_path):
    from module.onnx_runtime.config import OnnxRuntimeConfig
    from module.onnx_runtime.multi_model import OnnxMultiModelSpec, load_multi_model_bundle

    captured = {}

    class FakeSession:
        pass

    def fake_artifact_loader(repo_id, artifacts, **kwargs):
        captured["artifacts"] = (repo_id, dict(artifacts), kwargs)
        resolved = {}
        for name, filename in artifacts.items():
            target = tmp_path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(name, encoding="utf-8")
            resolved[name] = target
        return resolved

    def fake_support_loader(repo_id, files, **kwargs):
        captured["support"] = (repo_id, dict(files), kwargs)
        resolved = {}
        for name, filename in files.items():
            target = tmp_path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(name, encoding="utf-8")
            resolved[name] = target
        return resolved

    def fake_load_session_bundle(**kwargs):
        captured["bundle"] = kwargs
        return SimpleNamespace(
            sessions={"encoder": FakeSession(), "decoder": FakeSession()},
            providers=("CPUExecutionProvider",),
        )

    spec = OnnxMultiModelSpec(
        repo_id="repo/model",
        artifacts={"encoder": "encoder.onnx", "decoder": "decoder.onnx"},
        support_files={"config": "config.json"},
        local_dir=tmp_path / "cache",
        bundle_key="multi:model",
    )
    runtime = OnnxRuntimeConfig(execution_provider="cpu")

    bundle = load_multi_model_bundle(
        spec=spec,
        runtime_config=runtime,
        artifact_loader=fake_artifact_loader,
        support_file_loader=fake_support_loader,
        session_bundle_loader=fake_load_session_bundle,
        logger=lambda message: None,
    )

    assert set(bundle.sessions) == {"encoder", "decoder"}
    assert bundle.providers == ("CPUExecutionProvider",)
    assert bundle.artifact_paths["encoder"] == tmp_path / "encoder.onnx"
    assert bundle.support_paths["config"] == tmp_path / "config.json"
    assert captured["artifacts"][0] == "repo/model"
    assert captured["artifacts"][1] == {"encoder": "encoder.onnx", "decoder": "decoder.onnx"}
    assert callable(captured["artifacts"][2]["logger"])
    assert captured["support"][0] == "repo/model"
    assert captured["support"][1] == {"config": "config.json"}
    assert callable(captured["support"][2]["logger"])
    assert captured["bundle"]["bundle_key"] == "multi:model"
    assert captured["bundle"]["session_paths"] == {
        "encoder": tmp_path / "encoder.onnx",
        "decoder": tmp_path / "decoder.onnx",
    }
