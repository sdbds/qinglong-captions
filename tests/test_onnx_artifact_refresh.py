from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from module.onnx_runtime.config import OnnxRuntimeConfig
from module.onnx_runtime.multi_model import OnnxMultiModelSpec, load_multi_model_bundle
from module.onnx_runtime.session import clear_session_bundle_cache, load_session_bundle, refresh_session_artifacts
from module.onnx_runtime.single_model import OnnxModelSpec, load_single_model_bundle


class FileSession:
    def __init__(self, path, *, sess_options=None, providers=None):
        self.version = Path(path).read_text(encoding="utf-8")
        self.closed = False

    def get_inputs(self):
        return []

    def close(self):
        self.closed = True

    def run(self):
        assert not self.closed
        return self.version


@pytest.fixture(autouse=True)
def clean_session_cache():
    clear_session_bundle_cache()
    yield
    clear_session_bundle_cache()


def load_sessions(paths, *, runtime=None, key="refresh", factory=FileSession, revision=None):
    return load_session_bundle(
        bundle_key=key,
        session_paths=paths,
        runtime_config=runtime or OnnxRuntimeConfig(execution_provider="cpu"),
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=factory,
        session_options_factory=lambda: None,
        artifact_revision=revision,
    )


def load_wrapper(path, *, multi, runtime, artifact_loader, support_file_loader):
    common = dict(
        repo_id="test/model",
        local_dir=path.parent,
        bundle_key="refresh",
        support_files={"config": "config.json"},
    )
    if multi:
        spec = OnnxMultiModelSpec(artifacts={"model": path.name}, **common)
        loader = load_multi_model_bundle
    else:
        spec = OnnxModelSpec(onnx_filename=path.name, **common)
        loader = load_single_model_bundle

    def session_loader(**kwargs):
        return load_session_bundle(
            **kwargs,
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            session_factory=FileSession,
            session_options_factory=lambda: None,
        )

    return loader(
        spec=spec,
        runtime_config=runtime,
        artifact_loader=artifact_loader,
        support_file_loader=support_file_loader,
        session_bundle_loader=session_loader,
    )


@pytest.mark.parametrize(
    "cached_runtime",
    [
        OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=2),
        OnnxRuntimeConfig(execution_provider="cuda", intra_op_num_threads=1),
        OnnxRuntimeConfig(execution_provider="cuda", provider_options={"cuda": {"device_id": 1}}),
    ],
    ids=["threads", "provider", "provider-options"],
)
def test_force_refresh_invalidates_all_runtime_variants(tmp_path, cached_runtime):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    paths = {"model": path}
    old = load_sessions(paths, runtime=cached_runtime)
    other_path = tmp_path / "unrelated.onnx"
    other_path.write_text("unrelated", encoding="utf-8")
    unrelated = load_sessions({"model": other_path}, runtime=cached_runtime)

    path.write_text("v2", encoding="utf-8")
    refreshed = load_sessions(
        paths,
        runtime=OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=1, force_download=True),
    )

    current = load_sessions(paths, runtime=cached_runtime)
    assert current.sessions["model"].run() == "v2"
    assert current is not old
    assert old.sessions["model"].run() == "v1"
    assert load_sessions(paths, runtime=cached_runtime) is current
    assert load_sessions(paths, runtime=OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=1)) is refreshed
    assert load_sessions({"model": other_path}, runtime=cached_runtime) is unrelated


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("failure", ["artifact", "support"])
def test_wrapper_failed_refresh_invalidates_before_file_mutation(tmp_path, multi, failure):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    runtime = OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=2)

    def artifact_loader(*args, force_download=False, **kwargs):
        if force_download:
            path.write_text("v2", encoding="utf-8")
            if failure == "artifact":
                raise RuntimeError("artifact download failed after replacement")
        return {"model": path} if multi else path

    def support_loader(*args, force_download=False, **kwargs):
        if force_download and failure == "support":
            raise RuntimeError("support download failed")
        return {}

    def load(force=False):
        return load_wrapper(
            path, multi=multi, runtime=replace(runtime, force_download=force),
            artifact_loader=artifact_loader, support_file_loader=support_loader,
        )

    old = load()
    with pytest.raises(RuntimeError, match="download failed"):
        load(force=True)

    current = load()
    assert current.session_bundle.sessions["model"].run() == "v2"
    assert old.session_bundle.sessions["model"].run() == "v1"
    assert load().session_bundle is current.session_bundle


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_wrapper_evicts_existing_sessions_before_calling_artifact_loader(tmp_path, multi):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    paths = {"model": path}
    old = load_sessions(paths)

    def artifact_loader(*args, **kwargs):
        assert load_sessions(paths) is not old
        path.write_text("v2", encoding="utf-8")
        return {"model": path} if multi else path

    current = load_wrapper(
        path, multi=multi,
        runtime=OnnxRuntimeConfig(execution_provider="cpu", force_download=True),
        artifact_loader=artifact_loader, support_file_loader=lambda *args, **kwargs: {},
    )

    assert current.session_bundle.sessions["model"].run() == "v2"
    assert old.sessions["model"].run() == "v1"


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("start", ["before", "during"])
@pytest.mark.parametrize("finish", ["during", "after"])
def test_old_inflight_load_cannot_publish_across_wrapper_refresh(tmp_path, multi, start, finish):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    session_started, release_session = Event(), Event()
    download_started, release_download = Event(), Event()

    def old_factory(*args, **kwargs):
        session = FileSession(*args, **kwargs)
        session_started.set()
        assert release_session.wait(5)
        return session

    def artifact_loader(*args, **kwargs):
        download_started.set()
        assert release_download.wait(5)
        path.write_text("v2", encoding="utf-8")
        return {"model": path} if multi else path

    def support_loader(*args, **kwargs):
        raise RuntimeError("support download failed")

    def refresh():
        return load_wrapper(
            path, multi=multi,
            runtime=OnnxRuntimeConfig(execution_provider="cpu", force_download=True),
            artifact_loader=artifact_loader, support_file_loader=support_loader,
        )

    old_runtime = OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            if start == "before":
                pending = pool.submit(load_sessions, {"model": path}, runtime=old_runtime, factory=old_factory)
                assert session_started.wait(5)
                refreshing = pool.submit(refresh)
                assert download_started.wait(5)
            else:
                refreshing = pool.submit(refresh)
                assert download_started.wait(5)
                pending = pool.submit(load_sessions, {"model": path}, runtime=old_runtime, factory=old_factory)
                assert session_started.wait(5)

            if finish == "during":
                release_session.set()
                old = pending.result(timeout=5)
            release_download.set()
            with pytest.raises(RuntimeError, match="support download failed"):
                refreshing.result(timeout=5)
            release_session.set()
            old = pending.result(timeout=5)
        finally:
            release_session.set()
            release_download.set()

    current = load_sessions({"model": path}, runtime=old_runtime)
    assert current.sessions["model"].run() == "v2"
    assert old.sessions["model"].run() == "v1"
    assert load_sessions({"model": path}, runtime=old_runtime) is current


def test_refresh_invalidates_other_bundles_sharing_any_artifact(tmp_path):
    encoder, decoder = tmp_path / "encoder.onnx", tmp_path / "decoder.onnx"
    encoder.write_text("encoder-v1", encoding="utf-8")
    decoder.write_text("decoder-v1", encoding="utf-8")
    paths = {"encoder": encoder, "decoder": decoder}
    old = load_sessions(paths, key="multi", revision="old-revision")

    encoder.write_text("encoder-v2", encoding="utf-8")
    load_sessions(
        {"model": encoder}, key="single", revision="new-revision",
        runtime=OnnxRuntimeConfig(execution_provider="cpu", force_download=True),
    )

    current = load_sessions(paths, key="multi", revision="old-revision")
    assert current.sessions["encoder"].run() == "encoder-v2"
    assert current.sessions["decoder"].run() == "decoder-v1"
    assert old.sessions["encoder"].run() == "encoder-v1"
    assert old.sessions["decoder"].run() == "decoder-v1"


def test_refresh_matches_relative_and_absolute_artifact_paths(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    old = load_sessions({"model": Path("model.onnx")})

    path.write_text("v2", encoding="utf-8")
    load_sessions(
        {"model": path},
        runtime=OnnxRuntimeConfig(execution_provider="cpu", force_download=True),
    )

    current = load_sessions({"model": Path("model.onnx")})
    assert current.sessions["model"].run() == "v2"
    assert old.sessions["model"].run() == "v1"


def test_failed_session_refresh_closes_only_new_partial_bundle(tmp_path):
    encoder, decoder = tmp_path / "encoder.onnx", tmp_path / "decoder.onnx"
    paths = {"encoder": encoder, "decoder": decoder}
    for path in paths.values():
        path.write_text("v1", encoding="utf-8")
    runtime = OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=2)
    old = load_sessions(paths, runtime=runtime)
    for path in paths.values():
        path.write_text("v2", encoding="utf-8")
    created = []

    def failing_factory(path, **kwargs):
        if Path(path) == decoder:
            raise RuntimeError("decoder construction failed")
        session = FileSession(path, **kwargs)
        created.append(session)
        return session

    with pytest.raises(RuntimeError, match="decoder construction failed"):
        load_sessions(
            paths, factory=failing_factory,
            runtime=OnnxRuntimeConfig(execution_provider="cpu", force_download=True),
        )

    assert created[0].closed
    assert old.sessions["encoder"].run() == "v1"
    assert old.sessions["decoder"].run() == "v1"
    current = load_sessions(paths, runtime=runtime)
    assert current.sessions["encoder"].run() == "v2"
    assert current.sessions["decoder"].run() == "v2"


def test_normal_cache_reuse_remains_revision_scoped(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    old = load_sessions({"model": path}, revision="v1")
    path.write_text("v2", encoding="utf-8")
    current = load_sessions({"model": path}, revision="v2")

    assert old.sessions["model"].run() == "v1"
    assert current.sessions["model"].run() == "v2"
    assert load_sessions({"model": path}, revision="v2") is current


def test_force_refresh_preserves_unrelated_inflight_load(tmp_path):
    path, unrelated_path = tmp_path / "model.onnx", tmp_path / "unrelated.onnx"
    path.write_text("v2", encoding="utf-8")
    unrelated_path.write_text("unrelated", encoding="utf-8")
    started, release = Event(), Event()

    def delayed_factory(*args, **kwargs):
        session = FileSession(*args, **kwargs)
        started.set()
        assert release.wait(5)
        return session

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(load_sessions, {"model": unrelated_path}, factory=delayed_factory)
        try:
            assert started.wait(5)
            load_sessions(
                {"model": path},
                runtime=OnnxRuntimeConfig(execution_provider="cpu", force_download=True),
            )
        finally:
            release.set()
        unrelated = pending.result(timeout=5)

    assert load_sessions({"model": unrelated_path}) is unrelated
    assert unrelated.sessions["model"].run() == "unrelated"


def test_overlapping_refreshes_do_not_publish_intermediate_sessions(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    paths = {"model": path}

    with refresh_session_artifacts(paths):
        with refresh_session_artifacts(paths):
            pass
        intermediate = load_sessions(paths)
        path.write_text("v2", encoding="utf-8")

    current = load_sessions(paths)
    assert current.sessions["model"].run() == "v2"
    assert intermediate.sessions["model"].run() == "v1"
    assert load_sessions(paths) is current


def test_prefix_cache_clear_still_closes_matches_and_invalidates_pending_loads(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_text("v1", encoding="utf-8")
    paths = {"model": path}
    removed = load_sessions(paths, key="clear:cached")
    kept = load_sessions(paths, key="keep:cached")
    started, release = Event(), Event()

    def delayed_factory(*args, **kwargs):
        session = FileSession(*args, **kwargs)
        started.set()
        assert release.wait(5)
        return session

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(load_sessions, paths, key="clear:pending", factory=delayed_factory)
        try:
            assert started.wait(5)
            clear_session_bundle_cache("clear:")
            path.write_text("v2", encoding="utf-8")
        finally:
            release.set()
        old = pending.result(timeout=5)

    assert removed.sessions["model"].closed
    assert load_sessions(paths, key="keep:cached") is kept
    assert kept.sessions["model"].run() == "v1"
    assert old.sessions["model"].run() == "v1"
    assert load_sessions(paths, key="clear:pending").sessions["model"].run() == "v2"
