import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_collect_external_data_files_keeps_split_suffixes():
    from module.onnx_runtime.artifacts import collect_external_data_files

    repo_files = [
        "onnx/decoder_q4.onnx",
        "onnx/decoder_q4.onnx_data",
        "onnx/decoder_q4.onnx_data_1",
        "onnx/decoder_q4.onnx_data_2",
        "onnx/decoder_fp16.onnx_data",
    ]

    assert collect_external_data_files(repo_files, "onnx/decoder_q4.onnx") == (
        "onnx/decoder_q4.onnx_data",
        "onnx/decoder_q4.onnx_data_1",
        "onnx/decoder_q4.onnx_data_2",
    )


def test_download_onnx_artifact_downloads_model_and_matching_external_data(tmp_path):
    from module.onnx_runtime.artifacts import download_onnx_artifact

    requested = []
    logs = []

    def fake_download(*, repo_id, filename, local_dir=None, force_download=False):
        requested.append((repo_id, filename, local_dir, force_download))
        target = Path(local_dir or tmp_path) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(filename, encoding="utf-8")
        return str(target)

    model_path = download_onnx_artifact(
        "LiquidAI/LFM2.5-VL-1.6B-ONNX",
        "onnx/decoder_q4.onnx",
        local_dir=tmp_path,
        repo_files=(
            "onnx/decoder_q4.onnx",
            "onnx/decoder_q4.onnx_data",
            "onnx/decoder_q4.onnx_data_1",
            "onnx/decoder_fp16.onnx",
        ),
        downloader=fake_download,
        logger=logs.append,
    )

    assert model_path == tmp_path / "onnx" / "decoder_q4.onnx"
    assert [item[1] for item in requested] == [
        "onnx/decoder_q4.onnx",
        "onnx/decoder_q4.onnx_data",
        "onnx/decoder_q4.onnx_data_1",
    ]
    assert any("Downloading ONNX artifact" in message for message in logs)
    assert any("Downloaded ONNX artifact" in message for message in logs)


def test_download_onnx_artifact_logs_when_using_existing_file(tmp_path):
    from module.onnx_runtime.artifacts import download_onnx_artifact

    logs = []
    model_path = tmp_path / "onnx" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("onnx", encoding="utf-8")

    resolved_path = download_onnx_artifact(
        "repo/model",
        "onnx/model.onnx",
        local_dir=tmp_path,
        repo_files=("onnx/model.onnx",),
        logger=logs.append,
    )

    assert resolved_path == model_path
    assert logs == [f"[green]Using existing ONNX artifact[/green] {model_path}"]


def test_build_component_filename_uses_variant_suffixes():
    from module.onnx_runtime.artifacts import build_component_filename

    assert build_component_filename("embed_tokens", "fp16") == "onnx/embed_tokens_fp16.onnx"
    assert build_component_filename("embed_images", "q4") == "onnx/embed_images_q4.onnx"
    assert build_component_filename("decoder", "") == "onnx/decoder.onnx"


def test_select_execution_providers_prefers_cuda_and_falls_back_to_cpu():
    from module.onnx_runtime.session import select_execution_providers

    providers = select_execution_providers(
        preference="auto",
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_cache_dir=Path("cache"),
    )

    assert providers[0][0] == "CUDAExecutionProvider"
    assert providers[0][1] == {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
        "cudnn_conv_use_max_workspace": "1",
        "tunable_op_enable": True,
        "tunable_op_tuning_enable": True,
    }
    assert providers[-1] == "CPUExecutionProvider"


def test_load_session_bundle_caches_sessions(tmp_path):
    from module.onnx_runtime.config import OnnxRuntimeConfig
    from module.onnx_runtime.session import clear_session_bundle_cache, load_session_bundle

    calls = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            calls.append((path, providers))
            self.path = path
            self.providers = providers

    clear_session_bundle_cache()

    runtime = OnnxRuntimeConfig(execution_provider="cpu")
    session_paths = {
        "embed_tokens": tmp_path / "embed_tokens.onnx",
        "decoder": tmp_path / "decoder.onnx",
    }

    bundle1 = load_session_bundle(
        bundle_key="lfm:model",
        session_paths=session_paths,
        runtime_config=runtime,
        available_providers=["CPUExecutionProvider"],
        session_factory=FakeSession,
        session_options_factory=lambda: object(),
    )
    bundle2 = load_session_bundle(
        bundle_key="lfm:model",
        session_paths=session_paths,
        runtime_config=runtime,
        available_providers=["CPUExecutionProvider"],
        session_factory=FakeSession,
        session_options_factory=lambda: object(),
    )

    assert bundle1 is bundle2
    assert set(bundle1.sessions) == {"embed_tokens", "decoder"}
    assert len(calls) == 2


def test_runtime_config_prefers_onnx_section_over_legacy_model_section():
    from module.onnx_runtime.config import OnnxRuntimeConfig

    runtime = OnnxRuntimeConfig.from_runtime_sections(
        defaults={"execution_provider": "auto", "session": {"enable_mem_pattern": True}},
        legacy={"execution_provider": "cpu", "enable_mem_pattern": False},
        override={"execution_provider": "cuda"},
    )

    assert runtime.execution_provider == "cuda"
    assert runtime.enable_mem_pattern is False


def test_load_session_bundle_cache_key_includes_runtime_fingerprint(tmp_path):
    from module.onnx_runtime.config import OnnxRuntimeConfig
    from module.onnx_runtime.session import clear_session_bundle_cache, load_session_bundle

    calls = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            calls.append((path, providers))
            self.path = path
            self.providers = providers

    clear_session_bundle_cache()

    session_paths = {
        "model": tmp_path / "model.onnx",
    }

    runtime_a = OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=1)
    runtime_b = OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=8)

    bundle_a = load_session_bundle(
        bundle_key="single:model",
        session_paths=session_paths,
        runtime_config=runtime_a,
        available_providers=["CPUExecutionProvider"],
        session_factory=FakeSession,
        session_options_factory=lambda: object(),
    )
    bundle_b = load_session_bundle(
        bundle_key="single:model",
        session_paths=session_paths,
        runtime_config=runtime_b,
        available_providers=["CPUExecutionProvider"],
        session_factory=FakeSession,
        session_options_factory=lambda: object(),
    )

    assert bundle_a is not bundle_b
    assert len(calls) == 2


def test_load_session_bundle_uses_model_dir_for_tensorrt_cache_paths(tmp_path):
    from module.onnx_runtime.config import OnnxRuntimeConfig
    from module.onnx_runtime.session import clear_session_bundle_cache, load_session_bundle

    captured = {}

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            captured["providers"] = providers
            self.path = path
            self.providers = providers

    clear_session_bundle_cache()

    model_dir = tmp_path / "wd14_tagger_model" / "SmilingWolf_wd-v1-4-moat-tagger-v2"
    session_paths = {"model": model_dir / "model.onnx"}
    runtime = OnnxRuntimeConfig(execution_provider="tensorrt")

    load_session_bundle(
        bundle_key="single:model",
        session_paths=session_paths,
        runtime_config=runtime,
        available_providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=FakeSession,
        session_options_factory=lambda: object(),
    )

    tensorrt_options = captured["providers"][0][1]
    cuda_options = captured["providers"][1][1]
    assert tensorrt_options["trt_engine_cache_path"] == str(model_dir / "trt_engines")
    assert tensorrt_options["trt_timing_cache_path"] == str(model_dir)
    assert tensorrt_options["trt_builder_optimization_level"] == 3
    assert tensorrt_options["trt_max_partition_iterations"] == 1000
    assert tensorrt_options["trt_engine_hw_compatible"] is True
    assert tensorrt_options["trt_force_sequential_engine_build"] is False
    assert tensorrt_options["trt_context_memory_sharing_enable"] is True
    assert tensorrt_options["trt_sparsity_enable"] is True
    assert tensorrt_options["trt_min_subgraph_size"] == 7
    assert cuda_options == {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
        "cudnn_conv_use_max_workspace": "1",
        "tunable_op_enable": True,
        "tunable_op_tuning_enable": True,
    }


def test_load_session_bundle_uses_model_dir_for_nvtensorrtrtx_cache_paths(tmp_path):
    from module.onnx_runtime.config import OnnxRuntimeConfig
    from module.onnx_runtime.session import clear_session_bundle_cache, load_session_bundle

    captured = {}

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            captured["providers"] = providers
            self.path = path
            self.providers = providers

    clear_session_bundle_cache()

    model_dir = tmp_path / "wd14_tagger_model" / "SmilingWolf_wd-v1-4-moat-tagger-v2"
    session_paths = {"model": model_dir / "model.onnx"}
    runtime = OnnxRuntimeConfig(execution_provider="nvtensorrtrtx")

    load_session_bundle(
        bundle_key="single:model",
        session_paths=session_paths,
        runtime_config=runtime,
        available_providers=[
            "NvTensorRtRtxExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        session_factory=FakeSession,
        session_options_factory=lambda: object(),
    )

    nvtensorrtrtx_options = captured["providers"][0][1]
    cuda_options = captured["providers"][1][1]
    assert nvtensorrtrtx_options == {
        "nv_runtime_cache_path": str(model_dir / "trt_engines"),
        "nv_dump_subgraphs": False,
        "nv_detailed_build_log": True,
        "enable_cuda_graph": True,
        "nv_multi_profile_enable": False,
        "nv_use_external_data_initializer": False,
    }
    assert cuda_options == {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
        "cudnn_conv_use_max_workspace": "1",
        "tunable_op_enable": True,
        "tunable_op_tuning_enable": True,
    }
