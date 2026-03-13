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
    )

    assert model_path == tmp_path / "onnx" / "decoder_q4.onnx"
    assert [item[1] for item in requested] == [
        "onnx/decoder_q4.onnx",
        "onnx/decoder_q4.onnx_data",
        "onnx/decoder_q4.onnx_data_1",
    ]


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
