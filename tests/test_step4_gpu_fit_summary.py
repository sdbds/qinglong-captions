import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def _load_step4_module(module_name: str):
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    step4_caption = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    gui_path = str(ROOT / "gui")
    original_sys_path = list(sys.path)
    if gui_path not in sys.path:
        try:
            insert_at = original_sys_path.index(str(ROOT)) + 1
        except ValueError:
            insert_at = len(sys.path)
        sys.path.insert(insert_at, gui_path)
    try:
        spec.loader.exec_module(step4_caption)
    finally:
        sys.path[:] = original_sys_path

    return step4_caption


def _make_probe(total_vram_gb: float, *, bf16_supported: bool = True):
    from module.gpu_profile import GPUDeviceInfo, GPUProbeResult, classify_vram_tier, tier_label

    total_vram_bytes = int(total_vram_gb * 1024**3)
    tier = classify_vram_tier(total_vram_bytes, cuda_available=True)
    return GPUProbeResult(
        torch_available=True,
        cuda_available=True,
        cuda_version="12.8",
        device_count=1,
        current_device_index=0,
        devices=(
            GPUDeviceInfo(
                index=0,
                name="Fake GPU",
                capability=(8, 9),
                capability_label="8.9",
                sm="sm89",
                total_vram_bytes=total_vram_bytes,
                total_vram_gb=round(total_vram_bytes / 1024**3, 2),
                bf16_supported=bf16_supported,
            ),
        ),
        tier=tier,
        tier_label=tier_label(tier),
    )


def _make_multi_probe() -> object:
    from module.gpu_profile import GPUDeviceInfo, GPUProbeResult, classify_vram_tier, tier_label

    total_vram_bytes = 24 * 1024**3
    tier = classify_vram_tier(total_vram_bytes, cuda_available=True)
    return GPUProbeResult(
        torch_available=True,
        cuda_available=True,
        cuda_version="12.8",
        device_count=2,
        current_device_index=1,
        devices=(
            GPUDeviceInfo(
                index=0,
                name="GPU Zero",
                capability=(8, 9),
                capability_label="8.9",
                sm="sm89",
                total_vram_bytes=total_vram_bytes,
                total_vram_gb=24.0,
                bf16_supported=False,
            ),
            GPUDeviceInfo(
                index=1,
                name="GPU One",
                capability=(8, 9),
                capability_label="8.9",
                sm="sm89",
                total_vram_bytes=total_vram_bytes,
                total_vram_gb=24.0,
                bf16_supported=True,
            ),
        ),
        tier=tier,
        tier_label=tier_label(tier),
    )


def test_caption_step_builds_local_model_fit_entries_from_gpu_probe():
    step4 = _load_step4_module("test_step4_gpu_fit_entries")
    step4._load_current_route_model_ids = lambda: {
        "chandra_ocr": "datalab-to/chandra-ocr-2",
        "qwen_vl_local": "Qwen/Qwen3.5-9B",
        "music_flamingo_local": "henry1477/music-flamingo-2601-hf-fp8",
    }

    step = step4.CaptionStep()
    step.gpu_probe = _make_probe(10.0, bf16_supported=False)
    step.ocr_model = SimpleNamespace(value="chandra_ocr")
    step.vlm_image_model = SimpleNamespace(value="qwen_vl_local")
    step.alm_model = SimpleNamespace(value="music_flamingo_local")

    entries = step._build_local_model_fit_entries()

    assert [entry["family"] for entry in entries] == ["OCR", "VLM", "ALM"]

    ocr_entry = next(entry for entry in entries if entry["family"] == "OCR")
    assert ocr_entry["status"] == "ok"
    assert ocr_entry["current_model_id"] == "datalab-to/chandra-ocr-2"

    vlm_entry = next(entry for entry in entries if entry["family"] == "VLM")
    assert vlm_entry["route_name"] == "qwen_vl_local"
    assert vlm_entry["current_model_id"] == "Qwen/Qwen3.5-9B"
    assert vlm_entry["status"] == "warning"
    assert "May exceed" in vlm_entry["status_label"]

    alm_entry = next(entry for entry in entries if entry["family"] == "ALM")
    assert alm_entry["status"] == "ok"
    assert alm_entry["current_model_id"] == "henry1477/music-flamingo-2601-hf-fp8"


def test_caption_step_ignores_remote_routes_in_local_model_fit_entries():
    step4 = _load_step4_module("test_step4_gpu_fit_remote_routes")
    step4._load_current_route_model_ids = lambda: {}

    step = step4.CaptionStep()
    step.gpu_probe = _make_probe(12.0)
    step.ocr_model = SimpleNamespace(value="mistral_ocr")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="")

    assert step._build_local_model_fit_entries() == ()


def test_caption_step_formats_gpu_fit_header_from_cached_probe():
    step4 = _load_step4_module("test_step4_gpu_fit_header")

    step = step4.CaptionStep()
    step.gpu_probe = _make_probe(24.0)

    header = step._local_model_fit_header()

    assert "Fake GPU" in header
    assert "24.0 GB" in header
    assert ">16 GB" in header


def test_caption_step_formats_gpu_fit_header_for_multi_gpu_probe():
    step4 = _load_step4_module("test_step4_gpu_fit_multi_gpu_header")

    step = step4.CaptionStep()
    step.gpu_probe = _make_multi_probe()

    header = step._local_model_fit_header()

    assert "cuda:1" in header
    assert "GPU One" in header
    assert "2 GPUs" in header


def test_caption_step_uses_fresh_model_id_map_for_gpu_fit_entries():
    step4 = _load_step4_module("test_step4_gpu_fit_fresh_model_map")
    step4._load_current_route_model_ids = lambda: {"qwen_vl_local": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"}

    step = step4.CaptionStep()
    step.gpu_probe = _make_probe(10.0)
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="qwen_vl_local")
    step.alm_model = SimpleNamespace(value="")

    entries = step._build_local_model_fit_entries()

    assert len(entries) == 1
    assert entries[0]["current_model_id"] == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert entries[0]["status"] == "unknown"


def test_caption_step_gpu_fit_header_starts_in_pending_state():
    step4 = _load_step4_module("test_step4_gpu_fit_pending_header")

    step = step4.CaptionStep()

    header = step._local_model_fit_header()

    assert "检查中" in header


def test_caption_step_warns_for_large_custom_model_ids_from_name():
    step4 = _load_step4_module("test_step4_gpu_fit_custom_31b")
    step4._load_current_route_model_ids = lambda: {"qwen_vl_local": "Qwen/Qwen2.5-VL-31B-Instruct-AWQ"}

    step = step4.CaptionStep()
    step.gpu_probe = _make_probe(10.0)
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="qwen_vl_local")
    step.alm_model = SimpleNamespace(value="")

    entries = step._build_local_model_fit_entries()

    assert len(entries) == 1
    assert entries[0]["status"] == "warning"
    assert "31B" in entries[0]["status_label"]
