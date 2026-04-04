import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gui"))
sys.path.insert(2, str(ROOT / "gui" / "wizard"))

from gui.wizard import step6_tools
from module.gpu_profile import GPUDeviceInfo, GPUProbeResult, classify_vram_tier, tier_label


def _make_probe(total_vram_gb: float, *, bf16_supported: bool = True) -> GPUProbeResult:
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


def test_tools_step_applies_low_vram_see_through_defaults(monkeypatch):
    monkeypatch.setattr(step6_tools, "get_cached_gpu_probe", lambda: _make_probe(10.0, bf16_supported=False))

    step = step6_tools.ToolsStep()

    assert step.config["see_through_resolution"] == 1024
    assert step.config["see_through_resolution_depth"] == 720
    assert step.config["see_through_dtype"] == "float16"
    assert step.config["see_through_quant_mode"] == "none"
    assert step.config["see_through_group_offload"] is True


def test_tools_step_applies_high_vram_see_through_defaults(monkeypatch):
    monkeypatch.setattr(step6_tools, "get_cached_gpu_probe", lambda: _make_probe(24.0, bf16_supported=True))

    step = step6_tools.ToolsStep()

    assert step.config["see_through_resolution"] == 1280
    assert step.config["see_through_dtype"] == "bfloat16"
    assert step.config["see_through_quant_mode"] == "none"
    assert step.config["see_through_group_offload"] is False
