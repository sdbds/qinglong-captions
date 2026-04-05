import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gui"))
sys.path.insert(2, str(ROOT / "gui" / "wizard"))

from gui.wizard import step6_tools
from module.gpu_profile import GPUDeviceInfo, GPUProbeResult, classify_vram_tier, tier_label
from module.see_through.see_through_profile import recommend_see_through_config


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


def test_tools_step_init_keeps_conservative_defaults_before_gpu_probe(monkeypatch):
    monkeypatch.setattr(
        step6_tools,
        "_probe_see_through_defaults",
        lambda: (_ for _ in ()).throw(AssertionError("GPU probe should not run in __init__")),
    )

    step = step6_tools.ToolsStep()

    assert step.config["see_through_resolution"] == 768
    assert step.config["see_through_resolution_depth"] == 720
    assert step.config["see_through_dtype"] == "float32"
    assert step.config["see_through_quant_mode"] == "none"
    assert step.config["see_through_group_offload"] is False


def test_tools_step_applies_low_vram_probe_recommendation():
    step = step6_tools.ToolsStep()
    recommendation = recommend_see_through_config(_make_probe(10.0, bf16_supported=False))

    step._apply_see_through_recommendation(recommendation)

    assert step.config["see_through_resolution"] == 1024
    assert step.config["see_through_resolution_depth"] == 720
    assert step.config["see_through_dtype"] == "float16"
    assert step.config["see_through_quant_mode"] == "none"
    assert step.config["see_through_group_offload"] is True


def test_tools_step_applies_high_vram_probe_recommendation():
    step = step6_tools.ToolsStep()
    recommendation = recommend_see_through_config(_make_probe(24.0, bf16_supported=True))

    step._apply_see_through_recommendation(recommendation)

    assert step.config["see_through_resolution"] == 1280
    assert step.config["see_through_dtype"] == "bfloat16"
    assert step.config["see_through_quant_mode"] == "none"
    assert step.config["see_through_group_offload"] is False


class _DummyContainer:
    def __init__(self):
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyExecutionPanel:
    def __init__(self, *, show_start):
        self.show_start = show_start
        self.registered_buttons = []

    def register_external_start_button(self, button):
        self.registered_buttons.append(button)


def test_tools_step_only_builds_requested_panels(monkeypatch):
    step = step6_tools.ToolsStep()
    watermark = _DummyContainer()
    audio_separator = _DummyContainer()
    step._tool_tab_containers = {
        "watermark": watermark,
        "audio_separator": audio_separator,
    }
    calls = []

    monkeypatch.setattr(step, "_render_watermark_tool", lambda: calls.append("watermark"))
    monkeypatch.setattr(step, "_render_audio_separator_tool", lambda: calls.append("audio_separator"))

    step._ensure_tool_panel_rendered("watermark")
    step._ensure_tool_panel_rendered("watermark")
    step._ensure_tool_panel_rendered("audio_separator")

    assert calls == ["watermark", "audio_separator"]
    assert watermark.clear_calls == 1
    assert audio_separator.clear_calls == 1


def test_tools_step_defers_execution_panel_until_requested(monkeypatch):
    step = step6_tools.ToolsStep()
    step._execution_panel_container = _DummyContainer()
    step._tool_start_buttons = ["start-a", "start-b"]

    loads = []

    def _load_dummy_panel():
        loads.append("loaded")
        return _DummyExecutionPanel

    monkeypatch.setattr(step6_tools, "_load_execution_panel_cls", _load_dummy_panel)

    assert step.panel is None

    panel = step._ensure_execution_panel()

    assert panel.show_start is False
    assert panel.registered_buttons == ["start-a", "start-b"]
    assert step._ensure_execution_panel() is panel
    assert loads == ["loaded"]


def test_format_gpu_summary_uses_formatter_for_real_probe(monkeypatch):
    probe = object()

    class _FakeGpuProfileModule:
        @staticmethod
        def format_gpu_summary(value):
            assert value is probe
            return "RTX 4090 / 24GB"

    monkeypatch.setitem(sys.modules, "module.gpu_profile", _FakeGpuProfileModule())

    assert step6_tools._format_gpu_summary(probe) == "RTX 4090 / 24GB"


def test_tools_step_only_requests_gpu_probe_when_see_through_panel_renders(monkeypatch):
    step = step6_tools.ToolsStep()
    watermark = _DummyContainer()
    see_through = _DummyContainer()
    step._tool_tab_containers = {
        "watermark": watermark,
        "see_through": see_through,
    }
    calls = []

    monkeypatch.setattr(step, "_render_watermark_tool", lambda: calls.append("watermark"))
    monkeypatch.setattr(step, "_render_see_through_tool", lambda: calls.append("see_through"))
    monkeypatch.setattr(step, "_schedule_see_through_recommendation_refresh", lambda: calls.append("probe"))

    step._ensure_tool_panel_rendered("watermark")
    step._ensure_tool_panel_rendered("see_through")
    step._ensure_tool_panel_rendered("see_through")

    assert calls == ["watermark", "see_through", "probe"]
