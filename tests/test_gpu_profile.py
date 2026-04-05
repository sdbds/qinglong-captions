from types import SimpleNamespace

from module.gpu_profile import (
    GPUProbeResult,
    classify_vram_tier,
    format_gpu_summary,
    probe_gpu_environment,
    recommend_see_through_config,
)


def _make_probe(*, total_vram_gb: float, bf16_supported: bool = True, cuda_available: bool = True) -> GPUProbeResult:
    if not cuda_available:
        return GPUProbeResult(
            torch_available=True,
            cuda_available=False,
            cuda_version=None,
            device_count=0,
            current_device_index=None,
            devices=(),
            tier="cpu_only",
            tier_label="CPU only",
        )

    total_vram_bytes = int(total_vram_gb * 1024**3)
    return GPUProbeResult(
        torch_available=True,
        cuda_available=True,
        cuda_version="12.8",
        device_count=1,
        current_device_index=0,
        devices=(
            SimpleNamespace(
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
        tier=classify_vram_tier(total_vram_bytes, cuda_available=True),
        tier_label="",
    )


def test_classify_vram_tier_uses_expected_boundaries():
    assert classify_vram_tier(None, cuda_available=False) == "cpu_only"
    assert classify_vram_tier(int(7.9 * 1024**3), cuda_available=True) == "lt8gb"
    assert classify_vram_tier(int(8.0 * 1024**3), cuda_available=True) == "8_to_12gb"
    assert classify_vram_tier(int(12.0 * 1024**3), cuda_available=True) == "12_to_16gb"
    assert classify_vram_tier(int(16.0 * 1024**3), cuda_available=True) == "12_to_16gb"
    assert classify_vram_tier(int(16.1 * 1024**3), cuda_available=True) == "gt16gb"


def test_probe_gpu_environment_collects_name_sm_and_vram():
    class FakeProps:
        def __init__(self, total_memory, major, minor):
            self.total_memory = total_memory
            self.major = major
            self.minor = minor

    class FakeCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 1

        def current_device(self):
            return 0

        def get_device_name(self, index):
            assert index == 0
            return "RTX Test 5090"

        def get_device_properties(self, index):
            assert index == 0
            return FakeProps(total_memory=24 * 1024**3, major=12, minor=0)

        def is_bf16_supported(self):
            return True

    fake_torch = SimpleNamespace(cuda=FakeCuda(), version=SimpleNamespace(cuda="12.8"))

    probe = probe_gpu_environment(torch_module=fake_torch)

    assert probe.cuda_available is True
    assert probe.device_count == 1
    assert probe.current_device_index == 0
    assert probe.tier == "gt16gb"
    assert probe.devices[0].name == "RTX Test 5090"
    assert probe.devices[0].capability == (12, 0)
    assert probe.devices[0].capability_label == "12.0"
    assert probe.devices[0].sm == "sm120"
    assert probe.devices[0].total_vram_gb == 24.0


def test_format_gpu_summary_includes_nonzero_current_cuda_index():
    probe = GPUProbeResult(
        torch_available=True,
        cuda_available=True,
        cuda_version="12.8",
        device_count=2,
        current_device_index=1,
        devices=(
            SimpleNamespace(
                index=0,
                name="GPU Zero",
                capability=(8, 9),
                capability_label="8.9",
                sm="sm89",
                total_vram_bytes=24 * 1024**3,
                total_vram_gb=24.0,
                bf16_supported=False,
            ),
            SimpleNamespace(
                index=1,
                name="GPU One",
                capability=(8, 9),
                capability_label="8.9",
                sm="sm89",
                total_vram_bytes=24 * 1024**3,
                total_vram_gb=24.0,
                bf16_supported=True,
            ),
        ),
        tier="gt16gb",
        tier_label=">16 GB",
    )

    summary = format_gpu_summary(probe)

    assert "cuda:1" in summary
    assert "GPU One" in summary


def test_recommend_see_through_config_uses_conservative_profiles():
    under_8 = recommend_see_through_config(_make_probe(total_vram_gb=6.0))
    assert under_8.resolution == 768
    assert under_8.resolution_depth == 720
    assert under_8.quant_mode == "nf4"
    assert under_8.group_offload is True

    between_8_and_12 = recommend_see_through_config(_make_probe(total_vram_gb=10.0))
    assert between_8_and_12.resolution == 1024
    assert between_8_and_12.dtype == "bfloat16"
    assert between_8_and_12.quant_mode == "none"
    assert between_8_and_12.group_offload is True

    between_12_and_16 = recommend_see_through_config(_make_probe(total_vram_gb=14.0))
    assert between_12_and_16.resolution == 1280
    assert between_12_and_16.dtype == "bfloat16"
    assert between_12_and_16.group_offload is True

    above_16 = recommend_see_through_config(_make_probe(total_vram_gb=24.0))
    assert above_16.resolution == 1280
    assert above_16.dtype == "bfloat16"
    assert above_16.group_offload is False

    cpu_only = recommend_see_through_config(_make_probe(total_vram_gb=0.0, cuda_available=False))
    assert cpu_only.resolution == 768
    assert cpu_only.dtype == "float32"
    assert cpu_only.quant_mode == "none"
