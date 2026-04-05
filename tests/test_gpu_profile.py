from types import SimpleNamespace

from module.gpu_profile import (
    GPUProbeResult,
    clear_gpu_probe_cache,
    format_gpu_device_lines,
    format_gpu_summary,
    get_cached_gpu_probe,
    probe_gpu_environment,
    recommend_see_through_config,
)


def _make_probe(*, total_vram_gb: float, bf16_supported: bool = False, cuda_available: bool = True) -> GPUProbeResult:
    if not cuda_available:
        return GPUProbeResult(
            torch_available=True,
            cuda_available=False,
            cuda_version=None,
            device_count=0,
            current_device_index=None,
            devices=(),
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
    )


def test_probe_gpu_environment_collects_name_and_vram_from_nvidia_smi(monkeypatch):
    class _Completed:
        def __init__(self, stdout: str):
            self.stdout = stdout
            self.returncode = 0

    monkeypatch.setattr(
        "module.gpu_profile.subprocess.run",
        lambda *args, **kwargs: _Completed("0, RTX Test 5090, 24576\n1, RTX Test 5080, 16384\n"),
    )

    probe = probe_gpu_environment()

    assert probe.cuda_available is True
    assert probe.device_count == 2
    assert probe.current_device_index is None
    assert probe.cuda_version is None
    assert probe.devices[0].name == "RTX Test 5090"
    assert probe.devices[0].total_vram_gb == 24.0
    assert probe.devices[1].name == "RTX Test 5080"
    assert probe.devices[1].total_vram_gb == 16.0


def test_format_gpu_summary_uses_inventory_semantics():
    probe = GPUProbeResult(
        torch_available=True,
        cuda_available=True,
        cuda_version=None,
        device_count=2,
        current_device_index=None,
        devices=(
            SimpleNamespace(
                index=0,
                name="GPU Zero",
                capability=None,
                capability_label="unknown",
                sm=None,
                total_vram_bytes=24 * 1024**3,
                total_vram_gb=24.0,
                bf16_supported=False,
            ),
            SimpleNamespace(
                index=1,
                name="GPU One",
                capability=None,
                capability_label="unknown",
                sm=None,
                total_vram_bytes=24 * 1024**3,
                total_vram_gb=24.0,
                bf16_supported=False,
            ),
        ),
    )

    summary = format_gpu_summary(probe)

    assert "GPU Zero" in summary
    assert "cuda" not in summary.lower()
    assert "2 GPUs" in summary


def test_format_gpu_device_lines_lists_all_detected_gpus():
    probe = GPUProbeResult(
        torch_available=True,
        cuda_available=True,
        cuda_version=None,
        device_count=2,
        current_device_index=None,
        devices=(
            SimpleNamespace(
                index=0,
                name="GPU Zero",
                capability=None,
                capability_label="unknown",
                sm=None,
                total_vram_bytes=24 * 1024**3,
                total_vram_gb=24.0,
                bf16_supported=False,
            ),
            SimpleNamespace(
                index=1,
                name="GPU One",
                capability=None,
                capability_label="unknown",
                sm=None,
                total_vram_bytes=24 * 1024**3,
                total_vram_gb=24.0,
                bf16_supported=False,
            ),
        ),
    )

    lines = format_gpu_device_lines(probe)

    assert lines == (
        "GPU 0 | GPU Zero | 24.0 GB",
        "GPU 1 | GPU One | 24.0 GB",
    )


def test_get_cached_gpu_probe_uses_single_refreshable_inventory_cache(monkeypatch):
    clear_gpu_probe_cache()
    calls: list[GPUProbeResult] = []
    probes = (
        _make_probe(total_vram_gb=24.0),
        _make_probe(total_vram_gb=12.0),
    )

    def _fake_probe():
        probe = probes[len(calls)]
        calls.append(probe)
        return probe

    monkeypatch.setattr("module.gpu_profile.probe_gpu_environment", _fake_probe)

    first = get_cached_gpu_probe(refresh=True)
    second = get_cached_gpu_probe()

    assert first is second
    assert calls == [first]

    third = get_cached_gpu_probe()

    assert third is first

    fourth = get_cached_gpu_probe(refresh=True)

    assert fourth is not third
    assert calls == [first, fourth]


def test_recommend_see_through_config_uses_conservative_profiles():
    under_8 = recommend_see_through_config(_make_probe(total_vram_gb=6.0))
    assert under_8.resolution == 768
    assert under_8.resolution_depth == 720
    assert under_8.quant_mode == "nf4"
    assert under_8.group_offload is True
    assert under_8.min_vram_gb == 0.0

    between_8_and_12 = recommend_see_through_config(_make_probe(total_vram_gb=10.0))
    assert between_8_and_12.resolution == 1024
    assert between_8_and_12.dtype == "float16"
    assert between_8_and_12.quant_mode == "none"
    assert between_8_and_12.group_offload is True
    assert between_8_and_12.min_vram_gb == 8.0

    between_12_and_16 = recommend_see_through_config(_make_probe(total_vram_gb=14.0))
    assert between_12_and_16.resolution == 1280
    assert between_12_and_16.dtype == "float16"
    assert between_12_and_16.group_offload is True
    assert between_12_and_16.min_vram_gb == 12.0

    above_16 = recommend_see_through_config(_make_probe(total_vram_gb=24.0))
    assert above_16.resolution == 1280
    assert above_16.dtype == "float16"
    assert above_16.group_offload is False
    assert above_16.min_vram_gb == 16.0

    cpu_only = recommend_see_through_config(_make_probe(total_vram_gb=0.0, cuda_available=False))
    assert cpu_only.resolution == 768
    assert cpu_only.dtype == "float32"
    assert cpu_only.quant_mode == "none"
    assert cpu_only.min_vram_gb is None
