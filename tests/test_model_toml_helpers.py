import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "gui") not in sys.path:
    sys.path.insert(1, str(ROOT / "gui"))


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


def test_load_model_id_options_reads_model_list_repos_in_order(tmp_path):
    from gui.utils.toml_helpers import load_model_id_options

    config_path = tmp_path / "model.toml"
    config_path.write_text(
        """
[qwen_vl_local]
model_list."Qwen2.5 VL 3B Instruct" = { model_id = "Qwen/Qwen2.5-VL-3B-Instruct", meta = { min_vram_gb = 4 } }
model_list."Qwen2.5 VL 7B Instruct" = { model_id = "Qwen/Qwen2.5-VL-7B-Instruct", meta = { min_vram_gb = 8 } }
model_list."Qwen3.5 9B" = { model_id = "Qwen/Qwen3.5-9B", meta = { min_vram_gb = 12 } }
model_id = "Qwen/Qwen3.5-9B"
""",
        encoding="utf-8",
    )

    options = load_model_id_options("qwen_vl_local", config_path=config_path)

    assert list(options) == [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3.5-9B",
    ]
    assert options["Qwen/Qwen2.5-VL-3B-Instruct"] == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_load_model_id_options_keeps_current_custom_value_visible(tmp_path):
    from gui.utils.toml_helpers import load_model_id_options

    config_path = tmp_path / "model.toml"
    config_path.write_text(
        """
[qwen_vl_local]
model_list."Qwen3.5 9B" = { model_id = "Qwen/Qwen3.5-9B", meta = { min_vram_gb = 12 } }
model_id = "Qwen/Qwen3.5-9B"
""",
        encoding="utf-8",
    )

    options = load_model_id_options(
        "qwen_vl_local",
        current_model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        config_path=config_path,
    )

    assert "Qwen/Qwen2.5-VL-7B-Instruct-AWQ" in options
    assert options["Qwen/Qwen2.5-VL-7B-Instruct-AWQ"] == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"


def test_assess_current_model_fit_uses_toml_min_vram_meta(tmp_path):
    from gui.utils.toml_helpers import assess_current_model_fit

    config_path = tmp_path / "model.toml"
    config_path.write_text(
        """
[qwen_vl_local]
model_list."Qwen3.5 9B" = { model_id = "Qwen/Qwen3.5-9B", meta = { min_vram_gb = 12 } }
model_id = "Qwen/Qwen3.5-9B"
""",
        encoding="utf-8",
    )

    fit = assess_current_model_fit(
        "qwen_vl_local",
        current_model_id="Qwen/Qwen3.5-9B",
        probe=_make_probe(10.0),
        config_path=config_path,
    )

    assert fit is not None
    assert fit.status == "warning"
    assert fit.source == "model_list"
    assert "needs >= 12 GB" in fit.status_label


def test_assess_current_model_fit_falls_back_to_formula_for_custom_model_ids(tmp_path):
    from gui.utils.toml_helpers import assess_current_model_fit

    config_path = tmp_path / "model.toml"
    config_path.write_text(
        """
[qwen_vl_local]
model_list."Qwen3.5 9B" = { model_id = "Qwen/Qwen3.5-9B", meta = { min_vram_gb = 12 } }
model_id = "Qwen/Qwen3.5-9B"
""",
        encoding="utf-8",
    )

    fit = assess_current_model_fit(
        "qwen_vl_local",
        current_model_id="Qwen/Qwen2.5-VL-12B-Instruct-fp8",
        probe=_make_probe(16.0),
        config_path=config_path,
    )

    assert fit is not None
    assert fit.status == "unknown"
    assert fit.source == "heuristic"
    assert "12 GB" in fit.status_label


def test_load_current_route_model_ids_uses_provider_section_aliases(tmp_path):
    from gui.utils.toml_helpers import load_current_route_model_ids

    config_path = tmp_path / "model.toml"
    config_path.write_text(
        """
[penguin]
model_id = "tencent/Penguin-VL-8B"
""",
        encoding="utf-8",
    )

    route_model_ids = load_current_route_model_ids(("penguin_vl_local",), config_path=config_path)

    assert route_model_ids == {"penguin_vl_local": "tencent/Penguin-VL-8B"}
