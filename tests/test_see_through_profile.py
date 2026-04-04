from types import SimpleNamespace

from module.gpu_profile import GPUProbeResult, recommend_see_through_config
from module.see_through.see_through_profile import SEE_THROUGH_REPO_MAP, resolve_see_through_repo_ids


def test_resolve_see_through_repo_ids_uses_quant_mode_defaults_for_known_repo_family():
    resolved = resolve_see_through_repo_ids(
        quant_mode="nf4",
        repo_id_layerdiff=SEE_THROUGH_REPO_MAP["none"]["layerdiff"],
        repo_id_depth=SEE_THROUGH_REPO_MAP["none"]["depth"],
    )

    assert resolved.repo_id_layerdiff == SEE_THROUGH_REPO_MAP["nf4"]["layerdiff"]
    assert resolved.repo_id_depth == SEE_THROUGH_REPO_MAP["nf4"]["depth"]


def test_resolve_see_through_repo_ids_preserves_explicit_custom_overrides():
    resolved = resolve_see_through_repo_ids(
        quant_mode="nf4",
        repo_id_layerdiff="custom/layerdiff",
        repo_id_depth="custom/depth",
    )

    assert resolved.repo_id_layerdiff == "custom/layerdiff"
    assert resolved.repo_id_depth == "custom/depth"


def test_recommend_see_through_config_exposes_nf4_profile_for_lt8gb():
    probe = GPUProbeResult(
        torch_available=True,
        cuda_available=True,
        cuda_version="12.8",
        device_count=1,
        current_device_index=0,
        devices=(
            SimpleNamespace(
                index=0,
                name="Tiny GPU",
                capability=(8, 6),
                capability_label="8.6",
                sm="sm86",
                total_vram_bytes=int(6 * 1024**3),
                total_vram_gb=6.0,
                bf16_supported=False,
            ),
        ),
        tier="lt8gb",
        tier_label="<8 GB",
    )

    recommendation = recommend_see_through_config(probe)

    assert recommendation.quant_mode == "nf4"
    assert recommendation.group_offload is True
    assert recommendation.repo_id_layerdiff == SEE_THROUGH_REPO_MAP["nf4"]["layerdiff"]
