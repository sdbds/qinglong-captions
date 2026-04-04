from __future__ import annotations

from dataclasses import dataclass

from module.gpu_profile import GPUProbeResult


DEFAULT_DEPTH_RESOLUTION = 720
DEFAULT_DEPTH_INFERENCE_STEPS = -1
DEFAULT_SEED = 42
DEFAULT_QUANT_MODE = "none"

SEE_THROUGH_REPO_MAP = {
    "none": {
        "layerdiff": "layerdifforg/seethroughv0.0.2_layerdiff3d",
        "depth": "24yearsold/seethroughv0.0.1_marigold",
    },
    "nf4": {
        "layerdiff": "24yearsold/seethroughv0.0.2_layerdiff3d_nf4",
        "depth": "24yearsold/seethroughv0.0.1_marigold_nf4",
    },
}


@dataclass(frozen=True)
class ResolvedSeeThroughRepos:
    repo_id_layerdiff: str
    repo_id_depth: str


@dataclass(frozen=True)
class SeeThroughRecommendation:
    tier: str
    tier_label: str
    resolution: int
    resolution_depth: int
    dtype: str
    offload_policy: str
    group_offload: bool
    quant_mode: str
    repo_id_layerdiff: str
    repo_id_depth: str
    note: str | None = None


def normalize_quant_mode(quant_mode: str | None) -> str:
    normalized = str(quant_mode or DEFAULT_QUANT_MODE).strip().lower()
    if normalized not in SEE_THROUGH_REPO_MAP:
        raise ValueError(f"Unsupported see-through quant_mode: {quant_mode}")
    return normalized


def _resolve_repo_id(requested_repo: str | None, *, default_repo: str, known_repos: set[str]) -> str:
    normalized_repo = str(requested_repo or "").strip()
    if not normalized_repo:
        return default_repo
    if normalized_repo in known_repos:
        return default_repo
    return normalized_repo


def resolve_see_through_repo_ids(
    *,
    quant_mode: str | None,
    repo_id_layerdiff: str | None = None,
    repo_id_depth: str | None = None,
) -> ResolvedSeeThroughRepos:
    normalized_quant_mode = normalize_quant_mode(quant_mode)
    defaults = SEE_THROUGH_REPO_MAP[normalized_quant_mode]
    known_layerdiff_repos = {repos["layerdiff"] for repos in SEE_THROUGH_REPO_MAP.values()}
    known_depth_repos = {repos["depth"] for repos in SEE_THROUGH_REPO_MAP.values()}
    return ResolvedSeeThroughRepos(
        repo_id_layerdiff=_resolve_repo_id(
            repo_id_layerdiff,
            default_repo=defaults["layerdiff"],
            known_repos=known_layerdiff_repos,
        ),
        repo_id_depth=_resolve_repo_id(
            repo_id_depth,
            default_repo=defaults["depth"],
            known_repos=known_depth_repos,
        ),
    )


def recommend_see_through_config(probe: GPUProbeResult) -> SeeThroughRecommendation:
    repos_none = resolve_see_through_repo_ids(quant_mode="none")
    if not probe.cuda_available or probe.primary_device is None:
        return SeeThroughRecommendation(
            tier=probe.tier,
            tier_label=probe.tier_label,
            resolution=768,
            resolution_depth=DEFAULT_DEPTH_RESOLUTION,
            dtype="float32",
            offload_policy="delete",
            group_offload=False,
            quant_mode="none",
            repo_id_layerdiff=repos_none.repo_id_layerdiff,
            repo_id_depth=repos_none.repo_id_depth,
            note="CUDA unavailable; keep the profile conservative.",
        )

    primary_device = probe.primary_device
    preferred_half = "bfloat16" if primary_device.bf16_supported else "float16"

    if probe.tier == "lt8gb":
        repos_nf4 = resolve_see_through_repo_ids(quant_mode="nf4")
        return SeeThroughRecommendation(
            tier=probe.tier,
            tier_label=probe.tier_label,
            resolution=768,
            resolution_depth=DEFAULT_DEPTH_RESOLUTION,
            dtype=preferred_half,
            offload_policy="delete",
            group_offload=True,
            quant_mode="nf4",
            repo_id_layerdiff=repos_nf4.repo_id_layerdiff,
            repo_id_depth=repos_nf4.repo_id_depth,
            note="Below 8 GB VRAM is still tight; start from NF4 plus group offload.",
        )

    if probe.tier == "8_to_12gb":
        return SeeThroughRecommendation(
            tier=probe.tier,
            tier_label=probe.tier_label,
            resolution=1024,
            resolution_depth=DEFAULT_DEPTH_RESOLUTION,
            dtype=preferred_half,
            offload_policy="delete",
            group_offload=True,
            quant_mode="none",
            repo_id_layerdiff=repos_none.repo_id_layerdiff,
            repo_id_depth=repos_none.repo_id_depth,
            note="8-12 GB keeps the standard repo with group offload; switch to NF4 if you still hit OOM.",
        )

    if probe.tier == "12_to_16gb":
        return SeeThroughRecommendation(
            tier=probe.tier,
            tier_label=probe.tier_label,
            resolution=1280,
            resolution_depth=DEFAULT_DEPTH_RESOLUTION,
            dtype=preferred_half,
            offload_policy="delete",
            group_offload=True,
            quant_mode="none",
            repo_id_layerdiff=repos_none.repo_id_layerdiff,
            repo_id_depth=repos_none.repo_id_depth,
        )

    return SeeThroughRecommendation(
        tier=probe.tier,
        tier_label=probe.tier_label,
        resolution=1280,
        resolution_depth=DEFAULT_DEPTH_RESOLUTION,
        dtype=preferred_half,
        offload_policy="delete",
        group_offload=False,
        quant_mode="none",
        repo_id_layerdiff=repos_none.repo_id_layerdiff,
        repo_id_depth=repos_none.repo_id_depth,
    )
