from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import toml as tomllib

from .options import ModelVariant

DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "muscriptor_batch_profiles.toml"
)


@dataclass(frozen=True)
class ModelBatchProfile:
    model: ModelVariant
    repo_id: str
    minimum_vram_bytes: int
    peak_bs1_bytes: int
    bs2_minus_bs1_bytes: int
    validated_batch_size: int
    validation_peak_reserved_bytes: int
    validation_chunks: int

    @property
    def peak_bs2_bytes(self) -> int:
        return self.peak_bs1_bytes + self.bs2_minus_bs1_bytes


class InsufficientVRAMError(ValueError):
    def __init__(self, model: ModelVariant, total_bytes: int, required_bytes: int):
        self.model = model
        self.total_bytes = total_bytes
        self.required_bytes = required_bytes
        super().__init__(
            f"MuScriptor {model.value} requires at least "
            f"{required_bytes / 1024**3:.2f} GiB total VRAM"
        )


@dataclass(frozen=True)
class BatchProfileCatalog:
    reserve_bytes: int
    low_vram_max_bytes: int
    low_vram_reserve_bytes: int
    prefer_even: bool
    profiles: dict[ModelVariant, ModelBatchProfile]
    benchmark: dict[str, Any]

    def reserve_for(self, total_vram_bytes: int) -> int:
        if int(total_vram_bytes) <= self.low_vram_max_bytes:
            return self.low_vram_reserve_bytes
        return self.reserve_bytes

    def recommend(self, model: ModelVariant | str, total_vram_bytes: int) -> int:
        variant = ModelVariant(model)
        profile = self.profiles[variant]
        total_vram = int(total_vram_bytes)
        reserve_bytes = self.reserve_for(total_vram)
        minimum_total = profile.minimum_vram_bytes + reserve_bytes
        if total_vram < minimum_total:
            raise InsufficientVRAMError(variant, total_vram, minimum_total)
        budget = total_vram - reserve_bytes
        if budget < profile.peak_bs2_bytes:
            return 1
        initial_marginal = profile.bs2_minus_bs1_bytes
        validated_marginal = (
            profile.validation_peak_reserved_bytes - profile.peak_bs2_bytes
        ) / (profile.validated_batch_size - 2)
        marginal = max(initial_marginal, validated_marginal)
        selected = 2 + int((budget - profile.peak_bs2_bytes) // marginal)
        selected = max(2, min(1024, selected))
        if self.prefer_even and selected > 2 and selected % 2:
            selected -= 1
        return selected


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def load_batch_profile_catalog(
    path: str | Path = DEFAULT_PROFILE_PATH,
) -> BatchProfileCatalog:
    profile_path = Path(path)
    data = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported MuScriptor batch profile schema")
    if tuple(data.get("probe_batch_sizes", ())) != (1, 2):
        raise ValueError("MuScriptor batch profiles must use BS1 and BS2")

    reserve_bytes = _positive_int(data.get("reserve_bytes"), "reserve bytes")
    low_vram_max_bytes = _positive_int(
        data.get("low_vram_max_bytes"),
        "low VRAM maximum",
    )
    low_vram_reserve_bytes = _positive_int(
        data.get("low_vram_reserve_bytes"),
        "low VRAM reserve",
    )
    if low_vram_reserve_bytes >= reserve_bytes:
        raise ValueError("low VRAM reserve must be below the default reserve")
    raw_models = data.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("MuScriptor batch profiles are missing models")

    profiles: dict[ModelVariant, ModelBatchProfile] = {}
    for variant in ModelVariant:
        raw = raw_models.get(variant.value)
        if not isinstance(raw, dict):
            raise ValueError(f"missing MuScriptor batch profile: {variant.value}")
        peak_bs1 = _positive_int(raw.get("peak_bs1_bytes"), "BS1 peak")
        profile = ModelBatchProfile(
            model=variant,
            repo_id=str(raw.get("repo_id") or variant.repo_id),
            minimum_vram_bytes=_positive_int(
                raw.get("minimum_vram_bytes"),
                "minimum VRAM",
            ),
            peak_bs1_bytes=peak_bs1,
            bs2_minus_bs1_bytes=_positive_int(
                raw.get("bs2_minus_bs1_bytes"),
                "BS2-BS1 memory",
            ),
            validated_batch_size=_positive_int(
                raw.get("validated_batch_size"),
                "validated batch size",
            ),
            validation_peak_reserved_bytes=_positive_int(
                raw.get("validation_peak_reserved_bytes"),
                "validation peak",
            ),
            validation_chunks=_positive_int(
                raw.get("validation_chunks"),
                "validation chunks",
            ),
        )
        if profile.validated_batch_size <= 2:
            raise ValueError(f"validation batch is too small: {variant.value}")
        if profile.minimum_vram_bytes < profile.peak_bs1_bytes:
            raise ValueError(f"minimum VRAM is below the BS1 peak: {variant.value}")
        if profile.validation_peak_reserved_bytes <= profile.peak_bs2_bytes:
            raise ValueError(f"invalid validation peak: {variant.value}")
        profiles[variant] = profile

    benchmark = data.get("benchmark")
    catalog = BatchProfileCatalog(
        reserve_bytes=reserve_bytes,
        low_vram_max_bytes=low_vram_max_bytes,
        low_vram_reserve_bytes=low_vram_reserve_bytes,
        prefer_even=bool(data.get("prefer_even", True)),
        profiles=profiles,
        benchmark=dict(benchmark) if isinstance(benchmark, dict) else {},
    )
    benchmark_vram = _positive_int(
        catalog.benchmark.get("total_vram_bytes"),
        "benchmark VRAM",
    )
    for variant, profile in profiles.items():
        if catalog.recommend(variant, benchmark_vram) != profile.validated_batch_size:
            raise ValueError(f"stale MuScriptor validated batch size: {variant.value}")
    return catalog


@lru_cache(maxsize=1)
def get_batch_profile_catalog() -> BatchProfileCatalog:
    return load_batch_profile_catalog()


def recommend_batch_size(
    model: ModelVariant | str,
    total_vram_bytes: int,
) -> int:
    return get_batch_profile_catalog().recommend(model, total_vram_bytes)


def reserve_bytes_for_total(total_vram_bytes: int) -> int:
    return get_batch_profile_catalog().reserve_for(total_vram_bytes)
