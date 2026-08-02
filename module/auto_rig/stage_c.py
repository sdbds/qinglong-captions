from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .artifacts import atomic_write_bytes, atomic_write_json
from .capabilities import CapabilityPlan
from .control_bindings import ControlBindingPlan
from .control_registry import ControlRegistryPlan
from .export_symbols import build_global_export_symbol_table
from .format_plans import build_format_plan_set
from .manifests import (
    StageManifest,
    build_stage_manifest,
    manifest_relative_path,
    write_stage_manifest,
)
from .preset_library import PresetLibraryPlan
from .primitive_candidates import enumerate_primitive_candidates
from .projections import (
    MotionManifestProjection,
    RigReportProjection,
    motion_manifest_bytes,
    project_motion_manifest,
    project_rig_report,
    rig_report_bytes,
    validate_motion_manifest_projection,
    validate_rig_report_projection,
)
from .rig_document import (
    RigDocument,
    build_rig_document,
    load_rig_document,
    rig_document_bytes,
)
from .rig_geometry import RigGeometryCache
from .texture_plan import (
    CanonicalTexturePageSet,
    TexturePagePlan,
    build_texture_page_plan,
    materialize_canonical_texture_pages,
    texture_region_input,
)
from .texture_sources import LoadedTextureRegion

STAGE_C_SCHEMA_VERSION = 1
STAGE_C_ALGORITHM_VERSION = "stage-c-rig-document-v1"
STAGE_C_FAILURE_SCHEMA_VERSION = "stage-failure-v1"


class StageCError(RuntimeError):
    """Raised when Stage C cannot publish a complete validated transaction."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class StageCResult:
    rig: RigDocument
    motion_manifest: MotionManifestProjection
    report: RigReportProjection
    canonical_texture_pages: CanonicalTexturePageSet
    manifest: StageManifest
    rig_bytes: bytes
    motion_manifest_bytes: bytes
    report_bytes: bytes


def _path(root: Path, relative_path: str) -> Path:
    return root / Path(*relative_path.split("/"))


def _write_failure(item_root: Path, error: StageCError) -> None:
    failure_path = item_root / "rig" / "cache" / "C" / "failure.json"
    atomic_write_json(
        failure_path,
        {
            "schema_version": STAGE_C_FAILURE_SCHEMA_VERSION,
            "status": "stage_failed",
            "stage_name": "C",
            "retryable": False,
            "diagnostics": [
                {
                    "code": error.code,
                    "stage": "C",
                    "message": str(error),
                }
            ],
        },
    )


def _publish_file(staging_root: Path, item_root: Path, relative_path: str) -> None:
    source = _path(staging_root, relative_path)
    if not source.is_file():
        raise StageCError(
            "stage_c_transaction_incomplete",
            f"staging output is missing: {relative_path}",
        )
    atomic_write_bytes(_path(item_root, relative_path), source.read_bytes())


def execute_stage_c(
    item_root: str | Path,
    *,
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    loaded_regions: Iterable[LoadedTextureRegion],
    expected_texture_plan: TexturePagePlan,
    profile_id: str,
    upstream_manifests: Mapping[str, str],
    relevant_config_fingerprint: str,
    native_composite_mode_by_part: dict[str, str] | None = None,
    native_quality_by_part: dict[str, dict[str, object]] | None = None,
) -> StageCResult:
    """Build, validate, and commit C-owned public files with the marker last."""

    root = Path(item_root).resolve(strict=True)
    marker = _path(root, manifest_relative_path("C"))
    failure_path = root / "rig" / "cache" / "C" / "failure.json"
    marker.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    try:
        regions = tuple(loaded_regions)
        if not regions or any(
            not isinstance(region, LoadedTextureRegion) for region in regions
        ):
            raise StageCError(
                "input_contract_mismatch",
                "Stage C requires authenticated texture-region records",
            )
        actual_texture_plan = build_texture_page_plan(
            texture_region_input(region) for region in regions
        )
        if actual_texture_plan != expected_texture_plan:
            raise StageCError(
                "texture_plan_mismatch",
                "C recomputation differs from A's final TexturePagePlan",
            )
        texture_page_ids = tuple(
            f"texture-page/page_{page.index}" for page in actual_texture_plan.pages
        )
        candidates = enumerate_primitive_candidates(
            cache,
            controls,
            presets,
            bindings,
            texture_page_ids=texture_page_ids,
        )
        symbols = build_global_export_symbol_table(candidates, controls)
        formats = build_format_plan_set(
            cache,
            controls,
            presets,
            capabilities,
            bindings,
            candidates,
            symbols,
            profile_id=profile_id,
        )

        cache_dir = root / "rig" / "cache" / "C"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="staging-", dir=cache_dir) as temporary:
            staging_root = Path(temporary)
            canonical_pages = materialize_canonical_texture_pages(
                actual_texture_plan,
                regions,
                item_root=staging_root,
            )
            rig = build_rig_document(
                cache,
                controls,
                presets,
                capabilities,
                bindings,
                formats,
                candidates,
                symbols,
                actual_texture_plan,
                canonical_pages,
                native_composite_mode_by_part=native_composite_mode_by_part,
                native_quality_by_part=native_quality_by_part,
            )
            rig_bytes_value = rig_document_bytes(rig)
            atomic_write_bytes(_path(staging_root, "rig/rig.json"), rig_bytes_value)
            disk_rig = load_rig_document(_path(staging_root, "rig/rig.json"))
            if disk_rig != rig:
                raise StageCError(
                    "stage_c_transaction_incomplete",
                    "staged RigDocument changed after disk round-trip",
                )
            motion = project_motion_manifest(disk_rig)
            report = project_rig_report(disk_rig)
            validate_motion_manifest_projection(motion, disk_rig)
            validate_rig_report_projection(report, disk_rig)
            motion_bytes_value = motion_manifest_bytes(motion)
            report_bytes_value = rig_report_bytes(report)
            atomic_write_bytes(
                _path(staging_root, "rig/motion_manifest.json"),
                motion_bytes_value,
            )
            atomic_write_bytes(
                _path(staging_root, "rig/report.json"),
                report_bytes_value,
            )

            output_paths = [
                "rig/rig.json",
                "rig/motion_manifest.json",
                "rig/report.json",
                *(page.relative_path for page in canonical_pages.pages),
            ]
            for relative_path in output_paths:
                _publish_file(staging_root, root, relative_path)

        status = (
            "stage_validated_with_degradation"
            if cache.degradation_state == "degraded"
            else "stage_validated"
        )
        manifest = build_stage_manifest(
            root,
            stage_name="C",
            stage_schema_version=STAGE_C_SCHEMA_VERSION,
            algorithm_version=STAGE_C_ALGORITHM_VERSION,
            upstream_manifests=upstream_manifests,
            input_file_sha256=(),
            target_input_fingerprint=cache.target_input_fingerprint,
            native_variant_set_sha256=cache.native_variant_set_sha256,
            native_variant_eligibility_sha256=cache.native_variant_eligibility_sha256,
            relevant_config_fingerprint=relevant_config_fingerprint,
            rig_overrides_sha256=cache.rig_overrides_sha256,
            output_paths=output_paths,
            status=status,
        )
        write_stage_manifest(root, manifest)
        failure_path.unlink(missing_ok=True)
        return StageCResult(
            rig=rig,
            motion_manifest=motion,
            report=report,
            canonical_texture_pages=canonical_pages,
            manifest=manifest,
            rig_bytes=rig_bytes_value,
            motion_manifest_bytes=motion_bytes_value,
            report_bytes=report_bytes_value,
        )
    except StageCError as exc:
        _write_failure(root, exc)
        raise
    except Exception as exc:
        wrapped = StageCError("stage_c_failed", str(exc))
        _write_failure(root, wrapped)
        raise wrapped from exc


__all__ = [
    "STAGE_C_ALGORITHM_VERSION",
    "STAGE_C_FAILURE_SCHEMA_VERSION",
    "STAGE_C_SCHEMA_VERSION",
    "StageCError",
    "StageCResult",
    "execute_stage_c",
]
