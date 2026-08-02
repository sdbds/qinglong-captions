from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .artifacts import (
    FileDigest,
    atomic_write_bytes,
    atomic_write_json,
    describe_file,
    sha256_file,
)
from .export.spine.animations import (
    SpineAnimationPlan,
    build_spine_animation_plan,
)
from .export.spine.atlas import (
    SpineAtlasPlan,
    build_spine_atlas_plan,
    serialize_spine_atlas,
)
from .export.spine.bind_plan import SpineBindPlan, build_spine_bind_plan
from .export.spine.coordinates import (
    SpineCoordinatePlan,
    build_spine_coordinate_plan,
)
from .export.spine.model import SpineDocument, build_spine_document
from .export.spine.serializer import (
    build_spine_encoding_descriptor,
    parse_spine_document,
    serialize_spine_document,
    serialize_spine_report,
)
from .export.spine.symbols import SpineSymbolView, build_spine_symbol_view
from .export.spine.validator import (
    SpineBundleValidationReport,
    validate_spine_bundle,
)
from .jcs import jcs_sha256
from .manifests import (
    StageManifest,
    build_stage_manifest,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)
from .rig_document import RigDocument, load_rig_document

STAGE_D_SCHEMA_VERSION = 1
STAGE_D_ALGORITHM_VERSION = "stage-d-spine-4-2-v1"
STAGE_D_FAILURE_SCHEMA_VERSION = "stage-failure-v1"
SPINE_EXPORT_REPORT_VERSION = "spine-export-report-v1"


class StageDError(RuntimeError):
    """Raised when Stage D cannot publish a validated Spine 4.2 transaction."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class StageDResult:
    rig: RigDocument
    coordinate_plan: SpineCoordinatePlan
    bind_plan: SpineBindPlan
    symbol_view: SpineSymbolView
    atlas_plan: SpineAtlasPlan
    animation_plan: SpineAnimationPlan
    document: SpineDocument
    validation: SpineBundleValidationReport
    report: dict[str, object]
    manifest: StageManifest
    skeleton_bytes: bytes
    atlas_bytes: bytes
    report_bytes: bytes


def _path(root: Path, relative_path: str) -> Path:
    return root / Path(*relative_path.split("/"))


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_failure(item_root: Path, error: StageDError) -> None:
    atomic_write_json(
        item_root / "rig" / "cache" / "D" / "failure.json",
        {
            "schema_version": STAGE_D_FAILURE_SCHEMA_VERSION,
            "status": "stage_failed",
            "stage_name": "D",
            "retryable": False,
            "diagnostics": [
                {
                    "code": error.code,
                    "stage": "D",
                    "message": str(error),
                }
            ],
        },
    )


def _publish_file(staging_root: Path, item_root: Path, relative_path: str) -> None:
    source = _path(staging_root, relative_path)
    if not source.is_file():
        raise StageDError(
            "stage_d_transaction_incomplete",
            f"staging output is missing: {relative_path}",
        )
    atomic_write_bytes(_path(item_root, relative_path), source.read_bytes())


def _validate_c_upstream(
    root: Path,
    upstream_manifests: Mapping[str, str],
) -> StageManifest:
    if set(upstream_manifests) != {"C"}:
        raise StageDError(
            "input_contract_mismatch",
            "Stage D must bind exactly the current C manifest",
        )
    c_manifest = read_stage_manifest(root, "C")
    marker = _path(root, manifest_relative_path("C"))
    actual_marker_sha256 = sha256_file(marker)
    if upstream_manifests.get("C") != actual_marker_sha256:
        raise StageDError(
            "upstream_manifest_mismatch",
            "Stage D upstream digest differs from the current C marker",
        )
    if c_manifest.status not in {
        "stage_validated",
        "stage_validated_with_degradation",
    }:
        raise StageDError(
            "input_contract_mismatch",
            "Stage C is not in a reusable validated status",
        )
    return c_manifest


def _source_pages(
    root: Path,
    rig_payload: Mapping[str, object],
    atlas: SpineAtlasPlan,
) -> tuple[dict[str, bytes], tuple[FileDigest, ...]]:
    raw_pages = rig_payload.get("texture_pages")
    if not isinstance(raw_pages, list):
        raise StageDError(
            "input_contract_mismatch", "RigDocument texture_pages is not a list"
        )
    by_index = {
        raw.get("index"): raw
        for raw in raw_pages
        if isinstance(raw, Mapping) and isinstance(raw.get("index"), int)
    }
    pages: dict[str, bytes] = {}
    inputs = []
    for page in atlas.pages:
        raw = by_index.get(page.index)
        if raw is None:
            raise StageDError(
                "input_contract_mismatch",
                f"RigDocument lacks canonical texture page {page.index}",
            )
        source_relative = raw.get("relative_path")
        if not isinstance(source_relative, str):
            raise StageDError(
                "input_contract_mismatch",
                f"texture page {page.index} has no canonical source path",
            )
        source = _path(root, source_relative)
        if not source.is_file():
            raise StageDError(
                "input_contract_mismatch",
                f"canonical texture page is missing: {source_relative}",
            )
        payload = source.read_bytes()
        if _sha256(payload) != page.source_encoded_png_sha256:
            raise StageDError(
                "canonical_texture_page_mismatch",
                f"canonical texture page digest changed: {source_relative}",
            )
        pages[page.path] = payload
        inputs.append(describe_file(root, source_relative))
    return pages, tuple(inputs)


def _build_export_report(
    *,
    rig: RigDocument,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
    atlas: SpineAtlasPlan,
    animations: SpineAnimationPlan,
    validation: SpineBundleValidationReport,
    skeleton_bytes: bytes,
    atlas_bytes: bytes,
    pages: Mapping[str, bytes],
    status: str,
) -> dict[str, object]:
    rig_payload = rig.to_dict()
    format_plans = rig_payload["format_plans"]
    profile = format_plans["profile"]
    model_plan = next(
        item
        for item in format_plans["model_plans"]
        if item["format_id"] == "spine_4_2"
    )
    if model_plan["status"] != "supported":
        raise StageDError(
            "spine_model_not_supported",
            "Stage C did not authorize a Spine 4.2 model artifact",
        )
    encoding = build_spine_encoding_descriptor()
    artifacts = [
        {
            "path": "rig/spine/skeleton.json",
            "size": len(skeleton_bytes),
            "sha256": _sha256(skeleton_bytes),
        },
        {
            "path": "rig/spine/skeleton.atlas",
            "size": len(atlas_bytes),
            "sha256": _sha256(atlas_bytes),
        },
        *(
            {
                "path": f"rig/spine/{path}",
                "size": len(payload),
                "sha256": _sha256(payload),
            }
            for path, payload in sorted(pages.items())
        ),
    ]
    base = {
        "schema_version": SPINE_EXPORT_REPORT_VERSION,
        "algorithm_version": STAGE_D_ALGORITHM_VERSION,
        "status": status,
        "format_id": "spine_4_2",
        "profile_id": profile["profile_id"],
        "rig_json_sha256": rig.document_sha256,
        "format_model_plan_sha256": model_plan["plan_sha256"],
        "format_plan_set_sha256": format_plans["plan_sha256"],
        "global_symbol_table_sha256": rig_payload["export_symbols"]["table_sha256"],
        "plans": {
            "coordinate_plan_sha256": coordinates.plan_sha256,
            "bind_plan_sha256": bind.plan_sha256,
            "symbol_view_sha256": symbols.view_sha256,
            "atlas_plan_sha256": atlas.plan_sha256,
            "animation_plan_sha256": animations.plan_sha256,
        },
        "encoding": encoding.to_dict(),
        "animations": [record.to_dict() for record in animations.records],
        "artifacts": artifacts,
        "validation": validation.to_dict(),
        "official_spine_runtime_gate": {
            "status": "not_run",
            "reason": "Spine 4.2 Editor/runtime is an external opt-in release gate",
        },
    }
    return {**base, "report_sha256": jcs_sha256(base)}


def execute_stage_d(
    item_root: str | Path,
    *,
    upstream_manifests: Mapping[str, str],
    relevant_config_fingerprint: str,
) -> StageDResult:
    """Compile, validate, and commit D-owned Spine 4.2 files marker-last."""

    root = Path(item_root).resolve(strict=True)
    marker = _path(root, manifest_relative_path("D"))
    failure_path = root / "rig" / "cache" / "D" / "failure.json"
    marker.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    try:
        c_manifest = _validate_c_upstream(root, upstream_manifests)
        rig_path = root / "rig" / "rig.json"
        rig = load_rig_document(rig_path)
        payload = rig.to_dict()
        if (
            rig.document_sha256
            != next(
                item.sha256
                for item in c_manifest.output_file_sha256
                if item.path == "rig/rig.json"
            )
        ):
            raise StageDError(
                "input_contract_mismatch",
                "RigDocument digest differs from the committed C manifest",
            )
        if (
            payload["input_fingerprint"] != c_manifest.target_input_fingerprint
            or payload["input"]["native_variant_set_sha256"]
            != c_manifest.native_variant_set_sha256
            or payload["input"]["native_variant_eligibility_sha256"]
            != c_manifest.native_variant_eligibility_sha256
        ):
            raise StageDError(
                "input_contract_mismatch",
                "RigDocument semantic identity differs from Stage C",
            )

        coordinates = build_spine_coordinate_plan(
            payload["canvas"], input_fingerprint=payload["input_fingerprint"]
        )
        bind = build_spine_bind_plan(payload["bones"], payload["meshes"], coordinates)
        symbols = build_spine_symbol_view(payload["export_symbols"])
        atlas = build_spine_atlas_plan(
            payload["texture_pages"], payload["parts"], symbols
        )
        setup = build_spine_document(rig, coordinates, bind, symbols, atlas)
        animations = build_spine_animation_plan(
            rig, coordinates, bind, symbols, setup
        )
        document = build_spine_document(
            rig,
            coordinates,
            bind,
            symbols,
            atlas,
            animations=animations.animations,
        )
        skeleton_bytes = serialize_spine_document(document)
        atlas_bytes = serialize_spine_atlas(atlas)
        source_pages, page_input_digests = _source_pages(root, payload, atlas)
        input_digests = (
            describe_file(root, "rig/rig.json"),
            *page_input_digests,
        )
        degraded = payload["provenance"]["degradation_state"] == "degraded"
        status = "stage_validated_with_degradation" if degraded else "stage_validated"

        cache_dir = root / "rig" / "cache" / "D"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="staging-", dir=cache_dir) as temporary:
            staging_root = Path(temporary)
            atomic_write_bytes(
                _path(staging_root, "rig/spine/skeleton.json"), skeleton_bytes
            )
            atomic_write_bytes(
                _path(staging_root, "rig/spine/skeleton.atlas"), atlas_bytes
            )
            for page_path, page_bytes in source_pages.items():
                atomic_write_bytes(
                    _path(staging_root, f"rig/spine/{page_path}"), page_bytes
                )
            staged_pages = {
                page.path: _path(
                    staging_root, f"rig/spine/{page.path}"
                ).read_bytes()
                for page in atlas.pages
            }
            validation = validate_spine_bundle(
                _path(staging_root, "rig/spine/skeleton.json").read_bytes(),
                _path(staging_root, "rig/spine/skeleton.atlas").read_bytes(),
                staged_pages,
                rig,
                coordinates,
                bind,
                symbols,
                atlas,
                animations,
            )
            report = _build_export_report(
                rig=rig,
                coordinates=coordinates,
                bind=bind,
                symbols=symbols,
                atlas=atlas,
                animations=animations,
                validation=validation,
                skeleton_bytes=skeleton_bytes,
                atlas_bytes=atlas_bytes,
                pages=source_pages,
                status=status,
            )
            report_bytes = serialize_spine_report(report)
            if parse_spine_document(report_bytes) != report:
                raise StageDError(
                    "stage_d_transaction_incomplete",
                    "staged export report did not round-trip canonical JCS",
                )
            if report["report_sha256"] != jcs_sha256(
                {key: value for key, value in report.items() if key != "report_sha256"}
            ):
                raise StageDError(
                    "stage_d_transaction_incomplete",
                    "staged export report digest is invalid",
                )
            atomic_write_bytes(
                _path(staging_root, "rig/spine/export_report.json"),
                report_bytes,
            )
            output_paths = [
                "rig/spine/skeleton.json",
                "rig/spine/skeleton.atlas",
                *(f"rig/spine/{page.path}" for page in atlas.pages),
                "rig/spine/export_report.json",
            ]
            for relative_path in output_paths:
                _publish_file(staging_root, root, relative_path)

        manifest = build_stage_manifest(
            root,
            stage_name="D",
            stage_schema_version=STAGE_D_SCHEMA_VERSION,
            algorithm_version=STAGE_D_ALGORITHM_VERSION,
            upstream_manifests=upstream_manifests,
            input_file_sha256=input_digests,
            target_input_fingerprint=c_manifest.target_input_fingerprint,
            native_variant_set_sha256=c_manifest.native_variant_set_sha256,
            native_variant_eligibility_sha256=(
                c_manifest.native_variant_eligibility_sha256
            ),
            relevant_config_fingerprint=relevant_config_fingerprint,
            rig_overrides_sha256=c_manifest.rig_overrides_sha256,
            output_paths=output_paths,
            status=status,
        )
        write_stage_manifest(root, manifest)
        failure_path.unlink(missing_ok=True)
        return StageDResult(
            rig=rig,
            coordinate_plan=coordinates,
            bind_plan=bind,
            symbol_view=symbols,
            atlas_plan=atlas,
            animation_plan=animations,
            document=document,
            validation=validation,
            report=report,
            manifest=manifest,
            skeleton_bytes=skeleton_bytes,
            atlas_bytes=atlas_bytes,
            report_bytes=report_bytes,
        )
    except StageDError as exc:
        _write_failure(root, exc)
        raise
    except Exception as exc:
        wrapped = StageDError("stage_d_failed", str(exc))
        _write_failure(root, wrapped)
        raise wrapped from exc


__all__ = [
    "SPINE_EXPORT_REPORT_VERSION",
    "STAGE_D_ALGORITHM_VERSION",
    "STAGE_D_FAILURE_SCHEMA_VERSION",
    "STAGE_D_SCHEMA_VERSION",
    "StageDError",
    "StageDResult",
    "execute_stage_d",
]
