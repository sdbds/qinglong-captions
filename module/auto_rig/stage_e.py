from __future__ import annotations

import hashlib
import json
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
from .export.live2d.animations import (
    Live2DAnimationPlan,
    build_live2d_animation_plan,
    encode_live2d_animation_asset,
)
from .export.live2d.artmesh import (
    Live2DArtMeshPlan,
    build_live2d_artmesh_plan,
)
from .export.live2d.attestation import Live2DRuntimeAttestation
from .export.live2d.binding_plan import (
    Live2DBindingPlan,
    build_live2d_binding_plan,
)
from .export.live2d.coordinates import (
    Live2DCoordinatePlan,
    build_live2d_coordinate_plan,
)
from .export.live2d.document import build_live2d_moc3_document
from .export.live2d.keyforms import (
    Live2DKeyformPlan,
    build_live2d_keyform_plan,
)
from .export.live2d.moc3_codec import Moc3V400Document, encode_moc3_v400
from .export.live2d.release_validator import (
    Live2DReleaseValidationReport,
    validate_live2d_release_bundle,
)
from .export.live2d.rigid_drivers import (
    RigidDriverRegistry,
    build_rigid_driver_registry,
)
from .export.live2d.runtime_assets import (
    Live2DRuntimeAssetPlan,
    build_live2d_runtime_asset_plan,
    encode_live2d_runtime_asset,
)
from .export.live2d.runtime_toolchain import (
    Live2DRuntimeToolchain,
    Live2DRuntimeToolchainError,
)
from .export.live2d.symbols import (
    Live2DSymbolView,
    build_live2d_symbol_view,
)
from .export.live2d.validator import (
    Live2DStructureValidationReport,
    build_live2d_structure_validation_report,
)
from .jcs import jcs_bytes, jcs_sha256
from .manifests import (
    StageManifest,
    build_stage_manifest,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)
from .rig_document import RigDocument, load_rig_document

STAGE_E_SCHEMA_VERSION = 1
STAGE_E_ALGORITHM_VERSION = "stage-e-live2d-moc3-v3"
STAGE_E_FAILURE_SCHEMA_VERSION = "stage-failure-v1"
LIVE2D_EXPORT_REPORT_VERSION = "live2d-export-report-v1"
LIVE2D_VALIDATION_TIERS = frozenset({"structural", "release"})


class StageEError(RuntimeError):
    """Raised when Stage E cannot publish a validated Live2D transaction."""

    def __init__(self, code: str, message: str, *, log_path: Path | None = None) -> None:
        self.code = code
        self.log_path = log_path
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class StageEResult:
    rig: RigDocument
    symbols: Live2DSymbolView
    rigid_registry: RigidDriverRegistry
    bindings: Live2DBindingPlan
    coordinates: Live2DCoordinatePlan
    artmeshes: Live2DArtMeshPlan
    keyforms: Live2DKeyformPlan
    document: Moc3V400Document
    animations: Live2DAnimationPlan
    runtime_assets: Live2DRuntimeAssetPlan
    structure_validation: Live2DStructureValidationReport
    release_validation: Live2DReleaseValidationReport | None
    report: dict[str, object]
    manifest: StageManifest
    moc_bytes: bytes
    report_bytes: bytes


def _path(root: Path, relative_path: str) -> Path:
    return root / Path(*relative_path.split("/"))


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_failure(item_root: Path, error: StageEError) -> None:
    diagnostic: dict[str, object] = {
        "code": error.code,
        "stage": "E",
        "message": str(error),
    }
    if error.log_path is not None:
        diagnostic["log_path"] = str(error.log_path)
    atomic_write_json(
        item_root / "rig" / "cache" / "E" / "failure.json",
        {
            "schema_version": STAGE_E_FAILURE_SCHEMA_VERSION,
            "status": "stage_failed",
            "stage_name": "E",
            "retryable": False,
            "diagnostics": [diagnostic],
        },
    )


def publish_stage_e_toolchain_failure(
    item_root: str | Path,
    error: Live2DRuntimeToolchainError,
) -> StageEError:
    """Publish a pre-transaction Stage E toolchain failure and preserve its code."""

    root = Path(item_root).resolve(strict=True)
    _path(root, manifest_relative_path("E")).unlink(missing_ok=True)
    wrapped = StageEError(
        error.code,
        str(error),
        log_path=error.log_path,
    )
    _write_failure(root, wrapped)
    return wrapped


def _publish_file(staging_root: Path, item_root: Path, relative_path: str) -> None:
    source = _path(staging_root, relative_path)
    if not source.is_file():
        raise StageEError(
            "stage_e_transaction_incomplete",
            f"staging output is missing: {relative_path}",
        )
    atomic_write_bytes(_path(item_root, relative_path), source.read_bytes())


def _validate_c_upstream(
    root: Path,
    upstream_manifests: Mapping[str, str],
) -> StageManifest:
    if set(upstream_manifests) != {"C"}:
        raise StageEError(
            "input_contract_mismatch",
            "Stage E must bind exactly the current C manifest",
        )
    c_manifest = read_stage_manifest(root, "C")
    marker = _path(root, manifest_relative_path("C"))
    if upstream_manifests.get("C") != sha256_file(marker):
        raise StageEError(
            "upstream_manifest_mismatch",
            "Stage E upstream digest differs from the current C marker",
        )
    if c_manifest.status not in {
        "stage_validated",
        "stage_validated_with_degradation",
    }:
        raise StageEError(
            "input_contract_mismatch",
            "Stage C is not in a reusable validated status",
        )
    return c_manifest


def _validate_rig_identity(
    rig: RigDocument,
    c_manifest: StageManifest,
) -> None:
    outputs = {record.path: record for record in c_manifest.output_file_sha256}
    expected = outputs.get("rig/rig.json")
    if expected is None or rig.document_sha256 != expected.sha256:
        raise StageEError(
            "input_contract_mismatch",
            "RigDocument digest differs from the committed C manifest",
        )
    payload = rig.to_dict()
    if (
        payload["input_fingerprint"] != c_manifest.target_input_fingerprint
        or payload["input"]["native_variant_set_sha256"] != c_manifest.native_variant_set_sha256
        or payload["input"]["native_variant_eligibility_sha256"] != c_manifest.native_variant_eligibility_sha256
    ):
        raise StageEError(
            "input_contract_mismatch",
            "RigDocument semantic identity differs from Stage C",
        )


def _source_pages(
    root: Path,
    rig_payload: Mapping[str, object],
    runtime_assets: Live2DRuntimeAssetPlan,
) -> tuple[dict[str, bytes], tuple[FileDigest, ...]]:
    raw_pages = rig_payload.get("texture_pages")
    if not isinstance(raw_pages, list):
        raise StageEError(
            "input_contract_mismatch",
            "RigDocument texture_pages is not a list",
        )
    by_index = {raw.get("index"): raw for raw in raw_pages if isinstance(raw, Mapping) and isinstance(raw.get("index"), int)}
    expected_paths = tuple(f"textures/page_{index}.png" for index in range(len(by_index)))
    if runtime_assets.referenced_texture_paths != expected_paths:
        raise StageEError(
            "input_contract_mismatch",
            "Live2D texture references differ from canonical page indices",
        )
    pages: dict[str, bytes] = {}
    inputs = []
    for index, destination_relative in enumerate(expected_paths):
        raw = by_index.get(index)
        if raw is None:
            raise StageEError(
                "input_contract_mismatch",
                f"RigDocument lacks canonical texture page {index}",
            )
        source_relative = raw.get("relative_path")
        expected_digest = raw.get("encoded_png_sha256")
        if not isinstance(source_relative, str) or not isinstance(expected_digest, str):
            raise StageEError(
                "input_contract_mismatch",
                f"texture page {index} lacks a canonical file contract",
            )
        source = _path(root, source_relative)
        if not source.is_file():
            raise StageEError(
                "input_contract_mismatch",
                f"canonical texture page is missing: {source_relative}",
            )
        encoded = source.read_bytes()
        if _sha256(encoded) != expected_digest:
            raise StageEError(
                "canonical_texture_page_mismatch",
                f"canonical texture page digest changed: {source_relative}",
            )
        pages[destination_relative] = encoded
        inputs.append(describe_file(root, source_relative))
    return pages, tuple(inputs)


def _build_export_report(
    *,
    rig: RigDocument,
    symbols: Live2DSymbolView,
    rigid_registry: RigidDriverRegistry,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
    keyforms: Live2DKeyformPlan,
    animations: Live2DAnimationPlan,
    runtime_assets: Live2DRuntimeAssetPlan,
    structure_validation: Live2DStructureValidationReport,
    release_validation: Live2DReleaseValidationReport | None,
    artifacts: Mapping[str, bytes],
    validation_tier: str,
    status: str,
    runtime_attestation: Live2DRuntimeAttestation | None,
) -> dict[str, object]:
    payload = rig.to_dict()
    format_plans = payload["format_plans"]
    model_plans = format_plans["model_plans"]
    model_matches = [record for record in model_plans if record["format_id"] == "live2d_moc3_v4_00"]
    if len(model_matches) != 1 or model_matches[0]["status"] != "supported":
        raise StageEError(
            "live2d_model_not_supported",
            "Stage C did not authorize a Live2D MOC3 V4.00 artifact",
        )
    release_record: dict[str, object]
    if release_validation is None:
        release_record = {
            "status": "not_run",
            "reason": "structural tier does not execute official Core/SDK gates",
        }
    else:
        if runtime_attestation is None:
            raise StageEError(
                "stage_e_transaction_incomplete",
                "release validation has no selected runtime attestation",
            )
        release_record = {
            "status": "passed",
            "platform_id": runtime_attestation.platform_id,
            "backend_id": runtime_attestation.backend_id,
            "attestation_record_sha256": runtime_attestation.record_sha256,
            "report": release_validation.to_dict(),
        }
    base = {
        "schema_version": LIVE2D_EXPORT_REPORT_VERSION,
        "algorithm_version": STAGE_E_ALGORITHM_VERSION,
        "status": status,
        "validation_tier": validation_tier,
        "formal_release_eligible": release_validation is not None,
        "format_id": "live2d_moc3_v4_00",
        "profile_id": format_plans["profile"]["profile_id"],
        "rig_json_sha256": rig.document_sha256,
        "format_model_plan_sha256": model_matches[0]["plan_sha256"],
        "format_plan_set_sha256": format_plans["plan_sha256"],
        "global_symbol_table_sha256": payload["export_symbols"]["table_sha256"],
        "plans": {
            "symbol_view_sha256": symbols.view_sha256,
            "rigid_driver_registry_sha256": rigid_registry.registry_sha256,
            "binding_plan_sha256": bindings.plan_sha256,
            "coordinate_plan_sha256": coordinates.plan_sha256,
            "artmesh_plan_sha256": artmeshes.plan_sha256,
            "keyform_plan_sha256": keyforms.plan_sha256,
            "animation_plan_sha256": animations.plan_sha256,
            "runtime_asset_plan_sha256": runtime_assets.plan_sha256,
        },
        "parameters": [record.to_dict() for record in bindings.parameters],
        "rotation_instances": [record.to_dict() for record in bindings.rotation_instances],
        "bindings": [record.to_dict() for record in bindings.bindings],
        "artifacts": [
            {
                "path": path,
                "size": len(encoded),
                "sha256": _sha256(encoded),
            }
            for path, encoded in sorted(artifacts.items())
        ],
        "structure_validation": structure_validation.to_dict(),
        "release_validation": release_record,
    }
    return {**base, "report_sha256": jcs_sha256(base)}


def execute_stage_e(
    item_root: str | Path,
    *,
    upstream_manifests: Mapping[str, str],
    relevant_config_fingerprint: str,
    validation_tier: str = "release",
    runtime_toolchain: Live2DRuntimeToolchain | None = None,
    runtime_attestation: Live2DRuntimeAttestation | None = None,
) -> StageEResult:
    """Compile, validate, and commit E-owned Live2D files marker-last."""

    root = Path(item_root).resolve(strict=True)
    marker = _path(root, manifest_relative_path("E"))
    failure_path = root / "rig" / "cache" / "E" / "failure.json"
    marker.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    try:
        if validation_tier not in LIVE2D_VALIDATION_TIERS:
            raise StageEError(
                "input_contract_mismatch",
                "validation_tier must be structural or release",
            )
        if validation_tier == "release" and (
            runtime_toolchain is None or runtime_attestation is None
        ):
            raise StageEError(
                "live2d_release_gate_unavailable",
                "release tier requires a resolved Cubism runtime toolchain and attestation",
            )
        c_manifest = _validate_c_upstream(root, upstream_manifests)
        rig = load_rig_document(root / "rig" / "rig.json")
        _validate_rig_identity(rig, c_manifest)
        payload = rig.to_dict()

        symbols = build_live2d_symbol_view(payload["export_symbols"])
        rigid_registry = build_rigid_driver_registry(payload["control_specs"])
        bindings = build_live2d_binding_plan(rig, symbols, rigid_registry)
        coordinates = build_live2d_coordinate_plan(rig, bindings)
        artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
        keyforms = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)
        document = build_live2d_moc3_document(rig, bindings, coordinates, artmeshes, keyforms)
        moc_bytes = encode_moc3_v400(document)
        animations = build_live2d_animation_plan(rig, symbols, bindings, keyforms)
        runtime_assets = build_live2d_runtime_asset_plan(rig, symbols, bindings, artmeshes, animations)
        source_pages, page_input_digests = _source_pages(root, payload, runtime_assets)
        input_digests = (
            describe_file(root, "rig/rig.json"),
            *page_input_digests,
        )
        status = (
            "stage_validated_with_degradation" if payload["provenance"]["degradation_state"] == "degraded" else "stage_validated"
        )

        cache_dir = root / "rig" / "cache" / "E"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="staging-", dir=cache_dir) as temporary:
            staging_root = Path(temporary)
            bundle_root = staging_root / "rig" / "live2d"
            artifact_bytes: dict[str, bytes] = {
                "rig/live2d/model.moc3": moc_bytes,
                f"rig/live2d/{runtime_assets.model3.relative_path}": (encode_live2d_runtime_asset(runtime_assets.model3)),
                f"rig/live2d/{runtime_assets.cdi3.relative_path}": (encode_live2d_runtime_asset(runtime_assets.cdi3)),
                **{
                    f"rig/live2d/{asset.relative_path}": (encode_live2d_animation_asset(asset))
                    for asset in (
                        *animations.motion_assets,
                        *animations.expression_assets,
                    )
                },
                **{f"rig/live2d/{relative_path}": encoded for relative_path, encoded in source_pages.items()},
            }
            for relative_path, encoded in artifact_bytes.items():
                atomic_write_bytes(_path(staging_root, relative_path), encoded)

            structure_validation = build_live2d_structure_validation_report(
                moc_bytes,
                rig,
                bindings,
                coordinates,
                artmeshes,
                keyforms,
            )
            release_validation = None
            if validation_tier == "release":
                release_validation = validate_live2d_release_bundle(
                    bundle_root,
                    runtime_toolchain=runtime_toolchain,
                    runtime_attestation=runtime_attestation,
                    rig=rig,
                    bindings=bindings,
                    coordinates=coordinates,
                    artmeshes=artmeshes,
                    keyforms=keyforms,
                    animations=animations,
                    runtime_assets=runtime_assets,
                    structure_report=structure_validation,
                )
            report = _build_export_report(
                rig=rig,
                symbols=symbols,
                rigid_registry=rigid_registry,
                bindings=bindings,
                coordinates=coordinates,
                artmeshes=artmeshes,
                keyforms=keyforms,
                animations=animations,
                runtime_assets=runtime_assets,
                structure_validation=structure_validation,
                release_validation=release_validation,
                artifacts=artifact_bytes,
                validation_tier=validation_tier,
                status=status,
                runtime_attestation=runtime_attestation,
            )
            report_bytes = jcs_bytes(report)
            if json.loads(report_bytes) != report:
                raise StageEError(
                    "stage_e_transaction_incomplete",
                    "staged export report did not round-trip canonical JCS",
                )
            if report["report_sha256"] != jcs_sha256({key: value for key, value in report.items() if key != "report_sha256"}):
                raise StageEError(
                    "stage_e_transaction_incomplete",
                    "staged export report digest is invalid",
                )
            report_path = "rig/live2d/export_report.json"
            atomic_write_bytes(_path(staging_root, report_path), report_bytes)
            output_paths = [*sorted(artifact_bytes), report_path]
            for relative_path in output_paths:
                _publish_file(staging_root, root, relative_path)

        manifest = build_stage_manifest(
            root,
            stage_name="E",
            stage_schema_version=STAGE_E_SCHEMA_VERSION,
            algorithm_version=STAGE_E_ALGORITHM_VERSION,
            upstream_manifests=upstream_manifests,
            input_file_sha256=input_digests,
            target_input_fingerprint=c_manifest.target_input_fingerprint,
            native_variant_set_sha256=c_manifest.native_variant_set_sha256,
            native_variant_eligibility_sha256=(c_manifest.native_variant_eligibility_sha256),
            relevant_config_fingerprint=relevant_config_fingerprint,
            rig_overrides_sha256=c_manifest.rig_overrides_sha256,
            output_paths=output_paths,
            status=status,
        )
        write_stage_manifest(root, manifest)
        failure_path.unlink(missing_ok=True)
        return StageEResult(
            rig=rig,
            symbols=symbols,
            rigid_registry=rigid_registry,
            bindings=bindings,
            coordinates=coordinates,
            artmeshes=artmeshes,
            keyforms=keyforms,
            document=document,
            animations=animations,
            runtime_assets=runtime_assets,
            structure_validation=structure_validation,
            release_validation=release_validation,
            report=report,
            manifest=manifest,
            moc_bytes=moc_bytes,
            report_bytes=report_bytes,
        )
    except StageEError as exc:
        _write_failure(root, exc)
        raise
    except Exception as exc:
        wrapped = StageEError("stage_e_failed", str(exc))
        _write_failure(root, wrapped)
        raise wrapped from exc


__all__ = [
    "LIVE2D_EXPORT_REPORT_VERSION",
    "LIVE2D_VALIDATION_TIERS",
    "STAGE_E_ALGORITHM_VERSION",
    "STAGE_E_FAILURE_SCHEMA_VERSION",
    "STAGE_E_SCHEMA_VERSION",
    "StageEError",
    "StageEResult",
    "execute_stage_e",
    "publish_stage_e_toolchain_failure",
]
