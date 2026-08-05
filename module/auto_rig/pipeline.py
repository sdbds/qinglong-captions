from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Mapping

from .anatomy import AnatomyMaskGeometry, build_anatomy_mask_geometry
from .artifacts import (
    FileDigest,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    describe_file,
    sha256_file,
)
from .bone_graph import build_bone_graph
from .capabilities import derive_capabilities
from .component_geometry import load_component_geometry, load_mesh_component_sources
from .component_plan import (
    MaskComponentPlan,
    build_base_mask_component_plan,
    extend_mask_component_plan_with_variants,
)
from .contracts import load_auto_rig_input_contract
from .control_bindings import build_control_binding_plan
from .control_registry import build_control_registry_plan
from .draw_order import build_ordinary_draw_order
from .export.live2d.attestation import (
    Live2DAttestationError,
    load_packaged_live2d_frame_attestation,
    select_runtime_attestation,
)
from .export.live2d.runtime_toolchain import (
    LIVE2D_VALIDATOR_PROTOCOL_DIGEST,
    Live2DRuntimeToolchainError,
    ensure_live2d_runtime_toolchain,
)
from .format_plans import load_capability_profile
from .generic_variants import (
    GenericVariantSynthesisPlan,
    build_generic_variant_synthesis_plan,
)
from .input_identity import build_target_input_identity
from .joint_pipeline import build_stage_a_joint_plan
from .manifests import (
    StageManifest,
    build_stage_fingerprint,
    build_stage_manifest,
    clear_stage_public_outputs,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)
from .mask_sources import load_validated_part_alphas
from .mesh_builder import build_mesh_plan
from .native_variant_admission import (
    admit_native_variant_resources,
    expand_draw_order_with_admitted_variants,
)
from .native_variant_quality import build_native_variant_quality_plan
from .native_variants import NativeVariantSet, load_native_variant_set
from .overrides import load_rig_override_source, validate_rig_override_source
from .pose.contracts import PoseMode
from .pose.factory import PoseProviderPool, make_builtin_pose_resolver
from .pose.integration import PoseExecutionResult, execute_pose_observation
from .preset_library import build_preset_library_plan
from .rig_geometry import (
    RIG_GEOMETRY_CACHE_PATH,
    RigGeometryCache,
    build_rig_geometry_cache,
    load_rig_geometry_cache,
    rig_geometry_cache_bytes,
    validate_rig_geometry_cache_payload,
)
from .skinning import build_skinning_plan
from .stage_c import (
    STAGE_C_ALGORITHM_VERSION,
    STAGE_C_SCHEMA_VERSION,
    StageCResult,
    execute_stage_c,
)
from .stage_d import (
    STAGE_D_ALGORITHM_VERSION,
    STAGE_D_SCHEMA_VERSION,
    StageDResult,
    execute_stage_d,
)
from .stage_e import (
    LIVE2D_VALIDATION_TIERS,
    STAGE_E_ALGORITHM_VERSION,
    STAGE_E_SCHEMA_VERSION,
    StageEResult,
    execute_stage_e,
    publish_stage_e_toolchain_failure,
)
from .stage_g import execute_stage_g_success
from .stage_graph import StageGraphValidator
from .terminal import (
    EXPORT_MANIFEST_PATH,
    StageFailureRecord,
    TerminalFinalizationResult,
    finalize_failure,
    invalidate_terminal,
    is_item_completed,
)
from .texture_plan import texture_region_input
from .texture_sources import (
    LoadedTextureRegion,
    build_render_texture_regions,
    load_base_texture_regions,
    load_native_texture_regions,
)

AUTO_RIG_PIPELINE_VERSION = "auto-rig-item-runner-v3"
STAGE_A_SCHEMA_VERSION = 2
STAGE_A_ALGORITHM_VERSION = "stage-a-mask-joints-pose-v3"
STAGE_A_OBSERVATIONS_PATH = "rig/cache/A/geometry_observations.json"
STAGE_A_POSE_REPORT_PATH = "rig/cache/A/pose_report.json"
STAGE_B_SCHEMA_VERSION = 1
STAGE_B_ALGORITHM_VERSION = "stage-b-mesh-skinning-v3"
_ACTIVE_STAGE_PATH = "rig/cache/pipeline/active_stage.json"


class AutoRigPipelineError(RuntimeError):
    """Raised when the item runner receives an impossible execution request."""


@dataclass(frozen=True, slots=True)
class AutoRigPipelineResult:
    item_root: Path
    stage_manifests: Mapping[str, StageManifest]
    pose_execution: PoseExecutionResult
    geometry_cache: RigGeometryCache
    stage_c: StageCResult | None
    stage_d: StageDResult | None
    stage_e: StageEResult | None
    terminal: TerminalFinalizationResult | None
    reused_stages: tuple[str, ...] = ()


def _item_root(value: str | Path) -> Path:
    candidate = Path(value).expanduser().absolute()
    if candidate.is_file():
        if candidate.name.casefold() != "final.psd":
            raise AutoRigPipelineError("an auto-rig file input must be the item's final.psd")
        candidate = candidate.parent
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise AutoRigPipelineError(f"item root is unavailable: {candidate}") from exc
    if not root.is_dir():
        raise AutoRigPipelineError(f"item root is not a directory: {root}")
    return root


def _marker_sha256(root: Path, stage_name: str) -> str:
    return sha256_file(root / Path(*manifest_relative_path(stage_name).split("/")))


def _native_input_digests(
    root: Path,
    variant_set: NativeVariantSet,
) -> tuple[FileDigest, ...]:
    paths: set[str] = set()
    if variant_set.manifest_path is not None:
        paths.add(variant_set.manifest_path.relative_to(root).as_posix())
    paths.update(entry.png_path.relative_to(root).as_posix() for entry in variant_set.entries if entry.source_kind == "authored")
    return tuple(describe_file(root, path) for path in sorted(paths))


def _stage_a_output_paths(
    component_plan: MaskComponentPlan,
    variant_set: NativeVariantSet,
) -> tuple[str, ...]:
    paths = {STAGE_A_OBSERVATIONS_PATH, STAGE_A_POSE_REPORT_PATH}
    paths.update(part.qcl_file.path for part in component_plan.parts)
    paths.update(partition.qcl_file.path for partition in component_plan.variant_partitions if partition.qcl_file is not None)
    paths.update(entry.relative_path for entry in variant_set.entries if entry.source_kind == "generated")
    return tuple(sorted(paths))


def _write_stage_a_observations(
    root: Path,
    *,
    target_payload: dict[str, object],
    component_plan: MaskComponentPlan,
    ordinary_draw_payload: dict[str, object],
    native_variant_set: NativeVariantSet,
    generic_variant_synthesis: GenericVariantSynthesisPlan,
    quality_payload: dict[str, object],
    eligibility_payload: dict[str, object],
    final_draw_payload: dict[str, object],
    anatomy: AnatomyMaskGeometry,
    joint_payload: dict[str, object],
    pose_execution_payload: dict[str, object],
) -> None:
    atomic_write_json(
        root / Path(*STAGE_A_OBSERVATIONS_PATH.split("/")),
        {
            "schema_version": "stage-a-geometry-observations-v1",
            "target": target_payload,
            "component_plan": {
                **component_plan.semantic_payload(),
                "plan_sha256": component_plan.plan_sha256,
            },
            "ordinary_draw_order": ordinary_draw_payload,
            "native_variant_set": {
                **native_variant_set.semantic_payload(),
                "native_variant_set_sha256": (native_variant_set.native_variant_set_sha256),
            },
            "generic_variant_synthesis": {
                **generic_variant_synthesis.semantic_payload(),
                "plan_sha256": generic_variant_synthesis.plan_sha256,
            },
            "native_variant_quality": quality_payload,
            "native_variant_eligibility": eligibility_payload,
            "final_draw_order": final_draw_payload,
            "anatomy": {
                **anatomy.plan.semantic_payload(),
                "plan_sha256": anatomy.plan.plan_sha256,
            },
            "joint_plan": joint_payload,
            "pose_execution": pose_execution_payload,
        },
    )


def _native_metadata(
    variant_set: NativeVariantSet,
    render_variant_ids: tuple[str, ...],
    quality_plan,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    admitted = set(render_variant_ids)
    composite_modes = {entry.part_id: entry.composite_mode for entry in variant_set.entries if entry.variant_id in admitted}
    quality_by_id = {result.variant_id: result.to_dict() for result in quality_plan.candidate_results}
    quality = {entry.part_id: quality_by_id[entry.variant_id] for entry in variant_set.entries if entry.variant_id in admitted}
    return composite_modes, quality


def _selected_regions(
    base_regions: tuple[LoadedTextureRegion, ...],
    native_regions: tuple[LoadedTextureRegion, ...],
    render_variant_ids: tuple[str, ...],
) -> tuple[LoadedTextureRegion, ...]:
    admitted = set(render_variant_ids)
    selected = tuple(region for region in native_regions if region.variant_id in admitted)
    return (*base_regions, *selected)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        return _jsonable(namespace)
    slots = getattr(value, "__slots__", ())
    if slots:
        return _jsonable({name: getattr(value, name) for name in slots})
    return str(value)


def _pose_provider_identities(providers: Mapping[str, object] | None) -> dict[str, object]:
    identities: dict[str, object] = {}
    for key, provider in sorted((providers or {}).items()):
        identities[key] = {
            "class": f"{type(provider).__module__}.{type(provider).__qualname__}",
            "runtime_report": _jsonable(getattr(provider, "runtime_report", None)),
        }
    return identities


def _external_file_identity(value: str | Path | None) -> dict[str, object] | None:
    if value is None:
        return None
    requested = Path(value).expanduser().absolute()
    try:
        resolved = requested.resolve(strict=True)
    except OSError:
        return {"path": str(requested), "sha256": None}
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def _resolve_stage_e_runtime(
    root: Path,
    *,
    sdk_root: str | Path | None,
):
    try:
        toolchain = ensure_live2d_runtime_toolchain(sdk_root=sdk_root)
        attestation_payload = load_packaged_live2d_frame_attestation()
        runtime_attestation = select_runtime_attestation(
            attestation_payload,
            platform_id=toolchain.platform_id,
            backend_id=toolchain.backend_id,
            core_sha256=toolchain.core_sha256,
            validator_protocol_digest=LIVE2D_VALIDATOR_PROTOCOL_DIGEST,
        )
        return toolchain, runtime_attestation
    except Live2DRuntimeToolchainError as exc:
        raise publish_stage_e_toolchain_failure(root, exc) from exc
    except (Live2DAttestationError, OSError) as exc:
        toolchain_error = Live2DRuntimeToolchainError(
            "live2d_coordinate_schema_unverified",
            str(exc),
        )
        raise publish_stage_e_toolchain_failure(root, toolchain_error) from exc


def _stage_reusable(
    root: Path,
    *,
    target_stage: str,
    expected_fingerprints: Mapping[str, str],
) -> bool:
    try:
        return (
            StageGraphValidator(root)
            .validate(
                target_stage=target_stage,
                expected_fingerprints=expected_fingerprints,
            )
            .reusable
        )
    except (OSError, ValueError):
        return False


def _stored_stage_outputs_are_intact(root: Path, stage_name: str) -> bool:
    """Check committed bytes before deterministic planners can rematerialize them."""

    try:
        return (
            StageGraphValidator(root)
            .validate(
                target_stage=stage_name,
                expected_fingerprints={},
                required_fingerprint_stages=(),
            )
            .reusable
        )
    except (OSError, ValueError):
        return False


def _invalidate_stage_markers(root: Path, stages: tuple[str, ...]) -> None:
    for stage in stages:
        (root / Path(*manifest_relative_path(stage).split("/"))).unlink(missing_ok=True)
        (root / "rig" / "cache" / stage / "failure.json").unlink(missing_ok=True)
    invalidate_terminal(root)


def _load_pose_execution(root: Path, cache: RigGeometryCache, stage_a: StageManifest) -> PoseExecutionResult:
    try:
        report = json.loads((root / Path(*STAGE_A_POSE_REPORT_PATH.split("/"))).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AutoRigPipelineError("reusable Stage A pose report is unreadable") from exc
    if not isinstance(report, dict):
        raise AutoRigPipelineError("reusable Stage A pose report is not an object")
    selected = report.get("selected_provider_id")
    if selected is not None and not isinstance(selected, str):
        raise AutoRigPipelineError("reusable Stage A pose provider identity is invalid")
    return PoseExecutionResult(
        joint_plan=cache.joint_plan,
        selected_provider_id=selected,
        degraded=stage_a.status == "stage_validated_with_degradation",
        report=report,
    )


def _load_terminal_result(root: Path) -> TerminalFinalizationResult:
    try:
        payload = json.loads((root / Path(*EXPORT_MANIFEST_PATH.split("/"))).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AutoRigPipelineError("completed terminal manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise AutoRigPipelineError("completed terminal manifest is not an object")
    return TerminalFinalizationResult(
        terminal_path=root / Path(*EXPORT_MANIFEST_PATH.split("/")),
        g_manifest=read_stage_manifest(root, "G"),
        payload=payload,
    )


def _export_input_digests(c_manifest: StageManifest) -> tuple[FileDigest, ...]:
    selected = tuple(
        record
        for record in c_manifest.output_file_sha256
        if record.path == "rig/rig.json" or (record.path.startswith("rig/shared/textures/") and record.path.endswith(".png"))
    )
    if not selected or selected[0].path != "rig/rig.json":
        raise AutoRigPipelineError("Stage C manifest lacks exporter input artifacts")
    return selected


def _set_active_stage(root: Path, stage_name: str) -> None:
    atomic_write_json(
        root / Path(*_ACTIVE_STAGE_PATH.split("/")),
        {
            "schema_version": "auto-rig-active-stage-v1",
            "stage_name": stage_name,
        },
    )


def _active_stage(root: Path) -> str:
    try:
        payload = json.loads((root / Path(*_ACTIVE_STAGE_PATH.split("/"))).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "A"
    stage = payload.get("stage_name") if isinstance(payload, dict) else None
    return stage if stage in {"A", "B", "C", "D", "E", "G"} else "A"


def _observed_input_set_sha256(root: Path) -> str:
    paths: set[str] = set()
    for relative in (
        "final.psd",
        "src_img.png",
        "optimized/info.json",
        "run_meta.json",
        "rig_overrides.json",
        "rig_inputs/variants/manifest.json",
    ):
        if (root / Path(*relative.split("/"))).is_file():
            paths.add(relative)
    variants = root / "rig_inputs" / "variants"
    if variants.is_dir():
        paths.update(path.relative_to(root).as_posix() for path in variants.rglob("*") if path.is_file())
    records = tuple(describe_file(root, path).to_dict() for path in sorted(paths))
    return canonical_json_sha256({"observed_inputs": records})


def _load_stage_failure_record(root: Path, stage_name: str, error: Exception) -> StageFailureRecord:
    relative = f"rig/cache/{stage_name}/failure.json"
    path = root / Path(*relative.split("/"))
    payload: dict[str, object] | None = None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    diagnostics = payload.get("diagnostics") if payload is not None else None
    retryable = payload.get("retryable") if payload is not None else None
    if not isinstance(diagnostics, list) or any(not isinstance(item, dict) for item in diagnostics):
        code = getattr(error, "code", "auto_rig_stage_failed")
        diagnostics = [
            {
                "code": str(code),
                "stage": stage_name,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        ]
        retryable = False
        atomic_write_json(
            path,
            {
                "schema_version": "stage-failure-v1",
                "status": "stage_failed",
                "stage_name": stage_name,
                "retryable": retryable,
                "diagnostics": diagnostics,
            },
        )
    return StageFailureRecord(
        stage_name=stage_name,
        record_path=relative,
        diagnostics=tuple(diagnostics),
        retryable=retryable if isinstance(retryable, bool) else False,
    )


def _completed_manifests_before(root: Path, failed_stage: str) -> dict[str, StageManifest]:
    order = ("A", "B", "C", "D", "E")
    completed: dict[str, StageManifest] = {}
    for stage in order[: order.index(failed_stage)]:
        try:
            completed[stage] = read_stage_manifest(root, stage)
        except (OSError, ValueError):
            break
    return completed


def _publish_pipeline_failure(
    root: Path,
    *,
    error: Exception,
    profile_id: str,
    validation_tier: str,
    invocation_config: Mapping[str, object],
) -> None:
    failed_stage = _active_stage(root)
    if failed_stage == "G":
        invalidate_terminal(root)
        return
    (root / Path(*manifest_relative_path(failed_stage).split("/"))).unlink(missing_ok=True)
    record = _load_stage_failure_record(root, failed_stage, error)
    completed = _completed_manifests_before(root, failed_stage)
    latest = completed[next(reversed(completed))] if completed else None
    override_path = root / "rig_overrides.json"
    overrides_sha256 = sha256_file(override_path) if override_path.is_file() else canonical_json_sha256({"rig_overrides": "absent"})
    if latest is not None:
        overrides_sha256 = latest.rig_overrides_sha256
    try:
        profile = load_capability_profile(profile_id)
        profile_sha256 = profile.profile_sha256
    except ValueError:
        profile_sha256 = canonical_json_sha256({"invalid_profile": profile_id})
    finalize_failure(
        root,
        item_id=root.name,
        target_input_fingerprint=(latest.target_input_fingerprint if latest is not None else None),
        observed_input_set_sha256=_observed_input_set_sha256(root),
        native_variant_set_sha256=(latest.native_variant_set_sha256 if latest is not None else None),
        native_variant_eligibility_sha256=(latest.native_variant_eligibility_sha256 if latest is not None else None),
        config_fingerprint=canonical_json_sha256(dict(invocation_config)),
        rig_overrides_sha256=overrides_sha256,
        profile=profile_id,
        profile_fingerprint=profile_sha256,
        validation_tier=validation_tier,
        failure_records=(record,),
        completed_stage_names=tuple(completed),
    )


def _run_auto_rig_item_impl(
    item_root_or_final_psd: str | Path,
    *,
    profile_id: str = "dual_runtime_core_v1",
    validation_tier: str = "release",
    sdk_root: str | Path | None = None,
    spine_runtime_path: str | Path | None = None,
    pose_mode: PoseMode = "auto",
    pose_providers: Mapping[str, object] | None = None,
    pose_provider_pool: PoseProviderPool | None = None,
    pose_model_cache_dir: str | Path | None = None,
    sdpose_bundle_path: str | Path | None = None,
    detrpose_weights_path: str | Path | None = None,
    pose_device: str | None = None,
    prefer_pose_fa2: bool = True,
    finalize: bool = True,
) -> AutoRigPipelineResult:
    """Run one completed see-through item through deterministic stages A-E/G."""

    if validation_tier not in LIVE2D_VALIDATION_TIERS:
        raise AutoRigPipelineError("validation_tier must be either 'structural' or 'release'")
    if finalize and validation_tier != "release":
        raise AutoRigPipelineError("formal finalization requires release-tier Live2D validation")

    root = _item_root(item_root_or_final_psd)
    # Component/QCL and generated-variant planners are deterministic but may
    # recreate missing A-owned bytes. Preserve the pre-planning integrity fact
    # so that repair cannot masquerade as a cache hit.
    stored_a_outputs_intact = _stored_stage_outputs_are_intact(root, "A")
    _set_active_stage(root, "A")
    profile = load_capability_profile(profile_id)
    required_formats = frozenset(profile.required_formats)
    resolved_spine_runtime = None
    spine_runtime_sha256 = None
    if spine_runtime_path is not None:
        try:
            resolved_spine_runtime = Path(spine_runtime_path).resolve(strict=True)
            spine_runtime_sha256 = sha256_file(resolved_spine_runtime)
        except (OSError, ValueError) as exc:
            raise AutoRigPipelineError(f"Spine Runtime validator is unavailable: {spine_runtime_path}") from exc

    override_source = load_rig_override_source(root)
    contract = load_auto_rig_input_contract(
        root,
        tag_aliases=dict(override_source.tag_aliases),
    )
    target = build_target_input_identity(contract)
    overrides = validate_rig_override_source(override_source, target)

    loaded_alphas = load_validated_part_alphas(contract)
    base_component_plan = build_base_mask_component_plan(contract)
    ordinary_draw = build_ordinary_draw_order(base_component_plan.parts)
    authored_variant_set = load_native_variant_set(
        contract,
        base_component_plan,
        ordinary_draw,
    )
    source_base_regions = load_base_texture_regions(contract)
    native_variant_set, generic_variant_synthesis = build_generic_variant_synthesis_plan(
        contract,
        base_component_plan,
        ordinary_draw,
        source_base_regions,
        authored_variant_set,
        item_root=root,
    )
    component_plan = extend_mask_component_plan_with_variants(
        base_component_plan,
        native_variant_set,
        root,
    )
    quality = build_native_variant_quality_plan(
        loaded_alphas,
        component_plan,
        ordinary_draw,
        native_variant_set,
        item_root=root,
    )
    source_native_regions = load_native_texture_regions(native_variant_set)
    base_regions, native_regions = build_render_texture_regions(
        component_plan,
        item_root=root,
        base_regions=source_base_regions,
        native_regions=source_native_regions,
    )
    eligibility = admit_native_variant_resources(
        tuple(texture_region_input(region) for region in base_regions),
        tuple(texture_region_input(region) for region in native_regions),
        component_plan,
        quality,
    )
    final_draw = expand_draw_order_with_admitted_variants(
        ordinary_draw,
        native_variant_set,
        eligibility,
    )

    loaded_geometry = load_component_geometry(component_plan, item_root=root)
    anatomy = build_anatomy_mask_geometry(
        loaded_geometry,
        canvas_edge=contract.canvas.resolution,
    )
    geometry_joints = build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
    )

    a_inputs = [*target.input_files, *_native_input_digests(root, authored_variant_set)]
    if override_source.identity.present:
        a_inputs.append(describe_file(root, "rig_overrides.json"))
    a_inputs = sorted({record.path: record for record in a_inputs}.values(), key=lambda item: item.path)
    a_config = canonical_json_sha256(
        {
            "pipeline_version": AUTO_RIG_PIPELINE_VERSION,
            "pose_mode": pose_mode,
            "pose_device": pose_device,
            "prefer_pose_fa2": prefer_pose_fa2,
            "pose_providers": _pose_provider_identities(pose_providers),
            "sdpose_bundle": _external_file_identity(sdpose_bundle_path),
            "detrpose_weights": _external_file_identity(detrpose_weights_path),
        }
    )
    a_fingerprint = build_stage_fingerprint(
        stage_name="A",
        stage_schema_version=STAGE_A_SCHEMA_VERSION,
        algorithm_version=STAGE_A_ALGORITHM_VERSION,
        upstream_manifests={},
        input_file_sha256=a_inputs,
        target_input_fingerprint=target.target_input_fingerprint,
        native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
        native_variant_eligibility_sha256=eligibility.plan_sha256,
        relevant_config_fingerprint=a_config,
        rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
        status="stage_validated",
    )
    b_config = canonical_json_sha256(
        {
            "pipeline_version": AUTO_RIG_PIPELINE_VERSION,
            "stage_b_algorithm": STAGE_B_ALGORITHM_VERSION,
        }
    )
    reused: list[str] = []
    stage_a: StageManifest
    stage_b: StageManifest
    pose_execution: PoseExecutionResult
    cache: RigGeometryCache
    reuse_ab = stored_a_outputs_intact and _stage_reusable(
        root,
        target_stage="A",
        expected_fingerprints={"A": a_fingerprint},
    )
    b_fingerprint = ""
    if reuse_ab:
        b_upstream = {"A": _marker_sha256(root, "A")}
        b_fingerprint = build_stage_fingerprint(
            stage_name="B",
            stage_schema_version=STAGE_B_SCHEMA_VERSION,
            algorithm_version=STAGE_B_ALGORITHM_VERSION,
            upstream_manifests=b_upstream,
            input_file_sha256=(),
            target_input_fingerprint=target.target_input_fingerprint,
            native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
            native_variant_eligibility_sha256=eligibility.plan_sha256,
            relevant_config_fingerprint=b_config,
            rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
            status="stage_validated",
        )
        reuse_ab = _stage_reusable(
            root,
            target_stage="B",
            expected_fingerprints={"A": a_fingerprint, "B": b_fingerprint},
        )
    if reuse_ab:
        try:
            stage_a = read_stage_manifest(root, "A")
            stage_b = read_stage_manifest(root, "B")
            cache = load_rig_geometry_cache(root, target=target)
            if (
                cache.component_plan_sha256 != component_plan.plan_sha256
                or cache.native_variant_set_sha256 != native_variant_set.native_variant_set_sha256
                or cache.native_variant_eligibility_sha256 != eligibility.plan_sha256
                or cache.final_draw_order.plan_sha256 != final_draw.plan_sha256
                or cache.stage_a_fingerprint != a_fingerprint
                or cache.stage_b_fingerprint != b_fingerprint
            ):
                raise AutoRigPipelineError("reusable Stage B cache differs from current Stage A plans")
            pose_execution = _load_pose_execution(root, cache, stage_a)
            reused.extend(("A", "B"))
        except (OSError, ValueError, AutoRigPipelineError):
            reuse_ab = False

    if not reuse_ab:
        _invalidate_stage_markers(root, ("A", "B", "C", "D", "E"))
        from PIL import Image

        with Image.open(root / "src_img.png") as source_image:
            pose_image = source_image.copy()
        owned_pose_pool = None
        resolver = pose_provider_pool
        if resolver is None:
            owned_pose_pool = make_builtin_pose_resolver(
                model_cache_dir=pose_model_cache_dir,
                sdpose_bundle_path=sdpose_bundle_path,
                detrpose_weights_path=detrpose_weights_path,
                device=pose_device,
                prefer_fa2=prefer_pose_fa2,
            )
            resolver = owned_pose_pool
        try:
            pose_execution = execute_pose_observation(
                mode=pose_mode,
                anatomy=anatomy,
                geometry_plan=geometry_joints,
                target=target,
                overrides=overrides,
                image=pose_image,
                providers=pose_providers,
                provider_resolver=resolver,
            )
        finally:
            if owned_pose_pool is not None:
                owned_pose_pool.close()
        joints = pose_execution.joint_plan
        atomic_write_json(root / Path(*STAGE_A_POSE_REPORT_PATH.split("/")), pose_execution.report)
        a_status = "stage_validated_with_degradation" if pose_execution.degraded else "stage_validated"
        _write_stage_a_observations(
            root,
            target_payload={**target.semantic_payload(), "target_input_fingerprint": target.target_input_fingerprint},
            component_plan=component_plan,
            ordinary_draw_payload={**ordinary_draw.semantic_payload(), "plan_sha256": ordinary_draw.plan_sha256},
            native_variant_set=native_variant_set,
            generic_variant_synthesis=generic_variant_synthesis,
            quality_payload={**quality.semantic_payload(), "plan_sha256": quality.plan_sha256},
            eligibility_payload={**eligibility.semantic_payload(), "plan_sha256": eligibility.plan_sha256},
            final_draw_payload={**final_draw.semantic_payload(), "plan_sha256": final_draw.plan_sha256},
            anatomy=anatomy,
            joint_payload={**joints.semantic_payload(), "plan_sha256": joints.plan_sha256},
            pose_execution_payload=pose_execution.report,
        )
        stage_a = build_stage_manifest(
            root,
            stage_name="A",
            stage_schema_version=STAGE_A_SCHEMA_VERSION,
            algorithm_version=STAGE_A_ALGORITHM_VERSION,
            upstream_manifests={},
            input_file_sha256=a_inputs,
            target_input_fingerprint=target.target_input_fingerprint,
            native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
            native_variant_eligibility_sha256=eligibility.plan_sha256,
            relevant_config_fingerprint=a_config,
            rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
            output_paths=_stage_a_output_paths(component_plan, native_variant_set),
            status=a_status,
        )
        if stage_a.stage_fingerprint != a_fingerprint:
            raise AutoRigPipelineError("Stage A precomputed fingerprint changed at commit")
        write_stage_manifest(root, stage_a)

        _set_active_stage(root, "B")
        bones = build_bone_graph(joints)
        mesh_plan = build_mesh_plan(
            component_plan,
            item_root=root,
            render_variant_ids=eligibility.render_variant_ids,
            final_part_ids=final_draw.part_order,
        )
        component_sources = load_mesh_component_sources(
            component_plan,
            item_root=root,
            render_variant_ids=eligibility.render_variant_ids,
        )
        skinning = build_skinning_plan(
            mesh_plan,
            bones,
            joints,
            normalized_parts=component_plan.parts,
            component_sources=component_sources,
            native_variant_set=native_variant_set,
        )
        b_upstream = {"A": _marker_sha256(root, "A")}
        b_fingerprint = build_stage_fingerprint(
            stage_name="B",
            stage_schema_version=STAGE_B_SCHEMA_VERSION,
            algorithm_version=STAGE_B_ALGORITHM_VERSION,
            upstream_manifests=b_upstream,
            input_file_sha256=(),
            target_input_fingerprint=target.target_input_fingerprint,
            native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
            native_variant_eligibility_sha256=eligibility.plan_sha256,
            relevant_config_fingerprint=b_config,
            rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
            status="stage_validated",
        )
        cache = build_rig_geometry_cache(
            target=target,
            component_plan=component_plan,
            final_draw=final_draw,
            joints=joints,
            bone_graph=bones,
            mesh_plan=mesh_plan,
            skinning_plan=skinning,
            native_variant_set=native_variant_set,
            stage_a_fingerprint=a_fingerprint,
            stage_b_fingerprint=b_fingerprint,
        )
        cache_path = root / Path(*RIG_GEOMETRY_CACHE_PATH.split("/"))
        cache_bytes = rig_geometry_cache_bytes(cache)
        atomic_write_bytes(cache_path, cache_bytes)
        validate_rig_geometry_cache_payload(json.loads(cache_bytes))
        stage_b = build_stage_manifest(
            root,
            stage_name="B",
            stage_schema_version=STAGE_B_SCHEMA_VERSION,
            algorithm_version=STAGE_B_ALGORITHM_VERSION,
            upstream_manifests=b_upstream,
            input_file_sha256=(),
            target_input_fingerprint=target.target_input_fingerprint,
            native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
            native_variant_eligibility_sha256=eligibility.plan_sha256,
            relevant_config_fingerprint=b_config,
            rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
            output_paths=(RIG_GEOMETRY_CACHE_PATH,),
            status=("stage_validated_with_degradation" if cache.degradation_state == "degraded" else "stage_validated"),
        )
        if stage_b.stage_fingerprint != b_fingerprint:
            raise AutoRigPipelineError("Stage B precomputed fingerprint changed at commit")
        write_stage_manifest(root, stage_b)

    _set_active_stage(root, "C")
    controls = build_control_registry_plan()
    presets = build_preset_library_plan(controls)
    capabilities = derive_capabilities(
        cache,
        anatomy.plan,
        presets,
        native_variant_set=native_variant_set,
    )
    bindings = build_control_binding_plan(
        cache,
        anatomy.plan,
        controls,
        presets,
        capabilities,
        native_variant_set=native_variant_set,
    )
    selected_regions = _selected_regions(
        base_regions,
        native_regions,
        eligibility.render_variant_ids,
    )
    composite_modes, native_quality = _native_metadata(
        native_variant_set,
        eligibility.render_variant_ids,
        quality,
    )
    c_config = canonical_json_sha256(
        {
            "pipeline_version": AUTO_RIG_PIPELINE_VERSION,
            "profile_id": profile_id,
            "control_registry_sha256": controls.plan_sha256,
            "preset_library_sha256": presets.plan_sha256,
        }
    )
    c_upstream = {"A": _marker_sha256(root, "A"), "B": _marker_sha256(root, "B")}
    c_fingerprint = build_stage_fingerprint(
        stage_name="C",
        stage_schema_version=STAGE_C_SCHEMA_VERSION,
        algorithm_version=STAGE_C_ALGORITHM_VERSION,
        upstream_manifests=c_upstream,
        input_file_sha256=(),
        target_input_fingerprint=target.target_input_fingerprint,
        native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
        native_variant_eligibility_sha256=eligibility.plan_sha256,
        relevant_config_fingerprint=c_config,
        rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
        status="stage_validated",
    )
    expected_fingerprints = {"A": a_fingerprint, "B": b_fingerprint, "C": c_fingerprint}
    stage_c: StageCResult | None = None
    if _stage_reusable(root, target_stage="C", expected_fingerprints=expected_fingerprints):
        c_manifest = read_stage_manifest(root, "C")
        reused.append("C")
    else:
        _invalidate_stage_markers(root, ("C", "D", "E"))
        stage_c = execute_stage_c(
            root,
            cache=cache,
            controls=controls,
            presets=presets,
            capabilities=capabilities,
            bindings=bindings,
            loaded_regions=selected_regions,
            expected_texture_plan=eligibility.final_texture_plan,
            profile_id=profile_id,
            upstream_manifests=c_upstream,
            relevant_config_fingerprint=c_config,
            native_composite_mode_by_part=composite_modes,
            native_quality_by_part=native_quality,
        )
        c_manifest = stage_c.manifest
    c_marker = _marker_sha256(root, "C")
    exporter_inputs = _export_input_digests(c_manifest)
    d_config = canonical_json_sha256(
        {
            "pipeline_version": AUTO_RIG_PIPELINE_VERSION,
            "profile_id": profile_id,
            "exporter": "spine_4_2",
            "official_runtime_executable_sha256": spine_runtime_sha256,
        }
    )
    d_fingerprint = build_stage_fingerprint(
        stage_name="D",
        stage_schema_version=STAGE_D_SCHEMA_VERSION,
        algorithm_version=STAGE_D_ALGORITHM_VERSION,
        upstream_manifests={"C": c_marker},
        input_file_sha256=exporter_inputs,
        target_input_fingerprint=target.target_input_fingerprint,
        native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
        native_variant_eligibility_sha256=eligibility.plan_sha256,
        relevant_config_fingerprint=d_config,
        rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
        status="stage_validated",
    )
    expected_fingerprints["D"] = d_fingerprint
    stage_d: StageDResult | None = None
    _set_active_stage(root, "D")
    if _stage_reusable(root, target_stage="D", expected_fingerprints=expected_fingerprints):
        d_manifest = read_stage_manifest(root, "D")
        reused.append("D")
    else:
        _invalidate_stage_markers(root, ("D",))
        stage_d = execute_stage_d(
            root,
            upstream_manifests={"C": c_marker},
            relevant_config_fingerprint=d_config,
            spine_runtime_path=resolved_spine_runtime,
        )
        d_manifest = stage_d.manifest

    stage_e: StageEResult | None = None
    if "live2d_moc3_v4_00" in required_formats:
        _set_active_stage(root, "E")
        runtime_toolchain = None
        runtime_attestation = None
        if validation_tier == "release":
            runtime_toolchain, runtime_attestation = _resolve_stage_e_runtime(
                root,
                sdk_root=sdk_root,
            )
            runtime_identity: object = runtime_toolchain.fingerprint_payload(
                attestation_record_sha256=runtime_attestation.record_sha256,
            )
        else:
            runtime_identity = "not-required"
        e_config = canonical_json_sha256(
            {
                "pipeline_version": AUTO_RIG_PIPELINE_VERSION,
                "profile_id": profile_id,
                "exporter": "live2d_moc3_v4_00",
                "validation_tier": validation_tier,
                "runtime_toolchain": runtime_identity,
            }
        )
        e_fingerprint = build_stage_fingerprint(
            stage_name="E",
            stage_schema_version=STAGE_E_SCHEMA_VERSION,
            algorithm_version=STAGE_E_ALGORITHM_VERSION,
            upstream_manifests={"C": c_marker},
            input_file_sha256=exporter_inputs,
            target_input_fingerprint=target.target_input_fingerprint,
            native_variant_set_sha256=native_variant_set.native_variant_set_sha256,
            native_variant_eligibility_sha256=eligibility.plan_sha256,
            relevant_config_fingerprint=e_config,
            rig_overrides_sha256=overrides.identity.rig_overrides_sha256,
            status="stage_validated",
        )
        expected_fingerprints["E"] = e_fingerprint
        if _stage_reusable(
            root,
            target_stage="E",
            expected_fingerprints=expected_fingerprints,
        ):
            e_manifest = read_stage_manifest(root, "E")
            reused.append("E")
        else:
            _invalidate_stage_markers(root, ("E",))
            stage_e = execute_stage_e(
                root,
                upstream_manifests={"C": c_marker},
                relevant_config_fingerprint=e_config,
                validation_tier=validation_tier,
                runtime_toolchain=runtime_toolchain,
                runtime_attestation=runtime_attestation,
            )
            e_manifest = stage_e.manifest
    else:
        clear_stage_public_outputs(root, "E")
        (root / "rig" / "cache" / "E" / "failure.json").unlink(missing_ok=True)
        invalidate_terminal(root)
    manifests: dict[str, StageManifest] = {
        "A": stage_a,
        "B": stage_b,
        "C": c_manifest,
        "D": d_manifest,
    }
    if "live2d_moc3_v4_00" in required_formats:
        manifests["E"] = e_manifest
    terminal = None
    if finalize:
        _set_active_stage(root, "G")
        if is_item_completed(root, expected_stage_fingerprints=expected_fingerprints):
            terminal = _load_terminal_result(root)
            reused.append("G")
        else:
            invalidate_terminal(root)
            terminal = execute_stage_g_success(
                root,
                config_fingerprint=canonical_json_sha256(
                    {
                        "pipeline_version": AUTO_RIG_PIPELINE_VERSION,
                        "profile_id": profile_id,
                        "validation_tier": validation_tier,
                    }
                ),
                expected_stage_fingerprints=expected_fingerprints,
            )
        manifests["G"] = terminal.g_manifest

    return AutoRigPipelineResult(
        item_root=root,
        stage_manifests=manifests,
        pose_execution=pose_execution,
        geometry_cache=cache,
        stage_c=stage_c,
        stage_d=stage_d,
        stage_e=stage_e,
        terminal=terminal,
        reused_stages=tuple(reused),
    )


def run_auto_rig_item(
    item_root_or_final_psd: str | Path,
    *,
    profile_id: str = "dual_runtime_core_v1",
    validation_tier: str = "release",
    sdk_root: str | Path | None = None,
    spine_runtime_path: str | Path | None = None,
    pose_mode: PoseMode = "auto",
    pose_providers: Mapping[str, object] | None = None,
    pose_provider_pool: PoseProviderPool | None = None,
    pose_model_cache_dir: str | Path | None = None,
    sdpose_bundle_path: str | Path | None = None,
    detrpose_weights_path: str | Path | None = None,
    pose_device: str | None = None,
    prefer_pose_fa2: bool = True,
    finalize: bool = True,
) -> AutoRigPipelineResult:
    """Run one item; formal success and every A-E failure get an authoritative terminal."""

    # Invocation and registry errors are process configuration failures, not
    # failures of the item at stage A. Validate them before touching item state.
    if validation_tier not in LIVE2D_VALIDATION_TIERS:
        raise AutoRigPipelineError("validation_tier must be either 'structural' or 'release'")
    if finalize and validation_tier != "release":
        raise AutoRigPipelineError("formal finalization requires release-tier Live2D validation")
    profile = load_capability_profile(profile_id)
    if finalize and not profile.terminal_delivery:
        raise AutoRigPipelineError("formal finalization requires a terminal dual-runtime capability profile")

    root = _item_root(item_root_or_final_psd)
    invocation_config = {
        "pipeline_version": AUTO_RIG_PIPELINE_VERSION,
        "profile_id": profile_id,
        "validation_tier": validation_tier,
        "sdk_root": str(sdk_root) if sdk_root is not None else None,
        "spine_runtime_path": (str(spine_runtime_path) if spine_runtime_path is not None else None),
        "pose_mode": pose_mode,
        "pose_device": pose_device,
        "prefer_pose_fa2": prefer_pose_fa2,
        "pose_providers": _pose_provider_identities(pose_providers),
        "sdpose_bundle_path": (str(sdpose_bundle_path) if sdpose_bundle_path is not None else None),
        "detrpose_weights_path": (str(detrpose_weights_path) if detrpose_weights_path is not None else None),
        "finalize": finalize,
    }
    try:
        result = _run_auto_rig_item_impl(
            root,
            profile_id=profile_id,
            validation_tier=validation_tier,
            sdk_root=sdk_root,
            spine_runtime_path=spine_runtime_path,
            pose_mode=pose_mode,
            pose_providers=pose_providers,
            pose_provider_pool=pose_provider_pool,
            pose_model_cache_dir=pose_model_cache_dir,
            sdpose_bundle_path=sdpose_bundle_path,
            detrpose_weights_path=detrpose_weights_path,
            pose_device=pose_device,
            prefer_pose_fa2=prefer_pose_fa2,
            finalize=finalize,
        )
    except Exception as exc:
        try:
            _publish_pipeline_failure(
                root,
                error=exc,
                profile_id=profile_id,
                validation_tier=validation_tier,
                invocation_config=invocation_config,
            )
        except Exception as terminal_exc:
            (root / Path(*_ACTIVE_STAGE_PATH.split("/"))).unlink(missing_ok=True)
            raise AutoRigPipelineError(
                f"item execution failed and its terminal failure record could not be published: {terminal_exc}"
            ) from exc
        (root / Path(*_ACTIVE_STAGE_PATH.split("/"))).unlink(missing_ok=True)
        raise
    (root / Path(*_ACTIVE_STAGE_PATH.split("/"))).unlink(missing_ok=True)
    return result


__all__ = [
    "AUTO_RIG_PIPELINE_VERSION",
    "STAGE_A_POSE_REPORT_PATH",
    "AutoRigPipelineError",
    "AutoRigPipelineResult",
    "run_auto_rig_item",
]
