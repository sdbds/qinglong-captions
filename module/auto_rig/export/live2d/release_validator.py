from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from ...jcs import jcs_sha256
from ...rig_document import RigDocument
from .animations import (
    Live2DAnimationAsset,
    Live2DAnimationPlan,
    encode_live2d_animation_asset,
)
from .artmesh import Live2DArtMeshPlan
from .attestation import load_live2d_frame_attestation
from .binding_plan import Live2DBindingPlan
from .coordinates import Live2DCoordinatePlan
from .cubism_core import (
    CubismCoreError,
    CubismModelState,
    exercise_moc_with_core,
    probe_cubism_core,
)
from .cubism_renderer import (
    LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST,
    CubismRendererError,
    render_moc_with_offscreen_harness,
)
from .keyforms import Live2DKeyformPlan
from .moc3_codec import decode_moc3_v400
from .runtime_assets import (
    Live2DRuntimeAssetPlan,
    encode_live2d_runtime_asset,
)
from .validator import (
    Live2DMoc3State,
    Live2DStructureValidationReport,
    evaluate_live2d_moc3_state,
)

LIVE2D_RELEASE_VALIDATOR_VERSION = "live2d-release-validator-v1"
LIVE2D_RELEASE_REPORT_VERSION = "live2d-release-report-v1"
LIVE2D_RENDER_WIDTH = 512
LIVE2D_RENDER_HEIGHT = 512


class Live2DReleaseGateError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(f"live2d_release_gate_failed: {message}")


def _error(message: str) -> Live2DReleaseGateError:
    return Live2DReleaseGateError(message)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _resolve_file(path: str | Path, *, field: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _error(f"{field} is unavailable") from exc
    if not resolved.is_file():
        raise _error(f"{field} is unavailable")
    return resolved


def _bundle_file(root: Path, relative_path: str) -> Path:
    if not relative_path or "\\" in relative_path:
        raise _error("runtime bundle path is not canonical")
    candidate = (root / Path(*relative_path.split("/"))).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise _error("runtime bundle path escapes its root") from exc
    if not candidate.is_file():
        raise _error(f"runtime bundle file is missing: {relative_path}")
    return candidate


def _attested_core(core_path: Path):
    try:
        probe = probe_cubism_core(core_path)
    except (OSError, CubismCoreError) as exc:
        raise _error("Cubism Core release gate is unavailable") from exc
    attestation_path = (
        Path(__file__).with_name("attestations") / "live2d-frames-v1.json"
    )
    payload = load_live2d_frame_attestation(attestation_path.read_bytes())
    descriptor = payload.get("contract_descriptor")
    if not isinstance(descriptor, Mapping):
        raise _error("packaged Live2D attestation is malformed")
    approved = descriptor.get("approved_core_binaries")
    if not isinstance(approved, list):
        raise _error("packaged Live2D attestation has no Core allowlist")
    if (
        descriptor.get("e0_validator_protocol_digest")
        != LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST
    ):
        raise _error("packaged Live2D renderer protocol is not attested")
    expected_digest = f"sha256:{probe.sha256}"
    matches = [
        record
        for record in approved
        if isinstance(record, Mapping)
        and record.get("platform") == "windows"
        and record.get("arch") == "x86_64"
        and record.get("core_sha256") == expected_digest
        and record.get("core_version") == probe.version.label
    ]
    if len(matches) != 1 or not probe.capabilities.attestation_candidate:
        raise _error("configured Cubism Core is not attested for formal release")
    return probe


@dataclass(frozen=True, slots=True)
class Live2DParameterReleaseEvidence:
    parameter_id: str
    tested_value: float
    target_export_names: tuple[str, ...]
    visible_change: bool
    core_to_structural_maximum_residual: float
    core_vertex_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "tested_value": self.tested_value,
            "target_export_names": list(self.target_export_names),
            "visible_change": self.visible_change,
            "core_to_structural_maximum_residual": (
                self.core_to_structural_maximum_residual
            ),
            "core_vertex_sha256": self.core_vertex_sha256,
        }


@dataclass(frozen=True, slots=True)
class Live2DMotionReleaseEvidence:
    preset_id: str
    sample_times: tuple[float, ...]
    rgba_sha256: tuple[str, ...]
    observed_parameter_maximum_residual: float
    nonzero_alpha: bool
    visible_change: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "sample_times": list(self.sample_times),
            "rgba_sha256": list(self.rgba_sha256),
            "observed_parameter_maximum_residual": (
                self.observed_parameter_maximum_residual
            ),
            "nonzero_alpha": self.nonzero_alpha,
            "visible_change": self.visible_change,
        }


@dataclass(frozen=True, slots=True)
class Live2DExpressionReleaseEvidence:
    preset_id: str
    rgba_sha256: str
    observed_parameter_maximum_residual: float
    nonzero_alpha: bool
    visible_change: bool
    restored_after_clear: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "rgba_sha256": self.rgba_sha256,
            "observed_parameter_maximum_residual": (
                self.observed_parameter_maximum_residual
            ),
            "nonzero_alpha": self.nonzero_alpha,
            "visible_change": self.visible_change,
            "restored_after_clear": self.restored_after_clear,
        }


@dataclass(frozen=True, slots=True)
class Live2DReleaseValidationReport:
    schema_version: str
    validator_version: str
    structure_report_sha256: str
    core_sha256: str
    core_version: str
    renderer_sha256: str
    renderer_protocol_digest: str
    moc_sha256: str
    core_consistency: bool
    default_rest_maximum_residual: float
    baseline_rgba_sha256: str
    parameter_evidence: tuple[Live2DParameterReleaseEvidence, ...]
    motion_evidence: tuple[Live2DMotionReleaseEvidence, ...]
    expression_evidence: tuple[Live2DExpressionReleaseEvidence, ...]
    report_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "validator_version": self.validator_version,
            "structure_report_sha256": self.structure_report_sha256,
            "core_sha256": self.core_sha256,
            "core_version": self.core_version,
            "renderer_sha256": self.renderer_sha256,
            "renderer_protocol_digest": self.renderer_protocol_digest,
            "moc_sha256": self.moc_sha256,
            "core_consistency": self.core_consistency,
            "default_rest_maximum_residual": (
                self.default_rest_maximum_residual
            ),
            "baseline_rgba_sha256": self.baseline_rgba_sha256,
            "parameter_evidence": [
                record.to_dict() for record in self.parameter_evidence
            ],
            "motion_evidence": [
                record.to_dict() for record in self.motion_evidence
            ],
            "expression_evidence": [
                record.to_dict() for record in self.expression_evidence
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "report_sha256": self.report_sha256}


def _state(result, *, field: str) -> CubismModelState:
    if result.model_state is None:
        raise _error(f"Cubism Core omitted model state for {field}")
    return result.model_state


def _core_point_to_canvas(
    point: tuple[float, float], coordinates: Live2DCoordinatePlan
) -> tuple[float, float]:
    return (
        point[0] * coordinates.ppu + coordinates.width / 2.0,
        coordinates.height / 2.0 - point[1] * coordinates.ppu,
    )


def _state_residual(
    core_state: CubismModelState,
    structural: Live2DMoc3State,
    coordinates: Live2DCoordinatePlan,
) -> float:
    expected = {record.artmesh_id: record for record in structural.artmeshes}
    if set(expected) != {drawable.id for drawable in core_state.drawables}:
        raise _error("Core drawable IDs differ from the structural document")
    residual = 0.0
    for drawable in core_state.drawables:
        target = expected[drawable.id]
        if len(drawable.vertex_positions) != len(target.canvas_positions):
            raise _error("Core drawable topology differs from the structural document")
        for actual, wanted in zip(
            drawable.vertex_positions, target.canvas_positions, strict=True
        ):
            canvas = _core_point_to_canvas(actual, coordinates)
            residual = max(
                residual,
                math.hypot(canvas[0] - wanted[0], canvas[1] - wanted[1]),
            )
        if abs(drawable.opacity - target.opacity) > 1 / 255:
            raise _error(
                "Core drawable opacity differs from structural evaluation: "
                f"{drawable.id} core={drawable.opacity:g} structural={target.opacity:g}"
            )
        if drawable.draw_order != round(target.draw_order):
            raise _error("Core drawable order differs from structural evaluation")
    return residual


def _visible_state_change(
    baseline: CubismModelState, changed: CubismModelState
) -> bool:
    before = {drawable.id: drawable for drawable in baseline.drawables}
    after = {drawable.id: drawable for drawable in changed.drawables}
    if set(before) != set(after):
        raise _error("Core drawable identity changed after setting a parameter")
    for identity in before:
        left = before[identity]
        right = after[identity]
        if abs(left.opacity - right.opacity) > 1 / 255:
            return True
        if any(
            math.hypot(a[0] - b[0], a[1] - b[1]) > 0.1 / 8192.0
            for a, b in zip(
                left.vertex_positions, right.vertex_positions, strict=True
            )
        ):
            return True
    return False


def _motion_curve_value(
    asset: Live2DAnimationAsset, parameter_id: str, time: float
) -> float:
    curves = asset.payload.get("Curves")
    if not isinstance(curves, list):
        raise _error("motion asset has no curve list")
    matches = [
        curve
        for curve in curves
        if isinstance(curve, Mapping) and curve.get("Id") == parameter_id
    ]
    if len(matches) != 1:
        raise _error("motion parameter curve is missing or duplicated")
    segments = matches[0].get("Segments")
    if not isinstance(segments, list) or len(segments) < 5:
        raise _error("motion parameter segments are invalid")
    points = [(float(segments[0]), float(segments[1]))]
    for index in range(2, len(segments), 3):
        if segments[index] != 0:
            raise _error("release validator supports only linear motion segments")
        points.append((float(segments[index + 1]), float(segments[index + 2])))
    duration = float(asset.payload["Meta"]["Duration"])
    loop = bool(asset.payload["Meta"]["Loop"])
    evaluation_time = time
    if loop and time > duration:
        evaluation_time = time % duration
    for point_time, value in points:
        if abs(evaluation_time - point_time) <= 1e-7:
            return value
    for (left_time, left), (right_time, right) in zip(points, points[1:]):
        if left_time < evaluation_time < right_time:
            ratio = (evaluation_time - left_time) / (right_time - left_time)
            return left + (right - left) * ratio
    if abs(evaluation_time - duration) <= 1e-7:
        return points[-1][1]
    raise _error("motion sample time lies outside its curve")


def _motion_sample_times(
    asset: Live2DAnimationAsset,
    *,
    defaults: Mapping[str, float],
) -> tuple[float, ...]:
    meta = asset.payload.get("Meta")
    curves = asset.payload.get("Curves")
    if not isinstance(meta, Mapping) or not isinstance(curves, list):
        raise _error("motion has no canonical sampling data")
    duration = float(meta["Duration"])
    candidate_times: set[float] = set()
    for curve in curves:
        if not isinstance(curve, Mapping):
            raise _error("motion curve is invalid")
        segments = curve.get("Segments")
        if not isinstance(segments, list) or len(segments) < 5:
            raise _error("motion parameter segments are invalid")
        candidate_times.add(float(segments[0]))
        candidate_times.update(
            float(segments[index + 1]) for index in range(2, len(segments), 3)
        )
    effect_time = max(
        sorted(candidate_times),
        key=lambda time: (
            sum(
                abs(
                    _motion_curve_value(asset, parameter_id, time)
                    - defaults[parameter_id]
                )
                for parameter_id in asset.parameter_ids
            ),
            -time,
        ),
    )
    return tuple(dict.fromkeys((duration / 2.0, duration, effect_time)))


def _verify_bundle(
    root: Path,
    animations: Live2DAnimationPlan,
    runtime_assets: Live2DRuntimeAssetPlan,
    structure_report: Live2DStructureValidationReport,
) -> tuple[Path, tuple[Path, ...]]:
    moc_path = _bundle_file(root, "model.moc3")
    if _file_sha256(moc_path) != structure_report.moc_sha256:
        raise _error("MOC3 digest differs from the structural validation report")
    for asset in (runtime_assets.model3, runtime_assets.cdi3):
        path = _bundle_file(root, asset.relative_path)
        if path.read_bytes() != encode_live2d_runtime_asset(asset):
            raise _error("model3/cdi3 bytes differ from their runtime plan")
    for asset in (*animations.motion_assets, *animations.expression_assets):
        path = _bundle_file(root, asset.relative_path)
        if path.read_bytes() != encode_live2d_animation_asset(asset):
            raise _error("motion/expression bytes differ from their runtime plan")
    texture_paths = tuple(
        _bundle_file(root, relative_path)
        for relative_path in runtime_assets.referenced_texture_paths
    )
    return moc_path, texture_paths


def validate_live2d_release_bundle(
    bundle_root: str | Path,
    *,
    core_path: str | Path,
    renderer_path: str | Path,
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
    keyforms: Live2DKeyformPlan,
    animations: Live2DAnimationPlan,
    runtime_assets: Live2DRuntimeAssetPlan,
    structure_report: Live2DStructureValidationReport,
) -> Live2DReleaseValidationReport:
    if (
        structure_report.rig_document_sha256 != rig.document_sha256
        or structure_report.binding_plan_sha256 != bindings.plan_sha256
        or structure_report.coordinate_plan_sha256 != coordinates.plan_sha256
        or structure_report.artmesh_plan_sha256 != artmeshes.plan_sha256
        or structure_report.keyform_plan_sha256 != keyforms.plan_sha256
        or animations.rig_document_sha256 != rig.document_sha256
        or runtime_assets.rig_document_sha256 != rig.document_sha256
        or runtime_assets.animation_plan_sha256 != animations.plan_sha256
    ):
        raise _error("release plans and structural report do not share one input graph")
    try:
        root = Path(bundle_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _error("Live2D release bundle is unavailable") from exc
    if not root.is_dir():
        raise _error("Live2D release bundle is unavailable")
    core_file = _resolve_file(core_path, field="Cubism Core")
    probe = _attested_core(core_file)
    renderer_file = _resolve_file(renderer_path, field="official SDK renderer")
    moc_path, texture_paths = _verify_bundle(
        root, animations, runtime_assets, structure_report
    )
    moc_payload = moc_path.read_bytes()
    document = decode_moc3_v400(moc_payload)
    try:
        baseline_result = exercise_moc_with_core(
            core_file, moc_path, capture_model_state=True
        )
    except CubismCoreError as exc:
        raise _error("Cubism Core rejected the formal MOC3") from exc
    if baseline_result.consistency is not True or not baseline_result.finite_vertices:
        raise _error("Cubism Core consistency/model update gate failed")
    if (
        baseline_result.parameter_count != len(bindings.parameters)
        or baseline_result.part_count != len(artmeshes.parts)
        or baseline_result.drawable_count != len(artmeshes.artmeshes)
        or baseline_result.nonzero_drawable_count <= 0
    ):
        raise _error("Cubism Core model counts differ from the structural plans")
    baseline_state = _state(baseline_result, field="default state")
    if {parameter.id for parameter in baseline_state.parameters} != {
        parameter.export_name for parameter in bindings.parameters
    }:
        raise _error("Cubism Core parameter IDs differ from the binding plan")
    if {part.id for part in baseline_state.parts} != {
        part.export_name for part in artmeshes.parts
    }:
        raise _error("Cubism Core Part IDs differ from the ArtMesh plan")
    structural_default = evaluate_live2d_moc3_state(document)
    default_residual = _state_residual(
        baseline_state, structural_default, coordinates
    )
    if default_residual > 0.1:
        raise _error("Cubism Core default state exceeds the 0.1 px rest gate")

    targets_by_parameter: dict[str, set[str]] = {
        parameter.parameter_id: set() for parameter in bindings.parameters
    }
    for binding in bindings.bindings:
        targets_by_parameter[binding.parameter_id].add(
            binding.primitive_export_name
        )
    parameter_evidence = []
    for parameter in bindings.parameters:
        test_value = max(
            (parameter.minimum, parameter.maximum),
            key=lambda value: (abs(value - parameter.default), value),
        )
        if test_value == parameter.default:
            raise _error("emitted parameter has no non-default test point")
        try:
            result = exercise_moc_with_core(
                core_file,
                moc_path,
                parameter_values={parameter.export_name: test_value},
                capture_model_state=True,
            )
        except CubismCoreError as exc:
            raise _error(f"Cubism Core rejected parameter {parameter.export_name}") from exc
        state = _state(result, field=parameter.export_name)
        observed = next(
            (item for item in state.parameters if item.id == parameter.export_name),
            None,
        )
        if observed is None or abs(observed.value - test_value) > 1e-6:
            raise _error("Cubism Core did not apply the requested parameter value")
        structural = evaluate_live2d_moc3_state(
            document, {parameter.export_name: test_value}
        )
        residual = _state_residual(state, structural, coordinates)
        if residual > 0.1:
            raise _error("Cubism Core parameter state exceeds structural residual")
        visible_change = _visible_state_change(baseline_state, state)
        if not visible_change:
            raise _error(f"parameter has no visible target effect: {parameter.export_name}")
        parameter_evidence.append(
            Live2DParameterReleaseEvidence(
                parameter_id=parameter.export_name,
                tested_value=test_value,
                target_export_names=tuple(
                    sorted(targets_by_parameter[parameter.parameter_id])
                ),
                visible_change=visible_change,
                core_to_structural_maximum_residual=residual,
                core_vertex_sha256=f"sha256:{state.vertex_position_sha256}",
            )
        )
    restored = exercise_moc_with_core(
        core_file, moc_path, capture_model_state=True
    )
    restored_state = _state(restored, field="restored default state")
    if (
        restored_state.vertex_position_sha256
        != baseline_state.vertex_position_sha256
        or tuple(part.opacity for part in restored_state.parts)
        != tuple(part.opacity for part in baseline_state.parts)
    ):
        raise _error("resetting all parameters does not restore setup state")

    try:
        baseline_render = render_moc_with_offscreen_harness(
            renderer_file,
            moc_path,
            texture_paths,
            width=LIVE2D_RENDER_WIDTH,
            height=LIVE2D_RENDER_HEIGHT,
        )
    except CubismRendererError as exc:
        raise _error("official SDK renderer rejected setup state") from exc
    if baseline_render.nonzero_alpha_pixels <= 0:
        raise _error("official SDK setup render has empty alpha")

    motion_evidence = []
    parameter_defaults = {
        parameter.export_name: parameter.default
        for parameter in bindings.parameters
    }
    for asset in animations.motion_assets:
        motion_path = _bundle_file(root, asset.relative_path)
        meta = asset.payload.get("Meta")
        if not isinstance(meta, Mapping):
            raise _error("motion has no Meta object")
        sample_times = _motion_sample_times(
            asset, defaults=parameter_defaults
        )
        hashes = []
        maximum_parameter_residual = 0.0
        nonzero = True
        visible = False
        for sample_time in sample_times:
            try:
                evidence = render_moc_with_offscreen_harness(
                    renderer_file,
                    moc_path,
                    texture_paths,
                    motion_path=motion_path,
                    evaluation_time=sample_time,
                    observe_parameter_ids=asset.parameter_ids,
                    width=LIVE2D_RENDER_WIDTH,
                    height=LIVE2D_RENDER_HEIGHT,
                )
            except CubismRendererError as exc:
                raise _error(f"official SDK rejected motion {asset.preset_id}") from exc
            hashes.append(evidence.rgba_sha256)
            nonzero = nonzero and evidence.nonzero_alpha_pixels > 0
            visible = visible or evidence.rgba_sha256 != baseline_render.rgba_sha256
            for parameter_id in asset.parameter_ids:
                expected = _f32(
                    _motion_curve_value(asset, parameter_id, sample_time)
                )
                maximum_parameter_residual = max(
                    maximum_parameter_residual,
                    abs(evidence.parameter_values[parameter_id] - expected),
                )
        if maximum_parameter_residual > 1e-6:
            raise _error("motion runtime parameter evaluation differs from canonical curve")
        if not nonzero or not visible:
            raise _error(f"motion has empty or unchanged render: {asset.preset_id}")
        motion_evidence.append(
            Live2DMotionReleaseEvidence(
                preset_id=asset.preset_id,
                sample_times=sample_times,
                rgba_sha256=tuple(hashes),
                observed_parameter_maximum_residual=maximum_parameter_residual,
                nonzero_alpha=nonzero,
                visible_change=visible,
            )
        )

    expression_evidence = []
    for asset in animations.expression_assets:
        expression_path = _bundle_file(root, asset.relative_path)
        parameters = asset.payload.get("Parameters")
        if not isinstance(parameters, list):
            raise _error("expression has no parameter targets")
        expected = {
            str(item["Id"]): float(item["Value"])
            for item in parameters
            if isinstance(item, Mapping)
        }
        try:
            evidence = render_moc_with_offscreen_harness(
                renderer_file,
                moc_path,
                texture_paths,
                expression_path=expression_path,
                observe_parameter_ids=asset.parameter_ids,
                width=LIVE2D_RENDER_WIDTH,
                height=LIVE2D_RENDER_HEIGHT,
            )
            cleared = render_moc_with_offscreen_harness(
                renderer_file,
                moc_path,
                texture_paths,
                width=LIVE2D_RENDER_WIDTH,
                height=LIVE2D_RENDER_HEIGHT,
            )
        except CubismRendererError as exc:
            raise _error(f"official SDK rejected expression {asset.preset_id}") from exc
        parameter_residual = max(
            (
                abs(evidence.parameter_values[identity] - value)
                for identity, value in expected.items()
            ),
            default=0.0,
        )
        visible = evidence.rgba_sha256 != baseline_render.rgba_sha256
        restored_after_clear = cleared.rgba_sha256 == baseline_render.rgba_sha256
        if (
            parameter_residual > 1e-6
            or evidence.nonzero_alpha_pixels <= 0
            or not visible
            or not restored_after_clear
        ):
            raise _error("expression apply/clear render gate failed")
        expression_evidence.append(
            Live2DExpressionReleaseEvidence(
                preset_id=asset.preset_id,
                rgba_sha256=evidence.rgba_sha256,
                observed_parameter_maximum_residual=parameter_residual,
                nonzero_alpha=True,
                visible_change=visible,
                restored_after_clear=restored_after_clear,
            )
        )
    values = {
        "schema_version": LIVE2D_RELEASE_REPORT_VERSION,
        "validator_version": LIVE2D_RELEASE_VALIDATOR_VERSION,
        "structure_report_sha256": structure_report.report_sha256,
        "core_sha256": f"sha256:{probe.sha256}",
        "core_version": probe.version.label,
        "renderer_sha256": _file_sha256(renderer_file),
        "renderer_protocol_digest": baseline_render.validator_protocol_digest,
        "moc_sha256": structure_report.moc_sha256,
        "core_consistency": True,
        "default_rest_maximum_residual": default_residual,
        "baseline_rgba_sha256": baseline_render.rgba_sha256,
        "parameter_evidence": tuple(parameter_evidence),
        "motion_evidence": tuple(motion_evidence),
        "expression_evidence": tuple(expression_evidence),
    }
    provisional = Live2DReleaseValidationReport(**values, report_sha256="")
    return Live2DReleaseValidationReport(
        **values, report_sha256=jcs_sha256(provisional.semantic_payload())
    )


__all__ = [
    "LIVE2D_RELEASE_REPORT_VERSION",
    "LIVE2D_RELEASE_VALIDATOR_VERSION",
    "Live2DExpressionReleaseEvidence",
    "Live2DMotionReleaseEvidence",
    "Live2DParameterReleaseEvidence",
    "Live2DReleaseGateError",
    "Live2DReleaseValidationReport",
    "validate_live2d_release_bundle",
]
