from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .animations import (
    Live2DAnimationPlan,
    encode_live2d_animation_asset,
)
from .artmesh import Live2DArtMeshPlan
from .binding_plan import Live2DBindingPlan
from .e0_assets import cubism_runtime_json_bytes
from .symbols import Live2DSymbolView, validate_live2d_symbol_view

LIVE2D_RUNTIME_ASSET_PLAN_VERSION = "live2d-runtime-asset-plan-v4"
LIVE2D_RUNTIME_JSON_ENCODER_VERSION = "cubism-runtime-json-v1"
LIVE2D_ARTIFACT_BASENAME = "model"
LIVE2D_MODEL3_VERSION = 3
LIVE2D_CDI3_VERSION = 3


class Live2DRuntimeAssetError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_runtime_asset: {message}")


def _error(message: str) -> Live2DRuntimeAssetError:
    return Live2DRuntimeAssetError(message)


def _runtime_sha256(payload: Mapping[str, object]) -> str:
    return f"sha256:{sha256(cubism_runtime_json_bytes(payload)).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Live2DRuntimeJsonAsset:
    asset_kind: str
    relative_path: str
    payload: dict[str, object]
    runtime_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_kind": self.asset_kind,
            "relative_path": self.relative_path,
            "payload": self.payload,
            "runtime_sha256": self.runtime_sha256,
        }


@dataclass(frozen=True, slots=True)
class Live2DRuntimeAssetPlan:
    schema_version: str
    runtime_json_encoder_version: str
    artifact_basename: str
    rig_document_sha256: str
    symbol_view_sha256: str
    binding_plan_sha256: str
    artmesh_plan_sha256: str
    animation_plan_sha256: str
    model3: Live2DRuntimeJsonAsset
    cdi3: Live2DRuntimeJsonAsset
    referenced_animation_paths: tuple[str, ...]
    referenced_texture_paths: tuple[str, ...]
    asset_set_sha256: str
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_json_encoder_version": self.runtime_json_encoder_version,
            "artifact_basename": self.artifact_basename,
            "rig_document_sha256": self.rig_document_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "binding_plan_sha256": self.binding_plan_sha256,
            "artmesh_plan_sha256": self.artmesh_plan_sha256,
            "animation_plan_sha256": self.animation_plan_sha256,
            "model3": self.model3.to_dict(),
            "cdi3": self.cdi3.to_dict(),
            "referenced_animation_paths": list(self.referenced_animation_paths),
            "referenced_texture_paths": list(self.referenced_texture_paths),
            "asset_set_sha256": self.asset_set_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def validate_cubism_runtime_json_bytes(encoded: bytes, expected_payload: Mapping[str, object]) -> bytes:
    if not isinstance(encoded, bytes) or encoded != cubism_runtime_json_bytes(expected_payload):
        raise _error("Cubism runtime JSON encoding is not canonical")
    if not encoded.endswith(b"\n") or b'\n  "' not in encoded:
        raise _error("Cubism runtime JSON encoding must be indented with a newline")
    return encoded


def encode_live2d_runtime_asset(asset: Live2DRuntimeJsonAsset) -> bytes:
    if not isinstance(asset, Live2DRuntimeJsonAsset):
        raise _error("runtime JSON asset has the wrong type")
    encoded = cubism_runtime_json_bytes(asset.payload)
    validate_cubism_runtime_json_bytes(encoded, asset.payload)
    if f"sha256:{sha256(encoded).hexdigest()}" != asset.runtime_sha256:
        raise _error("runtime JSON asset digest mismatch")
    return encoded


def _hit_area_entries(
    rig: RigDocument,
    artmeshes: Live2DArtMeshPlan,
) -> list[dict[str, str]]:
    payload = rig.to_dict()
    part_by_id = {
        part["part_id"]: part for part in payload["parts"] if part["source_kind"] == "see_through" and part["setup_visibility"] == 1
    }
    mesh_by_id = {mesh["mesh_id"]: mesh for mesh in payload["meshes"]}
    joint_by_id = {
        joint["joint_id"]: (float(joint["x"]), float(joint["y"]))
        for joint in payload["joints"]
        if joint["status"] == "resolved" and joint["x"] is not None and joint["y"] is not None
    }

    def bbox(artmesh) -> tuple[int, int, int, int]:
        return tuple(mesh_by_id[artmesh.mesh_id]["component_bbox"])

    def area(artmesh) -> tuple[int, str]:
        x1, y1, x2, y2 = bbox(artmesh)
        return ((x2 - x1) * (y2 - y1), artmesh.export_name)

    def largest_for_tags(base_tags: tuple[str, ...]):
        for base_tag in base_tags:
            candidate_part_ids = {part_id for part_id, part in part_by_id.items() if part["base_tag"] == base_tag}
            candidates = tuple(artmesh for artmesh in artmeshes.artmeshes if artmesh.part_id in candidate_part_ids)
            if not candidates:
                continue

            return max(candidates, key=area)
        return None

    def limb_pair(
        *,
        base_tags: tuple[str, ...],
        joint_stems: tuple[str, ...],
    ) -> dict[str, object]:
        candidates = tuple(
            artmesh
            for artmesh in artmeshes.artmeshes
            if artmesh.part_id in part_by_id and part_by_id[artmesh.part_id]["base_tag"] in base_tags
        )

        def score(artmesh, side: str) -> tuple[int, int, float, int, str]:
            part_side = part_by_id[artmesh.part_id]["side"]
            points = tuple(joint_by_id[joint_id] for stem in joint_stems if (joint_id := f"joint/{stem}.{side}") in joint_by_id)
            x1, y1, x2, y2 = bbox(artmesh)
            contained = sum(x1 <= x <= x2 and y1 <= y <= y2 for x, y in points)

            def distance_squared(point: tuple[float, float]) -> float:
                x, y = point
                dx = max(float(x1) - x, 0.0, x - float(x2))
                dy = max(float(y1) - y, 0.0, y - float(y2))
                return dx * dx + dy * dy

            return (
                int(part_side == side),
                contained,
                -sum(distance_squared(point) for point in points),
                area(artmesh)[0],
                artmesh.export_name,
            )

        choices: dict[str, tuple[object, ...]] = {}
        for side in ("xmin", "xmax"):
            if not any(f"joint/{stem}.{side}" in joint_by_id for stem in joint_stems):
                choices[side] = ()
                continue
            choices[side] = tuple(artmesh for artmesh in candidates if part_by_id[artmesh.part_id]["side"] in {None, side})
        pairs = tuple(
            (left, right) for left in choices["xmin"] for right in choices["xmax"] if left.export_name != right.export_name
        )

        def pair_score(pair) -> tuple[object, ...]:
            left_score = score(pair[0], "xmin")
            right_score = score(pair[1], "xmax")
            return (
                left_score[0] + right_score[0],
                left_score[1] + right_score[1],
                left_score[2] + right_score[2],
                left_score[3] + right_score[3],
                left_score[4],
                right_score[4],
            )

        explicit_pairs = tuple(
            pair
            for pair in pairs
            if part_by_id[pair[0].part_id]["side"] == "xmin" and part_by_id[pair[1].part_id]["side"] == "xmax"
        )
        if explicit_pairs:
            left, right = max(explicit_pairs, key=pair_score)
            return {"xmin": left, "xmax": right}

        pairs = tuple(pair for pair in pairs if score(pair[0], "xmin")[1] > 0 and score(pair[1], "xmax")[1] > 0)
        if pairs:
            left, right = max(pairs, key=pair_score)
            return {"xmin": left, "xmax": right}
        if not candidates or not all(choices[side] for side in ("xmin", "xmax")):
            return {}

        def combined_score(artmesh) -> tuple[object, ...]:
            left_score = score(artmesh, "xmin")
            right_score = score(artmesh, "xmax")
            return (
                left_score[1] + right_score[1],
                left_score[2] + right_score[2],
                area(artmesh)[0],
                artmesh.export_name,
            )

        combined_candidates = tuple(
            artmesh
            for artmesh in candidates
            if part_by_id[artmesh.part_id]["side"] is None and score(artmesh, "xmin")[1] > 0 and score(artmesh, "xmax")[1] > 0
        )
        if not combined_candidates:
            return {}
        return {"combined": max(combined_candidates, key=combined_score)}

    entries = []
    for name, base_tags in (
        ("Head", ("face",)),
        ("Body", ("topwear", "bottomwear")),
    ):
        artmesh = largest_for_tags(base_tags)
        if artmesh is not None:
            entries.append({"Id": artmesh.export_name, "Name": name})
    for prefix, base_tags, joint_stems in (
        ("Arm", ("handwear",), ("shoulder", "elbow", "wrist")),
        ("Leg", ("legwear", "footwear"), ("hip", "knee", "ankle")),
    ):
        pair = limb_pair(base_tags=base_tags, joint_stems=joint_stems)
        combined = pair.get("combined")
        if combined is not None:
            entries.append({"Id": combined.export_name, "Name": f"{prefix}s"})
        else:
            for side, suffix in (("xmin", "ScreenLeft"), ("xmax", "ScreenRight")):
                artmesh = pair.get(side)
                if artmesh is not None:
                    entries.append({"Id": artmesh.export_name, "Name": f"{prefix}{suffix}"})
    return entries


def _assemble(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    artmeshes: Live2DArtMeshPlan,
    animations: Live2DAnimationPlan,
) -> Live2DRuntimeAssetPlan:
    motion_entries = [
        {
            # Cubism ignores unknown motion-entry fields, while ZIP viewers can
            # retain this stable label after replacing File with a data URI.
            "Name": asset.artifact_export_name,
            "File": asset.relative_path,
            "FadeInTime": 0.0,
            "FadeOutTime": 0.0,
        }
        for asset in animations.motion_assets
    ]
    expression_entries = [
        {"Name": asset.artifact_export_name, "File": asset.relative_path} for asset in animations.expression_assets
    ]
    texture_paths = tuple(f"textures/page_{index}.png" for index in range(artmeshes.texture_page_count))
    references: dict[str, object] = {
        "Moc": f"{LIVE2D_ARTIFACT_BASENAME}.moc3",
        "Textures": list(texture_paths),
        "DisplayInfo": f"{LIVE2D_ARTIFACT_BASENAME}.cdi3.json",
        "Motions": {"Presets": motion_entries},
    }
    if expression_entries:
        references["Expressions"] = expression_entries
    used_parameter_ids = {
        parameter_id for asset in (*animations.motion_assets, *animations.expression_assets) for parameter_id in asset.parameter_ids
    }
    groups = []
    lip_ids = sorted(
        parameter.export_name
        for parameter in bindings.parameters
        if parameter.control_id == "control/mouth_open" and parameter.export_name in used_parameter_ids
    )
    # Eye openness is driven only by the explicit blink motion and expressions.
    # An EyeBlink group opts compatible runtimes into global random blinking,
    # which leaks into unrelated motions and can overwrite one-eye expressions.
    for name, identities in (("LipSync", lip_ids),):
        if identities:
            groups.append({"Target": "Parameter", "Name": name, "Ids": identities})
    model3_payload: dict[str, object] = {
        "Version": LIVE2D_MODEL3_VERSION,
        "FileReferences": references,
    }
    if groups:
        model3_payload["Groups"] = groups
    hit_areas = _hit_area_entries(rig, artmeshes)
    if hit_areas:
        model3_payload["HitAreas"] = hit_areas
    cdi3_payload: dict[str, object] = {
        "Version": LIVE2D_CDI3_VERSION,
        "Parameters": [
            {
                "Id": parameter.export_name,
                "GroupId": "",
                "Name": parameter.export_name,
            }
            for parameter in bindings.parameters
        ],
        "ParameterGroups": [],
        "Parts": [{"Id": part.export_name, "Name": part.export_name} for part in artmeshes.parts],
    }
    model3 = Live2DRuntimeJsonAsset(
        asset_kind="model3",
        relative_path=f"{LIVE2D_ARTIFACT_BASENAME}.model3.json",
        payload=model3_payload,
        runtime_sha256=_runtime_sha256(model3_payload),
    )
    cdi3 = Live2DRuntimeJsonAsset(
        asset_kind="cdi3",
        relative_path=f"{LIVE2D_ARTIFACT_BASENAME}.cdi3.json",
        payload=cdi3_payload,
        runtime_sha256=_runtime_sha256(cdi3_payload),
    )
    animation_paths = tuple(asset.relative_path for asset in (*animations.motion_assets, *animations.expression_assets))
    asset_payload = {
        "model3": model3.to_dict(),
        "cdi3": cdi3.to_dict(),
        "animation_runtime_sha256": [asset.runtime_sha256 for asset in (*animations.motion_assets, *animations.expression_assets)],
        "texture_paths": list(texture_paths),
    }
    values = {
        "schema_version": LIVE2D_RUNTIME_ASSET_PLAN_VERSION,
        "runtime_json_encoder_version": LIVE2D_RUNTIME_JSON_ENCODER_VERSION,
        "artifact_basename": LIVE2D_ARTIFACT_BASENAME,
        "rig_document_sha256": rig.document_sha256,
        "symbol_view_sha256": symbols.view_sha256,
        "binding_plan_sha256": bindings.plan_sha256,
        "artmesh_plan_sha256": artmeshes.plan_sha256,
        "animation_plan_sha256": animations.plan_sha256,
        "model3": model3,
        "cdi3": cdi3,
        "referenced_animation_paths": animation_paths,
        "referenced_texture_paths": texture_paths,
        "asset_set_sha256": jcs_sha256(asset_payload),
    }
    provisional = Live2DRuntimeAssetPlan(**values, plan_sha256="")
    return Live2DRuntimeAssetPlan(**values, plan_sha256=jcs_sha256(provisional.semantic_payload()))


def build_live2d_runtime_asset_plan(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    artmeshes: Live2DArtMeshPlan,
    animations: Live2DAnimationPlan,
) -> Live2DRuntimeAssetPlan:
    validate_rig_document(rig)
    validate_live2d_symbol_view(symbols)
    if not isinstance(animations, Live2DAnimationPlan) or animations.plan_sha256 != jcs_sha256(animations.semantic_payload()):
        raise _error("animation plan type or digest is invalid")
    for asset in (*animations.motion_assets, *animations.expression_assets):
        encode_live2d_animation_asset(asset)
    return _assemble(rig, symbols, bindings, artmeshes, animations)


def validate_live2d_runtime_asset_plan(
    plan: Live2DRuntimeAssetPlan,
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    artmeshes: Live2DArtMeshPlan,
    animations: Live2DAnimationPlan,
) -> Live2DRuntimeAssetPlan:
    if not isinstance(plan, Live2DRuntimeAssetPlan):
        raise _error("runtime asset plan has the wrong type")
    encode_live2d_runtime_asset(plan.model3)
    encode_live2d_runtime_asset(plan.cdi3)
    expected = _assemble(rig, symbols, bindings, artmeshes, animations)
    if plan != expected:
        raise _error("runtime asset plan differs from canonical Stage C/E plans")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("runtime asset plan digest mismatch")
    return plan


__all__ = [
    "LIVE2D_ARTIFACT_BASENAME",
    "LIVE2D_CDI3_VERSION",
    "LIVE2D_MODEL3_VERSION",
    "LIVE2D_RUNTIME_ASSET_PLAN_VERSION",
    "LIVE2D_RUNTIME_JSON_ENCODER_VERSION",
    "Live2DRuntimeAssetError",
    "Live2DRuntimeAssetPlan",
    "Live2DRuntimeJsonAsset",
    "build_live2d_runtime_asset_plan",
    "encode_live2d_runtime_asset",
    "validate_cubism_runtime_json_bytes",
    "validate_live2d_runtime_asset_plan",
]
