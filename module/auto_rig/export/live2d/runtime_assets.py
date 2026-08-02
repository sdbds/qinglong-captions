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

LIVE2D_RUNTIME_ASSET_PLAN_VERSION = "live2d-runtime-asset-plan-v1"
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


def validate_cubism_runtime_json_bytes(
    encoded: bytes, expected_payload: Mapping[str, object]
) -> bytes:
    if not isinstance(encoded, bytes) or encoded != cubism_runtime_json_bytes(
        expected_payload
    ):
        raise _error("Cubism runtime JSON encoding is not canonical")
    if not encoded.endswith(b"\n") or b"\n  \"" not in encoded:
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


def _assemble(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    artmeshes: Live2DArtMeshPlan,
    animations: Live2DAnimationPlan,
) -> Live2DRuntimeAssetPlan:
    motion_entries = [
        {
            "File": asset.relative_path,
            "FadeInTime": 0.0,
            "FadeOutTime": 0.0,
        }
        for asset in animations.motion_assets
    ]
    expression_entries = [
        {"Name": asset.artifact_export_name, "File": asset.relative_path}
        for asset in animations.expression_assets
    ]
    texture_paths = tuple(
        f"textures/page_{index}.png"
        for index in range(artmeshes.texture_page_count)
    )
    references: dict[str, object] = {
        "Moc": f"{LIVE2D_ARTIFACT_BASENAME}.moc3",
        "Textures": list(texture_paths),
        "DisplayInfo": f"{LIVE2D_ARTIFACT_BASENAME}.cdi3.json",
        "Motions": {"Presets": motion_entries},
    }
    if expression_entries:
        references["Expressions"] = expression_entries
    used_parameter_ids = {
        parameter_id
        for asset in (*animations.motion_assets, *animations.expression_assets)
        for parameter_id in asset.parameter_ids
    }
    groups = []
    eye_ids = sorted(
        parameter.export_name
        for parameter in bindings.parameters
        if parameter.control_id
        in {"control/eye_open.xmin", "control/eye_open.xmax"}
        and parameter.export_name in used_parameter_ids
    )
    lip_ids = sorted(
        parameter.export_name
        for parameter in bindings.parameters
        if parameter.control_id == "control/mouth_open"
        and parameter.export_name in used_parameter_ids
    )
    for name, identities in (("EyeBlink", eye_ids), ("LipSync", lip_ids)):
        if identities:
            groups.append(
                {"Target": "Parameter", "Name": name, "Ids": identities}
            )
    model3_payload: dict[str, object] = {
        "Version": LIVE2D_MODEL3_VERSION,
        "FileReferences": references,
    }
    if groups:
        model3_payload["Groups"] = groups
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
        "Parts": [
            {"Id": part.export_name, "Name": part.export_name}
            for part in artmeshes.parts
        ],
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
    animation_paths = tuple(
        asset.relative_path
        for asset in (*animations.motion_assets, *animations.expression_assets)
    )
    asset_payload = {
        "model3": model3.to_dict(),
        "cdi3": cdi3.to_dict(),
        "animation_runtime_sha256": [
            asset.runtime_sha256
            for asset in (*animations.motion_assets, *animations.expression_assets)
        ],
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
    return Live2DRuntimeAssetPlan(
        **values, plan_sha256=jcs_sha256(provisional.semantic_payload())
    )


def build_live2d_runtime_asset_plan(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    artmeshes: Live2DArtMeshPlan,
    animations: Live2DAnimationPlan,
) -> Live2DRuntimeAssetPlan:
    validate_rig_document(rig)
    validate_live2d_symbol_view(symbols)
    if (
        not isinstance(animations, Live2DAnimationPlan)
        or animations.plan_sha256 != jcs_sha256(animations.semantic_payload())
    ):
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
