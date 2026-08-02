from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from module.auto_rig.export.spine.animations import build_spine_animation_plan
from module.auto_rig.export.spine.atlas import (
    build_spine_atlas_plan,
    serialize_spine_atlas,
)
from module.auto_rig.export.spine.bind_plan import build_spine_bind_plan
from module.auto_rig.export.spine.coordinates import build_spine_coordinate_plan
from module.auto_rig.export.spine.model import build_spine_document
from module.auto_rig.export.spine.serializer import (
    SPINE_SERIALIZATION_VERSION,
    build_spine_encoding_descriptor,
    parse_spine_document,
    serialize_spine_document,
    serialize_spine_report,
)
from module.auto_rig.export.spine.symbols import build_spine_symbol_view
from module.auto_rig.export.spine.validator import (
    SPINE_BUNDLE_VALIDATOR_VERSION,
    SpineBundleValidationError,
    validate_spine_bundle,
)
from tests.test_auto_rig_stage_c import _execute


@pytest.fixture(scope="module")
def spine_bundle(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("spine-validator"))
    *_inputs, result = _execute(root)
    rig = result.rig
    payload = rig.to_dict()
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
    pages = {
        page.path: (
            root
            / Path(
                *next(
                    raw["relative_path"].split("/")
                    for raw in payload["texture_pages"]
                    if raw["index"] == page.index
                )
            )
        ).read_bytes()
        for page in atlas.pages
    }
    return (
        rig,
        coordinates,
        bind,
        symbols,
        atlas,
        animations,
        document,
        skeleton_bytes,
        atlas_bytes,
        pages,
    )


def _first_attachment(payload: dict[str, object], document):
    record = document.component_records[0]
    skin = payload["skins"][0]
    return skin["attachments"][record["slot_name"]][record["attachment_key_name"]]


def _add_curve(value: object) -> bool:
    if isinstance(value, dict):
        for child in value.values():
            if _add_curve(child):
                return True
    elif isinstance(value, list):
        if value and all(isinstance(item, dict) and "time" in item for item in value):
            value[0]["curve"] = "linear"
            return True
        for child in value:
            if _add_curve(child):
                return True
    return False


def test_spine_serializers_are_canonical_and_descriptor_is_versioned(
    spine_bundle,
) -> None:
    *_, document, skeleton_bytes, _atlas_bytes, _pages = spine_bundle
    assert serialize_spine_document(document) == skeleton_bytes
    assert parse_spine_document(skeleton_bytes) == document.to_dict()
    assert serialize_spine_report({"z": 1, "a": 2}) == b'{"a":2,"z":1}'
    descriptor = build_spine_encoding_descriptor()
    assert descriptor.schema_version == SPINE_SERIALIZATION_VERSION
    assert descriptor.descriptor_sha256.startswith("sha256:")


def test_spine_bundle_validator_reconstructs_setup_and_public_references(
    spine_bundle,
) -> None:
    rig, coordinates, bind, symbols, atlas, animations, _document, skeleton, atlas_bytes, pages = (
        spine_bundle
    )
    report = validate_spine_bundle(
        skeleton,
        atlas_bytes,
        pages,
        rig,
        coordinates,
        bind,
        symbols,
        atlas,
        animations,
    )
    assert report.schema_version == SPINE_BUNDLE_VALIDATOR_VERSION
    assert report.validated is True
    assert report.maximum_setup_residual_px <= 0.1
    assert report.animation_count == len(animations.animations)
    assert report.page_count == len(pages)
    assert report.validator_fingerprint.startswith("sha256:")


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_version",
        "nested_topology",
        "illegal_curve",
        "missing_attachment",
        "reversed_slots",
        "bad_bone_index",
        "mismatched_region_path",
        "stale_symbol",
    ),
)
def test_spine_bundle_validator_rejects_structural_json_mutations(
    spine_bundle,
    mutation: str,
) -> None:
    rig, coordinates, bind, symbols, atlas, animations, document, skeleton, atlas_bytes, pages = (
        spine_bundle
    )
    payload = deepcopy(parse_spine_document(skeleton))
    if mutation == "wrong_version":
        payload["skeleton"]["spine"] = "4.1"
    elif mutation == "nested_topology":
        attachment = _first_attachment(payload, document)
        attachment["triangles"] = [attachment["triangles"]]
    elif mutation == "illegal_curve":
        assert _add_curve(payload["animations"])
    elif mutation == "missing_attachment":
        record = document.component_records[0]
        del payload["skins"][0]["attachments"][record["slot_name"]]
    elif mutation == "reversed_slots":
        payload["slots"].reverse()
    elif mutation == "bad_bone_index":
        record = next(item for item in document.component_records if item["weighted"])
        attachment = payload["skins"][0]["attachments"][record["slot_name"]][
            record["attachment_key_name"]
        ]
        attachment["vertices"][1] = len(payload["bones"])
    elif mutation == "mismatched_region_path":
        _first_attachment(payload, document)["path"] = "missing_region"
    else:
        payload["bones"][1]["name"] = "stale_bone_name"

    with pytest.raises(SpineBundleValidationError):
        validate_spine_bundle(
            serialize_spine_report(payload),
            atlas_bytes,
            pages,
            rig,
            coordinates,
            bind,
            symbols,
            atlas,
            animations,
        )


def test_spine_bundle_validator_rejects_noncanonical_json(spine_bundle) -> None:
    rig, coordinates, bind, symbols, atlas, animations, document, _skeleton, atlas_bytes, pages = (
        spine_bundle
    )
    noncanonical = json.dumps(document.to_dict(), indent=2).encode("utf-8")
    with pytest.raises(SpineBundleValidationError, match="canonical"):
        validate_spine_bundle(
            noncanonical,
            atlas_bytes,
            pages,
            rig,
            coordinates,
            bind,
            symbols,
            atlas,
            animations,
        )


@pytest.mark.parametrize("mutation", ("pma", "missing_region", "missing_page", "changed_page"))
def test_spine_bundle_validator_rejects_atlas_and_page_mutations(
    spine_bundle,
    mutation: str,
) -> None:
    rig, coordinates, bind, symbols, atlas, animations, _document, skeleton, atlas_bytes, pages = (
        spine_bundle
    )
    changed_atlas = atlas_bytes
    changed_pages = dict(pages)
    if mutation == "pma":
        changed_atlas = atlas_bytes.replace(b"pma: false", b"pma: true", 1)
    elif mutation == "missing_region":
        lines = atlas_bytes.decode("ascii").splitlines()
        region_index = lines.index(atlas.pages[0].regions[0].name)
        del lines[region_index : region_index + 5]
        changed_atlas = ("\n".join(lines) + "\n").encode("ascii")
    elif mutation == "missing_page":
        del changed_pages[next(iter(changed_pages))]
    else:
        page_name = next(iter(changed_pages))
        changed_pages[page_name] += b"changed"

    with pytest.raises(SpineBundleValidationError):
        validate_spine_bundle(
            skeleton,
            changed_atlas,
            changed_pages,
            rig,
            coordinates,
            bind,
            symbols,
            atlas,
            animations,
        )
