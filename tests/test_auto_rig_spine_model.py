from __future__ import annotations

from pathlib import Path

from module.auto_rig.export.spine.atlas import build_spine_atlas_plan
from module.auto_rig.export.spine.bind_plan import build_spine_bind_plan
from module.auto_rig.export.spine.coordinates import build_spine_coordinate_plan
from module.auto_rig.export.spine.model import (
    SPINE_DOCUMENT_VERSION,
    build_spine_document,
    validate_spine_document,
)
from module.auto_rig.export.spine.symbols import build_spine_symbol_view
from tests.test_auto_rig_rig_document import _build


def test_spine_setup_document_has_one_visible_attachment_per_component(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    coordinates = build_spine_coordinate_plan(
        payload["canvas"],
        input_fingerprint=payload["input_fingerprint"],
        landmarks=tuple(
            tuple(point)
            for bone in payload["bones"]
            for point in (bone["head"], bone["tail"])
            if point is not None
        ),
    )
    bind = build_spine_bind_plan(payload["bones"], payload["meshes"], coordinates)
    symbols = build_spine_symbol_view(payload["export_symbols"])
    atlas = build_spine_atlas_plan(
        payload["texture_pages"], payload["parts"], symbols
    )
    document = build_spine_document(rig, coordinates, bind, symbols, atlas)
    skeleton = document.to_dict()

    assert document.schema_version == SPINE_DOCUMENT_VERSION
    assert skeleton["skeleton"]["spine"] == "4.2"
    assert skeleton["bones"][0] == {"name": "root"}
    assert len(skeleton["slots"]) == len(payload["meshes"])
    assert len({slot["name"] for slot in skeleton["slots"]}) == len(payload["meshes"])
    assert all("attachment" in slot for slot in skeleton["slots"])
    assert [slot["name"] for slot in skeleton["slots"]] == document.slot_names

    skin = skeleton["skins"][0]
    assert skin["name"] == "default"
    assert set(skin["attachments"]) == set(document.slot_names)
    assert all(len(attachments) == 1 for attachments in skin["attachments"].values())
    assert [record["component_draw_rank"] for record in document.component_records] == list(
        range(len(payload["meshes"]))
    )
    assert validate_spine_document(document, rig, coordinates, bind, symbols, atlas) is document


def test_spine_setup_document_preserves_region_sharing_and_setup_visibility(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    coordinates = build_spine_coordinate_plan(
        payload["canvas"], input_fingerprint=payload["input_fingerprint"]
    )
    bind = build_spine_bind_plan(payload["bones"], payload["meshes"], coordinates)
    symbols = build_spine_symbol_view(payload["export_symbols"])
    atlas = build_spine_atlas_plan(
        payload["texture_pages"], payload["parts"], symbols
    )
    document = build_spine_document(rig, coordinates, bind, symbols, atlas)

    records_by_part = {}
    for record in document.component_records:
        records_by_part.setdefault(record["part_id"], []).append(record)
    for records in records_by_part.values():
        assert len({record["atlas_region_name"] for record in records}) == 1
    part_by_id = {part["part_id"]: part for part in payload["parts"]}
    slot_by_name = {slot["name"]: slot for slot in document.to_dict()["slots"]}
    for record in document.component_records:
        expected_hidden = part_by_id[record["part_id"]]["setup_visibility"] == 0
        assert (slot_by_name[record["slot_name"]].get("color") == "ffffff00") is expected_hidden
