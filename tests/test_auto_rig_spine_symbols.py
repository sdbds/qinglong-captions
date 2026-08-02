from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.export.spine.symbols import (
    SpineSymbolError,
    build_spine_symbol_view,
    require_spine_symbol,
    validate_spine_symbol_view,
)
from tests.test_auto_rig_rig_document import _build


def test_spine_symbol_view_resolves_typed_names_without_exporter_sanitize(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    view = build_spine_symbol_view(payload["export_symbols"])

    root = require_spine_symbol(
        view, kind="spine_bone", source_internal_ids=("bone/root",)
    )
    first_mesh = payload["meshes"][0]
    slot = require_spine_symbol(
        view,
        kind="spine_slot",
        source_internal_ids=(first_mesh["part_id"], first_mesh["component_id"]),
    )
    region = require_spine_symbol(
        view,
        kind="spine_atlas_region",
        source_internal_ids=(first_mesh["part_id"],),
    )

    assert root.export_name == "root"
    assert slot.export_name.isascii()
    assert region.namespace == "atlas_region"
    assert validate_spine_symbol_view(view) is view
    assert all(record.format_id == "spine_4_2" for record in view.symbols)


def test_spine_symbol_view_rejects_renamed_or_missing_typed_key(tmp_path: Path) -> None:
    *_, rig = _build(tmp_path)
    table = rig.to_dict()["export_symbols"]
    first = next(
        symbol
        for symbol in table["symbols"]
        if symbol["typed_primitive_key"]["format_id"] == "spine_4_2"
    )
    first["export_name"] = "exporter_re_sanitized"

    with pytest.raises(SpineSymbolError, match="digest"):
        build_spine_symbol_view(table)

    *_, clean_rig = _build(tmp_path / "clean")
    view = build_spine_symbol_view(clean_rig.to_dict()["export_symbols"])
    with pytest.raises(SpineSymbolError, match="missing"):
        require_spine_symbol(
            view, kind="spine_bone", source_internal_ids=("bone/not-present",)
        )
