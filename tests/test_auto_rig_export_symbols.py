from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export_symbols import (
    EXPORT_NAME_CODEC_VERSION,
    GLOBAL_EXPORT_SYMBOL_TABLE_VERSION,
    GlobalExportSymbolTableError,
    _resolve_export_symbols,
    build_global_export_symbol_table,
    encode_base_export_name,
    validate_global_export_symbol_table,
)
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.primitive_candidates import _key
from tests.test_auto_rig_primitive_candidates import _enumerate


def _symbol(table, *, kind: str, base_source: str, parameter_id: str | None = None):
    matches = tuple(
        symbol
        for symbol in table.symbols
        if symbol.typed_primitive_key.kind == kind
        and symbol.typed_primitive_key.base_source_internal_id == base_source
        and symbol.typed_primitive_key.parameter_id == parameter_id
    )
    assert len(matches) == 1
    return matches[0]


def test_global_symbols_match_golden_names_and_real_namespaces(tmp_path: Path) -> None:
    _cache, controls, _presets, _bindings, candidates = _enumerate(tmp_path)

    table = build_global_export_symbol_table(candidates, controls)

    assert table.schema_version == GLOBAL_EXPORT_SYMBOL_TABLE_VERSION
    assert table.name_codec_version == EXPORT_NAME_CODEC_VERSION
    assert validate_global_export_symbol_table(table, candidates, controls) is table
    assert _symbol(
        table,
        kind="live2d_rotation_deformer",
        base_source="bone/torso",
        parameter_id="parameter/auto_idle",
    ).export_name == "torso__rot_auto_idle"
    assert _symbol(
        table,
        kind="live2d_parameter",
        base_source="parameter/angle_x",
        parameter_id="parameter/angle_x",
    ).export_name == "ParamAngleX"
    assert _symbol(
        table,
        kind="spine_animation",
        base_source="clip/wave.xmin",
    ).export_name == "wave_xmin"

    topwear = tuple(
        symbol
        for symbol in table.symbols
        if symbol.typed_primitive_key.base_source_internal_id == "part/topwear"
        and symbol.typed_primitive_key.kind
        in {
            "spine_slot",
            "spine_attachment_key",
            "spine_attachment_object",
            "spine_atlas_region",
            "live2d_part",
            "live2d_artmesh",
        }
    )
    assert {symbol.export_name for symbol in topwear} == {"topwear"}
    assert len({symbol.namespace_key.key_sha256 for symbol in topwear}) == len(topwear)

    candidate_key_ids = {
        candidate.typed_primitive_key.key_sha256 for candidate in candidates.candidates
    }
    assert {symbol.typed_primitive_key.key_sha256 for symbol in table.symbols} == (
        candidate_key_ids
    )


def test_name_codec_resolves_all_collision_members_and_keeps_namespaces_separate() -> None:
    assert encode_base_export_name("part/front-hair") == "front_hair"
    first = _key("spine_4_2", "spine_bone", ("part/a-b",), "part/a-b")
    second = _key("spine_4_2", "spine_bone", ("part/a.b",), "part/a.b")
    other_namespace = _key(
        "live2d_moc3_v4_00",
        "live2d_part",
        ("part/a-b",),
        "part/a-b",
    )

    collided = _resolve_export_symbols((first, second), parameter_export_names={})
    assert all(symbol.export_name.startswith("a_b_") for symbol in collided)
    assert all(len(symbol.export_name.encode("ascii")) < 64 for symbol in collided)
    assert len({symbol.export_name for symbol in collided}) == 2

    separate = _resolve_export_symbols(
        (first, other_namespace), parameter_export_names={}
    )
    assert [symbol.export_name for symbol in separate] == ["a_b", "a_b"]
    assert separate[0].namespace_key != separate[1].namespace_key


def test_name_codec_enforces_the_63_byte_boundary() -> None:
    stem_63 = "a" * 63
    stem_64 = "b" * 64
    exact = _key(
        "spine_4_2",
        "spine_bone",
        (f"part/{stem_63}",),
        f"part/{stem_63}",
    )
    too_long = _key(
        "spine_4_2",
        "spine_bone",
        (f"part/{stem_64}",),
        f"part/{stem_64}",
    )

    symbols = _resolve_export_symbols((exact, too_long), parameter_export_names={})
    by_base = {symbol.base_export_name: symbol for symbol in symbols}
    assert by_base[stem_63].export_name == stem_63
    assert len(by_base[stem_64].export_name.encode("ascii")) == 63
    assert by_base[stem_64].export_name.startswith("b" * 46 + "_")


def test_symbol_validator_rejects_rehashed_survival_set_renaming(tmp_path: Path) -> None:
    _cache, controls, _presets, _bindings, candidates = _enumerate(tmp_path)
    table = build_global_export_symbol_table(candidates, controls)
    rows = list(table.symbols)
    changed = replace(rows[0], export_name="renamed", symbol_sha256="")
    changed = replace(changed, symbol_sha256=jcs_sha256(changed.semantic_payload()))
    rows[0] = changed
    provisional = replace(table, symbols=tuple(rows), table_sha256="")
    tampered = replace(
        provisional,
        table_sha256=jcs_sha256(provisional.semantic_payload()),
    )

    with pytest.raises(GlobalExportSymbolTableError) as exc_info:
        validate_global_export_symbol_table(tampered, candidates, controls)

    assert exc_info.value.code == "invalid_global_export_symbol_table"
