from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.artmesh import build_live2d_artmesh_plan
from module.auto_rig.export.live2d.coordinates import live2d_local_to_canvas
from module.auto_rig.export.live2d.cubism_core import exercise_moc_with_core
from module.auto_rig.export.live2d.document import (
    _build as _build_live2d_moc3_document,
)
from module.auto_rig.export.live2d.document import (
    build_live2d_moc3_document,
)
from module.auto_rig.export.live2d.keyforms import build_live2d_keyform_plan
from module.auto_rig.export.live2d.moc3 import parse_moc3_v400_envelope
from module.auto_rig.export.live2d.moc3_codec import (
    Moc3V400Document,
    decode_moc3_v400,
    encode_moc3_v400,
)
from module.auto_rig.export.live2d.moc3_sections_kernel import (
    MOC3_V400_SECTION_SPECS,
    moc3_element_size,
)
from module.auto_rig.export.live2d.validator import (
    build_live2d_structure_validation_report,
    evaluate_live2d_moc3_state,
    validate_live2d_moc3_document,
    validate_live2d_moc3_payload,
)
from tests.test_auto_rig_live2d_coordinates import _coordinate_plans


def _document_plans(tmp_path: Path):
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(tmp_path)
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    keyforms = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)
    document = build_live2d_moc3_document(rig, bindings, coordinates, artmeshes, keyforms)
    return rig, bindings, coordinates, artmeshes, keyforms, document


@pytest.fixture(scope="module")
def document_plans(tmp_path_factory: pytest.TempPathFactory):
    return _document_plans(tmp_path_factory.mktemp("live2d-document"))


def _flatten(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    return tuple(value for point in points for value in point)


def test_document_compiles_complete_v400_sections_deterministically(
    document_plans,
) -> None:
    rig, bindings, coordinates, artmeshes, keyforms, document = document_plans

    assert validate_live2d_moc3_document(document, rig, bindings, coordinates, artmeshes, keyforms) is document
    assert document.counts[0] == len(artmeshes.parts)
    assert document.counts[1] == len(bindings.rotation_instances)
    assert document.counts[2] == 0
    assert document.counts[3] == len(bindings.rotation_instances)
    assert document.counts[4] == len(artmeshes.artmeshes)
    assert document.counts[5] == len(bindings.parameters)
    assert document.counts[20:] == (0, 0, 0)
    assert document.section("rotation_deformer_keyform.reflect_xs") == (False,) * document.counts[8]
    assert document.section("rotation_deformer_keyform.reflect_ys") == (False,) * document.counts[8]
    for name in document.sections:
        if ".runtime_space" in name:
            assert document.section(name) == ()

    payload = encode_moc3_v400(document)
    assert len(payload) <= 64 * 1024 * 1024
    assert decode_moc3_v400(payload) == document
    assert encode_moc3_v400(decode_moc3_v400(payload)) == payload
    assert validate_live2d_moc3_payload(payload, rig, bindings, coordinates, artmeshes, keyforms) == document
    report = build_live2d_structure_validation_report(payload, rig, bindings, coordinates, artmeshes, keyforms)
    assert report.moc_file_size == len(payload)
    assert report.counts == document.counts
    assert report.maximum_default_position_residual <= 0.1
    assert report.maximum_default_opacity_residual <= 1 / 255
    assert all(record.target_export_names for record in report.parameter_target_closure)
    assert build_live2d_moc3_document(rig, bindings, coordinates, artmeshes, keyforms) == document


def test_default_state_reconstructs_rest_and_nondefault_rotation_moves_vertices(
    document_plans,
) -> None:
    rig, bindings, coordinates, artmeshes, keyforms, document = document_plans

    default_state = evaluate_live2d_moc3_state(document)
    actual_by_name = {record.artmesh_id: record for record in default_state.artmeshes}
    for setup in artmeshes.artmeshes:
        expected = tuple(live2d_local_to_canvas(coordinates, setup.parent_instance_id, point) for point in setup.positions)
        actual = actual_by_name[setup.export_name]
        assert _flatten(actual.canvas_positions) == pytest.approx(_flatten(expected), abs=0.1)
        assert actual.opacity == pytest.approx(setup.setup_opacity, abs=1 / 255)
        assert actual.draw_order == setup.draw_order + 500

    head = next(parameter for parameter in bindings.parameters if parameter.control_id == "control/head_nod")
    moved_state = evaluate_live2d_moc3_state(document, {head.export_name: head.maximum})
    assert any(
        _flatten(before.canvas_positions) != pytest.approx(_flatten(after.canvas_positions), abs=0.1)
        for before, after in zip(default_state.artmeshes, moved_state.artmeshes, strict=True)
    )


def test_runtime_visibility_flags_do_not_encode_setup_opacity(
    document_plans,
) -> None:
    rig, bindings, coordinates, artmeshes, keyforms, _document = document_plans
    target_index = next(
        index for index, record in enumerate(keyforms.artmesh_keyforms) if record.parameter_id == "parameter/mouth_open_y"
    )
    target_mesh = artmeshes.artmeshes[target_index]
    target_part_index = next(index for index, part in enumerate(artmeshes.parts) if part.part_id == target_mesh.part_id)
    parts = list(artmeshes.parts)
    parts[target_part_index] = replace(parts[target_part_index], setup_opacity=0.0)
    meshes = list(artmeshes.artmeshes)
    meshes[target_index] = replace(meshes[target_index], setup_opacity=0.0)
    hidden_at_rest = replace(
        artmeshes,
        parts=tuple(parts),
        artmeshes=tuple(meshes),
    )

    document = _build_live2d_moc3_document(rig, bindings, coordinates, hidden_at_rest, keyforms)

    assert document.section("part.visibles")[target_part_index] is True
    assert document.section("art_mesh.visibles")[target_index] is True


@pytest.mark.parametrize(
    ("section_name", "replacement", "message"),
    (
        ("deformer.parent_deformer_indices", (999,), "parent_deformer"),
        ("rotation_deformer_keyform.reflect_xs", (True,), "reflect_xs"),
        ("parameter.default_values", (999.0,), "default_values"),
        ("uv.xys", (0.0,), "uv.xys"),
        ("art_mesh.texture_indices", (3,), "texture_indices"),
        ("art_mesh.visibles", (False,), "visibles"),
        ("art_mesh_keyform.draw_orders", (999.0,), "draw_orders"),
        ("keyform_position.xys", (12345.0,), "keyform_position"),
    ),
)
def test_structural_validator_rejects_mutated_sections(
    document_plans,
    section_name: str,
    replacement: tuple[object, ...],
    message: str,
) -> None:
    rig, bindings, coordinates, artmeshes, keyforms, document = document_plans
    sections = dict(document.sections)
    original = sections[section_name]
    sections[section_name] = replacement + original[len(replacement) :]
    broken = Moc3V400Document(
        counts=document.counts,
        canvas=document.canvas,
        sections=sections,
    )

    with pytest.raises(ValueError, match=message):
        validate_live2d_moc3_document(broken, rig, bindings, coordinates, artmeshes, keyforms)


def test_payload_validator_rejects_nonzero_writer_padding(document_plans) -> None:
    rig, bindings, coordinates, artmeshes, keyforms, document = document_plans
    payload = bytearray(encode_moc3_v400(document))
    envelope = parse_moc3_v400_envelope(bytes(payload))
    padding_offset = None
    for index, spec in enumerate(MOC3_V400_SECTION_SPECS):
        section_end = envelope.sot_offsets[spec.sot_index] + envelope.counts[spec.count_index] * moc3_element_size(
            spec.element_kind
        )
        next_start = (
            envelope.sot_offsets[MOC3_V400_SECTION_SPECS[index + 1].sot_index]
            if index + 1 < len(MOC3_V400_SECTION_SPECS)
            else len(payload)
        )
        if section_end < next_start:
            padding_offset = section_end
            break
    assert padding_offset is not None
    assert payload[padding_offset] == 0
    payload[padding_offset] = 1

    with pytest.raises(ValueError, match="padding"):
        validate_live2d_moc3_payload(bytes(payload), rig, bindings, coordinates, artmeshes, keyforms)


def test_document_validator_rejects_count_section_mismatch(document_plans) -> None:
    rig, bindings, coordinates, artmeshes, keyforms, document = document_plans
    counts = list(document.counts)
    counts[4] += 1
    broken = replace(document, counts=tuple(counts))

    with pytest.raises(ValueError, match="art_mesh"):
        validate_live2d_moc3_document(broken, rig, bindings, coordinates, artmeshes, keyforms)


def test_official_core_accepts_compiled_document(document_plans, tmp_path: Path) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not core_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH is not configured")
    _rig, bindings, _coordinates, artmeshes, _keyforms, document = document_plans
    moc_path = tmp_path / "model.moc3"
    moc_path.write_bytes(encode_moc3_v400(document))

    result = exercise_moc_with_core(core_path, moc_path, capture_model_state=True)
    assert result.consistency is True
    assert result.finite_vertices is True
    assert result.parameter_count == len(bindings.parameters)
    assert result.part_count == len(artmeshes.parts)
    assert result.drawable_count == len(artmeshes.artmeshes)


def test_official_core_can_reveal_an_artmesh_hidden_at_rest(document_plans, tmp_path: Path) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not core_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH is not configured")
    rig, bindings, coordinates, artmeshes, keyforms, _document = document_plans
    target_index = next(
        index for index, record in enumerate(keyforms.artmesh_keyforms) if record.parameter_id == "parameter/mouth_open_y"
    )
    target_keyforms = keyforms.artmesh_keyforms[target_index]
    target_mesh = artmeshes.artmeshes[target_index]
    target_part_index = next(index for index, part in enumerate(artmeshes.parts) if part.part_id == target_mesh.part_id)
    parameter = next(record for record in bindings.parameters if record.parameter_id == target_keyforms.parameter_id)
    opacities = tuple(0.0 if value == target_keyforms.parameter_default else 1.0 for value in target_keyforms.parameter_values)
    keyform_records = list(keyforms.artmesh_keyforms)
    keyform_records[target_index] = replace(
        target_keyforms,
        opacities=opacities,
        has_opacity=True,
    )
    parts = list(artmeshes.parts)
    parts[target_part_index] = replace(parts[target_part_index], setup_opacity=0.0)
    meshes = list(artmeshes.artmeshes)
    meshes[target_index] = replace(meshes[target_index], setup_opacity=0.0)
    hidden_at_rest = replace(
        artmeshes,
        parts=tuple(parts),
        artmeshes=tuple(meshes),
    )
    dynamic_keyforms = replace(
        keyforms,
        artmesh_keyforms=tuple(keyform_records),
    )
    document = _build_live2d_moc3_document(rig, bindings, coordinates, hidden_at_rest, dynamic_keyforms)
    moc_path = tmp_path / "hidden-at-rest.moc3"
    moc_path.write_bytes(encode_moc3_v400(document))

    default = exercise_moc_with_core(core_path, moc_path, capture_model_state=True).model_state
    revealed = exercise_moc_with_core(
        core_path,
        moc_path,
        parameter_values={parameter.export_name: parameter.maximum},
        capture_model_state=True,
    ).model_state
    assert default is not None
    assert revealed is not None
    default_drawable = next(record for record in default.drawables if record.id == target_mesh.export_name)
    revealed_drawable = next(record for record in revealed.drawables if record.id == target_mesh.export_name)

    assert default_drawable.opacity == pytest.approx(0.0, abs=1 / 255)
    assert revealed_drawable.opacity == pytest.approx(1.0, abs=1 / 255)
