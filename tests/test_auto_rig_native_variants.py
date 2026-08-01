from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

import module.auto_rig.native_variants as native_variants_module
from module.auto_rig.artifacts import sha256_file
from module.auto_rig.component_plan import build_mask_component_plan
from module.auto_rig.contracts import (
    AUTO_RIG_INPUT_CONTRACT_VERSION,
    AutoRigCanvasContract,
    AutoRigContractError,
    AutoRigInputContract,
    AutoRigPartContract,
    ValidatedPartSource,
)
from module.auto_rig.draw_order import build_ordinary_draw_order
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.mask_sources import LoadedPartAlpha
from module.auto_rig.native_variants import (
    NATIVE_VARIANT_MANIFEST_VERSION,
    load_native_variant_set,
)

_SHA256_ZERO = "sha256:" + "0" * 64


def _alpha(width: int, height: int, points: set[tuple[int, int]]) -> bytes:
    values = bytearray(width * height)
    for x, y in points:
        values[y * width + x] = 255
    return bytes(values)


def _base_part(
    *,
    source_tag: str,
    base_tag: str,
    part_id: str,
    side: str | None,
    bbox: tuple[int, int, int, int],
    points: set[tuple[int, int]],
    depth: float,
) -> LoadedPartAlpha:
    source = ValidatedPartSource(
        mode="png",
        color_path=Path("unused-color.png"),
        depth_path=Path("unused-depth.png"),
        color_sha256=_SHA256_ZERO,
        depth_sha256=_SHA256_ZERO,
        layer_name=None,
    )
    part = AutoRigPartContract(
        source_tag=source_tag,
        base_tag=base_tag,
        semantic_slug=base_tag.replace(" ", "-"),
        side=side,
        part_id=part_id,
        xyxy=bbox,
        depth_median=depth,
        source=source,
    )
    return LoadedPartAlpha(
        part=part,
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        alpha_u8=_alpha(bbox[2] - bbox[0], bbox[3] - bbox[1], points),
    )


def _context(root: Path, loaded: tuple[LoadedPartAlpha, ...]):
    component_plan = build_mask_component_plan(
        loaded,
        canvas_edge=1024,
        item_root=root,
    )
    draw_order = build_ordinary_draw_order(component_plan.parts)
    contract = AutoRigInputContract(
        schema_version=AUTO_RIG_INPUT_CONTRACT_VERSION,
        item_root=root,
        tag_version="v3",
        canvas=AutoRigCanvasContract(width=1024, height=1024, resolution=1024),
        save_to_psd=False,
        tblr_split=False,
        payload_mode="png",
        source_image_path=root / "src_img.png",
        source_image_sha256=_SHA256_ZERO,
        layerdiff_manifest_path=root / "layerdiff" / "manifest.json",
        optimized_manifest_path=root / "optimized" / "manifest.json",
        optimized_info_path=root / "optimized" / "info.json",
        parts=tuple(record.part for record in loaded),
    )
    return contract, component_plan, draw_order


def _mouth_context(root: Path):
    mouth = _base_part(
        source_tag="mouth",
        base_tag="mouth",
        part_id="part/mouth",
        side=None,
        bbox=(100, 120, 104, 124),
        points={(0, 0), (1, 0), (0, 1), (1, 1)},
        depth=0.1,
    )
    return _context(root, (mouth,))


def _eye_context(root: Path):
    loaded: list[LoadedPartAlpha] = []
    for index, base_tag in enumerate(("eyewhite", "irides", "eyelash")):
        loaded.append(
            _base_part(
                source_tag=base_tag,
                base_tag=base_tag,
                part_id=f"part/{base_tag}",
                side=None,
                bbox=(100, 100 + index * 10, 112, 104 + index * 10),
                points={
                    (0, 0), (1, 0), (0, 1), (1, 1),
                    (9, 1), (10, 1), (9, 2), (10, 2),
                },
                depth=0.2 + index * 0.01,
            )
        )
    return _context(root, tuple(loaded))


def _entry(
    *,
    variant_id: str,
    role: str,
    base_part_ids: list[str],
    anchor: str,
    xyxy: tuple[int, int, int, int] = (10, 20, 14, 24),
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "semantic_role": role,
        "composite_mode": "occluding_overlay_v1",
        "base_part_ids": base_part_ids,
        "draw_anchor_part_id": anchor,
        "xyxy": list(xyxy),
        "path": f"{variant_id}.png",
        "rgba_mode": "RGBA",
        "alpha_mode": "straight",
        "color_space": "sRGB",
        "file_sha256": _SHA256_ZERO,
    }


def _write_manifest(
    root: Path,
    entries: list[dict[str, object]],
    *,
    indent: int | None = 2,
) -> None:
    directory = root / "rig_inputs" / "variants"
    directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = directory / str(entry["path"])
        if not path.exists():
            bbox = entry["xyxy"]
            size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            Image.new("RGBA", size, (120, 80, 60, 255)).save(path)
        entry["file_sha256"] = sha256_file(path)
    payload = {"schema_version": NATIVE_VARIANT_MANIFEST_VERSION, "variants": entries}
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=indent, allow_nan=True),
        encoding="utf-8",
    )


def _load(context):
    return load_native_variant_set(*context)


def test_missing_directory_and_empty_manifest_have_the_same_empty_semantic_set(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    empty_root = tmp_path / "empty"
    missing_root.mkdir()
    empty_root.mkdir()
    missing_context = _mouth_context(missing_root)
    empty_context = _mouth_context(empty_root)
    _write_manifest(empty_root, [])

    missing = _load(missing_context)
    empty = _load(empty_context)

    assert missing.entries == empty.entries == ()
    assert missing.native_variant_set_sha256 == empty.native_variant_set_sha256
    assert missing.present is False
    assert empty.present is True


def test_loader_rejects_component_plan_from_a_different_canvas(tmp_path: Path) -> None:
    contract, component_plan, draw_order = _mouth_context(tmp_path)
    mismatched = replace(component_plan, canvas_edge=768)

    with pytest.raises(AutoRigContractError, match="canvas"):
        load_native_variant_set(contract, mismatched, draw_order)


def test_loader_rejects_a_component_plan_that_preoccupies_the_native_namespace(
    tmp_path: Path,
) -> None:
    contract, component_plan, _ = _mouth_context(tmp_path)
    original = component_plan.parts[0]
    native_part_id = "part/native.mouth_open"
    native_components = tuple(
        replace(component, part_id=native_part_id) for component in original.components
    )
    native_part = replace(
        original,
        part_id=native_part_id,
        components=native_components,
    )
    provisional = replace(component_plan, parts=(native_part,), plan_sha256=_SHA256_ZERO)
    occupied = replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    draw_order = build_ordinary_draw_order(occupied.parts)

    with pytest.raises(AutoRigContractError, match="already contains variants"):
        load_native_variant_set(contract, occupied, draw_order)


def test_manifest_decodes_rgba_and_semantic_digest_ignores_json_formatting_and_order(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_context = _mouth_context(first_root)
    second_context = _mouth_context(second_root)
    smile = _entry(
        variant_id="mouth_smile",
        role="mouth_smile",
        base_part_ids=["part/mouth"],
        anchor="part/mouth",
    )
    opened = _entry(
        variant_id="mouth_open",
        role="mouth_open",
        base_part_ids=["part/mouth"],
        anchor="part/mouth",
    )
    _write_manifest(first_root, [smile, opened], indent=2)
    _write_manifest(second_root, [dict(reversed(tuple(opened.items()))), dict(reversed(tuple(smile.items())))], indent=None)

    first = _load(first_context)
    second = _load(second_context)

    assert tuple(entry.variant_id for entry in first.entries) == ("mouth_open", "mouth_smile")
    assert first.native_variant_set_sha256 == second.native_variant_set_sha256
    assert first.entries[0].alpha_u8 == bytes((255,)) * 16
    assert first.entries[0].part_id == "part/native.mouth_open"


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("invalid-id", "variant_id"),
        ("reserved-id", "reserved"),
        ("wrong-path", "path"),
        ("unknown-role", "semantic_role"),
        ("unknown-field", "fields"),
        ("wrong-sha", "SHA"),
        ("wrong-size", "dimensions"),
        ("wrong-mode", "RGBA"),
        ("transparent", "alpha"),
        ("extra-file", "inventory"),
    ),
)
def test_manifest_rejects_invalid_identity_path_or_png(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    context = _mouth_context(tmp_path)
    entry = _entry(
        variant_id="mouth_open",
        role="mouth_open",
        base_part_ids=["part/mouth"],
        anchor="part/mouth",
    )
    if mutation == "invalid-id":
        entry["variant_id"] = "Bad ID"
        entry["path"] = "Bad ID.png"
    elif mutation == "reserved-id":
        entry["variant_id"] = "con"
        entry["path"] = "con.png"
    elif mutation == "wrong-path":
        entry["path"] = "../mouth_open.png"
    elif mutation == "unknown-role":
        entry["semantic_role"] = "mouth_unknown"
    elif mutation == "unknown-field":
        entry["extra"] = True
    _write_manifest(tmp_path, [entry])
    directory = tmp_path / "rig_inputs" / "variants"
    png = directory / str(entry["path"])
    if mutation == "wrong-sha":
        payload = json.loads((directory / "manifest.json").read_text())
        payload["variants"][0]["file_sha256"] = _SHA256_ZERO
        (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong-size":
        Image.new("RGBA", (3, 4), (1, 2, 3, 255)).save(png)
        payload = json.loads((directory / "manifest.json").read_text())
        payload["variants"][0]["file_sha256"] = sha256_file(png)
        (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong-mode":
        Image.new("RGB", (4, 4), (1, 2, 3)).save(png)
        payload = json.loads((directory / "manifest.json").read_text())
        payload["variants"][0]["file_sha256"] = sha256_file(png)
        (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "transparent":
        Image.new("RGBA", (4, 4), (1, 2, 3, 0)).save(png)
        payload = json.loads((directory / "manifest.json").read_text())
        payload["variants"][0]["file_sha256"] = sha256_file(png)
        (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "extra-file":
        (directory / "stale.png").write_bytes(b"stale")

    with pytest.raises(AutoRigContractError, match=match):
        _load(context)


def test_manifest_rejects_duplicate_ids_roles_and_base_ids(tmp_path: Path) -> None:
    context = _mouth_context(tmp_path)
    first = _entry(
        variant_id="mouth_a",
        role="mouth_open",
        base_part_ids=["part/mouth"],
        anchor="part/mouth",
    )
    second = _entry(
        variant_id="mouth_b",
        role="mouth_open",
        base_part_ids=["part/mouth", "part/mouth"],
        anchor="part/mouth",
    )
    _write_manifest(tmp_path, [first, second])

    with pytest.raises(AutoRigContractError, match="duplicate"):
        _load(context)


def test_manifest_rejects_duplicate_base_ids_before_normalization(tmp_path: Path) -> None:
    context = _mouth_context(tmp_path)
    entry = _entry(
        variant_id="mouth_open",
        role="mouth_open",
        base_part_ids=["part/mouth", "part/mouth"],
        anchor="part/mouth",
    )
    _write_manifest(tmp_path, [entry])

    with pytest.raises(AutoRigContractError, match="base_part_ids contains duplicate"):
        _load(context)


def test_manifest_rejects_duplicate_json_key_and_nonfinite_number(tmp_path: Path) -> None:
    context = _mouth_context(tmp_path)
    directory = tmp_path / "rig_inputs" / "variants"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        '{"schema_version":"native-variant-manifest-v1","variants":[],"variants":[]}',
        encoding="utf-8",
    )
    with pytest.raises(AutoRigContractError, match="duplicate"):
        _load(context)

    (directory / "manifest.json").write_text(
        '{"schema_version":"native-variant-manifest-v1","variants":[NaN]}',
        encoding="utf-8",
    )
    with pytest.raises(AutoRigContractError, match="non-finite"):
        _load(context)


def test_manifest_rejects_symlinked_png_before_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _mouth_context(tmp_path)
    entry = _entry(
        variant_id="mouth_open",
        role="mouth_open",
        base_part_ids=["part/mouth"],
        anchor="part/mouth",
    )
    _write_manifest(tmp_path, [entry])
    png = tmp_path / "rig_inputs" / "variants" / "mouth_open.png"
    external = tmp_path / "external.png"
    png.replace(external)
    try:
        os.symlink(external, png)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    def unexpected_read(path: Path) -> str:
        raise AssertionError(f"symlink was read before rejection: {path}")

    monkeypatch.setattr(native_variants_module, "sha256_file", unexpected_read)

    with pytest.raises(AutoRigContractError, match="symlink|reparse"):
        _load(context)


def test_eye_and_mouth_roles_require_registry_derived_base_sets_and_front_anchor(
    tmp_path: Path,
) -> None:
    eye_root = tmp_path / "eyes"
    mouth_root = tmp_path / "mouth"
    eye_root.mkdir()
    mouth_root.mkdir()
    eye_context = _eye_context(eye_root)
    mouth_context = _mouth_context(mouth_root)
    eye_parts = eye_context[1].parts
    xmin_bases = sorted(
        part.part_id
        for part in eye_parts
        if part.side == "xmin" and part.base_tag in {"eyewhite", "irides", "eyelash"}
    )
    eye_rank = {part_id: rank for rank, part_id in enumerate(eye_context[2].ordinary_part_order)}
    eye_anchor = max(xmin_bases, key=eye_rank.__getitem__)
    eye_entry = _entry(
        variant_id="blink_xmin",
        role="eye_closed.xmin",
        base_part_ids=xmin_bases,
        anchor=eye_anchor,
    )
    mouth_entry = _entry(
        variant_id="mouth_open",
        role="mouth_open",
        base_part_ids=["part/mouth"],
        anchor="part/mouth",
    )
    _write_manifest(eye_root, [eye_entry])
    _write_manifest(mouth_root, [mouth_entry])

    eye = _load(eye_context).entries[0]
    mouth = _load(mouth_context).entries[0]

    assert eye.base_part_ids == tuple(xmin_bases)
    assert eye.draw_anchor_part_id == eye_anchor
    assert eye.anchor_base_tag == "eyelash"
    assert mouth.anchor_base_tag == "mouth"


def test_coupled_eye_role_requires_every_existing_eye_family_part(tmp_path: Path) -> None:
    context = _eye_context(tmp_path)
    parts = context[1].parts
    rank = {part_id: index for index, part_id in enumerate(context[2].ordinary_part_order)}
    bases = sorted(
        part.part_id for part in parts if part.base_tag in {"eyewhite", "irides", "eyelash"}
    )
    _write_manifest(
        tmp_path,
        [
            _entry(
                variant_id="blink_coupled",
                role="eye_closed.coupled",
                base_part_ids=bases,
                anchor=max(bases, key=rank.__getitem__),
            )
        ],
    )

    candidate = _load(context).entries[0]

    assert candidate.base_part_ids == tuple(bases)
    part_by_id = {part.part_id: part for part in parts}
    assert {part_by_id[part_id].side for part_id in bases} == {"xmin", "xmax"}


@pytest.mark.parametrize(
    "mutation",
    ("missing-base", "extra-base", "unknown-base", "anchor-outside", "anchor-not-front"),
)
def test_role_reference_contract_rejects_inexact_base_or_anchor(
    tmp_path: Path,
    mutation: str,
) -> None:
    context = _eye_context(tmp_path)
    parts = context[1].parts
    bases = sorted(
        part.part_id
        for part in parts
        if part.side == "xmin" and part.base_tag in {"eyewhite", "irides", "eyelash"}
    )
    rank = {part_id: index for index, part_id in enumerate(context[2].ordinary_part_order)}
    anchor = max(bases, key=rank.__getitem__)
    if mutation == "missing-base":
        bases = bases[:-1]
    elif mutation == "extra-base":
        bases = [*bases, next(part.part_id for part in parts if part.side == "xmax")]
    elif mutation == "unknown-base":
        bases = [*bases, "part/not-there"]
    elif mutation == "anchor-outside":
        anchor = next(part.part_id for part in parts if part.side == "xmax")
    elif mutation == "anchor-not-front":
        anchor = min(bases, key=rank.__getitem__)
    entry = _entry(
        variant_id="blink_xmin",
        role="eye_closed.xmin",
        base_part_ids=bases,
        anchor=anchor,
    )
    _write_manifest(tmp_path, [entry])

    with pytest.raises(AutoRigContractError):
        _load(context)


def test_coupled_and_single_eye_roles_cannot_overlap(tmp_path: Path) -> None:
    context = _eye_context(tmp_path)
    parts = context[1].parts
    rank = {part_id: index for index, part_id in enumerate(context[2].ordinary_part_order)}
    all_bases = sorted(
        part.part_id for part in parts if part.base_tag in {"eyewhite", "irides", "eyelash"}
    )
    xmin_bases = sorted(part.part_id for part in parts if part.side == "xmin")
    entries = [
        _entry(
            variant_id="blink_coupled",
            role="eye_closed.coupled",
            base_part_ids=all_bases,
            anchor=max(all_bases, key=rank.__getitem__),
        ),
        _entry(
            variant_id="blink_xmin",
            role="eye_closed.xmin",
            base_part_ids=xmin_bases,
            anchor=max(xmin_bases, key=rank.__getitem__),
        ),
    ]
    _write_manifest(tmp_path, entries)

    with pytest.raises(AutoRigContractError, match="overlap"):
        _load(context)


def test_incomplete_blink_or_mouth_form_bundle_remains_a_valid_candidate_set(
    tmp_path: Path,
) -> None:
    eye_root = tmp_path / "eye"
    mouth_root = tmp_path / "mouth"
    eye_root.mkdir()
    mouth_root.mkdir()
    eye_context = _eye_context(eye_root)
    mouth_context = _mouth_context(mouth_root)
    eye_parts = eye_context[1].parts
    eye_rank = {part_id: index for index, part_id in enumerate(eye_context[2].ordinary_part_order)}
    xmin_bases = sorted(part.part_id for part in eye_parts if part.side == "xmin")
    _write_manifest(
        eye_root,
        [
            _entry(
                variant_id="blink_xmin",
                role="eye_closed.xmin",
                base_part_ids=xmin_bases,
                anchor=max(xmin_bases, key=eye_rank.__getitem__),
            )
        ],
    )
    _write_manifest(
        mouth_root,
        [
            _entry(
                variant_id="mouth_smile",
                role="mouth_smile",
                base_part_ids=["part/mouth"],
                anchor="part/mouth",
            )
        ],
    )

    assert len(_load(eye_context).entries) == 1
    assert len(_load(mouth_context).entries) == 1
