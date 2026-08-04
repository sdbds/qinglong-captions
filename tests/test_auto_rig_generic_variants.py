from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from module.auto_rig.artifacts import sha256_file
from module.auto_rig.component_plan import build_base_mask_component_plan
from module.auto_rig.contracts import (
    AUTO_RIG_INPUT_CONTRACT_VERSION,
    AutoRigCanvasContract,
    AutoRigInputContract,
    AutoRigPartContract,
    ValidatedPartSource,
)
from module.auto_rig.draw_order import build_ordinary_draw_order
from module.auto_rig.generic_variants import (
    GENERIC_VARIANT_SYNTHESIS_VERSION,
    build_generic_variant_synthesis_plan,
)
from module.auto_rig.mask_sources import load_validated_part_alphas
from module.auto_rig.native_variants import make_native_variant_set
from module.auto_rig.texture_sources import load_base_texture_regions


def _part(
    root: Path,
    *,
    source_tag: str,
    base_tag: str,
    part_id: str,
    side: str | None,
    xyxy: tuple[int, int, int, int],
    image: Image.Image,
    depth: float,
) -> AutoRigPartContract:
    color = root / f"{part_id.removeprefix('part/').replace('/', '_')}.png"
    depth_path = root / f"{part_id.removeprefix('part/').replace('/', '_')}.depth"
    image.save(color, format="PNG", compress_level=9, optimize=False)
    depth_path.write_bytes(b"depth")
    return AutoRigPartContract(
        source_tag=source_tag,
        base_tag=base_tag,
        semantic_slug=source_tag.replace(".", "-"),
        side=side,
        part_id=part_id,
        xyxy=xyxy,
        depth_median=depth,
        source=ValidatedPartSource(
            mode="png",
            color_path=color,
            depth_path=depth_path,
            color_sha256=sha256_file(color),
            depth_sha256=sha256_file(depth_path),
            layer_name=None,
        ),
    )


def _rgba(size: tuple[int, int], color: tuple[int, int, int, int]) -> Image.Image:
    return Image.new("RGBA", size, color)


def _fixture(
    root: Path,
    *,
    mouth_image: Image.Image | None = None,
) -> AutoRigInputContract:
    root.mkdir(parents=True, exist_ok=True)
    parts: list[AutoRigPartContract] = []
    parts.append(
        _part(
            root,
            source_tag="face",
            base_tag="face",
            part_id="part/face",
            side=None,
            xyxy=(16, 8, 112, 120),
            image=_rgba((96, 112), (244, 205, 190, 255)),
            depth=0.50,
        )
    )
    for side, x1 in (("xmin", 32), ("xmax", 76)):
        eye = _rgba((20, 10), (248, 248, 248, 255))
        iris = _rgba((6, 7), (80, 130, 170, 255))
        lash = _rgba((22, 3), (42, 28, 38, 255))
        for base_tag, image, xyxy, depth in (
            ("eyewhite", eye, (x1, 42, x1 + 20, 52), 0.30),
            ("irides", iris, (x1 + 7, 44, x1 + 13, 51), 0.25),
            ("eyelash", lash, (x1 - 1, 40, x1 + 21, 43), 0.20),
        ):
            parts.append(
                _part(
                    root,
                    source_tag=f"{base_tag}-{side}",
                    base_tag=base_tag,
                    part_id=f"part/{base_tag}.{side}",
                    side=side,
                    xyxy=xyxy,
                    image=image,
                    depth=depth,
                )
            )
        brow = _rgba((18, 3), (55, 35, 45, 255))
        parts.append(
            _part(
                root,
                source_tag=f"eyebrow-{side}",
                base_tag="eyebrow",
                part_id=f"part/eyebrow.{side}",
                side=side,
                xyxy=(x1 + 1, 32, x1 + 19, 35),
                image=brow,
                depth=0.18,
            )
        )
    mouth = mouth_image or Image.new("RGBA", (24, 12), (0, 0, 0, 0))
    if mouth_image is None:
        draw = ImageDraw.Draw(mouth)
        draw.ellipse((2, 1, 21, 10), fill=(105, 30, 45, 255))
        draw.ellipse((6, 3, 17, 8), fill=(40, 12, 18, 0))
    parts.append(
        _part(
            root,
            source_tag="mouth",
            base_tag="mouth",
            part_id="part/mouth",
            side=None,
            xyxy=(52, 72, 76, 84),
            image=mouth,
            depth=0.15,
        )
    )
    placeholder = root / "placeholder"
    return AutoRigInputContract(
        schema_version=AUTO_RIG_INPUT_CONTRACT_VERSION,
        item_root=root,
        tag_version="v3",
        canvas=AutoRigCanvasContract(width=128, height=128, resolution=128),
        save_to_psd=False,
        tblr_split=True,
        payload_mode="png",
        source_image_path=placeholder,
        source_image_sha256="sha256:" + "0" * 64,
        layerdiff_manifest_path=placeholder,
        optimized_manifest_path=placeholder,
        optimized_info_path=placeholder,
        parts=tuple(parts),
    )


def _build(root: Path, authored=(), *, mouth_image: Image.Image | None = None):
    contract = _fixture(root, mouth_image=mouth_image)
    load_validated_part_alphas(contract)
    components = build_base_mask_component_plan(contract)
    draw = build_ordinary_draw_order(components.parts)
    authored_set = make_native_variant_set(
        present=bool(authored),
        manifest_path=None,
        entries=tuple(authored),
    )
    return build_generic_variant_synthesis_plan(
        contract,
        components,
        draw,
        load_base_texture_regions(contract),
        authored_set,
        item_root=root,
    )


def test_generic_variants_are_deterministic_and_use_crossfade_for_open_closed(
    tmp_path: Path,
) -> None:
    first_set, first_plan = _build(tmp_path / "first")
    second_set, second_plan = _build(tmp_path / "second")

    assert first_plan.schema_version == GENERIC_VARIANT_SYNTHESIS_VERSION
    assert first_plan.plan_sha256 == second_plan.plan_sha256
    by_role = {entry.semantic_role: entry for entry in first_set.entries}
    assert set(by_role) == {
        "eye_closed.xmin",
        "eye_closed.xmax",
        "mouth_closed",
        "mouth_smile",
        "mouth_frown",
    }
    assert by_role["eye_closed.xmin"].composite_mode == "crossfade_overlay_v1"
    assert by_role["eye_closed.xmax"].composite_mode == "crossfade_overlay_v1"
    assert by_role["mouth_closed"].composite_mode == "crossfade_overlay_v1"
    assert by_role["mouth_smile"].composite_mode == "occluding_overlay_v1"
    assert by_role["mouth_frown"].composite_mode == "occluding_overlay_v1"
    assert all(entry.source_kind == "generated" for entry in first_set.entries)
    assert [entry.file_sha256 for entry in first_set.entries] == [entry.file_sha256 for entry in second_set.entries]


def test_closed_curved_mouth_is_not_misclassified_as_an_open_mouth(
    tmp_path: Path,
) -> None:
    mouth = Image.new("RGBA", (24, 12), (0, 0, 0, 0))
    ImageDraw.Draw(mouth).line(
        ((2, 5), (7, 4), (12, 3), (17, 4), (21, 5)),
        fill=(105, 30, 45, 255),
        width=2,
    )

    variants, plan = _build(tmp_path, mouth_image=mouth)

    assert plan.mouth_state_observation == "closed_or_ambiguous"
    assert "mouth_closed" not in {entry.semantic_role for entry in variants.entries}
    mouth_record = next(record for record in plan.records if record.semantic_role == "mouth_closed")
    assert mouth_record.status == "unavailable"
    assert mouth_record.reason == "mouth_base_not_open"


def test_authored_role_wins_and_generation_fills_only_missing_roles(tmp_path: Path) -> None:
    generated, _ = _build(tmp_path / "seed")
    authored_eye = replace(
        next(entry for entry in generated.entries if entry.semantic_role == "eye_closed.xmin"),
        source_kind="authored",
        generator_version=None,
    )

    merged, plan = _build(tmp_path / "merged", authored=(authored_eye,))

    xmin = next(entry for entry in merged.entries if entry.semantic_role == "eye_closed.xmin")
    assert xmin.source_kind == "authored"
    assert {record.semantic_role: record.status for record in plan.records}["eye_closed.xmin"] == "authored_override"
    assert len({entry.semantic_role for entry in merged.entries}) == len(merged.entries)


def test_generated_mouth_forms_keep_a_visible_curve_after_runtime_downscale(
    tmp_path: Path,
) -> None:
    variants, _plan = _build(tmp_path)
    by_role = {entry.semantic_role: entry for entry in variants.entries}
    face_color = (244, 205, 190)

    for role in ("mouth_smile", "mouth_frown"):
        image = Image.open(by_role[role].png_path).convert("RGB")
        centerline: list[float | None] = []
        for x in range(image.width):
            ink_rows = [
                y
                for y in range(image.height)
                if max(abs(channel - background) for channel, background in zip(image.getpixel((x, y)), face_color, strict=True))
                > 30
            ]
            centerline.append(sum(ink_rows) / len(ink_rows) if ink_rows else None)
        endpoints = [
            value
            for index, value in enumerate(centerline)
            if value is not None and (0.08 <= index / image.width <= 0.25 or 0.75 <= index / image.width <= 0.92)
        ]
        center = [value for index, value in enumerate(centerline) if value is not None and 0.4 <= index / image.width <= 0.6]
        curve_amplitude = abs(sum(center) / len(center) - sum(endpoints) / len(endpoints))

        assert curve_amplitude >= image.height * 0.25
