from __future__ import annotations

from collections.abc import Mapping

from .frame_kernel import apply_rotation_stack
from .moc3 import Moc3CanvasInfo
from .moc3_codec import Moc3V400Document, empty_v400_sections, encode_moc3_v400
from .uv_kernel import canonical_top_left_to_core_api_uv, canonical_top_left_to_moc_uv

STATIC_E0_CANVAS_SIZE = (1024.0, 768.0)
STATIC_E0_ROOT_VERTICES = (
    (-0.34375, 0.25),
    (0.1875, 0.28125),
    (0.28125, -0.15625),
    (-0.28125, -0.21875),
)
STATIC_E0_MOC_VERTICES = tuple((x, -y) for x, y in STATIC_E0_ROOT_VERTICES)
STATIC_E0_UVS = (
    (0.125, 0.25),
    (0.75, 0.125),
    (0.875, 0.75),
    (0.25, 0.875),
)
STATIC_E0_MOC_UVS = tuple(canonical_top_left_to_moc_uv(uv) for uv in STATIC_E0_UVS)
STATIC_E0_CORE_UVS = tuple(canonical_top_left_to_core_api_uv(uv) for uv in STATIC_E0_UVS)

_STATIC_E0_COUNTS = (1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 8, 1, 2, 1, 1, 8, 6, 1, 1, 1, 0, 0, 0)

DEFORMER_E0_CANVAS_SIZE = (1024.0, 768.0)
DEFORMER_E0_DIRECT_LOCAL_VERTICES = (
    (0.15, 0.20),
    (0.45, 0.15),
    (0.48, 0.45),
    (0.12, 0.50),
)
DEFORMER_E0_NESTED_LOCAL_VERTICES = (
    (-42.0, -28.0),
    (58.0, -22.0),
    (64.0, 36.0),
    (-48.0, 42.0),
)
DEFORMER_E0_UVS = (
    (0.125, 0.75),
    (0.75, 0.875),
    (0.875, 0.25),
    (0.25, 0.125),
)

_DEFORMER_E0_COUNTS = (1, 3, 1, 2, 2, 4, 1, 3, 6, 2, 40, 5, 9, 5, 11, 16, 12, 1, 1, 2, 0, 0, 0)
_DEFORMER_E0_PARAMETER_DEFAULTS = {
    "ParamStatic": 0.0,
    "ParamBreath": 0.0,
    "ParamOuter": 0.0,
    "ParamInner": 6.0,
}
_DEFORMER_E0_WARP_GRIDS_CANVAS = (
    (
        (180.0, 140.0),
        (820.0, 140.0),
        (180.0, 650.0),
        (820.0, 650.0),
    ),
    (
        (180.0, 120.0),
        (820.0, 120.0),
        (180.0, 660.0),
        (820.0, 660.0),
    ),
    (
        (180.0, 100.0),
        (820.0, 100.0),
        (180.0, 680.0),
        (820.0, 680.0),
    ),
)
_DEFORMER_E0_OUTER_ORIGINS = ((0.28, 0.42), (0.30, 0.40), (0.34, 0.37))
_DEFORMER_E0_INNER_ORIGIN = (100.0, 60.0)


def build_static_e0_document() -> Moc3V400Document:
    sections = empty_v400_sections()
    sections.update(
        {
            "part.ids": ("PartRoot",),
            "part.keyform_binding_band_indices": (1,),
            "part.keyform_begin_indices": (0,),
            "part.keyform_counts": (1,),
            "part.visibles": (True,),
            "part.enables": (True,),
            "part.parent_part_indices": (-1,),
            "art_mesh.ids": ("ArtMeshFixture",),
            "art_mesh.keyform_binding_band_indices": (0,),
            "art_mesh.keyform_begin_indices": (0,),
            "art_mesh.keyform_counts": (1,),
            "art_mesh.visibles": (True,),
            "art_mesh.enables": (True,),
            "art_mesh.parent_part_indices": (0,),
            "art_mesh.parent_deformer_indices": (-1,),
            "art_mesh.texture_indices": (0,),
            "art_mesh.drawable_flags": (4,),
            "art_mesh.position_index_counts": (4,),
            "art_mesh.uv_begin_indices": (0,),
            "art_mesh.position_index_begin_indices": (0,),
            "art_mesh.vertex_counts": (6,),
            "art_mesh.mask_begin_indices": (0,),
            "art_mesh.mask_counts": (0,),
            "parameter.ids": ("ParamOpacity",),
            "parameter.max_values": (1.0,),
            "parameter.min_values": (0.0,),
            "parameter.default_values": (1.0,),
            "parameter.repeats": (False,),
            "parameter.decimal_places": (1,),
            "parameter.keyform_binding_begin_indices": (0,),
            "parameter.keyform_binding_counts": (1,),
            "part_keyform.draw_orders": (500.0,),
            "art_mesh_keyform.opacities": (1.0,),
            "art_mesh_keyform.draw_orders": (500.0,),
            "art_mesh_keyform.keyform_position_begin_indices": (0,),
            "keyform_position.xys": tuple(
                coordinate for point in STATIC_E0_MOC_VERTICES for coordinate in point
            ),
            "keyform_binding_index.indices": (0,),
            "keyform_binding_band.begin_indices": (0, 0),
            "keyform_binding_band.counts": (1, 0),
            "keyform_binding.keys_begin_indices": (0,),
            "keyform_binding.keys_counts": (1,),
            "keys.values": (1.0,),
            "uv.xys": tuple(coordinate for point in STATIC_E0_MOC_UVS for coordinate in point),
            "position_index.indices": (0, 1, 2, 0, 2, 3),
            "drawable_mask.art_mesh_indices": (-1,),
            "draw_order_group.object_begin_indices": (0,),
            "draw_order_group.object_counts": (1,),
            "draw_order_group.object_total_counts": (1,),
            "draw_order_group.min_draw_orders": (1000,),
            "draw_order_group.max_draw_orders": (200,),
            "draw_order_group_object.types": (0,),
            "draw_order_group_object.indices": (0,),
            "draw_order_group_object.group_indices": (-1,),
        }
    )
    return Moc3V400Document(
        counts=_STATIC_E0_COUNTS,
        canvas=Moc3CanvasInfo(
            pixels_per_unit=1024.0,
            origin_x=512.0,
            origin_y=384.0,
            width=STATIC_E0_CANVAS_SIZE[0],
            height=STATIC_E0_CANVAS_SIZE[1],
            flag=0,
        ),
        sections=sections,
    )


def build_static_e0_moc() -> bytes:
    return encode_moc3_v400(build_static_e0_document())


def _canvas_to_moc_root(point: tuple[float, float]) -> tuple[float, float]:
    width, height = DEFORMER_E0_CANVAS_SIZE
    ppu = max(width, height)
    return ((point[0] - width / 2.0) / ppu, (point[1] - height / 2.0) / ppu)


def _flatten(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    return tuple(coordinate for point in points for coordinate in point)


def build_deformer_e0_document() -> Moc3V400Document:
    sections = empty_v400_sections()
    warp_positions = tuple(
        coordinate
        for grid in _DEFORMER_E0_WARP_GRIDS_CANVAS
        for coordinate in _flatten(tuple(_canvas_to_moc_root(point) for point in grid))
    )
    sections.update(
        {
            "part.ids": ("PartRoot",),
            "part.keyform_binding_band_indices": (2,),
            "part.keyform_begin_indices": (0,),
            "part.keyform_counts": (1,),
            "part.visibles": (True,),
            "part.enables": (True,),
            "part.parent_part_indices": (-1,),
            "deformer.ids": ("WarpBreath", "RotationOuter", "RotationInner"),
            "deformer.keyform_binding_band_indices": (3, 4, 5),
            "deformer.visibles": (True, True, True),
            "deformer.enables": (True, True, True),
            "deformer.parent_part_indices": (0, 0, 0),
            "deformer.parent_deformer_indices": (-1, 0, 1),
            "deformer.types": (0, 1, 1),
            "deformer.specific_indices": (0, 0, 1),
            "warp_deformer.keyform_binding_band_indices": (6,),
            "warp_deformer.keyform_begin_indices": (0,),
            "warp_deformer.keyform_counts": (3,),
            "warp_deformer.vertex_counts": (4,),
            "warp_deformer.rows": (1,),
            "warp_deformer.cols": (1,),
            "rotation_deformer.keyform_binding_band_indices": (7, 8),
            "rotation_deformer.keyform_begin_indices": (0, 3),
            "rotation_deformer.keyform_counts": (3, 3),
            "rotation_deformer.base_angles": (0.0, 0.0),
            "art_mesh.ids": ("ArtMeshDirect", "ArtMeshNested"),
            "art_mesh.keyform_binding_band_indices": (0, 1),
            "art_mesh.keyform_begin_indices": (0, 1),
            "art_mesh.keyform_counts": (1, 1),
            "art_mesh.visibles": (True, True),
            "art_mesh.enables": (True, True),
            "art_mesh.parent_part_indices": (0, 0),
            "art_mesh.parent_deformer_indices": (0, 2),
            "art_mesh.texture_indices": (0, 0),
            "art_mesh.drawable_flags": (4, 4),
            "art_mesh.position_index_counts": (4, 4),
            "art_mesh.uv_begin_indices": (0, 8),
            "art_mesh.position_index_begin_indices": (0, 6),
            "art_mesh.vertex_counts": (6, 6),
            "art_mesh.mask_begin_indices": (0, 0),
            "art_mesh.mask_counts": (0, 0),
            "parameter.ids": tuple(_DEFORMER_E0_PARAMETER_DEFAULTS),
            "parameter.max_values": (1.0, 1.0, 20.0, 30.0),
            "parameter.min_values": (0.0, -1.0, -20.0, -30.0),
            "parameter.default_values": tuple(_DEFORMER_E0_PARAMETER_DEFAULTS.values()),
            "parameter.repeats": (False,) * 4,
            "parameter.decimal_places": (1,) * 4,
            "parameter.keyform_binding_begin_indices": (0, 2, 3, 4),
            "parameter.keyform_binding_counts": (2, 1, 1, 1),
            "part_keyform.draw_orders": (500.0,),
            "warp_deformer_keyform.opacities": (1.0, 1.0, 1.0),
            "warp_deformer_keyform.keyform_position_begin_indices": (16, 24, 32),
            "rotation_deformer_keyform.opacities": (1.0,) * 6,
            "rotation_deformer_keyform.angles": (-20.0, 0.0, 20.0, -30.0, 0.0, 30.0),
            "rotation_deformer_keyform.origin_xs": (
                *(origin[0] for origin in _DEFORMER_E0_OUTER_ORIGINS),
                *(_DEFORMER_E0_INNER_ORIGIN[0] for _ in range(3)),
            ),
            "rotation_deformer_keyform.origin_ys": (
                *(origin[1] for origin in _DEFORMER_E0_OUTER_ORIGINS),
                *(_DEFORMER_E0_INNER_ORIGIN[1] for _ in range(3)),
            ),
            "rotation_deformer_keyform.scales": (1.0 / 640.0,) * 3 + (1.0,) * 3,
            "rotation_deformer_keyform.reflect_xs": (False,) * 6,
            "rotation_deformer_keyform.reflect_ys": (False,) * 6,
            "art_mesh_keyform.opacities": (1.0, 1.0),
            "art_mesh_keyform.draw_orders": (500.0, 500.0),
            "art_mesh_keyform.keyform_position_begin_indices": (0, 8),
            "keyform_position.xys": (
                *_flatten(DEFORMER_E0_DIRECT_LOCAL_VERTICES),
                *_flatten(DEFORMER_E0_NESTED_LOCAL_VERTICES),
                *warp_positions,
            ),
            "keyform_binding_index.indices": (0, 1, 2, 3, 4),
            "keyform_binding_band.begin_indices": (0, 1, 0, 0, 0, 0, 2, 3, 4),
            "keyform_binding_band.counts": (1, 1, 0, 0, 0, 0, 1, 1, 1),
            "keyform_binding.keys_begin_indices": (0, 1, 2, 5, 8),
            "keyform_binding.keys_counts": (1, 1, 3, 3, 3),
            "keys.values": (0.0, 0.0, -1.0, 0.0, 1.0, -20.0, 0.0, 20.0, -30.0, 6.0, 30.0),
            "uv.xys": _flatten(tuple(canonical_top_left_to_moc_uv(uv) for uv in DEFORMER_E0_UVS)) * 2,
            "position_index.indices": (0, 1, 2, 0, 2, 3) * 2,
            "drawable_mask.art_mesh_indices": (-1,),
            "draw_order_group.object_begin_indices": (0,),
            "draw_order_group.object_counts": (2,),
            "draw_order_group.object_total_counts": (2,),
            "draw_order_group.min_draw_orders": (1000,),
            "draw_order_group.max_draw_orders": (200,),
            "draw_order_group_object.types": (0, 0),
            "draw_order_group_object.indices": (1, 0),
            "draw_order_group_object.group_indices": (-1, -1),
            "additional.quad_transforms": (True,),
        }
    )
    return Moc3V400Document(
        counts=_DEFORMER_E0_COUNTS,
        canvas=Moc3CanvasInfo(
            pixels_per_unit=1024.0,
            origin_x=512.0,
            origin_y=384.0,
            width=DEFORMER_E0_CANVAS_SIZE[0],
            height=DEFORMER_E0_CANVAS_SIZE[1],
            flag=0,
        ),
        sections=sections,
    )


def build_deformer_e0_moc() -> bytes:
    return encode_moc3_v400(build_deformer_e0_document())


def build_deformer_e0_missing_default_moc() -> bytes:
    """Build the E0 negative control whose non-midpoint default is not a rest key."""

    positive = build_deformer_e0_document()
    sections = dict(positive.sections)
    keys = list(sections["keys.values"])
    keys[-2] = 0.0
    sections["keys.values"] = tuple(keys)
    return encode_moc3_v400(
        Moc3V400Document(
            counts=positive.counts,
            canvas=positive.canvas,
            sections=sections,
        )
    )


def _interpolate_triplet(
    value: float,
    minimum: float,
    middle: float,
    maximum: float,
    triplet: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    if value <= middle:
        amount = (value - minimum) / (middle - minimum)
        start, end = triplet[0], triplet[1]
    else:
        amount = (value - middle) / (maximum - middle)
        start, end = triplet[1], triplet[2]
    return (
        start[0] + (end[0] - start[0]) * amount,
        start[1] + (end[1] - start[1]) * amount,
    )


def _interpolate_grid(value: float) -> tuple[tuple[float, float], ...]:
    return tuple(
        _interpolate_triplet(
            value,
            -1.0,
            0.0,
            1.0,
            (negative, rest, positive),
        )
        for negative, rest, positive in zip(*_DEFORMER_E0_WARP_GRIDS_CANVAS)
    )


def _apply_grid(
    point: tuple[float, float],
    grid: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    u, v = point
    top = (
        grid[0][0] + (grid[1][0] - grid[0][0]) * u,
        grid[0][1] + (grid[1][1] - grid[0][1]) * u,
    )
    bottom = (
        grid[2][0] + (grid[3][0] - grid[2][0]) * u,
        grid[2][1] + (grid[3][1] - grid[2][1]) * u,
    )
    return (
        top[0] + (bottom[0] - top[0]) * v,
        top[1] + (bottom[1] - top[1]) * v,
    )


def evaluate_deformer_e0_canvas(
    parameter_values: Mapping[str, float] | None = None,
) -> dict[str, tuple[tuple[float, float], ...]]:
    values = dict(_DEFORMER_E0_PARAMETER_DEFAULTS)
    if parameter_values:
        values.update(parameter_values)
    grid = _interpolate_grid(values["ParamBreath"])
    outer_origin = _interpolate_triplet(
        values["ParamOuter"],
        -20.0,
        0.0,
        20.0,
        _DEFORMER_E0_OUTER_ORIGINS,
    )
    outer_pivot_canvas = _apply_grid(outer_origin, grid)
    nested_canvas_offsets = _evaluate_nested_canvas_offsets(values, reverse_stack=False)
    return {
        "ArtMeshDirect": tuple(_apply_grid(point, grid) for point in DEFORMER_E0_DIRECT_LOCAL_VERTICES),
        "ArtMeshNested": tuple(
            (outer_pivot_canvas[0] + point[0], outer_pivot_canvas[1] + point[1])
            for point in nested_canvas_offsets
        ),
    }


def _evaluate_nested_canvas_offsets(
    values: Mapping[str, float],
    *,
    reverse_stack: bool,
) -> tuple[tuple[float, float], ...]:
    outer_rank, inner_rank = ((20, 10) if reverse_stack else (10, 20))
    inner_angle = _interpolate_triplet(
        values["ParamInner"],
        -30.0,
        6.0,
        30.0,
        ((-30.0, 0.0), (0.0, 0.0), (30.0, 0.0)),
    )[0]
    entries = (
        (
            outer_rank,
            0.0,
            0.0,
            values["ParamOuter"],
            max(DEFORMER_E0_CANVAS_SIZE) / 640.0,
        ),
        (
            inner_rank,
            _DEFORMER_E0_INNER_ORIGIN[0],
            _DEFORMER_E0_INNER_ORIGIN[1],
            inner_angle,
            1.0,
        ),
    )
    return tuple(apply_rotation_stack(point, entries) for point in DEFORMER_E0_NESTED_LOCAL_VERTICES)


def evaluate_deformer_e0_reversed_stack_canvas(
    parameter_values: Mapping[str, float] | None = None,
) -> tuple[tuple[float, float], ...]:
    values = dict(_DEFORMER_E0_PARAMETER_DEFAULTS)
    if parameter_values:
        values.update(parameter_values)
    grid = _interpolate_grid(values["ParamBreath"])
    outer_origin = _interpolate_triplet(
        values["ParamOuter"],
        -20.0,
        0.0,
        20.0,
        _DEFORMER_E0_OUTER_ORIGINS,
    )
    pivot = _apply_grid(outer_origin, grid)
    return tuple(
        (pivot[0] + point[0], pivot[1] + point[1])
        for point in _evaluate_nested_canvas_offsets(values, reverse_stack=True)
    )


__all__ = [
    "DEFORMER_E0_CANVAS_SIZE",
    "DEFORMER_E0_DIRECT_LOCAL_VERTICES",
    "DEFORMER_E0_NESTED_LOCAL_VERTICES",
    "DEFORMER_E0_UVS",
    "STATIC_E0_CANVAS_SIZE",
    "STATIC_E0_CORE_UVS",
    "STATIC_E0_MOC_VERTICES",
    "STATIC_E0_MOC_UVS",
    "STATIC_E0_ROOT_VERTICES",
    "STATIC_E0_UVS",
    "build_deformer_e0_document",
    "build_deformer_e0_moc",
    "build_deformer_e0_missing_default_moc",
    "build_static_e0_document",
    "build_static_e0_moc",
    "evaluate_deformer_e0_canvas",
    "evaluate_deformer_e0_reversed_stack_canvas",
]
