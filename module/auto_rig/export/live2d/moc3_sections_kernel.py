from __future__ import annotations

from dataclasses import dataclass

MOC3_SECTION_KERNEL_VERSION = "moc3-v400-sections-v1"
MOC3_RUNTIME_ELEMENT_SIZE = 8
MOC3_STRING_ELEMENT_SIZE = 64


@dataclass(frozen=True, slots=True)
class Moc3SectionSpec:
    sot_index: int
    name: str
    element_kind: str
    count_index: int
    writer_alignment: int


_A = 64
_N = 1
_R = "runtime"
_S = "str64"
_I = "i32"
_F = "f32"
_H = "i16"
_B = "bool32"
_U = "u8"

_BASE_SECTION_ROWS = (
    ("part.runtime_space", _R, 0, _A),
    ("part.ids", _S, 0, _N),
    ("part.keyform_binding_band_indices", _I, 0, _A),
    ("part.keyform_begin_indices", _I, 0, _A),
    ("part.keyform_counts", _I, 0, _A),
    ("part.visibles", _B, 0, _A),
    ("part.enables", _B, 0, _A),
    ("part.parent_part_indices", _I, 0, _A),
    ("deformer.runtime_space", _R, 1, _A),
    ("deformer.ids", _S, 1, _N),
    ("deformer.keyform_binding_band_indices", _I, 1, _A),
    ("deformer.visibles", _B, 1, _A),
    ("deformer.enables", _B, 1, _A),
    ("deformer.parent_part_indices", _I, 1, _A),
    ("deformer.parent_deformer_indices", _I, 1, _A),
    ("deformer.types", _I, 1, _A),
    ("deformer.specific_indices", _I, 1, _A),
    ("warp_deformer.keyform_binding_band_indices", _I, 2, _A),
    ("warp_deformer.keyform_begin_indices", _I, 2, _A),
    ("warp_deformer.keyform_counts", _I, 2, _A),
    ("warp_deformer.vertex_counts", _I, 2, _A),
    ("warp_deformer.rows", _I, 2, _A),
    ("warp_deformer.cols", _I, 2, _A),
    ("rotation_deformer.keyform_binding_band_indices", _I, 3, _A),
    ("rotation_deformer.keyform_begin_indices", _I, 3, _A),
    ("rotation_deformer.keyform_counts", _I, 3, _A),
    ("rotation_deformer.base_angles", _F, 3, _A),
    ("art_mesh.runtime_space_0", _R, 4, _A),
    ("art_mesh.runtime_space_1", _R, 4, _A),
    ("art_mesh.runtime_space_2", _R, 4, _A),
    ("art_mesh.runtime_space_3", _R, 4, _A),
    ("art_mesh.ids", _S, 4, _N),
    ("art_mesh.keyform_binding_band_indices", _I, 4, _A),
    ("art_mesh.keyform_begin_indices", _I, 4, _A),
    ("art_mesh.keyform_counts", _I, 4, _A),
    ("art_mesh.visibles", _B, 4, _A),
    ("art_mesh.enables", _B, 4, _A),
    ("art_mesh.parent_part_indices", _I, 4, _A),
    ("art_mesh.parent_deformer_indices", _I, 4, _A),
    ("art_mesh.texture_indices", _I, 4, _A),
    ("art_mesh.drawable_flags", _U, 4, _A),
    ("art_mesh.position_index_counts", _I, 4, _A),
    ("art_mesh.uv_begin_indices", _I, 4, _A),
    ("art_mesh.position_index_begin_indices", _I, 4, _A),
    ("art_mesh.vertex_counts", _I, 4, _A),
    ("art_mesh.mask_begin_indices", _I, 4, _A),
    ("art_mesh.mask_counts", _I, 4, _A),
    ("parameter.runtime_space", _R, 5, _A),
    ("parameter.ids", _S, 5, _N),
    ("parameter.max_values", _F, 5, _A),
    ("parameter.min_values", _F, 5, _A),
    ("parameter.default_values", _F, 5, _A),
    ("parameter.repeats", _B, 5, _A),
    ("parameter.decimal_places", _I, 5, _A),
    ("parameter.keyform_binding_begin_indices", _I, 5, _A),
    ("parameter.keyform_binding_counts", _I, 5, _A),
    ("part_keyform.draw_orders", _F, 6, _A),
    ("warp_deformer_keyform.opacities", _F, 7, _A),
    ("warp_deformer_keyform.keyform_position_begin_indices", _I, 7, _A),
    ("rotation_deformer_keyform.opacities", _F, 8, _A),
    ("rotation_deformer_keyform.angles", _F, 8, _A),
    ("rotation_deformer_keyform.origin_xs", _F, 8, _A),
    ("rotation_deformer_keyform.origin_ys", _F, 8, _A),
    ("rotation_deformer_keyform.scales", _F, 8, _A),
    ("rotation_deformer_keyform.reflect_xs", _B, 8, _A),
    ("rotation_deformer_keyform.reflect_ys", _B, 8, _A),
    ("art_mesh_keyform.opacities", _F, 9, _A),
    ("art_mesh_keyform.draw_orders", _F, 9, _A),
    ("art_mesh_keyform.keyform_position_begin_indices", _I, 9, _A),
    ("keyform_position.xys", _F, 10, _A),
    ("keyform_binding_index.indices", _I, 11, _A),
    ("keyform_binding_band.begin_indices", _I, 12, _A),
    ("keyform_binding_band.counts", _I, 12, _A),
    ("keyform_binding.keys_begin_indices", _I, 13, _A),
    ("keyform_binding.keys_counts", _I, 13, _A),
    ("keys.values", _F, 14, _A),
    ("uv.xys", _F, 15, _A),
    ("position_index.indices", _H, 16, _A),
    ("drawable_mask.art_mesh_indices", _I, 17, _A),
    ("draw_order_group.object_begin_indices", _I, 18, _A),
    ("draw_order_group.object_counts", _I, 18, _A),
    ("draw_order_group.object_total_counts", _I, 18, _A),
    ("draw_order_group.min_draw_orders", _I, 18, _A),
    ("draw_order_group.max_draw_orders", _I, 18, _A),
    ("draw_order_group_object.types", _I, 19, _A),
    ("draw_order_group_object.indices", _I, 19, _A),
    ("draw_order_group_object.group_indices", _I, 19, _A),
    ("glue.runtime_space", _R, 20, _A),
    ("glue.ids", _S, 20, _N),
    ("glue.keyform_binding_band_indices", _I, 20, _A),
    ("glue.keyform_begin_indices", _I, 20, _A),
    ("glue.keyform_counts", _I, 20, _A),
    ("glue.art_mesh_index_as", _I, 20, _A),
    ("glue.art_mesh_index_bs", _I, 20, _A),
    ("glue.info_begin_indices", _I, 20, _A),
    ("glue.info_counts", _I, 20, _A),
    ("glue_info.weights", _F, 21, _A),
    ("glue_info.position_indices", _H, 21, _A),
    ("glue_keyform.intensities", _F, 22, _A),
)

MOC3_V400_SECTION_SPECS = tuple(
    Moc3SectionSpec(index + 2, name, element_kind, count_index, alignment)
    for index, (name, element_kind, count_index, alignment) in enumerate(
        (*_BASE_SECTION_ROWS, ("additional.quad_transforms", _B, 2, _A))
    )
)


def moc3_element_size(element_kind: str) -> int:
    if element_kind == _R:
        return MOC3_RUNTIME_ELEMENT_SIZE
    if element_kind == _S:
        return MOC3_STRING_ELEMENT_SIZE
    if element_kind in (_I, _F, _B):
        return 4
    if element_kind == _H:
        return 2
    if element_kind == _U:
        return 1
    raise ValueError("unknown MOC3 element kind")


__all__ = [
    "MOC3_RUNTIME_ELEMENT_SIZE",
    "MOC3_SECTION_KERNEL_VERSION",
    "MOC3_STRING_ELEMENT_SIZE",
    "MOC3_V400_SECTION_SPECS",
    "Moc3SectionSpec",
    "moc3_element_size",
]
