"""
Wizard steps for Qinglong Captions GUI.
青龙字幕工具向导步骤。
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "render_setup_step": ".step0_setup",
    "render_import_step": ".step1_import",
    "render_video_split_step": ".step2_video_split",
    "render_tagger_step": ".step3_tagger",
    "render_caption_step": ".step4_caption",
    "render_export_step": ".step5_export",
    "render_tools_step": ".step6_tools",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
