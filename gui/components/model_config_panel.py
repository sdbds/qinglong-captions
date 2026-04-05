"""ModelConfigPanel — inline model.toml configuration editor.

Displays editable parameters for a local model (OCR/VLM/ALM) below its
dropdown. Uses tomlkit for format-preserving read/write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import tomlkit
from nicegui import ui

from gui.utils.i18n import t
from gui.utils.toml_helpers import guess_slider_range, load_model_id_options, load_model_list_entries, unwrap

# Resolved at import time; gui/ is added to sys.path before GUI runs
from module.providers.catalog import provider_config_sections

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "model.toml"

# Fields rendered as read-only labels
_READ_ONLY_FIELDS = {"svg_model_id"}

# Fields rendered as user-editable dropdowns (allow custom HuggingFace repo IDs)
_MODEL_ID_FIELDS = {"model_id"}

# Row layout shared between all field rows
_ROW_STYLE = (
    "border-bottom: 1px solid var(--color-border, #e5e7eb); "
    "padding: 6px 0; gap: 8px;"
)

# Outer wrapper style — visually distinguishes panel from parent card
_PANEL_STYLE = (
    "background: rgba(5, 150, 105, 0.08); "
    "box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.06); "
    "border-radius: 10px; "
    "padding: 8px 12px; "
    "margin-top: 6px; "
    "border: none; outline: none;"
)


class ModelConfigPanel:
    """Shows and edits the model.toml section for a selected local model.

    Usage::

        container = ui.column().classes("w-full")
        panel = ModelConfigPanel(container)
        panel.show("chandra_ocr")   # render the [chandra_ocr] section
        panel.hide()                # collapse and clear
    """

    def __init__(self, parent: ui.column, on_save: Optional[Callable[[], None]] = None) -> None:
        self._parent = parent
        self._current_route: Optional[str] = None
        self._section_name: Optional[str] = None
        self._data: Dict[str, Any] = {}   # plain Python copy for widget binding
        self._on_save = on_save

    # ── Public API ────────────────────────────────────────────────────

    def show(self, route_name: str) -> None:
        """Render the config panel for *route_name*. No-op if already shown."""
        if route_name == self._current_route:
            return
        self._current_route = route_name
        self._section_name = None
        self._data = {}
        self._parent.clear()
        with self._parent:
            self._render(route_name)

    def hide(self) -> None:
        """Hide the panel and reset state."""
        self._current_route = None
        self._section_name = None
        self._data = {}
        self._parent.clear()

    # ── Section lookup ─────────────────────────────────────────────────

    def _find_section(self, route_name: str) -> Optional[Tuple[str, dict]]:
        """Return (section_name, plain_dict) or None if not found in model.toml."""
        if not _CONFIG_PATH.exists():
            return None
        try:
            doc = tomlkit.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

        candidates = provider_config_sections(route_name)
        for candidate in candidates:
            if candidate in doc:
                raw = doc[candidate]
                return candidate, unwrap(raw)
        return None

    @staticmethod
    def _build_model_id_options(
        route_name: str,
        current_model_id: str,
        probe=None,
    ) -> dict[str, str]:
        options = load_model_id_options(
            route_name,
            current_model_id=current_model_id,
        )
        if options:
            return options
        if current_model_id:
            return {current_model_id: current_model_id}
        return {}

    @staticmethod
    def _set_control_value(control: Any, value: Any) -> None:
        if hasattr(control, "set_value"):
            control.set_value(value)
            return
        control.value = value

    # ── Rendering ──────────────────────────────────────────────────────

    def _render(self, route_name: str) -> None:
        result = self._find_section(route_name)

        with ui.column().classes("w-full").style(_PANEL_STYLE):
            with ui.expansion(
                text=t("model_config_panel", "模型配置"),
                icon="tune",
                value=True,
            ).classes("w-full model-config-expansion").props("dense"):
                if result is None:
                    ui.label(
                        t("model_config_no_section", "未找到此模型的配置。")
                    ).classes("text-caption").style("color: var(--color-text-secondary); padding: 8px 0;")
                    return

                section_name, data = result
                self._section_name = section_name
                self._data = data

                # Section hint
                ui.label(f"[{section_name}]").classes("text-caption text-weight-bold").style(
                    "color: var(--color-text-secondary); padding: 4px 0 8px 0; font-family: monospace;"
                )

                with ui.column().classes("w-full gap-0"):
                    for key, value in data.items():
                        if key == "model_list":
                            continue
                        if isinstance(value, dict):
                            self._render_nested_section(key, value)
                        else:
                            self._render_field(key, value, data, read_only=key in _READ_ONLY_FIELDS)

                # Save + Restore buttons
                with ui.row().classes("w-full justify-end q-mt-sm gap-0"):
                    ui.button(
                        t("model_config_restore", "恢复默认"),
                        icon="restart_alt",
                        on_click=self._restore_defaults,
                    ).props("dense flat").style(
                        "color: var(--color-text-secondary); font-size: 12px;"
                    )
                    ui.button(
                        t("model_config_save", "保存配置"),
                        icon="save",
                        on_click=self._save,
                    ).props("dense flat").style(
                        "color: var(--color-primary); font-size: 12px;"
                    )

    def _render_nested_section(self, name: str, data: dict) -> None:
        """Render a nested TOML sub-table (e.g. paddle_ocr.save) as a sub-expansion."""
        with ui.expansion(
            text=name,
            icon="subdirectory_arrow_right",
            value=False,
        ).classes("w-full q-ml-sm").props("dense"):
            with ui.column().classes("w-full gap-0"):
                for key, value in data.items():
                    self._render_field(key, value, data, read_only=key in _READ_ONLY_FIELDS)

    def _render_field(
        self,
        key: str,
        value: Any,
        parent_dict: dict,
        *,
        read_only: bool = False,
    ) -> None:
        """Type-dispatched field widget renderer."""
        if key == "model_list":
            return

        if read_only:
            with ui.row().classes("w-full items-center").style(_ROW_STYLE):
                ui.label(key).classes("text-caption text-weight-medium").style(
                    "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
                )
                ui.label(str(value)).classes("text-caption").style(
                    "color: var(--color-text); font-family: monospace; flex: 1; overflow-wrap: anywhere;"
                )
                ui.label(
                    t("model_config_read_only", "只读")
                ).classes("text-caption").style(
                    "color: var(--color-text-secondary); opacity: 0.6; font-size: 10px; margin-left: 4px;"
                )
            return

        # model_id → model_list select (if available) + free-text input
        if key in _MODEL_ID_FIELDS and isinstance(value, str):
            route = self._current_route or ""
            entries = load_model_list_entries(route)
            if entries:
                self._render_model_list_and_model_id(parent_dict, entries)
            else:
                self._render_model_id_input(key, value, parent_dict)
            return

        # bool must come before int (bool is subclass of int in Python)
        if isinstance(value, bool):
            with ui.row().classes("w-full items-center").style(_ROW_STYLE):
                ui.label(key).classes("text-caption text-weight-medium").style(
                    "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
                )
                sw = ui.switch(value=value).props("dense color=amber")
                sw.on_value_change(lambda e, k=key, p=parent_dict: p.__setitem__(k, e.value))

        elif isinstance(value, int):
            min_v, max_v, step_v = guess_slider_range(key, value, False)
            with ui.row().classes("w-full items-center").style(_ROW_STYLE):
                ui.label(key).classes("text-caption text-weight-medium").style(
                    "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
                )
                slider = (
                    ui.slider(min=min_v, max=max_v, step=step_v, value=value)
                    .props("dense label-always")
                    .style("flex: 1; min-width: 100px;")
                )
                num = (
                    ui.number(value=value, step=step_v)
                    .props("dense outlined")
                    .style("width: 80px; margin-left: 8px;")
                )

                def _sync_int_sl(e, k=key, p=parent_dict, sl=slider, nm=num):
                    v = int(e.value) if e.value is not None else 0
                    p[k] = v
                    if nm.value != v:
                        nm.value = v

                def _sync_int_nm(e, k=key, p=parent_dict, sl=slider):
                    v = int(e.value) if e.value is not None else 0
                    p[k] = v
                    if sl.value != v:
                        sl.value = v

                slider.on_value_change(_sync_int_sl)
                num.on_value_change(_sync_int_nm)

        elif isinstance(value, float):
            min_v, max_v, step_v = guess_slider_range(key, value, True)
            with ui.row().classes("w-full items-center").style(_ROW_STYLE):
                ui.label(key).classes("text-caption text-weight-medium").style(
                    "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
                )
                slider = (
                    ui.slider(min=min_v, max=max_v, step=step_v, value=value)
                    .props("dense label-always")
                    .style("flex: 1; min-width: 100px;")
                )
                num = (
                    ui.number(value=value, step=step_v, format="%.4g")
                    .props("dense outlined")
                    .style("width: 90px; margin-left: 8px;")
                )

                def _sync_float_sl(e, k=key, p=parent_dict, sl=slider, nm=num):
                    v = float(e.value) if e.value is not None else 0.0
                    p[k] = v
                    if nm.value != v:
                        nm.value = v

                def _sync_float_nm(e, k=key, p=parent_dict, sl=slider):
                    v = float(e.value) if e.value is not None else 0.0
                    p[k] = v
                    if sl.value != v:
                        sl.value = v

                slider.on_value_change(_sync_float_sl)
                num.on_value_change(_sync_float_nm)

        elif isinstance(value, str):
            is_multiline = "\n" in value or len(value) > 100
            if is_multiline:
                with ui.column().classes("w-full gap-0").style(_ROW_STYLE):
                    with ui.expansion(text=key, icon="article", value=False).classes("w-full").props("dense"):
                        ta = (
                            ui.textarea(value=value)
                            .classes("w-full")
                            .props(
                                "outlined autogrow "
                                'input-style="font-family: Consolas, Monaco, monospace; font-size: 12px;"'
                            )
                        )
                        ta.on_value_change(lambda e, k=key, p=parent_dict: p.__setitem__(k, e.value or ""))
            else:
                with ui.row().classes("w-full items-center").style(_ROW_STYLE):
                    ui.label(key).classes("text-caption text-weight-medium").style(
                        "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
                    )
                    inp = (
                        ui.input(value=value)
                        .classes("flex-1")
                        .props("dense outlined")
                        .style("font-size: 13px;")
                    )
                    inp.on_value_change(lambda e, k=key, p=parent_dict: p.__setitem__(k, e.value or ""))

        elif isinstance(value, list):
            # Render as newline-separated textarea for string lists
            display = "\n".join(str(v) for v in value)
            with ui.column().classes("w-full gap-0").style(_ROW_STYLE):
                with ui.expansion(text=key, icon="list", value=False).classes("w-full").props("dense"):
                    ta = (
                        ui.textarea(value=display)
                        .classes("w-full")
                        .props(
                            "outlined autogrow "
                            'input-style="font-family: Consolas, Monaco, monospace; font-size: 12px;"'
                        )
                    )

                    def _sync_list(e, k=key, p=parent_dict, orig=value):
                        lines = [l for l in (e.value or "").splitlines()]
                        # preserve original element types
                        if orig and isinstance(orig[0], str):
                            p[k] = lines
                        else:
                            # try to cast back to original type
                            try:
                                p[k] = [type(orig[0])(l) for l in lines if l.strip()]
                            except Exception:
                                p[k] = lines

                    ta.on_value_change(_sync_list)

    def _render_model_id_input(self, key: str, value: str, parent_dict: dict) -> None:
        """Plain model_id input with datalist autocomplete (no model_list)."""
        suggestions = list(self._build_model_id_options(self._current_route or "", value).keys())
        dl_id = f"dl_{id(parent_dict)}_{key}"
        with ui.row().classes("w-full items-center").style(_ROW_STYLE):
            ui.label(key).classes("text-caption text-weight-medium").style(
                "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
            )
            inp = (
                ui.input(value=value, placeholder="HuggingFace repo ID...")
                .classes("flex-1")
                .props(f'dense outlined list="{dl_id}"')
                .style("font-size: 13px; font-family: monospace;")
            )
            if suggestions:
                opts_html = "".join(f'<option value="{s}"/>' for s in suggestions)
                ui.html(f'<datalist id="{dl_id}">{opts_html}</datalist>')
        inp.on_value_change(lambda e, k=key, p=parent_dict: p.__setitem__(k, e.value or ""))

    def _render_model_list_and_model_id(self, parent_dict: dict, entries: tuple) -> None:
        """Render model_list select + model_id input with linkage."""
        route = self._current_route or ""
        model_id_value = str(parent_dict.get("model_id", "") or "").strip()

        name_to_id = {e.name: e.model_id for e in entries}
        id_to_name = {e.model_id: e.name for e in entries}
        list_options = {e.name: e.name for e in entries}
        current_name = id_to_name.get(model_id_value) or None

        # model_list row — searchable select of friendly product names
        with ui.row().classes("w-full items-center").style(_ROW_STYLE):
            ui.label("model_list").classes("text-caption text-weight-medium").style(
                "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
            )
            list_select = (
                ui.select(
                    options=list_options,
                    value=current_name,
                    label="",
                )
                .classes("flex-1 modern-select force-light-bg")
                .props(
                    'dense use-input fill-input hide-selected clearable '
                    'dropdown-icon="search" placeholder="Search product model..."'
                )
                .style("font-size: 13px;")
            )

        # model_id row — free-text input
        suggestions = list(self._build_model_id_options(route, model_id_value).keys())
        dl_id = f"dl_{id(parent_dict)}_model_id"
        with ui.row().classes("w-full items-center").style(_ROW_STYLE):
            ui.label("model_id").classes("text-caption text-weight-medium").style(
                "color: var(--color-text-secondary); min-width: 160px; flex-shrink: 0;"
            )
            model_id_inp = (
                ui.input(value=model_id_value, placeholder="HuggingFace repo ID...")
                .classes("flex-1")
                .props(f'dense outlined list="{dl_id}"')
                .style("font-size: 13px; font-family: monospace;")
            )
            if suggestions:
                opts_html = "".join(f'<option value="{s}"/>' for s in suggestions)
                ui.html(f'<datalist id="{dl_id}">{opts_html}</datalist>')

        # Linkage: list → id
        def _on_list_change(e):
            name = str(e.value or "").strip()
            if name and name in name_to_id:
                new_id = name_to_id[name]
                parent_dict["model_id"] = new_id
                model_id_inp.value = new_id

        # Linkage: id → list
        def _on_id_change(e):
            new_id = str(e.value or "").strip()
            parent_dict["model_id"] = new_id
            matched = id_to_name.get(new_id) or None
            if list_select.value != matched:
                list_select.value = matched

        list_select.on_value_change(_on_list_change)
        model_id_inp.on_value_change(_on_id_change)

    # ── Save ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._section_name or not _CONFIG_PATH.exists():
            ui.notify(t("model_config_save_error", "保存模型配置失败"), type="negative", position="top")
            return
        try:
            doc = tomlkit.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
            self._apply_values(doc[self._section_name], self._data)
            _CONFIG_PATH.write_text(tomlkit.dumps(doc), encoding="utf-8")
            if self._on_save is not None:
                self._on_save()
            ui.notify(t("model_config_saved", "模型配置已保存"), type="positive", position="top")
        except Exception as exc:
            ui.notify(
                f"{t('model_config_save_error', '保存模型配置失败')}: {exc}",
                type="negative",
                position="top",
            )

    # ── Restore defaults ───────────────────────────────────────────────

    def _restore_defaults(self) -> None:
        """Re-read model.toml from disk and re-render with original values."""
        route = self._current_route
        if not route:
            return
        # Clear route guard so show() will re-render
        self._current_route = None
        self.show(route)
        if self._on_save is not None:
            self._on_save()
        ui.notify(t("model_config_restored", "已恢复默认配置"), type="info", position="top")

    @staticmethod
    def _apply_values(toml_node: Any, data: dict) -> None:
        """Recursively write plain Python values back into a tomlkit document node."""
        for key, value in data.items():
            if key == "model_list":
                continue
            if key not in toml_node:
                continue
            if isinstance(value, dict):
                ModelConfigPanel._apply_values(toml_node[key], value)
            else:
                toml_node[key] = value
