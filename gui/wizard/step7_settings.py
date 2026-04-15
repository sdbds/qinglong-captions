"""配置管理悬浮窗 - 结构化表单编辑 prompts.toml / model.toml / general.toml + 环境变量"""

from pathlib import Path
from typing import Any, Dict, List

import tomlkit
from gui.utils.toml_helpers import unwrap as _unwrap, guess_slider_range as _guess_slider_range
from nicegui import ui

from theme import COLORS
from gui.utils.i18n import t, get_i18n
from gui.utils.env_config import (
    ENV_VAR_DEFINITIONS,
    load_env_config,
    save_env_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

_CONFIG_FILES = {
    "prompts": "prompts.toml",
    "model": "model.toml",
    "general": "general.toml",
}


# ── Scoped CSS for settings dialog ─────────────────────────────────

_SETTINGS_CSS = """
/* ---- Settings dialog background ---- */
.settings-dialog-bg {
    background: #e8f5e9 !important;          /* light: distinct mint-green */
}
body.dark-mode .settings-dialog-bg {
    background: #001a11 !important;          /* dark: very deep green-black */
}

/* ---- Top-level section cards ---- */
.settings-dialog-bg .settings-section-card {
    background: #ffffff !important;          /* light: white card */
    border: 1px solid rgba(5,150,105,0.25) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
body.dark-mode .settings-dialog-bg .settings-section-card {
    background: #0a3d2e !important;          /* dark: teal-green card */
    border: 1px solid rgba(16,185,129,0.3) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}

/* ---- Nested sub-section cards ---- */
.settings-dialog-bg .settings-nested-card {
    background: #f1f8e9 !important;          /* light: lime tint */
    border: 1px solid rgba(5,150,105,0.18) !important;
}
body.dark-mode .settings-dialog-bg .settings-nested-card {
    background: #072a22 !important;          /* dark: slightly lighter than card */
    border: 1px solid rgba(16,185,129,0.2) !important;
}

/* ---- Section header gradient strip ---- */
.settings-dialog-bg .settings-section-header {
    background: linear-gradient(135deg,
        rgba(5,150,105,0.10), rgba(251,191,36,0.05)) !important;
}
body.dark-mode .settings-dialog-bg .settings-section-header {
    background: linear-gradient(135deg,
        rgba(16,185,129,0.20), rgba(251,191,36,0.10)) !important;
}

/* ---- Clickable key label ---- */
.settings-key-label {
    cursor: pointer;
    border-radius: 4px;
    padding: 1px 4px;
    transition: background 0.15s;
}
.settings-key-label:hover {
    background: rgba(5,150,105,0.10);
}
body.dark-mode .settings-key-label:hover {
    background: rgba(16,185,129,0.15);
}
"""


class SettingsDialog:
    """配置管理悬浮窗 — structured TOML form editor."""

    def __init__(self):
        self.data: Dict[str, dict] = {}
        self.tab_containers: Dict[str, ui.column] = {}
        self.dialog = None
        self._active_key = "prompts"
        self.env_data: Dict[str, str] = {}
        self._env_container: ui.column | None = None
        self._pending_tabs: set = set()

    # ── I/O ──────────────────────────────────────────────────────────

    def _load(self, key: str) -> dict:
        """Load a config file as plain Python dicts (JSON-safe for NiceGUI)."""
        path = CONFIG_DIR / _CONFIG_FILES[key]
        if path.exists():
            doc = tomlkit.parse(path.read_text(encoding="utf-8"))
            return _unwrap(doc)
        return {}

    def _save_current(self):
        key = self._active_key
        if key == "env":
            try:
                save_env_config(self.env_data)
                ui.notify(t("save_success"), type="positive", position="top")
            except Exception as exc:
                ui.notify(
                    f"{t('save_failed')}: {exc}", type="negative", position="top"
                )
            return
        try:
            content = tomlkit.dumps(self.data[key])
            (CONFIG_DIR / _CONFIG_FILES[key]).write_text(content, encoding="utf-8")
            ui.notify(t("save_success"), type="positive", position="top")
        except Exception as exc:
            ui.notify(
                f"{t('save_failed')}: {exc}", type="negative", position="top"
            )

    def _reload_current(self):
        key = self._active_key
        if key == "env":
            self.env_data = load_env_config()
            self._render_env_tab()
            ui.notify(t("reload_success"), type="info", position="top")
            return
        self.data[key] = self._load(key)
        self._render_tab(key)
        ui.notify(t("reload_success"), type="info", position="top")

    # ── Shared styles ─────────────────────────────────────────────

    _LABEL_STYLE = (
        "min-width: 200px; max-width: 200px; color: var(--color-text); "
        "word-break: break-all;"
    )
    _ROW_STYLE = (
        "border-bottom: 1px solid var(--card-border); padding: 6px 0;"
    )

    # ── Click-to-edit key label ──────────────────────────────────────

    def _editable_key_label(self, key: str, parent: dict):
        """Render a key label that turns into an input on click."""
        container = ui.element("div").style(
            "min-width: 200px; max-width: 200px; display: inline-flex; "
            "align-items: center;"
        )
        with container:
            label = (
                ui.label(key)
                .classes("text-body2 text-weight-medium settings-key-label")
                .style("color: var(--color-text);")
            )
            inp = (
                ui.input(value=key)
                .props("dense borderless")
                .style(
                    "font-size: 13px; font-weight: 600; max-width: 190px; "
                    "color: var(--color-text);"
                )
            )
            inp.visible = False

            cur_key = [key]

            def _start_edit(_e, lb=label, ip=inp):
                lb.visible = False
                ip.visible = True
                ui.run_javascript(
                    f"var c=document.getElementById('c{ip.id}');"
                    f"if(c){{var i=c.querySelector('input');if(i)i.focus();}}"
                )

            def _finish_edit(_e, lb=label, ip=inp, ck=cur_key, p=parent):
                new = (ip.value or "").strip()
                old = ck[0]
                if new and new != old and new not in p:
                    p[new] = p.pop(old)
                    ck[0] = new
                    lb.text = new
                else:
                    ip.value = ck[0]
                ip.visible = False
                lb.visible = True

            label.on("click", _start_edit)
            inp.on("blur", _finish_edit)
            inp.on("keydown.enter", _finish_edit)

        return container, cur_key

    # ── Field renderers ──────────────────────────────────────────────

    def _render_field(
        self,
        key: str,
        value: Any,
        parent: dict,
        *,
        show_delete: bool = True,
    ):
        """Render one key-value pair with an appropriate form control."""
        # bool must come before int (bool is subclass of int in Python)
        if isinstance(value, bool):
            with ui.row().classes("w-full items-center").style(self._ROW_STYLE):
                self._editable_key_label(key, parent)
                sw = ui.switch(value=value).props("dense color=amber")
                sw.on_value_change(
                    lambda e, k=key, p=parent: p.__setitem__(k, e.value)
                )
                ui.element("div").style("flex: 1;")
                if show_delete:
                    self._delete_btn(key, parent)

        elif isinstance(value, int):
            min_v, max_v, step_v = _guess_slider_range(key, value, False)
            with ui.row().classes("w-full items-center").style(self._ROW_STYLE):
                self._editable_key_label(key, parent)
                slider = (
                    ui.slider(min=min_v, max=max_v, step=step_v, value=value)
                    .props("dense label-always")
                    .style("flex: 1; min-width: 120px;")
                )
                num = (
                    ui.number(value=value, step=step_v)
                    .props("dense outlined")
                    .style("width: 90px; margin-left: 8px;")
                )

                def _sync_int_slider(e, k=key, p=parent, sl=slider, nm=num):
                    v = int(e.value) if e.value is not None else 0
                    p[k] = v
                    if nm.value != v:
                        nm.value = v

                def _sync_int_num(e, k=key, p=parent, sl=slider, nm=num):
                    v = int(e.value) if e.value is not None else 0
                    p[k] = v
                    if sl.value != v:
                        sl.value = v

                slider.on_value_change(_sync_int_slider)
                num.on_value_change(_sync_int_num)
                if show_delete:
                    self._delete_btn(key, parent)

        elif isinstance(value, float):
            min_v, max_v, step_v = _guess_slider_range(key, value, True)
            with ui.row().classes("w-full items-center").style(self._ROW_STYLE):
                self._editable_key_label(key, parent)
                slider = (
                    ui.slider(min=min_v, max=max_v, step=step_v, value=value)
                    .props("dense label-always")
                    .style("flex: 1; min-width: 120px;")
                )
                num = (
                    ui.number(value=value, step=step_v, format="%.4g")
                    .props("dense outlined")
                    .style("width: 100px; margin-left: 8px;")
                )

                def _sync_float_slider(e, k=key, p=parent, sl=slider, nm=num):
                    v = float(e.value) if e.value is not None else 0.0
                    p[k] = v
                    if nm.value != v:
                        nm.value = v

                def _sync_float_num(e, k=key, p=parent, sl=slider, nm=num):
                    v = float(e.value) if e.value is not None else 0.0
                    p[k] = v
                    if sl.value != v:
                        sl.value = v

                slider.on_value_change(_sync_float_slider)
                num.on_value_change(_sync_float_num)
                if show_delete:
                    self._delete_btn(key, parent)

        elif isinstance(value, str):
            is_multiline = "\n" in value or len(value) > 120
            if is_multiline:
                # Collapsible multiline text — collapsed by default
                with ui.column().classes("w-full gap-0").style(self._ROW_STYLE):
                    with ui.row().classes("w-full items-center"):
                        with ui.expansion(
                            text=key,
                            icon="article",
                            value=False,
                        ).classes("flex-1").style(
                            "border: none; margin: 0; padding: 0;"
                        ) as exp:
                            exp.props("dense dense-toggle")
                            ta = (
                                ui.textarea(value=value)
                                .classes("w-full")
                                .props(
                                    "outlined autogrow "
                                    'input-style="font-family: Consolas, Monaco, '
                                    'monospace; font-size: 12px; line-height: 1.4;"'
                                )
                            )
                            ta.on_value_change(
                                lambda e, k=key, p=parent: p.__setitem__(
                                    k, e.value or ""
                                )
                            )
                        if show_delete:
                            self._delete_btn(key, parent)
            else:
                with ui.row().classes("w-full items-center").style(self._ROW_STYLE):
                    self._editable_key_label(key, parent)
                    inp = (
                        ui.input(value=value)
                        .props("dense outlined")
                        .classes("flex-1")
                    )
                    inp.on_value_change(
                        lambda e, k=key, p=parent: p.__setitem__(
                            k, e.value or ""
                        )
                    )
                    if show_delete:
                        self._delete_btn(key, parent)

        elif isinstance(value, list):
            self._render_list_field(key, value, parent, show_delete=show_delete)

    def _delete_btn(self, key: str, parent: dict):
        """Render a small delete button for a field."""

        def _do_delete(_e, k=key, p=parent):
            if k in p:
                del p[k]
            self._render_tab(self._active_key)

        btn = ui.button(
            icon="remove_circle_outline", on_click=_do_delete
        ).props("flat round dense size=sm")
        btn.style(
            "color: var(--color-text-secondary); opacity: 0.5; margin-left: 4px;"
        )
        btn.tooltip(f"Delete '{key}'")

    def _render_list_field(
        self,
        key: str,
        items: list,
        parent: dict,
        *,
        show_delete: bool = True,
    ):
        """Render a list value."""
        if all(isinstance(v, dict) for v in items) and items:
            self._render_dict_array(
                key, items, parent, show_delete=show_delete
            )
            return

        if all(isinstance(v, str) for v in items):
            with ui.column().classes("w-full gap-1").style(self._ROW_STYLE):
                with ui.row().classes("w-full items-center"):
                    ui.label(f"{key}  ({len(items)} items)").classes(
                        "text-body2 text-weight-medium"
                    ).style("color: var(--color-text); flex: 1;")
                    if show_delete:
                        self._delete_btn(key, parent)
                ta = (
                    ui.textarea(value="\n".join(items))
                    .classes("w-full")
                    .props(
                        "outlined autogrow "
                        'input-style="font-family: Consolas, Monaco, '
                        'monospace; font-size: 12px;"'
                    )
                )
                ta.on_value_change(
                    lambda e, k=key, p=parent: p.__setitem__(
                        k,
                        [
                            ln
                            for ln in (e.value or "").splitlines()
                            if ln.strip()
                        ],
                    )
                )
        else:
            with ui.row().classes("w-full items-center").style(self._ROW_STYLE):
                ui.label(key).classes(
                    "text-body2 text-weight-medium"
                ).style(self._LABEL_STYLE)
                text = ", ".join(str(v) for v in items)
                inp = (
                    ui.input(value=text)
                    .props("dense outlined")
                    .classes("flex-1")
                )
                inp.on_value_change(
                    lambda e, k=key, p=parent: p.__setitem__(
                        k,
                        [
                            s.strip()
                            for s in (e.value or "").split(",")
                            if s.strip()
                        ],
                    )
                )
                if show_delete:
                    self._delete_btn(key, parent)

    def _render_dict_array(
        self,
        key: str,
        items: List[dict],
        parent: dict,
        *,
        show_delete: bool = True,
    ):
        """Render an array of dicts as a mini-table of editable rows."""
        if not items:
            return
        field_names = list(items[0].keys())
        with ui.column().classes("w-full gap-1").style(self._ROW_STYLE):
            with ui.row().classes("w-full items-center"):
                ui.label(f"{key}  ({len(items)} items)").classes(
                    "text-body2 text-weight-medium"
                ).style("color: var(--color-text); flex: 1;")
                if show_delete:
                    self._delete_btn(key, parent)
            for idx, item in enumerate(items):
                with ui.row().classes("w-full items-center gap-2").style(
                    "padding: 4px 0; "
                    "border-bottom: 1px solid var(--card-border);"
                ):
                    ui.label(f"#{idx}").classes(
                        "text-caption text-weight-bold"
                    ).style(
                        "min-width: 28px; "
                        "color: var(--color-text-secondary);"
                    )
                    for fk in field_names:
                        fv = item.get(fk, "")
                        if isinstance(fv, bool):
                            sw = ui.switch(value=fv).props("dense")
                            sw.tooltip(fk)
                            sw.on_value_change(
                                lambda e, it=item, f=fk: it.__setitem__(
                                    f, e.value
                                )
                            )
                        elif isinstance(fv, int):
                            inp = (
                                ui.number(label=fk, value=fv, step=1)
                                .props("dense outlined")
                                .style("width: 110px;")
                            )
                            inp.on_value_change(
                                lambda e, it=item, f=fk: it.__setitem__(
                                    f,
                                    int(e.value) if e.value is not None else 0,
                                )
                            )
                        elif isinstance(fv, float):
                            inp = (
                                ui.number(label=fk, value=fv, step=0.1)
                                .props("dense outlined")
                                .style("width: 110px;")
                            )
                            inp.on_value_change(
                                lambda e, it=item, f=fk: it.__setitem__(
                                    f,
                                    float(e.value)
                                    if e.value is not None
                                    else 0.0,
                                )
                            )
                        else:
                            inp = (
                                ui.input(label=fk, value=str(fv))
                                .props("dense outlined")
                                .style("flex: 1; min-width: 100px;")
                            )
                            inp.on_value_change(
                                lambda e, it=item, f=fk: it.__setitem__(
                                    f, e.value or ""
                                )
                            )
                    # Delete row button
                    def _del_row(_e, k=key, p=parent, i=idx):
                        lst = p.get(k, [])
                        if 0 <= i < len(lst):
                            lst.pop(i)
                        self._render_tab(self._active_key)

                    del_btn = ui.button(
                        icon="remove_circle_outline", on_click=_del_row
                    ).props("flat round dense size=sm")
                    del_btn.style(
                        "color: var(--color-text-secondary); opacity: 0.5;"
                    )

    # ── Section renderer ─────────────────────────────────────────────

    _SECTION_CARD_LAYOUT = (
        "border-radius: 12px; padding: 0; margin-bottom: 12px; "
        "overflow: hidden;"
    )
    _SECTION_HEADER_LAYOUT = (
        "border-bottom: 1px solid var(--card-border); padding: 8px 16px; "
        "cursor: pointer; user-select: none;"
    )
    _NESTED_CARD_LAYOUT = (
        "border-radius: 8px; padding: 0; margin-top: 8px; overflow: hidden;"
    )

    def _render_section(
        self,
        display_path: str,
        data: dict,
        parent: dict,
        key_in_parent: str,
        *,
        is_nested: bool = False,
    ):
        """Render one TOML [section] as a collapsible card container."""
        simple: list = []
        nested: list = []
        for k, v in data.items():
            (nested if isinstance(v, dict) else simple).append((k, v))

        if is_nested:
            card_cls = "w-full settings-nested-card"
            card_layout = self._NESTED_CARD_LAYOUT
        else:
            card_cls = "w-full settings-section-card"
            card_layout = self._SECTION_CARD_LAYOUT

        with ui.column().classes(card_cls).style(card_layout):
            # ── Clickable header with editable section name ──
            cur_key = [key_in_parent]
            collapsed = [False]  # default: expanded

            header_row = ui.row().classes(
                "w-full items-center gap-2 settings-section-header"
            ).style(self._SECTION_HEADER_LAYOUT)

            body_col = ui.column().classes("w-full gap-0").style(
                "padding: 12px 16px;"
            )

            with header_row:
                # Collapse/expand chevron
                chevron = ui.icon("expand_more", size="20px").style(
                    "color: var(--color-text-secondary); "
                    "transition: transform 0.2s;"
                )

                ui.icon(
                    "subdirectory_arrow_right"
                    if is_nested
                    else "folder_open",
                    size="18px",
                ).style(f"color: {COLORS['primary']};")

                # Section name — click-to-edit label / input pair
                name_label = (
                    ui.label(key_in_parent)
                    .classes(
                        "text-body1 text-weight-bold settings-key-label"
                    )
                    .style("color: var(--color-text); font-size: 14px;")
                )
                name_inp = (
                    ui.input(value=key_in_parent)
                    .props("dense borderless")
                    .style(
                        "font-weight: 700; font-size: 14px; max-width: 400px; "
                        "color: var(--color-text);"
                    )
                )
                name_inp.visible = False

                def _start_name_edit(
                    _e, lb=name_label, ip=name_inp
                ):
                    lb.visible = False
                    ip.visible = True

                def _finish_name_edit(
                    _e, lb=name_label, ip=name_inp, ck=cur_key, p=parent
                ):
                    new = (ip.value or "").strip()
                    old = ck[0]
                    if new and new != old and new not in p:
                        p[new] = p.pop(old)
                        ck[0] = new
                        lb.text = new
                    else:
                        ip.value = ck[0]
                    ip.visible = False
                    lb.visible = True

                name_label.on("click.stop", _start_name_edit)
                name_inp.on("blur", _finish_name_edit)
                name_inp.on("keydown.enter", _finish_name_edit)
                # Prevent header click-to-collapse when interacting with input
                name_inp.on("click.stop", lambda _e: None)

                ui.label(f"[{display_path}]").classes("text-caption").style(
                    "color: var(--color-text-secondary); margin-left: auto;"
                )

                # Toggle collapse on header click
                def _toggle_collapse(
                    _e,
                    bc=body_col,
                    ch=chevron,
                    flag=collapsed,
                ):
                    flag[0] = not flag[0]
                    bc.visible = not flag[0]
                    ch.style(
                        "color: var(--color-text-secondary); "
                        "transition: transform 0.2s; "
                        + ("transform: rotate(-90deg);" if flag[0] else "")
                    )

            header_row.on("click", _toggle_collapse)

            # ── Card body: fields (visible by default) ──
            with body_col:
                for k, v in simple:
                    self._render_field(k, v, data)

                for k, v in nested:
                    self._render_section(
                        f"{display_path}.{k}", v, data, k, is_nested=True
                    )

                # ── Add new item button ──
                def _add_item(_e, d=data):
                    with ui.dialog() as add_dlg, ui.card().style(
                        "min-width: 340px; "
                        "background: var(--color-surface) !important; "
                        "border: 1px solid var(--card-border);"
                    ):
                        ui.label(t("settings_title")).classes(
                            "text-subtitle1 text-weight-bold"
                        ).style("color: var(--color-text);")

                        key_input = (
                            ui.input(label="Key", placeholder="new_key")
                            .props("dense outlined")
                            .classes("w-full")
                        )

                        type_select = (
                            ui.select(
                                {
                                    "str": "String",
                                    "int": "Integer",
                                    "float": "Float",
                                    "bool": "Boolean",
                                    "dict": "Section {}",
                                },
                                value="str",
                                label="Type",
                            )
                            .props("outlined")
                            .classes("w-full q-mt-sm")
                        )

                        with ui.row().classes(
                            "w-full justify-end gap-2 q-mt-sm"
                        ):
                            ui.button(
                                icon="close", on_click=add_dlg.close
                            ).props("flat dense")

                            def _confirm(
                                _ev,
                                dlg=add_dlg,
                                ki=key_input,
                                ts=type_select,
                                dd=d,
                            ):
                                name = (ki.value or "").strip()
                                if not name or name in dd:
                                    ui.notify(
                                        "Key empty or already exists",
                                        type="warning",
                                    )
                                    return
                                defaults = {
                                    "str": "",
                                    "int": 0,
                                    "float": 0.0,
                                    "bool": False,
                                    "dict": {},
                                }
                                dd[name] = defaults.get(ts.value, "")
                                dlg.close()
                                self._render_tab(self._active_key)

                            ui.button(
                                icon="check", on_click=_confirm
                            ).props("flat dense").style(
                                f"color: {COLORS['primary']};"
                            )
                    add_dlg.open()

                with ui.row().classes("w-full justify-center").style(
                    "padding-top: 8px;"
                ):
                    add_btn = ui.button(
                        icon="add", on_click=_add_item
                    ).props("flat round dense")
                    add_btn.style(
                        f"color: {COLORS['primary']}; "
                        "border: 1px dashed var(--card-border); "
                        "opacity: 0.7;"
                    )
                    add_btn.tooltip("Add new item")

    # ── Tab renderer ─────────────────────────────────────────────────

    def _render_tab(self, key: str):
        """(Re-)render all sections for one config file."""
        container = self.tab_containers.get(key)
        if container is None:
            return
        container.clear()
        with container:
            data = self.data.get(key, {})
            for section_name, section_val in data.items():
                if isinstance(section_val, dict):
                    self._render_section(
                        section_name, section_val, data, section_name
                    )
                else:
                    self._render_field(section_name, section_val, data)

    # ── Environment variables tab ────────────────────────────────────

    _GROUP_LABELS = {
        "runtime": "env_group_runtime",
        "uv": "env_group_uv",
        "network": "env_group_network",
    }
    _GROUP_ICONS = {
        "runtime": "memory",
        "uv": "inventory_2",
        "network": "language",
    }

    def _render_env_tab(self):
        """(Re-)render the environment variables tab."""
        container = self._env_container
        if container is None:
            return
        container.clear()

        # Collect keys that belong to predefined groups
        predefined_keys: set[str] = {d["key"] for d in ENV_VAR_DEFINITIONS}

        # Group definitions by group key
        groups: Dict[str, list] = {}
        for defn in ENV_VAR_DEFINITIONS:
            groups.setdefault(defn["group"], []).append(defn)

        # Collect custom (user-added) keys
        custom_keys = [k for k in self.env_data if k not in predefined_keys]

        with container:
            # ── Predefined groups ──
            for group_key, items in groups.items():
                label_key = self._GROUP_LABELS.get(group_key, group_key)
                icon = self._GROUP_ICONS.get(group_key, "settings")

                with ui.column().classes(
                    "w-full settings-section-card"
                ).style(self._SECTION_CARD_LAYOUT):
                    # Section header
                    with ui.row().classes(
                        "w-full items-center gap-2 settings-section-header"
                    ).style(self._SECTION_HEADER_LAYOUT):
                        ui.icon(icon, size="18px").style(
                            f"color: {COLORS['primary']};"
                        )
                        ui.label(t(label_key)).classes(
                            "text-body1 text-weight-bold"
                        ).style(
                            "color: var(--color-text); font-size: 14px;"
                        )

                    # Fields
                    with ui.column().classes("w-full gap-0").style(
                        "padding: 12px 16px;"
                    ):
                        for defn in items:
                            self._render_env_row(
                                defn["key"],
                                deletable=False,
                                desc=defn,
                            )

            # ── Custom (user-added) env vars ──
            if custom_keys:
                with ui.column().classes(
                    "w-full settings-section-card"
                ).style(self._SECTION_CARD_LAYOUT):
                    with ui.row().classes(
                        "w-full items-center gap-2 settings-section-header"
                    ).style(self._SECTION_HEADER_LAYOUT):
                        ui.icon("tune", size="18px").style(
                            f"color: {COLORS['primary']};"
                        )
                        ui.label(t("env_group_custom")).classes(
                            "text-body1 text-weight-bold"
                        ).style(
                            "color: var(--color-text); font-size: 14px;"
                        )

                    with ui.column().classes("w-full gap-0").style(
                        "padding: 12px 16px;"
                    ):
                        for env_key in custom_keys:
                            self._render_env_row(env_key, deletable=True)

            # ── Add new env var button ──
            with ui.row().classes("w-full justify-center").style(
                "padding-top: 8px;"
            ):
                def _add_env_var(_e):
                    with ui.dialog() as dlg, ui.card().style(
                        "min-width: 340px; "
                        "background: var(--color-surface) !important; "
                        "border: 1px solid var(--card-border);"
                    ):
                        ui.label(t("env_add_title")).classes(
                            "text-subtitle1 text-weight-bold"
                        ).style("color: var(--color-text);")

                        key_input = (
                            ui.input(
                                label="Key",
                                placeholder="MY_ENV_VAR",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        val_input = (
                            ui.input(
                                label="Value",
                                placeholder=t("env_empty_hint"),
                            )
                            .props("dense outlined")
                            .classes("w-full q-mt-sm")
                        )

                        with ui.row().classes(
                            "w-full justify-end gap-2 q-mt-sm"
                        ):
                            ui.button(
                                icon="close", on_click=dlg.close
                            ).props("flat dense")

                            def _confirm(
                                _ev, d=dlg, ki=key_input, vi=val_input
                            ):
                                name = (ki.value or "").strip()
                                if not name:
                                    ui.notify(
                                        "Key cannot be empty",
                                        type="warning",
                                    )
                                    return
                                if name in self.env_data:
                                    ui.notify(
                                        f"'{name}' already exists",
                                        type="warning",
                                    )
                                    return
                                self.env_data[name] = (vi.value or "").strip()
                                d.close()
                                self._render_env_tab()

                            ui.button(
                                icon="check", on_click=_confirm
                            ).props("flat dense").style(
                                f"color: {COLORS['primary']};"
                            )
                    dlg.open()

                add_btn = ui.button(
                    icon="add", on_click=_add_env_var
                ).props("flat round dense")
                add_btn.style(
                    f"color: {COLORS['primary']}; "
                    "border: 1px dashed var(--card-border); "
                    "opacity: 0.7;"
                )
                add_btn.tooltip(t("env_add_title"))

    def _render_env_row(
        self,
        env_key: str,
        *,
        deletable: bool = False,
        desc: dict | None = None,
    ):
        """Render a single environment variable row."""
        current_val = self.env_data.get(env_key, "")

        with ui.row().classes("w-full items-center").style(self._ROW_STYLE):
            # Label
            ui.label(env_key).classes(
                "text-body2 text-weight-medium"
            ).style(self._LABEL_STYLE)

            # Input
            inp = (
                ui.input(
                    value=current_val,
                    placeholder=t("env_empty_hint"),
                )
                .props("dense outlined")
                .classes("flex-1")
            )
            inp.on_value_change(
                lambda e, k=env_key: self.env_data.__setitem__(
                    k, e.value or ""
                )
            )

            # Delete button
            if deletable:
                def _do_delete(_e, k=env_key):
                    self.env_data.pop(k, None)
                    self._render_env_tab()

                del_btn = ui.button(
                    icon="remove_circle_outline", on_click=_do_delete
                ).props("flat round dense size=sm")
                del_btn.style(
                    "color: var(--color-text-secondary); "
                    "opacity: 0.5; margin-left: 4px;"
                )
                del_btn.tooltip(f"Delete '{env_key}'")

        # Description hint (for predefined vars)
        if desc:
            lang = get_i18n().lang
            desc_key = "desc_zh" if lang == "zh" else "desc_en"
            hint = desc.get(desc_key, desc.get("desc_en", ""))
            if hint:
                ui.label(hint).classes("text-caption").style(
                    "color: var(--color-text-secondary); "
                    "padding-left: 204px; margin-top: -4px; "
                    "margin-bottom: 4px;"
                )

    # ── Dialog construction ──────────────────────────────────────────

    def build(self):
        """Build the full dialog shell (call once)."""
        ui.add_css(_SETTINGS_CSS)

        self.dialog = ui.dialog().props(
            "maximized transition-show=slide-up transition-hide=slide-down"
        )
        with self.dialog, ui.card().classes(
            "w-full h-full q-pa-none settings-dialog-bg"
        ).style("display: flex; flex-direction: column; overflow: hidden;"):
            # ── Sticky top bar ───────────────────────────────────────
            with ui.row().classes(
                "w-full items-center justify-between q-px-md q-py-sm"
            ).style(
                f"border-bottom: 1px solid {COLORS['accent']}; "
                "flex-shrink: 0; z-index: 10;"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("settings", size="26px").style(
                        f"color: {COLORS['primary']};"
                    )
                    with ui.column().classes("gap-0"):
                        ui.label(t("settings_title")).classes(
                            "text-h6 text-weight-bold"
                        ).style("color: var(--color-text);")
                        ui.label(t("settings_desc")).classes(
                            "text-caption"
                        ).style("color: var(--color-text-secondary);")

                with ui.row().classes("items-center gap-2"):
                    reload_btn = ui.button(
                        t("reload"),
                        icon="refresh",
                        on_click=self._reload_current,
                    ).props("flat dense")
                    reload_btn.classes("modern-btn-secondary")
                    reload_btn.classes(
                        remove="bg-primary bg-secondary text-white"
                    )

                    save_btn = ui.button(
                        t("save"),
                        icon="save",
                        on_click=self._save_current,
                    ).props("dense")
                    save_btn.classes("modern-btn-primary")
                    save_btn.classes(
                        remove="bg-primary bg-secondary text-white"
                    )

                    close_btn = ui.button(
                        icon="close", on_click=self.dialog.close
                    ).props("flat round dense")
                    close_btn.style(f"color: {COLORS['accent']};")

            # ── Tabs + scrollable content ────────────────────────────
            with ui.column().classes("w-full").style(
                "flex: 1; overflow: hidden; "
                "display: flex; flex-direction: column;"
            ):
                with ui.tabs().classes("w-full").style(
                    "flex-shrink: 0;"
                ) as tabs:
                    prompts_tab = ui.tab(t("prompts_config"))
                    model_tab = ui.tab(t("model_config"))
                    general_tab = ui.tab(t("general_config"))
                    env_tab = ui.tab(t("env_config"))

                tab_map = {
                    t("prompts_config"): "prompts",
                    t("model_config"): "model",
                    t("general_config"): "general",
                    t("env_config"): "env",
                }

                def _on_tab(e, tm=tab_map):
                    self._active_key = tm.get(e.value, "prompts")
                    self._ensure_tab_loaded(self._active_key)

                tabs.on_value_change(_on_tab)

                with ui.tab_panels(tabs, value=prompts_tab).classes(
                    "w-full"
                ).style("flex: 1; overflow: auto; padding: 8px;"):
                    with ui.tab_panel(prompts_tab).classes("q-pa-sm"):
                        self.tab_containers["prompts"] = (
                            ui.column().classes("w-full gap-3")
                        )
                    with ui.tab_panel(model_tab).classes("q-pa-sm"):
                        self.tab_containers["model"] = (
                            ui.column().classes("w-full gap-3")
                        )
                    with ui.tab_panel(general_tab).classes("q-pa-sm"):
                        self.tab_containers["general"] = (
                            ui.column().classes("w-full gap-3")
                        )
                    with ui.tab_panel(env_tab).classes("q-pa-sm"):
                        self._env_container = (
                            ui.column().classes("w-full gap-3")
                        )

    def open(self):
        """Open the dialog, loading fresh data on demand."""
        if self.dialog is None:
            self.build()
        self._active_key = "prompts"
        # Only load + render the first visible tab; others load lazily
        self.data["prompts"] = self._load("prompts")
        self._render_tab("prompts")
        self._pending_tabs = {"model", "general", "env"}
        self.dialog.open()

    def _ensure_tab_loaded(self, key: str):
        """Lazy-load a tab's data and render it on first switch."""
        if key not in self._pending_tabs:
            return
        self._pending_tabs.discard(key)
        if key == "env":
            self.env_data = load_env_config()
            self._render_env_tab()
        else:
            self.data[key] = self._load(key)
            self._render_tab(key)


def create_settings_dialog() -> SettingsDialog:
    """Create and return a SettingsDialog (call once per page)."""
    dlg = SettingsDialog()
    dlg.build()
    return dlg
