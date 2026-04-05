"""
Advanced Input Components
Includes: Editable Slider, Toggle Switch, Searchable Dropdown
From sd-scripts/gui with enhancements
"""

from nicegui import ui, core
from typing import Optional, Callable, Dict, Any, List
from gui.utils.i18n import t, get_i18n
from gui.theme import COLORS
import uuid


def editable_slider(
    label_key: str,
    value_ref: Dict[str, Any],
    value_key: str,
    min_val: float,
    max_val: float,
    step: float = 1,
    decimals: int = 0,
    label_default: str = None,
    flex: int = 1,
    on_change: Callable = None,
):
    """
    Create an editable slider component with two-way binding

    Args:
        label_key: Translation key for the label
        value_ref: Dictionary containing the value (e.g., self.config)
        value_key: Key in the dictionary for this value
        min_val: Minimum value
        max_val: Maximum value
        step: Step size
        decimals: Number of decimal places to display
        label_default: Default label text if translation not found
        flex: Flex grow value for layout
        on_change: Callback when value changes
    """
    with ui.element("div").classes("editable-slider").style(f"flex: {flex}; margin: 0; padding: 0; min-width: 140px;"):
        # Label row with value display
        with ui.row().classes("w-full items-center justify-between no-wrap").style("margin: 0; padding: 0; min-height: 20px;"):
            label_el = (
                ui.label(t(label_key, label_default or label_key))
                .classes("slider-label")
                .style(
                    "min-width: 60px; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0; padding: 0;"
                )
            )

            # Editable value display
            current_val = value_ref.get(value_key, min_val)
            value_btn = ui.button(f"{current_val:.{decimals}f}").props('flat dense type="button"').classes("slider-value")
            value_btn.style("padding: 0 4px; min-height: 18px; height: 18px; font-size: 11px; margin: 0;")

        # Note: i18n bind not supported in this implementation
        pass

        # NiceGUI native slider
        slider = (
            ui.slider(min=min_val, max=max_val, step=step, value=current_val)
            .classes("w-full")
            .style("margin: 0; padding: 0; min-height: 16px; height: 16px;")
        )
        # 移除颜色props，让CSS控制颜色
        slider.props("dense")

        # Sync value display when slider changes
        def sync_display():
            val = slider.value
            value_ref[value_key] = val
            value_btn.set_text(f"{val:.{decimals}f}")
            if on_change:
                on_change(val)

        slider.on_value_change(sync_display)

        # Click on value to edit
        def start_edit():
            current_val = value_ref.get(value_key, min_val)
            value_btn.visible = False

            edit_container = ui.element("span")
            with edit_container:
                edit_input = ui.input(value=f"{current_val:.{decimals}f}").classes("slider-edit-input").style("width: 60px;")

            finished = [False]

            def finish_edit():
                if finished[0]:
                    return
                finished[0] = True

                try:
                    new_val = float(edit_input.value)
                    new_val = max(min_val, min(max_val, new_val))
                    if decimals == 0:
                        new_val = int(new_val)
                    else:
                        new_val = round(new_val, decimals)

                    value_ref[value_key] = new_val
                    slider.set_value(new_val)
                    value_btn.set_text(f"{new_val:.{decimals}f}")

                    if on_change:
                        on_change(new_val)
                except ValueError:
                    pass
                finally:
                    edit_container.delete()
                    value_btn.visible = True

            edit_input.on("blur", finish_edit)
            edit_input.on("keyup.enter", finish_edit)

            # 使用 NiceGUI 元素 id 找到 Quasar 容器，然后找到内部 input
            container_id = edit_input.id
            ui.run_javascript(f"""
                (function() {{
                    // 通过 NiceGUI 元素 id 找到 Quasar 容器
                    var container = document.getElementById('c{container_id}');
                    if (!container) {{
                        console.error('Editable slider: container not found');
                        return;
                    }}
                    
                    // 在容器内找到实际的 input 元素
                    var input = container.querySelector('input.q-field__native') || 
                                container.querySelector('input') ||
                                container;
                    
                    // 聚焦并选中
                    if (input && input.focus) {{
                        input.focus();
                        if (input.select) input.select();
                    }}
                    
                    // 获取 q-field 容器
                    var field = container.closest ? container.closest('.q-field') : container;
                    
                    // 点击处理器 - 点击外部时 blur
                    function onClickOutside(e) {{
                        if (!field || !field.contains(e.target)) {{
                            if (input && input.blur) input.blur();
                            document.removeEventListener('mousedown', onClickOutside);
                            document.removeEventListener('touchstart', onClickOutside);
                        }}
                    }}
                    
                    // 延迟绑定事件
                    setTimeout(function() {{
                        document.addEventListener('mousedown', onClickOutside);
                        document.addEventListener('touchstart', onClickOutside);
                    }}, 100);
                }})();
            """)

        value_btn.on_click(start_edit)

    return slider


def toggle_switch(label_key: str, value_ref: Dict[str, Any], value_key: str, label_default: str = None, on_change: Callable = None):
    """
    Create a toggle switch button (turn on/off style)

    Args:
        label_key: Translation key for the label
        value_ref: Dictionary containing the value
        value_key: Key in the dictionary for this value
        label_default: Default label text if translation not found
        on_change: Callback when value changes
    """
    value = value_ref.get(value_key, False)

    btn = ui.button().props('flat unelevated type="button"').classes(f"toggle-container {'active' if value else ''}")

    with btn:
        with ui.element("div").classes("toggle-switch"):
            ui.element("div").classes("toggle-knob")

        label_el = ui.label(t(label_key, label_default or label_key)).classes("toggle-label")

        status_text = t("status_on") if value else t("status_off")
        status_label = ui.label(status_text).classes("toggle-status")

    def apply_value(new_value: bool):
        new_value = bool(new_value)
        value_ref[value_key] = new_value

        if new_value:
            btn.classes("active")
            status_label.set_text(t("status_on"))
        else:
            btn.classes(remove="active")
            status_label.set_text(t("status_off"))

        if on_change:
            on_change(new_value)

    def toggle():
        apply_value(not value_ref.get(value_key, False))

    btn.on_click(toggle)
    btn.set_toggle_value = apply_value
    return btn


def toggle_switch_simple(label: str, value: bool = False, on_change: Callable = None):
    """
    Create a toggle switch button with simple value binding

    Args:
        label: Label text
        value: Initial value
        on_change: Callback when value changes, receives new_value
    """
    btn = ui.button().props('flat unelevated type="button"').classes(f"toggle-container {'active' if value else ''}")

    with btn:
        with ui.element("div").classes("toggle-switch"):
            ui.element("div").classes("toggle-knob")

        label_el = ui.label(label).classes("toggle-label")

        from gui.utils.i18n import t

        status_text = t("status_on") if value else t("status_off")
        status_label = ui.label(status_text).classes("toggle-status")

    # Store current value
    current_value = [value]

    def toggle():
        new_value = not current_value[0]
        current_value[0] = new_value

        if new_value:
            btn.classes("active")
            status_label.set_text(t("status_on"))
        else:
            btn.classes(remove="active")
            status_label.set_text(t("status_off"))

        if on_change:
            on_change(new_value)

    btn.on_click(toggle)
    return btn, lambda: current_value[0]


def searchable_select(
    options: Dict[str, str],
    value_ref: Dict[str, Any],
    value_key: str,
    label_key: str = None,
    label_default: str = None,
    placeholder_key: str = None,
    placeholder_default: str = "Search or select...",
    on_change: Callable = None,
    classes: str = "",
    style: str = "",
):
    """
    Create a searchable dropdown select with input filtering

    Args:
        options: Dictionary of {value: label} pairs
        value_ref: Dictionary containing the value
        value_key: Key in the dictionary for this value
        label_key: Translation key for the label
        label_default: Default label text
        placeholder_key: Translation key for placeholder
        placeholder_default: Default placeholder text
        on_change: Callback when value changes
        classes: Additional CSS classes
        style: Additional inline styles
    """
    current_value = value_ref.get(value_key, list(options.keys())[0] if options else None)

    with ui.column().classes(f"w-full {classes}").style(style):
        if label_key:
            label_el = ui.label(t(label_key, label_default or label_key)).classes("text-sm font-medium q-mb-xs")

            # Note: i18n bind not supported in this implementation
            pass

        select = ui.select(options, value=current_value, label="").classes("w-full")

        # Enable search/filter functionality
        select.props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
        select.props(f'placeholder="{t(placeholder_key, placeholder_default)}"')

        def on_value_change(e):
            value_ref[value_key] = e.value
            if on_change:
                on_change(e.value)

        select.on_value_change(on_value_change)

    return select


def model_selector(
    value_ref: Dict[str, Any],
    value_key: str = "pretrained_model",
    label_key: str = "pretrained_model",
    label_default: str = "Pretrained Model",
    on_change: Callable = None,
):
    """
    Create a searchable model selector with common SD models
    """
    # Common model options - can be extended
    model_options = {
        "": t("select_model", "Select a model..."),
        "runwayml/stable-diffusion-v1-5": "SD 1.5",
        "stabilityai/stable-diffusion-2-1": "SD 2.1",
        "stabilityai/stable-diffusion-xl-base-1.0": "SDXL 1.0",
        "stabilityai/stable-diffusion-xl-refiner-1.0": "SDXL Refiner",
        "madebyollin/sdxl-vae-fp16-fix": "SDXL VAE FP16",
        "black-forest-labs/FLUX.2-dev": "FLUX.2 Dev",
        "black-forest-labs/FLUX.2-schnell": "FLUX.2 Schnell",
    }

    return searchable_select(
        options=model_options,
        value_ref=value_ref,
        value_key=value_key,
        label_key=label_key,
        label_default=label_default,
        placeholder_key="search_model",
        placeholder_default="Search or type model name...",
        on_change=on_change,
    )


def styled_select(
    options: Dict[str, str],
    value: str = None,
    label: str = "",
    icon: str = "arrow_drop_down",
    icon_color: str = None,
    placeholder: str = "Search or select...",
    on_change: Callable = None,
    flex: int = None,
    new_value_mode: str = None,
    searchable: bool = True,
):
    """
    创建带图标前缀和小标题的现代化下拉框

    Args:
        options: 选项字典 {value: label}
        value: 默认值
        label: 小标题标签
        icon: 前缀图标
        icon_color: 图标颜色 (默认使用主色)
        placeholder: 占位符文本
        on_change: 变更回调
        flex: flex 布局权重
        new_value_mode: 新模式 (add/add-unique/toggle)
        searchable: 是否启用搜索输入模式
    """
    from gui.theme import COLORS

    icon_color = icon_color or COLORS["primary"]
    style = f"flex: {flex};" if flex else ""

    with ui.column().classes("w-full styled-select-container").style(style):
        # 小标题带图标
        if label:
            with ui.row().classes("items-center gap-2 q-mb-xs"):
                ui.icon(icon, size="18px").style(f"color: {icon_color};")
                ui.label(label).classes("text-caption text-weight-medium").style("color: var(--color-text-secondary);")

        # 下拉框 - 使用标准样式（非outlined）避免深色块问题
        select = ui.select(options=options, value=value, label="").classes("w-full modern-select force-light-bg")

        # 不使用 outlined，避免 Quasar 默认深色背景
        dropdown_icon = "search" if searchable else "arrow_drop_down"
        props = f'dense stack-label dropdown-icon="{dropdown_icon}" placeholder="{placeholder}"'
        if searchable:
            props += ' use-input fill-input hide-selected input-debounce="0"'
        if new_value_mode:
            props += f' new-value-mode="{new_value_mode}"'
        select.props(props)

        if on_change:
            select.on_value_change(lambda e: on_change(e.value))

    return select


def styled_input(
    value: str = "",
    label: str = "",
    icon: str = "edit",
    icon_color: str = None,
    placeholder: str = "",
    password: bool = False,
    on_change: Callable = None,
    flex: int = None,
):
    """
    创建带图标前缀和小标题的现代化输入框

    Args:
        value: 默认值
        label: 小标题标签
        icon: 前缀图标
        icon_color: 图标颜色
        placeholder: 占位符文本
        password: 是否为密码输入
        on_change: 变更回调
        flex: flex 布局权重
    """
    from gui.theme import COLORS

    icon_color = icon_color or COLORS["primary"]
    style = f"flex: {flex};" if flex else ""

    with ui.column().classes("w-full styled-input-container").style(style):
        # 小标题带图标
        if label:
            with ui.row().classes("items-center gap-2 q-mb-xs"):
                ui.icon(icon, size="18px").style(f"color: {icon_color};")
                ui.label(label).classes("text-caption text-weight-medium").style("color: var(--color-text-secondary);")

        # 输入框 - 使用标准样式（非outlined）避免深色块问题
        inp = ui.input(value=value, label="", password=password).classes("w-full modern-input force-light-bg")
        inp.props(f'dense stack-label placeholder="{placeholder}"')

        if on_change:
            inp.on("change", lambda e: on_change(e.value))

    return inp
