"""
Modern Theme System for Musubi Tuner GUI
现代化主题系统 - 深色模式、圆角、渐变、阴影

整合自 sd-scripts/gui/styles.py - 提供两种主题系统:
1. Modern Theme (默认) - 深绿+金色自然主题
2. Green Gold Theme (可选) - 明亮绿金主题 (来自 sd-scripts)
"""

from nicegui import ui
from typing import Dict, Any, Optional

# ============== Theme 1: Modern Theme (默认) ==============

MODERN_COLORS = {
    "primary": "#059669",  # Deep Emerald 600 (dress color)
    "primary_dark": "#047857",  # Deep Emerald 700
    "primary_light": "#4ade80",  # Light Green 400 (hair highlight)
    "secondary": "#fbbf24",  # Golden Amber 400 (horns/accessories)
    "accent": "#10b981",  # Emerald 500 (bright green)
    "success": "#22c55e",  # Green 500
    "warning": "#f59e0b",  # Amber 500
    "error": "#ef4444",  # Red 500
    "info": "#2dd4bf",  # Teal 400
    "background": "#022c22",  # Deep Forest Green
    "surface": "#064e3b",  # Dark Emerald
    "surface_light": "#065f46",  # Medium Emerald
    "surface_dark": "#022c22",  # Deep Forest
    "text": "#ecfdf5",  # Mint 50 (light green-white)
    "text_secondary": "#6ee7b7",  # Light Emerald 300
    "text_muted": "#34d399",  # Emerald 400
    "border": "#059669",  # Emerald 600
    # 额外颜色 - 兼容 sd-scripts 样式
    "blue": "#3b82f6",
    "blue_light": "#60a5fa",
    "blue_dark": "#2563eb",
    "mint": "#6ee7b7",
    "emerald": "#10b981",
    "forest": "#166534",
    "bg_light": "#f0fdf4",
    "text_on_primary": "#ffffff",
    "text_on_accent": "#78350f",
    "bg_dark": "#0f281e",
    "text_dark": "#e2e8f0",
}

# CSS Classes for Modern Styling
MODERN_CLASSES = {
    # Cards
    "card": "modern-card",
    "card_hover": "modern-card-hover",
    "card_header": "modern-card-header",
    # Buttons
    "btn_primary": "modern-btn-primary",
    "btn_secondary": "modern-btn-secondary",
    "btn_danger": "modern-btn-danger",
    "btn_success": "modern-btn-success",
    "btn_ghost": "modern-btn-ghost",
    # Inputs
    "input": "modern-input",
    "select": "modern-select",
    # Layout
    "page_container": "modern-page-container",
    "section": "modern-section",
    "section_title": "modern-section-title",
    "header": "modern-header",
    # Navigation
    "nav_btn": "modern-nav-btn",
    "nav_btn_active": "modern-nav-btn-active",
    # Tags/Badges
    "badge": "modern-badge",
    "badge_primary": "modern-badge-primary",
    "badge_success": "modern-badge-success",
    # 来自 sd-scripts 的额外类
    "section_card": "section-card",
    "gold_btn": "gold-btn",
    "green_btn": "green-btn",
    "red_btn": "red-btn",
    "footer_green": "footer-green",
    "header_green": "header-green",
}


# ============== Theme 2: Green Gold Theme (来自 sd-scripts) ==============


def get_green_gold_colors():
    """Green/Gold theme color palette from sd-scripts"""
    return {
        # Deep green (like the dress) - Primary
        "primary": "#1a4d3a",
        "primary_light": "#2d6a4f",
        "primary_dark": "#0d3326",
        # Medium green (like the hair)
        "secondary": "#4ade80",
        "secondary_light": "#6ee7a0",
        "secondary_dark": "#22c55e",
        # Gold/Yellow (crown accents)
        "accent": "#fbbf24",
        "accent_light": "#fcd34d",
        "accent_dark": "#f59e0b",
        # Blue (for primary action buttons)
        "blue": "#3b82f6",
        "blue_light": "#60a5fa",
        "blue_dark": "#2563eb",
        # Additional greens
        "mint": "#6ee7b7",
        "emerald": "#10b981",
        "forest": "#166534",
        # Neutral
        "bg_light": "#f0fdf4",
        "border": "#cbd5e1",
        "text_on_primary": "#ffffff",
        "text_on_accent": "#78350f",
        # Dark mode
        "bg_dark": "#0f281e",
        "text_dark": "#e2e8f0",
        # 兼容 modern theme
        "success": "#22c55e",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "info": "#2dd4bf",
        "background": "#022c22",
        "surface": "#064e3b",
        "surface_light": "#065f46",
        "surface_dark": "#022c22",
        "text": "#ecfdf5",
        "text_secondary": "#6ee7b7",
        "text_muted": "#34d399",
        "primary_dark_theme": "#047857",
    }


def get_green_gold_styles(COLORS: Dict[str, str]) -> str:
    """Generate complete CSS styles with color interpolation from sd-scripts"""

    return f"""
    <style>
    :root {{
        --q-primary: #f59e0b !important;  /* Gold/Amber instead of blue */
        --q-secondary: {COLORS["secondary"]} !important;
        --q-accent: {COLORS["accent"]} !important;
        --q-positive: {COLORS["emerald"]} !important;
    }}
    
    /* ===== Base Styles - Flash Prevention ===== */
    /* Force html element to never be black */
    html {{
        background: #f0fdf4 !important;
        background-color: #f0fdf4 !important;
    }}
    
    /* Force body background - prevents black flash */
    body {{
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%) !important;
        background-color: #f0fdf4 !important;
        background-attachment: fixed !important;
    }}
    
    /* Force Vue app root to never be black */
    #app, #q-app {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    
    /* Force Quasar page container */
    .q-page, .nicegui-page, [class*="page-container"] {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    
    /* ===== Header & Footer ===== */
    .header-green {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_light"]} 100%) !important;
        color: {COLORS["text_on_primary"]} !important;
        border-bottom: 3px solid {COLORS["accent"]} !important;
    }}
    
    .footer-green {{
        background: linear-gradient(135deg, {COLORS["primary_dark"]} 0%, {COLORS["primary"]} 100%) !important;
        color: {COLORS["text_on_primary"]} !important;
        border-top: 3px solid {COLORS["accent"]} !important;
    }}
    
    /* ===== Button Styles ===== */
    /* Gold/Amber button - PRIMARY ACTION */
    .gold-btn {{
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%) !important;
        color: #78350f !important;
        font-weight: bold !important;
        border: 2px solid #d97706 !important;
    }}
    
    .gold-btn:hover {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.5) !important;
    }}
    
    body:not(.dark-mode) .gold-btn .q-btn__content,
    body:not(.dark-mode) .gold-btn .q-icon {{
        color: white !important;
    }}
    body.dark-mode .gold-btn .q-btn__content,
    body.dark-mode .gold-btn .q-icon {{
        color: #78350f !important;
    }}

    /* Green button - BROWSE BUTTONS */
    body:not(.dark-mode) .green-btn {{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }}

    .green-btn:hover {{
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
    }}

    body:not(.dark-mode) .green-btn .q-btn__content {{
        color: white !important;
    }}

    body:not(.dark-mode) .red-btn {{
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        font-weight: bold !important;
    }}
    body.dark-mode .red-btn {{
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: #fef2f2 !important;
        font-weight: bold !important;
    }}
    
    /* ===== Card Styles ===== */
    .section-card {{
        background: linear-gradient(145deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%) !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 15px rgba(26, 77, 58, 0.08) !important;
        transition: all 0.3s ease !important;
        margin-bottom: 8px !important;
    }}
    
    .section-card:hover {{
        box-shadow: 0 8px 25px rgba(26, 77, 58, 0.15) !important;
        border-color: {COLORS["accent"]} !important;
    }}
    
    .section-card .q-card__section {{
        padding: 10px 14px !important;
    }}
    
    /* ===== Title Styles ===== */
    .section-title {{
        color: {COLORS["primary"]} !important;
        font-weight: bold !important;
        font-size: 0.95em !important;
        border-left: 3px solid {COLORS["accent"]} !important;
        padding-left: 8px !important;
        margin-bottom: 8px !important;
        margin-top: 2px !important;
    }}
    
    .section-subtitle {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
        font-size: 0.85em !important;
        margin-top: 12px !important;
        margin-bottom: 6px !important;
        opacity: 0.8;
    }}
    
    /* ===== Tab Styles ===== */
    .q-tab--active {{
        color: {COLORS["primary"]} !important;
        background: linear-gradient(180deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.05) 100%) !important;
        border-bottom: 3px solid {COLORS["accent"]} !important;
    }}
    
    .q-tab__indicator {{
        background: {COLORS["accent"]} !important;
        height: 3px !important;
    }}
    
    .q-tab {{
        min-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    .q-tab__content {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
    }}
    
    .q-tab__icon {{
        font-size: 20px !important;
        margin: 0 !important;
    }}
    
    .q-tab__label {{
        font-size: 14px !important;
        font-weight: 500 !important;
    }}
    
    /* ===== Input/Field Styles - Enhanced ===== */
    .q-field {{
        margin-bottom: 8px !important;
    }}
    
    /* Input control - covers both input and select */
    .q-field__control,
    .q-field--outlined .q-field__control,
    .q-field--filled .q-field__control {{
        border-radius: 10px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid {COLORS["border"]} !important;
        transition: all 0.2s ease !important;
        padding-left: 12px !important;
        min-height: 48px !important;
    }}
    
    /* CRITICAL: Tab panel fields - prevent black flash */
    html body:not(.dark-mode) .q-tab-panel .q-field__control,
    html body:not(.dark-mode) .q-tab-panel .q-field--outlined .q-field__control,
    html body:not(.dark-mode) .q-tab-panel .q-field--filled .q-field__control,
    html body:not(.dark-mode) .q-tab-panel .q-field--standard .q-field__control {{
        background: rgba(255, 255, 255, 0.9) !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* Remove Quasar default field indicators (round bar artifacts) */
    .q-field--outlined .q-field__control::before,
    .q-field--outlined .q-field__control::after,
    .q-field--standard .q-field__control::before,
    .q-field--standard .q-field__control::after,
    .q-field--filled .q-field__control::before,
    .q-field--filled .q-field__control::after {{
        border: none !important;
        display: none !important;
    }}
    
    .q-field__control:hover,
    .q-field--outlined .q-field__control:hover {{
        border-color: {COLORS["secondary"]} !important;
        background: rgba(255, 255, 255, 1) !important;
    }}
    
    .q-field--focused .q-field__control,
    .q-field--outlined.q-field--focused .q-field__control {{
        border-color: {COLORS["emerald"]} !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
        background: rgba(255, 255, 255, 1) !important;
    }}
    
    .q-field__label {{
        color: {COLORS["primary"]} !important;
        font-weight: 500 !important;
        padding-left: 4px !important;
    }}
    
    /* Input text color */
    .q-field__native,
    .q-field__input,
    .q-field__input input,
    .q-input input,
    .q-select input {{
        color: {COLORS["primary"]} !important;
        font-weight: 500 !important;
    }}
    
    /* Placeholder color */
    .q-field__input::placeholder,
    .q-input::placeholder {{
        color: rgba(26, 77, 58, 0.5) !important;
    }}
    
    /* Select dropdown specific */
    .q-select .q-field__control {{
        cursor: pointer !important;
    }}
    
    /* Path input specific - browse button area */
    .q-field__append,
    .q-field__prepend {{
        color: {COLORS["primary"]} !important;
    }}
    
    /* File picker and path selector styles - Clean Style matching select dropdown */
    .q-file .q-field__control,
    .path-input .q-field__control {{
        background: transparent !important;
        border: 2px solid rgba(5, 150, 105, 0.25) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }}
    
    .path-input .q-field__control:hover {{
        border-color: rgba(5, 150, 105, 0.5) !important;
        background: rgba(255, 255, 255, 0.5) !important;
    }}
    
    .path-input.q-field--focused .q-field__control {{
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.15) !important;
        background: rgba(255, 255, 255, 0.8) !important;
    }}
    
    /* Dark mode path input - matching select */
    body.dark-mode .path-input .q-field__control {{
        background: transparent !important;
        border-color: rgba(251, 191, 36, 0.3) !important;
        box-shadow: none !important;
    }}
    
    body.dark-mode .path-input .q-field__control:hover {{
        border-color: rgba(251, 191, 36, 0.5) !important;
        background: rgba(6, 78, 59, 0.3) !important;
    }}
    
    body.dark-mode .path-input.q-field--focused .q-field__control {{
        background: rgba(6, 78, 59, 0.5) !important;
        border-color: #4ade80 !important;
        box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.2) !important;
    }}
    
    /* Path input native - transparent like select */
    .path-input .q-field__native {{
        background: transparent !important;
        color: var(--color-text) !important;
    }}
    
    /* ===== Slider Styles ===== */
    .editable-slider {{
        margin-bottom: 4px !important;
        flex: 1 1 0 !important;
        min-width: 120px !important;
        max-width: 100% !important;
        padding: 2px 0 !important;
    }}
    
    .slider-label-row {{
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 6px !important;
    }}
    
    .slider-label {{
        font-size: 11px !important;
        font-weight: 500 !important;
        color: {COLORS["primary"]} !important;
        line-height: 1.2 !important;
    }}
    
    /* Slider value button - override text-primary with maximum specificity */
    html body .q-btn[class*="slider-value"],
    html body button[class*="slider-value"],
    html body .q-btn.text-primary[class*="slider-value"],
    html body button.q-btn.text-primary[class*="slider-value"],
    html body [class*="editable-slider"] .q-btn[class*="slider-value"],
    html body [class*="editable-slider"] button[class*="slider-value"] {{
        font-size: 10px !important;
        font-family: 'Consolas', monospace !important;
        color: #f59e0b !important;
        background: transparent !important;
        border: 1px solid rgba(245, 158, 11, 0.5) !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        text-transform: none !important;
        min-height: 18px !important;
        height: 18px !important;
        padding: 0 4px !important;
        min-width: 32px !important;
    }}
    
    html body .q-btn[class*="slider-value"]:hover,
    html body button[class*="slider-value"]:hover,
    html body .q-btn.text-primary[class*="slider-value"]:hover,
    html body button.q-btn.text-primary[class*="slider-value"]:hover,
    html body [class*="editable-slider"] .q-btn[class*="slider-value"]:hover,
    html body [class*="editable-slider"] button[class*="slider-value"]:hover {{
        background: rgba(245, 158, 11, 0.15) !important;
        border-color: rgba(245, 158, 11, 0.8) !important;
        color: #f59e0b !important;
    }}
    
    .slider-container {{
        position: relative !important;
        height: 20px !important;
        display: flex !important;
        align-items: center !important;
    }}
    
    .slider-track {{
        position: absolute !important;
        left: 0 !important;
        right: 0 !important;
        height: 6px !important;
        background: transparent !important;
        border-radius: 3px !important;
        box-shadow: none !important;
    }}
    
    .slider-fill {{
        position: absolute !important;
        left: 0 !important;
        height: 6px !important;
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%) !important;
        border-radius: 3px !important;
        transition: width 0.15s ease !important;
    }}
    
    .slider-thumb {{
        position: absolute !important;
        width: 18px !important;
        height: 18px !important;
        background: white !important;
        border: 3px solid {COLORS["emerald"]} !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
        transition: left 0.15s ease !important;
        z-index: 2 !important;
    }}
    
    .slider-input {{
        position: absolute !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        opacity: 0 !important;
        cursor: pointer !important;
        z-index: 3 !important;
        margin: 0 !important;
    }}
    
    .slider-edit-input {{
        width: 70px !important;
        font-size: 12px !important;
        font-family: 'Consolas', monospace !important;
        text-align: right !important;
        border: 2px solid #f59e0b !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
        outline: none !important;
        background: white !important;
        color: #78350f !important;
    }}
    
    body.dark-mode .slider-edit-input {{
        background: rgba(30, 20, 10, 0.9) !important;
        color: #fbbf24 !important;
        border-color: #f59e0b !important;
    }}
    
    /* ===== Quasar Slider Override - Gold Theme ===== */
    /* Override inline color classes */
    .q-slider [class*="text-#"] {{
        color: #f59e0b !important;
    }}
    
    .q-slider [class*="bg-#"] {{
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%) !important;
    }}
    
    /* ===== Quasar Slider Styles - 金色/透明对比 ===== */
    
    /* 滑块整体变量 */
    .q-slider {{
        --q-color-primary: #f59e0b !important;
        --q-primary: #f59e0b !important;
    }}
    
    /* 轨道容器 - 完全透明 */
    .q-slider__track-container,
    .q-slider__track-container--h,
    .q-slider__track-container--v {{
        background: transparent !important;
        border-radius: 3px !important;
    }}
    
    /* 轨道 - 默认透明 */
    .q-slider__track {{
        background: transparent !important;
    }}
    
    /* 轨道 - 有内容/宽度时金色（已填充部分） */
    .q-slider__track[style*="width"]:not([style*="width: 0"]) {{
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%) !important;
    }}
    
    /* 深色模式下的轨道 - 透明 */
    body.dark-mode .q-slider__track-container,
    body.dark-mode .q-slider__track-container--h,
    body.dark-mode .q-slider__track-container--v {{
        background: transparent !important;
    }}
    
    /* 滑块按钮 */
    .q-slider__thumb {{
        color: #f59e0b !important;
    }}
    
    .q-slider__thumb-circle {{
        background: #f59e0b !important;
        border-color: #f59e0b !important;
    }}
    
    /* 禁用状态 - 也透明 */
    .q-slider--disabled .q-slider__track-container,
    .q-slider--disabled .q-slider__track-container--h,
    .q-slider--disabled .q-slider__track-container--v {{
        background: transparent !important;
    }}
    
    /* ===== Toggle Switch Styles ===== */
    .toggle-container {{
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        cursor: pointer !important;
        padding: 4px 10px !important;
        background: rgba(134, 239, 172, 0.3) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(5, 150, 105, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
        min-height: 28px !important;
        margin: 2px !important;
    }}
    
    body.dark-mode .toggle-container {{
        background: rgba(6, 78, 59, 0.6) !important;
        border-color: rgba(251, 191, 36, 0.3) !important;
    }}
    
    .toggle-container .q-btn__content {{
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 0 !important;
    }}
    
    .toggle-container:hover {{
        background: rgba(134, 239, 172, 0.5) !important;
        border-color: rgba(5, 150, 105, 0.6) !important;
    }}
    
    body.dark-mode .toggle-container:hover {{
        background: rgba(6, 78, 59, 0.8) !important;
        border-color: rgba(251, 191, 36, 0.5) !important;
    }}
    
    .toggle-container.active {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        border-color: #d97706 !important;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.4) !important;
    }}
    
    .toggle-switch {{
        width: 36px !important;
        height: 20px !important;
        background: rgba(5, 150, 105, 0.3) !important;
        border-radius: 10px !important;
        position: relative !important;
        transition: all 0.3s ease !important;
        flex-shrink: 0 !important;
    }}
    
    body.dark-mode .toggle-switch {{
        background: rgba(6, 78, 59, 0.8) !important;
    }}
    
    .toggle-container.active .toggle-switch {{
        background: {COLORS["primary"]} !important;
    }}
    
    .toggle-knob {{
        width: 16px !important;
        height: 16px !important;
        background: white !important;
        border-radius: 50% !important;
        position: absolute !important;
        top: 2px !important;
        left: 2px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }}
    
    .toggle-container.active .toggle-knob {{
        left: 18px !important;
    }}
    
    .toggle-label {{
        font-size: 12px !important;
        font-weight: 500 !important;
        color: {COLORS["primary"]} !important;
        user-select: none !important;
        white-space: nowrap !important;
    }}
    
    .toggle-status {{
        font-size: 10px !important;
        color: {COLORS["primary_light"]} !important;
        font-weight: 600 !important;
        margin-left: 4px !important;
    }}
    
    .toggle-container.active .toggle-status {{
        color: {COLORS["primary"]} !important;
    }}
    
    /* ===== Scrollbar Styles ===== */
    ::-webkit-scrollbar {{
        width: 10px !important;
        height: 10px !important;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(16, 185, 129, 0.05) !important;
        border-radius: 5px !important;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS["emerald"]} 0%, #059669 100%) !important;
        border-radius: 5px !important;
        border: 2px solid transparent !important;
        background-clip: content-box !important;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, #34d399 0%, {COLORS["emerald"]} 100%) !important;
    }}
    
    /* Fix: Force light background on tab panel to prevent dark flash */
    /* CRITICAL: Prevent black flash during tab switches */
    /* 同时去掉 tab panel 容器的正方形边框 */
    html body:not(.dark-mode) .q-tab-panels,
    html body:not(.dark-mode) .q-tab-panel,
    html body:not(.dark-mode) .nicegui-tab-panel,
    .q-tab-panels,
    .q-tab-panel,
    .nicegui-tab-panel {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        /* 禁用切换时的过渡动画，避免闪烁 */
        transition: none !important;
        animation: none !important;
    }}
    
    html body:not(.dark-mode) .q-tab-panel {{
        padding: 12px !important;
    }}
    
    /* 禁用tab panel的过渡效果 */
    .q-tab-panels {{
        transition: none !important;
    }}
    
    /* Ensure all direct children of tab panel have proper background */
    html body:not(.dark-mode) .q-tab-panel > div,
    html body:not(.dark-mode) .q-tab-panel > section {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    
    /* Cards inside tab panel - keep rounded card style */
    html body:not(.dark-mode) .q-tab-panel .q-card,
    html body:not(.dark-mode) .q-tab-panel .nicegui-card,
    html body:not(.dark-mode) .q-tab-panel .modern-card,
    html body:not(.dark-mode) .q-tab-panel .q-card.nicegui-card,
    html body:not(.dark-mode) .q-tab-panel .q-card.modern-card,
    html body:not(.dark-mode) .q-tab-panel .nicegui-card.modern-card,
    html body:not(.dark-mode) .q-tab-panel .q-card.nicegui-card.modern-card,
    html body:not(.dark-mode) .q-tab-panel [class*="q-card"][class*="nicegui-card"] {{
        background: rgba(255, 255, 255, 0.5) !important;
        background-color: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(5, 150, 105, 0.15) !important;
        transition: none !important;
        animation: none !important;
    }}
    
    /* ============================================
       Tab Panel Fix (Targeted - NOT nuclear)
       ============================================ */
    
    /* Tab panels and panels transparent background */
    html body:not(.dark-mode) .q-tab-panels,
    html body:not(.dark-mode) .q-tab-panel {{
        background: transparent !important;
        transition: none !important;
    }}
    
    /* Cards in tab panels - glass effect */
    html body:not(.dark-mode) .q-tab-panel .q-card,
    html body:not(.dark-mode) .q-tab-panel .nicegui-card,
    html body:not(.dark-mode) .q-tab-panel .modern-card,
    html body:not(.dark-mode) .q-tab-panel .section-card {{
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(5, 150, 105, 0.15) !important;
    }}
    
    /* Form controls in tab panels */
    html body:not(.dark-mode) .q-tab-panel .q-field__control {{
        background: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* Buttons in tab panels - ensure correct theme */
    html body:not(.dark-mode) .q-tab-panel .gold-btn,
    html body:not(.dark-mode) .q-tab-panel .green-btn,
    html body:not(.dark-mode) .q-tab-panel .modern-btn-primary,
    html body:not(.dark-mode) .q-tab-panel .modern-nav-btn {{
        background: var(--btn-primary-bg) !important;
        color: var(--btn-primary-text) !important;
    }}
    
    /* Flat buttons remain transparent */
    html body:not(.dark-mode) .q-tab-panel .q-btn--flat,
    html body:not(.dark-mode) .q-tab-panel .modern-btn-ghost {{
        background: transparent !important;
    }}
    
    /* ===== Log Output Styles ===== */
    .log-container {{
        background: #1a1a2e !important;
        color: #00ff88 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
        border: 2px solid {COLORS["primary"]} !important;
        border-radius: 8px !important;
    }}
    
    .log-output {{
        background: rgba(15, 40, 30, 0.95) !important;
        color: #e2e8f0 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
        border-radius: 12px !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3) !important;
        line-height: 1.4 !important;
        font-size: 0.85em !important;
    }}
    
    .title-glow {{
        text-shadow: 0 0 20px rgba(74, 222, 128, 0.5), 0 0 40px rgba(251, 191, 36, 0.3) !important;
    }}
    
    /* ===== Language Selector - Light Mode ===== */
    .lang-selector {{
        border-radius: 8px !important;
        font-weight: 500 !important;
    }}
    
    .lang-selector .q-field__control {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid {COLORS["primary"]} !important;
        border-radius: 8px !important;
        height: 40px !important;
        overflow: hidden !important;
    }}
    
    .lang-selector .q-field__control:hover {{
        background: white !important;
        border-color: {COLORS["secondary"]} !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.2) !important;
    }}
    
    .lang-selector .q-field__native {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }}
    
    .lang-selector .q-icon {{
        color: {COLORS["primary"]} !important;
    }}
    
    .lang-selector .q-field__marginal {{
        height: 100% !important;
    }}
    
    /* ===== Dropdown/Menu Styles ===== */
    .q-menu {{
        background: white !important;
        border: 2px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        animation: slideDown 0.25s ease-out !important;
        transform-origin: top !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
    }}
    
    .q-menu .q-item {{
        color: {COLORS["primary"]} !important;
    }}
    
    .q-menu .q-item:hover {{
        background: rgba(16, 185, 129, 0.1) !important;
    }}
    
    body:not(.dark-mode) .q-menu .q-item--active {{
        background: {COLORS["emerald"]} !important;
        color: white !important;
    }}
    
    @keyframes slideDown {{
        from {{
            opacity: 0;
            transform: translateY(-10px) scaleY(0.95);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) scaleY(1);
        }}
    }}
    
    .q-item {{
        color: {COLORS["primary"]} !important;
        transition: all 0.15s ease !important;
        border-radius: 6px !important;
        margin: 2px 4px !important;
    }}
    
    .q-menu .q-item,
    .q-menu .q-item__label,
    .q-menu .q-item__section {{
        color: {COLORS["primary"]} !important;
    }}
    
    .q-item:hover {{
        background: rgba(16, 185, 129, 0.12) !important;
        color: {COLORS["primary_light"]} !important;
    }}
    
    body:not(.dark-mode) .q-item--active,
    body:not(.dark-mode) .q-menu .q-item--active {{
        background: {COLORS["emerald"]} !important;
        color: white !important;
        font-weight: bold !important;
    }}

    body:not(.dark-mode) .q-item--active .q-item__label,
    body:not(.dark-mode) .q-menu .q-item--active .q-item__label {{
        color: white !important;
    }}
    
    .q-select__dropdown-icon {{
        color: {COLORS["primary"]} !important;
        transition: transform 0.3s ease !important;
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        height: auto !important;
    }}
    
    .q-field--focused .q-select__dropdown-icon {{
        transform: translateY(-50%) rotate(180deg) !important;
    }}
    
    .q-virtual-scroll__content {{
        padding: 4px !important;
    }}
    
    .q-select__input {{
        color: {COLORS["primary"]} !important;
        font-weight: 500 !important;
    }}

    /* Fix: Remove color block on selected value in select */
    .q-select .q-field__native > span {{
        background: transparent !important;
        padding: 0 !important;
    }}

    .q-select .q-chip {{
        background: transparent !important;
        color: inherit !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }}

    .q-select .q-chip__content {{
        color: inherit !important;
    }}

    .q-select .q-field__control-container {{
        padding-top: 0 !important;
    }}

    /* ===== Dark Mode Styles ===== */
    body.dark-mode {{
        background: linear-gradient(135deg, #0d1f1a 0%, #1a2e26 50%, #0f1f18 100%) !important;
    }}
    
    body.dark-mode .q-item__label,
    body.dark-mode .q-item__section,
    body.dark-mode .q-field__label,
    body.dark-mode .q-field__native,
    body.dark-mode .q-field__input,
    body.dark-mode .q-field__counter,
    body.dark-mode .q-field__messages,
    body.dark-mode .q-select__input,
    body.dark-mode .text-subtitle1,
    body.dark-mode .text-caption,
    body.dark-mode .text-body2,
    body.dark-mode .q-tab__label,
    body.dark-mode .q-toolbar__title,
    body.dark-mode label,
    body.dark-mode .section-title,
    body.dark-mode .section-subtitle {{
        color: #fbbf24 !important;
    }}
    
    body.dark-mode .q-input input,
    body.dark-mode .q-select input,
    body.dark-mode .q-field__native,
    body.dark-mode textarea,
    body.dark-mode input,
    body.dark-mode select {{
        color: #fcd34d !important;
    }}
    
    /* OVERRIDE: Button text colors */
    /* Dark mode gold button */
    body.dark-mode .gold-btn {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        color: #78350f !important;
        border-color: #b45309 !important;
    }}
    
    body.dark-mode .gold-btn:hover {{
        background: linear-gradient(135deg, #fcd34d 0%, #fbbf24 100%) !important;
        box-shadow: 0 4px 15px rgba(251, 191, 36, 0.6) !important;
    }}
    
    body.dark-mode .gold-btn .q-btn__content,
    body.dark-mode button.gold-btn .q-btn__content,
    body.dark-mode .q-btn.gold-btn .q-btn__content {{
        color: #78350f !important;
    }}
    
    body.dark-mode .green-btn,
    body.dark-mode .green-btn .q-btn__content,
    body.dark-mode button.green-btn,
    body.dark-mode button.green-btn .q-btn__content,
    body.dark-mode .q-btn.green-btn,
    body.dark-mode .q-btn.green-btn .q-btn__content {{
        color: #ecfdf5 !important;
    }}
    
    body.dark-mode .header-green {{
        background: linear-gradient(135deg, #2d7d5a 0%, #34d399 50%, #2d7d5a 100%) !important;
    }}
    
    body.dark-mode .footer-green {{
        background: linear-gradient(135deg, #2d7d5a 0%, #34d399 50%, #2d7d5a 100%) !important;
    }}
    
    /* Dark mode - OVERRIDE ALL CARD BACKGROUNDS */
    body.dark-mode .q-card,
    body.dark-mode .section-card,
    body.dark-mode [class*="section-card"] {{
        background: transparent !important;
        border-color: rgba(251, 191, 36, 0.2) !important;
        transition: none !important;
        animation: none !important;
    }}

    body.dark-mode .section-card,
    body.dark-mode .q-tab-panel .section-card {{
        background: rgba(6, 78, 59, 0.3) !important;
        border: 1px solid rgba(251, 191, 36, 0.2) !important;
    }}

    /* Dark mode - tab panel 容器本身去掉正方形边框（保留内部卡片圆角） */
    body.dark-mode .q-tab-panel,
    body.dark-mode .q-tab-panel.nicegui-tab-panel,
    body.dark-mode .q-tab-panels {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        transition: none !important;
        animation: none !important;
    }}

    body.dark-mode .q-tab-panel > div {{
        background: transparent !important;
    }}
    
    body.dark-mode .q-input,
    body.dark-mode .q-select,
    body.dark-mode .q-field__control,
    body.dark-mode .q-field__marginal {{
        background: rgba(15, 40, 30, 0.9) !important;
        border-color: rgba(251, 191, 36, 0.3) !important;
    }}
    
    body.dark-mode .q-menu,
    body.dark-mode .q-list {{
        background: rgba(20, 45, 35, 0.98) !important;
        border: 1px solid rgba(251, 191, 36, 0.3) !important;
    }}
    
    body.dark-mode .q-item,
    body.dark-mode .q-menu .q-item,
    body.dark-mode .q-menu .q-item__label,
    body.dark-mode .q-menu .q-item__section,
    body.dark-mode .q-item__label {{
        color: #fbbf24 !important;
    }}
    
    body.dark-mode .q-item:hover {{
        background: rgba(251, 191, 36, 0.15) !important;
    }}
    
    body.dark-mode .q-item--active,
    body.dark-mode .q-menu .q-item--active {{
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: #ecfdf5 !important;
        font-weight: bold !important;
    }}

    body.dark-mode .q-item--active .q-item__label,
    body.dark-mode .q-menu .q-item--active .q-item__label {{
        color: #ecfdf5 !important;
    }}
    
    /* Slider dark mode - override text-primary with maximum specificity */
    html body.dark-mode .q-btn[class*="slider-value"],
    html body.dark-mode button[class*="slider-value"],
    html body.dark-mode .q-btn.text-primary[class*="slider-value"],
    html body.dark-mode button.q-btn.text-primary[class*="slider-value"],
    html body.dark-mode [class*="editable-slider"] .q-btn[class*="slider-value"],
    html body.dark-mode [class*="editable-slider"] button[class*="slider-value"] {{
        background: transparent !important;
        color: #fbbf24 !important;
        border-color: rgba(251, 191, 36, 0.5) !important;
    }}
    
    body.dark-mode .editable-slider .text-xs {{
        color: #fbbf24 !important;
    }}
    
    body.dark-mode .toggle-label {{
        color: #fbbf24 !important;
    }}
    
    body.dark-mode .toggle-status {{
        color: #fcd34d !important;
    }}
    
    body.dark-mode .toggle-container .toggle-label {{
        color: #ecfdf5 !important;
    }}
    
    body.dark-mode .toggle-container .toggle-status {{
        color: #cbd5e1 !important;
    }}
    
    body.dark-mode .toggle-container.active {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        border-color: #f59e0b !important;
        box-shadow: 0 0 12px rgba(251, 191, 36, 0.5) !important;
    }}
    
    body.dark-mode .toggle-container.active .toggle-switch {{
        background: rgba(120, 53, 15, 0.5) !important;
    }}
    
    body.dark-mode .toggle-container.active .toggle-label {{
        color: #78350f !important;
        font-weight: bold !important;
    }}
    
    body.dark-mode .toggle-container.active .toggle-status {{
        color: #78350f !important;
        font-weight: bold !important;
    }}
    
    body.dark-mode .section-title {{
        color: #fbbf24 !important;
        border-left-color: #fbbf24 !important;
    }}
    
    /* Light mode: override Quasar text-primary blue color */
    html body:not(.dark-mode) .text-primary,
    html body:not(.dark-mode) .q-btn.text-primary,
    html body:not(.dark-mode) button.q-btn.text-primary {{
        color: #f59e0b !important;
    }}
    
    body.dark-mode .text-primary,
    body.dark-mode [class*="text-"],
    body.dark-mode .q-field--focused .q-field__label,
    body.dark-mode .q-select--focused .q-field__label {{
        color: #fbbf24 !important;
    }}
    
    body.dark-mode .q-tab:not(.q-tab--active) {{
        color: #94a3b8 !important;
    }}
    
    body.dark-mode .q-tab__icon {{
        color: #fbbf24 !important;
    }}
    
    body.dark-mode .green-btn,
    body.dark-mode button.green-btn,
    body.dark-mode .q-btn.green-btn {{
        color: #ecfdf5 !important;
    }}

    /* Dark mode gold button - FIX: use gold colors not blue */
    body.dark-mode .gold-btn,
    body.dark-mode button.gold-btn,
    body.dark-mode .q-btn.gold-btn {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        color: #78350f !important;
        border-color: #b45309 !important;
    }}
    
    /* Dark mode modern navigation buttons */
    body.dark-mode .modern-nav-btn {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        color: #78350f !important;
        border: 2px solid #b45309 !important;
    }}
    
    body.dark-mode .modern-nav-btn:hover {{
        background: linear-gradient(135deg, #fcd34d 0%, #fbbf24 100%) !important;
        box-shadow: 0 4px 15px rgba(251, 191, 36, 0.6) !important;
    }}
    
    body.dark-mode .modern-nav-btn .q-btn__content {{
        color: #78350f !important;
    }}
    
    /* Dark mode modern buttons */
    body.dark-mode .modern-btn-primary {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        color: #78350f !important;
        border: 2px solid #b45309 !important;
    }}
    
    body.dark-mode .modern-btn-primary .q-btn__content {{
        color: #78350f !important;
    }}
    
    body.dark-mode .modern-btn-secondary {{
        background: rgba(251, 191, 36, 0.15) !important;
        color: #fbbf24 !important;
        border: 2px solid rgba(251, 191, 36, 0.5) !important;
    }}
    
    body.dark-mode .modern-btn-secondary:hover {{
        background: rgba(251, 191, 36, 0.25) !important;
        border-color: rgba(251, 191, 36, 0.7) !important;
    }}
    
    body.dark-mode .modern-btn-secondary .q-btn__content {{
        color: #fbbf24 !important;
    }}
    
    /* Dark mode: override text-primary blue color on buttons */
    body.dark-mode .q-btn.text-primary,
    body.dark-mode button.q-btn.text-primary,
    body.dark-mode .modern-nav-btn.text-primary,
    body.dark-mode .modern-btn-primary.text-primary,
    body.dark-mode .modern-btn-secondary.text-primary {{
        color: inherit !important;
    }}
    
    /* Dark mode green button - keep green not blue */
    body.dark-mode .green-btn,
    body.dark-mode button.green-btn,
    body.dark-mode .q-btn.green-btn {{
        background: linear-gradient(135deg, {COLORS["primary_light"]} 0%, {COLORS["primary"]} 100%) !important;
        color: #ecfdf5 !important;
    }}
    </style>
    """


# ============== Shared Component Styles ==============


def get_shared_component_css() -> str:
    """Shared component styles used by both modern and green-gold themes.
    These cover widgets (slider, toggle, log) and Quasar override fixes
    that are identical regardless of theme."""
    return """
    /* ===== Shared: Quasar Field Indicator Removal ===== */
    .q-field--outlined .q-field__control::before,
    .q-field--outlined .q-field__control::after,
    .q-field--standard .q-field__control::before,
    .q-field--standard .q-field__control::after,
    .q-field--filled .q-field__control::before,
    .q-field--filled .q-field__control::after {
        border: none !important;
        display: none !important;
    }

    /* ===== Shared: Select chip cleanup ===== */
    .q-select .q-field__native > span {
        background: transparent !important;
        padding: 0 !important;
    }
    .q-select .q-chip {
        background: transparent !important;
        color: inherit !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }
    .q-select .q-chip__content {
        color: inherit !important;
    }
    .q-select .q-field__control-container {
        padding-top: 0 !important;
    }

    /* ===== Shared: Tab Panel Flash Prevention + 去掉正方形边框 ===== */
    .q-tab-panels,
    .q-tab-panel,
    .nicegui-tab-panel,
    body:not(.dark-mode) .q-tab-panels,
    body:not(.dark-mode) .q-tab-panel,
    body.dark-mode .q-tab-panels,
    body.dark-mode .q-tab-panel,
    body.dark-mode .q-tab-panel.nicegui-tab-panel {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        transition: none !important;
    }
    body:not(.dark-mode) .q-tab-panel > div,
    body:not(.dark-mode) .q-tab-panel > section {
        background: transparent !important;
    }

    /* ===== Shared: Stepper Title Fix ===== */
    .q-stepper__title,
    .q-stepper__tab .q-stepper__title,
    .q-stepper__tab--active .q-stepper__title {
        text-shadow: none !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        -webkit-background-clip: initial !important;
        background-clip: initial !important;
        border: none !important;
        -webkit-text-fill-color: inherit !important;
    }
    .q-stepper *,
    .q-stepper__header *,
    .q-stepper__tab * {
        text-shadow: none !important;
    }
    .q-stepper__tab .q-focus-helper,
    .q-stepper__tab--active .q-focus-helper {
        background: transparent !important;
        opacity: 0 !important;
    }

    /* ===== Shared: Dropdown Animation ===== */
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px) scaleY(0.95); }
        to { opacity: 1; transform: translateY(0) scaleY(1); }
    }

    /* ===== Shared: Footer hidden ===== */
    .q-footer {
        display: none !important;
    }
    """


# ============== Theme Toggle & Language Selector Styles ==============


def get_theme_toggle_styles(COLORS: Dict[str, str]) -> str:
    """Get styles for theme toggle button and language selector"""
    return f"""
    <style>
    /* Theme Toggle Button */
    .theme-toggle-btn {{
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid {COLORS["accent"]} !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }}
    
    .theme-toggle-btn:hover {{
        background: rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 0 15px rgba(251, 191, 36, 0.4) !important;
        transform: scale(1.1) !important;
    }}
    
    .theme-toggle-btn .q-icon {{
        color: {COLORS["accent"]} !important;
        font-size: 22px !important;
    }}
    
    /* Language Selector - Light Mode */
    .lang-selector {{
        min-width: 140px !important;
    }}
    
    .lang-selector .q-field__control {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid {COLORS["primary"]} !important;
        border-radius: 8px !important;
        height: 40px !important;
        min-height: 40px !important;
    }}
    
    .lang-selector .q-field__control:hover {{
        background: white !important;
        border-color: {COLORS["secondary"]} !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.2) !important;
    }}
    
    .lang-selector .q-field__native {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
    }}
    
    .lang-selector .q-icon {{
        color: {COLORS["primary"]} !important;
    }}
    
    .lang-icon-wrapper {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    /* Dark mode adjustments */
    body.dark-mode .theme-toggle-btn {{
        background: rgba(251, 191, 36, 0.1) !important;
        border-color: {COLORS["accent"]} !important;
    }}
    
    body.dark-mode .theme-toggle-btn:hover {{
        background: rgba(251, 191, 36, 0.2) !important;
        box-shadow: 0 0 15px rgba(251, 191, 36, 0.6) !important;
    }}
    
    body.dark-mode .lang-selector .q-field__control {{
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: {COLORS["accent"]} !important;
    }}
    
    body.dark-mode .lang-selector .q-field__native {{
        color: {COLORS["accent"]} !important;
    }}
    </style>
    """


# ============== Modern Theme CSS (原有样式) ==============


def get_modern_css() -> str:
    """Get the modern theme CSS with light/dark mode support"""
    return f"""
        /* ===== Modern Theme CSS ===== */
        
        /* Root Variables - Light Mode (Default) */
        :root {{
            /* Override Quasar variables - EMERALD GREEN THEME (light mode) */
            --q-primary: #059669 !important;
            --q-secondary: {MODERN_COLORS["secondary"]} !important;
            --q-accent: {MODERN_COLORS["accent"]} !important;

            /* Semantic theme colors */
            --color-primary: {MODERN_COLORS["primary"]};
            --color-primary-dark: {MODERN_COLORS["primary_dark"]};
            --color-primary-light: {MODERN_COLORS["primary_light"]};
            --color-secondary: {MODERN_COLORS["secondary"]};
            --color-accent: {MODERN_COLORS["accent"]};
            --color-success: {MODERN_COLORS["success"]};
            --color-warning: {MODERN_COLORS["warning"]};
            --color-error: {MODERN_COLORS["error"]};
            --color-info: {MODERN_COLORS["info"]};

            /* Gold/Amber palette */
            --color-gold: #f59e0b;
            --color-gold-light: #fbbf24;
            --color-gold-lighter: #fcd34d;
            --color-gold-dark: #d97706;
            --color-gold-darker: #b45309;
            --color-gold-text: #78350f;

            /* Emerald/Green palette */
            --color-emerald-900: #064e3b;
            --color-emerald-800: #065f46;
            --color-emerald-700: #047857;
            --color-emerald-600: #059669;
            --color-emerald-500: #10b981;
            --color-emerald-400: #34d399;
            --color-emerald-300: #6ee7b7;
            --color-emerald-200: #a7f3d0;
            --color-emerald-100: #d1fae5;
            --color-emerald-50: #ecfdf5;
            --color-green-400: #4ade80;
            --color-green-300: #86efac;
            --color-green-50: #f0fdf4;
            --color-green-100: #dcfce7;
            --color-forest: #022c22;

            /* Slate palette */
            --color-slate-900: #0f172a;
            --color-slate-700: #334155;
            --color-slate-500: #64748b;
            --color-slate-300: #cbd5e1;
            --color-slate-200: #e2e8f0;

            /* Blue/Violet accents */
            --color-blue-50: #f0f9ff;
            --color-blue-100: #e0f2fe;
            --color-violet-50: #ede9fe;

            /* Light mode semantic colors */
            --color-bg: #f0fdf4;
            --color-surface: #ffffff;
            --color-surface-light: #f0fdf4;
            --color-text: #0f172a;
            --color-text-secondary: #334155;
            --color-text-muted: #64748b;
            --color-border: #86efac;
            --color-border-subtle: rgba(5, 150, 105, 0.25);

            /* Button semantic variables — 浅色模式使用翡翠绿系按钮 */
            --btn-primary-bg: linear-gradient(135deg, #059669 0%, #10b981 100%);
            --btn-primary-bg-hover: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --btn-primary-color: #059669;
            --btn-primary-text: #ffffff;
            --btn-primary-border: #047857;
            --btn-primary-shadow: rgba(5, 150, 105, 0.35);
            --btn-primary-shadow-hover: rgba(5, 150, 105, 0.5);

            /* Card semantic variables */
            --card-bg: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%);
            --card-border: rgba(5, 150, 105, 0.25);
            --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.08), 0 0 0 1px rgba(5, 150, 105, 0.05);
            --card-shadow-hover: 0 8px 30px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(5, 150, 105, 0.2);

            /* Input semantic variables */
            --input-bg: rgba(255, 255, 255, 0.8);
            --input-bg-hover: rgba(255, 255, 255, 1);
            --input-border: rgba(203, 213, 225, 0.8);
            --input-focus-shadow: 0 0 0 3px rgba(5, 150, 105, 0.15);
        }}
        
        /* Dark Mode Variables */
        body.dark-mode {{
            --color-bg: #022c22;
            --color-surface: #064e3b;
            --color-surface-light: #065f46;
            --color-text: #ecfdf5;
            --color-text-secondary: #6ee7b7;
            --color-text-muted: #34d399;
            --color-border: #059669;
            --color-border-subtle: rgba(5, 150, 105, 0.3);

            /* Button overrides */
            --btn-primary-bg: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            --btn-primary-bg-hover: linear-gradient(135deg, #fcd34d 0%, #fbbf24 100%);
            --btn-primary-color: #fbbf24;
            --btn-primary-text: #78350f;
            --btn-primary-border: #b45309;
            --btn-primary-shadow: rgba(245, 158, 11, 0.4);
            --btn-primary-shadow-hover: rgba(251, 191, 36, 0.6);

            /* Card overrides */
            --card-bg: rgba(6, 78, 59, 0.6);
            --card-border: rgba(5, 150, 105, 0.3);
            --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(5, 150, 105, 0.1);
            --card-shadow-hover: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(5, 150, 105, 0.3);

            /* Input overrides */
            --input-bg: rgba(2, 44, 34, 0.8);
            --input-bg-hover: rgba(2, 44, 34, 0.9);
            --input-border: rgba(251, 191, 36, 0.3);
            --input-focus-shadow: 0 0 0 3px rgba(74, 222, 128, 0.2);
        }}
        
        /* Body Background - Light Mode (Default) */
        body {{
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 25%, #e0f2fe 50%, #ede9fe 75%, #ecfdf5 100%);
            background-attachment: fixed;
            color: var(--color-text);
        }}
        
        /* Body Background - Dark Mode is in Dark Mode Overrides section below */

        /* ===== Cards ===== */
        .modern-card {{
            background: var(--card-bg) !important;
            backdrop-filter: blur(10px);
            border: 1px solid var(--card-border) !important;
            border-radius: 16px !important;
            box-shadow: var(--card-shadow) !important;
            transition: all 0.3s ease;
        }}

        .modern-card:hover {{
            box-shadow: var(--card-shadow-hover) !important;
            transform: translateY(-2px);
        }}

        .modern-card-hover {{
            cursor: pointer;
        }}

        .modern-card-hover:hover {{
            background: linear-gradient(135deg, var(--color-green-100) 0%, var(--color-emerald-100) 50%, var(--color-blue-100) 100%) !important;
            border-color: var(--color-border-subtle) !important;
        }}

        body.dark-mode .modern-card-hover:hover {{
            background: rgba(51, 65, 85, 0.9) !important;
            border-color: rgba(5, 150, 105, 0.4) !important;
        }}

        .modern-card-header {{
            background: linear-gradient(135deg, rgba(5, 150, 105, 0.2), rgba(251, 191, 36, 0.1)) !important;
            border-bottom: 1px solid var(--color-border-subtle) !important;
            border-radius: 16px 16px 0 0 !important;
            padding: 16px 20px !important;
        }}
        
        /* ===== Buttons - Using CSS variables ===== */
        button.q-btn.modern-btn-primary,
        .q-btn.q-btn-item.modern-btn-primary,
        html body button.modern-btn-primary,
        [class*="modern-btn-primary"] {{
            background: var(--btn-primary-bg) !important;
            background-color: var(--btn-primary-color) !important;
            color: var(--btn-primary-text) !important;
            border: 2px solid var(--btn-primary-border) !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600;
            text-transform: none;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px var(--btn-primary-shadow) !important;
            transition: all 0.3s ease;
        }}

        button.q-btn.modern-btn-primary:hover,
        html body button.modern-btn-primary:hover {{
            background: var(--btn-primary-bg-hover) !important;
            box-shadow: 0 6px 20px var(--btn-primary-shadow-hover) !important;
            transform: translateY(-2px);
        }}

        button.q-btn.modern-btn-secondary,
        html body button.modern-btn-secondary,
        [class*="modern-btn-secondary"] {{
            background: rgba(5, 150, 105, 0.1) !important;
            border: 2px solid rgba(5, 150, 105, 0.35) !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            color: #047857 !important;
            font-weight: 600;
            text-transform: none;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }}

        button.q-btn.modern-btn-secondary:hover,
        html body button.modern-btn-secondary:hover {{
            background: rgba(5, 150, 105, 0.18) !important;
            border-color: rgba(5, 150, 105, 0.5) !important;
        }}

        .modern-btn-danger {{
            background: linear-gradient(135deg, var(--color-error), #dc2626) !important;
            border: 2px solid transparent !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600;
            text-transform: none;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
            transition: all 0.3s ease;
        }}

        /* Success button - uses same gold/amber as primary */
        button.q-btn.modern-btn-success,
        .q-btn.q-btn-item.modern-btn-success,
        html body button.modern-btn-success,
        [class*="modern-btn-success"] {{
            background: var(--btn-primary-bg) !important;
            background-color: var(--btn-primary-color) !important;
            color: var(--btn-primary-text) !important;
            border: 2px solid var(--btn-primary-border) !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600;
            text-transform: none;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px var(--btn-primary-shadow) !important;
            transition: all 0.3s ease;
        }}

        body.dark-mode button.q-btn.modern-btn-success .q-btn__content,
        body.dark-mode .q-btn.q-btn-item.modern-btn-success .q-btn__content {{
            color: var(--color-gold-text) !important;
        }}

        button.q-btn.modern-btn-ghost,
        html body button.modern-btn-ghost,
        [class*="modern-btn-ghost"] {{
            background: transparent !important;
            border: 2px solid rgba(5, 150, 105, 0.35) !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            color: #059669 !important;
            font-weight: 600;
            text-transform: none;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }}

        button.q-btn.modern-btn-ghost:hover,
        html body button.modern-btn-ghost:hover {{
            background: rgba(5, 150, 105, 0.1) !important;
            border-color: rgba(5, 150, 105, 0.5) !important;
            color: #047857 !important;
        }}
        
        /* ===== Inputs ===== */
        /* Fix: Do NOT set background on input elements - only .q-field__control should have background */
        /* This prevents the "double input box" effect caused by overlapping backgrounds */
        .modern-input input,
        .force-light-bg input {{
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            color: var(--color-text) !important;
            padding: 8px 12px !important;
            transition: all 0.3s ease !important;
        }}
        
        body.dark-mode .force-light-bg input {{
            background: transparent !important;
        }}

        /* Fix: Ensure modern-select has correct background */
        .modern-select .q-field__control,
        .force-light-bg .q-field__control,
        .styled-select-container .q-field__control,
        .styled-input-container .q-field__control {{
            background: var(--input-bg) !important;
            border: 2px solid var(--input-border) !important;
            border-radius: 10px !important;
        }}

        .modern-select .q-field__control:hover,
        .force-light-bg .q-field__control:hover,
        .styled-select-container .q-field__control:hover,
        .styled-input-container .q-field__control:hover {{
            background: var(--input-bg-hover) !important;
        }}

        /* Fix: Remove the inner rounded box overlay on select dropdown values */
        .modern-select .q-field__native {{
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            color: var(--color-text) !important;
            padding: 0 !important;
            min-height: auto !important;
            line-height: 1.5 !important;
        }}

        /* Fix: Prevent label and value overlap in select with use-input */
        .modern-select.q-field--float .q-field__native {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }}

        /* Fix: Adjust select input when using use-input prop */
        .modern-select.q-field--outlined .q-field__control-container {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }}

        /* Fix: Ensure select value doesn't overlap with label */
        .modern-select .q-field__input {{
            color: var(--color-text) !important;
            font-weight: 500 !important;
            padding: 0 !important;
        }}

        /* Fix: Add spacing between select label and value */
        .modern-select .q-field__label {{
            top: -12px !important;
            font-size: 11px !important;
            transform: translateY(-100%) scale(0.9) !important;
            color: var(--color-text-secondary) !important;
            background: transparent !important;
            padding: 0 4px !important;
            margin-bottom: 4px !important;
        }}

        /* Fix: Adjust select control padding to accommodate label */
        .modern-select.q-field--outlined .q-field__control {{
            padding-top: 8px !important;
            padding-bottom: 8px !important;
        }}

        /* Fix: Input element should NOT have background - only .q-field__control should */
        body.dark-mode .modern-input input {{
            background: transparent !important;
            border: none !important;
        }}

        body.dark-mode .modern-select .q-field__native {{
            background: transparent !important;
        }}

        /* Fix: Remove focus box-shadow/border from input - let .q-field__control handle it */
        .modern-input input:focus {{
            outline: none !important;
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }}

        /* Select focus state - no inner box shadow */
        .modern-select.q-field--focused .q-field__native {{
            background: transparent !important;
            box-shadow: none !important;
        }}

        body.dark-mode .modern-input input:focus {{
            background: transparent !important;
        }}
        
        /* ===== Path Input Specific Styles ===== */
        .path-input .q-field__control {{
            background: var(--input-bg) !important;
            border: 2px solid var(--input-border) !important;
            border-radius: 10px !important;
        }}

        .path-input .q-field__control:hover {{
            background: var(--input-bg-hover) !important;
        }}

        .path-input.q-field--focused .q-field__control {{
            background: var(--input-bg-hover) !important;
            box-shadow: var(--input-focus-shadow) !important;
        }}
        
        .path-input .q-field__native {{
            background: transparent !important;
            color: var(--color-text) !important;
        }}
        
        .modern-input .q-field__label,
        .modern-select .q-field__label {{
            color: var(--color-text-secondary) !important;
        }}
        
        /* ===== Generic Field Styling (all inputs/selects) ===== */
        .q-field__control {{
            border-radius: 10px !important;
            background: var(--input-bg) !important;
            border: 2px solid var(--input-border) !important;
            transition: all 0.2s ease;
            min-height: 44px !important;
            padding-left: 12px !important;
        }}

        /* Field indicator removal is in get_shared_component_css() */

        .q-field__control:hover {{
            background: var(--input-bg-hover) !important;
        }}

        .q-field--focused .q-field__control {{
            box-shadow: var(--input-focus-shadow) !important;
            background: var(--input-bg-hover) !important;
        }}
        
        .q-field__label {{
            color: var(--color-text) !important;
            font-weight: 500 !important;
            padding-left: 4px !important;
        }}
        
        .q-field__native,
        .q-field__input {{
            color: var(--color-text) !important;
            font-weight: 500 !important;
        }}
        
        .q-field__marginal {{
            color: var(--color-text-secondary) !important;
        }}
        
        .q-field__bottom {{
            padding: 2px 12px 0 !important;
        }}
        
        /* Select dropdown arrow */
        .q-select__dropdown-icon {{
            color: var(--color-text-secondary) !important;
            transition: transform 0.3s ease !important;
        }}
        
        .q-field--focused .q-select__dropdown-icon {{
            transform: rotate(180deg) !important;
        }}

        /* Select chip cleanup is in get_shared_component_css() */

        /* Placeholder colors */
        .q-field__input::placeholder,
        .q-input::placeholder {{
            color: rgba(6, 78, 59, 0.4) !important;
        }}
        
        /* Select specific */
        .q-select .q-field__control {{
            cursor: pointer !important;
        }}
        
        .q-select__input {{
            color: var(--color-text) !important;
            font-weight: 500 !important;
        }}
        
        /* ===== Header & Navigation - Improved ===== */
        .modern-header, .q-header {{
            background: linear-gradient(135deg, 
                rgba(6, 78, 59, 0.85), 
                rgba(5, 150, 105, 0.8),
                rgba(6, 95, 70, 0.85)) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
            border-bottom: 2px solid rgba(251, 191, 36, 0.4) !important;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.15) !important;
            padding: 8px 24px !important;
        }}
        
        body.dark-mode .modern-header,
        body.dark-mode .q-header {{
            background: linear-gradient(135deg, 
                rgba(2, 44, 34, 0.9), 
                rgba(5, 150, 105, 0.85),
                rgba(6, 78, 59, 0.9)) !important;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4) !important;
        }}
        
        /* Light mode header - keep dark green bg so white text stays visible */
        body:not(.dark-mode) .q-header,
        body:not(.dark-mode) .q-header.bg-primary {{
            background: linear-gradient(135deg,
                rgba(6, 78, 59, 0.92),
                rgba(5, 150, 105, 0.88),
                rgba(6, 95, 70, 0.92)) !important;
            border-bottom: 2px solid rgba(251, 191, 36, 0.4) !important;
        }}
        
        /* ============================================
           Navigation Buttons - Consolidated Styles
           ============================================ */

        /* Base nav button style - both light and dark mode */
        body .q-btn.modern-nav-btn,
        body button.q-btn.modern-nav-btn {{
            background: var(--btn-primary-bg) !important;
            background-color: var(--btn-primary-color) !important;
            border: 2px solid var(--btn-primary-border) !important;
            border-radius: 12px !important;
            color: var(--btn-primary-text) !important;
            font-weight: 600;
            padding: 10px 18px !important;
            box-shadow: 0 2px 8px var(--btn-primary-shadow), inset 0 1px 0 rgba(255,255,255,0.3) !important;
            --q-btn-active-opacity: 1 !important;
            transition: all 0.2s ease;
        }}

        /* Nav button hover */
        body .q-btn.modern-nav-btn:hover,
        body button.q-btn.modern-nav-btn:hover {{
            background: var(--btn-primary-bg-hover) !important;
            border-color: var(--color-gold-darker) !important;
            box-shadow: 0 4px 16px var(--btn-primary-shadow-hover), 0 0 20px var(--btn-primary-shadow) !important;
            transform: translateY(-1px);
        }}

        /* Nav button active/focus states */
        body .q-btn.modern-nav-btn:active,
        body button.q-btn.modern-nav-btn:active,
        body .q-btn.modern-nav-btn:focus,
        body button.q-btn.modern-nav-btn:focus,
        body .q-btn.modern-nav-btn.q-btn--active,
        body button.q-btn.modern-nav-btn.q-btn--active {{
            background: var(--btn-primary-bg-hover) !important;
            color: var(--btn-primary-text) !important;
            border-color: var(--color-gold-darker) !important;
            box-shadow: 0 0 20px var(--btn-primary-shadow-hover), inset 0 1px 0 rgba(255,255,255,0.3) !important;
            outline: none !important;
        }}

        /* Nav button content color */
        body .q-btn.modern-nav-btn .q-btn__content,
        body button.q-btn.modern-nav-btn .q-btn__content {{
            color: var(--btn-primary-text) !important;
        }}

        /* Active indicator for current page nav button */
        body .q-btn.modern-nav-btn-active,
        body button.q-btn.modern-nav-btn-active {{
            background: var(--btn-primary-bg-hover) !important;
            box-shadow: 0 0 20px var(--btn-primary-shadow-hover), inset 0 1px 0 rgba(255,255,255,0.3) !important;
            position: relative;
        }}

        body .q-btn.modern-nav-btn-active::after,
        body button.q-btn.modern-nav-btn-active::after {{
            content: '';
            position: absolute;
            bottom: -4px;
            left: 50%;
            transform: translateX(-50%);
            width: 20px;
            height: 3px;
            background: var(--color-gold-light);
            border-radius: 2px;
        }}
        
        /* ===== Stepper (Q-Stepper) ===== */
        .q-stepper {{
            background: transparent !important;
            box-shadow: none !important;
        }}
        
        /* Light mode stepper header - force transparent background */
        html body:not(.dark-mode) .q-stepper__header,
        .q-stepper__header {{
            background: rgba(240, 253, 244, 0.8) !important;
            border-radius: 12px !important;
            padding: 8px !important;
            border: 1px solid rgba(5, 150, 105, 0.2) !important;
        }}
        
        .q-stepper__tab,
        /* Light mode stepper tab - force transparent background */
        html body:not(.dark-mode) .q-stepper__header .q-stepper__tab,
        .q-stepper__header .q-stepper__tab {{
            color: #0f172a !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            background: transparent !important;
            background-color: transparent !important;
        }}
        
        .q-stepper__tab--active,
        /* Light mode stepper tab active - force transparent background */
        html body:not(.dark-mode) .q-stepper__header .q-stepper__tab--active,
        .q-stepper__header .q-stepper__tab--active {{
            color: #059669 !important;
            background: transparent !important;
            background-color: transparent !important;
            font-weight: 600 !important;
        }}
        
        /* Stepper title fix is in get_shared_component_css() */
        
        .q-stepper__dot,
        .q-stepper__dot .q-icon,
        .q-stepper__dot .material-icons {{
            background: transparent !important;
            background-color: transparent !important;
            color: var(--color-emerald-600) !important;
        }}

        /* Stepper dot active state */
        .q-stepper__tab--active .q-stepper__dot,
        .q-stepper__tab--active .q-stepper__dot .q-icon,
        .q-stepper__tab--active .q-stepper__dot .material-icons {{
            background: transparent !important;
            background-color: transparent !important;
            color: var(--color-emerald-600) !important;
        }}
        
        .q-stepper__line {{
            background: transparent !important;
        }}
        
        /* Focus helper fix is in get_shared_component_css() */

        /* Light mode stepper content - force transparent background */
        html body:not(.dark-mode) .q-stepper__content,
        .q-stepper__content {{
            background: transparent !important;
            border-radius: 12px !important;
            margin-top: 16px !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Footer hidden is in get_shared_component_css() */

        /* ===== Section Titles ===== */
        .modern-section-title {{
            color: var(--color-text) !important;
            font-weight: 700 !important;
            font-size: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            position: relative;
            padding-left: 16px !important;
        }}
        
        .modern-section-title::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 4px;
            height: 24px;
            background: linear-gradient(180deg, var(--color-primary), var(--color-secondary));
            border-radius: 2px;
        }}
        
        /* ===== Page Container ===== */
        .modern-page-container {{
            max-width: 1400px !important;
            margin: 0 auto !important;
            padding: 24px !important;
        }}
        
        /* ===== Tabs ===== */
        .q-tab {{
            color: var(--color-text-secondary) !important;
            font-weight: 500 !important;
            text-transform: none !important;
            border-radius: 10px !important;
            margin: 0 4px !important;
        }}
        
        .q-tab--active {{
            color: var(--color-primary-light) !important;
            background: rgba(5, 150, 105, 0.15) !important;
        }}
        
        .q-tab__indicator {{
            background: linear-gradient(90deg, var(--color-primary), var(--color-secondary)) !important;
            height: 3px !important;
            border-radius: 3px !important;
        }}
        
        /* ===== Expansion Panels ===== */
        .q-expansion-item {{
            background: rgba(240, 253, 244, 0.6) !important;
            border: 1px solid rgba(5, 150, 105, 0.15) !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
            overflow: hidden;
        }}
        
        .q-expansion-item__container {{
            border-radius: 12px !important;
        }}
        
        /* ===== Badges ===== */
        .modern-badge {{
            background: rgba(5, 150, 105, 0.2) !important;
            color: var(--color-primary-light) !important;
            border-radius: 20px !important;
            padding: 4px 12px !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
        }}
        
        .modern-badge-primary {{
            background: linear-gradient(135deg, var(--color-primary), var(--color-accent)) !important;
            color: #ecfdf5 !important;
        }}
        
        .modern-badge-success {{
            background: rgba(251, 191, 36, 0.2) !important;
            color: #34d399 !important;
        }}
        
        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(5, 150, 105, 0.08);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, var(--color-primary), var(--color-accent));
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: linear-gradient(180deg, var(--color-primary-light), var(--color-secondary));
        }}
        
        /* ===== Notifications ===== */
        .q-notification {{
            border-radius: 12px !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
        }}
        
        /* ===== Checkboxes & Radio ===== */
        .q-checkbox__bg,
        .q-radio__bg {{
            border: 2px solid rgba(5, 150, 105, 0.4) !important;
            border-radius: 6px !important;
        }}
        
        .q-checkbox__svg,
        .q-radio__inner--truthy .q-radio__check {{
            color: var(--color-primary) !important;
        }}
        
        /* ===== Menu / Dropdown ===== */
        .q-menu {{
            background: rgba(255, 255, 255, 0.98) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(5, 150, 105, 0.2) !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.12) !important;
            animation: slideDown 0.25s ease-out !important;
            transform-origin: top !important;
        }}
        
        /* Ensure dropdown items have visible text color */
        .q-item {{
            color: var(--color-emerald-800) !important;
            transition: all 0.15s ease;
            border-radius: 6px !important;
            margin: 2px 4px !important;
        }}

        .q-menu .q-item,
        .q-menu .q-item__label,
        .q-menu .q-item__section {{
            color: var(--color-emerald-800) !important;
        }}

        .q-item:hover,
        .q-menu .q-item:hover {{
            background: rgba(5, 150, 105, 0.12) !important;
            color: var(--color-emerald-700) !important;
        }}
        
        body:not(.dark-mode) .q-item--active,
        body:not(.dark-mode) .q-menu .q-item--active {{
            background: linear-gradient(135deg, {MODERN_COLORS["primary"]}, {MODERN_COLORS["accent"]}) !important;
            color: white !important;
            font-weight: 600 !important;
        }}

        body:not(.dark-mode) .q-item--active .q-item__label,
        body:not(.dark-mode) .q-menu .q-item--active .q-item__label,
        body:not(.dark-mode) .q-menu .q-item--active .q-item__section {{
            color: white !important;
        }}
        
        .q-virtual-scroll__content {{
            padding: 4px !important;
        }}
        
        /* ===== Dialog ===== */
        .q-dialog__backdrop {{
            background: rgba(0, 0, 0, 0.7) !important;
            backdrop-filter: blur(5px) !important;
        }}
        
        .q-dialog .q-card--dark {{
            background: rgba(30, 41, 59, 0.95) !important;
            border: 1px solid rgba(5, 150, 105, 0.2) !important;
        }}
        
        /* ===== Log Viewer ===== */
        .modern-log {{
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(5, 150, 105, 0.2) !important;
            border-radius: 12px !important;
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        }}
        
        /* ===== Step Cards (Homepage) - Improved Layout ===== */
        .step-card {{
            background: var(--color-surface) !important;
            border: 2px solid var(--color-emerald-600) !important;
            border-radius: 16px !important;
            padding: 24px 16px !important;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
            min-height: 180px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            cursor: pointer;
        }}

        body.dark-mode .step-card {{
            background: rgba(6, 78, 59, 0.95) !important;
            border-color: rgba(251, 191, 36, 0.5) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        }}

        .step-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--color-emerald-600), var(--color-gold-light), var(--color-emerald-600));
            opacity: 0;
            transition: opacity 0.3s ease;
        }}

        .step-card:hover {{
            background: var(--color-surface) !important;
            border-color: rgba(5, 150, 105, 0.8) !important;
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(5, 150, 105, 0.2) !important;
        }}

        body.dark-mode .step-card:hover {{
            background: rgba(8, 100, 75, 1) !important;
            border-color: rgba(251, 191, 36, 0.8) !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4) !important;
        }}

        .step-card:hover::before {{
            opacity: 1;
        }}

        .step-card .q-icon {{
            background: linear-gradient(135deg, var(--color-green-400), var(--color-gold-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 48px !important;
            margin-bottom: 12px !important;
            transition: transform 0.3s ease;
        }}

        .step-card:hover .q-icon {{
            transform: scale(1.1);
        }}

        /* Step card text colors */
        .step-card .text-h6,
        .step-card .text-subtitle1,
        .step-card .text-body2 {{
            color: var(--color-text) !important;
        }}

        .step-card .text-caption {{
            color: var(--color-text-secondary) !important;
            background: transparent !important;
        }}

        body.dark-mode .step-card .text-h6,
        body.dark-mode .step-card .text-subtitle1,
        body.dark-mode .step-card .text-body2,
        body.dark-mode .step-card .text-caption {{
            color: var(--color-text) !important;
            background: transparent !important;
        }}
        
        /* ===== Model Architecture Grid ===== */
        .model-item {{
            background: linear-gradient(135deg, var(--color-green-50) 0%, var(--color-emerald-50) 100%) !important;
            border: 1px solid var(--color-border-subtle) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }}

        body.dark-mode .model-item {{
            background: rgba(6, 78, 59, 0.8) !important;
            border-color: rgba(251, 191, 36, 0.3) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        }}

        .model-item:hover {{
            background: linear-gradient(135deg, var(--color-green-100) 0%, var(--color-emerald-100) 100%) !important;
            border-color: rgba(5, 150, 105, 0.6) !important;
            transform: translateX(4px);
            box-shadow: 0 4px 16px rgba(5, 150, 105, 0.2) !important;
        }}

        body.dark-mode .model-item:hover {{
            background: rgba(5, 150, 105, 0.7) !important;
            border-color: rgba(251, 191, 36, 0.4) !important;
        }}

        /* Model item text colors */
        .model-item .text-subtitle2,
        .model-item .text-caption,
        .model-item .model-name,
        .model-item .model-desc,
        .model-item .feature-desc {{
            color: var(--color-emerald-800) !important;
        }}

        body.dark-mode .model-item .text-subtitle2,
        body.dark-mode .model-item .text-caption,
        body.dark-mode .model-item .model-name,
        body.dark-mode .model-item .model-desc,
        body.dark-mode .model-item .feature-desc {{
            color: var(--color-text) !important;
        }}

        /* Step card text colors */
        .step-card .step-label {{
            color: var(--color-emerald-600) !important;
        }}

        .step-card .step-title {{
            color: var(--color-emerald-800) !important;
        }}

        .step-card .step-desc {{
            color: var(--color-emerald-700) !important;
        }}

        body.dark-mode .step-card .step-label {{
            color: var(--color-green-400) !important;
        }}

        body.dark-mode .step-card .step-title {{
            color: var(--color-text) !important;
        }}

        body.dark-mode .step-card .step-desc {{
            color: var(--color-emerald-300) !important;
        }}

        /* Section text colors */
        .section-title {{
            color: var(--color-emerald-800) !important;
        }}

        .section-subtitle {{
            color: var(--color-emerald-700) !important;
        }}

        .app-desc {{
            color: var(--color-emerald-700) !important;
        }}

        body.dark-mode .section-title {{
            color: var(--color-text) !important;
        }}

        body.dark-mode .section-subtitle {{
            color: var(--color-emerald-300) !important;
        }}

        body.dark-mode .app-desc {{
            color: var(--color-emerald-300) !important;
        }}

        /* Header text colors - header always has dark bg, so always light text */
        .header-title {{
            color: var(--color-emerald-50) !important;
        }}

        .header-version {{
            color: var(--color-emerald-200) !important;
        }}
        
        /* ===== Animation Keyframes ===== */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .animate-fade-in {{
            animation: fadeIn 0.5s ease forwards;
        }}
        
        @keyframes glow {{
            0%, 100% {{ box-shadow: 0 0 5px rgba(5, 150, 105, 0.5); }}
            50% {{ box-shadow: 0 0 20px rgba(5, 150, 105, 0.8), 0 0 40px rgba(251, 191, 36, 0.4); }}
        }}
        
        .animate-glow {{
            animation: glow 2s ease-in-out infinite;
        }}
        
        /* ================================================================
           Component Styles: Editable Slider, Toggle Switch, Log Output
           ================================================================ */
        
        /* ===== Editable Slider ===== */
        .editable-slider {{
            margin-bottom: 4px !important;
            flex: 1 1 0 !important;
            min-width: 120px !important;
            max-width: 100% !important;
            padding: 2px 0 !important;
        }}
        
        .slider-label-row {{
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            margin-bottom: 6px !important;
        }}
        
        .slider-label {{
            font-size: 11px !important;
            font-weight: 500 !important;
            color: var(--color-text) !important;
            line-height: 1.2 !important;
        }}
        
        /* Slider value button */
        html body .q-btn[class*="slider-value"],
        html body button[class*="slider-value"],
        html body .q-btn.text-primary[class*="slider-value"],
        html body button.q-btn.text-primary[class*="slider-value"],
        html body [class*="editable-slider"] .q-btn[class*="slider-value"],
        html body [class*="editable-slider"] button[class*="slider-value"] {{
            font-size: 10px !important;
            font-family: 'Consolas', monospace;
            color: var(--color-gold) !important;
            background: transparent !important;
            border: 1px solid rgba(245, 158, 11, 0.5) !important;
            border-radius: 4px !important;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: none;
            min-height: 18px !important;
            height: 18px !important;
            padding: 0 4px !important;
            min-width: 32px !important;
        }}

        html body .q-btn[class*="slider-value"]:hover,
        html body button[class*="slider-value"]:hover,
        html body .q-btn.text-primary[class*="slider-value"]:hover,
        html body button.q-btn.text-primary[class*="slider-value"]:hover,
        html body [class*="editable-slider"] .q-btn[class*="slider-value"]:hover,
        html body [class*="editable-slider"] button[class*="slider-value"]:hover {{
            background: rgba(245, 158, 11, 0.15) !important;
            border-color: rgba(245, 158, 11, 0.8) !important;
            color: var(--color-gold) !important;
        }}
        
        .editable-slider .text-xs {{
            color: var(--color-text) !important;
        }}
        
        .slider-container {{
            position: relative !important;
            height: 20px !important;
            display: flex !important;
            align-items: center !important;
        }}
        
        .slider-track {{
            position: absolute !important;
            left: 0 !important;
            right: 0 !important;
            height: 6px !important;
            background: linear-gradient(90deg, #e2e8f0 0%, #cbd5e1 50%, #e2e8f0 100%) !important;
            border-radius: 3px !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.1) !important;
        }}
        
        .slider-fill {{
            position: absolute !important;
            left: 0 !important;
            height: 6px !important;
            background: linear-gradient(90deg, var(--color-gold) 0%, var(--color-gold-light) 100%) !important;
            border-radius: 3px !important;
            transition: width 0.15s ease;
        }}

        .slider-thumb {{
            position: absolute !important;
            width: 18px !important;
            height: 18px !important;
            background: white !important;
            border: 3px solid var(--color-accent) !important;
            border-radius: 50% !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            transition: left 0.15s ease;
            z-index: 2;
        }}
        
        .slider-input {{
            position: absolute !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            opacity: 0 !important;
            cursor: pointer !important;
            z-index: 3 !important;
            margin: 0 !important;
        }}
        
        .slider-edit-input {{
            width: 70px !important;
            font-size: 12px !important;
            font-family: 'Consolas', monospace;
            text-align: right;
            border: 2px solid var(--color-gold) !important;
            border-radius: 6px !important;
            padding: 2px 8px !important;
            outline: none !important;
            background: white !important;
            color: var(--color-gold-text) !important;
        }}

        body.dark-mode .slider-edit-input {{
            background: rgba(30, 20, 10, 0.9) !important;
            color: var(--color-gold-light) !important;
            border-color: var(--color-gold) !important;
        }}
        
        /* ===== Quasar Slider Override - Gold Theme ===== */
        .q-slider [class*="text-#"] {{
            color: var(--color-gold) !important;
        }}

        .q-slider [class*="bg-#"] {{
            background: linear-gradient(90deg, var(--color-gold) 0%, var(--color-gold-light) 100%) !important;
        }}

        .q-slider {{
            --q-color-primary: var(--color-gold) !important;
            --q-primary: var(--color-gold) !important;
        }}
        
        /* 轨道容器 - 完全透明 */
        .q-slider__track-container,
        .q-slider__track-container--h,
        .q-slider__track-container--v {{
            background: transparent !important;
            border-radius: 3px !important;
        }}
        
        /* 轨道 - 默认透明 */
        .q-slider__track {{
            background: transparent !important;
        }}
        
        /* 轨道 - 有宽度时金色（已填充部分） */
        .q-slider__track[style*="width"]:not([style*="width: 0"]) {{
            background: linear-gradient(90deg, var(--color-gold) 0%, var(--color-gold-light) 100%) !important;
        }}

        /* 滑块按钮 */
        .q-slider__thumb {{
            color: var(--color-gold) !important;
        }}

        .q-slider__thumb-circle {{
            background: var(--color-gold) !important;
            border-color: var(--color-gold) !important;
        }}
        
        /* 禁用状态 - 也透明 */
        .q-slider--disabled .q-slider__track-container,
        .q-slider--disabled .q-slider__track-container--h,
        .q-slider--disabled .q-slider__track-container--v {{
            background: transparent !important;
        }}
        
        /* ===== Toggle Switch ===== */
        .toggle-container {{
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            cursor: pointer !important;
            padding: 4px 10px !important;
            background: rgba(134, 239, 172, 0.3) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(5, 150, 105, 0.4) !important;
            transition: all 0.3s ease !important;
            text-transform: none !important;
            min-height: 28px !important;
            margin: 2px !important;
        }}
        
        .toggle-container .q-btn__content {{
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
            padding: 0 !important;
        }}
        
        .toggle-container:hover {{
            background: rgba(134, 239, 172, 0.5) !important;
            border-color: rgba(5, 150, 105, 0.6) !important;
        }}
        
        .toggle-container.active {{
            background: var(--btn-primary-bg) !important;
            border-color: var(--color-gold-dark) !important;
            box-shadow: 0 2px 8px var(--btn-primary-shadow) !important;
        }}

        .toggle-switch {{
            width: 36px !important;
            height: 20px !important;
            background: rgba(5, 150, 105, 0.3) !important;
            border-radius: 10px !important;
            position: relative !important;
            transition: all 0.3s ease;
            flex-shrink: 0;
        }}

        .toggle-container.active .toggle-switch {{
            background: var(--color-primary) !important;
        }}
        
        .toggle-knob {{
            width: 16px !important;
            height: 16px !important;
            background: white !important;
            border-radius: 50% !important;
            position: absolute !important;
            top: 2px !important;
            left: 2px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        }}
        
        .toggle-container.active .toggle-knob {{
            left: 18px !important;
        }}
        
        .toggle-label {{
            font-size: 12px !important;
            font-weight: 500 !important;
            color: var(--color-text) !important;
            user-select: none !important;
            white-space: nowrap !important;
        }}
        
        .toggle-status {{
            font-size: 10px !important;
            color: var(--color-text-secondary) !important;
            font-weight: 600 !important;
            margin-left: 4px !important;
        }}
        
        .toggle-container.active .toggle-label {{
            color: var(--color-gold-text) !important;
        }}

        .toggle-container.active .toggle-status {{
            color: var(--color-gold-text) !important;
        }}
        
        /* ===== Log Output ===== */
        .log-container {{
            background: #1a1a2e !important;
            color: #00ff88 !important;
            font-family: 'Consolas', 'Monaco', monospace !important;
            border: 2px solid {MODERN_COLORS["primary"]} !important;
            border-radius: 8px !important;
        }}
        
        .log-output {{
            background: rgba(15, 40, 30, 0.95) !important;
            color: #e2e8f0 !important;
            font-family: 'Consolas', 'Monaco', monospace !important;
            border-radius: 12px !important;
            border: 1px solid rgba(16, 185, 129, 0.3) !important;
            box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3) !important;
            line-height: 1.4 !important;
            font-size: 0.85em !important;
        }}
        
        /* .modern-log is defined above in Log Viewer section */

        /* ===== Language Selector ===== */
        .title-glow {{
            text-shadow: 0 0 20px rgba(74, 222, 128, 0.5), 0 0 40px rgba(251, 191, 36, 0.3) !important;
        }}
        
        /* ================================================================
           Dark Mode Overrides
           Now using CSS variables - most colors auto-switch via :root / body.dark-mode vars
           Only Quasar-specific overrides remain here
           ================================================================ */

        /* Body */
        body.dark-mode {{
            background: linear-gradient(135deg, #0d1f1a 0%, #1a2e26 50%, #0f1f18 100%) !important;
        }}

        /* Text colors (.q-field__label is overridden to text-secondary below) */
        body.dark-mode .q-field__native,
        body.dark-mode .q-field__input,
        body.dark-mode .q-field__counter,
        body.dark-mode .q-field__messages,
        body.dark-mode .q-select__input,
        body.dark-mode .text-subtitle1,
        body.dark-mode .text-caption,
        body.dark-mode .text-body2,
        body.dark-mode .q-tab__label,
        body.dark-mode .q-toolbar__title,
        body.dark-mode label {{
            color: var(--color-text) !important;
        }}

        body.dark-mode .q-input input,
        body.dark-mode .q-select input,
        body.dark-mode textarea,
        body.dark-mode input {{
            color: var(--color-emerald-100) !important;
        }}

        body.dark-mode .q-field__label {{
            color: var(--color-text-secondary) !important;
        }}

        /* Cards - use CSS variable */
        body.dark-mode .q-card,
        body.dark-mode .modern-card {{
            background: var(--card-bg) !important;
            border-color: var(--card-border) !important;
        }}

        /* Dark mode tab panel transparency is in the BOTH MODES section below */

        /* Inputs - use CSS variables */
        body.dark-mode .q-field__control,
        body.dark-mode .q-field--outlined .q-field__control,
        body.dark-mode .q-field--filled .q-field__control {{
            background: var(--input-bg) !important;
            border: 2px solid var(--input-border) !important;
        }}

        body.dark-mode .q-field__control:hover,
        body.dark-mode .q-field--outlined .q-field__control:hover {{
            background: var(--input-bg-hover) !important;
        }}

        body.dark-mode .q-field--focused .q-field__control {{
            box-shadow: var(--input-focus-shadow) !important;
        }}

        body.dark-mode .q-field__append,
        body.dark-mode .q-field__prepend {{
            color: var(--color-text-secondary) !important;
        }}

        /* Menu / Dropdown */
        body.dark-mode .q-menu {{
            background: rgba(6, 78, 59, 0.98) !important;
            border: 1px solid var(--color-border-subtle) !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important;
        }}

        body.dark-mode .q-item,
        body.dark-mode .q-menu .q-item,
        body.dark-mode .q-menu .q-item__label,
        body.dark-mode .q-menu .q-item__section,
        body.dark-mode .q-item__label {{
            color: var(--color-text) !important;
        }}

        body.dark-mode .q-item:hover,
        body.dark-mode .q-menu .q-item:hover {{
            background: rgba(5, 150, 105, 0.25) !important;
            color: var(--color-green-400) !important;
        }}

        body.dark-mode .q-item--active,
        body.dark-mode .q-menu .q-item--active {{
            background: linear-gradient(135deg, var(--color-primary), var(--color-accent)) !important;
            color: #ecfdf5 !important;
            font-weight: 600;
        }}

        body.dark-mode .q-item--active .q-item__label,
        body.dark-mode .q-menu .q-item--active .q-item__label {{
            color: #ecfdf5 !important;
        }}

        /* Dark mode placeholder */
        body.dark-mode .q-field__input::placeholder,
        body.dark-mode .q-input::placeholder {{
            color: rgba(110, 231, 183, 0.5) !important;
        }}

        body.dark-mode .q-select__input {{
            color: var(--color-text) !important;
        }}

        /* Tabs */
        body.dark-mode .q-tab:not(.q-tab--active) {{
            color: var(--color-emerald-300) !important;
        }}

        body.dark-mode .q-tab--active {{
            color: var(--color-green-400) !important;
            background: rgba(5, 150, 105, 0.2) !important;
        }}
        
        /* Stepper dark mode */
        body.dark-mode .q-stepper__header {{
            background: rgba(6, 78, 59, 0.6) !important;
            border-color: rgba(5, 150, 105, 0.3) !important;
        }}
        
        body.dark-mode .q-stepper__content {{
            background: rgba(6, 78, 59, 0.4) !important;
            border-color: rgba(5, 150, 105, 0.2) !important;
        }}
        
        body.dark-mode .q-stepper__dot,
        body.dark-mode .q-stepper__dot .q-icon,
        body.dark-mode .q-stepper__dot .material-icons {{
            background: transparent !important;
            background-color: transparent !important;
            color: var(--color-emerald-300) !important;
        }}

        body.dark-mode .q-stepper__tab--active .q-stepper__dot,
        body.dark-mode .q-stepper__tab--active .q-stepper__dot .q-icon,
        body.dark-mode .q-stepper__tab--active .q-stepper__dot .material-icons {{
            background: transparent !important;
            background-color: transparent !important;
            color: var(--color-emerald-300) !important;
        }}
        
        body.dark-mode .q-stepper__line {{
            background: transparent !important;
        }}
        
        /* Focus helper fix is in get_shared_component_css() */
        
        body.dark-mode .q-stepper__tab,
        body.dark-mode .q-stepper__header .q-stepper__tab {{
            color: var(--color-emerald-300) !important;
            background: transparent !important;
            background-color: transparent !important;
        }}
        
        /* Dark mode stepper title fix is in get_shared_component_css()
           (non-mode-specific rules apply to both modes) */
        
        /* Expansion panel dark mode */
        body.dark-mode .q-expansion-item {{
            background: rgba(6, 78, 59, 0.6) !important;
            border-color: rgba(5, 150, 105, 0.2) !important;
        }}
        
        /* Slider dark mode - override text-primary */
        html body.dark-mode .q-btn[class*="slider-value"],
        html body.dark-mode button[class*="slider-value"],
        html body.dark-mode .q-btn.text-primary[class*="slider-value"],
        html body.dark-mode button.q-btn.text-primary[class*="slider-value"],
        html body.dark-mode [class*="editable-slider"] .q-btn[class*="slider-value"],
        html body.dark-mode [class*="editable-slider"] button[class*="slider-value"] {{
            background: transparent !important;
            color: var(--color-gold-light) !important;
            border-color: rgba(251, 191, 36, 0.5) !important;
        }}

        html body.dark-mode .q-btn[class*="slider-value"]:hover,
        html body.dark-mode button[class*="slider-value"]:hover,
        html body.dark-mode .q-btn.text-primary[class*="slider-value"]:hover,
        html body.dark-mode button.q-btn.text-primary[class*="slider-value"]:hover,
        html body.dark-mode [class*="editable-slider"] .q-btn[class*="slider-value"]:hover,
        html body.dark-mode [class*="editable-slider"] button[class*="slider-value"]:hover {{
            background: rgba(251, 191, 36, 0.15) !important;
            color: var(--color-gold-light) !important;
        }}
        
        body.dark-mode .editable-slider .text-xs {{
            color: var(--color-text) !important;
        }}
        
        body.dark-mode .slider-track {{
            background: transparent !important;
        }}
        
        /* Toggle dark mode */
        body.dark-mode .toggle-container {{
            background: rgba(6, 78, 59, 0.6) !important;
            border-color: rgba(5, 150, 105, 0.3) !important;
        }}
        
        body.dark-mode .toggle-container:hover {{
            background: rgba(6, 78, 59, 0.8) !important;
            border-color: rgba(5, 150, 105, 0.5) !important;
        }}
        
        body.dark-mode .toggle-container.active {{
            background: var(--btn-primary-bg) !important;
            border-color: var(--color-gold) !important;
            box-shadow: 0 0 12px rgba(251, 191, 36, 0.5) !important;
        }}

        body.dark-mode .toggle-container.active .toggle-switch {{
            background: rgba(120, 53, 15, 0.5) !important;
        }}

        body.dark-mode .toggle-container.active .toggle-label {{
            color: var(--color-gold-text) !important;
            font-weight: bold;
        }}

        body.dark-mode .toggle-container.active .toggle-status {{
            color: var(--color-gold-text) !important;
            font-weight: bold;
        }}
        
        body.dark-mode .toggle-label {{
            color: var(--color-text) !important;
        }}
        
        body.dark-mode .toggle-status {{
            color: var(--color-text-secondary) !important;
        }}
        
        body.dark-mode .toggle-switch {{
            background: rgba(6, 78, 59, 0.8) !important;
        }}
        
        /* Section title dark mode */
        body.dark-mode .section-title {{
            color: var(--color-text) !important;
            border-left-color: var(--color-green-400) !important;
        }}

        /* Dark mode buttons - use CSS variables (auto-switch via --btn-primary-*) */
        body.dark-mode .modern-btn-primary {{
            background: var(--btn-primary-bg) !important;
            color: var(--btn-primary-text) !important;
            border-color: var(--btn-primary-border) !important;
        }}

        body.dark-mode .modern-btn-primary:hover {{
            background: var(--btn-primary-bg-hover) !important;
            box-shadow: 0 6px 20px var(--btn-primary-shadow-hover) !important;
        }}

        body.dark-mode .modern-btn-primary .q-btn__content {{
            color: var(--btn-primary-text) !important;
        }}

        body.dark-mode .modern-btn-secondary {{
            background: rgba(251, 191, 36, 0.2) !important;
            border-color: rgba(251, 191, 36, 0.5) !important;
            color: var(--color-gold-light) !important;
        }}

        body.dark-mode .modern-btn-secondary:hover {{
            background: rgba(251, 191, 36, 0.3) !important;
            border-color: rgba(251, 191, 36, 0.7) !important;
        }}

        body.dark-mode .modern-btn-ghost {{
            border-color: rgba(251, 191, 36, 0.4) !important;
            color: var(--color-gold-light) !important;
        }}

        body.dark-mode .modern-btn-ghost:hover {{
            background: rgba(251, 191, 36, 0.15) !important;
            border-color: rgba(251, 191, 36, 0.6) !important;
        }}

        body.dark-mode .modern-btn-danger .q-btn__content {{
            color: #fef2f2 !important;
        }}
        
        /* Scrollbar dark mode */
        body.dark-mode ::-webkit-scrollbar-track {{
            background: rgba(2, 44, 34, 0.5);
        }}
        
        /* Checkbox/radio in dark mode */
        body.dark-mode .q-checkbox__bg,
        body.dark-mode .q-radio__bg {{
            border-color: rgba(5, 150, 105, 0.5) !important;
        }}
        
        /* Dark mode placeholder is defined above */

        /* Dialog dark mode */
        body.dark-mode .q-dialog .q-card--dark {{
            background: rgba(6, 78, 59, 0.95) !important;
            border: 1px solid rgba(5, 150, 105, 0.3) !important;
        }}
        
        /* ============================================ */
        /* ULTRA CRITICAL: Final protection against flash */
        /* ============================================ */
        
        /* DARK MODE: Override Quasar primary color to gold */
        html body.dark-mode .q-btn,
        html body.dark-mode button.q-btn,
        html body.dark-mode .q-btn *,
        html body.dark-mode button.q-btn * {{
            --q-color-primary: var(--color-gold-light) !important;
            --q-primary: var(--color-gold-light) !important;
        }}

        /* Override text-primary with gold in dark mode */
        html body.dark-mode .text-primary,
        html body.dark-mode .q-btn.text-primary,
        html body.dark-mode button.text-primary,
        html body.dark-mode .q-btn.q-btn--standard.text-primary,
        html body.dark-mode .q-btn.q-btn--outline.text-primary {{
            color: var(--color-gold-light) !important;
        }}
        
        /* LIGHT MODE: Prevent black flash on tab panel - ULTIMATE protection */
        html body:not(.dark-mode) .q-tab-panel,
        html body:not(.dark-mode) .q-tab-panels,
        html body:not(.dark-mode) .q-tab-panel *,
        html body:not(.dark-mode) .q-tab-panels * {{
            --q-color-dark: transparent !important;
            --q-dark: transparent !important;
        }}

        /* BOTH MODES: Tab panel 容器本身去掉正方形边框 */
        html body .q-tab-panel,
        html body .q-tab-panel.nicegui-tab-panel,
        html body .q-tab-panels,
        html body.dark-mode .q-tab-panel,
        html body.dark-mode .q-tab-panel.nicegui-tab-panel,
        html body.dark-mode .q-tab-panels {{
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Prevent any dark background from Quasar variables in light mode */
        html body:not(.dark-mode) {{
            --q-color-dark: transparent !important;
            --q-dark: transparent !important;
            --q-dark-page: transparent !important;
        }}
    """


# ============== 主题应用函数 ==============

# 全局颜色变量 (默认使用 Modern Theme)
COLORS = MODERN_COLORS.copy()


def apply_theme(theme_name: str = "modern", use_green_gold: bool = False):
    """
    Apply theme CSS to NiceGUI app

    Args:
        theme_name: 'modern' (默认) 或 'green-gold' (来自 sd-scripts)
        use_green_gold: 是否使用 green-gold 主题 (兼容旧API)
    """
    global COLORS

    if use_green_gold or theme_name == "green-gold":
        # 应用 Green Gold 主题
        green_gold_colors = get_green_gold_colors()
        COLORS = {**MODERN_COLORS, **green_gold_colors}  # 合并以确保兼容性
        styles = get_green_gold_styles(COLORS)
        ui.add_head_html(styles, shared=True)
        # 添加共享组件样式
        shared_css = get_shared_component_css()
        ui.add_head_html(f"<style>{shared_css}</style>", shared=True)
        # 添加主题切换样式
        toggle_styles = get_theme_toggle_styles(COLORS)
        ui.add_head_html(toggle_styles, shared=True)
    else:
        # 应用 Modern 主题 (默认)
        COLORS = MODERN_COLORS.copy()

        # 首先添加防闪烁样式 - 确保在CSS加载前就有正确背景
        ui.add_head_html(
            """
        <style>
        /* Pre-load styles: Prevent flash before main CSS loads */
        /* Must use raw hex values since CSS variables aren't defined yet */
        html { background: #f0fdf4 !important; }
        body { background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%) !important; }
        :root { --q-primary: #059669 !important; --q-color-primary: #059669 !important; }

        /* Override Quasar bg-primary on custom button classes — LIGHT MODE ONLY */
        body:not(.dark-mode) .q-btn.modern-nav-btn.bg-primary,
        body:not(.dark-mode) .q-btn.gold-btn.bg-primary,
        body:not(.dark-mode) .q-btn.modern-btn-primary.bg-primary,
        body:not(.dark-mode) .q-btn.modern-btn-success.bg-primary,
        body:not(.dark-mode) button.modern-nav-btn.bg-primary,
        body:not(.dark-mode) button.gold-btn.bg-primary,
        body:not(.dark-mode) button.modern-btn-primary.bg-primary,
        body:not(.dark-mode) button.modern-btn-success.bg-primary {{
            background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
            color: #ffffff !important;
        }}

        /* Override text-primary on custom button classes — LIGHT MODE ONLY */
        body:not(.dark-mode) .q-btn.modern-nav-btn.text-primary,
        body:not(.dark-mode) .q-btn.gold-btn.text-primary,
        body:not(.dark-mode) .q-btn.modern-btn-primary.text-primary,
        body:not(.dark-mode) button.modern-nav-btn.text-primary,
        body:not(.dark-mode) button.gold-btn.text-primary,
        body:not(.dark-mode) button.modern-btn-primary.text-primary {{
            color: #047857 !important;
        }}

        /* Green buttons keep white text */
        body:not(.dark-mode) .q-btn.green-btn.text-primary,
        body:not(.dark-mode) button.green-btn.text-primary {{
            color: white !important;
        }}

        /* Dark mode: keep gold/amber buttons */
        body.dark-mode .q-btn.modern-nav-btn.bg-primary,
        body.dark-mode .q-btn.gold-btn.bg-primary,
        body.dark-mode .q-btn.modern-btn-primary.bg-primary,
        body.dark-mode .q-btn.modern-btn-success.bg-primary,
        body.dark-mode button.modern-nav-btn.bg-primary,
        body.dark-mode button.gold-btn.bg-primary,
        body.dark-mode button.modern-btn-primary.bg-primary,
        body.dark-mode button.modern-btn-success.bg-primary {{
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
            color: #78350f !important;
        }}

        /* Tab panels: prevent dark background flash in light mode */
        body:not(.dark-mode) .q-tab-panels,
        body:not(.dark-mode) .q-tab-panel {{
            background: transparent !important;
            transition: none !important;
        }}

        /* Dark mode: force gold primary */
        body.dark-mode {{ --q-primary: #fbbf24 !important; --q-color-primary: #fbbf24 !important; }}
        </style>
        """,
            shared=True,
        )

        css = get_modern_css()
        ui.add_css(css, shared=True)
        # 添加共享组件样式
        shared_css = get_shared_component_css()
        ui.add_head_html(f"<style>{shared_css}</style>", shared=True)
        # 添加主题切换样式
        toggle_styles = get_theme_toggle_styles(COLORS)
        ui.add_head_html(toggle_styles, shared=True)


def toggle_dark_mode():
    """Toggle dark mode and save preference"""
    ui.run_javascript("""
        (function() {
            const isDark = document.body.classList.toggle('dark-mode');
            localStorage.setItem('dark_mode', isDark);
            // Also save to file for persistence across sessions
            fetch('/api/dark_mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dark_mode: isDark})
            });
            return isDark;
        })();
    """)


def load_theme_preference():
    """Load and apply saved theme preference"""
    ui.run_javascript("""
        (function() {
            const isDark = localStorage.getItem('dark_mode') === 'true';
            if (isDark) {
                document.body.classList.add('dark-mode');
            }
            return isDark;
        })();
    """)


def apply_green_gold_styles(ui_instance=None):
    """
    应用 Green Gold 主题样式 (来自 sd-scripts)
    与 sd-scripts/gui/styles.py 中的 apply_styles 函数兼容

    Args:
        ui_instance: NiceGUI ui 实例 (可选，为兼容性保留)
    """
    global COLORS
    green_gold_colors = get_green_gold_colors()
    COLORS = {**MODERN_COLORS, **green_gold_colors}
    styles = get_green_gold_styles(COLORS)
    ui.add_head_html(styles, shared=True)


def get_classes(name: str) -> str:
    """Get CSS classes for a component type"""
    return MODERN_CLASSES.get(name, "")


# ============== 便捷函数 ==============


def apply_card(element, hover: bool = False):
    """Apply modern card styling"""
    classes = "modern-card"
    if hover:
        classes += " modern-card-hover"
    element.classes(classes)
    return element


def apply_button(element, variant: str = "primary"):
    """Apply modern button styling"""
    class_map = {
        "primary": "modern-btn-primary",
        "secondary": "modern-btn-secondary",
        "danger": "modern-btn-danger",
        "success": "modern-btn-success",
        "ghost": "modern-btn-ghost",
        "gold": "gold-btn",
        "green": "green-btn",
        "red": "red-btn",
    }
    # Remove Quasar default bg classes to prevent color override
    element.classes(remove="bg-primary bg-secondary bg-positive bg-negative bg-info bg-warning")
    element.classes(class_map.get(variant, "modern-btn-primary"))
    return element


def apply_input(element):
    """Apply modern input styling"""
    element.classes("modern-input")
    return element


def apply_section_card(element):
    """Apply section-card styling (from sd-scripts)"""
    element.classes("section-card")
    return element


def apply_section_title(element):
    """Apply section-title styling (from sd-scripts)"""
    element.classes("section-title")
    return element
