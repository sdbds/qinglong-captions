"""
Qinglong Captions GUI — Unified Theme System
Clean dark-first design inspired by Linear + Supabase.

Single CSS Custom Properties system — no duplicate themes, no !important abuse.
Light/dark mode switching via CSS variables on body.dark-mode class.
"""

from nicegui import ui

# ============================================================
#  Color Palette — Single Source of Truth
# ============================================================

COLORS = {
    # Brand — green from character hair, gold from rose
    "primary":        "#6ee07a",   # Mint green (hair color)
    "primary_dark":   "#3cb54a",   # Deeper green
    "primary_light":  "#8df097",   # Lighter green
    "secondary":      "#daa520",   # Goldenrod (rose color)
    "accent":         "#6ee07a",   # Alias for primary

    # Semantic
    "success":        "#6ee07a",
    "warning":        "#daa520",
    "error":          "#f85149",
    "info":           "#58a6ff",

    # Surfaces (dark mode defaults — teal-dark from clothing)
    "background":     "#0f2828",
    "surface":        "#1a3a3a",
    "surface_light":  "#213f3f",
    "surface_dark":   "#0f2828",

    # Text
    "text":           "#e6f0ef",
    "text_secondary": "#8aa8a6",
    "text_muted":     "#4a706e",

    # Borders
    "border":         "#2a5050",

    # Legacy keys
    "blue":           "#daa520",
    "blue_light":     "#e6b422",
    "blue_dark":      "#c49318",
    "mint":           "#6ee07a",
    "emerald":        "#6ee07a",
    "forest":         "#0f2828",
    "bg_light":       "#f0f6f6",
    "text_on_primary": "#0f2828",
    "text_on_accent": "#0f2828",
    "bg_dark":        "#0f2828",
    "text_dark":      "#e6f0ef",
}

# CSS class name mapping — consumed via get_classes()
MODERN_CLASSES = {
    "card":             "ql-card",
    "card_hover":       "ql-card ql-card--hover",
    "card_header":      "ql-card-header",
    "btn_primary":      "ql-btn-primary",
    "btn_secondary":    "ql-btn-secondary",
    "btn_danger":       "ql-btn-danger",
    "btn_success":      "ql-btn-primary",
    "btn_ghost":        "ql-btn-ghost",
    "input":            "ql-input",
    "select":           "ql-select",
    "page_container":   "ql-page",
    "section":          "ql-section",
    "section_title":    "ql-section-title",
    "header":           "ql-header",
    "nav_btn":          "ql-nav-btn",
    "nav_btn_active":   "ql-nav-btn ql-nav-btn--active",
    "badge":            "ql-badge",
    "badge_primary":    "ql-badge ql-badge--primary",
    "badge_success":    "ql-badge ql-badge--success",
    # Legacy aliases
    "section_card":     "ql-card",
    "gold_btn":         "ql-btn-primary",
    "green_btn":        "ql-btn-primary",
    "red_btn":          "ql-btn-danger",
    "footer_green":     "ql-header",
    "header_green":     "ql-header",
}


# ============================================================
#  CSS Generation
# ============================================================

def _css_variables() -> str:
    """CSS Custom Properties for light & dark mode."""
    return """
/* ===== CSS Custom Properties ===== */
:root {
    /* Surfaces — teal-dark from clothing */
    --ql-bg:              #0f2828;
    --ql-surface:         #1a3a3a;
    --ql-surface-raised:  #213f3f;
    --ql-overlay:         #284848;

    /* Brand accent — mint green (hair) */
    --ql-accent:          #6ee07a;
    --ql-accent-hover:    #8df097;
    --ql-accent-muted:    rgba(110, 224, 122, 0.10);
    --ql-accent-border:   rgba(110, 224, 122, 0.20);
    --ql-secondary:       #daa520;

    /* Text */
    --ql-text:            #e6f0ef;
    --ql-text-secondary:  #8aa8a6;
    --ql-text-muted:      #4a706e;
    --ql-text-on-accent:  #0f2828;

    /* Borders */
    --ql-border:          #2a5050;
    --ql-border-hover:    #3a6868;

    /* Semantic */
    --ql-success:         #6ee07a;
    --ql-warning:         #daa520;
    --ql-error:           #f85149;
    --ql-info:            #daa520;

    /* Buttons — gold (rose color) */
    --ql-btn-bg:          #daa520;
    --ql-btn-text:        #ffffff;
    --ql-btn-hover:       #e6b422;
    --ql-btn-border:      #c49318;
    --ql-btn-shadow:      rgba(218, 165, 32, 0.30);

    /* Cards */
    --ql-card-bg:         #1a3a3a;
    --ql-card-border:     #2a5050;
    --ql-card-shadow:     0 1px 3px rgba(0,0,0,0.3);

    /* Inputs */
    --ql-input-bg:        #122e2e;
    --ql-input-border:    #2a5050;
    --ql-input-focus:     rgba(218, 165, 32, 0.35);

    /* Quasar overrides */
    --q-primary: #daa520;
    --q-color-primary: #daa520;

    /* Legacy aliases — backward compat for wizard pages using var(--color-*) */
    --color-primary:        var(--ql-accent);
    --color-primary-dark:   #059669;
    --color-primary-light:  var(--ql-accent-hover);
    --color-secondary:      var(--ql-secondary);
    --color-accent:         var(--ql-accent);
    --color-success:        var(--ql-success);
    --color-warning:        var(--ql-warning);
    --color-error:          var(--ql-error);
    --color-info:           var(--ql-info);
    --color-bg:             var(--ql-bg);
    --color-surface:        var(--ql-surface);
    --color-surface-light:  var(--ql-surface-raised);
    --color-text:           var(--ql-text);
    --color-text-secondary: var(--ql-text-secondary);
    --color-text-muted:     var(--ql-text-muted);
    --color-border:         var(--ql-border);
    --color-border-subtle:  var(--ql-accent-border);
    --card-bg:              var(--ql-card-bg);
    --card-border:          var(--ql-card-border);
    --card-shadow:          var(--ql-card-shadow);
    --input-bg:             var(--ql-input-bg);
    --input-border:         var(--ql-input-border);
    --input-focus-shadow:   0 0 0 2px var(--ql-input-focus);
    --btn-primary-bg:       var(--ql-btn-bg);
    --btn-primary-text:     var(--ql-btn-text);
    --btn-primary-border:   var(--ql-btn-border);
    --btn-primary-shadow:   var(--ql-btn-shadow);
    --btn-primary-color:    var(--ql-accent);
    --color-gold:           var(--ql-accent);
    --color-gold-light:     var(--ql-accent-hover);
    --color-gold-text:      var(--ql-text-on-accent);
    --color-emerald-50:     #ecfdf5;
    --color-emerald-100:    #d1fae5;
    --color-emerald-300:    var(--ql-text-secondary);
    --color-emerald-600:    var(--ql-accent);
    --color-emerald-800:    var(--ql-text);
    --color-green-400:      var(--ql-accent-hover);
    --color-green-50:       var(--ql-surface-raised);
}

/* Light mode overrides */
body:not(.dark-mode) {
    --ql-bg:              #f0f6f6;
    --ql-surface:         #ffffff;
    --ql-surface-raised:  #f0f6f6;
    --ql-overlay:         #ffffff;

    --ql-accent:          #1a7a28;
    --ql-accent-hover:    #15661f;
    --ql-accent-muted:    rgba(26, 122, 40, 0.10);
    --ql-accent-border:   rgba(26, 122, 40, 0.25);

    --ql-text:            #0a1515;
    --ql-text-secondary:  #1e3838;
    --ql-text-muted:      #3d5e5e;
    --ql-text-on-accent:  #ffffff;

    --ql-border:          #b8d4d2;
    --ql-border-hover:    #8aa8a6;

    --ql-btn-bg:          #c49318;
    --ql-btn-text:        #ffffff;
    --ql-btn-hover:       #b08516;
    --ql-btn-border:      #a07a14;
    --ql-btn-shadow:      rgba(196, 147, 24, 0.20);

    --ql-card-bg:         #ffffff;
    --ql-card-border:     #b8d4d2;
    --ql-card-shadow:     0 1px 3px rgba(15,40,40,0.06);

    --ql-input-bg:        #ffffff;
    --ql-input-border:    #b8d4d2;
    --ql-input-focus:     rgba(218, 165, 32, 0.30);

    --q-primary: #c49318;
    --q-color-primary: #c49318;

    /* Legacy aliases must be re-declared here so they pick up light-mode --ql-* values */
    --color-primary:        var(--ql-accent);
    --color-primary-dark:   #15661f;
    --color-primary-light:  var(--ql-accent-hover);
    --color-secondary:      var(--ql-secondary);
    --color-accent:         var(--ql-accent);
    --color-success:        var(--ql-success);
    --color-warning:        var(--ql-warning);
    --color-error:          var(--ql-error);
    --color-info:           var(--ql-info);
    --color-bg:             var(--ql-bg);
    --color-surface:        var(--ql-surface);
    --color-surface-light:  var(--ql-surface-raised);
    --color-text:           var(--ql-text);
    --color-text-secondary: var(--ql-text-secondary);
    --color-text-muted:     var(--ql-text-muted);
    --color-border:         var(--ql-border);
    --color-border-subtle:  var(--ql-accent-border);
    --card-bg:              var(--ql-card-bg);
    --card-border:          var(--ql-card-border);
    --card-shadow:          var(--ql-card-shadow);
    --input-bg:             var(--ql-input-bg);
    --input-border:         var(--ql-input-border);
    --input-focus-shadow:   0 0 0 2px var(--ql-input-focus);
    --btn-primary-bg:       var(--ql-btn-bg);
    --btn-primary-text:     var(--ql-btn-text);
    --btn-primary-border:   var(--ql-btn-border);
    --btn-primary-shadow:   var(--ql-btn-shadow);
    --btn-primary-color:    var(--ql-accent);
    --color-gold:           var(--ql-accent);
    --color-gold-light:     var(--ql-accent-hover);
    --color-gold-text:      var(--ql-text-on-accent);
    --color-emerald-50:     #f0f6f6;
    --color-emerald-100:    #e0eded;
    --color-emerald-300:    var(--ql-text-secondary);
    --color-emerald-600:    var(--ql-accent);
    --color-emerald-800:    var(--ql-text);
    --color-green-400:      var(--ql-accent-hover);
    --color-green-50:       var(--ql-surface-raised);
}
"""


def _base_styles() -> str:
    """Global base styles."""
    return """
/* ===== Base ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html {
    background: var(--ql-bg);
}

body {
    font-family: 'Inter', -apple-system, 'Microsoft YaHei', 'Segoe UI', sans-serif;
    background: var(--ql-bg);
    color: var(--ql-text);
    text-shadow: 0 1px 2px rgba(139, 90, 30, 0.25);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

body:not(.dark-mode) {
    text-shadow: none;
}

#app, #q-app {
    background: transparent;
}

/* ===== Page Container ===== */
.ql-page {
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
    box-sizing: border-box;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--ql-border);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--ql-border-hover);
}
"""


def _header_styles() -> str:
    """Header & navigation."""
    return """
/* ===== Header ===== */
.ql-header, .q-header {
    background: var(--ql-surface);
    border-bottom: 1px solid var(--ql-border);
    box-shadow: none;
    padding: 6px 24px;
    backdrop-filter: blur(12px);
}

/* Logo title */
.header-title { color: var(--ql-text); font-weight: 600; }
.header-version { color: var(--ql-text-muted); font-size: 12px; }

/* ===== Nav Buttons — Gold filled ===== */
.q-btn.ql-nav-btn,
.q-btn.ql-nav-btn.bg-primary,
.q-btn.ql-nav-btn.text-primary {
    background: var(--ql-btn-bg) !important;
    color: #ffffff !important;
    border: 1px solid var(--ql-btn-border);
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 600;
    font-size: 13px;
    text-transform: none;
    letter-spacing: 0;
    box-shadow: 0 1px 3px var(--ql-btn-shadow);
    transition: background 0.15s, box-shadow 0.15s;
}

.q-btn.ql-nav-btn:hover,
.q-btn.ql-nav-btn.text-primary:hover {
    background: var(--ql-btn-hover) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px var(--ql-btn-shadow);
}

.q-btn.ql-nav-btn .q-btn__content {
    color: inherit;
}

.q-btn.ql-nav-btn .q-icon {
    font-size: 18px;
    color: inherit;
}

/* Active nav */
.q-btn.ql-nav-btn--active,
.q-btn.ql-nav-btn--active.bg-primary,
.q-btn.ql-nav-btn--active.text-primary {
    background: var(--ql-btn-hover) !important;
    color: #ffffff !important;
    border-color: var(--ql-btn-border);
    font-weight: 700;
    box-shadow: 0 2px 10px var(--ql-btn-shadow);
}

.q-btn.ql-nav-btn--active .q-icon {
    color: #ffffff;
}
"""


def _card_styles() -> str:
    """Card component."""
    return """
/* ===== Cards ===== */
.ql-card, .modern-card, .section-card {
    background: var(--ql-card-bg);
    border: 1px solid var(--ql-card-border);
    border-radius: 12px;
    box-shadow: var(--ql-card-shadow);
    transition: border-color 0.2s;
}

.ql-card:hover, .modern-card:hover, .section-card:hover {
    border-color: var(--ql-border-hover);
}

.ql-card--hover { cursor: pointer; }
.ql-card--hover:hover {
    border-color: var(--ql-accent-border);
}

.ql-card-header, .modern-card-header {
    border-bottom: 1px solid var(--ql-border);
    padding: 14px 20px;
    border-radius: 12px 12px 0 0;
}

/* Quasar card override */
body.dark-mode .q-card {
    background: var(--ql-card-bg);
    border: 1px solid var(--ql-card-border);
}
body:not(.dark-mode) .q-card {
    background: var(--ql-card-bg);
    border: 1px solid var(--ql-card-border);
}

.q-card__section { padding: 12px 16px; }
"""


def _button_styles() -> str:
    """Button components."""
    return """
/* ===== Buttons ===== */

/* Primary */
.ql-btn-primary, .modern-btn-primary, .modern-btn-success, .gold-btn, .green-btn {
    background: var(--ql-btn-bg);
    color: var(--ql-btn-text);
    border: 1px solid var(--ql-btn-border);
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
    text-transform: none;
    letter-spacing: 0;
    box-shadow: 0 1px 2px var(--ql-btn-shadow);
    transition: background 0.15s, box-shadow 0.15s;
}

.ql-btn-primary:hover, .modern-btn-primary:hover, .modern-btn-success:hover,
.gold-btn:hover, .green-btn:hover {
    background: var(--ql-btn-hover);
    box-shadow: 0 2px 8px var(--ql-btn-shadow);
}

.ql-btn-primary .q-btn__content, .modern-btn-primary .q-btn__content,
.modern-btn-success .q-btn__content, .gold-btn .q-btn__content,
.green-btn .q-btn__content {
    color: var(--ql-btn-text);
}

/* Secondary */
.ql-btn-secondary, .modern-btn-secondary {
    background: transparent;
    color: var(--ql-text);
    border: 1px solid var(--ql-border);
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 13px;
    text-transform: none;
    box-shadow: none;
    transition: background 0.15s, border-color 0.15s;
}

.ql-btn-secondary:hover, .modern-btn-secondary:hover {
    background: var(--ql-accent-muted);
    border-color: var(--ql-border-hover);
}

.ql-btn-secondary .q-btn__content, .modern-btn-secondary .q-btn__content {
    color: var(--ql-text);
}

/* Danger */
.ql-btn-danger, .modern-btn-danger, .red-btn {
    background: var(--ql-error);
    color: #ffffff;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
    text-transform: none;
    box-shadow: 0 1px 2px rgba(248, 81, 73, 0.25);
    transition: background 0.15s;
}

.ql-btn-danger:hover, .modern-btn-danger:hover, .red-btn:hover {
    background: #da3633;
}

.ql-btn-danger .q-btn__content, .modern-btn-danger .q-btn__content,
.red-btn .q-btn__content {
    color: #ffffff;
}

/* Ghost */
.ql-btn-ghost, .modern-btn-ghost {
    background: transparent;
    color: var(--ql-text-secondary);
    border: 1px solid var(--ql-border);
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 13px;
    text-transform: none;
    box-shadow: none;
    transition: color 0.15s, border-color 0.15s;
}

.ql-btn-ghost:hover, .modern-btn-ghost:hover {
    color: var(--ql-text);
    border-color: var(--ql-border-hover);
    background: var(--ql-accent-muted);
}

/* Remove Quasar default button colors on our custom buttons */
.ql-btn-primary.bg-primary, .ql-btn-secondary.bg-primary,
.ql-btn-danger.bg-primary, .ql-btn-ghost.bg-primary,
.ql-nav-btn.bg-primary,
.modern-btn-primary.bg-primary, .modern-btn-secondary.bg-primary,
.gold-btn.bg-primary, .green-btn.bg-primary, .red-btn.bg-primary {
    background: inherit;
}
"""


def _input_styles() -> str:
    """Input / select / field components."""
    return """
/* ===== Inputs & Fields ===== */
.q-field__control {
    background: var(--ql-input-bg);
    border: 1px solid var(--ql-input-border);
    border-radius: 8px;
    min-height: 40px;
    padding-left: 12px;
    transition: border-color 0.15s, box-shadow 0.15s;
}

.q-field__control:hover {
    border-color: var(--ql-border-hover);
}

.q-field--focused .q-field__control {
    border-color: var(--ql-accent);
    box-shadow: 0 0 0 2px var(--ql-input-focus);
}

/* Remove Quasar default field decorations */
.q-field--outlined .q-field__control::before,
.q-field--outlined .q-field__control::after,
.q-field--standard .q-field__control::before,
.q-field--standard .q-field__control::after,
.q-field--filled .q-field__control::before,
.q-field--filled .q-field__control::after {
    border: none;
    display: none;
}

.q-field__label {
    color: var(--ql-text-secondary);
    font-weight: 500;
    font-size: 13px;
}

.q-field__native,
.q-field__input {
    color: var(--ql-text);
    font-weight: 400;
}

.q-field__marginal {
    color: var(--ql-text-muted);
}

.q-field__bottom {
    padding: 2px 12px 0;
}

.q-field__input::placeholder,
.q-input::placeholder {
    color: var(--ql-text-muted);
}

/* Select specific */
.q-select .q-field__control { cursor: pointer; }
.q-select__input { color: var(--ql-text); font-weight: 400; }

.q-select__dropdown-icon {
    color: var(--ql-text-muted);
    transition: transform 0.2s;
}
.q-field--focused .q-select__dropdown-icon {
    transform: rotate(180deg);
}

/* Select chip cleanup */
.q-select .q-field__native > span { background: transparent; padding: 0; }
.q-select .q-chip { background: transparent; color: inherit; padding: 0; margin: 0; border: none; }
.q-select .q-chip__content { color: inherit; }
.q-select .q-field__control-container { padding-top: 0; }

/* Path input */
.path-input .q-field__control {
    background: var(--ql-input-bg);
    border: 1px solid var(--ql-input-border);
    border-radius: 8px;
}
.path-input .q-field__native { background: transparent; color: var(--ql-text); }

/* Force light background classes (backward compat) */
.force-light-bg input { background: transparent; border: none; color: var(--ql-text); }
.modern-input input { background: transparent; border: none; color: var(--ql-text); }
.modern-input input:focus { outline: none; box-shadow: none; background: transparent; }
.modern-select .q-field__native { background: transparent; color: var(--ql-text); }
"""


def _tab_styles() -> str:
    """Tab components."""
    return """
/* ===== Tabs ===== */
.q-tab {
    color: var(--ql-text-secondary);
    font-weight: 500;
    text-transform: none;
    border-radius: 8px;
    margin: 0 2px;
    min-height: 40px;
}

.q-tab--active {
    color: var(--ql-accent);
    background: var(--ql-accent-muted);
}

.q-tab__indicator {
    background: var(--ql-accent);
    height: 2px;
    border-radius: 2px;
}

.q-tab__content { display: flex; align-items: center; gap: 6px; }
.q-tab__icon { font-size: 18px; margin: 0; }
.q-tab__label { font-size: 13px; font-weight: 500; }

/* Tab panels — transparent, no flash */
.q-tab-panels,
.q-tab-panel,
.nicegui-tab-panel {
    background: transparent;
    border: none;
    box-shadow: none;
    transition: none;
}
.q-tab-panel > div,
.q-tab-panel > section {
    background: transparent;
}
"""


def _menu_styles() -> str:
    """Menu / dropdown / dialog."""
    return """
/* ===== Menu / Dropdown ===== */
.q-menu {
    background: var(--ql-surface-raised);
    border: 1px solid var(--ql-border);
    border-radius: 10px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    animation: ql-slide-down 0.15s ease-out;
    transform-origin: top;
}

@keyframes ql-slide-down {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
}

.q-item {
    color: var(--ql-text);
    border-radius: 6px;
    margin: 2px 4px;
    transition: background 0.1s;
}

.q-menu .q-item,
.q-menu .q-item__label,
.q-menu .q-item__section {
    color: var(--ql-text);
}

.q-item:hover,
.q-menu .q-item:hover {
    background: var(--ql-accent-muted);
}

.q-item--active,
.q-menu .q-item--active {
    background: var(--ql-accent);
    color: var(--ql-text-on-accent);
    font-weight: 600;
}

.q-item--active .q-item__label,
.q-menu .q-item--active .q-item__label {
    color: var(--ql-text-on-accent);
}

.q-virtual-scroll__content { padding: 4px; }

/* ===== Dialog ===== */
.q-dialog__backdrop {
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
}

.q-dialog .q-card--dark {
    background: var(--ql-surface);
    border: 1px solid var(--ql-border);
}
"""


def _component_styles() -> str:
    """Slider, toggle, log, stepper, expansion, badge, checkbox, notification."""
    return """
/* ===== Editable Slider ===== */
.editable-slider {
    margin-bottom: 4px;
    flex: 1 1 0;
    min-width: 120px;
    max-width: 100%;
    padding: 2px 0;
}

.slider-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.slider-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--ql-text);
    line-height: 1.2;
}

/* Slider value button */
.q-btn[class*="slider-value"],
button[class*="slider-value"],
.q-btn.text-primary[class*="slider-value"],
[class*="editable-slider"] .q-btn[class*="slider-value"],
[class*="editable-slider"] button[class*="slider-value"] {
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    color: var(--ql-accent);
    background: transparent;
    border: 1px solid var(--ql-accent-border);
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
    text-transform: none;
    min-height: 18px;
    height: 18px;
    padding: 0 4px;
    min-width: 32px;
}

.q-btn[class*="slider-value"]:hover,
button[class*="slider-value"]:hover {
    background: var(--ql-accent-muted);
}

.editable-slider .text-xs { color: var(--ql-text); }

.slider-container {
    position: relative;
    height: 20px;
    display: flex;
    align-items: center;
}

.slider-track {
    position: absolute; left: 0; right: 0;
    height: 4px;
    background: var(--ql-border);
    border-radius: 2px;
}

.slider-fill {
    position: absolute; left: 0;
    height: 4px;
    background: var(--ql-accent);
    border-radius: 2px;
    transition: width 0.15s ease;
}

.slider-thumb {
    position: absolute;
    width: 14px; height: 14px;
    background: var(--ql-accent);
    border: 2px solid var(--ql-surface);
    border-radius: 50%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    transition: left 0.15s ease;
    z-index: 2;
}

.slider-input {
    position: absolute; left: 0;
    width: 100%; height: 100%;
    opacity: 0; cursor: pointer;
    z-index: 3; margin: 0;
}

.slider-edit-input {
    width: 70px;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    text-align: right;
    border: 1px solid var(--ql-accent);
    border-radius: 6px;
    padding: 2px 8px;
    outline: none;
    background: var(--ql-input-bg);
    color: var(--ql-text);
}

/* Quasar Slider Override */
.q-slider {
    --q-color-primary: var(--ql-secondary);
    --q-primary: var(--ql-secondary);
}

.q-slider__track-container,
.q-slider__track-container--h { background: var(--ql-border); border-radius: 2px; }
.q-slider__track { background: transparent; }
.q-slider__track[style*="width"]:not([style*="width: 0"]) {
    background: var(--ql-secondary);
}
.q-slider__thumb { color: var(--ql-secondary); }
.q-slider__thumb-circle { background: var(--ql-secondary); border-color: var(--ql-secondary); }
.q-slider--disabled .q-slider__track-container { background: var(--ql-border); }

/* ===== Toggle Switch ===== */
.toggle-container {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    padding: 4px 10px;
    background: var(--ql-accent-muted);
    border-radius: 16px;
    border: 1px solid var(--ql-border);
    transition: all 0.2s;
    text-transform: none;
    min-height: 28px;
    margin: 2px;
}

.toggle-container .q-btn__content { display: flex; align-items: center; gap: 8px; padding: 0; }

.toggle-container:hover {
    border-color: var(--ql-border-hover);
}

.toggle-container.active {
    background: var(--ql-secondary);
    border-color: var(--ql-btn-border);
}

.toggle-switch {
    width: 32px; height: 18px;
    background: var(--ql-border);
    border-radius: 9px;
    position: relative;
    transition: background 0.2s;
    flex-shrink: 0;
}

.toggle-container.active .toggle-switch { background: rgba(0,0,0,0.2); }

.toggle-knob {
    width: 14px; height: 14px;
    background: white;
    border-radius: 50%;
    position: absolute;
    top: 2px; left: 2px;
    transition: left 0.2s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.toggle-container.active .toggle-knob { left: 16px; }

.toggle-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--ql-text);
    user-select: none;
    white-space: nowrap;
}

.toggle-status {
    font-size: 10px;
    color: var(--ql-text-secondary);
    font-weight: 600;
    margin-left: 2px;
}

.toggle-container.active .toggle-label { color: var(--ql-text-on-accent); }
.toggle-container.active .toggle-status { color: var(--ql-text-on-accent); }

/* ===== Log Output ===== */
.log-container {
    background: #0d1117;
    color: var(--ql-accent);
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    border: 1px solid var(--ql-border);
    border-radius: 8px;
}

.log-output {
    background: #0d1117;
    color: var(--ql-text);
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    border-radius: 8px;
    border: 1px solid var(--ql-border);
    line-height: 1.5;
    font-size: 13px;
}

.modern-log {
    background: #0d1117;
    border: 1px solid var(--ql-border);
    border-radius: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}

.title-glow {
    text-shadow: 0 0 20px rgba(62, 207, 142, 0.4);
}

/* ===== Stepper ===== */
.q-stepper { background: transparent; box-shadow: none; }

.q-stepper__header {
    background: var(--ql-surface);
    border-radius: 10px;
    padding: 8px;
    border: 1px solid var(--ql-border);
}

.q-stepper__tab {
    color: var(--ql-text-secondary);
    border-radius: 8px;
    font-weight: 500;
    background: transparent;
}

.q-stepper__tab--active {
    color: var(--ql-accent);
    background: transparent;
    font-weight: 600;
}

.q-stepper__title,
.q-stepper__tab .q-stepper__title,
.q-stepper__tab--active .q-stepper__title {
    text-shadow: none;
    background: transparent;
    box-shadow: none;
    -webkit-background-clip: initial;
    background-clip: initial;
    border: none;
    -webkit-text-fill-color: inherit;
}

.q-stepper *, .q-stepper__header * { text-shadow: none; }

.q-stepper__tab .q-focus-helper,
.q-stepper__tab--active .q-focus-helper {
    background: transparent;
    opacity: 0;
}

.q-stepper__dot, .q-stepper__dot .q-icon {
    background: transparent;
    color: var(--ql-accent);
}

.q-stepper__line { background: transparent; }

.q-stepper__content {
    background: transparent;
    border-radius: 10px;
    margin-top: 12px;
    border: none;
    box-shadow: none;
}

/* ===== Expansion Panels ===== */
.q-expansion-item {
    background: var(--ql-surface);
    border: 1px solid var(--ql-border);
    border-radius: 10px;
    margin-bottom: 8px;
    overflow: hidden;
}

.q-expansion-item__container { border-radius: 10px; }

/* ===== Badges ===== */
.ql-badge {
    background: var(--ql-accent-muted);
    color: var(--ql-accent);
    border-radius: 16px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
}

.ql-badge--primary {
    background: var(--ql-accent);
    color: var(--ql-text-on-accent);
}

.ql-badge--success {
    background: var(--ql-accent-muted);
    color: var(--ql-accent);
}

/* Legacy badge classes */
.modern-badge {
    background: var(--ql-accent-muted); color: var(--ql-accent);
    border-radius: 16px; padding: 2px 10px;
    font-size: 12px; font-weight: 600;
}
.modern-badge-primary {
    background: var(--ql-accent); color: var(--ql-text-on-accent);
}
.modern-badge-success {
    background: var(--ql-accent-muted); color: var(--ql-accent);
}

/* ===== Checkboxes & Radio ===== */
.q-checkbox__bg, .q-radio__bg {
    border: 2px solid var(--ql-border);
    border-radius: 4px;
}

.q-checkbox__svg,
.q-radio__inner--truthy .q-radio__check {
    color: var(--ql-accent);
}

/* ===== Notifications ===== */
.q-notification {
    border-radius: 10px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

/* ===== Section Titles ===== */
.ql-section-title, .section-title, .modern-section-title {
    color: var(--ql-text);
    font-weight: 600;
    font-size: 14px;
    border-left: 3px solid var(--ql-accent);
    padding-left: 10px;
    margin-bottom: 8px;
    margin-top: 2px;
}

.section-subtitle {
    color: var(--ql-text-secondary);
    font-weight: 500;
    font-size: 13px;
    margin-top: 10px;
    margin-bottom: 6px;
}

/* ===== Footer hidden ===== */
.q-footer { display: none; }
"""


def _homepage_styles() -> str:
    """Homepage-specific: step cards, model items."""
    return """
/* ===== Step Cards (Homepage) ===== */
.step-card {
    background: var(--ql-surface);
    border: 1px solid var(--ql-border);
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
    box-shadow: var(--ql-card-shadow);
    min-height: 160px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    cursor: pointer;
}

.step-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--ql-accent);
    opacity: 0;
    transition: opacity 0.2s;
}

.step-card:hover {
    border-color: var(--ql-accent-border);
    transform: translateY(-2px);
}

.step-card:hover::before { opacity: 1; }

.step-card .q-icon {
    color: var(--ql-accent);
    font-size: 40px;
    margin-bottom: 12px;
    transition: transform 0.2s;
}

.step-card:hover .q-icon { transform: scale(1.1); }

.step-card .step-title,
.step-card .text-h6 { color: var(--ql-text); font-weight: 600; }
.step-card .step-desc,
.step-card .text-body2 { color: var(--ql-text-secondary); }
.step-card .step-label { color: var(--ql-accent); }
.step-card .text-caption { color: var(--ql-text-muted); background: transparent; }

/* ===== Model Items ===== */
.model-item {
    background: var(--ql-surface-raised);
    border: 1px solid var(--ql-border);
    border-radius: 8px;
    padding: 10px 14px;
    transition: border-color 0.15s;
}

.model-item:hover {
    border-color: var(--ql-accent-border);
}

.model-item .model-name { color: var(--ql-text); font-weight: 500; }
.model-item .model-desc,
.model-item .feature-desc { color: var(--ql-text-secondary); }

/* ===== App Desc ===== */
.app-desc { color: var(--ql-text-secondary); }
"""


def _theme_toggle_styles() -> str:
    """Theme toggle button and language selector."""
    return """
/* ===== Theme Toggle ===== */
.theme-toggle-btn {
    background: transparent;
    border: 1px solid var(--ql-border);
    border-radius: 50%;
    width: 36px; height: 36px;
    padding: 0;
    display: flex; align-items: center; justify-content: center;
    transition: border-color 0.15s, background 0.15s;
    cursor: pointer;
}

.theme-toggle-btn:hover {
    background: var(--ql-accent-muted);
    border-color: var(--ql-accent-border);
}

.theme-toggle-btn .q-icon {
    color: var(--ql-text-secondary);
    font-size: 20px;
}

/* ===== Language Selector ===== */
.lang-selector {
    min-width: 130px;
}

.lang-selector .q-field__control {
    background: var(--ql-input-bg);
    border: 1px solid var(--ql-border);
    border-radius: 8px;
    height: 36px; min-height: 36px;
}

.lang-selector .q-field__control:hover {
    border-color: var(--ql-border-hover);
}

.lang-selector .q-field__native {
    color: var(--ql-text);
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.lang-selector .q-icon { color: var(--ql-text-secondary); }
"""


def _responsive_styles() -> str:
    """Responsive breakpoints."""
    return """
/* ===== Responsive ===== */
@media (max-width: 1100px) {
    .ql-header .q-btn__content > span.block,
    .ql-nav-btn .q-btn__content > span.block {
        display: none;
    }
}

@media (max-width: 768px) {
    .ql-header > .row { flex-wrap: wrap; }
    .ql-page { padding: 12px; }
}
"""


def _quasar_overrides() -> str:
    """Override Quasar defaults that fight with our theme."""
    return """
/* ===== Quasar Dark Variable Neutralization ===== */
body:not(.dark-mode) {
    --q-color-dark: transparent;
    --q-dark: transparent;
    --q-dark-page: transparent;
}

/* Override text-primary to use gold */
.text-primary { color: #daa520 !important; }

/* Force --q-primary on body to beat Quasar inline style on html */
body {
    --q-primary: #daa520 !important;
    --q-color-primary: #daa520 !important;
}
body:not(.dark-mode) {
    --q-primary: #c49318 !important;
    --q-color-primary: #c49318 !important;
}

/* White text/icons inside gold bg-primary buttons */
.q-btn.bg-primary .q-btn__content,
.q-btn.bg-primary .q-icon {
    color: #ffffff !important;
}

/* Flat buttons with text-primary → gold text */
.q-btn--flat.text-primary,
.q-btn--flat.text-primary .q-btn__content {
    color: #daa520 !important;
}
.q-btn--flat.text-primary:hover {
    background: rgba(218, 165, 32, 0.12) !important;
}

/* Slider value button (editable-slider) → gold text */
.slider-value,
.slider-value.q-btn {
    color: #daa520 !important;
}

/* Icons — gold by default, white inside filled buttons */
.q-icon { color: #daa520; }
.q-field__marginal .q-icon { color: #daa520; }
.q-tab .q-icon { color: inherit; }
.q-stepper__dot .q-icon { color: #daa520; }
.q-btn.bg-primary .q-icon { color: #ffffff !important; }

/* Slider accent → gold */
.q-slider__thumb { color: var(--ql-secondary); }
.q-slider__thumb-circle {
    background: var(--ql-secondary) !important;
    border-color: var(--ql-secondary) !important;
}

/* Toggle accent → gold */
.q-toggle__inner--truthy .q-toggle__track { background: var(--ql-secondary) !important; }
.q-toggle__inner--truthy .q-toggle__thumb::after { background: var(--ql-secondary) !important; }

/* Quasar buttons: remove default bg-primary/text-primary interference */
body .q-btn.ql-nav-btn,
body .q-btn.ql-btn-primary,
body .q-btn.ql-btn-secondary,
body .q-btn.ql-btn-danger,
body .q-btn.ql-btn-ghost,
body .q-btn.modern-btn-primary,
body .q-btn.modern-btn-secondary,
body .q-btn.modern-btn-danger,
body .q-btn.modern-btn-success,
body .q-btn.modern-btn-ghost,
body .q-btn.gold-btn,
body .q-btn.green-btn,
body .q-btn.red-btn {
    --q-btn-active-opacity: 1;
}

/* Animation Keyframes */
@keyframes ql-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in { animation: ql-fade-in 0.3s ease forwards; }

@keyframes ql-glow {
    0%, 100% { box-shadow: 0 0 4px rgba(218, 165, 32, 0.3); }
    50% { box-shadow: 0 0 16px rgba(218, 165, 32, 0.6); }
}

.animate-glow { animation: ql-glow 2s ease-in-out infinite; }
"""


# ============================================================
#  Pre-load CSS — prevents flash before main CSS loads
# ============================================================

_PRELOAD_CSS = """<style>
html { background: #0f2828; }
body { background: #0f2828; color: #e6f0ef;
       --q-primary: #daa520 !important; --q-color-primary: #daa520 !important; }
body:not(.dark-mode) { background: #f0f6f6; color: #0a1515;
       --q-primary: #c49318 !important; --q-color-primary: #c49318 !important; }
body.dark-mode { --q-primary: #daa520 !important; --q-color-primary: #daa520 !important; }
.q-tab-panels, .q-tab-panel { background: transparent; transition: none; }
</style>"""


# ============================================================
#  Public API
# ============================================================

def apply_theme(theme_name: str = "modern", use_green_gold: bool = False):
    """
    Apply theme CSS to NiceGUI app.

    Args:
        theme_name: Ignored (single unified theme). Kept for backward compat.
        use_green_gold: Ignored. Kept for backward compat.
    """
    # Pre-load styles to prevent flash
    ui.add_head_html(_PRELOAD_CSS, shared=True)

    # Assemble full CSS
    css = "\n".join([
        _css_variables(),
        _base_styles(),
        _header_styles(),
        _card_styles(),
        _button_styles(),
        _input_styles(),
        _tab_styles(),
        _menu_styles(),
        _component_styles(),
        _homepage_styles(),
        _theme_toggle_styles(),
        _responsive_styles(),
        _quasar_overrides(),
    ])
    ui.add_css(css, shared=True)


def get_classes(name: str) -> str:
    """Get CSS classes for a component type."""
    return MODERN_CLASSES.get(name, "")


def toggle_dark_mode():
    """Toggle dark mode and save preference."""
    ui.run_javascript("""
        (function() {
            const isDark = document.body.classList.toggle('dark-mode');
            localStorage.setItem('dark_mode', isDark);
            return isDark;
        })();
    """)


def load_theme_preference():
    """Load and apply saved theme preference."""
    ui.run_javascript("""
        (function() {
            const isDark = localStorage.getItem('dark_mode') === 'true';
            if (isDark) document.body.classList.add('dark-mode');
            return isDark;
        })();
    """)


# ============================================================
#  Convenience Functions (backward compat)
# ============================================================

def apply_card(element, hover: bool = False):
    """Apply card styling."""
    classes = "ql-card"
    if hover:
        classes += " ql-card--hover"
    element.classes(classes)
    return element


def apply_button(element, variant: str = "primary"):
    """Apply button styling."""
    class_map = {
        "primary":   "ql-btn-primary",
        "secondary": "ql-btn-secondary",
        "danger":    "ql-btn-danger",
        "success":   "ql-btn-primary",
        "ghost":     "ql-btn-ghost",
        "gold":      "ql-btn-primary",
        "green":     "ql-btn-primary",
        "red":       "ql-btn-danger",
    }
    element.classes(remove="bg-primary bg-secondary bg-positive bg-negative bg-info bg-warning")
    element.classes(class_map.get(variant, "ql-btn-primary"))
    return element


def apply_input(element):
    """Apply input styling."""
    element.classes("ql-input")
    return element


def apply_section_card(element):
    """Apply section-card styling."""
    element.classes("ql-card")
    return element


def apply_section_title(element):
    """Apply section-title styling."""
    element.classes("ql-section-title")
    return element


# Backward compat: Green Gold theme (now just calls the unified theme)
def apply_green_gold_styles(ui_instance=None):
    """Apply Green Gold theme (now maps to unified theme)."""
    apply_theme()


def get_green_gold_colors():
    """Return colors dict (backward compat)."""
    return COLORS.copy()


# Legacy alias
MODERN_COLORS = COLORS
