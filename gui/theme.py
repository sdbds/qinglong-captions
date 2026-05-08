"""
Qinglong Captions GUI — Unified Theme System
Light palette from the first reference image; dark palette from the second.

Single CSS Custom Properties system — no duplicate themes, no !important abuse.
Light/dark mode switching via CSS variables with html/body dark-mode sync.
"""

from nicegui import ui

# ============================================================
#  Color Palette — Single Source of Truth
# ============================================================

COLORS = {
    # Theme-aware semantic colors
    "primary":         "var(--ql-accent)",
    "primary_dark":    "var(--ql-accent-strong)",
    "primary_light":   "var(--ql-accent-hover)",
    "secondary":       "var(--ql-secondary)",
    "accent":          "var(--ql-accent)",

    # Semantic
    "success":         "var(--ql-success)",
    "warning":         "var(--ql-warning)",
    "error":           "var(--ql-error)",
    "info":            "var(--ql-info)",

    # Surfaces
    "background":      "var(--ql-bg)",
    "surface":         "var(--ql-surface)",
    "surface_light":   "var(--ql-surface-raised)",
    "surface_dark":    "var(--ql-surface-strong)",

    # Text
    "text":            "var(--ql-text)",
    "text_secondary":  "var(--ql-text-secondary)",
    "text_muted":      "var(--ql-text-muted)",

    # Borders
    "border":          "var(--ql-border)",

    # Legacy keys
    "blue":            "var(--ql-info)",
    "blue_light":      "var(--ql-info)",
    "blue_dark":       "var(--ql-info)",
    "mint":            "var(--ql-accent)",
    "emerald":         "var(--ql-success)",
    "forest":          "var(--ql-surface-strong)",
    "bg_light":        "var(--ql-surface-raised)",
    "text_on_primary": "var(--ql-btn-text)",
    "text_on_accent":  "var(--ql-text-on-accent)",
    "bg_dark":         "var(--ql-surface-strong)",
    "text_dark":       "var(--ql-text)",
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
    /* Dark theme — slate ink base + mint highlight + gold auxiliary */
    --ql-bg:              #171b22;
    --ql-surface:         #212934;
    --ql-surface-raised:  #2a3440;
    --ql-surface-strong:  #141b24;
    --ql-overlay:         #313b48;
    --ql-inset-bg:        rgba(255, 255, 255, 0.055);
    --ql-inset-border:    rgba(255, 255, 255, 0.11);
    --ql-console-bg:      #10161e;
    --ql-console-text:    #eef2f6;
    --ql-console-border:  rgba(114, 216, 158, 0.18);

    --ql-accent:          #72d89e;
    --ql-accent-strong:   #58bf84;
    --ql-accent-hover:    #8ce4b1;
    --ql-accent-muted:    rgba(114, 216, 158, 0.16);
    --ql-accent-border:   rgba(114, 216, 158, 0.28);
    --ql-accent-soft:     rgba(114, 216, 158, 0.14);
    --ql-secondary:       #d8b85e;
    --ql-secondary-hover: #e5c874;
    --ql-secondary-muted: rgba(216, 184, 94, 0.18);
    --ql-secondary-border: rgba(216, 184, 94, 0.28);

    --ql-text:            #f3f5f8;
    --ql-text-secondary:  #c3cad6;
    --ql-text-muted:      #8e98a9;
    --ql-text-dim:        rgba(195, 202, 214, 0.74);
    --ql-text-faint:      rgba(195, 202, 214, 0.58);
    --ql-text-ghost:      rgba(195, 202, 214, 0.38);
    --ql-text-on-accent:  #141920;

    --ql-border:          #465262;
    --ql-border-hover:    #5b6a7d;

    --ql-success:         #72d89e;
    --ql-success-soft:    rgba(114, 216, 158, 0.12);
    --ql-success-border:  rgba(114, 216, 158, 0.22);
    --ql-warning:         #d8b85e;
    --ql-warning-soft:    rgba(216, 184, 94, 0.14);
    --ql-warning-border:  rgba(216, 184, 94, 0.24);
    --ql-error:           #c76388;
    --ql-error-hover:     #b45579;
    --ql-error-soft:      rgba(199, 99, 136, 0.14);
    --ql-info:            #c06a80;
    --ql-info-soft:       rgba(192, 106, 128, 0.14);

    --ql-btn-bg:          var(--ql-secondary);
    --ql-btn-text:        #ffffff;
    --ql-btn-icon:        var(--ql-accent);
    --ql-btn-hover:       var(--ql-secondary-hover);
    --ql-btn-border:      #e5c874;
    --ql-btn-shadow:      rgba(216, 184, 94, 0.24);
    --ql-toggle-active-bg: var(--ql-secondary);
    --ql-toggle-active-border: var(--ql-btn-border);
    --ql-toggle-active-text: #ffffff;
    --ql-nav-bg:          rgba(114, 216, 158, 0.06);
    --ql-nav-hover:       rgba(114, 216, 158, 0.12);
    --ql-nav-border:      rgba(114, 216, 158, 0.16);
    --ql-nav-text:        #d7e4dc;
    --ql-nav-icon:        var(--ql-accent);
    --ql-nav-active-bg:   rgba(114, 216, 158, 0.20);
    --ql-nav-active-text: #b7f0ce;
    --ql-nav-active-icon: #9ce8bb;
    --ql-nav-active-border: rgba(114, 216, 158, 0.36);
    --ql-nav-shadow:      rgba(12, 17, 24, 0.18);

    --ql-card-bg:         var(--ql-surface-raised);
    --ql-card-border:     var(--ql-border);
    --ql-card-shadow:     0 16px 38px rgba(4, 9, 14, 0.24);

    --ql-input-bg:        #19212b;
    --ql-input-border:    var(--ql-border);
    --ql-input-focus:     rgba(114, 216, 158, 0.24);

    --q-primary: var(--ql-accent);
    --q-color-primary: var(--ql-accent);

    /* Legacy aliases — backward compat for wizard pages using var(--color-*) */
    --color-primary:        var(--ql-accent);
    --color-primary-dark:   var(--ql-accent-strong);
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
    --btn-primary-color:    var(--ql-btn-bg);
    --color-gold:           var(--ql-secondary);
    --color-gold-light:     var(--ql-secondary-hover);
    --color-gold-text:      var(--ql-text-on-accent);
    --color-emerald-50:     var(--ql-surface-raised);
    --color-emerald-100:    var(--ql-surface);
    --color-emerald-300:    var(--ql-text-secondary);
    --color-emerald-600:    var(--ql-success);
    --color-emerald-800:    var(--ql-text);
    --color-green-400:      var(--ql-accent-hover);
    --color-green-50:       var(--ql-surface-raised);
}

/* Light mode overrides */
body:not(.dark-mode) {
    /* Light theme — warm ivory paper + ribbon mauve + champagne */
    --ql-bg:              #f7f0e6;
    --ql-surface:         #fffaf4;
    --ql-surface-raised:  #f2e8dc;
    --ql-surface-strong:  #eadfce;
    --ql-overlay:         #fffdf8;
    --ql-inset-bg:        rgba(128, 97, 138, 0.05);
    --ql-inset-border:    rgba(128, 97, 138, 0.12);
    --ql-console-bg:      #f3ece3;
    --ql-console-text:    #392f2b;
    --ql-console-border:  rgba(128, 97, 138, 0.16);

    --ql-accent:          #80618a;
    --ql-accent-strong:   #6d5978;
    --ql-accent-hover:    #9578a0;
    --ql-accent-muted:    rgba(128, 97, 138, 0.12);
    --ql-accent-border:   rgba(128, 97, 138, 0.22);
    --ql-accent-soft:     rgba(128, 97, 138, 0.08);
    --ql-secondary:       #b88746;
    --ql-secondary-hover: #c99955;
    --ql-secondary-muted: rgba(184, 135, 70, 0.16);
    --ql-secondary-border: rgba(184, 135, 70, 0.24);

    --ql-text:            #392f2b;
    --ql-text-secondary:  #6c5f58;
    --ql-text-muted:      #93857b;
    --ql-text-dim:        rgba(108, 95, 88, 0.72);
    --ql-text-faint:      rgba(108, 95, 88, 0.58);
    --ql-text-ghost:      rgba(108, 95, 88, 0.38);
    --ql-text-on-accent:  #ffffff;

    --ql-border:          #c6b49e;
    --ql-border-hover:    #b09d86;

    --ql-success:         #567860;
    --ql-success-soft:    rgba(86, 120, 96, 0.10);
    --ql-success-border:  rgba(86, 120, 96, 0.20);
    --ql-warning:         #8d602f;
    --ql-warning-soft:    rgba(141, 96, 47, 0.12);
    --ql-warning-border:  rgba(141, 96, 47, 0.22);
    --ql-error:           #a65466;
    --ql-error-hover:     #924759;
    --ql-error-soft:      rgba(166, 84, 102, 0.12);
    --ql-info:            #a65466;
    --ql-info-soft:       rgba(166, 84, 102, 0.12);

    --ql-btn-bg:          #e4d5c0;
    --ql-btn-text:        #392f2b;
    --ql-btn-icon:        var(--ql-accent);
    --ql-btn-hover:       #dcc7aa;
    --ql-btn-border:      #c5ae8a;
    --ql-btn-shadow:      rgba(90, 70, 52, 0.14);
    --ql-toggle-active-bg: #d7b455;
    --ql-toggle-active-border: #e4c46f;
    --ql-toggle-active-text: #392f2b;
    --ql-nav-bg:          rgba(128, 97, 138, 0.08);
    --ql-nav-hover:       rgba(128, 97, 138, 0.14);
    --ql-nav-border:      rgba(128, 97, 138, 0.22);
    --ql-nav-text:        #6d5978;
    --ql-nav-icon:        var(--ql-accent);
    --ql-nav-active-bg:   var(--ql-accent);
    --ql-nav-active-text: #ffffff;
    --ql-nav-active-icon: var(--ql-accent);
    --ql-nav-active-border: var(--ql-accent-strong);
    --ql-nav-shadow:      rgba(128, 97, 138, 0.18);

    --ql-card-bg:         var(--ql-surface);
    --ql-card-border:     #d2c0aa;
    --ql-card-shadow:     0 8px 22px rgba(58, 47, 43, 0.08);

    --ql-input-bg:        var(--ql-surface);
    --ql-input-border:    #b6a28a;
    --ql-input-focus:     rgba(128, 97, 138, 0.20);

    --q-primary: var(--ql-accent);
    --q-color-primary: var(--ql-accent);

    /* Legacy aliases must be re-declared here so they pick up light-mode --ql-* values */
    --color-primary:        var(--ql-accent);
    --color-primary-dark:   var(--ql-accent-strong);
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
    --btn-primary-color:    var(--ql-btn-bg);
    --color-gold:           var(--ql-secondary);
    --color-gold-light:     var(--ql-secondary-hover);
    --color-gold-text:      var(--ql-text-on-accent);
    --color-emerald-50:     var(--ql-surface-raised);
    --color-emerald-100:    var(--ql-surface);
    --color-emerald-300:    var(--ql-text-secondary);
    --color-emerald-600:    var(--ql-success);
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
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

body.dark-mode {
    background:
        radial-gradient(circle at top right, rgba(114, 216, 158, 0.06), transparent 18%),
        radial-gradient(circle at bottom left, rgba(216, 184, 94, 0.05), transparent 20%),
        linear-gradient(180deg, #090d12 0%, #0d1218 42%, #111821 100%);
    background-attachment: fixed;
}

#app, #q-app {
    background: transparent;
}

/* ===== Page Container ===== */
.ql-page {
    width: min(1240px, calc(100% - 104px));
    max-width: 1240px;
    margin: 28px auto 40px;
    padding: 32px;
    position: relative;
    border-radius: 30px;
    box-sizing: border-box;
    overflow: clip;
}

body:not(.dark-mode) .ql-page {
    border: 1px solid rgba(210, 192, 170, 0.92);
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.72) 0%, rgba(255, 255, 255, 0.24) 14%, rgba(255, 255, 255, 0.10) 100%),
        rgba(255, 250, 244, 0.96);
    box-shadow:
        0 24px 64px rgba(58, 47, 43, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.45);
}

body.dark-mode .ql-page {
    border: 1px solid rgba(86, 103, 125, 0.95);
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.065) 0%, rgba(255, 255, 255, 0.024) 14%, rgba(255, 255, 255, 0.010) 100%),
        rgba(31, 40, 52, 0.98);
    box-shadow:
        0 36px 96px rgba(2, 5, 9, 0.54),
        inset 0 1px 0 rgba(255, 255, 255, 0.065);
    backdrop-filter: blur(12px);
}

body:not(.dark-mode) .ql-page::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    background:
        radial-gradient(circle at top center, rgba(128, 97, 138, 0.05), transparent 34%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.24), transparent 24%);
}

body.dark-mode .ql-page::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    background:
        radial-gradient(circle at top center, rgba(114, 216, 158, 0.07), transparent 34%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.022), transparent 22%);
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

/* ===== Nav Buttons ===== */
.q-btn.ql-nav-btn,
.q-btn.ql-nav-btn.bg-primary,
.q-btn.ql-nav-btn.text-primary {
    background: var(--ql-nav-bg) !important;
    color: var(--ql-nav-text) !important;
    border: 1px solid var(--ql-nav-border);
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 600;
    font-size: 13px;
    text-transform: none;
    letter-spacing: 0;
    box-shadow: 0 1px 3px var(--ql-nav-shadow);
    transition: background 0.15s, box-shadow 0.15s;
}

.q-btn.ql-nav-btn:hover,
.q-btn.ql-nav-btn.text-primary:hover {
    background: var(--ql-nav-hover) !important;
    color: var(--ql-nav-text) !important;
    border-color: var(--ql-nav-active-border);
    box-shadow: 0 2px 8px var(--ql-nav-shadow);
}

.q-btn.ql-nav-btn .q-btn__content {
    color: inherit;
}

.q-btn.ql-nav-btn .q-icon {
    font-size: 18px;
    color: var(--ql-nav-icon);
}

/* Active nav */
.q-btn.ql-nav-btn--active,
.q-btn.ql-nav-btn--active.bg-primary,
.q-btn.ql-nav-btn--active.text-primary {
    background: var(--ql-nav-active-bg) !important;
    color: var(--ql-nav-active-text) !important;
    border-color: var(--ql-nav-active-border);
    font-weight: 700;
    box-shadow: 0 2px 10px var(--ql-nav-shadow);
}

.q-btn.ql-nav-btn--active .q-icon {
    color: var(--ql-nav-active-icon);
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

body.dark-mode .ql-card,
body.dark-mode .modern-card,
body.dark-mode .section-card,
body.dark-mode .q-card {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.028) 0%, rgba(255, 255, 255, 0.012) 100%),
        var(--ql-card-bg);
    box-shadow: var(--ql-card-shadow);
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
.q-btn.ql-btn-primary, .q-btn.modern-btn-primary, .q-btn.modern-btn-success,
.q-btn.gold-btn, .q-btn.green-btn,
.ql-btn-primary, .modern-btn-primary, .modern-btn-success, .gold-btn, .green-btn {
    background: var(--ql-btn-bg) !important;
    color: var(--ql-btn-text) !important;
    border: 1px solid var(--ql-btn-border) !important;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
    text-transform: none;
    letter-spacing: 0;
    box-shadow: 0 1px 2px var(--ql-btn-shadow) !important;
    transition: background 0.15s, box-shadow 0.15s;
}

.q-btn.ql-btn-primary:hover, .q-btn.modern-btn-primary:hover, .q-btn.modern-btn-success:hover,
.q-btn.gold-btn:hover, .q-btn.green-btn:hover,
.ql-btn-primary:hover, .modern-btn-primary:hover, .modern-btn-success:hover,
.gold-btn:hover, .green-btn:hover {
    background: var(--ql-btn-hover) !important;
    box-shadow: 0 2px 8px var(--ql-btn-shadow) !important;
}

.ql-btn-primary .q-btn__content > span, .modern-btn-primary .q-btn__content > span,
.modern-btn-success .q-btn__content > span, .gold-btn .q-btn__content > span,
.green-btn .q-btn__content > span {
    color: var(--ql-btn-text) !important;
}

.ql-btn-primary .q-icon, .modern-btn-primary .q-icon,
.modern-btn-success .q-icon, .gold-btn .q-icon,
.green-btn .q-icon {
    color: var(--ql-btn-icon) !important;
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
    box-shadow: 0 1px 2px var(--ql-error-soft);
    transition: background 0.15s;
}

.ql-btn-danger:hover, .modern-btn-danger:hover, .red-btn:hover {
    background: var(--ql-error-hover);
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

body.dark-mode .q-field__control {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.018) 0%, rgba(255, 255, 255, 0) 100%),
        var(--ql-input-bg);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
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

.q-tabs {
    padding: 6px;
    border-radius: 16px;
}

.q-tab {
    padding: 0 18px;
}

body:not(.dark-mode) .q-tabs {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.52) 0%, rgba(255, 255, 255, 0.18) 100%),
        var(--ql-surface-raised);
    border: 1px solid var(--ql-border);
    box-shadow: 0 10px 24px rgba(58, 47, 43, 0.06);
}

body:not(.dark-mode) .q-tab--active {
    border: 1px solid var(--ql-accent-border);
}

body:not(.dark-mode) .q-tab__indicator {
    opacity: 0;
}

body.dark-mode .q-tabs {
    padding: 6px;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.024) 0%, rgba(255, 255, 255, 0.01) 100%),
        var(--ql-surface-strong);
    border: 1px solid var(--ql-border);
    border-radius: 16px;
    box-shadow: 0 10px 24px rgba(7, 12, 18, 0.18);
}

body.dark-mode .q-tab--active {
    color: #b7f0ce;
    background: linear-gradient(180deg, rgba(114, 216, 158, 0.18), rgba(114, 216, 158, 0.10));
    border: 1px solid var(--ql-accent-border);
}

body.dark-mode .q-tab__indicator {
    opacity: 0;
}

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

/* Editable slider needs a thin rail, not a filled container block */
.editable-slider .q-slider__track-container,
.editable-slider .q-slider__track-container--h {
    height: 3px !important;
    background: transparent !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    border: none !important;
}
.editable-slider .q-slider__track[style*="width"]:not([style*="width: 0"]) {
    background: var(--ql-secondary) !important;
    border-radius: 999px !important;
}
.editable-slider .q-slider__thumb {
    color: var(--ql-secondary);
    background: transparent !important;
    border-radius: 999px !important;
    box-shadow: none !important;
}
.editable-slider .q-slider__thumb-circle {
    color: var(--ql-secondary);
    background: var(--ql-secondary) !important;
    border-color: var(--ql-secondary) !important;
    border-radius: 999px !important;
}

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
    background: var(--ql-toggle-active-bg);
    border-color: var(--ql-toggle-active-border);
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

.toggle-container.active .toggle-label { color: var(--ql-toggle-active-text); }
.toggle-container.active .toggle-status { color: var(--ql-toggle-active-text); }

/* ===== Log Output ===== */
.log-container {
    background: var(--ql-console-bg);
    color: var(--ql-accent);
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    border: 1px solid var(--ql-border);
    border-radius: 8px;
}

.log-output {
    background: var(--ql-console-bg);
    color: var(--ql-text);
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    border-radius: 8px;
    border: 1px solid var(--ql-border);
    line-height: 1.5;
    font-size: 13px;
}

.modern-log {
    background: var(--ql-console-bg);
    border: 1px solid var(--ql-border);
    border-radius: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}

body.dark-mode .log-container,
body.dark-mode .log-output,
body.dark-mode .modern-log {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0) 100%),
        var(--ql-console-bg);
    border-color: var(--ql-console-border);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.title-glow {
    text-shadow: 0 0 20px var(--ql-accent-border);
}

/* ===== Stepper ===== */
.q-stepper { background: transparent; box-shadow: none; }

.q-stepper__header {
    background: var(--ql-surface);
    border-radius: 10px;
    padding: 8px;
    border: 1px solid var(--ql-border);
}

body.dark-mode .q-stepper__header {
    background: var(--ql-surface-raised);
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

body.dark-mode .q-expansion-item {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.022) 0%, rgba(255, 255, 255, 0.008) 100%),
        var(--ql-surface-raised);
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
    .ql-page {
        width: calc(100% - 16px);
        margin: 8px auto 20px;
        padding: 16px;
        border-radius: 18px;
    }
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

/* Override text-primary to follow theme primary accent */
.text-primary { color: var(--ql-accent) !important; }

/* Force --q-primary on body to beat Quasar inline style on html */
body {
    --q-primary: var(--ql-accent) !important;
    --q-color-primary: var(--ql-accent) !important;
}

/* Theme-aware text inside generic primary buttons */
.q-btn.bg-primary .q-btn__content > span {
    color: var(--ql-btn-text) !important;
}

/* Flat buttons with text-primary → theme accent */
.q-btn--flat.text-primary,
.q-btn--flat.text-primary .q-btn__content {
    color: var(--ql-accent) !important;
}
.q-btn--flat.text-primary:hover {
    background: var(--ql-accent-muted) !important;
}

/* Slider value button (editable-slider) → theme gold */
.slider-value,
.slider-value.q-btn {
    color: var(--ql-secondary) !important;
}

/* Icons keep their own component/local colors. */
.q-tab .q-icon { color: inherit; }

/* Slider accent → secondary accent */
.q-slider__thumb { color: var(--ql-secondary); }
.q-slider__thumb-circle {
    background: var(--ql-secondary) !important;
    border-color: var(--ql-secondary) !important;
}

/* Toggle accent → dedicated active toggle color */
.q-toggle__inner--truthy .q-toggle__track { background: var(--ql-toggle-active-bg) !important; }
.q-toggle__inner--truthy .q-toggle__thumb::after { background: var(--ql-toggle-active-bg) !important; }

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

/* Final override for custom filled buttons: beat Quasar's default bg-primary */
body .q-btn.ql-btn-primary,
body .q-btn.modern-btn-primary,
body .q-btn.modern-btn-success,
body .q-btn.gold-btn,
body .q-btn.green-btn {
    background: var(--ql-btn-bg) !important;
    color: var(--ql-btn-text) !important;
    border: 1px solid var(--ql-btn-border) !important;
    box-shadow: 0 1px 2px var(--ql-btn-shadow) !important;
}

body .q-btn.ql-btn-primary:hover,
body .q-btn.modern-btn-primary:hover,
body .q-btn.modern-btn-success:hover,
body .q-btn.gold-btn:hover,
body .q-btn.green-btn:hover {
    background: var(--ql-btn-hover) !important;
    box-shadow: 0 2px 8px var(--ql-btn-shadow) !important;
}

body .q-btn.ql-btn-primary .q-btn__content > span,
body .q-btn.modern-btn-primary .q-btn__content > span,
body .q-btn.modern-btn-success .q-btn__content > span,
body .q-btn.gold-btn .q-btn__content > span,
body .q-btn.green-btn .q-btn__content > span {
    color: var(--ql-btn-text) !important;
}

body .q-btn.ql-btn-primary .q-icon,
body .q-btn.modern-btn-primary .q-icon,
body .q-btn.modern-btn-success .q-icon,
body .q-btn.gold-btn .q-icon,
body .q-btn.green-btn .q-icon {
    color: var(--ql-btn-icon) !important;
}

/* Final override for custom ghost buttons */
body .q-btn.ql-btn-ghost,
body .q-btn.modern-btn-ghost {
    background: transparent !important;
    color: var(--ql-text-secondary) !important;
    border: 1px solid var(--ql-border) !important;
    box-shadow: none !important;
}

body .q-btn.ql-btn-ghost:hover,
body .q-btn.modern-btn-ghost:hover {
    background: var(--ql-accent-muted) !important;
    color: var(--ql-text) !important;
    border-color: var(--ql-border-hover) !important;
}

body .q-btn.ql-btn-ghost .q-btn__content,
body .q-btn.modern-btn-ghost .q-btn__content,
body .q-btn.ql-btn-ghost .q-icon,
body .q-btn.modern-btn-ghost .q-icon {
    color: inherit !important;
}

/* Animation Keyframes */
@keyframes ql-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in { animation: ql-fade-in 0.3s ease forwards; }

@keyframes ql-glow {
    0%, 100% { box-shadow: 0 0 4px var(--ql-accent-border); }
    50% { box-shadow: 0 0 16px var(--ql-accent-muted); }
}

.animate-glow { animation: ql-glow 2s ease-in-out infinite; }
"""


# ============================================================
#  Pre-load CSS — prevents flash before main CSS loads
# ============================================================

_PRELOAD_THEME_INIT = """<script>
(function() {
    var saved = localStorage.getItem('dark_mode');
    var isDark = saved === null ? true : saved === 'true';
    document.documentElement.classList.toggle('dark-mode', isDark);
    document.documentElement.classList.toggle('light-mode', !isDark);
    document.addEventListener('DOMContentLoaded', function() {
        if (!document.body) return;
        document.body.classList.toggle('dark-mode', isDark);
    });
})();
</script>"""

_PRELOAD_CSS = """<style>
html {
    background: #171b22;
    color: #f3f5f8;
    --ql-bg: #171b22;
    --ql-surface: #212934;
    --ql-surface-raised: #2a3440;
    --ql-surface-strong: #141b24;
    --ql-overlay: #313b48;
    --ql-border: #465262;
    --ql-border-hover: #5b6a7d;
    --ql-inset-bg: rgba(255, 255, 255, 0.055);
    --ql-inset-border: rgba(255, 255, 255, 0.11);
    --ql-console-bg: #10161e;
    --ql-console-text: #eef2f6;
    --ql-console-border: rgba(114, 216, 158, 0.18);
    --ql-accent: #72d89e;
    --ql-accent-strong: #58bf84;
    --ql-accent-hover: #8ce4b1;
    --ql-accent-muted: rgba(114, 216, 158, 0.16);
    --ql-accent-border: rgba(114, 216, 158, 0.28);
    --ql-accent-soft: rgba(114, 216, 158, 0.14);
    --ql-secondary: #d8b85e;
    --ql-secondary-hover: #e5c874;
    --ql-secondary-muted: rgba(216, 184, 94, 0.18);
    --ql-secondary-border: rgba(216, 184, 94, 0.28);
    --ql-success: #72d89e;
    --ql-warning: #d8b85e;
    --ql-info: #c06a80;
    --ql-text: #f3f5f8;
    --ql-text-secondary: #c3cad6;
    --ql-text-muted: #8e98a9;
    --ql-text-dim: rgba(195, 202, 214, 0.74);
    --ql-text-faint: rgba(195, 202, 214, 0.58);
    --ql-text-ghost: rgba(195, 202, 214, 0.38);
    --ql-btn-text: #ffffff;
    --ql-btn-icon: #72d89e;
    --ql-btn-bg: #d8b85e;
    --ql-btn-hover: #e5c874;
    --ql-btn-border: #e5c874;
    --ql-nav-bg: rgba(114, 216, 158, 0.06);
    --ql-nav-hover: rgba(114, 216, 158, 0.12);
    --ql-nav-border: rgba(114, 216, 158, 0.16);
    --ql-nav-text: #d7e4dc;
    --ql-nav-icon: #72d89e;
    --ql-nav-active-bg: rgba(114, 216, 158, 0.20);
    --ql-nav-active-text: #b7f0ce;
    --ql-nav-active-icon: #9ce8bb;
    --ql-nav-active-border: rgba(114, 216, 158, 0.36);
    --ql-card-bg: #2a3440;
    --ql-card-border: #465262;
    --q-primary: #72d89e !important;
    --q-color-primary: #72d89e !important;
}
html:not(.dark-mode) {
    background: #f7f0e6;
    color: #392f2b;
    --ql-bg: #f7f0e6;
    --ql-surface: #fffaf4;
    --ql-surface-raised: #f2e8dc;
    --ql-surface-strong: #eadfce;
    --ql-overlay: #fffdf8;
    --ql-border: #c6b49e;
    --ql-border-hover: #b09d86;
    --ql-accent: #80618a;
    --ql-accent-strong: #6d5978;
    --ql-accent-hover: #9578a0;
    --ql-accent-muted: rgba(128, 97, 138, 0.12);
    --ql-accent-border: rgba(128, 97, 138, 0.22);
    --ql-accent-soft: rgba(128, 97, 138, 0.08);
    --ql-secondary: #b88746;
    --ql-secondary-hover: #c99955;
    --ql-secondary-muted: rgba(184, 135, 70, 0.16);
    --ql-secondary-border: rgba(184, 135, 70, 0.24);
    --ql-success: #567860;
    --ql-warning: #8d602f;
    --ql-info: #a65466;
    --ql-text: #392f2b;
    --ql-text-secondary: #6c5f58;
    --ql-text-muted: #93857b;
    --ql-text-dim: rgba(108, 95, 88, 0.72);
    --ql-text-faint: rgba(108, 95, 88, 0.58);
    --ql-text-ghost: rgba(108, 95, 88, 0.38);
    --ql-console-bg: #f3ece3;
    --ql-console-text: #392f2b;
    --ql-console-border: rgba(128, 97, 138, 0.16);
    --ql-btn-text: #392f2b;
    --ql-btn-icon: #80618a;
    --ql-btn-bg: #e4d5c0;
    --ql-btn-hover: #dcc7aa;
    --ql-btn-border: #c5ae8a;
    --ql-nav-bg: rgba(128, 97, 138, 0.08);
    --ql-nav-hover: rgba(128, 97, 138, 0.14);
    --ql-nav-border: rgba(128, 97, 138, 0.22);
    --ql-nav-text: #6d5978;
    --ql-nav-icon: #80618a;
    --ql-nav-active-bg: #80618a;
    --ql-nav-active-text: #ffffff;
    --ql-nav-active-icon: #80618a;
    --ql-nav-active-border: #6d5978;
    --ql-card-bg: #fffaf4;
    --ql-card-border: #d2c0aa;
    --q-primary: #80618a !important;
    --q-color-primary: #80618a !important;
}
body {
    background: inherit;
    color: inherit;
}
.ql-page {
    width: min(1240px, calc(100% - 104px));
    max-width: 1240px;
    margin: 28px auto 40px;
    padding: 32px;
    position: relative;
    border-radius: 30px;
    box-sizing: border-box;
    overflow: clip;
}
html:not(.dark-mode) .ql-page,
body:not(.dark-mode) .ql-page {
    border: 1px solid rgba(210, 192, 170, 0.92);
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.72) 0%, rgba(255, 255, 255, 0.24) 14%, rgba(255, 255, 255, 0.10) 100%),
        rgba(255, 250, 244, 0.96);
    box-shadow:
        0 24px 64px rgba(58, 47, 43, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.45);
}
html.dark-mode,
body.dark-mode {
    background:
        radial-gradient(circle at top right, rgba(114, 216, 158, 0.06), transparent 18%),
        radial-gradient(circle at bottom left, rgba(216, 184, 94, 0.05), transparent 20%),
        linear-gradient(180deg, #090d12 0%, #0d1218 42%, #111821 100%);
}
.dark-mode .ql-page {
    width: min(1240px, calc(100% - 104px));
    max-width: 1240px;
    margin: 28px auto 40px;
    padding: 32px;
    position: relative;
    border-radius: 30px;
    border: 1px solid rgba(86, 103, 125, 0.95);
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.065) 0%, rgba(255, 255, 255, 0.024) 14%, rgba(255, 255, 255, 0.010) 100%),
        rgba(31, 40, 52, 0.98);
    box-shadow:
        0 36px 96px rgba(2, 5, 9, 0.54),
        inset 0 1px 0 rgba(255, 255, 255, 0.065);
    backdrop-filter: blur(12px);
}
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
    ui.add_head_html(_PRELOAD_THEME_INIT, shared=True)
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
