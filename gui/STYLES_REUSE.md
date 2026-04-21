# Theme System — Style Guide

Unified theme system based on CSS Custom Properties. The light palette is
derived from the first reference image (ivory + mauve + champagne); the dark
palette is derived from the second reference image (deep teal + mint + amber).

## Architecture

- **Single theme** — no duplicate green-gold / modern split
- **CSS Custom Properties** — `--ql-*` namespace, auto-switch via `body.dark-mode`
- **Legacy aliases** — old `var(--color-*)`, `var(--card-*)`, `var(--btn-primary-*)` still work
- **Scoped `!important` only for Quasar overrides** — keep component styles token-driven

## Quick Start

```python
from gui.theme import apply_theme, get_classes, COLORS

apply_theme()  # Call once per page in page_base()

# Use CSS classes
with ui.card().classes(get_classes('card')):
    ui.label('Content')

# Use theme-aware color values
ui.icon("star").style(f"color: {COLORS['primary']};")
```

`COLORS[...]` now resolves to `var(--ql-*)` strings so inline styles follow
light/dark mode automatically. Do not append alpha suffixes like `22` to
`COLORS` values; define a CSS token instead.

## Color Palette

| Key | Value | Usage |
|-----|-------|-------|
| `primary` | `var(--ql-accent)` | Theme accent: mauve in light, mint in dark |
| `primary_dark` | `var(--ql-accent-strong)` | Stronger accent for borders / pressed states |
| `primary_light` | `var(--ql-accent-hover)` | Softer accent highlight |
| `secondary` | `var(--ql-secondary)` | Champagne / amber secondary accent |
| `success` | `var(--ql-success)` | Success state |
| `warning` | `var(--ql-warning)` | Warning state |
| `error` | `var(--ql-error)` | Error / danger state |
| `info` | `var(--ql-info)` | Information state |

## CSS Class Mapping

Use `get_classes(name)` to get class strings:

| Name | CSS Class | Usage |
|------|-----------|-------|
| `card` | `ql-card` | Standard card |
| `card_hover` | `ql-card ql-card--hover` | Clickable card |
| `page_container` | `ql-page` | Page wrapper |
| `nav_btn` | `ql-nav-btn` | Navigation button |
| `btn_primary` | `ql-btn-primary` | Primary action |
| `btn_secondary` | `ql-btn-secondary` | Secondary action |
| `btn_danger` | `ql-btn-danger` | Danger action |
| `btn_ghost` | `ql-btn-ghost` | Ghost button |
| `badge` | `ql-badge` | Badge |
| `section_title` | `ql-section-title` | Section heading |
| `section_card` | `ql-card` | Alias for card |

Legacy classes (`modern-card`, `modern-btn-primary`, `gold-btn`, `green-btn`,
`red-btn`, `section-card`, `section-title`) are all still defined in CSS.

## CSS Variables

All variables use the `--ql-` prefix. In inline styles, prefer these:

```
var(--ql-text)            — primary text color
var(--ql-text-secondary)  — secondary text
var(--ql-text-muted)      — muted text
var(--ql-accent)          — primary accent
var(--ql-secondary)       — champagne / amber secondary accent
var(--ql-border)          — standard border
var(--ql-surface)         — surface background
var(--ql-card-bg)         — card background
var(--ql-input-bg)        — input background
var(--ql-success)         — success color
var(--ql-warning)         — warning color
var(--ql-error)           — error color
var(--ql-info)            — info color
var(--ql-inset-bg)        — soft inset panel background
var(--ql-console-bg)      — log / console background
```

Legacy `var(--color-text)`, `var(--card-border)`, etc. are aliased and
still work but should not be used in new code.

## Dark / Light Mode

Dark mode is the default. Toggle with `body.dark-mode` class.
CSS variables auto-switch — no separate dark-mode CSS needed.

```javascript
// Toggle
window.toggleDarkMode();  // defined in THEME_SCRIPT
```

## Convenience Functions

```python
from gui.theme import apply_card, apply_button, apply_input

apply_card(element, hover=True)
apply_button(element, variant='primary')  # primary|secondary|danger|ghost
apply_input(element)
```

## Backward Compatibility

- `apply_theme('green-gold')` and `apply_green_gold_styles()` still exist
  but now call the unified theme
- `MODERN_COLORS` is an alias for `COLORS`
- All old CSS class names are defined as aliases in the stylesheet
