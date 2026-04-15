# Theme System — Style Guide

Unified theme system based on CSS Custom Properties. Dark-first design
inspired by Linear + Supabase.

## Architecture

- **Single theme** — no duplicate green-gold / modern split
- **CSS Custom Properties** — `--ql-*` namespace, auto-switch via `body.dark-mode`
- **Legacy aliases** — old `var(--color-*)`, `var(--card-*)`, `var(--btn-primary-*)` still work
- **No `!important`** — clean cascade, no specificity wars

## Quick Start

```python
from theme import apply_theme, get_classes, COLORS

apply_theme()  # Call once per page in page_base()

# Use CSS classes
with ui.card().classes(get_classes('card')):
    ui.label('Content')

# Use color values
ui.icon("star").style(f"color: {COLORS['primary']};")
```

## Color Palette

| Key | Value | Usage |
|-----|-------|-------|
| `primary` | `#3ecf8e` | Brand accent (emerald) |
| `primary_dark` | `#059669` | Deeper emerald |
| `primary_light` | `#4ade80` | Light highlights |
| `secondary` | `#818cf8` | Indigo accent |
| `success` | `#3ecf8e` | Success state |
| `warning` | `#d29922` | Warning state |
| `error` | `#f85149` | Error/danger |
| `info` | `#58a6ff` | Information |

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
var(--ql-accent)          — brand accent
var(--ql-secondary)       — secondary accent (indigo)
var(--ql-border)          — standard border
var(--ql-surface)         — surface background
var(--ql-card-bg)         — card background
var(--ql-input-bg)        — input background
var(--ql-success)         — success color
var(--ql-warning)         — warning color
var(--ql-error)           — error color
var(--ql-info)            — info color
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
from theme import apply_card, apply_button, apply_input

apply_card(element, hover=True)
apply_button(element, variant='primary')  # primary|secondary|danger|ghost
apply_input(element)
```

## Backward Compatibility

- `apply_theme('green-gold')` and `apply_green_gold_styles()` still exist
  but now call the unified theme
- `MODERN_COLORS` is an alias for `COLORS`
- All old CSS class names are defined as aliases in the stylesheet
