# OvisOCR2 GUI Config Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render OvisOCR2 runtime and visual-region enums as dropdowns while displaying the official default prompt in the GUI without copying it into TOML.

**Architecture:** Move the official prompt into a lightweight contract module imported by both the Provider and `ModelConfigPanel`. Keep enum and display-default behavior section-scoped in the existing panel metadata, and normalize an unchanged default prompt back to the existing empty-string sentinel at GUI event boundaries.

**Tech Stack:** Python 3.10+, NiceGUI, tomlkit, pytest, Ruff

## Global Constraints

- Persisted OvisOCR2 enum values are exactly `direct`/`openai` and `crop`/`drop`.
- `config/model.toml` and `config/config.toml` keep `prompt = ""` as the default sentinel.
- The official prompt has one source and must not be duplicated in GUI or TOML configuration.
- Prompt equivalence ignores only leading/trailing whitespace and CRLF/LF differences; internal whitespace remains significant.
- Loading the GUI config panel must not import the OvisOCR2 inference implementation or its Torch/Transformers chain.
- Preserve existing uncommitted changes in `config/general.toml`, `tests/test_runtime_backends.py`, and `utils/transformer_loader.py`.

---

### Task 1: Extract The Lightweight OvisOCR2 Prompt Contract

**Files:**
- Create: `module/providers/ocr/ovis_ocr2_contract.py`
- Modify: `module/providers/ocr/ovis_ocr2.py:24-28,270-278`
- Test: `tests/test_ovis_ocr2_provider.py`

**Interfaces:**
- Produces: `OVIS_OCR2_DEFAULT_PROMPT: str`
- Consumed by: `OvisOCR2Provider.default_prompt` and the GUI display-default metadata in Task 3.

- [ ] **Step 1: Write the failing shared-contract test**

```python
from module.providers.ocr.ovis_ocr2_contract import OVIS_OCR2_DEFAULT_PROMPT


def test_provider_uses_shared_official_prompt_contract():
    assert OvisOCR2Provider.default_prompt == OVIS_OCR2_DEFAULT_PROMPT
    assert OVIS_OCR2_DEFAULT_PROMPT.startswith("\nExtract all readable content")
    assert '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />' in OVIS_OCR2_DEFAULT_PROMPT
```

- [ ] **Step 2: Run the test and confirm the contract module is missing**

Run: `pytest tests/test_ovis_ocr2_provider.py::test_provider_uses_shared_official_prompt_contract -q`

Expected: FAIL during collection with `ModuleNotFoundError: module.providers.ocr.ovis_ocr2_contract`.

- [ ] **Step 3: Add the contract and make the Provider consume it**

```python
# module/providers/ocr/ovis_ocr2_contract.py
OVIS_OCR2_DEFAULT_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order and output the result as a single "
    "Markdown document. For charts or images, represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."
)
```

Move the existing complete official string verbatim from `OvisOCR2Provider.default_prompt` into this constant, import it in `ovis_ocr2.py`, and set:

```python
default_prompt = OVIS_OCR2_DEFAULT_PROMPT
```

- [ ] **Step 4: Run the focused Provider test**

Run: `pytest tests/test_ovis_ocr2_provider.py::test_provider_uses_shared_official_prompt_contract -q`

Expected: PASS.

### Task 2: Render OvisOCR2 Enum Fields As Fixed Selects

**Files:**
- Modify: `gui/components/model_config_panel.py:29-33`
- Modify: `tests/test_model_config_panel.py:139-162`

**Interfaces:**
- Consumes: existing `ModelConfigPanel._enum_options_for_field(key: str) -> tuple[str, ...]`.
- Produces: section-scoped enum entries for OvisOCR2 without changing other sections.

- [ ] **Step 1: Add failing dropdown tests**

```python
@pytest.mark.parametrize(
    ("key", "value", "expected_options"),
    [
        ("runtime_backend", "direct", {"direct": "direct", "openai": "openai"}),
        ("visual_region_mode", "crop", {"crop": "crop", "drop": "drop"}),
    ],
)
def test_model_config_panel_renders_ovis_ocr2_enums_as_fixed_selects(
    monkeypatch, key, value, expected_options
):
    panel_module = _load_model_config_panel_module(f"test_ovis_ocr2_{key}_select")
    fake_select = _FakeSelect()
    monkeypatch.setattr(panel_module, "ui", _FakeUI([fake_select]))
    panel = panel_module.ModelConfigPanel(parent=SimpleNamespace(clear=lambda: None))
    panel._section_name = "ovis_ocr2"
    panel._current_route = "ovis_ocr2"
    parent_dict = {key: value}

    panel._render_field(key, value, parent_dict)

    assert fake_select.options == expected_options
    assert fake_select.value == value
```

- [ ] **Step 2: Run the dropdown tests and verify they fail as string inputs**

Run: `pytest tests/test_model_config_panel.py -k ovis_ocr2_enums -q`

Expected: FAIL because `_FakeUI.input()` is used and no fixed select is created.

- [ ] **Step 3: Add only the section-scoped enum entries**

```python
_ENUM_FIELD_OPTIONS = {
    ("paddle_ocr", "model_tier"): ("tiny", "small", "medium"),
    ("ovis_ocr2", "runtime_backend"): ("direct", "openai"),
    ("ovis_ocr2", "visual_region_mode"): ("crop", "drop"),
}
```

- [ ] **Step 4: Run the dropdown and existing panel tests**

Run: `pytest tests/test_model_config_panel.py -q`

Expected: all tests PASS.

### Task 3: Display The Official Prompt And Preserve The Empty Sentinel

**Files:**
- Modify: `gui/components/model_config_panel.py:15-33,123-133,312-337`
- Modify: `tests/test_model_config_panel.py:43-110`
- Test: `tests/test_model_config_panel.py`

**Interfaces:**
- Consumes: `OVIS_OCR2_DEFAULT_PROMPT` from Task 1.
- Produces: `_display_value_for_field(key: str, value: Any) -> Any` and `_stored_value_for_field(key: str, value: Any) -> Any`.

- [ ] **Step 1: Extend the fake UI with textarea support**

Reuse `_FakeSelect` as the chainable fake control and add:

```python
self.created_textareas = []

def textarea(self, *_args, **kwargs):
    fake_textarea = self._fake_controls.pop(0)
    fake_textarea.value = kwargs.get("value")
    self.created_textareas.append(fake_textarea)
    return fake_textarea
```

- [ ] **Step 2: Add failing prompt display and normalization tests**

```python
def test_model_config_panel_displays_ovis_default_prompt_without_persisting_it(monkeypatch):
    sys.modules.pop("module.providers.ocr.ovis_ocr2", None)
    panel_module = _load_model_config_panel_module("test_ovis_ocr2_default_prompt")
    assert "module.providers.ocr.ovis_ocr2" not in sys.modules
    fake_textarea = _FakeSelect()
    monkeypatch.setattr(panel_module, "ui", _FakeUI([fake_textarea]))
    panel = panel_module.ModelConfigPanel(parent=SimpleNamespace(clear=lambda: None))
    panel._section_name = "ovis_ocr2"
    panel._current_route = "ovis_ocr2"
    parent_dict = {"prompt": ""}

    panel._render_field("prompt", "", parent_dict)

    assert fake_textarea.value == OVIS_OCR2_DEFAULT_PROMPT
    assert parent_dict["prompt"] == ""
    fake_textarea.on_value_change_handler(
        SimpleNamespace(value=f"  {OVIS_OCR2_DEFAULT_PROMPT.replace(chr(10), chr(13) + chr(10))}  ")
    )
    assert parent_dict["prompt"] == ""
    fake_textarea.on_value_change_handler(SimpleNamespace(value="custom prompt"))
    assert parent_dict["prompt"] == "custom prompt"
```

- [ ] **Step 3: Run the prompt tests and confirm blank input is still rendered as a text input**

Run: `pytest tests/test_model_config_panel.py -k ovis_default_prompt -q`

Expected: FAIL because the panel displays `""` through `ui.input()` and has no default-value normalization.

- [ ] **Step 4: Add section-scoped display defaults and normalization**

```python
from module.providers.ocr.ovis_ocr2_contract import OVIS_OCR2_DEFAULT_PROMPT

_FIELD_DISPLAY_DEFAULTS = {
    ("ovis_ocr2", "prompt"): OVIS_OCR2_DEFAULT_PROMPT,
}

def _display_value_for_field(self, key: str, value: Any) -> Any:
    default = self._display_default_for_field(key)
    if default is not None and isinstance(value, str) and not value.strip():
        return default
    return value

def _stored_value_for_field(self, key: str, value: Any) -> Any:
    default = self._display_default_for_field(key)
    if default is None or not isinstance(value, str):
        return value
    if not value.strip():
        return ""
    normalize = lambda text: text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "" if normalize(value) == normalize(default) else value
```

Use the resolved display value when choosing the string widget and setting its value. Route both textarea and input change handlers through `_stored_value_for_field`.

- [ ] **Step 5: Run all model config panel and OvisOCR2 tests**

Run: `pytest tests/test_model_config_panel.py tests/test_ovis_ocr2_provider.py tests/test_ovis_ocr2_integration.py -q`

Expected: all tests PASS.

### Task 4: Verify The Integrated Change

**Files:**
- Verify only; do not edit unrelated dirty files.

**Interfaces:**
- Consumes: all behavior from Tasks 1-3.
- Produces: verification evidence for the complete GUI fix.

- [ ] **Step 1: Run the relevant regression suite**

Run: `pytest -q tests/test_model_config_panel.py tests/test_ovis_ocr2_provider.py tests/test_ovis_ocr2_integration.py tests/test_provider_catalog.py tests/test_provider_routes.py`

Expected: all tests PASS.

- [ ] **Step 2: Run Ruff on changed Python files**

Run: `ruff check gui/components/model_config_panel.py module/providers/ocr/ovis_ocr2.py module/providers/ocr/ovis_ocr2_contract.py tests/test_model_config_panel.py tests/test_ovis_ocr2_provider.py`

Expected: `All checks passed!`

- [ ] **Step 3: Check whitespace and ownership boundaries**

Run: `git diff --check && git status --short`

Expected: `git diff --check` exits 0; existing user modifications remain unstaged and unchanged.
