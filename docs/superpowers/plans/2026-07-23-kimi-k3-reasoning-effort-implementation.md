# Kimi K3 Reasoning Effort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the K3 Thinking-mode override with config-only, top-level
`reasoning_effort` support for both Kimi providers and synchronize the GUI.

**Architecture:** A small shared provider module validates the exact K3 model
ID and the three supported effort values. Runtime providers read their own
config sections, while a focused GUI helper performs format-preserving reads
and writes of the canonical split config and legacy mirror.

**Tech Stack:** Python 3.10+, pytest, OpenAI-compatible client, NiceGUI,
tomlkit, PowerShell.

## Global Constraints

- The only supported values are `low`, `high`, and `max`.
- The project default is `low`.
- The wire field is top-level `reasoning_effort`.
- Reasoning effort comes only from configuration files, never CLI arguments.
- `config/model.toml` is canonical; `config/config.toml` is kept as a legacy
  mirror.
- Existing non-K3 Thinking behavior remains unchanged.
- Do not retain or migrate obsolete K3 Thinking keys or values.
- Preserve all pre-existing unrelated working-tree edits. Do not stage or
  commit an overlapping file if doing so would capture unrelated changes.

---

### Task 1: Configuration Contract And CLI Removal

**Files:**
- Modify: `tests/test_cloud_concurrency_config.py`
- Modify: `config/model.toml`
- Modify: `config/config.toml`
- Modify: `module/captioner.py`
- Modify: `4.captioner.ps1`

**Interfaces:**
- Consumes: Existing split runtime configuration loader.
- Produces: `kimi_vl.reasoning_effort` and `kimi_code.reasoning_effort`
  configuration values; no K3 reasoning CLI option.

- [x] **Step 1: Write failing configuration and CLI tests**

```python
def test_shipped_kimi_configs_default_k3_reasoning_effort_to_low():
    for path in (ROOT / "config/model.toml", ROOT / "config/config.toml"):
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        assert config["kimi_vl"]["reasoning_effort"] == "low"
        assert config["kimi_code"]["reasoning_effort"] == "low"
        assert "thinking_mode" not in config["kimi_code"]


def test_captioner_parser_rejects_removed_kimi_thinking_option():
    with pytest.raises(SystemExit):
        setup_parser().parse_args(["dataset", "--kimi_code_thinking=max"])
```

Also assert that `4.captioner.ps1` contains neither
`$kimi_code_thinking` nor `--kimi_code_thinking`.

- [x] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_cloud_concurrency_config.py -q
```

Expected: failures show the old `thinking_mode`, parser option, and
PowerShell variable are still present.

- [x] **Step 3: Implement the minimal config and CLI cleanup**

Use this contract in both TOML files:

```toml
[kimi_vl]
thinking = "enabled"
reasoning_effort = "low"  # low | high | max

[kimi_code]
reasoning_effort = "low"  # low | high | max
```

Delete the `--kimi_code_thinking` parser definition and the matching
PowerShell variable/argument forwarding.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_cloud_concurrency_config.py -q
```

Expected: all tests pass.

---

### Task 2: Provider Validation And Wire Payload

**Files:**
- Create: `module/providers/cloud_vlm/kimi_reasoning.py`
- Modify: `module/providers/cloud_vlm/kimi_vl.py`
- Modify: `module/providers/cloud_vlm/kimi_code.py`
- Modify: `tests/test_provider_v2.py`

**Interfaces:**
- Produces:
  - `is_k3_model(model_path: object) -> bool`
  - `normalize_k3_reasoning_effort(value: object, default: str = "low") -> str`
  - `configured_k3_reasoning_effort(config: Mapping[str, Any], section: str) -> str`
- The shared request function accepts `reasoning_effort: str = ""`.

- [x] **Step 1: Write failing normalizer and provider tests**

```python
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_k3_reasoning_effort_is_sent_as_top_level_field(effort):
    # Build the existing fake OpenAI stream, call attempt_kimi_vl, then:
    assert create.call_args.kwargs["extra_body"] == {
        "reasoning_effort": effort,
    }


def test_invalid_k3_reasoning_effort_fails_before_request():
    mock_client = MagicMock()
    with pytest.raises(ValueError, match="Unsupported K3 reasoning effort"):
        attempt_kimi_vl(
            client=mock_client,
            model_path="k3",
            messages=[],
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            uri="/fake.jpg",
            reasoning_effort="medium",
        )
    mock_client.chat.completions.create.assert_not_called()
```

Add provider tests showing that Kimi reads `kimi_vl.reasoning_effort`,
Kimi-Code reads `kimi_code.reasoning_effort`, a missing key yields `low`,
and non-K3 models omit the field.

- [x] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_provider_v2.py -k "kimi and (reasoning or k3)" -q
```

Expected: the old `thinking_effort` path and `thinking_mode` mapping violate
the new assertions.

- [x] **Step 3: Implement the shared normalizer**

```python
K3_MODEL_ID = "k3"
DEFAULT_K3_REASONING_EFFORT = "low"
SUPPORTED_K3_REASONING_EFFORTS = frozenset({"low", "high", "max"})


def is_k3_model(model_path: object) -> bool:
    return str(model_path or "").strip().lower() == K3_MODEL_ID


def normalize_k3_reasoning_effort(value: object, default: str = DEFAULT_K3_REASONING_EFFORT) -> str:
    effort = str(value or "").strip().lower() or default
    if effort not in SUPPORTED_K3_REASONING_EFFORTS:
        expected = ", ".join(sorted(SUPPORTED_K3_REASONING_EFFORTS))
        raise ValueError(
            f"Unsupported K3 reasoning effort: {value}. Expected one of: {expected}"
        )
    return effort
```

`configured_k3_reasoning_effort` reads `config[section]`, defaults a missing
value to `low`, and delegates validation to the normalizer.

- [x] **Step 4: Replace provider branching**

For exact K3 models, pass only the section-specific normalized
`reasoning_effort`. Remove `thinking_effort`, the encoded mode constants,
and all CLI lookups. For non-K3 Kimi-Code models, pass
`thinking="enabled"`. In `attempt_kimi_vl`, validate before calling the
client and emit:

```python
extra_body = {"reasoning_effort": normalized_effort}
```

- [x] **Step 5: Run the focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_provider_v2.py -k "kimi" -q
```

Expected: all selected tests pass.

---

### Task 3: GUI Configuration Persistence

**Files:**
- Create: `gui/utils/kimi_reasoning_config.py`
- Create: `tests/test_kimi_reasoning_config.py`

**Interfaces:**
- Consumes: `normalize_k3_reasoning_effort`.
- Produces:
  - `load_kimi_reasoning_effort(section: str, *, config_path: Path = MODEL_CONFIG_PATH) -> str`
  - `save_kimi_reasoning_effort(section: str, value: object, *, model_config_path: Path = MODEL_CONFIG_PATH, legacy_config_path: Path = LEGACY_CONFIG_PATH) -> str`

- [x] **Step 1: Write failing persistence tests**

Use temporary TOML files containing `[kimi_vl]` and `[kimi_code]`.
Assert that load defaults a missing key to `low`, save changes only the
requested section in both files, comments/unrelated sections survive, and an
invalid value changes neither file.

- [x] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_kimi_reasoning_config.py -q
```

Expected: import failure because the helper does not exist.

- [x] **Step 3: Implement format-preserving persistence**

Parse both target files with `tomlkit` before writing either one. Normalize
the value first, require the requested section to exist, update only
`reasoning_effort`, then serialize both documents. Return the normalized
value. Load from `config/model.toml` and default a missing key to `low`.

- [x] **Step 4: Run the tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_kimi_reasoning_config.py -q
```

Expected: all tests pass.

---

### Task 4: Caption GUI Synchronization

**Files:**
- Modify: `gui/wizard/step4_caption.py`
- Modify: `gui/utils/i18n.py`
- Modify: `tests/test_penguin_dependencies.py`
- Modify: `tests/test_gui_i18n.py`

**Interfaces:**
- Consumes: GUI persistence functions from Task 3.
- Produces: independent Kimi and Kimi-Code effort selects and no effort CLI
  argument.

- [x] **Step 1: Replace old GUI tests with failing new-contract tests**

Assert:

```python
assert CaptionStep.KIMI_REASONING_EFFORT_OPTIONS == {
    "low": "low",
    "high": "high",
    "max": "max",
}
assert CaptionStep().config["kimi_reasoning_effort"] == "low"
assert CaptionStep().config["kimi_code_reasoning_effort"] == "low"
```

Add `k3` to the Kimi model-list expectation. Test visibility handlers with
fake containers, persistence callbacks with a patched save helper, and
`_build_caption_args` to ensure no argument contains
`kimi_code_thinking` or `reasoning_effort`.

- [x] **Step 2: Run the GUI tests and verify RED**

Run:

```powershell
python -m pytest tests/test_penguin_dependencies.py tests/test_gui_i18n.py -k "kimi or i18n" -q
```

Expected: failures reference old Thinking options and CLI construction.

- [x] **Step 3: Implement the GUI controls**

Add `k3` to Kimi models. Replace the old Kimi-Code control with one effort
select per provider. Each select initializes from its provider section,
persists through `save_kimi_reasoning_effort`, and lives in a container whose
visibility is:

```python
container.set_visibility(is_k3_model(model_path))
```

On save failure, call `ui.notify` with the localized `save_failed` label.
Remove all effort handling from `_build_caption_args`.

- [x] **Step 4: Update translations**

Replace `kimi_code_thinking` with `kimi_reasoning_effort` in English,
Simplified Chinese, Japanese, and Korean. Use labels equivalent to
"Reasoning effort" rather than "Thinking mode".

- [x] **Step 5: Run focused and full verification**

Run:

```powershell
python -m pytest tests/test_cloud_concurrency_config.py tests/test_provider_v2.py tests/test_kimi_reasoning_config.py tests/test_penguin_dependencies.py tests/test_gui_i18n.py -q
python -m compileall module/providers/cloud_vlm gui config
```

Expected: all tests pass and compileall reports no errors.

- [x] **Step 6: Visually inspect the GUI**

Start the existing GUI, open both API panels, and verify at desktop and narrow
viewport widths that:

- Kimi hides the control for its default non-K3 model and shows it for `k3`.
- Kimi-Code shows the control for its default `k3` model.
- All three options fit without overlap.
- Switching away from K3 hides the control without shifting unrelated fields.
