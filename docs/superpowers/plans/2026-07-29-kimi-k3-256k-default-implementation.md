# Kimi K3-256K Default Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `k3-256k` to Kimi and Kimi-Code, make it their shared default, and give it exactly the existing K3 reasoning behavior.

**Architecture:** Define the K3 model family and default model once in `kimi_reasoning.py`, then consume those constants from providers, the parser, and the GUI. Keep the PowerShell literal synchronized through tests because it cannot import Python constants.

**Tech Stack:** Python 3, pytest, NiceGUI configuration code, PowerShell.

## Global Constraints

- Keep `k3` selectable and preserve an explicitly selected model ID.
- Put `k3-256k` first and make it the default for both Kimi providers.
- Both K3 IDs use `reasoning_effort = "low" | "high" | "max"`; the project default remains `low`.
- Reasoning effort remains configuration-only; do not add a CLI override.
- Preserve non-K3 Thinking behavior.
- Do not change API keys, base URLs, or configuration ownership.
- Do not run browser or screenshot tests.
- Preserve all pre-existing uncommitted work; do not commit implementation files that already contain unrelated or earlier K3 changes.

---

### Task 1: Centralize The K3 Model Family

**Files:**
- Modify: `tests/test_provider_v2.py`
- Modify: `module/providers/cloud_vlm/kimi_reasoning.py`
- Modify: `module/providers/cloud_vlm/kimi_code.py`
- Modify: `module/providers/cloud_vlm/kimi_vl.py`

**Interfaces:**
- Produces: `K3_MODEL_IDS: frozenset[str]`
- Produces: `DEFAULT_K3_MODEL_ID: str`
- Preserves: `is_k3_model(model_path: object) -> bool`
- Consumes: `configured_k3_reasoning_effort(config, section) -> str`

- [ ] **Step 1: Write failing model-family and provider tests**

Add tests that require both K3 IDs to match and require `k3-256k` to take the
same provider path as `k3`:

```python
@pytest.mark.parametrize("model_path", ["k3", " K3 ", "k3-256k", " K3-256K "])
def test_k3_model_family_recognizes_both_model_ids(model_path):
    from module.providers.cloud_vlm.kimi_reasoning import is_k3_model

    assert is_k3_model(model_path) is True


@pytest.mark.parametrize("model_path", ["kimi-k2.6", "kimi-for-coding", "", None])
def test_k3_model_family_rejects_non_k3_model_ids(model_path):
    from module.providers.cloud_vlm.kimi_reasoning import is_k3_model

    assert is_k3_model(model_path) is False
```

Parameterize the existing Kimi and Kimi-Code provider reasoning tests with:

```python
@pytest.mark.parametrize("model_path", ["k3", "k3-256k"])
```

Use `model_path` in each provider context and assert the selected model reaches
`attempt_kimi_vl` unchanged:

```python
assert mock_attempt.call_args.kwargs["model_path"] == model_path
assert mock_attempt.call_args.kwargs["reasoning_effort"] == expected_effort
```

Add a default contract assertion:

```python
def test_kimi_code_empty_model_uses_shared_k3_256k_default():
    from module.providers.cloud_vlm.kimi_code import KimiCodeProvider
    from module.providers.cloud_vlm.kimi_reasoning import DEFAULT_K3_MODEL_ID

    assert DEFAULT_K3_MODEL_ID == "k3-256k"
    assert KimiCodeProvider.KIMI_CODE_DEFAULT_MODEL == DEFAULT_K3_MODEL_ID
    assert KimiCodeProvider.normalize_model_path("") == DEFAULT_K3_MODEL_ID
```

Add a Kimi provider fallback test using a `SimpleNamespace` without
`kimi_model_path`; mock `attempt_kimi_vl` and assert:

```python
assert mock_attempt.call_args.kwargs["model_path"] == "k3-256k"
assert mock_attempt.call_args.kwargs["reasoning_effort"] == "low"
```

- [ ] **Step 2: Run tests and verify the new expectations fail**

Run:

```powershell
python -m pytest tests/test_provider_v2.py -k "k3_model_family or k3_reads_reasoning or kimi_provider_reads_k3 or empty_model_uses_shared" -q
```

Expected: failures show that `k3-256k` is not recognized and the default is
still `k3`.

- [ ] **Step 3: Implement the shared model family and provider defaults**

Replace the single model constant in `kimi_reasoning.py` with:

```python
K3_MODEL_IDS = frozenset({"k3", "k3-256k"})
DEFAULT_K3_MODEL_ID = "k3-256k"


def is_k3_model(model_path: object) -> bool:
    return str(model_path or "").strip().lower() in K3_MODEL_IDS
```

Import `DEFAULT_K3_MODEL_ID` in both provider modules. In `kimi_code.py`, keep
the public class constant as an alias:

```python
KIMI_CODE_DEFAULT_MODEL = DEFAULT_K3_MODEL_ID
```

In `kimi_vl.py`, replace the fallback literal:

```python
model_path = getattr(self.ctx.args, "kimi_model_path", DEFAULT_K3_MODEL_ID)
```

- [ ] **Step 4: Run provider tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_provider_v2.py -k "k3 or kimi_code or kimi_provider" -q
```

Expected: all selected tests pass, including existing non-K3 coverage.

---

### Task 2: Synchronize CLI And PowerShell Defaults

**Files:**
- Modify: `tests/test_cloud_concurrency_config.py`
- Modify: `module/captioner.py`
- Modify: `4.captioner.ps1`

**Interfaces:**
- Consumes: `DEFAULT_K3_MODEL_ID`
- Produces: parser defaults `kimi_model_path == kimi_code_model_path == "k3-256k"`
- Produces: PowerShell defaults matching the parser

- [ ] **Step 1: Change tests to require the new defaults**

Update the parser assertions:

```python
assert default_args.kimi_model_path == "k3-256k"
assert default_args.kimi_code_model_path == "k3-256k"
```

Update the PowerShell assertions and guard:

```python
assert '$kimi_model_path = "k3-256k"' in script
assert '$kimi_code_model_path = "k3-256k"' in script
assert '$kimi_code_model_path -ne "k3-256k"' in script
```

- [ ] **Step 2: Run tests and verify they fail on the old defaults**

Run:

```powershell
python -m pytest tests/test_cloud_concurrency_config.py -k "current_kimi_models" -q
```

Expected: parser and PowerShell assertions fail because they still use
`kimi-k2.6` and `k3`.

- [ ] **Step 3: Implement parser and launcher defaults**

Import `DEFAULT_K3_MODEL_ID` in `module/captioner.py` and use it for both Kimi
arguments:

```python
default=DEFAULT_K3_MODEL_ID
```

Set both PowerShell variables to `"k3-256k"` and update only the Kimi-Code
redundant-argument check:

```powershell
if ($kimi_code_model_path -and $kimi_code_model_path -ne "k3-256k") {
```

- [ ] **Step 4: Run the CLI and PowerShell tests**

Run:

```powershell
python -m pytest tests/test_cloud_concurrency_config.py -k "kimi" -q
```

Expected: all selected tests pass and the removed reasoning CLI remains
rejected.

---

### Task 3: Synchronize GUI Models And Reasoning Visibility

**Files:**
- Modify: `tests/test_penguin_dependencies.py`
- Modify: `gui/wizard/step4_caption.py`

**Interfaces:**
- Consumes: `DEFAULT_K3_MODEL_ID`
- Consumes: `is_k3_model(model_path: object) -> bool`
- Produces: both GUI model lists beginning with `["k3-256k", "k3"]`
- Produces: both GUI defaults equal to `"k3-256k"`

- [ ] **Step 1: Change GUI tests to require the new model order and defaults**

Update the model-list expectations:

```python
assert CaptionStep.API_CONFIGS["Kimi"]["models"][:2] == ["k3-256k", "k3"]
assert CaptionStep.API_CONFIGS["Kimi"]["default_model"] == "k3-256k"
assert CaptionStep.API_CONFIGS["Kimi-Code"]["models"][:2] == ["k3-256k", "k3"]
assert CaptionStep.API_CONFIGS["Kimi-Code"]["default_model"] == "k3-256k"
```

Exercise visibility for both IDs:

```python
step._sync_kimi_reasoning_effort_visibility("Kimi", "k3-256k")
step._sync_kimi_reasoning_effort_visibility("Kimi-Code", " K3 ")
assert kimi_container.visible is True
assert kimi_code_container.visible is True
```

Use `k3-256k` in the command-building test and assert:

```python
assert "--kimi_code_model_path=k3-256k" in args
assert not any("reasoning_effort" in arg for arg in args)
```

- [ ] **Step 2: Run GUI unit tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_penguin_dependencies.py -k "kimi" -q
```

Expected: failures identify the old list order, defaults, model normalization,
and K3 visibility behavior.

- [ ] **Step 3: Implement GUI defaults using the shared constant**

Import both shared symbols:

```python
from module.providers.cloud_vlm.kimi_reasoning import (
    DEFAULT_K3_MODEL_ID,
    is_k3_model,
)
```

Use `DEFAULT_K3_MODEL_ID` in `_normalize_kimi_code_model_path`, both GUI
`default_model` values, and the first position of both model lists:

```python
"models": [
    DEFAULT_K3_MODEL_ID,
    "k3",
    ...
],
"default_model": DEFAULT_K3_MODEL_ID,
```

No separate visibility edit is needed because `is_k3_model()` now owns the
family behavior.

- [ ] **Step 4: Run GUI unit tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_penguin_dependencies.py -k "kimi" -q
```

Expected: all selected tests pass without launching a browser.

---

### Task 4: Verify The Integrated Change

**Files:**
- Verify: `module/providers/cloud_vlm/kimi_reasoning.py`
- Verify: `module/providers/cloud_vlm/kimi_code.py`
- Verify: `module/providers/cloud_vlm/kimi_vl.py`
- Verify: `module/captioner.py`
- Verify: `gui/wizard/step4_caption.py`
- Verify: `4.captioner.ps1`
- Verify: `tests/test_provider_v2.py`
- Verify: `tests/test_cloud_concurrency_config.py`
- Verify: `tests/test_penguin_dependencies.py`

**Interfaces:**
- Verifies the complete K3 model-family, default propagation, and GUI contract.

- [ ] **Step 1: Scan for stale Kimi defaults**

Run:

```powershell
rg -n 'KIMI_CODE_DEFAULT_MODEL = "k3"|default="kimi-k2\.6"|default="k3"|\$kimi_model_path = "kimi-k2\.6"|\$kimi_code_model_path = "k3"|default_model": "kimi-k2\.6"|default_model": "k3"' module gui tests 4.captioner.ps1
```

Expected: no stale default definitions remain. Test fixtures that explicitly
exercise the preserved `k3` model may still contain `"k3"`.

- [ ] **Step 2: Run all targeted test files**

Run:

```powershell
python -m pytest tests/test_provider_v2.py tests/test_cloud_concurrency_config.py tests/test_penguin_dependencies.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and diff checks**

Run:

```powershell
python -m compileall -q module/providers/cloud_vlm/kimi_reasoning.py module/providers/cloud_vlm/kimi_code.py module/providers/cloud_vlm/kimi_vl.py module/captioner.py gui/wizard/step4_caption.py
git diff --check
git status --short
```

Expected: compilation and whitespace checks succeed. `git status` may still
show the pre-existing dirty worktree plus the intended files from this change.
