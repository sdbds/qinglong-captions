# Kimi K3-256K Default Model Design

**Status:** Approved

## Context

The project currently exposes `k3` for both the Kimi and Kimi-Code providers,
but only Kimi-Code uses it as the default model. K3 reasoning behavior is
centralized in `kimi_reasoning.py` and currently recognizes only the exact
model ID `k3`.

Kimi now exposes `k3-256k`, a fixed 256K-context K3 variant. It supports the
same `reasoning_effort` values as `k3`: `low`, `high`, and `max`.

## Goals

- Add `k3-256k` to the Kimi and Kimi-Code model selectors.
- Put `k3-256k` first and make it the default model for both providers.
- Keep the existing `k3` model selectable.
- Treat `k3` and `k3-256k` as one K3 model family for reasoning behavior.
- Preserve the project-level K3 reasoning default of `low`.
- Keep CLI, PowerShell, provider fallbacks, GUI defaults, and tests aligned.

## Non-Goals

- Do not alias `k3` to `k3-256k` or alter an explicitly selected model ID.
- Do not remove `k3` or change its request behavior.
- Do not add a reasoning-effort CLI option.
- Do not change API keys, base URLs, or configuration section ownership.
- Do not change Thinking behavior for non-K3 models.

## Shared Model Contract

`module/providers/cloud_vlm/kimi_reasoning.py` owns the K3 model family and its
default model:

```python
K3_MODEL_IDS = frozenset({"k3", "k3-256k"})
DEFAULT_K3_MODEL_ID = "k3-256k"
```

`is_k3_model()` remains the shared predicate. It trims whitespace, compares
case-insensitively, and returns true for either K3 model ID. Keeping the
existing function avoids duplicating model-family checks across providers and
the GUI.

Both IDs use the existing reasoning contract:

```toml
reasoning_effort = "low"  # low | high | max
```

The selected effort remains configuration-only and independent for
`kimi_vl` and `kimi_code`.

## Default Propagation

The shared `DEFAULT_K3_MODEL_ID` constant is the source of truth for Python
runtime defaults:

- `module/captioner.py` uses it for both model CLI defaults and help text.
- `KimiVLProvider` uses it when `kimi_model_path` is absent.
- `KimiCodeProvider` uses it when `kimi_code_model_path` is absent or empty.
- `gui/wizard/step4_caption.py` uses it for both panels' default model and for
  empty Kimi-Code model normalization.

The PowerShell launcher cannot import Python constants, so its two model
defaults and Kimi-Code default-argument suppression check use the literal
`k3-256k`. Tests guard this unavoidable mirror.

Both GUI model lists start with:

```text
k3-256k
k3
```

The remaining provider-specific models retain their current order.

## Provider Behavior

For either Kimi provider:

1. Resolve the selected model without rewriting an explicit ID.
2. If `is_k3_model()` is true, read that provider's existing
   `reasoning_effort` configuration.
3. Send the normalized value as the top-level `reasoning_effort` request
   field.
4. Otherwise preserve the provider's existing non-K3 Thinking behavior.

This makes `k3-256k` behavior identical to `k3` without changing the wire
model name.

## GUI Behavior

The Kimi and Kimi-Code model selectors expose `k3-256k` first and initialize
to it. The Reasoning Effort control is visible for both `k3-256k` and `k3`,
and hidden for every non-K3 model.

Changing between the two K3 IDs does not change the saved effort. The current
provider-specific configuration persistence remains unchanged.

## Testing

Targeted tests cover:

- Shared model-family matching for `k3` and `k3-256k`, including normalized
  case and whitespace.
- Both parser defaults and provider fallbacks resolve to `k3-256k`.
- Both PowerShell defaults use `k3-256k`, and Kimi-Code omits the redundant
  default CLI argument.
- Both GUI lists place `k3-256k` before `k3` and use it as the default.
- Reasoning controls are visible for both K3 IDs.
- Both providers send the configured top-level `reasoning_effort` for
  `k3-256k`.
- Existing `k3` and non-K3 behavior remains covered.

## Reference

- Kimi Code model configuration:
  https://www.kimi.com/code/docs/kimi-code/models.html
