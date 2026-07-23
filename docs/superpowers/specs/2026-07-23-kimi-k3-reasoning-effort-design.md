# Kimi K3 Reasoning Effort Design

**Status:** Approved

## Context

The current K3 integration encodes both the request field and its value inside
`kimi_code_thinking`, for example `thinking.effort:max` or
`reasoning_effort:max`. It also defaults to `max` and transports the selection
through a CLI option.

K3 instead needs one top-level request field:

```json
{
  "reasoning_effort": "low"
}
```

The accepted values are `low`, `high`, and `max`. This project will use `low`
as its default, even though the Kimi service default is currently documented as
`high`.

## Goals

- Configure K3 reasoning effort separately for the Kimi and Kimi-Code
  providers.
- Use `reasoning_effort` as the configuration key and wire field.
- Accept only `low`, `high`, and `max`, with `low` as the project default.
- Make configuration files the only source of reasoning effort.
- Keep the caption GUI synchronized with the configuration files.
- Preserve non-K3 Thinking behavior.

## Non-Goals

- Do not retain or migrate `kimi_code_thinking`, `thinking_mode`,
  `thinking.effort:*`, or the associated CLI option.
- Do not add a CLI override for either provider.
- Do not send `reasoning_effort` to non-K3 models.
- Do not change API keys, base URLs, or model-selection transport.

## Configuration Contract

The split runtime configuration is canonical:

```toml
[kimi_vl]
thinking = "enabled"
reasoning_effort = "low"  # low | high | max

[kimi_code]
reasoning_effort = "low"  # low | high | max
```

`config/model.toml` is read by the current split configuration loader.
`config/config.toml` remains the legacy single-file mirror and must contain the
same values. The obsolete `kimi_code.thinking_mode` key is removed from both.

Missing values resolve to `low`. A present value outside `low`, `high`, and
`max` is a configuration error and must fail before an API request is sent.
This avoids silently turning a typo into a different cost or quality level.

## Provider Behavior

A shared K3 effort normalizer owns the supported values, the default, and
validation. Model matching is case-insensitive after trimming and applies only
to the exact model ID `k3`.

For the Kimi provider:

1. Read the selected model from the existing runtime argument.
2. When the model is `k3`, read `kimi_vl.reasoning_effort`.
3. Pass the normalized value to the shared Kimi request function.
4. For all other models, keep using `kimi_vl.thinking` and omit
   `reasoning_effort`.

For the Kimi-Code provider:

1. Normalize the existing model aliases as before.
2. When the normalized model is `k3`, read
   `kimi_code.reasoning_effort`.
3. Pass the normalized value to the shared Kimi request function.
4. For Kimi for Coding models, retain the existing enabled Thinking request
   and omit `reasoning_effort`.

The shared request function accepts one optional `reasoning_effort` value. When
present, it emits:

```python
extra_body = {"reasoning_effort": reasoning_effort}
```

The obsolete `thinking_effort` request path is removed.

## CLI And Script Cleanup

`module/captioner.py` removes `--kimi_code_thinking`. The shipped PowerShell
launcher removes `$kimi_code_thinking` and no longer appends that option.

Model IDs and credentials continue through their existing CLI arguments. Only
K3 reasoning effort is configuration-only.

## GUI Behavior

The Kimi and Kimi-Code API panels each own an independent Reasoning Effort
select with `low`, `high`, and `max` options.

- Kimi includes `k3` in its selectable model list.
- The control is visible only when that panel's selected model is exactly
  `k3`.
- Kimi initializes from `kimi_vl.reasoning_effort`.
- Kimi-Code initializes from `kimi_code.reasoning_effort`.
- Missing configuration initializes to `low`.
- Changing a select validates the value and writes it to both
  `config/model.toml` and `config/config.toml` with format-preserving TOML
  updates.
- A write failure produces a GUI error notification and does not introduce a
  CLI fallback.
- Caption command construction never includes a reasoning-effort argument.

The two controls remain independent because the providers use different
credentials and endpoints and may need different cost/quality settings.

## Data Flow

```text
GUI selection
    -> model.toml + legacy config.toml
    -> caption process loads split runtime configuration
    -> active K3 provider reads its own section
    -> shared request builder emits top-level reasoning_effort
    -> Kimi endpoint
```

Non-GUI runs follow the same flow beginning at the configuration files.

## Testing

Targeted tests will cover:

- Both shipped configuration files declare `reasoning_effort = "low"`.
- The parser no longer accepts `--kimi_code_thinking`.
- The PowerShell launcher contains no K3 Thinking CLI option.
- `low`, `high`, and `max` produce the matching top-level wire field.
- Missing configuration defaults to `low`.
- Invalid configuration fails before the request.
- Kimi and Kimi-Code read their own configuration sections.
- Non-K3 models omit `reasoning_effort` and retain existing Thinking behavior.
- GUI model lists, conditional visibility, initial values, persistence, and
  command construction follow the new contract.
- Existing provider, configuration, and GUI tests remain green.

## Reference

- Kimi Code model configuration:
  https://www.kimi.com/code/docs/kimi-code/models.html
