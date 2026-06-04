# Gemma 4 12B Unified Local Adaptation Spec

Date: 2026-06-04

## Review Verdict

This spec is revised after code-review feedback. The correction is important:

- The runtime default is `google/gemma-4-E4B-it` from `config/model.toml`, not the class attribute `Gemma4LocalProvider.default_model_id = "google/gemma-4-E2B-it"`.
- Adding Gemma 4 12B Unified is primarily a data/capability patch, not a provider rewrite.
- Audio ordering, `parse_response()`, thinking control, and dependency lower bounds are real follow-up topics, but they must not ride on the 12B model-list patch.

The implementation should therefore be split:

- **Patch A**: add `google/gemma-4-12B(-it)` support with zero behavior breakage.
- **Patch B**: separately evaluate prompt-order, response parsing, thinking, and dependency changes.

## Verified External Facts

Sources checked on 2026-06-04:

- Hugging Face model: <https://huggingface.co/google/gemma-4-12B>
- Hugging Face model API: <https://huggingface.co/api/models/google/gemma-4-12B>
- Hugging Face config: <https://huggingface.co/google/gemma-4-12B/raw/main/config.json>
- Hugging Face instruction-tuned model API: <https://huggingface.co/api/models/google/gemma-4-12B-it>

Observed facts from the model APIs:

- `google/gemma-4-12B`
  - `pipeline_tag = "any-to-any"`
  - `architecture = "Gemma4UnifiedForConditionalGeneration"`
  - `model_type = "gemma4_unified"`
  - BF16 parameters: `11,959,730,224`
  - used storage: about `23.95 GB`
  - no `chat_template.jinja` sibling
- `google/gemma-4-12B-it`
  - base model: `google/gemma-4-12B`
  - same unified architecture and parameter count
  - includes `chat_template.jinja`
- The model card says audio is supported on E2B, E4B, and 12B; 26B/31B are text-image only.

These facts justify adding both 12B ids, preferring `-it` for app usage, and treating 12B as audio-capable.

## Local Current Facts

Relevant files:

- `module/providers/local_vlm/gemma4_local.py`
- `config/model.toml`
- `config/config.toml`
- `tests/test_gemma4_local.py`
- `tests/test_penguin_dependencies.py`

Current local facts:

- `gemma4_local` already serves image, video, audio ASR, and audio AST routes.
- Class fallback default is `google/gemma-4-E2B-it`, but normal runtime config default is `google/gemma-4-E4B-it`.
- `config/model.toml` lists E2B, E4B, 26B A4B, 31B, and quantized variants; it does not list 12B.
- `_model_supports_audio()` currently allows E2B/E4B, rejects 26B/31B, and allows unknown ids.
- `_chat_template_source_repo()` knows E2B/E4B/26B/31B template fallback repos, but not 12B.
- Existing `tests/test_gemma4_local.py` locks the current direct-runtime audio content order as audio-first.
- Current direct decode path uses `processor.tokenizer.decode(..., skip_special_tokens=True)`.

## First Principles

The real requirement is:

```text
Allow the existing gemma4_local route to use Gemma 4 12B Unified, especially for audio routes.
```

This is a model capability data change. The clean representation is a small set of audio-capable model prefixes, not a new provider and not a broad capability dict.

Required properties:

1. One route stays canonical: `gemma4_local`.
2. Default model does not change.
3. Unknown explicit custom ids keep the current permissive behavior.
4. 26B/31B remain audio-rejected.
5. Patch A must not require changing existing behavior tests.

## Patch A: 12B Data Patch

Patch A is the only work that should be bundled with the 12B adaptation.

### Goals

1. Add `google/gemma-4-12B-it` to `[gemma4_local].model_list`.
2. Add `google/gemma-4-12B` to `[gemma4_local].model_list` as an advanced/base option.
3. Treat `gemma-4-12b` ids as audio-capable.
4. Preserve unknown custom id audio permissiveness.
5. Add 12B chat-template fallback to `google/gemma-4-12B-it`.
6. Do not change direct-runtime prompt ordering.
7. Do not change decode / parsing behavior.
8. Do not change `transformers` dependency lower bound.
9. Do not change the default model.

### Code Shape

Use data, not a capability object:

```python
_AUDIO_CAPABLE_MODEL_PREFIXES = (
    "gemma-4-e2b",
    "gemma-4-e4b",
    "gemma-4-12b",
)

_AUDIO_INCAPABLE_MODEL_PREFIXES = (
    "gemma-4-26b",
    "gemma-4-31b",
)
```

Then keep the existing unknown-id behavior:

```python
def _model_supports_audio(self, model_id: str | None = None) -> bool:
    normalized = self._normalize_model_id(model_id or self.model_id)
    if not normalized:
        return True
    if any(prefix in normalized for prefix in _AUDIO_CAPABLE_MODEL_PREFIXES):
        return True
    if any(prefix in normalized for prefix in _AUDIO_INCAPABLE_MODEL_PREFIXES):
        return False
    return True
```

Template fallback:

```python
if "gemma-4-12b" in normalized:
    return "google/gemma-4-12B-it"
```

Model catalog:

```toml
model_list."Gemma 4 12B Unified it" = { model_id = "google/gemma-4-12B-it", meta = { min_vram_gb = 32, supports_audio = true } }
model_list."Gemma 4 12B Unified base" = { model_id = "google/gemma-4-12B", meta = { min_vram_gb = 32, supports_audio = true, advanced = true } }
```

Rationale:

- `-it` is the right user-facing option for caption / ASR / AST.
- Base remains selectable because the requested URL is the base repo.
- `min_vram_gb = 32` avoids overpromising 24 GB cards based on file size alone.

### Tests

Add or update only 12B-focused tests:

1. `test_gemma4_model_toml_lists_12b_unified_variants`
   - asserts both model ids exist
   - asserts `min_vram_gb == 32`
   - asserts `supports_audio = true`
2. `test_gemma4_12b_supports_audio`
   - prepares audio with `model_id = "google/gemma-4-12B-it"`
   - must not raise `GEMMA4_AUDIO_UNSUPPORTED_MODEL`
3. `test_gemma4_12b_base_chat_template_falls_back_to_it_repo`
   - model id `google/gemma-4-12B`
   - fallback repo must be `google/gemma-4-12B-it`
4. `test_gemma4_unknown_custom_model_id_still_supports_audio`
   - locks current permissive behavior for explicit/custom ids
5. Keep `test_gemma4_prepare_media_rejects_audio_for_non_audio_variants`
   - keep 26B/31B cases
   - do not add 12B to rejection cases

Acceptance:

```powershell
uv run pytest tests/test_gemma4_local.py
```

Patch A should not require changing the existing assertion that audio content is first in `test_gemma4_direct_runtime_normalizes_ast_result`.

## Patch B: Separate Runtime Behavior Review

These items are deliberately not part of Patch A.

### B1. Audio Message Order

Official Gemma 4 guidance says audio should appear after text. Changing this may be correct, but it breaks current tests and changes runtime behavior.

If implemented, the spec and tests must explicitly update:

- `_build_messages()`
- `_build_runtime_messages()` if OpenAI-compatible local runtime is expected to mirror Gemma direct runtime
- `tests/test_gemma4_local.py::test_gemma4_direct_runtime_normalizes_ast_result`

Required test update:

```python
assert captured_messages[1]["content"][0]["type"] == "text"
assert captured_messages[1]["content"][1]["type"] == "audio"
```

This change needs its own review because it can alter outputs even for E2B/E4B.

### B2. `parse_response()` and Decode Fallback

Using `processor.parse_response()` is desirable for Gemma 4 Unified, especially with thinking/channel tokens, but the fallback must not regress current clean output.

Correct fallback rule:

```python
parse_response = getattr(processor, "parse_response", None)
if callable(parse_response):
    response = processor.decode(new_tokens, skip_special_tokens=False)
    response_text = _coerce_gemma4_parse_response(parse_response(response))
else:
    response_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
```

Never fall back to raw `skip_special_tokens=False` text when `parse_response()` is unavailable.

Tests must cover:

- parse_response exists and returns final answer
- parse_response absent keeps current `skip_special_tokens=True` behavior
- parse_response coercion failure falls back to a clean decode, not special-token text

### B3. Thinking Control

`enable_thinking=False` is probably correct for caption / ASR / AST, but it is still a behavior change. It should be reviewed with response parsing because thinking output affects parser cleanliness.

If added:

```toml
[gemma4_local]
thinking = "disabled"  # disabled | enabled
```

Direct runtime should map it to:

```python
chat_template_kwargs["enable_thinking"] = thinking == "enabled"
```

The current `prepare_multimodal_inputs()` helper already passes optional chat-template kwargs and retries without unknown kwargs, so no helper rewrite should be necessary.

### B4. Dependency Lower Bound

Do not change `pyproject.toml` until a resolvable Transformers release is verified to support Gemma 4 Unified.

Current test `tests/test_penguin_dependencies.py::test_pyproject_declares_gemma4_local_extra` expects a `transformers[serving]>=5.5.0` dependency. If the lower bound changes, update that test in the same patch.

Runtime failure should be explicit if the installed version cannot load 12B Unified:

```text
GEMMA4_12B_TRANSFORMERS_UNSUPPORTED: installed transformers does not expose Gemma 4 Unified runtime.
```

## Non-Goals

- Do not create `gemma4_12b_local`.
- Do not change the default model to 12B.
- Do not claim 24 GB VRAM support.
- Do not add auto-segmentation.
- Do not add function calling.
- Do not add visual token budget controls until the exact processor argument is verified.
- Do not add ModelOpt/NVFP4 direct loading.

## Acceptance Criteria

Patch A:

- `config/model.toml` lists both `google/gemma-4-12B-it` and `google/gemma-4-12B`.
- `gemma4_local` audio route accepts 12B.
- 26B/31B audio rejection remains.
- Unknown custom ids remain audio-permissive.
- 12B base can fall back to the `google/gemma-4-12B-it` chat template.
- Existing direct-runtime audio order tests are not changed.
- No dependency lower bound changes.
- No default model change.

Patch B, if pursued later:

- Audio-order tests are updated explicitly.
- `parse_response()` fallback preserves `skip_special_tokens=True` when unavailable.
- Thinking control is covered by direct-runtime tests.
- Dependency lower bound is changed only after package-resolution verification.

## Decision

Proceed with Patch A first.

Hold Patch B for separate review. The 12B adaptation should be a small data patch with a narrow test surface, not a bundled runtime rewrite.
