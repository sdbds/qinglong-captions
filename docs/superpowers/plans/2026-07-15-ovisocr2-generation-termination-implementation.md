# OvisOCR2 Direct Generation Termination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop OvisOCR2 Direct inference when generation enters a stable periodic suffix, normalize only the verified trigger tail, and preserve the shared Provider pipeline and vLLM behavior.

**Architecture:** Refactor the model-card cleanup around one pure repeated-tail matcher returning immutable metadata. Add a lazy, duck-typed Transformers stopping criterion that checks bounded generated-token tails, then revalidate and normalize its recorded match after final decoding. The Provider retains the official shared cleanup for both backends.

**Tech Stack:** Python 3.11, PyTorch, Transformers 5.x generation API, pytest, Ruff.

## Global Constraints

- Work directly on `codex/ovisocr2`; the user explicitly rejected a worktree.
- Preserve unrelated dirty files and stage only explicitly named OvisOCR2 files.
- Keep `max_new_tokens=16384`, `do_sample=False`, and both message contracts.
- Apply generation stopping only to Direct; do not change `_OpenAIPageInferencer` or `OpenAIChatRuntime`.
- Do not add penalties, n-gram blocking, wall-clock stopping, configuration keys, or dependencies.
- Keep Transformers and Torch imports lazy so catalog import works without the Ovis extra.
- Constants are fixed: begin at 128 new tokens, check every 32, decode at most 768 new tokens, periods 1-200 characters, at least 8 repeats and 200 repeated characters.
- Fail fast for batches other than size 1.
- Do not create `uv.lock`; the repository has no tracked lock and this fix changes no dependencies.
- The current `.venv` lacks pytest. Install only `pytest` and `ruff==0.12.2`; do not run `uv sync --group test`, which would replace the working Torch 2.13 stack with the test group's Torch 2.11 pin.

## File Structure

- Modify: `module/providers/ocr/ovis_ocr2.py` - match data, pure matcher, bounded criterion, Direct integration, and logging.
- Modify: `tests/test_ovis_ocr2_provider.py` - algorithm boundaries, stopping state, Direct integration, and vLLM regression.
- Verify: `module/providers/ocr/ovis_ocr2_contract.py` - the single prompt contract remains unchanged.
- Reference: `docs/superpowers/specs/2026-07-14-ovisocr2-design.md`.

---

### Task 0: Prepare the Existing Runtime for TDD

**Files:**
- Modify: none

**Interfaces:**
- Produces: pytest and Ruff commands available through the existing `.venv` without replacing Torch or Transformers.

- [ ] **Step 1: Install only the missing test tools**

Run:

```powershell
uv pip install --python .\.venv\Scripts\python.exe pytest ruff==0.12.2
```

Expected: pytest and Ruff install without changing the installed Torch 2.13 or Transformers 5.13 packages.

- [ ] **Step 2: Verify the runtime versions**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import pytest, torch, transformers; print(pytest.__version__, torch.__version__, transformers.__version__)"
.\.venv\Scripts\python.exe -m ruff --version
```

Expected: pytest is importable, Ruff is 0.12.2, Torch remains `2.13.0+cu130`, and Transformers remains `5.13.1`.

---

### Task 1: Establish One Repeated-Tail Data Contract

**Files:**
- Modify: `module/providers/ocr/ovis_ocr2.py:1-63`
- Test: `tests/test_ovis_ocr2_provider.py:73-99`

**Interfaces:**
- Produces: `_RepeatedTailMatch(period_len, matched_chars, repeat_times, trailing_chars, suffix_fingerprint)`.
- Produces: `_find_repeated_tail(..., expected_period_len=None) -> Optional[_RepeatedTailMatch]`.
- Produces: `_collapse_repeated_tail(text, match) -> str`.
- Preserves: `_clean_truncated_repeats(...) -> str` and all model-card defaults.

- [ ] **Step 1: Write failing shortest-period and partial-fragment tests**

Add `from module.providers.ocr import ovis_ocr2 as ovis_module`, then add:

```python
def test_find_repeated_tail_selects_shortest_qualifying_period():
    match = ovis_module._find_repeated_tail(
        "header#" + "ab" * 100,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )

    assert match is not None
    assert match.period_len == 2
    assert match.matched_chars == 200
    assert match.repeat_times == 100
    assert match.trailing_chars == 0


def test_collapse_repeated_tail_preserves_partial_period():
    text = "header#" + "abc" * 70 + "ab"
    match = ovis_module._find_repeated_tail(
        text,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )

    assert match is not None
    assert match.period_len == 3
    assert match.trailing_chars == 2
    assert ovis_module._collapse_repeated_tail(text, match) == "header#abcab"


def test_early_match_requires_both_repeat_and_character_thresholds():
    seven_long_units = "".join(chr(0x400 + index) for index in range(30)) * 7
    eight_units = "".join(chr(0x500 + index) for index in range(25)) * 8

    assert ovis_module._find_repeated_tail(
        "header#" + seven_long_units,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    ) is None
    assert ovis_module._find_repeated_tail(
        "header#" + eight_units,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    ) is not None


def test_early_match_rejects_period_over_200_characters():
    boundary_unit = "".join(chr(0x600 + index) for index in range(200))
    oversized_unit = "".join(chr(0x800 + index) for index in range(201))

    boundary_match = ovis_module._find_repeated_tail(
        "header#" + boundary_unit * 8,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )
    assert boundary_match is not None
    assert boundary_match.period_len == 200
    assert ovis_module._find_repeated_tail(
        "header#" + oversized_unit * 8,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    ) is None
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ovis_ocr2_provider.py -k "find_repeated_tail or collapse_repeated_tail" -q
```

Expected: FAIL because both private functions are absent.

- [ ] **Step 3: Implement the match record and pure matcher**

Change the import to `from dataclasses import dataclass, replace`. Add fixed early-stop constants and:

```python
_EARLY_REPEAT_START_TOKENS = 128
_EARLY_REPEAT_CHECK_INTERVAL = 32
_EARLY_REPEAT_TAIL_TOKENS = 768
_EARLY_REPEAT_MIN_CHARS = 200
_EARLY_REPEAT_MIN_TIMES = 8
_REPEAT_FINGERPRINT_MIN_CHARS = 64


@dataclass(frozen=True)
class _RepeatedTailMatch:
    period_len: int
    matched_chars: int
    repeat_times: int
    trailing_chars: int
    suffix_fingerprint: str


def _find_repeated_tail(
    text: str,
    *,
    min_text_len: int,
    max_period: int,
    min_period: int,
    min_repeat_chars: int,
    min_repeat_times: int,
    expected_period_len: Optional[int] = None,
) -> Optional[_RepeatedTailMatch]:
    n = len(text)
    if n < min_text_len or n < 2:
        return None

    lower_period = max(1, min_period)
    upper_period = min(max_period, n - 1)
    periods = (
        (expected_period_len,)
        if expected_period_len is not None
        else range(lower_period, upper_period + 1)
    )
    for period_len in periods:
        if period_len < lower_period or period_len > upper_period:
            continue
        if text[n - 1] != text[n - 1 - period_len]:
            continue

        match_len = 1
        index = n - 2
        while index >= period_len and text[index] == text[index - period_len]:
            match_len += 1
            index -= 1

        matched_chars = match_len + period_len
        repeat_times = matched_chars // period_len
        trailing_chars = matched_chars % period_len
        if repeat_times < min_repeat_times or matched_chars < min_repeat_chars:
            continue

        fingerprint_len = min(
            matched_chars,
            max(_REPEAT_FINGERPRINT_MIN_CHARS, period_len * 2),
        )
        return _RepeatedTailMatch(
            period_len=period_len,
            matched_chars=matched_chars,
            repeat_times=repeat_times,
            trailing_chars=trailing_chars,
            suffix_fingerprint=text[-fingerprint_len:],
        )
    return None


def _collapse_repeated_tail(text: str, match: _RepeatedTailMatch) -> str:
    suffix_start = len(text) - match.matched_chars + match.period_len
    trailing = text[-match.trailing_chars :] if match.trailing_chars else ""
    return text[:suffix_start] + trailing
```

Refactor `_clean_truncated_repeats()` to call this matcher with its existing arguments and collapse only a returned match.

- [ ] **Step 4: Verify GREEN and official behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ovis_ocr2_provider.py -k "repeated_tail or clean_truncated_repeats" -q
```

Expected: new tests and existing 8000/100/5 boundary tests PASS.

---

### Task 2: Add a Bounded Direct Stopping State Machine

**Files:**
- Modify: `module/providers/ocr/ovis_ocr2.py`
- Test: `tests/test_ovis_ocr2_provider.py`

**Interfaces:**
- Consumes: `_find_repeated_tail()` and `_RepeatedTailMatch`.
- Produces: `_RepeatedTailStoppingCriteria(processor, prompt_length)` with `triggered_match` and `triggered_at_tokens` state.
- Produces: `_normalize_triggered_repeat(text, match) -> Optional[str]`; `None` means preserve the original text.

- [ ] **Step 1: Write failing schedule, bound, device, and batch tests**

Add:

```python
class _CharacterTokenProcessor:
    def __init__(self):
        self.decoded_widths = []

    def batch_decode(self, token_ids, **kwargs):
        values = token_ids[0].tolist()
        self.decoded_widths.append(len(values))
        return ["".join(chr(value) for value in values)]


def test_repeat_stopping_excludes_prompt_and_checks_on_schedule():
    processor = _CharacterTokenProcessor()
    prompt = torch.full((1, 250), ord("1"), dtype=torch.long)
    criterion = ovis_module._RepeatedTailStoppingCriteria(processor, prompt_length=250)

    ids = torch.cat([prompt, torch.full((1, 127), ord("1"))], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == []

    ids = torch.cat([prompt, torch.full((1, 128), ord("1"))], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == [128]

    ids = torch.cat([prompt, torch.full((1, 159), ord("1"))], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == [128]

    ids = torch.cat([prompt, torch.full((1, 160), ord("1"))], dim=1)
    assert not criterion(ids, None).item()
    assert processor.decoded_widths == [128, 160]


def test_repeat_stopping_returns_device_bool_and_bounds_tail():
    processor = _CharacterTokenProcessor()
    prompt = torch.tensor([[10, 11]], dtype=torch.long)
    generated = torch.full((1, 900), ord("1"), dtype=torch.long)
    criterion = ovis_module._RepeatedTailStoppingCriteria(processor, prompt_length=2)

    stopped = criterion(torch.cat([prompt, generated], dim=1), None)

    assert stopped.shape == (1,)
    assert stopped.dtype == torch.bool
    assert stopped.device == generated.device
    assert stopped.item()
    assert processor.decoded_widths == [768]
    assert criterion.triggered_match is not None
    assert criterion.triggered_at_tokens == 900


def test_repeat_stopping_rejects_batching():
    criterion = ovis_module._RepeatedTailStoppingCriteria(_CharacterTokenProcessor(), 2)

    with pytest.raises(ValueError, match="batch size 1"):
        criterion(torch.ones((2, 130), dtype=torch.long), None)


def test_repeat_stopping_keeps_normal_markdown_table_and_page_state_isolated():
    markdown = "| index | value |\n| --- | --- |\n" + "".join(
        f"| {index} | item-{index} |\n" for index in range(20)
    )
    prompt = torch.tensor([[10, 11]], dtype=torch.long)
    output = torch.tensor([[ord(char) for char in markdown]], dtype=torch.long)
    repeated = torch.full((1, 250), ord("1"), dtype=torch.long)

    first = ovis_module._RepeatedTailStoppingCriteria(_CharacterTokenProcessor(), 2)
    second = ovis_module._RepeatedTailStoppingCriteria(_CharacterTokenProcessor(), 2)

    assert first(torch.cat([prompt, repeated], dim=1), None).item()
    assert not second(torch.cat([prompt, output], dim=1), None).item()
    assert first.triggered_match is not None
    assert second.triggered_match is None
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ovis_ocr2_provider.py -k "repeat_stopping" -q
```

Expected: FAIL because `_RepeatedTailStoppingCriteria` is absent.

- [ ] **Step 3: Implement the lazy duck-typed criterion**

Add a class without a module-level Transformers or Torch import:

```python
class _RepeatedTailStoppingCriteria:
    def __init__(self, processor: Any, prompt_length: int) -> None:
        self.processor = processor
        self.prompt_length = int(prompt_length)
        self.triggered_match: Optional[_RepeatedTailMatch] = None
        self.triggered_at_tokens: Optional[int] = None
        self._next_check = _EARLY_REPEAT_START_TOKENS

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        import torch

        if int(input_ids.shape[0]) != 1:
            raise ValueError("OvisOCR2 repeated-tail stopping requires batch size 1")

        generated_tokens = max(0, int(input_ids.shape[1]) - self.prompt_length)
        if generated_tokens < self._next_check:
            return torch.zeros((1,), device=input_ids.device, dtype=torch.bool)

        self._next_check = generated_tokens + _EARLY_REPEAT_CHECK_INTERVAL
        tail_start = max(
            self.prompt_length,
            int(input_ids.shape[1]) - _EARLY_REPEAT_TAIL_TOKENS,
        )
        decoded = self.processor.batch_decode(
            input_ids[:, tail_start:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        match = _find_repeated_tail(
            decoded[0] if decoded else "",
            min_text_len=0,
            max_period=200,
            min_period=1,
            min_repeat_chars=_EARLY_REPEAT_MIN_CHARS,
            min_repeat_times=_EARLY_REPEAT_MIN_TIMES,
        )
        if match is not None:
            self.triggered_match = match
            self.triggered_at_tokens = generated_tokens
        return torch.full((1,), match is not None, device=input_ids.device, dtype=torch.bool)
```

- [ ] **Step 4: Write failing strict-revalidation tests**

Add:

```python
def test_normalize_triggered_repeat_requires_the_recorded_fingerprint():
    repeated = "1\n\n" * 80
    trigger = ovis_module._find_repeated_tail(
        repeated,
        min_text_len=0,
        max_period=200,
        min_period=1,
        min_repeat_chars=200,
        min_repeat_times=8,
    )

    assert trigger is not None
    assert ovis_module._normalize_triggered_repeat("header\n" + repeated, trigger) == "header\n1\n\n"
    assert ovis_module._normalize_triggered_repeat("header\n" + repeated[:-1] + "x", trigger) is None
```

- [ ] **Step 5: Implement and pass strict revalidation**

Implement:

```python
def _normalize_triggered_repeat(
    text: str,
    trigger: _RepeatedTailMatch,
) -> Optional[str]:
    if not text.endswith(trigger.suffix_fingerprint):
        return None
    full_match = _find_repeated_tail(
        text,
        min_text_len=0,
        max_period=trigger.period_len,
        min_period=trigger.period_len,
        min_repeat_chars=_EARLY_REPEAT_MIN_CHARS,
        min_repeat_times=_EARLY_REPEAT_MIN_TIMES,
        expected_period_len=trigger.period_len,
    )
    if full_match is None:
        return None
    return _collapse_repeated_tail(text, full_match)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ovis_ocr2_provider.py -k "repeat_stopping or normalize_triggered_repeat" -q
```

Expected: all selected tests PASS.

---

### Task 3: Wire the Criterion into Direct Inference

**Files:**
- Modify: `module/providers/ocr/ovis_ocr2.py:191-229`
- Test: `tests/test_ovis_ocr2_provider.py:171-291`

**Interfaces:**
- Consumes: the criterion and strict normalizer from Task 2.
- Preserves: `infer_page(rgb_image, prompt) -> str` for both backends.
- Preserves: vLLM messages, full-resolution PNG, and `extra_body` byte-for-byte.

- [ ] **Step 1: Extend the Direct contract test and verify RED**

Add:

```python
assert len(captured["generate_kwargs"]["stopping_criteria"]) == 1
assert isinstance(
    captured["generate_kwargs"]["stopping_criteria"][0],
    ovis_module._RepeatedTailStoppingCriteria,
)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ovis_ocr2_provider.py::test_direct_inferencer_uses_native_qwen35_contract -q
```

Expected: FAIL because `generate()` has no `stopping_criteria` argument.

- [ ] **Step 2: Pass one fresh criterion per page**

Immediately before `model.generate()`, create the criterion using `inputs["input_ids"].shape[1]`, then call:

```python
stopping_criterion = _RepeatedTailStoppingCriteria(
    processor,
    prompt_length=int(inputs["input_ids"].shape[1]),
)
generated_ids = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=self.max_new_tokens,
    stopping_criteria=[stopping_criterion],
)
```

A plain list is intentional: Transformers 5.x merges it into its internal `StoppingCriteriaList`, preserving lazy imports and the current fake-module test seam.

- [ ] **Step 3: Write a failing Direct normalization/logging test**

Reuse the Direct test's loader monkeypatch with a processor whose `batch_decode()` maps integer token IDs to characters. Use this model:

```python
class FakeRepeatingModel:
    device = torch.device("cpu")

    def __init__(self, final_text=None):
        self.final_text = final_text

    def generate(self, **kwargs):
        trigger_text = "header\n" + "1\n\n" * 80
        trigger_ids = torch.tensor(
            [[10, 11, *[ord(char) for char in trigger_text]]],
            dtype=torch.long,
        )
        assert kwargs["stopping_criteria"][0](trigger_ids, None).item()
        output_text = self.final_text if self.final_text is not None else trigger_text
        return torch.tensor(
            [[10, 11, *[ord(char) for char in output_text]]],
            dtype=torch.long,
        )
```

For the success case, assert:

```python
assert result == "header\n1\n\n"
log = console.file.getvalue()
assert "generated_tokens=" in log
assert "period_chars=3" in log
assert "repeat_times=80" in log
```

For the failure case instantiate `FakeRepeatingModel(final_text="header\nnot repeated")`; assert the result is unchanged and the console contains `could not revalidate repeated tail`.

- [ ] **Step 4: Normalize and log only after generation**

After final decoding, add this shallow branch and never print inside the per-token criterion:

```python
text = decoded[0] if decoded else ""
trigger = stopping_criterion.triggered_match
if trigger is not None:
    normalized = _normalize_triggered_repeat(text, trigger)
    if normalized is None:
        self.console.print(
            "[yellow]OvisOCR2 warning:[/yellow] "
            "could not revalidate repeated tail; preserving decoded output"
        )
    else:
        text = normalized
        self.console.print(
            "[yellow]OvisOCR2 stopped repeated output:[/yellow] "
            f"generated_tokens={stopping_criterion.triggered_at_tokens}, "
            f"period_chars={trigger.period_len}, repeat_times={trigger.repeat_times}"
        )
return text
```

- [ ] **Step 5: Run Direct and vLLM contract tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ovis_ocr2_provider.py -k "direct_inferencer or openai_inferencer or stopping or normalize" -q
```

Expected: all selected tests PASS and the existing vLLM assertion is unchanged.

---

### Task 4: Regression Verification and Real GPU Acceptance

**Files:**
- Verify: `module/providers/ocr/ovis_ocr2.py`
- Verify: `module/providers/ocr/ovis_ocr2_contract.py`
- Verify: `tests/test_ovis_ocr2_provider.py`
- Verify: `tests/test_ovis_ocr2_integration.py`
- Verify: `tests/test_model_config_panel.py`

**Interfaces:**
- Consumes: completed Direct stopping implementation.
- Produces: automated and live-smoke evidence, with no new production API.

- [ ] **Step 1: Run the related pytest set**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ovis_ocr2_provider.py tests\test_ovis_ocr2_integration.py tests\test_model_config_panel.py tests\test_provider_catalog.py tests\test_provider_routes.py -q
```

Expected: all tests PASS with no new OvisOCR2 warnings.

- [ ] **Step 2: Run Ruff and whitespace checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check module\providers\ocr\ovis_ocr2.py module\providers\ocr\ovis_ocr2_contract.py tests\test_ovis_ocr2_provider.py
git diff --check
```

Expected: both commands exit 0. Do not run `uv lock --check`; no tracked lock exists.

- [ ] **Step 3: Run the known first-image Direct GPU smoke**

Run:

```powershell
@'
import time
from pathlib import Path

from PIL import Image
from rich.console import Console

from module.providers.ocr.ovis_ocr2 import (
    _DirectPageInferencer,
    _find_repeated_tail,
)
from module.providers.ocr.ovis_ocr2_contract import OVIS_OCR2_DEFAULT_PROMPT

image_path = Path(r"datasets/[20240517]BanGDreamItsMyGOArt01/001.jpg")
image = Image.open(image_path).convert("RGB")
inferencer = _DirectPageInferencer(
    model_id="ATH-MaaS/OvisOCR2",
    max_new_tokens=16384,
    min_pixels=448 * 448,
    max_pixels=2880 * 2880,
    console=Console(),
)
started = time.perf_counter()
text = inferencer.infer_page(image, OVIS_OCR2_DEFAULT_PROMPT)
elapsed = time.perf_counter() - started
remaining_repeat = _find_repeated_tail(
    text,
    min_text_len=0,
    max_period=200,
    min_period=1,
    min_repeat_chars=200,
    min_repeat_times=8,
)
print(f"elapsed_seconds={elapsed:.1f}")
print(f"output_chars={len(text)}")
print(f"tail={text[-120:]!r}")
assert remaining_repeat is None
'@ | .\.venv\Scripts\python.exe -
```

Acceptance on the current RTX 4090 machine:

- generation stops far below 16384 new tokens;
- the console reports trigger token count, `period_chars`, and `repeat_times`;
- output does not retain a long `1\n\n` suffix;
- cold execution finishes in roughly two minutes or less as a manual observation, not an automated cross-machine assertion.

- [ ] **Step 4: Inspect scope before branch finalization**

Run:

```powershell
git diff -- module/providers/ocr/ovis_ocr2.py tests/test_ovis_ocr2_provider.py docs/superpowers/specs/2026-07-14-ovisocr2-design.md
git status --short
```

Expected: the fix is confined to planned OvisOCR2 files and all unrelated dirty files remain untouched. Do not commit or push until the user requests branch finalization.
