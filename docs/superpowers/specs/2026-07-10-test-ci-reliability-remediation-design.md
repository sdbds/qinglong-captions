# Test and CI Reliability Remediation Design

Date: 2026-07-10

## Review Decision

This design is intentionally separate from runtime correctness repair.

It covers:

- the currently failing full pytest suite;
- order-dependent test contamination;
- stale tests that assert superseded contracts;
- the broken Provider discovery command in GitHub Actions;
- incomplete CI path coverage and file-list-based suite selection;
- a narrow static correctness gate for undefined names.

It does not cover Hatchling, wheel building, package publication, credential handling, command-line secret logging, or unauthenticated cloud mode.

Runtime review issues 3 through 8 are specified in `2026-07-10-runtime-correctness-data-integrity-remediation-design.md`. Regression tests required to prove those runtime fixes remain owned by that spec.

## Baseline

The 2026-07-10 local baseline produced:

```text
1143 passed, 26 failed, 2 skipped
```

The failures are not one category:

1. deterministic workflow contract failures caused by `from providers.registry` instead of `from module.providers.registry`;
2. stale assertions for the current Gemma score limits, model configuration UI, and `CaptionResult` return contract;
3. order-dependent failures caused by module-level fake `cv2` objects left in `sys.modules`;
4. a real Blob v2 translation failure owned by the runtime correctness spec;
5. a missing `paddleocr-native` conflict declaration for the new `musvit-onnx` extra;
6. environment-sensitive optional-runtime tests that need explicit skip/fixture boundaries.

Several groups that failed in the full suite passed when run alone, including Unlimited OCR, runtime backends, Penguin dependency tests, and WDTagger OpenCV selection. That is direct evidence of test pollution rather than independent production failures.

The current workflow also has two structural gaps:

- the strict discovery command cannot import its target package;
- CI watches and runs only a manually selected subset of the repository.

## First Principles

1. A test must leave interpreter, environment, filesystem, and module state as it found them.
2. A test assertion must protect a current documented contract, not preserve an obsolete implementation accident.
3. There must be one authoritative command whose success means the repository's unit suite is green.
4. CI selection must be based on declared test classes and supported environments, not a growing filename whitelist.
5. A static correctness gate should reject definite runtime errors without forcing an unrelated style cleanup.
6. Tests for a runtime fix belong with that fix; infrastructure tests only validate collection, isolation, selection, and execution.

## Goals

1. Make the declared offline unit command pass from a fresh repository environment.
2. Make isolated and full-suite execution produce the same result for each test group.
3. Remove module-level optional-dependency stubs and direct `sys.modules` pollution.
4. Update stale tests to current explicit contracts without changing production behavior merely to satisfy old assertions.
5. Fix strict Provider discovery to use the canonical `module.providers` namespace.
6. Run an authoritative full unit suite on Windows and Linux.
7. Keep Python 3.10 and 3.12 compatibility coverage without multiplying the full-suite cost across every matrix cell.
8. Add a narrow Ruff gate for `F821` and `F823`.
9. Make CI run when any runtime, GUI, configuration, script, test, or CI helper path changes.

## Non-Goals

- Do not implement runtime review issues 3 through 8 in this spec.
- Do not change scoring, UI, or `CaptionResult` product behavior solely to preserve stale tests.
- Do not make all 707 current Ruff findings blocking.
- Do not perform broad formatting or import sorting.
- Do not add wheel-build or package-install smoke tests.
- Do not address credentials or cloud authentication.
- Do not require GPU hardware or model downloads in the default unit suite.
- Do not turn optional integration tests into mandatory online tests.

## Test Ownership Model

Classify tests into three explicit layers.

### Unit

Default marker: no marker required.

Properties:

- deterministic and offline;
- no GPU requirement;
- no model download;
- no live remote API;
- completes in the normal full-suite job.

The authoritative command is:

```powershell
python -m pytest tests -q --strict-markers -m "not optional_runtime and not gpu and not network"
```

Tests that cannot satisfy this contract must be marked as optional integration tests or rewritten with local fakes.

### Compatibility smoke

Marker:

```text
compat
```

This is a deliberately small set of tests covering:

- canonical Provider discovery and routing;
- configuration parsing;
- path safety and output writing;
- pure caption pipeline behavior;
- GUI lazy import and command construction;
- CI project rendering.

It runs on Python 3.10 and 3.12 on Windows and Linux. Membership is expressed with the marker, not a workflow filename list.

### Optional integration

Markers:

```text
optional_runtime
gpu
network
```

These tests require an optional package stack, GPU, model artifact, external executable, or network service. They are excluded from the authoritative offline unit command unless their prerequisites are detected. A skipped optional test must state the missing prerequisite.

Register all markers in `pyproject.toml` so unknown markers fail review.

## Design 1: Hermetic Import and Module State

### Remove module-level fake modules

The immediate contamination comes from module-level statements such as:

```python
sys.modules.setdefault("cv2", fake_cv2)
```

in preprocessing tests. The fake is a `SimpleNamespace` without a valid module spec. Later Transformers imports call `importlib.util.find_spec("cv2")` and raise `ValueError: cv2.__spec__ is not set`.

Required repair:

- remove every module-level `sys.modules` mutation used to fake optional dependencies;
- create fakes as `types.ModuleType` with a valid `ModuleSpec` when import machinery must inspect them;
- install them through `monkeypatch.setitem(sys.modules, ...)` inside a fixture or test;
- import or reload the module under test only after the fixture installs the fake;
- remove the imported project module during teardown when its module globals retain the fake;
- let `monkeypatch` restore the original dependency module automatically.

### Centralize repository path setup

Tests currently repeat `sys.path.insert(0, str(ROOT))` across many files. Pytest already runs from the repository root and CI sets `PYTHONPATH` to that root.

Required repair:

- keep root setup in one `tests/conftest.py` only if local invocation proves it is still necessary;
- remove per-file module-level root insertion;
- never insert `ROOT / "module"`, which can create a second top-level Provider package;
- tests that intentionally load a file under a synthetic module name may continue using `importlib.util.spec_from_file_location` without modifying global package identity.

### Isolation guard

Add `tests/test_test_isolation_contract.py` to reject:

- module-level `sys.modules[...] = ...`;
- module-level `sys.modules.setdefault(...)`;
- module-level `sys.path.insert(...)` outside `tests/conftest.py`;
- fake optional modules that are not `ModuleType` when inserted into `sys.modules`.

The guard may use a small explicit allowlist only for tests whose purpose is validating import isolation. The allowlist must name exact file and line behavior, not a directory wildcard.

### Order regression

Run the known contamination sequence in both directions using fresh subprocesses:

```text
preprocess tests -> Unlimited OCR/runtime backend tests
Unlimited OCR/runtime backend tests -> preprocess tests
```

Both orders must pass. This is more diagnostic than adding random ordering to every CI run.

## Design 2: Repair Stale Contract Tests

Production behavior is authoritative when it is represented by a newer explicit contract in code or focused tests. Do not restore obsolete production behavior to make stale assertions green.

### Gemma image scores

Current `_IMAGE_SCORE_LIMITS` assigns a maximum of 10 to every canonical score category. Tests that still expect `Setting & Environment Integration` and `Storytelling & Concept` to be forced to 5 are stale.

Required changes:

- update structured and free-form expectations to preserve valid values up to 10;
- add one direct boundary test proving values below 0 clamp to 0 and values above 10 clamp to 10;
- keep label alias normalization tests separate from value-limit tests.

No production scoring change belongs in this spec.

### Model configuration panel

The current UI intentionally renders:

- a searchable product-name `select` for `model_list`;
- a free-text `input` with datalist suggestions for `model_id`.

Tests still expect `model_id` to be a second select and patch a removed `_load_model_list_entries` method.

Required changes:

- extend `_FakeUI` with a fake input and HTML/datalist support;
- patch the module-level `load_model_list_entries` function;
- assert product selection fills the free-text model ID input;
- assert a custom model ID clears the product-name selection;
- assert arbitrary custom model IDs remain accepted.

Do not revert the UI to a closed select.

### CaptionResult return contract

`postprocess_caption_content()` now returns `CaptionResult`. Music Flamingo tests that compare it directly with a string are stale.

Required changes:

- assert the result is `CaptionResult`;
- compare normalized subtitle text through `result.raw`;
- preserve the existing closed and unclosed SRT fence cases.

### Dependency conflict invariant

`musvit-onnx` depends on `onnx-base`, which includes the Torch stack. The existing test correctly requires every Torch-stack extra to conflict with `paddleocr-native`.

Required change:

- add `musvit-onnx` to the appropriate `paddleocr-native` conflict declaration;
- keep the generic invariant test rather than weakening it or adding a test exception.

This is a configuration drift repair justified by an existing repository-wide compatibility rule.

### Runtime-owned failure

`tests/test_text_translation_pipeline.py::test_normalize_and_translate_dataset_roundtrip` remains assigned to the runtime correctness spec because its failure requires Blob v2-aware production persistence. This test spec must not mark it skipped or loosen its assertions.

## Design 3: Canonical CI Commands

### Strict Provider discovery

Replace:

```python
from providers.registry import get_registry
```

with:

```python
from module.providers.registry import get_registry
```

Keep `PYTHONPATH` rooted at the repository root. The existing workflow contract test must pass before the Provider suite runs.

### Static correctness job

Add a fast job that runs:

```powershell
python -m compileall -q module gui utils config tests
python -m ruff check module gui utils config tests --select F821,F823
python -m pytest tests/test_test_workflow.py tests/test_test_isolation_contract.py -q --strict-markers
```

Add Ruff to the declared test dependency group and run the environment's installed version; do not use an unpinned `uvx` download in CI. Only `F821` and `F823` are initially blocking. The rest of the Ruff backlog remains visible but is not part of this acceptance gate.

### Full unit job

Run on:

```text
Windows latest, Python 3.11
Ubuntu latest, Python 3.11
```

Command:

```powershell
& $env:VENV_PYTHON -m pytest tests -q --strict-markers -m "not optional_runtime and not gpu and not network" --durations=25
```

This is the authoritative repository test result. Do not enumerate test filenames.

### Compatibility matrix job

Run the `compat` marker on:

```text
Windows latest, Python 3.10 and 3.12
Ubuntu latest, Python 3.10 and 3.12
```

Command:

```powershell
& $env:VENV_PYTHON -m pytest tests -q --strict-markers -m compat
```

Python 3.11 is covered by the full unit job and does not need another compatibility cell.

### Provider discovery job

Provider discovery can remain a separate failure boundary because it gives a clearer error than a later routing test:

```powershell
& $env:VENV_PYTHON -c "from module.providers.registry import get_registry; get_registry().discover(strict=True)"
```

Run the focused Provider registry/routes/API tests after strict discovery in the same job or fold them into the full unit job once failure reporting remains clear.

## Design 4: CI Trigger Coverage

The test workflow must run for changes to:

```text
module/**
gui/**
utils/**
config/**
tests/**
third_party/**
*.ps1
*.sh
.github/scripts/**
.github/workflows/test.yml
pyproject.toml
```

Documentation-only changes may remain excluded.

Do not list individual production modules such as `module/captioner.py` or `module/videospilter.py`; the directory boundary is the stable ownership unit.

Add workflow contract assertions for all trigger roots so a new major package cannot be silently omitted.

## Design 5: Optional Dependency Discipline

The default test dependency group must be sufficient for the full offline suite. Tests may not rely on whichever optional packages happen to exist in a developer's long-lived `.venv`.

Rules:

- optional imports are faked inside hermetic fixtures;
- tests requiring the real optional package use `pytest.importorskip` with a clear reason and an `optional_runtime` marker;
- GPU tests require the `gpu` marker and skip when no compatible device exists;
- network tests require the `network` marker and are not part of pull-request gating;
- no unit test downloads a model or calls a live Provider;
- fake modules are restored after every test.

Add a clean-environment smoke command that uses the same rendered CI project and exported requirements as GitHub Actions. This verifies that local success does not depend on undeclared packages.

## Failure Classification and Repair Order

Repair failures in this order:

1. Fix the canonical Provider discovery import so CI can reach its tests.
2. Remove module-level `cv2` and other optional-module contamination.
3. Re-run the full suite and separate remaining deterministic failures from environment skips.
4. Update Gemma, model panel, and Music Flamingo stale assertions to current contracts.
5. Add the missing MuSViT/Paddle conflict declaration.
6. Land runtime-spec regression fixes, including Blob v2 translation.
7. Make `F821,F823` clean and enable the static job.
8. Replace filename test lists with full-unit and compatibility-marker jobs.
9. Expand workflow trigger paths.
10. Run clean-environment and both-order isolation checks.

Do not broaden CI to the full suite while the suite is knowingly red. First make the local authoritative command green, then make CI enforce it in the same change series.

## Acceptance Criteria

- The declared offline unit command passes from a fresh declared test environment.
- The authoritative full suite passes on Windows and Linux with Python 3.11.
- The `compat` marker passes on Python 3.10 and 3.12 on both operating systems.
- Strict Provider discovery imports only `module.providers.registry` and succeeds before Provider tests.
- Known contamination groups pass in both execution orders.
- No ordinary test performs module-level `sys.modules` or `sys.path` mutation.
- Gemma, model panel, and Music Flamingo tests assert current contracts.
- The generic Torch-stack conflict invariant includes `musvit-onnx` without an allowlist exception.
- Ruff `F821,F823` reports zero findings for `module`, `gui`, `utils`, `config`, and `tests`.
- CI triggers for all declared production, GUI, vendored/submodule, script, test, configuration, and CI-helper roots.
- CI no longer enumerates the authoritative unit suite by filename.
- Optional GPU, network, and model-runtime tests skip with explicit reasons when prerequisites are absent.
- No wheel or Hatchling work is included.

## Risks and Stop Conditions

### Hidden optional dependencies

Risk: the full suite may currently pass only because the developer environment contains undeclared packages. Treat a clean-environment import failure as a dependency or fixture bug. Do not add the entire optional runtime to the base test group without proving the test needs the real package.

### Stale test ambiguity

Risk: a stale assertion may be the only surviving record of an intended product rule. Update it only when a newer explicit contract exists in code, configuration, or focused tests. The Gemma score limit, model ID free-text input, and `CaptionResult` return type meet that standard.

### CI duration

Risk: a full cross-platform suite costs more than the current whitelist. Limit the authoritative full suite to Python 3.11 on two operating systems and use the `compat` marker for the version edges. Do not return to filename selection to save time.

### Random-order testing

Risk: adding random ordering immediately can make failures harder to reproduce. First enforce hermetic fixtures and the two known opposite-order subprocess checks. Add randomized ordering later only if order regressions recur.

### Runtime/test spec sequencing

The translation round-trip remains red until the runtime correctness spec lands. Do not suppress it. During a split rollout, record that single expected failure in branch coordination, but the final integrated acceptance state has zero expected failures.
