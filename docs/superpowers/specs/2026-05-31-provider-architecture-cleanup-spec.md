# Provider Architecture Cleanup Spec

Date: 2026-05-31

## Review Verdict

The global review is directionally correct: the project has a good Provider spine, but the current growth path creates duplicate truth sources and pushes provider-specific exceptions back into generic orchestration.

Two corrections matter before implementation:

1. The top-level `providers` import problem is real, but the exact failure mode is sharper than "the registry singleton always splits".
   - With normal root-only imports, the shim keeps `providers.registry` and `module.providers.registry` identical.
   - With `ROOT / "module"` inserted before the repo root, which many tests do, `providers.base` resolves directly to `module/providers/base.py` as a top-level package. Then `providers.base.CaptionResult is module.providers.base.CaptionResult` is false.
   - `providers.ocr.dots` and `module.providers.ocr.dots` also become different modules and register different class objects.
   - The registry module is partly protected by the current `sys.modules` hack, but Provider classes, dataclasses, monkeypatch targets, and pending registrations still split.

2. `third_party/SongPrep/.venv` exists locally, but it is not tracked by the main repository and is not tracked by the SongPrep submodule. The main `.gitignore` already contains `.venv`. Treat this as workspace hygiene, not as committed repository pollution.

3. The provider list problem is real, but not every list is identical data.
   - `_PROVIDER_MODULES` and `_priority_order` currently contain the same 39 provider names and must stay synchronized.
   - `ROUTE_SPECS` has 27 route-selectable provider names because some cloud providers are selected by credentials rather than CLI route values.
   - `PROVIDER_SPECS` only covers 15 providers with aliases/config/prompt fallbacks.
   - The issue is still "too many manual declarations", but the fix must preserve the distinction between provider existence, auto priority, explicit route choices, aliases, config sections, and prompt fallback names.

Everything else in the review is substantially correct:

- `captioner.api_process_batch()` unwraps `CaptionResult` back to `dict | str`.
- `orchestrator`, `postprocess`, `output_writer`, and `dataset_sync` then re-discover the result shape with `isinstance`.
- `pyproject.toml` repeats the same Windows `flash-attn` URL 21 times and has a 533-line conflict matrix.
- `orchestrator.py`, `catalog.py`, and `resolver.py` contain provider-name special cases.
- There are hundreds of broad `except Exception` blocks; some still silently drop write/display failures.
- Several large functions/files should be split, but they are not the first correctness risk.

## First Principles

The system needs one canonical identity for each thing:

- one import path for Provider code;
- one return type across the caption pipeline;
- one declaration record for each provider's module, priority, routes, config sections, prompt fallbacks, and runtime behavior;
- one place where provider-specific capabilities are expressed.

When the same concept is represented by two names, two module objects, or three hand-maintained lists, the code cannot be reasoned about locally. Fixing symptoms in call sites will only add more patches.

The right repair order is therefore:

1. Fix identity.
2. Fix data flow shape.
3. Fix declaration ownership.
4. Move provider-specific behavior into provider contracts.
5. Clean dependency and exception hygiene.

## Goals

1. Make `module.providers.*` the only supported Provider import path in production code, tests, scripts, and docs.
2. Remove test `sys.path` setup that allows `providers.*` to bypass the shim and load `module/providers` as a second top-level package.
3. Keep `CaptionResult` intact from `module.api_handler_v2.api_process_batch()` through postprocess, sidecar writing, and Lance dataset update.
4. Collapse provider module import, priority, route, alias, config section, and prompt fallback metadata into a single provider declaration model.
5. Remove provider-name checks from generic orchestration by adding provider capability/hooks.
6. Reduce dependency duplication in `pyproject.toml` without changing optional-extra semantics.
7. Add guard tests that fail when import identity or provider declaration consistency regresses.

## Non-Goals

- Do not rewrite all providers at once.
- Do not change model behavior, prompts, or output content unless required by `CaptionResult` plumbing.
- Do not remove backward-compatible user CLI flags in this cleanup.
- Do not solve every `except Exception` instance in one pass.
- Do not split every large file before the correctness fixes land.
- Do not rewrite the dependency resolver model if a smaller generated/check-based approach is enough.

## Phase 0: Lock The Current Risk With Tests

Add failing tests before edits:

1. `test_provider_import_identity_with_module_path_first`
   - Temporarily place `ROOT / "module"` before `ROOT` in `sys.path`.
   - Import `providers.base` and `module.providers.base`.
   - Assert this situation is rejected or cannot occur after the fix.

2. `test_provider_modules_have_single_identity`
   - For representative modules such as `base`, `registry`, `ocr.dots`, `local_alm.cohere_transcribe_local`, import canonical `module.providers.*`.
   - Assert there is no loaded `providers.*` counterpart in `sys.modules`.

3. `test_no_tests_import_top_level_providers`
   - Scan `tests/**/*.py`.
   - Fail on `from providers`, `import providers`, `providers.` monkeypatch strings.
   - Allow only comments inside this spec or explicit migration test fixtures.

4. `test_no_test_adds_module_directory_to_sys_path_for_providers`
   - Fail on `sys.path.insert(... ROOT / "module" ...)` in tests that import providers.
   - Keep unrelated legacy module-path cases only if they do not touch Provider packages, and mark them with a narrow allowlist.

These tests are not style policing. They protect Python module identity.

## Phase 1: One Provider Import Path

Canonical path:

```python
module.providers
module.providers.base
module.providers.registry
module.providers.ocr.dots
```

Required changes:

1. Replace all test imports from `providers.*` with `module.providers.*`.
2. Replace monkeypatch targets such as `providers.ocr.dots.encode_image_to_blob` with `module.providers.ocr.dots.encode_image_to_blob`.
3. Remove `ROOT / "module"` insertion from provider tests.
4. Update workflow/test scripts that run `from providers.registry import get_registry` to `from module.providers.registry import get_registry`.
5. Keep production `PYTHONPATH` rooted at the project root, not at the `module` directory.
6. After tests are migrated, delete the top-level `providers/` compatibility shim.
7. Delete the `sys.modules.setdefault(...)` alias hack from `module/providers/registry.py`.

Acceptance:

- `rg -n "from providers|import providers|providers\\." tests module gui utils` returns no Provider import/patch usage outside migration notes.
- `rg -n "ROOT / [\"']module[\"']" tests` has no Provider-related path insertion.
- Fresh Python import check:

```python
import module.providers.base as base
import module.providers.registry as registry
assert base.CaptionResult.__module__ == "module.providers.base"
assert registry.get_registry() is registry.get_registry()
```

- Running `pytest tests/test_provider_registry.py tests/test_api_handler_v2.py tests/test_dots_ocr_provider.py` passes after import target updates.

## Phase 2: Keep CaptionResult Through The Pipeline

Current bad boundary:

```python
def api_process_batch(...):
    result = _api_process_batch_v2(...)
    if hasattr(result, "parsed") and result.parsed is not None:
        return result.parsed
    if hasattr(result, "raw"):
        return result.raw
    return result
```

Replace this with a real contract:

```python
def api_process_batch(...) -> CaptionResult:
    return _api_process_batch_v2(...)
```

Extend `CaptionResult` with explicit output helpers:

```python
@dataclass
class CaptionResult:
    raw: str
    parsed: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def payload(self) -> Dict[str, Any] | str:
        return self.parsed if self.parsed is not None else self.raw

    @property
    def text(self) -> str:
        return self.description

    @property
    def caption_extension(self) -> str | None:
        ...

    def to_dataset_caption(self) -> str:
        ...
```

Then change these modules:

- `module/caption_pipeline/postprocess.py`
  - Accept `CaptionResult`.
  - Return `CaptionResult`.
  - For legacy list/string parsing, wrap normalized output back into `CaptionResult`.

- `utils/output_writer.py`
  - Accept `CaptionResult`.
  - Use `result.parsed` for JSON sidecar.
  - Use `result.description` or structured subtitle fields for text sidecar.

- `module/caption_pipeline/dataset_sync.py`
  - Accept `CaptionResult`.
  - Use `result.to_dataset_caption()`.
  - Keep old `list/dict/str` support during the migration, but mark it legacy.

- `module/caption_pipeline/orchestrator.py`
  - Stop branching on `isinstance(output, dict)` for provider result type.
  - Use `result.is_structured`, `result.get(...)`, and `result.description`.
  - Subtitle alignment should operate on `result.raw` or a method like `result.subtitle_text`.

Acceptance:

- `tests/test_api_handler_v2.py` asserts `api_process_batch()` returns `CaptionResult`.
- Add `tests/test_caption_result_pipeline.py` covering:
  - raw image caption;
  - structured image caption;
  - audio transcript payload;
  - AST translation SRT payload;
  - document markdown payload.
- `rg -n "hasattr\\(result, \"parsed\"\\)|hasattr\\(result, \"raw\"\\)" module utils tests` returns no production compatibility unwrapping.
- The old `dict | str` conversions are only in a clearly named legacy adapter, if still needed by tests during rollout.

## Phase 3: Provider Declaration As Data

Introduce one declaration object:

```python
@dataclass(frozen=True)
class ProviderDeclaration:
    name: str
    module_path: str
    priority: int
    routes: tuple[RouteSpec, ...] = ()
    aliases: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ()
    prompt_prefixes: tuple[str, ...] = ()
    capabilities: ProviderCapabilities = ProviderCapabilities()
```

Update the decorator:

```python
@register_provider(
    name="marlin_2b_local",
    module_path=__name__,
    priority=260,
    routes=(RouteSpec("vlm_image_model", "marlin_2b_local"),),
    config_sections=("marlin_2b_local", "marlin"),
    prompt_prefixes=("marlin",),
)
class Marlin2BLocalProvider(LocalVLMProvider):
    ...
```

Rules:

- The declaration lives next to the provider class.
- Registry discovery imports declarations.
- Catalog functions derive route choices, aliases, config sections, and prompt prefixes from declarations.
- Provider auto-selection derives priority from declarations.
- A provider without a route can still exist and be selected by credentials.
- A route without a provider is invalid.

Do this in two steps:

1. Add declarations while still generating from existing `_PROVIDER_MODULES`, `_priority_order`, `ROUTE_SPECS`, and `PROVIDER_SPECS`.
2. Once generated data matches current behavior, delete the old hand-maintained structures.

Acceptance:

- Test that every provider in discovery has a declaration.
- Test that declaration names are unique.
- Test that priority values are unique or tie-broken explicitly.
- Test that every route provider resolves to a declared provider.
- Test that old aliases such as `pixtral` and `pixtral_ocr` still canonicalize to `mistral_ocr`.
- Add a provider fixture with one declaration and prove it appears in registry, route choices, config section lookup, and prompt prefixes without editing four files.

## Phase 4: Move Provider Special Cases Into Provider Contracts

Generic orchestration must not know model names like `gemma4_local` or `marlin_2b_local`.

Add a small behavior contract:

```python
@dataclass(frozen=True)
class SegmentationPolicy:
    bypass: bool = False
    default_segment_seconds: int | None = 600
    max_direct_duration_ms: int | None = None
    safe_segment_seconds: int | None = None

class Provider:
    @classmethod
    def segmentation_policy(cls, args, mime: str, config) -> SegmentationPolicy:
        return SegmentationPolicy()

    @classmethod
    def prompt_fallback_keys(cls, mime: str, field: str) -> tuple[str, ...]:
        return ()
```

Provider-specific mappings:

- `gemma4_local`
  - `bypass=True` for its current audio/video direct mode.

- `marlin_2b_local`
  - `max_direct_duration_ms` and `safe_segment_seconds` from `[marlin_2b_local]` / `[marlin]`.

- `music_flamingo_local`
  - default audio segment time `1200`.

- `cohere_transcribe_local`
  - default audio segment time `None`.

- `kimi_code`, `kimi_vl`, `minimax_code`, `mistral_ocr`, `stepfun`, `step_vl_local`
  - prompt fallback keys returned by provider declaration/contract, not by `resolver.py` name checks.

Acceptance:

- `module/caption_pipeline/orchestrator.py` contains no string literals for provider names except generic route-field names.
- `module/providers/catalog.py` no longer sets segment defaults by checking `args.alm_model == ...`.
- `module/providers/resolver.py` no longer has provider-name fallback branches.
- Existing tests for Gemma4 bypass, Marlin segment cap, Music Flamingo default, Cohere no-segment default, and prompt fallbacks still pass through the new contract.

## Phase 5: Dependency Declaration Cleanup

Current facts:

- The Windows `flash-attn` wheel URL appears 21 times.
- There are 42 `flash-attn` dependency lines.
- The `conflicts` block is about 533 lines and has 274 `{ extra = ... }` entries.
- `[tool.uv.sources]` exists, but `flash-attn` is not centralized there.

Repair options, in preferred order:

1. If uv supports the exact marker/URL shape needed, move the Windows `flash-attn` URL into `[tool.uv.sources]` and keep extras using `flash-attn==...` style dependency lines.
2. If uv cannot express this cleanly, add a small generator script that produces the repeated extra fragments and conflict matrix from a compact table.
3. Add a validation test that parses `pyproject.toml` and fails when the same direct URL is repeated more than once outside the generated section.

Do not hand-edit 21 repeated URLs.

Acceptance:

- `pyproject.toml` has one source of truth for the Windows `flash-attn` wheel URL.
- Extras retain current install behavior on Windows, Linux, and macOS.
- `tests/test_pyproject_uv_conflicts.py` is extended to validate the generated/centralized conflict data.
- If generation is used, generated sections are clearly marked and the generator is checked in.

## Phase 6: Exception Hygiene

Do not try to fix all broad catches in one commit. Start where data loss is likely.

Priority targets:

1. `module/providers/ocr_base.py`
   - `write_markdown_output(...)` failures must log provider, file, output dir, and exception.
   - `display_markdown(...)` failures can be warning-level logs, not hard failures.

2. `module/providers/registry.py`
   - `can_handle()` exceptions during provider selection should be recorded for debug output, not silently skipped.

3. Caption sidecar writes
   - Sidecar write failures should raise unless explicitly configured as best-effort.

Policy:

- Recoverable optional UI/display failures: log and continue.
- Filesystem writes that represent the result: raise.
- Provider import failures: record and surface through existing strict discovery errors.
- `except Exception: pass` is forbidden in provider and pipeline modules.

Acceptance:

- `rg -n "except Exception:\\s*$" module/providers module/caption_pipeline utils/output_writer.py` returns no silent pass blocks.
- Tests cover OCR write failure surfacing and optional display failure logging.

## Phase 7: Large File And Dots Cleanup

Do this after identity and result-shape fixes.

`module/providers/ocr/dots.py`:

- Keep local wrappers only where upstream imports are broken or need compatibility glue.
- Prefer importing upstream `dots_ocr` utilities from `third_party/dots.ocr`.
- For copied helpers such as resize and output cleaning, either:
  - call upstream directly, or
  - isolate the forked implementation in a small module with a comment explaining the upstream incompatibility and the tests that lock behavior.

`module/videospilter.py`:

- Split `SceneDetector.align_subtitle()` into:
  - scene block construction;
  - segment boundary detection;
  - start-time coherence adjustment;
  - best scene boundary selection;
  - large-offset correction;
  - subtitle mutation.

`module/rewardmodel.py`:

- Split `main()` into:
  - config threshold loading;
  - dataset resolution;
  - model loading;
  - batch scoring;
  - quality folder materialization;
  - report writing.

Acceptance:

- No behavior changes without tests.
- New helper functions are named by domain operation, not by implementation step numbers.
- Existing tests continue to pass.

## Workspace Hygiene

Do not commit local virtual environments.

Facts:

- Main `.gitignore` already contains `.venv`.
- Main repo tracks `third_party/SongPrep` as a submodule entry, not its `.venv`.
- SongPrep submodule also does not track `.venv`.

Action:

- No code change required for main repo tracking.
- Developers can delete `third_party/SongPrep/.venv` locally if it is only a stale environment.
- Add a CI/workflow check only if accidental submodule pollution recurs.

## Implementation Order

1. Add import-identity regression tests.
2. Migrate tests and scripts to `module.providers.*`; remove `ROOT / "module"` provider-path setup.
3. Delete the top-level `providers` shim and registry alias hack.
4. Make `captioner.api_process_batch()` return `CaptionResult`; update postprocess/output/dataset sync.
5. Add provider declarations while preserving generated current catalog behavior.
6. Remove provider-name branches from orchestrator/catalog/resolver via provider contracts.
7. Clean `flash-attn` duplication and conflict declaration generation/validation.
8. Replace silent provider/pipeline exception swallowing in priority files.
9. Split large functions only after the behavioral contracts are covered.

## Stop Conditions

Stop and reassess if any of these happens:

- A public launcher or documented user command truly requires `providers.*` imports.
- `CaptionResult` propagation forces a broad rewrite of Lance export/import semantics.
- uv cannot centralize direct URL dependencies and a generator would make install behavior less transparent.
- Provider declarations require importing heavy optional model dependencies during CLI argument construction.

The fallback is not to keep hacks indefinitely. The fallback is to add a small, explicit compatibility adapter at process startup that fails loudly when it would create duplicate provider identity.
