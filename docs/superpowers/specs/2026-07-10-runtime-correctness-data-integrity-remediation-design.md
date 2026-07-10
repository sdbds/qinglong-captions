# Runtime Correctness and Data Integrity Remediation Design

Date: 2026-07-10

## Review Decision

This design implements the approved subset of the 2026-07-10 global review:

- issue 3: deterministic Provider entry-point crashes;
- issue 4: failed or empty caption results overwriting valid Lance captions;
- issue 5: WDTagger URI/probability misalignment after image decode failures;
- issue 6: See-Through vendor imports corrupting process-global import state;
- issue 7: concurrent native-console jobs sharing temporary files;
- issue 8: text translation updates failing against Lance Blob v2 datasets.

Review issues 1 and 2 are explicitly excluded. Test-suite and CI infrastructure repair is specified separately in `2026-07-10-test-ci-reliability-remediation-design.md`.

## Problem

The six approved failures look unrelated at the file level, but they share one structural defect: identity and ownership are implicit.

- Standalone Provider functions reach back into an instance that does not exist.
- Caption success, skip, failure, and empty output are represented by overlapping values.
- Preprocessed images are separated from the URIs that identify them.
- Vendored packages take ownership of global import names such as `utils`.
- Native jobs identify their files with a process ID shared by every job in that process.
- Partial Lance updates use `update_all` without supplying a full Blob v2 row.

Adding local conditionals would hide individual symptoms while preserving the same failure class. The repair must make identity explicit at each boundary.

## First Principles

1. A callable may only use state present in its arguments or explicit owner object.
2. A persisted caption must come from an explicitly successful, non-empty result.
3. A transformed value must retain the identity of the source value it came from.
4. A component may not leave process-global state different after it returns.
5. Concurrent jobs may not share mutable files unless sharing is part of the declared protocol.
6. A partial database update must preserve every column it does not own, including extension-backed Blob v2 data.

## Goals

1. Make QwenVL and the affected local OCR Provider entry points executable in a fresh process.
2. Introduce an explicit caption outcome contract and prevent non-successful results from reaching sidecar or Lance persistence.
3. Keep WDTagger URI/image/probability associations correct when any input in a batch fails to decode.
4. Remove See-Through's dependency on top-level `utils`, `modules`, and `annotators` import names.
5. Give each native-console run exclusive signal and log files with deterministic cleanup.
6. Use one shared, Blob v2-aware Lance row-update helper for caption and translation updates.
7. Preserve existing CLI flags, output formats, dataset tags, sidecar names, and successful Provider behavior.

## Non-Goals

- Do not address credential storage, command-line secret logging, or unauthenticated cloud mode.
- Do not repair unrelated Provider behavior, prompt content, scoring policy, or model selection.
- Do not rewrite the entire Provider hierarchy.
- Do not migrate the vendored See-Through implementation to a new upstream version.
- Do not redesign Lance schemas or remove Blob v2.
- Do not repair the general pytest/CI infrastructure in this change.
- Do not address Hatchling or wheel packaging.
- Do not combine this work with broad formatting, exception, or type-annotation cleanup.

## Compatibility Contract

- Existing `CaptionResult(raw=..., parsed=..., metadata=...)` construction remains valid.
- Existing successful raw and structured captions serialize exactly as before.
- Existing CLI and TOML names remain unchanged.
- Skipped or failed jobs advance progress and remain visible in logs, but do not modify sidecars, Lance captions, or success tags.
- Existing WDTagger output ordering is preserved for valid images.
- Existing See-Through public functions keep their arguments and return values while vendored modules move to their project namespace.
- Existing native-console UI behavior is preserved; only temporary resource ownership changes.
- Lance updates preserve schema, blobs, unmodified columns, version history, and tag behavior.

## Design 1: Provider Entry-Point Ownership

### QwenVL

`module/providers/cloud_vlm/qwenvl.py` already imports `Path` at module scope. The inner `from pathlib import Path` inside `QwenVLProvider.attempt()` makes `Path` local to the whole function and causes the first line to raise `UnboundLocalError`.

Required change:

- delete the inner import;
- use the module-level `Path` for both the primary media path and optional pair path;
- add a direct `QwenVLProvider.attempt()` regression test that reaches message construction without calling the real API.

No compatibility alias or fallback branch is needed.

### Standalone OCR functions

The following top-level functions currently read `self._supports_flex_attn` even though no `self` exists:

- `attempt_deepseek_ocr`;
- `attempt_firered_ocr`;
- `attempt_glm_ocr`;
- `attempt_hunyuan_ocr`;
- `attempt_lighton_ocr`;
- `attempt_logics_ocr`;
- `attempt_nanonets_ocr`;
- both affected OLMOCR loading paths.

Add an explicit keyword-only argument:

```python
supports_flex_attn: bool = False
```

The owning Provider instance passes:

```python
supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False))
```

The standalone function passes that value to `transformerLoader`. Tests and direct callers that omit it retain the current effective default of `False`.

Rules:

- no top-level Provider attempt function may reference `self`;
- do not replace the error with a hard-coded value at each failing line;
- the instance-to-function handoff is the single ownership boundary;
- OLMOCR must use the same argument for both loader creation paths.

## Design 2: Explicit Caption Outcomes

### Result state

Add a string enum to `module/providers/base.py`:

```python
class CaptionStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
```

Extend `CaptionResult` compatibly:

```python
@dataclass
class CaptionResult:
    raw: str
    parsed: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CaptionStatus = CaptionStatus.SUCCESS
    error: Optional[str] = None
```

Add named constructors:

```python
CaptionResult.success(...)
CaptionResult.skipped(reason: str, ...)
CaptionResult.failed(error: str, ...)
```

Add a persistence predicate:

```python
@property
def is_persistable(self) -> bool:
    return self.status is CaptionStatus.SUCCESS and self.has_content
```

`has_content` is semantic, not serialized-size based:

- an unstructured result has content only when `raw.strip()` is non-empty;
- a structured result has content only when at least one recognized content field is non-empty: `description`, `long_description`, `short_description`, `markdown`, `text`, `transcript`, or `translation_srt`;
- `{}`, metadata-only dictionaries, score-only dictionaries, and empty recognized fields do not count as caption content.

Change `CaptionResult.__bool__()` to return `is_persistable`. Diagnostic text carried by a failed result must not make that result truthy at a persistence boundary.

The semantic check protects the database from legacy branches that still construct empty or metadata-only results with the default success status during migration.

### Pipeline behavior

`module/caption_pipeline/orchestrator.py` must construct paired updates from ordered job results. Do not filter the URI and result arrays independently.

Introduce a small persistence record in `module/caption_pipeline/dataset_sync.py`:

```python
@dataclass(frozen=True)
class CaptionUpdate:
    uri: str
    caption: str
```

The orchestrator converts only persistable results into `CaptionUpdate` values. A skipped or failed result:

- advances progress;
- keeps its buffered log output;
- does not write a sidecar;
- does not enter the Lance update batch;
- does not cause the success tag to be created or advanced;
- does not erase an existing caption.

The sidecar write inside `_process_single_caption_job()` must use the same `is_persistable` predicate. Do not rely on object truthiness in one sink and explicit status in another.

`update_dataset_captions()` accepts paired updates as its canonical interface. A narrow legacy adapter may remain temporarily for existing internal tests, but production callers must use `CaptionUpdate`.

### Provider adoption

Convert known non-success branches in the affected path first:

- missing optional/required pair media returns `skipped` with a reason;
- retry exhaustion that is intentionally non-fatal returns `failed` with the final error;
- preprocessing that cannot produce media returns `failed`;
- explicit policy skips return `skipped`.

Exceptions that currently abort the whole run may continue to raise. This design does not force all exceptions into `CaptionResult.failed`.

### Logging

At the end of a batch, report counts for:

- successful and persisted;
- skipped;
- failed without persistence.

Do not print a generic "Successfully updated dataset" message when no updates were persisted.

## Design 3: WDTagger Batch Identity

Both WDTagger preprocessing paths must return the same shape:

```python
tuple[list[str], list[Any]]
```

Change `load_and_preprocess_batch(uris, ...)` to retain each successful `(uri, image)` pair and return `valid_uris, images`, matching `load_siglip2_rgb_batch()`.

Runner rules:

- never set `valid_uris = uris` after preprocessing;
- assert or fail clearly if `len(valid_uris) != len(batch_images)`;
- after inference, fail clearly if the number of probability rows differs from the number of valid URIs;
- zip only `valid_uris` with the corresponding probability rows;
- advance progress by the original input count, including rejected images;
- write no sidecar, JSON entry, or Lance update for a rejected image.

The repair must cover a failed image at the start, middle, and end of a batch. The middle failure is the regression that detects index shifting.

## Design 4: Namespaced See-Through Vendor Imports

The reviewed reversible-context design is rejected. A lock around `sys.modules` cannot prevent unrelated threads from importing a temporarily replaced `utils` package. Restoring the mapping afterward does not revoke wrong module objects already retained by those threads.

The simpler safe design is to make the existing vendor directory a real package namespace. It already contains `__init__.py` files and requires only a bounded import migration.

Canonical vendor paths are:

```text
module.see_through.vendor.modules
module.see_through.vendor.utils
module.see_through.vendor.annotators
```

Required changes:

1. Replace extracted-core vendor imports such as `from modules...` and `from utils.cv...` with fully qualified `module.see_through.vendor...` imports.
2. Keep genuine project imports, especially `utils.transformer_loader`, on the project namespace.
3. Replace absolute imports inside vendored `modules` and `utils` files with package-relative or fully qualified vendor imports.
4. Remove every `ensure_vendor_imports()` call.
5. Delete `module/see_through/vendor_bootstrap.py` after no caller imports it.
6. Update vendor-surface tests to understand canonical vendor paths rather than top-level `utils.*` names.
7. For the currently unbundled optional `annotators.lama_inpainter` path, use the canonical vendor path and raise a clear optional-component error if it is unavailable. Do not reintroduce a top-level `annotators` alias.

Cached LayerDiff and Marigold pipeline objects then retain stable class identity because their modules remain loaded under one canonical name. No `sys.path` or `sys.modules` mutation is permitted anywhere in the See-Through package.

Acceptance checks include:

```python
import utils.path_safety as before
import module.see_through.extracted.layerdiff_core
import module.see_through.extracted.marigold_core
import utils.path_safety as after
assert before is after
```

Also assert that importing and running mocked See-Through phases does not create top-level `modules`, vendor-backed `utils.*`, or `annotators` entries in `sys.modules`.

## Design 5: Per-Run Native Console Resources

`gui/utils/process_runner.py` must allocate resources per invocation, not per GUI process.

Introduce an internal run resource owner:

```python
@dataclass(frozen=True)
class NativeRunFiles:
    directory: Path
    exit_file: Path
    log_file: Path
```

Each `_run_native()` call creates a unique `TemporaryDirectory` using a job/run UUID. The wrapper receives paths inside that directory.

Lifecycle rules:

- the directory is created before the wrapper starts;
- no pre-run deletion of another run's files is necessary;
- the polling loop watches only its own exit file;
- the tail task reads only its own log file;
- cancellation terminates only the current wrapper/process tree;
- GUI completion is driven by the child exit signal, not by the wrapper window closing;
- after the signal appears, the tail task performs its final read and is cancelled and awaited before cleanup;
- temporary files may be deleted after the final tail read even while the wrapper window remains open at its "Press Enter" prompt, because the wrapper no longer consumes them;
- cleanup happens in `finally` for success, failure, startup error, and cancellation without waiting for the user to close the console window;
- the result code is read before cleanup;
- a missing or malformed result file produces an explicit runner error, not a success code.

Two `ProcessRunner` instances must be able to run concurrently in one Python process with different log content and exit codes.

## Design 6: Blob V2-Aware Lance Row Updates

### Why the current merge fails

Both caption persistence and text translation construct partial Arrow tables and call:

```python
dataset.merge_insert(on="uris").when_matched_update_all().execute(table)
```

On a Blob v2 dataset, `update_all` expects a source row compatible with the target schema. A table containing only `uris`, `captions`, and optional `chunk_offsets` does not carry the Blob v2 `data` field and can fail with `Blob struct missing data field`.

### Shared update helper

Create `utils/lance_updates.py` with a focused interface:

```python
@dataclass(frozen=True)
class LanceRowUpdate:
    uri: str
    values: Mapping[str, Any]

def merge_rows_preserving_schema(
    dataset,
    updates: Sequence[LanceRowUpdate],
    *,
    key: str = "uris",
    batch_size: int = 100,
):
    ...
```

Before creating any batches, the helper must reject duplicate update keys across the complete update sequence. Then it builds a lazy sequence of bounded full-schema record batches.

For each bounded record batch, the helper must:

1. scan matching rows with `with_row_id=True` and fail if a URI is missing or non-unique;
2. read all non-blob target values and retain each row's `_rowid`;
3. read actual Blob v2 bytes through `utils.lance_blob.take_blob_files(dataset, ids=row_ids, ...)` and `BlobFile.readall()`;
4. overlay only the fields owned by each `LanceRowUpdate`;
5. rebuild a full Arrow table in exact target-schema order using `build_lance_value_array`;
6. yield the full-schema batch to one `pyarrow.RecordBatchReader`.

After the reader is constructed, execute exactly one:

```python
dataset.merge_insert(on=key).when_matched_update_all().execute(reader)
```

`batch_size` controls source-reader memory, not the number of Lance commits. All batches therefore publish as one dataset version. If lookup, blob reading, Arrow construction, or reader consumption fails midway, no partial version may become latest.

Verify the reported update count when Lance exposes it.

This approach is preferred over SQL-string `dataset.update()` calls because captions and Markdown would require complex SQL escaping and one operation per row. Full-row reconstruction is bounded by `batch_size`, preserves Blob v2, keeps values as typed Arrow data, and commits atomically as one Lance version.

### Error taxonomy and fallback

Define distinct helper errors:

- `LanceUpdateValidationError` for duplicate keys, missing/non-unique target rows, unknown columns, or invalid values;
- `LanceUpdateConflictError` for exhausted optimistic-concurrency retries;
- `LanceUpdateStorageError` for Arrow/Lance read or commit failures.

None of these errors may enter the current catch-all "rebuild from sidecars" fallback. Validation and conflict errors must propagate. Storage errors must also propagate because translation may have no sidecars and because rebuilding can bypass the exact row-identity checks this design adds.

Remove the generic merge-error rebuild path from `update_dataset_captions()`. Preserve the standalone explicit rebuild tool; only the automatic fallback from a failed caption update is removed.

### Caption integration

`module/caption_pipeline/dataset_sync.py` converts each `CaptionUpdate` into:

```python
LanceRowUpdate(uri=update.uri, values={"captions": [update.caption]})
```

No non-persistable result reaches the helper.

### Translation integration

`module/texttranslate.py` supplies:

```python
LanceRowUpdate(
    uri=uri,
    values={
        "captions": [translated_markdown],
        "chunk_offsets": translated_offsets,
    },
)
```

`chunk_offsets` is omitted when the target schema does not contain it. Translation tag creation happens only after all update batches succeed.

### Preservation requirements

Before and after an update:

- blob bytes are identical;
- MIME, hash, duration, metadata, and unrelated columns are identical;
- row count is identical;
- URI values are identical;
- only requested caption/offset fields change;
- an update failure does not create or advance the success tag.

## Feature-Owned Regression Tests

These tests belong to this runtime repair, not to the separate test-infrastructure spec.

### Provider entry points

- Directly call QwenVL message construction with the API call mocked.
- Parameterize the eight OCR entry points with fake transformer classes and an empty loader cache.
- Assert no top-level attempt function contains or executes an implicit `self` reference.
- Run Ruff `F821,F823` on affected Provider files.

### Caption outcomes

- A successful caption creates a sidecar and updates Lance.
- A skipped result preserves an existing sidecar and Lance caption.
- A failed result preserves an existing sidecar and Lance caption.
- A legacy empty-success result is not persistable.
- Mixed concurrent results preserve original job order while persisting only successful pairs.
- A batch with zero successful updates does not move the dataset tag.

### WDTagger

- A corrupt first, middle, and last image never shifts valid URI/probability pairs.
- Rejected images do not receive sidecars or JSON records.
- Progress advances by the original input count.

### Vendor imports

- Project `utils.*` module identity is unchanged after importing and running mocked vendor phases.
- No top-level vendor-backed `modules`, `utils.*`, or `annotators` modules are created.
- Cached pipeline classes retain one canonical module identity.

### Native runner

- Two concurrent runs have different directories and files.
- Their output and exit codes do not cross.
- Cancellation cleans only the cancelled run.
- Startup failure and malformed exit files clean resources and report errors.

### Lance updates

- Update a Blob v2 dataset containing real blob bytes and verify byte equality.
- Update captions through `dataset_sync` without changing unrelated columns.
- Complete the normalize-and-translate round trip in `tests/test_text_translation_pipeline.py`.
- Verify missing and duplicate URIs fail before any dataset version or tag update.
- Inject a failure while consuming the second record batch and verify the latest dataset version is unchanged.
- Verify bounded streaming produces one new dataset version and the same final data as a single batch.

## Implementation Order

1. Add direct Provider entry-point regression tests, then repair QwenVL and OCR ownership.
2. Add `CaptionStatus`, named constructors, and `is_persistable` without changing persistence yet.
3. Add `CaptionUpdate` and filter ordered results at the orchestrator boundary.
4. Add the shared Blob v2-aware Lance update helper and migrate `dataset_sync`.
5. Migrate text translation to the same Lance helper.
6. Repair WDTagger batch identity and add corrupt-image tests.
7. Replace top-level vendor imports with canonical package imports and delete the bootstrap mutation.
8. Replace PID-based native-console files with per-run resources.
9. Run all feature-owned tests, then the full suite after the separate test-infrastructure repair lands.

The Lance helper precedes text translation because issues 4 and 8 require the same safe persistence primitive. Caption status precedes persistence filtering so the sink does not invent Provider semantics.

## Implementation Plan Boundaries

Keep this approved repair scope in one design spec, but do not execute it as one undifferentiated plan. Create independently reviewable plans for:

1. Provider entry points plus caption outcome and Blob v2 persistence, covering issues 3, 4, and 8;
2. WDTagger batch identity, covering issue 5;
3. See-Through vendor import isolation, covering issue 6;
4. native-console per-run resources, covering issue 7.

Each plan must end with its focused regression suite passing. The final integration checkpoint runs all four focused suites together before the repository-wide test/CI plan is considered complete.

## Acceptance Criteria

- QwenVL and all eight affected OCR entry points reach their mocked backend in a fresh process.
- Ruff reports no `F821` or `F823` in the affected Provider files.
- Non-successful or empty caption results never modify sidecars, Lance captions, or success tags.
- WDTagger outputs remain aligned after arbitrary decode failures within a batch.
- See-Through uses only canonical vendor package names and never mutates `sys.path` or project module identities.
- Concurrent native-console jobs use exclusive files and report their own logs and exit codes.
- Caption and translation updates succeed on Blob v2 datasets while preserving blob bytes and unrelated columns.
- Existing successful output formats, CLI flags, configuration names, and dataset tag names remain compatible.
- Review issues 1 and 2 remain untouched.

## Risks and Stop Conditions

### Caption status migration

Risk: a Provider may use an empty result as an undocumented control signal. The persistence predicate must remain conservative: empty output is never data worth overwriting an existing caption. If a genuinely valid empty-caption use case is found, model it as an explicit separate operation rather than weakening this rule.

### Vendor import migration

Risk: copied upstream files may add new absolute imports in a future sync. Keep an AST guard that rejects top-level `modules`, vendor-backed `utils`, and `annotators` imports under `module/see_through/vendor` and `module/see_through/extracted`.

### Blob memory

Risk: reconstructing full Blob v2 rows loads blob bytes for the current batch. Keep the batch bounded and configurable. Do not fall back to loading the entire dataset.

### Concurrent Lance writes

Risk: another writer may update a row between read and merge. Preserve Lance's conflict handling and surface exhausted conflicts. Do not silently retry by rebuilding the whole dataset from stale data.

### Native process cleanup

Risk: deleting the run directory before the tail task's final read loses output, while waiting for the wrapper window to close regresses current UX. The exit signal is the boundary: read the result, finish/cancel-and-await tailing, delete files, and return while the wrapper may continue waiting for Enter.
