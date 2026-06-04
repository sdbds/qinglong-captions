# Lance Blob v2 Caption Merge Fallback Spec

Date: 2026-06-04

## Linus Review Verdict

Root cause diagnosis is good. The original design shape was not.

The bug is not in caption generation. The broken abstraction is treating Lance `merge_insert(...).when_matched_update_all()` as a partial column update. It is not. It replaces matched rows with source rows. A source table containing only `uris` and `captions` is not a complete Lance row, so a target schema containing Blob v2 eventually fails validation with:

```text
Blob struct missing `data` field
```

Rejecting manual Blob v2 row synthesis is also correct. Rehydrating media bytes just to update text captions is a bad cost model and a data-safety trap.

The problem in the previous spec was duplicated design. This repository already has the right sidecar-rebuild pattern in `module/wdtagger/lance_io.py`:

- `resolve_lance_rebuild_source(...)`
- `load_rebuild_data(...)`
- `rebuild_lance_from_sidecars(...)`

The fix should extract or reuse that logic, not grow a second private implementation in `dataset_sync.py`.

Good patch shape:

- Keep the fast merge path.
- On Lance merge failure, call shared sidecar rebuild logic.
- Return the active dataset to the orchestrator.
- Feed the active dataset into `extract_from_lance_fn(...)`.
- Do not add a second rebuild implementation beside `wdtagger/lance_io.py`.

## Local Facts

Relevant files:

- `module/caption_pipeline/dataset_sync.py`
- `module/caption_pipeline/orchestrator.py`
- `module/lanceImport.py`
- `utils/lance_blob.py`
- `utils/lance_utils.py`
- `module/wdtagger/lance_io.py`
- `module/wdtagger/runner.py`
- `utils/output_writer.py`
- `tests/test_caption_pipeline.py`
- `tests/test_caption_result_pipeline.py`

Observed facts:

- `utils/lance_blob.py` maps Lance data storage version `>= 2.2`, `stable`, and `next` to `lance.blob.v2`.
- `module/lanceImport.py::transform2lance()` defaults to `data_storage_version="2.2"`.
- `module/caption_pipeline/dataset_sync.py::update_dataset_captions()` builds a source table with only `uris` and `captions`.
- `update_dataset_captions()` then calls `dataset.merge_insert(on="uris").when_matched_update_all().execute(table)`.
- Local `.venv` has `lance 7.0.0`, `lance.blob_field`, and `lance.blob_array`.
- A minimal local reproduction with a Blob v2 target dataset and a two-column `uris/captions` merge reproduces the same error:

```text
OSError: Invalid user input: Blob struct missing `data` field
```

Repository facts that must be reused:

- `module/wdtagger/lance_io.py::resolve_lance_rebuild_source()` already handles the "input is `.lance` path vs source directory vs opened dataset" decision.
- `module/wdtagger/lance_io.py::load_rebuild_data()` already has a fallback from directory scan to existing dataset `uris`.
- `module/wdtagger/lance_io.py::rebuild_lance_from_sidecars()` already rebuilds Lance from sidecar captions after merge failure.
- `module/wdtagger/runner.py` already treats Lance merge failure as recoverable and falls back to rebuild mode.
- `utils/output_writer.py::write_caption_output()` writes sidecars before `update_dataset_captions()` is called from `orchestrator.py`.
- `utils/output_writer.py` may write `.txt`, `.srt`, or `.md` depending on mime and structured payload.
- `module/lanceImport.py::_find_sidecar_caption()` reads `.txt`, `.md`, and `.srt` sidecars in that order.

## First Principles

The real requirement is:

```text
After caption generation succeeds, persist the generated captions into Lance without corrupting media rows.
```

The source of truth during captioning is split:

- Media metadata and optional blobs live in Lance.
- Generated captions are written as sidecar files before Lance sync.

Therefore the safe recovery path is:

1. Generate captions.
2. Write sidecar captions.
3. Try fast Lance caption merge.
4. If Lance rejects the partial merge, rebuild the Lance dataset from sidecars using shared rebuild logic.

The important simplification is not "write more fallback code". It is "move the existing sidecar-rebuild behavior to one shared place and call it from both wdtagger and caption pipeline".

## Goals

1. Fix caption pipeline crashes caused by Blob v2 target schemas during caption sync.
2. Keep the current fast merge path when it succeeds.
3. Reuse or extract existing `wdtagger/lance_io.py` rebuild logic.
4. Avoid a second sidecar rebuild implementation in `dataset_sync.py`.
5. Return the active dataset to the orchestrator so downstream export uses the updated dataset.
6. Preserve sidecar writes and `CaptionResult` serialization.
7. Keep provider failures fatal; only the Lance sync phase gets a fallback.
8. Add tests for merge success, merge fallback, rebuild-source resolution, sidecar extension coverage, and active dataset propagation.

## Non-Goals

- Do not implement custom Blob v2 row rehydration.
- Do not change Lance import schema defaults.
- Do not downgrade `data_storage_version`.
- Do not remove the `blob` column.
- Do not change provider output serialization.
- Do not change caption sidecar naming.
- Do not catch exceptions around the whole processing loop.
- Do not duplicate `rebuild_lance_from_sidecars()` in `dataset_sync.py`.
- Do not make rebuild preserve every historical import option in this patch.

## Design

### 1. Extract Shared Sidecar Rebuild Logic

Move the generic rebuild pieces out of `module/wdtagger/lance_io.py` into a shared module.

Preferred implementation:

- Create `utils/lance_rebuild.py`, or
- Add to `utils/lance_utils.py` only if top-level imports do not create a circular dependency.

Important circular import constraint:

- `module/lanceImport.py` already imports `update_or_create_tag` from `utils.lance_utils`.
- Therefore `utils.lance_utils` must not top-level import `module.lanceImport`.
- If the shared helpers live in `utils.lance_utils`, they must accept `load_data_fn` and `transform2lance_fn` as callables or use carefully scoped local imports that do not run during `module.lanceImport` initialization.

Shared API shape:

```python
def resolve_lance_rebuild_source(train_data_dir: Any, dataset_path: Path | None = None) -> Path | None:
    ...

def load_lance_rebuild_data(
    source_dir: Path | None,
    dataset: Any,
    *,
    load_data_fn,
    read_sidecar_caption_fn=None,
    caption_extension: str | None = None,
) -> list[dict[str, Any]]:
    ...

def rebuild_lance_from_sidecars(
    source_dir: Path | None,
    *,
    output_name: str,
    dataset: Any,
    tag: str,
    transform2lance_fn,
    load_data_fn,
    console,
    caption_extension: str | None = None,
    read_sidecar_caption_fn=None,
) -> Any | None:
    ...
```

The shared implementation should preserve the existing wdtagger behavior:

- Directory scan first: `load_data_fn(str(source_dir))`.
- If directory scan returns rows, use those rows.
- If directory scan fails or returns no rows, fall back to scanning existing dataset `uris`.
- For wdtagger dataset fallback, use the configured `caption_extension` and `read_sidecar_caption_fn`.
- For caption pipeline, prefer `load_data()` directory scanning because it already detects `.txt`, `.md`, and `.srt` sidecars.

### 2. Update Wdtagger To Use The Shared Helper

`module/wdtagger/lance_io.py` should stop owning the generic rebuild implementation.

It can keep thin wrappers if that minimizes churn:

```python
from utils.lance_rebuild import (
    resolve_lance_rebuild_source,
    rebuild_lance_from_sidecars as _shared_rebuild_lance_from_sidecars,
)

def rebuild_lance_from_sidecars(source_dir, output_name, dataset, caption_extension):
    return _shared_rebuild_lance_from_sidecars(
        source_dir,
        output_name=output_name,
        dataset=dataset,
        tag="WDtagger",
        transform2lance_fn=transform2lance,
        load_data_fn=load_data,
        read_sidecar_caption_fn=read_sidecar_caption,
        caption_extension=caption_extension,
        console=constants.console,
    )
```

This keeps wdtagger behavior stable while removing the duplicate implementation.

### 3. Make Caption Dataset Sync Return The Active Dataset

`update_dataset_captions()` currently returns `None`. Change it to return the dataset that later pipeline stages should use.

Fast path:

```python
active_dataset = update_dataset_captions(...)
extract_from_lance_fn(active_dataset, args.dataset_dir, clip_with_caption=not args.not_clip_with_caption)
```

If merge succeeds, return the original `dataset`.

If fallback rebuild succeeds, return the rebuilt dataset.

If fallback is unavailable or rebuild returns `None`, raise a clear error.

### 4. Add Minimal Fallback Inputs To Caption Sync

Extend `update_dataset_captions()` only enough to call the shared helper:

```python
def update_dataset_captions(
    dataset,
    processed_filepaths,
    results,
    merge_batch_size: int,
    console,
    tag_name: str = "gemini",
    *,
    dataset_dir=None,
    dataset_path=None,
    transform2lance_fn=None,
    load_data_fn=None,
):
    ...
```

`load_data_fn` can default to `module.lanceImport.load_data` only if doing so does not create an import cycle. Passing it from the orchestrator is acceptable and keeps dependencies explicit.

Caption fallback source:

```python
source_dir = resolve_lance_rebuild_source(dataset_dir, dataset_path)
```

For normal caption CLI usage, `dataset_dir` is a filesystem path and this resolves cleanly. If `dataset_dir` is an opened `LanceDataset`, it returns `None` and fallback is unavailable.

### 5. Scope The Fallback To Lance Sync

Wrap only the merge/tag phase, not provider execution.

Preferred shape:

```python
try:
    _merge_caption_batches(dataset, processed_filepaths, processed_captions, merge_batch_size, console)
    update_or_create_tag(dataset, tag_name)
    console.print("[green]Successfully updated dataset with new captions[/green]")
    return dataset
except Exception as merge_error:
    console.print("[yellow]Lance merge_insert failed; rebuilding dataset from sidecar caption files.[/yellow]")
    rebuilt_dataset = rebuild_lance_from_sidecars(...)
    if rebuilt_dataset is None:
        raise RuntimeError("Lance merge_insert failed and fallback rebuild is unavailable") from merge_error
    return rebuilt_dataset
```

Exception-chain responsibility is explicit: `update_dataset_captions()` catches the merge error and raises from it if fallback cannot produce a dataset.

If `rebuild_lance_from_sidecars()` itself raises:

```python
try:
    rebuilt_dataset = rebuild_lance_from_sidecars(...)
except Exception as rebuild_error:
    raise RuntimeError("Lance merge_insert failed and fallback rebuild also failed") from merge_error
```

The raised message should include both facts. The original merge error must remain the root context. The rebuild error should be logged before raising, or attached in the message, because Python exception chaining can only use one direct `from` cause.

### 6. Caption Rebuild Call

Caption pipeline rebuild should call the shared helper with caption-specific inputs:

```python
rebuilt_dataset = rebuild_lance_from_sidecars(
    source_dir,
    output_name="dataset",
    dataset=dataset,
    tag=tag_name,
    transform2lance_fn=transform2lance_fn,
    load_data_fn=load_data_fn,
    console=console,
    caption_extension=None,
)
```

`caption_extension=None` means "use `load_data()` sidecar detection", not "only `.txt`".

This matters because caption pipeline sidecars can be:

- `.txt` for images and many structured text outputs
- `.srt` for audio/video subtitle outputs
- `.md` for application/document outputs

`module/lanceImport.py::_find_sidecar_caption()` already detects `.txt`, `.md`, and `.srt`. The shared helper must not regress that by forcing a single extension for caption pipeline.

### 7. Orchestrator Wiring

Change:

```python
update_dataset_captions(...)
extract_from_lance_fn(dataset, args.dataset_dir, ...)
```

to:

```python
dataset = update_dataset_captions(
    dataset,
    processed_filepaths,
    results,
    merge_batch_size=getattr(args, "merge_batch_size", 1000),
    console=console_obj,
    dataset_dir=args.dataset_dir,
    transform2lance_fn=transform2lance_fn,
    load_data_fn=load_data,
)
extract_from_lance_fn(dataset, args.dataset_dir, clip_with_caption=not args.not_clip_with_caption)
```

If importing `load_data` in `orchestrator.py` is undesirable, pass a small wrapper from `captioner.py`, where `transform2lance` is already imported. The key point is explicit dependency injection, not hidden imports inside `dataset_sync.py`.

## Tests

### Shared Rebuild Tests

Add focused tests for the shared helper.

1. `test_resolve_lance_rebuild_source_accepts_directory`
   - input: source directory string
   - output: same directory path

2. `test_resolve_lance_rebuild_source_accepts_lance_path`
   - input: `dataset.lance`
   - output: parent directory

3. `test_resolve_lance_rebuild_source_rejects_open_dataset_object`
   - input: fake dataset object
   - output: `None`

4. `test_rebuild_data_falls_back_to_existing_dataset_uris_when_directory_scan_empty`
   - fake `load_data_fn` returns `[]`
   - fake dataset scanner yields `uris`
   - fake `read_sidecar_caption_fn` returns captions
   - output data preserves existing row set

5. `test_caption_rebuild_uses_load_data_sidecar_detection_for_txt_md_srt`
   - create three media files with `.txt`, `.md`, and `.srt` sidecars
   - use `module.lanceImport.load_data`
   - assert captions are loaded for all three sidecar extensions

### Caption Sync Tests

1. `test_update_dataset_captions_returns_original_dataset_after_successful_merge`
   - fake merge builder records the table
   - shared rebuild function is not called
   - return value is the original dataset

2. `test_update_dataset_captions_rebuilds_from_sidecars_when_merge_fails`
   - fake merge builder raises `OSError("Blob struct missing `data` field")`
   - fake shared rebuild returns `rebuilt_dataset`
   - return value is `rebuilt_dataset`

3. `test_update_dataset_captions_raises_from_merge_error_when_rebuild_unavailable`
   - fake merge builder raises
   - rebuild source resolves to `None` or shared rebuild returns `None`
   - assert raised `RuntimeError.__cause__` is the merge error

4. `test_update_dataset_captions_logs_rebuild_error_without_losing_merge_context`
   - fake merge builder raises
   - fake shared rebuild raises
   - assert final raised error says rebuild also failed
   - assert `__cause__` is still the merge error

5. `test_process_batch_uses_rebuilt_dataset_for_extract`
   - patch `update_dataset_captions` to return `rebuilt_dataset`
   - assert `extract_from_lance_fn` receives `rebuilt_dataset`, not the original

### Optional Lance Integration Regression

Optional but valuable:

```python
def test_lance_blob_v2_partial_merge_failure_uses_rebuild(tmp_path):
    ...
```

This test can create a tiny Blob v2 Lance dataset with `lance.blob_field()` and verify fallback. Skip it when the local Lance build does not expose Blob v2 APIs.

Unit tests above are required; this integration test is extra.

## Acceptance Criteria

Required:

- Caption generation still writes sidecar files before Lance sync.
- Successful `merge_insert` still works and does not rebuild.
- Blob v2 partial merge failure no longer crashes the caption command when a rebuild source is available.
- Caption pipeline uses the shared sidecar rebuild helper, not a second private implementation.
- Wdtagger still uses the same behavior through the shared helper or a thin wrapper.
- Caption rebuild reads `.txt`, `.md`, and `.srt` sidecars.
- Rebuild can fall back to existing dataset `uris` when directory scan returns no rows.
- `extract_from_lance_fn()` receives the rebuilt dataset after fallback.
- If fallback cannot run, the raised error says merge failed and rebuild is unavailable, with the merge exception as `__cause__`.
- Existing `CaptionResult` serialization behavior remains unchanged.

Targeted tests:

```powershell
pytest tests/test_caption_pipeline.py tests/test_caption_result_pipeline.py -q
```

If a new focused shared helper test file is added:

```powershell
pytest tests/test_lance_rebuild.py tests/test_caption_pipeline.py tests/test_caption_result_pipeline.py -q
```

## Risks

1. Rebuild is slower than merge.
   - Acceptable because it only runs after merge failure.

2. Rebuild can change the row set.
   - Merge only touches `processed_filepaths`. Rebuild runs a directory scan or dataset-uri fallback. If the source directory differs from the original dataset construction source, or if the original dataset used a different `caption_dir`, `include_text_assets`, `import_mode`, or external row source, the rebuilt dataset can gain or lose rows. This fallback is safe only when `dataset_dir` still represents the original source directory. The shared helper's existing dataset-URI fallback reduces this risk when directory scan returns no rows, but it does not reproduce every historical import option.

3. Rebuild uses current sidecar files.
   - This is correct for caption sync because sidecars are written before dataset sync. Stale sidecars in the source tree can still affect rebuild. This is already the import model and should not be changed inside this patch.

4. Rebuild with `save_binary=False` can drop embedded blob bytes from a dataset that previously saved binaries.
   - In the caption pipeline this is usually not a new regression: `_resolve_dataset()` already calls `transform2lance_fn(..., save_binary=False)` when cloud API keys are configured, and the default import path also defaults `save_binary=False`. The risk mainly applies to datasets that were manually created with embedded binaries before caption sync. Preserving embedded blobs would require a separate, heavier design.

5. Catching all merge exceptions can hide non-Blob Lance bugs.
   - The fallback should print/log the original exception and preserve exception context when fallback fails. Since sidecar captions are already written, rebuild remains a valid recovery for Lance write-path failures.

6. Putting shared rebuild helpers into `utils.lance_utils` can create circular imports.
   - Avoid top-level `module.lanceImport` imports from `utils.lance_utils`, or use a new `utils.lance_rebuild` module with dependency injection.

## Decision

Implement方案 1, revised:

- Reuse/extract existing sidecar rebuild logic from `module/wdtagger/lance_io.py`.
- Keep fast merge.
- Fall back to shared sidecar rebuild on merge failure.
- Return the active dataset and pass it to downstream export.

Do not implement Blob v2 manual row rehydration in this patch.
