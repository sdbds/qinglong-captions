# ModelScope Alternative Download Source Design

**Status:** Approved for implementation planning

**Date:** 2026-07-31

**Scope:** Design only. This document does not authorize code changes by itself.

## 1. Decision Summary

qinglong-captions will add ModelScope as an alternative model and dataset download
source for users in China.

The default remains Hugging Face. Users can select ModelScope explicitly. In
ModelScope mode, the application will:

1. Prefer already valid local files.
2. Reuse a valid ModelScope cache entry.
3. Reuse a valid Hugging Face cache entry without modifying it.
4. Download from ModelScope.
5. Validate the downloaded artifact by structure and, where applicable, first
   load.
6. If the ModelScope attempt fails, emit one warning and fall back to Hugging
   Face.
7. If Hugging Face also fails, report both causes in one final error.

The implementation will cover every first-party download path, not only the
shared Transformers loader.

The core architectural decision is to add a project-owned hub adapter in
`utils/model_hub.py`. The project will not implement this feature by changing
`HF_ENDPOINT`, monkeypatching `huggingface_hub`, or mixing ModelScope files into
the Hugging Face cache layout.

## 2. Linus Review

### 2.1 Is this a real problem?

Yes. The application downloads large and numerous artifacts. Hugging Face access
can be unreliable or slow in China, so a second infrastructure provider has
direct user value.

### 2.2 Is there a cheaper solution?

The project already supports `HF_ENDPOINT`, and the Chinese system defaults can
point at a Hugging Face mirror. That is cheaper than a second provider, but it
is not equivalent:

- A mirror still follows the Hugging Face repository and protocol.
- It is not an independent repository, account, token, or availability domain.
- Users specifically need ModelScope-hosted repositories.

Therefore `HF_ENDPOINT` remains supported, but it is not the solution to this
request.

### 2.3 What is the dangerous part?

The dangerous part is not adding another SDK. It is pretending the two providers
share repository identities, revisions, layouts, and caches.

They do not. Hugging Face and ModelScope use different cache structures and may
host different files under similar repository names. Any design that points both
SDKs at `huggingface/hub` risks corrupted metadata, partial snapshots, and errors
that only appear much later during model loading.

The design therefore keeps provider caches separate and treats reuse of the
existing Hugging Face cache as read-only candidate resolution.

## 3. Goals

- Keep Hugging Face as the default source with unchanged behavior.
- Add a user-selectable ModelScope source for China.
- Fall back automatically from ModelScope to Hugging Face after one warning.
- Cover all first-party model, dataset, template, metadata, and auxiliary-file
  downloads.
- Reuse already valid Hugging Face cache entries while ModelScope mode is active.
- Keep ModelScope and Hugging Face cache writes isolated.
- Preserve all existing canonical model IDs and caller configuration.
- Preserve explicit user-provided local model directories.
- Make source decisions observable without leaking credentials.
- Make the design testable without network access in normal CI.

## 4. Non-Goals

- Automatically detecting the user's country, IP address, or network quality.
- Replacing or removing `HF_ENDPOINT`, `HF_HOME`, or `HF_TOKEN`.
- Migrating, copying, hardlinking, or deduplicating old cache content.
- Making ModelScope write into Hugging Face `blobs/` or `snapshots/`.
- Renaming existing `model_id`, `repo_id`, or configuration values.
- Modifying vendored code under `third_party/**` or
  `module/see_through/vendor/**`.
- Rewriting every provider that already routes through the shared loader.
- Mass-editing PowerShell launch scripts.
- Changing cloud API providers or their network behavior.
- Guaranteeing fallback for arbitrary network access started inside third-party
  `trust_remote_code`.
- Hot-switching a model that is already loaded in a running task.

## 5. Existing Download Surface

The project does not have one download path today. The implementation must
centralize the following first-party families.

| Family | Current integration point | Required migration |
| --- | --- | --- |
| Transformers OCR, VLM, ALM, and LLM providers | `utils/transformer_loader.py` | Source-aware component and snapshot loading |
| See-Through models | Outer loaders plus shared transformer loader | Resolve local snapshots outside vendored code |
| ONNX single files and file sets | `module/onnx_runtime/artifacts.py` | Source-neutral file, list, snapshot, and external-data resolution |
| Water detection | Shared ONNX runtime | Covered automatically |
| Paddle OCR ONNX | Shared ONNX runtime | Covered automatically |
| LFM-VL ONNX | ONNX artifact helper | Covered automatically |
| MuSViT sheet-music OMR | Shared multi-model ONNX runtime | Covered automatically |
| Vocal MIDI | Shared multi-model ONNX runtime | Covered automatically |
| Audio separation model | Shared ONNX runtime plus direct metadata file | Migrate direct metadata download |
| WD Tagger | Shared ONNX runtime plus direct labels and hierarchy files | Migrate direct files and datasets |
| Dots OCR snapshot | `snapshot_download_with_reporting` | Preserve interface, change implementation |
| SigLIP2 tagger snapshot and processor | Shared snapshot helper plus direct `from_pretrained` | Make both source-aware |
| Gemma 4 chat template | Direct `hf_hub_download` | Route through adapter |
| Reward model | Direct `from_pretrained` | Route through source-aware pretrained loading |
| Mega-ASR | Shared snapshot helper plus direct model load | Return and load the actual resolved directory |
| MuScriptor | Upstream implicit Hugging Face loading | Resolve required files locally before calling upstream |
| Tag-highlighting metadata | Direct `hf_hub_download` | Route through adapter |

The inventory is deliberately based on first-party ownership rather than file
extension. A JSON label map and a Jinja chat template are part of the same
reliability problem as a multi-gigabyte weight file.

## 6. Design Principles

### 6.1 Canonical identity stays provider-neutral

Every existing repository ID remains the canonical project ID. Today that is
normally the Hugging Face ID, for example `owner/repository`.

Callers continue to pass that ID. They do not select provider-specific IDs and do
not contain ModelScope conditionals.

The adapter maps the canonical ID to a provider-specific ID only at the provider
boundary.

### 6.2 Source selection controls network order, not local usability

A valid local artifact is more valuable than a configured network preference.
Selecting ModelScope must not force a redownload when the exact required
artifact already exists in the Hugging Face cache.

This is read-only reuse. It does not make the cache formats compatible.

### 6.3 A source attempt is a transaction

A source attempt includes more than an SDK call. It includes:

- repository resolution;
- file or snapshot download;
- required-file validation;
- compound-artifact validation;
- and, for pretrained components, the first load.

Fallback happens when that complete attempt fails. A successful HTTP download
followed by an unusable model is not a successful source attempt.

### 6.4 Default mode must remain boring

When `MODEL_DOWNLOAD_SOURCE=huggingface`:

- ModelScope is not imported.
- No ModelScope cache is scanned.
- Existing Hugging Face arguments, endpoint behavior, authentication, progress,
  and exceptions remain unchanged unless wrapping is required to preserve the
  public helper interface.
- There is no new fallback in the opposite direction.

This protects the majority path and limits regression risk.

## 7. Configuration Contract

### 7.1 Environment variables

The following variables are added:

| Variable | Values | Default | Purpose |
| --- | --- | --- | --- |
| `MODEL_DOWNLOAD_SOURCE` | `huggingface`, `modelscope` | `huggingface` | Selects preferred network source |
| `MODELSCOPE_CACHE` | Filesystem path | `modelscope` relative to project root | Isolated ModelScope cache root |
| `MODELSCOPE_API_TOKEN` | Optional string | Empty | ModelScope authentication |
| `MODELSCOPE_ENDPOINT` | URL | ModelScope SDK default, currently `https://modelscope.cn` | Advanced endpoint override |

`MODEL_DOWNLOAD_SOURCE` is an enum, not a boolean. This avoids a configuration
name that becomes wrong when a third source is ever added.

An invalid value is a configuration error and must fail immediately with the
accepted values. It must not silently fall back to Hugging Face.

The following existing variables remain unchanged:

- `HF_HOME`
- `HF_ENDPOINT`
- `HF_TOKEN`

When ModelScope falls back to Hugging Face, the Hugging Face attempt uses the
existing Hugging Face configuration, including a configured mirror endpoint.

There is no separate fallback toggle in this design. ModelScope-to-Hugging-Face
fallback is fixed product behavior.

### 7.2 GUI

The environment configuration UI adds:

- a source dropdown for `MODEL_DOWNLOAD_SOURCE`;
- a path field for `MODELSCOPE_CACHE`;
- a secret field for `MODELSCOPE_API_TOKEN`;
- an advanced field for `MODELSCOPE_ENDPOINT`.

The source selector must be a dropdown with explicit labels, not a checkbox.

The GUI persists the values using the existing environment configuration path
and injects them into newly started tasks. Changing the value does not unload or
replace models already resident in memory.

### 7.3 Repository mapping file

Add `config/model_hub.toml` and merge it through `config/loader.py`.

The file maps canonical IDs to ModelScope IDs separately for models and
datasets:

```toml
[modelscope.models."canonical/huggingface-id"]
repo_id = "modelscope/actual-id"
revision = "master"

[modelscope.datasets."canonical/huggingface-dataset-id"]
repo_id = "modelscope/actual-dataset-id"
revision = "master"
```

Rules:

- Model and dataset namespaces are isolated.
- If no mapping exists, ModelScope uses the same repository ID.
- If the caller supplies a revision, ModelScope tries that revision unless the
  mapping explicitly overrides it.
- The mapping may change repository ID and revision, but not arbitrary filenames
  in the first version.
- Hugging Face always uses the original canonical ID and caller revision.
- Local filesystem paths are never mapped.
- Duplicate or malformed entries fail configuration validation.
- Tokens and endpoints do not belong in this file.

## 8. Hub Adapter

### 8.1 Ownership

Add `utils/model_hub.py` as the only first-party module allowed to perform direct
provider download and listing operations.

It owns:

- source enum parsing;
- mapping lookup;
- provider-specific repository resolution;
- token and endpoint separation;
- local cache candidate resolution;
- file, file-set, and snapshot download;
- provider listing operations;
- per-attempt diagnostics;
- warning deduplication;
- aggregate download errors;
- orchestration of a caller-supplied first-consumption validator;
- staging and promotion for explicit targets and compound artifacts.

It does not own:

- model-specific load kwargs;
- Transformers class selection;
- ONNX session construction;
- application task lifetime;
- GUI persistence;
- third-party code that independently accesses the network.

### 8.2 Public contract

Exact private names may change during implementation, but the module must expose
one source-neutral operation for each required primitive:

```python
download_repo_file(...)
download_repo_file_set(...)
download_repo_snapshot(...)
list_repo_files(...)
resolve_local_snapshot(...)
run_with_source_fallback(..., consume_candidate=...)
```

All primitives accept a canonical repository ID and explicit repository type.
Relevant operations also accept:

- revision;
- filename or file patterns;
- cache directory;
- explicit destination directory;
- force-download flag;
- provider-independent logging context;
- test-injected provider functions.

Results carry enough metadata for callers and tests to observe:

- actual local path;
- provider used;
- provider repository ID;
- revision used;
- whether the result came from local cache;
- whether fallback occurred.

Compatibility wrappers may return only a path where an existing public helper
already promises a string. Provider metadata remains available through the
operation context and structured logs.

`run_with_source_fallback` is the generic transaction boundary. The adapter
resolves one coherent local candidate, then calls a consumer-owned callback. The
callback may construct a Transformers component, ONNX session, MuScriptor model,
or another first runtime object. A callback failure is attached to that provider
attempt and can trigger the next source.

The adapter controls attempt ordering, warning deduplication, and aggregate
errors. The callback controls domain-specific loading and validation. This keeps
provider policy centralized without teaching `utils/model_hub.py` how to build
every model type.

### 8.3 Lazy imports

The ModelScope package is imported only inside a ModelScope provider branch.

Importing project modules, starting in default Hugging Face mode, and loading
from an explicit local path must not import ModelScope.

This is both a regression guard and a testable acceptance condition.

### 8.4 Authentication isolation

- Hugging Face receives only its existing token arguments or `HF_TOKEN`.
- ModelScope receives only `MODELSCOPE_API_TOKEN`.
- No provider token is forwarded to the other provider.
- Logs and exceptions redact tokens, authorization headers, and signed URL query
  parameters.

## 9. Resolution State Machines

### 9.1 Explicit local repository path

If a caller's repository argument is an existing local filesystem path, it is a
terminal local choice:

- no mapping is applied;
- no network provider is selected;
- no cache is rewritten;
- load errors are reported as local load errors rather than hidden by a network
  fallback.

This preserves user intent.

An explicit download destination is different from a local repository argument.
Downloads to an explicit destination use staging as described later.

### 9.2 Hugging Face mode

For `MODEL_DOWNLOAD_SOURCE=huggingface`, preserve the current Hugging Face flow:

```text
explicit local repository
    -> existing Hugging Face cache/network behavior
    -> success or existing Hugging Face error
```

The adapter must not:

- probe ModelScope;
- import ModelScope;
- translate repository IDs;
- or fall back to ModelScope.

### 9.3 ModelScope mode

For `MODEL_DOWNLOAD_SOURCE=modelscope`, use this order:

```text
explicit local repository
    -> valid ModelScope local cache
    -> valid Hugging Face local-only cache
    -> ModelScope network download
    -> required-file and first-load validation
    -> one warning on failure
    -> Hugging Face network download/load
    -> aggregate error if both providers fail
```

Important details:

- ModelScope cache lookup uses the mapped ModelScope repository identity.
- Hugging Face cache lookup uses the original canonical identity.
- Hugging Face cache probing is strictly local-only.
- A Hugging Face cache hit prevents both ModelScope and Hugging Face network
  requests.
- A ModelScope network success prevents a Hugging Face network request.
- The fallback Hugging Face attempt is made once per logical artifact request.
- A logical file set or model load is not allowed to mix files from both network
  sources.

### 9.4 Force download

`force_download=True` bypasses both provider caches:

```text
explicit local repository, when explicitly supplied
    -> ModelScope network
    -> warning on failure
    -> Hugging Face network
```

Force download does not delete, mutate, or invalidate old cache entries. It only
prevents them from satisfying the current request.

### 9.5 Non-fallback failures

Fallback is for source-specific availability or artifact failures, including:

- repository or revision not found;
- authentication failure;
- timeout or transport failure;
- missing required files;
- invalid or inconsistent file sets;
- checksum or validation failure;
- first pretrained load failure;
- unavailable or broken ModelScope provider import.

The adapter must not retry another provider for process-wide or local-machine
failures that the second provider cannot fix:

- user cancellation;
- `KeyboardInterrupt`;
- `SystemExit`;
- local permission denial for the destination;
- disk exhaustion;
- an invalid explicit local repository path supplied as the model itself.

These errors propagate immediately.

## 10. Cache Semantics

### 10.1 Cache roots stay separate

Hugging Face keeps using its existing cache, currently rooted through `HF_HOME`
and commonly stored under `huggingface/hub`.

ModelScope uses `MODELSCOPE_CACHE`, defaulting to project-relative
`modelscope/`.

Add `modelscope/` to `.gitignore`.

The implementation must never configure the ModelScope SDK to use
`huggingface/hub`.

### 10.2 Hugging Face cache reuse in ModelScope mode

The ModelScope SDK cannot natively reuse the Hugging Face cache because the
providers use incompatible directory and metadata layouts.

The project adapter can still reuse an already valid Hugging Face artifact:

- use supported Hugging Face local-only APIs rather than manually editing cache
  metadata;
- resolve the canonical repository and revision;
- verify the exact required file, file set, or loadable snapshot;
- pass the resolved path directly to the consumer;
- do not copy, hardlink, promote, delete, or relabel the Hugging Face entry.

This is a read-only shortcut, not shared-cache support.

The project also has legacy cache-like directories outside the formal
Hugging Face Hub layout, including shipped defaults such as `huggingface/`,
`huggingface/Mega-ASR`, and the MuSViT model directory. In ModelScope mode:

- an already valid legacy artifact remains a read-only local candidate;
- a shipped legacy `huggingface` default is not used as a ModelScope download
  destination;
- a new ModelScope artifact goes to `MODELSCOPE_CACHE`;
- an explicitly configured non-default user directory retains its current
  meaning and uses staged promotion when downloads are allowed there.

`module/onnx_runtime/config.py` must distinguish these shipped legacy defaults
from explicit custom destinations. Otherwise selecting ModelScope would still
write new files below a directory named `huggingface`, violating cache
isolation.

### 10.3 Validity is operation-specific

A directory existing is not enough to call it a cache hit.

- A single-file request requires the requested file.
- A known file set requires every file in the set.
- An ONNX artifact requires the graph and all discovered or declared external
  data files.
- A snapshot with allow-patterns requires every resulting required artifact.
- A Transformers component must complete its first
  `from_pretrained(local_path, local_files_only=True)` load.
- An ONNX bundle must construct its first session and satisfy the bundle's
  existing input/output validation.
- MuScriptor requires both its weight and configuration artifacts and a
  successful upstream local-weight load.
- Mega-ASR requires a successful first load from the resolved local directory.

An incomplete entry is skipped or rejected. It is never completed with files
from the other network provider inside the same logical result.

### 10.4 Staging and promotion

Downloads that target an explicit directory or assemble multiple related files
must use a source-specific staging directory.

The sequence is:

1. Create a unique staging directory under the destination's filesystem.
2. Download all files from one provider.
3. Validate the complete artifact.
4. Acquire the existing or adapter-owned destination lock.
5. Promote by atomic rename where supported.
6. Release the lock.
7. Remove adapter-owned staging data on failure using best effort.

The visible destination must therefore contain either the previous valid
artifact or the complete new artifact, never a half-downloaded mixture.

Provider-managed cache locking remains the provider SDK's responsibility. The
adapter adds locking only around project-owned staging and promotion.

## 11. Pretrained Component Loading

`utils/transformer_loader.py` remains the shared owner of Transformers-specific
loading.

### 11.1 Hugging Face mode

`load_pretrained_component` and `transformerLoader` preserve their current
Hugging Face call shapes and kwargs. This includes existing behavior for:

- `trust_remote_code`;
- dtype and device selection;
- tokenizer and processor classes;
- revisions and subfolders;
- local files;
- progress reporting;
- injected test doubles.

### 11.2 ModelScope mode

For a ModelScope attempt:

1. Resolve or download a complete ModelScope snapshot through the adapter.
2. Call the component's `from_pretrained` with the resolved local path.
3. Force the component load to remain local for that attempt.
4. Preserve all compatible caller load kwargs.
5. Treat a load failure as a failed ModelScope attempt.
6. Warn once and run the existing Hugging Face load against the canonical ID.

Using a local path for the ModelScope attempt is essential. Passing a ModelScope
ID directly to Transformers would silently route network behavior back through
Hugging Face and defeat source isolation.

### 11.3 Fallback scope

The warning and fallback context is shared across one logical component or model
request. A model with many files must not print one warning per failed shard.

Where a provider load has separate tokenizer, processor, config, and model
components, the implementation should reuse the resolved snapshot and attempt
context rather than redownload it.

### 11.4 Existing snapshot helper

Keep the public `snapshot_download_with_reporting` interface so existing callers
do not need provider logic. Internally it delegates to the adapter.

Its Hugging Face mode retains current Rich reporting. Its ModelScope mode uses a
provider progress callback or equivalent adapter reporting, with no duplicate
simultaneous progress bars.

## 12. ONNX and Compound Artifacts

`module/onnx_runtime/artifacts.py` becomes source-neutral.

It must support:

- one repository file;
- a known set of repository files;
- one ONNX graph;
- an ONNX graph plus external tensor data;
- multiple ONNX components;
- repository listing where external files must be discovered;
- support files such as vocabularies, labels, and JSON metadata.

The shared ONNX callers then inherit ModelScope support automatically:

- `module/waterdetect.py`
- `module/providers/ocr/paddle.py`
- `module/providers/local_vlm/lfm_vl_local.py`
- `module/sheet_music_omr/model.py`
- `module/vocal_midi.py`
- `module/audio_separator_core.py`
- `module/wdtagger/model_loader.py`

An ONNX artifact set is one transaction. If ModelScope provides the graph but
not one external-data file, the ModelScope attempt fails and the complete set is
retried from Hugging Face. The adapter does not combine the ModelScope graph with
a Hugging Face tensor file.

The shared single-model and multi-model bundle loaders participate in
`run_with_source_fallback`. Their first ONNX session construction and existing
model-contract validation are the consumer callback. If a complete-looking
ModelScope bundle cannot create a valid session, the full Hugging Face bundle is
retried before the final error.

Existing dependency-injection hooks for downloaders and repository listers must
remain usable in tests.

## 13. Direct First-Party Outliers

The following paths need explicit migration because the shared loader or ONNX
runtime does not fully cover them.

### 13.1 Audio separator metadata

`module/audio_separator_core.py` routes direct metadata downloads through the
adapter. Its ONNX model remains covered by the shared runtime.

### 13.2 WD Tagger

`module/wdtagger/model_loader.py`, `utils/wdtagger.py`, and related helpers route
model, labels, parent maps, and dataset-derived metadata through the adapter.
Existing injectable downloader contracts remain available or receive a
source-neutral equivalent.

### 13.3 Tag highlighting

`utils/tag_highlighting.py` uses the adapter for its auxiliary metadata file.

### 13.4 Gemma 4 template

`module/providers/local_vlm/gemma4_local.py` uses the adapter for
`chat_template.jinja`.

### 13.5 Reward model

`module/rewardmodel.py` uses the shared source-aware pretrained path rather than
calling a provider-bound `from_pretrained` directly.

### 13.6 SigLIP2 WD Tagger

`utils/wdtagger_siglip2.py` uses the source-aware snapshot and processor paths.
A processor load failure is part of the source attempt and can trigger fallback.

### 13.7 Mega-ASR

`module/providers/local_alm/mega_asr_local.py` must load from the actual directory
returned by source resolution.

Its first `Qwen3ASRModel` load is the consumer callback for the source
transaction. A ModelScope snapshot that downloads but cannot load is rejected
before Hugging Face fallback.

The existing `huggingface/Mega-ASR` directory remains a legacy local-cache
candidate. It is not renamed or moved.

### 13.8 MuScriptor

`module/muscriptor_tool/runtime.py` must stop relying on the upstream package to
implicitly download from Hugging Face.

The project resolves at least:

- `model.safetensors`;
- `config.json`.

It then calls the installed upstream `TranscriptionModel.load_model` with the
local weight path. The inspected upstream API accepts a local
`weights_path`, so no upstream patch or vendored edit is required.

ModelScope and Hugging Face attempts must each supply a coherent local pair.
The upstream first load is the consumer callback, so an unusable ModelScope pair
triggers Hugging Face fallback.

### 13.9 See-Through

Do not edit `module/see_through/vendor/**`. The project's outer loader resolves
the source to a local snapshot before invoking vendored code where required.

## 14. Dependency and Packaging

Add the official lightweight SDK as a normal runtime dependency:

```toml
modelscope-hub==0.1.8
```

Rationale:

- The feature is a supported user-facing mode, not a development extra.
- An exact pin prevents an alpha SDK update from silently changing behavior.
- The lightweight package avoids the broader dependency surface of the older
  all-in-one ModelScope package.
- Lazy import keeps the default Hugging Face startup path isolated.

The package is currently marked Alpha. Upgrades require explicit compatibility
testing of cache lookup, snapshot download, file download, listing, callbacks,
token handling, and endpoint override before changing the pin.

## 15. Diagnostics, Warnings, and Errors

### 15.1 Startup diagnostics

At the existing configuration diagnostic point, report:

- selected download source;
- resolved ModelScope cache path when relevant;
- whether a custom ModelScope endpoint is configured;
- whether a ModelScope token is present, as a boolean only.

Never print the token value.

### 15.2 Fallback warning

Emit exactly one warning per logical request when a ModelScope source attempt is
abandoned:

```text
ModelScope download/load failed for <canonical-id> at <stage>: <safe reason>.
Falling back to Hugging Face.
```

The warning records enough context to debug the failure without dumping raw
provider response bodies that may contain credentials.

### 15.3 Aggregate failure

If both providers fail, raise one project-owned error with:

- canonical repository ID and repository type;
- requested revision;
- ModelScope mapped ID and revision;
- ModelScope failure stage and sanitized cause;
- Hugging Face failure stage and sanitized cause;
- cache paths that were checked, without credentials.

Preserve each original exception as structured attempt data or exception
chaining. Do not flatten the result into "download failed."

### 15.4 Warning deduplication

Deduplication is scoped to an operation context, not global process state. A
later independent task must still receive its own warning.

## 16. Security

- Treat both provider tokens as secrets.
- Never include tokens in TOML mapping files.
- Redact authorization headers and signed URL parameters.
- Do not send Hugging Face credentials to ModelScope or the reverse.
- Resolve staging and destination paths before promotion and keep them inside
  the intended cache or explicit target.
- Do not execute downloaded code during structural validation.
- Existing `trust_remote_code` behavior remains an explicit caller choice.
- Loading remote code from a local ModelScope snapshot does not guarantee that
  the remote code itself will avoid unrelated network requests.

## 17. Testing Strategy

Normal CI uses provider fakes and temporary directories. It must not depend on
either public network.

### 17.1 Configuration tests

- Missing `MODEL_DOWNLOAD_SOURCE` selects Hugging Face.
- Both accepted enum values parse.
- Invalid and empty non-default values fail immediately.
- Project-relative `MODELSCOPE_CACHE` resolves consistently.
- GUI values persist and are injected into a new task.
- Changing the GUI source does not claim to hot-switch loaded models.

### 17.2 Mapping tests

- Same-ID fallback when no mapping exists.
- Different ModelScope model ID.
- Different ModelScope dataset ID.
- Model and dataset namespaces do not collide.
- Caller revision passes through when not overridden.
- Mapping revision overrides only the ModelScope attempt.
- Hugging Face fallback retains canonical ID and original revision.
- Local filesystem paths bypass mapping.

### 17.3 Cache tests

- Valid ModelScope cache hit performs no network request.
- ModelScope miss plus valid Hugging Face local cache performs no network
  request.
- Invalid or partial ModelScope entry is rejected.
- Invalid or partial Hugging Face local entry is rejected.
- Hugging Face cache reuse does not modify files or metadata.
- ModelScope download writes nothing under the Hugging Face cache.
- `force_download` bypasses both caches without deleting them.
- Existing legacy `huggingface/Mega-ASR` remains usable.

### 17.4 Download primitive tests

- Single file.
- Known file set.
- Snapshot.
- Allow and ignore patterns.
- Repository listing.
- Model versus dataset repository type.
- ONNX graph without external data.
- ONNX graph with one or more external-data files.
- Multi-component ONNX bundle.
- Explicit target staging and atomic promotion.
- Concurrent promotion does not expose a partial destination.

### 17.5 Fallback tests

For ModelScope, inject each of:

- not found;
- timeout;
- authentication error;
- missing required file;
- checksum or structural validation failure;
- pretrained first-load failure;
- ONNX session-construction or model-contract failure;
- MuScriptor or Mega-ASR first-load failure.

For every case, verify:

- one warning;
- one Hugging Face fallback attempt;
- no mixed-source artifact;
- no token in logs;
- success returns Hugging Face provider metadata.

When Hugging Face also fails, verify that the final error contains both sanitized
causes and the correct stages.

Verify that cancellation, disk exhaustion, and destination permission errors do
not trigger a pointless provider fallback.

### 17.6 Special-path tests

Add focused coverage for:

- audio separator metadata;
- WD Tagger labels and hierarchy data;
- tag-highlighting metadata;
- Gemma 4 template;
- reward model;
- SigLIP2 processor and snapshot;
- Mega-ASR resolved directory;
- MuScriptor local weight and config pair;
- See-Through outer-loader local resolution.

### 17.7 Default-mode regression tests

With no new configuration:

- Hugging Face remains selected.
- ModelScope is not imported.
- ModelScope cache is not touched.
- Existing injected Hugging Face download functions still work.
- Existing Hugging Face progress and error behavior remains valid.
- The existing test suite passes without ModelScope-specific setup.

### 17.8 Architecture enforcement

Add a test that scans first-party Python code and rejects new direct uses of:

- `hf_hub_download`;
- `snapshot_download`;
- `list_repo_files`;
- direct imports or calls into the ModelScope SDK.

Allowed locations are:

- `utils/model_hub.py`;
- the narrowly scoped Hugging Face progress bridge in
  `utils/transformer_loader.py`, if still required.

Exclude:

- `third_party/**`;
- `module/see_through/vendor/**`;
- tests and fixtures.

The scan is a guardrail, not a substitute for runtime tests. Direct
`from_pretrained` calls also require review, but some local-only uses are valid
and cannot be rejected by a simple string scan.

### 17.9 Optional live smoke test

Keep live-network coverage opt-in and download only a small configuration file:

```powershell
$env:RUN_MODELSCOPE_SMOKE = "1"
pytest tests/test_model_hub_smoke.py -q
```

The smoke test verifies repository mapping, authentication-free download,
provider metadata, and local cache reuse. It is not part of default CI.

## 18. Acceptance Criteria

Implementation is accepted only when all of the following hold:

1. With no new configuration, the complete existing test suite behaves as
   before and Hugging Face remains the only imported network provider.
2. In ModelScope mode, a successful ModelScope attempt causes no Hugging Face
   network request.
3. In ModelScope mode, a valid Hugging Face cache entry is reused without any
   network request.
4. A failed ModelScope download, validation, or first load emits one warning and
   performs one Hugging Face fallback.
5. If both providers fail, the final error contains both sanitized causes.
6. ModelScope never writes into the Hugging Face cache.
7. Existing model IDs, CLI arguments, and configuration values remain valid.
8. The GUI applies source changes to the next task and does not claim to replace
   already loaded models.
9. Every first-party direct download path is routed through the adapter or a
   shared source-aware loader.
10. `modelscope-hub==0.1.8` is pinned exactly and is imported lazily.

## 19. Risks and Mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Hugging Face revision such as `main` does not exist on ModelScope | ModelScope lookup fails | Mapping-level revision override, then Hugging Face fallback |
| Same repository ID has different contents | Download succeeds but load fails | Required-file and first-load validation |
| ModelScope repository is absent | Preferred source fails | Same-ID attempt, mapping support, one warned fallback |
| Alpha SDK behavior changes | Cache or API regression | Exact dependency pin and provider-contract tests |
| Provider tokens are confused | Authentication leak or failure | Strict token isolation and redaction |
| Partial multi-file download | Late runtime failure | Single-source transaction, staging, validation, atomic promotion |
| Existing HF cache is parsed incorrectly | False cache hit | Use supported local-only HF APIs and operation-specific validation |
| Vendored loader performs its own network request | Source policy bypass | Resolve local path in outer loader; document residual `trust_remote_code` risk |
| Feature spreads new conditionals across providers | Maintenance cost | One hub adapter plus existing shared loaders |
| Fallback hides persistent ModelScope problems | Silent degradation | Mandatory warning and provider metadata |

## 20. Implementation Boundary

### 20.1 Expected first-party touchpoints

The implementation plan must verify the exact diff, but code inspection already
identifies these ownership points:

| Area | Expected files |
| --- | --- |
| New provider boundary | `utils/model_hub.py` |
| Mapping configuration | `config/model_hub.toml`, `config/loader.py` |
| Shared pretrained loading | `utils/transformer_loader.py` |
| Shared ONNX download and first-load transaction | `module/onnx_runtime/artifacts.py`, `module/onnx_runtime/single_model.py`, `module/onnx_runtime/multi_model.py`, `module/onnx_runtime/config.py` |
| Direct auxiliary downloads | `module/audio_separator_core.py`, `module/wdtagger/model_loader.py`, `utils/wdtagger.py`, `utils/tag_highlighting.py`, `module/providers/local_vlm/gemma4_local.py` |
| Direct pretrained outliers | `module/rewardmodel.py`, `utils/wdtagger_siglip2.py`, `module/providers/local_alm/mega_asr_local.py`, `module/muscriptor_tool/runtime.py` |
| GUI configuration and launch diagnostics | `gui/utils/env_config.py`, `gui/wizard/step7_settings.py`, `gui/launch.py` |
| Packaging and cache ignore | `pyproject.toml`, `.gitignore` |
| Tests | focused adapter, loader, ONNX, GUI, outlier, architecture, and optional smoke-test modules |

Providers already covered by `transformerLoader`,
`snapshot_download_with_reporting`, or the shared ONNX runtime should not receive
mechanical source-selection conditionals.

Existing `config/model.toml` repository IDs remain unchanged. Any shipped
`huggingface` directory default is handled as a legacy local candidate by the
source-aware runtime rather than being blindly used as the ModelScope
destination.

### 20.2 Approved invariants

The later implementation plan may change private helper names, but it must not
weaken these approved invariants:

- Hugging Face is the unchanged default.
- ModelScope selection is explicit.
- ModelScope failure falls back to Hugging Face after one warning.
- Every first-party download path is covered.
- Canonical IDs remain unchanged.
- Local valid artifacts win.
- Hugging Face cache reuse is read-only.
- Provider cache writes remain separate.
- A logical artifact never mixes network sources.
- Fallback includes validation and first load.
- Both-provider failures preserve both causes.
- Vendored code is not edited.
- No region auto-detection or cache migration is introduced.

This design document is the input to a separate implementation plan. The plan
must enumerate concrete edits and tests before production code changes begin.

## 21. References

- [ModelScope Hub SDK repository](https://github.com/modelscope/modelscope_hub)
- [modelscope-hub 0.1.8 on PyPI](https://pypi.org/project/modelscope-hub/)
- [Hugging Face cache-system reference](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)
