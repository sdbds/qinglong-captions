# ModelScope Alternative Download Source Design

**Status:** Revised after independent review; pending implementation-plan approval

**Date:** 2026-07-31

**Revision:** 2

**Scope:** Design only. No production-code change is authorized by this document.

## 1. Decision Summary

qinglong-captions will support ModelScope as an alternative artifact source for
users in China.

Hugging Face remains the default. When ModelScope is selected, the application
will prefer a complete local model bundle, then try ModelScope, and automatically
fall back to a complete Hugging Face bundle after one warning.

The unit of resolution, validation, fallback, and in-memory cache commit is a
logical bundle, not a file and not an individual Transformers component.

A bundle may contain:

- one or more repositories;
- model weights;
- tokenizer, processor, or feature-extractor assets;
- remote-code Python files;
- ONNX graphs and external tensor files;
- label maps, templates, and metadata;
- dataset files;
- and every runtime object required for the first usable load.

The central invariant is:

> One automatically resolved bundle uses one provider. ModelScope and Hugging
> Face artifacts are never combined inside the same bundle.

For example, a ModelScope processor cannot remain cached when its ModelScope
model fails and the model falls back to Hugging Face. The entire tentative
ModelScope runtime is discarded and the entire bundle is retried from Hugging
Face.

## 2. Linus Review

### 2.1 Is this a real problem?

Yes. Large model downloads through Hugging Face are often slow or unreliable in
China. An independent ModelScope path has direct user value.

### 2.2 Is there a simpler solution?

`HF_ENDPOINT` and a Hugging Face mirror remain cheaper and continue to work.
They are not an independent repository, account, token, or availability domain,
so they do not satisfy the requirement.

The smallest acceptable design is:

1. one project-owned provider adapter;
2. one bundle data structure;
3. one source state machine;
4. one immutable local result per resolved bundle;
5. consumer-specific validation through local paths only.

A generic callback around individual downloads is not sufficient because it
cannot stop processor/model, graph/external-data, or model/metadata mixing.

### 2.3 What can this break?

The main compatibility risks are:

- successful Hugging Face behavior changes while adding the default-disabled
  provider;
- partial component caches survive a failed source attempt;
- upstream wrappers silently start their own Hugging Face downloads;
- explicit revisions are rewritten to unrelated ModelScope branches;
- ModelScope files are written into an existing Hugging Face directory;
- standalone PEP 723 scripts do not receive the new dependency;
- a local-machine failure is misclassified as a provider failure;
- Windows exposes a partially replaced model directory.

This revision makes each risk an explicit invariant and test target.

## 3. Required Invariants

The implementation must preserve all of these conditions:

1. `MODEL_DOWNLOAD_SOURCE` defaults to `huggingface`.
2. Hugging Face mode never imports ModelScope.
3. Every automatically resolved artifact in one bundle comes from one provider.
4. No tentative component enters a global cache before the complete bundle
   validates.
5. A ModelScope failure emits at most one warning per bundle and retries the
   complete bundle through Hugging Face.
6. A local-machine or caller-configuration failure does not trigger provider
   fallback.
7. Existing valid Hugging Face cache entries may be reused read-only in
   ModelScope mode.
8. ModelScope never writes into `huggingface/`, `HF_HOME`, or a Hugging Face Hub
   snapshot.
9. Existing canonical model IDs remain valid.
10. A caller-supplied explicit revision is never silently replaced.
11. Bundle directories are immutable after becoming ready.
12. No first-party consumer may receive a remote repository ID after bundle
    resolution; it receives local paths.
13. All first-party implicit-download wrappers are covered, including Eureka
    Audio and the production PaddleOCR paths.

## 4. Goals

- Provide a usable ModelScope source for China.
- Keep Hugging Face as the default.
- Automatically fall back from ModelScope to Hugging Face after one warning.
- Cover every first-party model, dataset, template, metadata, and support-file
  path.
- Guarantee provider coherence across all components of a logical model.
- Reuse complete existing Hugging Face local bundles without network access.
- Preserve explicit complete local model directories.
- Keep provider credentials and caches isolated.
- Produce deterministic diagnostics and aggregate provider errors.
- Make normal CI independent of public network.

## 5. Non-Goals

- IP, country, or latency auto-detection.
- Replacing or removing `HF_ENDPOINT`, `HF_HOME`, or `HF_TOKEN`.
- Falling back from Hugging Face to ModelScope in default mode.
- Hot-swapping objects already in use by a running inference call.
- Copying, moving, hardlinking, or deduplicating existing Hugging Face caches.
- Editing `third_party/**` or `module/see_through/vendor/**`.
- Changing cloud API provider traffic.
- Supporting ModelScope Studio repositories.
- Encrypting GUI environment configuration at rest in this feature.
- Guaranteeing arbitrary network behavior inside unmodified third-party remote
  code without a network-deny contract test.

## 6. Terminology and Data Model

### 6.1 Canonical repository

The canonical repository ID is the existing project-facing ID, normally the
current Hugging Face ID. Callers and user configuration keep using it.

Provider mapping occurs only inside the hub adapter.

### 6.2 Repository requirement

A `RepoRequirement` describes one repository role inside a bundle:

```python
RepoRequirement(
    role="model",
    canonical_repo_id="owner/repository",
    repo_type="model",
    requested_revision=None,
    subfolder=None,
)
```

A bundle can contain multiple repository roles. Examples include:

- a model and a different processor repository;
- WD Tagger model and hierarchy dataset repositories;
- Paddle detection and recognition repositories;
- a Gemma model and a separate template repository.

### 6.3 Artifact requirement

An `ArtifactRequirement` belongs to a repository role and declares what makes
that role complete:

```python
ArtifactRequirement(
    role="onnx_graph",
    repo_role="detector",
    paths=("inference.onnx", "inference.yml", "inference.json"),
)
```

Requirements may use exact paths or bounded patterns. Open-ended snapshot
downloads still record the exact remote file list and hashes in the resulting
manifest.

### 6.4 Bundle specification

A `BundleSpec` is the complete logical unit:

```python
BundleSpec(
    key="paddle_ocr:ppocrv6:medium",
    repositories=(detector_repo, recognizer_repo),
    artifacts=(detector_files, recognizer_files),
    legacy_candidates=(),
)
```

The bundle key is stable for a logical model shape. Runtime load settings such
as dtype and attention implementation do not change artifact identity, but they
do participate in the loaded-object cache key.

### 6.5 Resolved bundle

A `ResolvedBundle` contains only local paths and provenance:

- bundle key;
- one origin: `local-explicit`, `local-legacy`, `modelscope`, or
  `huggingface`;
- canonical and provider repository IDs;
- requested and provider revisions;
- exact file list, sizes, and available SHA-256 values;
- immutable local roots for every repository role;
- manifest fingerprint;
- provenance quality;
- cache-hit and fallback metadata.

A `ResolvedBundle` cannot contain both `modelscope` and `huggingface` origins.
Construction validates this invariant.

### 6.6 Loaded bundle

A consumer receives one `ResolvedBundle` and constructs every required runtime
object into a temporary `LoadedBundle`:

```python
LoadedBundle(
    resolution=resolved_bundle,
    components={
        "processor": processor,
        "model": model,
    },
)
```

The `LoadedBundle` enters a shared cache only after every component and contract
check succeeds.

The loaded cache key includes:

- bundle key;
- preferred source;
- resolved bundle fingerprint;
- component classes;
- load-configuration fingerprint.

The current processor and model caches keyed only by repository ID are not
sufficient and must not remain authoritative for multi-component loads.

## 7. Audited First-Party Download Surface

### 7.1 Shared Transformers paths

All first-party users of:

- `load_pretrained_component`;
- `transformerLoader.get_or_load_processor`;
- `transformerLoader.get_or_load_model`;
- direct `from_pretrained`;
- `snapshot_download_with_reporting`;

must resolve through a bundle.

This includes OCR, local VLM, local ALM, local LLM, and first-party cloud-named
providers that locally instantiate Transformers objects.

Providers that currently load processor/tokenizer and model in separate calls
must migrate to a single bundle load. The affected families include Chandra,
DeepSeek, Dots, FireRed, GLM, Hunyuan, Infinity Parser, LightOn, Logics,
Nanonets, olmOCR, OvisOCR2, Qianfan, Unlimited OCR, Penguin-VL, local LLMs, and
the local-loading Qwen/StepFun paths.

Single-component models use a one-component bundle rather than a separate
source state machine.

### 7.2 Shared ONNX paths

The shared ONNX runtime must bundle download and first session creation for:

- water detection;
- PP-OCRv6 components;
- LFM-VL;
- MuSViT sheet-music OMR;
- vocal MIDI;
- audio separation;
- WD Tagger;
- every graph with external tensor data;
- every support-file set used by those graphs.

### 7.3 Eureka Audio

`module/providers/local_alm/eureka_audio_local.py` currently passes the canonical
Hugging Face ID directly to `EurekaAudio`.

The outer provider must resolve a complete bundle and pass the immutable local
snapshot path as `model_path`. A contract test with the installed
`eureka-audio-local` extra must prove that construction performs no hub network
request.

If the upstream wrapper cannot load a local path, the implementation must adapt
the first-party outer wrapper. It must not restore an implicit remote ID.

### 7.4 Production PaddleOCR

The production `ppocrv6_onnx` path currently constructs `PaddleOCR` with model
names and allows PaddleOCR to download models implicitly. The production
`paddle_vl_native` path does the same for `PaddleOCRVL`.

The existing `ppocrv6_direct_onnx` path is experimental and ends in
`NotImplementedError`; it is not coverage for production.

The source-aware implementation must:

1. build one bundle for all enabled Paddle components;
2. resolve detection and recognition repositories together;
3. include optional orientation, unwarping, layout, chart, seal, and VL
   components only when enabled;
4. pass resolved local directories through PaddleOCR's existing
   `*_model_dir` arguments;
5. reject an enabled implicit component that has neither a declared repository
   nor an explicit local directory;
6. construct the real production `PaddleOCR` or `PaddleOCRVL` object as the
   bundle consumer validation.

For the default PP-OCRv6 configuration, detection and recognition models form
one provider-coherent bundle. A missing ModelScope recognition model causes the
complete bundle to retry through Hugging Face; it cannot pair with a ModelScope
detection model.

### 7.5 Direct auxiliary and specialized paths

Explicit migration remains required for:

- `module/audio_separator_core.py`;
- `module/wdtagger/model_loader.py`;
- `utils/wdtagger.py`;
- `utils/tag_highlighting.py`;
- `utils/wdtagger_siglip2.py`;
- `module/providers/local_vlm/gemma4_local.py`;
- `module/rewardmodel.py`;
- `module/providers/local_alm/mega_asr_local.py`;
- `module/muscriptor_tool/runtime.py`.

### 7.6 Cross-runtime bundles

A bundle can cross library boundaries.

Examples:

- LFM-VL combines a Transformers processor with several ONNX components.
- Audio separation combines an ONNX model with metadata.
- WD Tagger combines ONNX, labels, and hierarchy data.
- Gemma 4 may combine model assets with a separate chat-template repository.

All parts use the same provider attempt and commit together.

## 8. Configuration Contract

### 8.1 Environment variables

Add:

| Variable | Values | Default | Purpose |
| --- | --- | --- | --- |
| `MODEL_DOWNLOAD_SOURCE` | `huggingface`, `modelscope` | `huggingface` | Preferred network provider |
| `MODELSCOPE_CACHE` | Filesystem path | project-relative `modelscope` | Immutable ModelScope bundle store |
| `MODELSCOPE_API_TOKEN` | Optional string | empty | ModelScope authentication |
| `MODELSCOPE_ENDPOINT` | URL | SDK default, currently `https://modelscope.cn` | Advanced override |

An invalid `MODEL_DOWNLOAD_SOURCE` value fails configuration immediately. It
does not silently choose Hugging Face.

Existing variables remain unchanged:

- `HF_HOME`
- `HF_ENDPOINT`
- `HF_TOKEN`

Hugging Face fallback honors the configured Hugging Face endpoint, including an
existing mirror.

There is no fallback toggle. ModelScope-to-Hugging-Face fallback is fixed.

### 8.2 GUI

Add:

- a source dropdown;
- a ModelScope cache path field;
- a masked ModelScope token field;
- an advanced endpoint field.

The token field prevents shoulder surfing only. The existing environment
configuration persists values as plaintext JSON, like `HF_TOKEN`. This feature
does not claim encryption at rest. Documentation recommends process environment
injection when plaintext persistence is unacceptable.

A source change is applied to a newly launched task. An object already executing
in a task is not mutated. Any in-process loaded-object cache must include source
and bundle fingerprint so a later load request cannot silently reuse a bundle
from the previously selected provider.

### 8.3 Mapping and coverage file

Add `config/model_hub.toml` and merge it through `config/loader.py`.

Models and datasets use separate namespaces.

Verified mapping example:

```toml
[modelscope.models."davanstrien/dots.ocr-1.5"]
status = "verified"
repo_id = "rednote-hilab/dots.ocr-1.5"
default_revision = "master"
verified_at = "2026-07-31"

[modelscope.models."davanstrien/dots.ocr-1.5-svg"]
status = "verified"
repo_id = "rednote-hilab/dots.ocr-1.5-svg"
default_revision = "master"
verified_at = "2026-07-31"
```

Known unavailable example:

```toml
[modelscope.models."owner/repository"]
status = "unavailable"
checked_at = "2026-07-31"
reason = "No provider-equivalent repository was published."
```

Optional explicit revision equivalence:

```toml
[modelscope.models."owner/repository".revision_map]
"canonical-tag-or-commit" = "verified-modelscope-tag-or-commit"
```

Rules:

- Every shipped first-party canonical repository has an explicit
  `verified` or `unavailable` entry.
- A first-party `unavailable` entry skips a pointless ModelScope request, emits
  the bundle-level fallback warning, and uses Hugging Face.
- A user-supplied custom repository absent from the inventory tries the same ID.
- `default_revision` applies only when the caller supplied no revision.
- A caller-supplied revision is mapped only through an exact `revision_map`
  entry.
- Without an exact entry, ModelScope tries the caller's exact revision string.
- Failure to resolve that revision falls back to Hugging Face.
- Hugging Face always receives the canonical ID and original caller revision.
- Local paths are never mapped.
- Mapping entries never rename individual files.
- Tokens and endpoints never appear in this file.

The previous scalar `revision` override is rejected because it could silently
replace a requested commit with `master`.

### 8.4 Inventory completeness

The implementation must generate or maintain a first-party repository inventory
covering:

- every shipped `model_id` and `repo_id` in project config;
- hard-coded repository IDs in first-party Python;
- ONNX component repositories;
- dataset and auxiliary-file repositories;
- dynamically constructed Paddle component IDs;
- separate template and processor repositories.

A test fails when a shipped first-party repository has no `verified` or
`unavailable` ModelScope status.

This distinguishes path coverage from repository availability. A path can be
fully source-aware even when a particular model is not published on ModelScope.

## 9. Bundle Source State Machine

### 9.1 Explicit local bundle

An existing local repository argument bypasses mapping and network.

For a multi-repository bundle, every required role must be resolvable from the
explicit local configuration. The resolver does not complete a partial explicit
bundle from a network provider.

The consumer validates the complete local bundle. Failure is a local load error,
not a provider fallback.

### 9.2 Hugging Face mode

For `MODEL_DOWNLOAD_SOURCE=huggingface`:

```text
complete explicit/legacy local bundle
    -> Hugging Face cache/network using existing configuration
    -> complete bundle validation
    -> atomic loaded-bundle cache commit
    -> success or Hugging Face error
```

ModelScope is not imported or scanned.

Successful Hugging Face behavior, endpoint selection, token handling, and
provider errors remain compatible. One intentional internal change is allowed:
after a failed multi-component load, a successfully created processor is no
longer retained in a global partial cache.

### 9.3 ModelScope mode

For `MODEL_DOWNLOAD_SOURCE=modelscope`:

```text
complete explicit local bundle
    -> complete legacy local bundle
    -> complete ready ModelScope bundle
    -> complete Hugging Face local-only bundle
    -> strict ModelScope network materialization
    -> complete consumer validation
    -> one warning if the ModelScope attempt is unavailable or rejected
    -> complete Hugging Face network materialization/load
    -> aggregate error if both providers fail
```

Candidate rules:

- A cache candidate satisfies the entire `BundleSpec` or is skipped.
- ModelScope and Hugging Face cache entries cannot fill each other's missing
  roles.
- A partial ModelScope cache may be completed only through ModelScope network
  materialization into a new immutable bundle.
- Hugging Face fallback may reuse its own Hugging Face cache while downloading
  its missing files.
- The warning is per bundle, never per shard or component.
- A ModelScope success causes no Hugging Face network request.
- A complete Hugging Face local-only hit causes no network request.

### 9.4 Force download

`force_download=True` bypasses provider and legacy caches, except an explicitly
supplied local bundle:

```text
strict ModelScope network bundle
    -> one warning on provider/content failure
    -> Hugging Face network bundle
```

It does not delete old cache entries.

## 10. Hub Adapter

### 10.1 Ownership

Add `utils/model_hub.py` as the provider and bundle-policy boundary.

It owns:

- source parsing;
- `BundleSpec`, `ResolvedBundle`, and manifest types;
- mapping and availability lookup;
- provider ID and revision resolution;
- full-bundle candidate ordering;
- strict ModelScope file materialization;
- Hugging Face local-only probing;
- immutable bundle publication;
- warning deduplication;
- error classification entry points;
- aggregate provider errors;
- path-safety validation;
- test injection points.

Consumers own:

- model-specific class selection;
- load kwargs;
- ONNX session construction;
- runtime contract validation;
- conversion of known library exceptions into typed bundle-content errors.

### 10.2 Public operations

The exact private names may change, but the module must provide:

```python
resolve_bundle_candidates(bundle_spec, ...)
load_consistent_bundle(bundle_spec, consume_bundle, ...)
download_repo_file(...)
download_repo_file_set(...)
download_repo_snapshot(...)
list_repo_files(...)
```

The file and snapshot helpers are compatibility wrappers around one-artifact or
one-repository bundles. They do not have a separate fallback implementation.

`consume_bundle` receives only local paths. It must never receive a ModelScope or
Hugging Face repository ID.

### 10.3 Lazy provider imports

ModelScope imports occur only in a ModelScope provider attempt.

Default Hugging Face mode, explicit local bundles, module import, and
configuration loading do not import `modelscope_hub`.

### 10.4 Token isolation

- ModelScope receives only `MODELSCOPE_API_TOKEN`.
- Hugging Face receives only existing explicit token arguments or `HF_TOKEN`.
- Neither token is forwarded to the other provider.
- Logs redact tokens, authorization headers, cookies, and signed query strings.

## 11. Strict ModelScope Backend

### 11.1 Locked SDK constraint

Pin:

```toml
modelscope-hub==0.1.8
```

Version 0.1.8 has provider primitives the adapter can use, but its
`HubApi.download_repo` contract is unsuitable for this feature:

- it has no `force` argument;
- `local_files_only` accepts any non-empty cache directory while warning it
  cannot confirm the requested revision;
- parallel file exceptions are converted to strings and the method still
  returns the output directory.

Therefore first-party code must not call ModelScope
`HubApi.download_repo`.

### 11.2 Strict snapshot construction

The adapter builds a strict snapshot using official primitives:

1. call `HubApi.list_repo_files` for the exact repository type and revision;
2. normalize and validate every returned path;
3. apply allow and ignore patterns inside the adapter;
4. compute the expected bundle file set;
5. call `HubApi.download_file` for each file into an adapter-owned staging
   directory;
6. pass `force=True` when force download is requested;
7. pass provider SHA-256 metadata to `expected_sha256` when available;
8. propagate every file exception with its original type and path;
9. cancel pending work after a terminal local-machine failure;
10. verify the final file set and hashes before publication.

The adapter may use its own bounded executor, but it cannot discard worker
exceptions.

Single-file requests also use `HubApi.download_file`.

### 11.3 SDK contract tests

Tests against the exact installed package assert:

- `download_file` exposes `force`, `expected_sha256`, and
  `local_files_only`;
- `list_repo_files` returns the metadata fields consumed by the adapter;
- the adapter never calls `download_repo`;
- a per-file exception leaves no ready manifest;
- force download forwards `force=True` to every file;
- cancellation and local `OSError` subclasses retain their classification.

The optional live smoke test is additional evidence, not the only SDK contract
test.

## 12. Cache and Provenance

### 12.1 Separate roots

Hugging Face continues to use its existing cache and `HF_HOME`.

ModelScope bundle materialization uses `MODELSCOPE_CACHE`, defaulting to:

```text
modelscope/
    .staging/
    bundles/
```

Add `modelscope/` to `.gitignore`.

The adapter always supplies its own destination to ModelScope file downloads.
The SDK is never pointed at `huggingface/hub`.

### 12.2 Ready manifest

Every project-owned ModelScope bundle has a `bundle-manifest.json` written only
after complete validation.

It records:

- schema version;
- bundle key and fingerprint;
- provider;
- canonical and provider repository IDs;
- requested and provider revisions;
- exact normalized file paths;
- sizes and known or computed SHA-256 hashes;
- creation time;
- consumer validation version;
- provenance quality;
- package version used for materialization.

Only a directory with a valid ready manifest can satisfy an automatic
ModelScope cache lookup.

An unrelated pre-existing ModelScope SDK cache without this manifest is not
silently trusted. A user may pass it as an explicit local bundle and accept
local-path semantics.

### 12.3 Hugging Face read-only reuse

In ModelScope mode, the adapter may use supported Hugging Face
`local_files_only=True` operations to resolve every repository role.

The candidate is accepted only when the complete bundle validates.

The project does not:

- copy it into ModelScope cache;
- write a ModelScope manifest into it;
- change Hugging Face metadata;
- delete or promote files inside it;
- fill missing files from ModelScope.

Filesystem access-time changes caused by the operating system are outside this
write-isolation guarantee.

### 12.4 Legacy project directories

Existing locations such as:

- `huggingface/`;
- `huggingface/Mega-ASR`;
- existing MuSViT and ONNX directories;

remain complete local-candidate locations.

Because they may lack repository and revision provenance:

- they are marked `local-legacy` with `provenance="unverified"`;
- they must satisfy every structural and consumer validation;
- they are not combined with provider cache roles;
- they are skipped for an explicit immutable revision unless provenance can be
  proven;
- no manifest is written into the legacy directory;
- no new ModelScope download targets the legacy directory.

This preserves old local models without falsely claiming they correspond to a
specific provider revision.

## 13. Immutable Materialization on Windows

Directory replacement is not the atomicity mechanism.

Windows cannot atomically replace an existing non-empty directory. The adapter
therefore never promotes a new bundle over an existing ready bundle.

The publication sequence is:

1. create a unique staging directory on the same filesystem;
2. download and structurally validate all repository roles;
3. compute a bundle fingerprint from identity and file metadata;
4. acquire the bundle-fingerprint lock;
5. rename staging to a new, absent immutable version directory;
6. run first consumer validation from that final immutable path;
7. write `bundle-manifest.json.tmp`;
8. atomically replace the manifest file with `bundle-manifest.json`;
9. release the lock.

Consumers never move the directory after opening model or ONNX files.

If validation fails:

- no ready manifest is written;
- the directory is removed or quarantined using best effort;
- no reader treats it as a cache hit;
- failure cleanup cannot replace the original exception.

If the process crashes after directory rename but before the ready manifest, the
next locked resolver treats the directory as an orphan and validates or
quarantines it.

For explicit download destinations, new bundles live in an adapter-owned
immutable subdirectory and the returned path points there. Existing arbitrary
files in the user directory are not replaced. Single compatibility files may be
published with same-filesystem file-level `os.replace`, but compound consumers
read only through the ready bundle manifest.

## 14. Consumer Validation and Error Taxonomy

### 14.1 Temporary load and cache commit

For each candidate:

1. build all components into a temporary `LoadedBundle`;
2. run model-specific contract validation;
3. commit the whole `LoadedBundle` to cache only on success.

If any component fails:

- no component from that attempt remains in processor, model, session, or
  provider-global caches;
- temporary references are released;
- `gc.collect()` and device-cache cleanup may run as best effort;
- cleanup failures are attached as diagnostics and do not replace the root
  cause.

The model, processor, and provider base caches in:

- `utils/transformer_loader.py`;
- `module/providers/local_vlm_base.py`;
- `module/providers/local_alm_base.py`;
- `module/providers/local_llm_base.py`;

must key and commit complete bundle results, not repository IDs alone.

### 14.2 Fallback-eligible errors

Automatic provider fallback is allowed for:

- repository, revision, or required file not found;
- provider authentication rejection;
- timeout and transport failure;
- provider rate or availability error;
- missing required file;
- path or manifest mismatch;
- file hash or size mismatch;
- invalid safetensors or model configuration attributable to the artifact;
- invalid ONNX graph or external-data relationship;
- incompatible provider repository contents;
- typed first-consumption `BundleContentError`;
- missing ModelScope package while ModelScope is selected.

### 14.3 Non-fallback errors

These propagate immediately:

- `KeyboardInterrupt`, `SystemExit`, or user cancellation;
- disk exhaustion;
- destination permission denial;
- invalid destination path;
- CUDA or CPU out-of-memory;
- missing local runtime dependency such as bitsandbytes;
- unavailable attention backend;
- unsupported device, dtype, or quantization setting;
- invalid caller kwargs or project configuration;
- failure in an explicit local bundle;
- unknown consumer exceptions not classified as bundle content.

Unknown load failures default to non-fallback. Hiding an environment defect by
downloading the model again is worse than stopping with the first useful error.

### 14.4 Typed consumer classification

Generic substring matching is not the primary classifier.

Each consumer converts only known artifact-caused library failures into
`BundleContentError`. Examples:

- ONNX invalid protobuf or missing external tensor file;
- safetensors header/hash corruption;
- Transformers configuration or remote-code file absent from the resolved
  snapshot.

Existing local fallback behavior, such as attention implementation fallback,
runs inside the same provider attempt. It does not switch providers.

### 14.5 Warning and aggregate error

One ModelScope-to-Hugging-Face warning contains:

- bundle key;
- canonical repository IDs;
- failure stage;
- sanitized reason;
- statement that the complete bundle is being retried.

If Hugging Face also fails, one project-owned error preserves:

- ModelScope attempt stages and causes;
- Hugging Face attempt stages and causes;
- mapped identities and revisions;
- checked cache candidates;
- cleanup diagnostics;
- no credentials.

## 15. Consumer Integration

### 15.1 Transformers

Add a bundle load operation to `utils/transformer_loader.py`.

For a ModelScope or Hugging Face local candidate:

- every `from_pretrained` receives the role's resolved local path;
- `local_files_only=True` prevents hidden provider traffic;
- processor, tokenizer, config, and model are built in one temporary result;
- all are cached together only after success.

Existing separate helper methods may remain as one-component compatibility
wrappers. First-party multi-component callers cannot use them as independent
transactions.

The architecture test identifies any provider that loads multiple components
without a bundle context.

### 15.2 ONNX

`module/onnx_runtime/artifacts.py` becomes a bundle-spec builder rather than an
Hugging Face downloader.

`single_model.py` and `multi_model.py` consume a complete local bundle and create
all required ONNX sessions before commit.

An ONNX bundle includes:

- graph files;
- all discovered or declared external-data files;
- support metadata;
- every component in a multi-model runtime.

Session construction and existing input/output contract checks can raise typed
bundle-content errors. Device/provider configuration failures do not switch
download source.

### 15.3 LFM-VL

The processor and all ONNX components form one cross-runtime bundle. A
ModelScope processor cannot be combined with Hugging Face ONNX files.

### 15.4 Audio separator and WD Tagger

Audio model and metadata are one bundle.

WD Tagger ONNX model, labels, and required hierarchy data are one bundle even
when they span model and dataset repositories.

### 15.5 Gemma 4

The model and required chat template are one bundle. If the template lives in a
separate repository, both repository roles use the same provider attempt.

### 15.6 Reward model

Direct `from_pretrained` calls route through a one-model or model-plus-processor
bundle, depending on the scorer implementation.

### 15.7 SigLIP2

Snapshot and processor loading share one bundle transaction. A processor failure
cannot leave a cached ModelScope snapshot while the processor uses Hugging Face.

### 15.8 Mega-ASR

The resolver returns the actual immutable bundle directory. The first
`Qwen3ASRModel` load and all checkpoint support-file validation occur before
commit.

`huggingface/Mega-ASR` remains a legacy complete-bundle candidate.

### 15.9 MuScriptor

The bundle includes at least:

- `model.safetensors`;
- `config.json`.

The consumer calls the installed upstream
`TranscriptionModel.load_model(weights_path=<local-file>)`.

The upstream first load validates the candidate. No upstream repository ID is
passed.

### 15.10 See-Through

Do not edit vendored code.

The first-party outer loader resolves all required repositories and passes local
paths. A network-deny test during first load proves that the covered vendored
flow does not escape to Hugging Face.

If vendored code requires a separately hosted asset, that asset becomes another
role in the outer bundle.

## 16. Dependency and Process Boundaries

### 16.1 Project dependency

Add:

```toml
modelscope-hub==0.1.8
```

to normal project dependencies.

The exact pin contains alpha-SDK change risk. Upgrade requires provider-contract
tests before changing the version.

### 16.2 PEP 723 standalone scripts

`module/rewardmodel.py` and `module/waterdetect.py` have inline PEP 723
dependencies and are launched with `uv run <script>`.

uv ignores project dependencies when inline script metadata is present.
Therefore both inline dependency lists must also contain:

```toml
modelscope-hub==0.1.8
```

A test enumerates first-party PEP 723 scripts that transitively use the bundle
adapter and verifies the exact dependency pin. Adding only `pyproject.toml` is a
release-blocking failure.

PowerShell launch commands themselves do not need mass edits.

## 17. Security and Path Safety

- Tokens are isolated and redacted.
- GUI masking is not described as encryption.
- Every provider file path must be relative.
- Reject absolute paths, drive-prefixed paths, `..` traversal, empty normalized
  paths, and paths resolving outside staging.
- Reject normalized duplicate and Windows case-fold collisions.
- Reject symlink-like or special-file materialization.
- Resolve destination roots before creating or moving files.
- Never extract a provider archive for model and dataset bundles in this
  version.
- Consumer remote code executes only after structural publication into an
  immutable, non-ready directory and before the ready manifest.
- A network-deny harness detects first-load attempts to contact an undeclared
  repository.

Tests cover malicious file names, revision strings, and provider-returned paths.

## 18. Diagnostics

At task startup, report:

- selected source;
- resolved ModelScope cache path when relevant;
- whether a custom endpoint is configured;
- whether a ModelScope token is present, as a boolean;
- mapping inventory version.

For each resolved bundle, structured debug metadata includes:

- bundle key;
- origin;
- canonical and provider IDs;
- revisions;
- cache-hit status;
- fingerprint;
- fallback status.

Never print token values, cookies, or signed URLs.

ModelScope progress uses provider file callbacks or adapter progress. It must not
display duplicate bars from both SDK and project reporting.

## 19. Testing Strategy

Normal CI uses temporary directories, provider fakes, and the installed pinned
SDK contract. Public network is opt-in.

### 19.1 Configuration tests

- Default source is Hugging Face.
- Accepted enum values parse.
- Invalid values fail immediately.
- ModelScope cache resolves relative to project root.
- GUI persists and injects settings.
- Token field is masked while plaintext-storage behavior is documented.
- A new load after source change does not reuse a differently sourced bundle
  cache key.

### 19.2 Bundle-invariant tests

- `ResolvedBundle` rejects mixed provider origins.
- A ModelScope processor plus Hugging Face model cannot be constructed.
- Multiple repositories in one bundle use the same provider.
- No component cache entry appears before full validation.
- A failed second component discards the first component.
- Fallback retries every role from Hugging Face.
- Loaded cache key includes source, bundle fingerprint, and load settings.
- One-component wrappers use the same state machine.

### 19.3 Mapping tests

- Verified same-ID and different-ID mappings.
- Model and dataset namespace isolation.
- `unavailable` status skips ModelScope network.
- `default_revision` applies only when caller omitted revision.
- Exact revision map applies only to an exact caller revision.
- An unmapped explicit revision is tried unchanged.
- Hugging Face receives original ID and revision.
- Local paths bypass mapping.
- Dots 1.5 and SVG IDs map to the verified `rednote-hilab` ModelScope IDs.
- Every shipped first-party repository has a coverage status.

### 19.4 Cache and provenance tests

- Complete ModelScope ready-manifest hit uses no network.
- Non-empty ModelScope directory without ready manifest is not trusted.
- Complete Hugging Face local-only bundle uses no network.
- Partial ModelScope and complete Hugging Face caches do not mix.
- Partial Hugging Face cache and ModelScope cache do not mix.
- Legacy complete bundle loads with unverified provenance.
- Legacy bundle is skipped for an unproven explicit immutable revision.
- Force download bypasses provider and legacy caches.
- No ModelScope operation writes under Hugging Face roots.

### 19.5 Strict SDK tests

- Adapter uses `list_repo_files` plus `download_file`.
- Adapter never uses ModelScope `download_repo`.
- Every listed required file is downloaded.
- Force is forwarded.
- SHA-256 metadata is forwarded and verified.
- One file failure prevents ready publication.
- Worker exceptions remain typed.
- Cancellation, permission denial, and disk exhaustion do not trigger Hugging
  Face fallback.

### 19.6 Windows publication tests

- A new version directory has an absent final name before rename.
- Existing ready bundle is never directory-replaced.
- Reader ignores a directory without ready manifest.
- Crash-recovery handles orphan directories.
- A session is created from its final path, not staging.
- Explicit destinations preserve unrelated user files.
- File-level publication uses same-filesystem atomic replace only where
  applicable.

### 19.7 Fallback classification tests

Fallback cases:

- not found;
- authentication rejection;
- timeout;
- missing required file;
- hash mismatch;
- invalid ONNX artifact;
- typed Transformers artifact/content failure;
- known unavailable ModelScope mapping.

Non-fallback cases:

- CUDA OOM;
- missing bitsandbytes;
- unsupported attention backend after existing local fallback is exhausted;
- invalid dtype, device, or kwargs;
- disk full;
- permission denial;
- cancellation;
- unknown consumer exception.

Verify one warning per bundle and both sanitized causes when both providers
fail.

### 19.8 First-party path tests

Focused tests cover:

- every multi-component Transformers provider family;
- local LLM tokenizer plus model;
- LFM-VL processor plus ONNX components;
- production PP-OCRv6 `PaddleOCR`;
- production `PaddleOCRVL` enabled components;
- Eureka Audio local `model_path`;
- water detection;
- audio separator;
- WD Tagger;
- tag highlighting;
- Gemma template;
- reward model;
- SigLIP2;
- Mega-ASR;
- MuScriptor;
- MuSViT;
- vocal MIDI;
- See-Through outer loading.

Paddle and Eureka tests deny network after local bundle resolution.

### 19.9 Architecture enforcement

Static tests reject first-party direct uses outside the adapter or approved
progress bridge of:

- `hf_hub_download`;
- `snapshot_download`;
- `list_repo_files`;
- ModelScope SDK imports or calls;
- ModelScope `download_repo` anywhere in first-party code.

A provider-registry test requires every first-party local model provider with an
implicit repository ID to expose or build a `BundleSpec`.

A Transformers call-site test rejects separate first-party processor/model
transactions without one shared bundle context.

Exclusions:

- `third_party/**`;
- `module/see_through/vendor/**`;
- tests and fixtures.

Vendored exclusions do not exempt first-party outer loaders from network-deny
tests.

### 19.10 PEP 723 tests

- `rewardmodel.py` includes the exact ModelScope pin.
- `waterdetect.py` includes the exact ModelScope pin.
- every future inline script that transitively imports the adapter is detected.
- script-mode import succeeds in an isolated uv environment.

### 19.11 Optional live smoke test

```powershell
$env:RUN_MODELSCOPE_SMOKE = "1"
pytest tests/test_model_hub_smoke.py -q
```

The test downloads a small file, verifies the ready manifest, verifies a cache
hit, and exercises one known mapping. It is not default CI.

## 20. Acceptance Criteria

Implementation is accepted only when:

1. Without new configuration, Hugging Face remains selected and ModelScope is
   not imported.
2. Every first-party local download or implicit-download path declares a
   complete bundle.
3. No successful or failed operation produces a ModelScope/Hugging Face mixed
   bundle.
4. No partial runtime object enters a global cache.
5. A complete ModelScope success causes no Hugging Face network request.
6. A complete Hugging Face local-only bundle in ModelScope mode causes no
   network request.
7. A fallback-eligible ModelScope failure emits one warning and retries the
   complete bundle through Hugging Face.
8. A local-machine, dependency, device, or caller error does not switch
   providers.
9. Both-provider failure preserves both sanitized causes.
10. ModelScope writes nothing under Hugging Face cache or legacy roots.
11. Caller-supplied revisions are never replaced except by exact verified
    revision mapping.
12. ModelScope `HubApi.download_repo` is not used.
13. Ready bundles use immutable version directories and manifest-last
    publication.
14. Production PaddleOCR and Eureka Audio receive only resolved local model
    paths.
15. Every shipped first-party repository has a verified or unavailable
    ModelScope inventory entry.
16. Existing canonical model IDs and supported complete local directories remain
    valid.
17. `modelscope-hub==0.1.8` is pinned in the project and both affected PEP 723
    scripts.
18. The existing full test suite passes in default Hugging Face mode.

## 21. Independent-Review Resolution

| Finding | Revision 2 resolution |
| --- | --- |
| Eureka and production PaddleOCR omitted | Added explicit local-path bundle integration and network-deny tests |
| Component-level transaction allows mixed sources | Replaced with complete `BundleSpec`, temporary `LoadedBundle`, and atomic cache commit |
| PEP 723 scripts ignore project dependency | Added exact pin requirement and isolated script tests |
| ModelScope 0.1.8 snapshot API swallows errors and lacks force | Prohibited `download_repo`; strict list plus per-file download backend |
| Mapping can override explicit revision | Removed scalar override; exact revision map only |
| Every first-load error triggers fallback | Added typed artifact errors and fail-closed non-fallback taxonomy |
| Directory replacement is not atomic on Windows | Added immutable absent-target directories and manifest-last publication |
| Mapping has no completeness guarantee | Added verified/unavailable inventory and coverage gate |
| Legacy cache has no provenance | Added `local-legacy` unverified semantics and immutable-revision restriction |
| Provider paths can escape staging | Added normalization, traversal, collision, and containment rules |
| GUI token masking implies secure storage | Explicitly states plaintext persistence and scope |

## 22. Expected Implementation Touchpoints

Core:

- new `utils/model_hub.py`;
- new `config/model_hub.toml`;
- `config/loader.py`;
- `utils/transformer_loader.py`.

Loaded-object cache ownership:

- `module/providers/local_vlm_base.py`;
- `module/providers/local_alm_base.py`;
- `module/providers/local_llm_base.py`.

ONNX:

- `module/onnx_runtime/artifacts.py`;
- `module/onnx_runtime/single_model.py`;
- `module/onnx_runtime/multi_model.py`;
- `module/onnx_runtime/config.py`.

Production implicit-download wrappers:

- `module/providers/ocr/paddle.py`;
- `module/providers/local_alm/eureka_audio_local.py`.

Direct outliers:

- `module/audio_separator_core.py`;
- `module/wdtagger/model_loader.py`;
- `utils/wdtagger.py`;
- `utils/wdtagger_siglip2.py`;
- `utils/tag_highlighting.py`;
- `module/providers/local_vlm/gemma4_local.py`;
- `module/providers/local_vlm/lfm_vl_local.py`;
- `module/rewardmodel.py`;
- `module/waterdetect.py`;
- `module/providers/local_alm/mega_asr_local.py`;
- `module/muscriptor_tool/runtime.py`.

Multi-component provider call sites:

- every first-party caller currently issuing separate
  `get_or_load_processor`/`get_or_load_model` or repeated
  `load_pretrained_component` calls;
- outer See-Through loaders when vendored code needs local roots.

GUI and launch:

- `gui/utils/env_config.py`;
- `gui/wizard/step7_settings.py`;
- `gui/launch.py`.

Packaging:

- `pyproject.toml`;
- PEP 723 blocks in `module/rewardmodel.py` and `module/waterdetect.py`;
- `.gitignore`.

The implementation plan must enumerate the exact provider call sites found by
static scan. It must not restore provider conditionals inside each provider.

## 23. Residual Risks

- A verified mapping can become stale when a provider repository changes.
- Branch names such as `main` and `master` are mutable; manifests record the
  observed file identity, but force download is required to refresh.
- ModelScope 0.1.8 is alpha and remains a dependency risk despite the strict
  wrapper.
- Some remote code may attempt undeclared network access. Network-deny tests
  decide whether the outer bundle is complete.
- Full-bundle fallback can redownload more data than component fallback. That
  cost is intentional because source coherence is more important than retaining
  an unusable partial model.
- Enforcing complete explicit local bundles may reject previously accidental
  partial-local configurations. The implementation plan must inventory current
  public configuration fields and provide a clear error naming missing roles.

## 24. References

- [ModelScope Hub SDK repository](https://github.com/modelscope/modelscope_hub)
- [modelscope-hub 0.1.8 on PyPI](https://pypi.org/project/modelscope-hub/)
- [ModelScope 0.1.8 HubApi source](https://github.com/modelscope/modelscope_hub/blob/v0.1.8/src/modelscope_hub/api.py)
- [ModelScope 0.1.8 download implementation](https://github.com/modelscope/modelscope_hub/blob/v0.1.8/src/modelscope_hub/_download.py)
- [uv inline script dependency behavior](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)
- [Hugging Face cache-system reference](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)
- [ModelScope dots.ocr-1.5](https://modelscope.cn/models/rednote-hilab/dots.ocr-1.5)
- [ModelScope dots.ocr-1.5-svg](https://modelscope.cn/models/rednote-hilab/dots.ocr-1.5-svg)
