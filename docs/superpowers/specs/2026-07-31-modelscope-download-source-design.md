# ModelScope Alternative Download Source Design

**Status:** Revision 4 after review; pending implementation-plan approval

**Date:** 2026-07-31

**Revision:** 4

**Scope:** Design only. This document does not authorize production-code changes.

## 1. Decision Summary

qinglong-captions will support ModelScope as an opt-in model and artifact
download source for users in China.

Hugging Face remains the default. When ModelScope is selected, each compatible
artifact group is resolved independently:

1. reuse an already complete local group when possible;
2. otherwise try ModelScope;
3. on a fallback-eligible ModelScope failure, emit one warning;
4. retry that complete group through Hugging Face.

The central rule is:

> Artifacts inside one `CoherenceGroup` use one origin. Independent groups may
> resolve from different origins.

Examples:

- model weights, config, tokenizer, and processor files that must agree form one
  group;
- one ONNX graph and its external tensor files form one group;
- a classifier graph and its index-ordered labels form one group;
- independent Paddle detection and recognition packages form separate groups.

Provider is retrieval provenance, not artifact identity. SHA-256 values verify
content, but Revision 4 does not use equal hashes to permit provider mixing
inside a group. A future design may relax that only when the project owns a
trusted, complete cross-provider digest manifest.

The ModelScope feature ends at artifact resolution and structural validation.
It does not change the cardinality or key semantics of loaded model caches.
Transactional runtime-object caching is a separate bug fix and requires a
separate design.

## 2. Problem and Design Judgment

### 2.1 User requirement

The product requirement is explicit:

- add ModelScope for China;
- keep Hugging Face as the default;
- enable ModelScope only through configuration;
- warn and fall back to Hugging Face when ModelScope fails;
- cover every first-party download path.

This requirement does not depend on proving that an existing Hugging Face mirror
is always insufficient. `HF_ENDPOINT` remains supported, but it is a transport
override for the Hugging Face API and cache model, not an independent repository
inventory selected by the user.

### 2.2 Root failure mode

The correctness failure is compatibility skew, not merely different provider
names.

Two files are unsafe to combine when their shared contract is not independently
proven. Common contracts include:

- tokenizer vocabulary and embedding rows;
- processor configuration and model input layout;
- ONNX graph and external tensor offsets;
- label order and classifier output indices;
- remote-code modules and the configuration that imports them.

The design therefore models the smallest compatibility unit explicitly. It does
not turn a complete OCR pipeline or application feature into one giant
transaction when its components are independent.

### 2.3 Why not mix equal hashes in Revision 4?

Equal hashes prove equal bytes for the compared files. They do not prove that:

- every required file was included in the comparison;
- two mutable branch names identify the same release;
- an omitted support file is irrelevant;
- a provider mapping points to the intended upstream release.

Revision 4 chooses the conservative rule of one origin per coherence group.
Hashes remain mandatory integrity and cache-identity inputs where available.

### 2.4 Main regression risks

- changing successful Hugging Face behavior while the feature is disabled;
- defining groups too broadly and causing unnecessary multi-gigabyte fallback;
- defining groups too narrowly and allowing compatibility skew;
- treating a mutable branch name as immutable identity;
- trusting an incomplete ModelScope snapshot because the SDK returned a path;
- duplicating existing ONNX download helpers;
- changing loaded-cache keys and increasing CPU, RAM, or VRAM residency;
- rebuilding source-selection logic already maintained by PaddleX;
- tests that are always skipped in the actual CI dependency environment;
- a third-party wrapper silently bypassing the selected source.

## 3. Required Invariants

The implementation must preserve all of these conditions:

1. `MODEL_DOWNLOAD_SOURCE` defaults to `huggingface`.
2. Hugging Face mode does not import or call the ModelScope adapter.
3. One `CoherenceGroup` has exactly one resolved origin.
4. Different groups in one `BundleSpec` may have different origins.
5. A failed ModelScope attempt never contributes files to that group's Hugging
   Face retry.
6. A successful independent group is not redownloaded because another group
   falls back.
7. ModelScope failure emits at most one project warning per project-managed
   group before its Hugging Face retry.
8. Local-machine, dependency, device, and caller-configuration failures do not
   trigger provider fallback.
9. Existing complete Hugging Face cache entries may be reused read-only in
   ModelScope mode.
10. Partial Hugging Face cache state cannot fill only some roles of a group.
11. ModelScope never writes into `huggingface/`, `HF_HOME`, or a Hugging Face
    Hub snapshot.
12. Existing canonical model IDs remain valid.
13. An explicit caller revision is never replaced by a provider default.
14. A ready project-managed group directory is immutable.
15. Default Hugging Face mode preserves current loaded-object cache behavior.
16. Source selection is fixed before optional model frameworks are imported.
17. Every first-party download path is either project-managed or explicitly
    registered as upstream-managed.
18. User-supplied complete local paths bypass network source selection.
19. An indeterminate mutable-revision revalidation reuses a valid ready
    ModelScope group as stale instead of contacting Hugging Face.
20. `upstream-managed` ownership never substitutes for concrete repository
    availability evidence.
21. An uncertain coherence boundary defaults to the broader group.

## 4. Goals

- Provide a useful ModelScope path for China.
- Preserve Hugging Face as the zero-configuration behavior.
- Fall back from ModelScope to Hugging Face at coherence-group granularity.
- Cover first-party models, datasets, templates, metadata, and support files.
- Reuse complete local Hugging Face artifacts without copying them.
- Keep provider credentials and cache roots isolated.
- Preserve explicit revisions and canonical IDs.
- Produce deterministic, sanitized diagnostics.
- Keep normal CI independent of public model services.
- Deliver the feature in independently reviewable and reversible stages.

## 5. Non-Goals

- Country, IP, or latency auto-detection.
- Replacing or removing `HF_ENDPOINT`, `HF_HOME`, or `HF_TOKEN`.
- Adding Hugging-Face-to-ModelScope fallback to project-managed paths in
  default mode.
- Mixing providers inside a coherence group, even when observed hashes match.
- Treating the same `main` or `master` name on two hubs as proof of equivalence.
- Copying, moving, hardlinking, or deduplicating Hugging Face cache files.
- Changing current loaded-model cache keys or allowing multiple dtype/device
  variants to coexist.
- Fixing every existing partial runtime-object cache behavior in this feature.
- Editing `third_party/**` or `module/see_through/vendor/**`.
- Reimplementing PaddleX's official-model hoster inventory.
- Guaranteeing that arbitrary third-party remote code is safe.
- Encrypting GUI environment values at rest.

## 6. Terminology and Data Model

### 6.1 Canonical repository ID

The canonical repository ID is the existing project-facing ID, normally the
current Hugging Face ID. User configuration and provider call sites continue to
use it.

Provider mapping is internal. ModelScope repository names never leak into
provider configuration or become new public model IDs.

### 6.2 Artifact identity

`ArtifactIdentity` describes what the project requested, independently of where
the bytes were fetched:

```python
ArtifactIdentity(
    canonical_repo_id="owner/repository",
    repo_type="model",
    requested_revision="v1.2",
    selection=("config.json", "model.safetensors", "tokenizer.json"),
)
```

The resolved identity additionally records:

- the provider repository ID;
- the exact provider revision request;
- an immutable provider commit ID when exposed;
- the exact selected remote file set;
- remote size and digest metadata;
- a computed manifest fingerprint;
- the mapping inventory version.

`requested_revision`, a provider branch name, and a content fingerprint are
different fields. They must not be collapsed into one string.

### 6.3 Repository requirement

A `RepoRequirement` represents one repository role:

```python
RepoRequirement(
    role="model",
    canonical_repo_id="owner/repository",
    repo_type="model",
    requested_revision=None,
    subfolder=None,
)
```

One coherence group may span multiple repositories when those repositories
jointly implement one compatibility contract.

### 6.4 Artifact requirement

An `ArtifactRequirement` declares which files make a repository role complete:

```python
ArtifactRequirement(
    role="onnx_graph",
    repo_role="detector",
    paths=("inference.onnx", "inference.yml", "inference.json"),
)
```

Requirements may contain exact paths or bounded patterns. A full snapshot is
allowed, but the resolved manifest records its exact file set.

### 6.5 Coherence group

`CoherenceGroupSpec` is the unit of source choice, cache publication, warning,
and fallback:

```python
CoherenceGroupSpec(
    key="wd_tagger:model_and_labels",
    repositories=(model_repo, labels_repo),
    artifacts=(onnx_graph, labels_csv),
    validator="wd_tagger_contract",
)
```

All automatically resolved repository roles in this group use one origin:

- `local-explicit`;
- `local-legacy`;
- `modelscope`;
- `huggingface`;
- or `upstream-managed` for a registered external source manager.

The group key describes a compatibility contract, not a UI feature.

### 6.6 Bundle specification

`BundleSpec` is what one consumer needs before construction:

```python
BundleSpec(
    key="paddle_ocr:ppocrv6:medium",
    groups=(detector_group, recognizer_group),
)
```

The bundle does not impose one provider across its groups. In this example,
detection may reuse a complete Hugging Face cache while recognition downloads
from ModelScope.

### 6.7 Resolved group

`ResolvedGroup` contains only local paths and provenance:

- group key;
- one origin;
- canonical and provider repository IDs;
- requested and resolved revisions;
- role-to-local-path mapping;
- exact file list, sizes, and available SHA-256 values;
- manifest fingerprint, remote version token, and token basis;
- provenance quality, freshness state, and freshness-check time;
- cache-hit and fallback metadata.

Construction rejects more than one origin. A ModelScope file and a Hugging Face
file cannot coexist in one `ResolvedGroup`.

### 6.8 Resolved bundle

`ResolvedBundle` is an ordered collection of complete `ResolvedGroup` objects.
Its groups may have different origins.

Every group resolves before consumer construction starts. The resolver never
returns a partial group.

### 6.9 Loaded-object cache boundary

Artifact resolution and runtime-object caching are separate concerns.

Revision 4 does not add source, manifest, dtype, attention implementation, or
device settings to the existing loaded-object cache key. Source configuration is
fixed for the process. Changing it through the GUI launches a new process.

If an embedding use case later requires in-process source changes, it must evict
and rebuild the existing cache entry rather than silently keep multiple model
copies resident.

## 7. Coherence Boundary Rules

### 7.1 Same group

Artifacts belong to the same group when one cannot be validated or interpreted
correctly without the other. Required examples:

- Transformers weights, config, tokenizer, processor, and required remote code
  for one model release;
- one ONNX graph and all external-data shards;
- a classifier graph and its ordered label table;
- a model and a separately hosted chat template when that template is required
  by the model's public prompt contract;
- an audio model and metadata that defines its output indices;
- multiple repositories that together implement one indivisible runtime model.

### 7.2 Separate groups

Artifacts are separate when they can be loaded and validated independently and
communicate only through a stable public data contract. Required examples:

- Paddle text detection and text recognition packages;
- optional Paddle orientation and unwarping packages;
- independent stages in a pipeline that exchange images or plain text;
- optional hierarchy metadata that does not define the classifier's output
  index order.

Separate groups may resolve from different providers without violating source
consistency.

### 7.3 Review rule

Every proposed group must state its compatibility contract in one sentence.

If the sentence only says that artifacts are "used by the same feature", the
group is too broad. If two artifacts share an index, vocabulary, tensor layout,
or executable import contract but are in different groups, the groups are too
narrow.

The errors are asymmetric. An overly broad group costs download time and is
visible and recoverable. An overly narrow group can produce silent semantic
corruption. Therefore uncertainty defaults to the safer, broader group.

Splitting an existing group requires the PR to:

1. state the stable interface between the proposed groups;
2. show that each side can be validated without files from the other side;
3. add independent validators or contract tests for both sides.

Absence of that evidence means the artifacts remain in one group.

## 8. First-Party Download Ownership

### 8.1 Two ownership classes

Every audited first-party path has exactly one owner:

1. `project-managed`: qinglong-captions resolves a `CoherenceGroupSpec`;
2. `upstream-managed`: a registered dependency already implements source
   selection, and qinglong-captions configures and tests that mechanism.

Unregistered implicit downloads are defects.

### 8.2 Shared Transformers paths

Project-managed paths include first-party uses of:

- `load_pretrained_component`;
- `transformerLoader.get_or_load_processor`;
- `transformerLoader.get_or_load_model`;
- direct `from_pretrained`;
- `snapshot_download_with_reporting`.

Call sites that load processor/tokenizer and model separately must first resolve
one shared group and pass its local role path to both loads in ModelScope mode.

The current static scan shows this is a real multi-call pattern across OCR,
local VLM, local ALM, local LLM, and cloud-named providers that instantiate
local Transformers objects. The implementation plan must enumerate exact call
sites rather than rely on a prose family list.

In default Hugging Face mode, compatibility wrappers preserve current arguments,
progress reporting, and loaded-cache behavior.

### 8.3 Shared ONNX paths

The implementation extends, rather than duplicates:

- `module/onnx_runtime/artifacts.py`;
- `module/onnx_runtime/config.py`;
- `module/onnx_runtime/single_model.py`;
- `module/onnx_runtime/multi_model.py`.

Existing `download_repo_file` and `download_repo_file_set` remain compatible
entry points. Their downloader and file-lister injection points are retained.

`resolve_model_cache_dir()` remains the owner of existing ONNX destination
defaults. The provider adapter supplies resolved local files without inventing a
second competing ONNX cache configuration.

### 8.4 Production PaddleOCR

Production `PaddleOCR` and `PaddleOCRVL` downloads are upstream-managed.

PaddleX 3.7 already supports:

- `PADDLE_PDX_MODEL_SOURCE`;
- `PADDLE_PDX_HUGGING_FACE_ENDPOINT`;
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`;
- Hugging Face, ModelScope, AIStudio, and BOS hosters.

The bridge relies on the 3.7 hoster contract. Both Paddle extras constrain the
same minor line:

```toml
paddleocr>=3.7.0,<3.8.0
paddleocr[doc-parser]>=3.7.0,<3.8.0
```

The upper bound and an installed-package contract test are both required.
Changing either requires revalidating source order, package-name expansion, and
explicit `local_dir` behavior.

The application sets `PADDLE_PDX_MODEL_SOURCE` before importing PaddleOCR:

- `huggingface` in default mode;
- `modelscope` in ModelScope mode.

If `PADDLE_PDX_HUGGING_FACE_ENDPOINT` is not already set, an existing
`HF_ENDPOINT` value is forwarded to it. An explicit Paddle endpoint remains
authoritative. The application does not force
`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`.

With ModelScope selected, PaddleX orders ModelScope first and Hugging Face as the
first fallback. PaddleX may continue to AIStudio and BOS if both fail. This is
an explicit exception to the project-owned two-provider state machine; replacing
it would require duplicating PaddleX's maintained model inventory.

With Hugging Face selected, PaddleX still owns its existing broader fallback
chain and may eventually try other hosters, including ModelScope. "Hugging Face
is the default" means the preferred source for this upstream-managed path, not
an exclusivity guarantee that PaddleX does not provide.

Each Paddle official-model package is treated as an independent coherence group.
Detection, recognition, orientation, unwarping, layout, chart, seal, and VL
packages are not combined merely because one pipeline enables them.

PaddleX checks support after stripping `_onnx` or `_safetensors`, but passes the
original suffixed name to the hoster download method. The inventory therefore
records the exact expanded provider repository ID, not only the unsuffixed
`OCR_MODELS` membership.

The 2026-07-31 design verification received successful ModelScope metadata
responses for the current production IDs:

- `PaddlePaddle/PP-OCRv6_tiny_det_onnx`;
- `PaddlePaddle/PP-OCRv6_tiny_rec_onnx`;
- `PaddlePaddle/PP-OCRv6_small_det_onnx`;
- `PaddlePaddle/PP-OCRv6_small_rec_onnx`;
- `PaddlePaddle/PP-OCRv6_medium_det_onnx`;
- `PaddlePaddle/PP-OCRv6_medium_rec_onnx`;
- `PaddlePaddle/PaddleOCR-VL`;
- `PaddlePaddle/PaddleOCR-VL-1.5`;
- `PaddlePaddle/PaddleOCR-VL-1.6`.

This observation is evidence for the design, not a permanent guarantee. Phase 0
and release metadata checks regenerate it from the installed Paddle version.
The same check found `inference.onnx`, `inference.yml`, and `inference.json` in
all six ONNX repositories and non-empty file listings for all three VL
repositories.

Caller-supplied `*_model_dir` values bypass source configuration. The project
must not overwrite them with model names or downloaded paths.

The experimental direct-ONNX Paddle path is project-managed and continues to use
the shared ONNX adapter, with detection and recognition as separate groups.

### 8.5 Eureka Audio

Eureka Audio is project-managed because the first-party wrapper currently passes
a repository ID directly into `EurekaAudio`.

The wrapper phase resolves one complete model group and passes the resulting
local directory as `model_path`. This local-path capability is implemented and
reviewed as a separate commit so it can be reverted independently, but it is
required before the overall "all first-party paths" acceptance gate passes.

### 8.6 Vendored and extracted code

Do not edit vendored code.

First-party outer loaders for See-Through and extracted pipelines must either:

- resolve every declared repository and inject local paths; or
- register a tested upstream-managed source mechanism.

A network-deny test after project resolution detects undeclared downloads.

### 8.7 Direct auxiliary paths

The audit and implementation plan must cover at least:

- audio separator assets;
- WD Tagger model, labels, and hierarchy data;
- tag-highlighting resources;
- Gemma chat templates;
- reward model files;
- water-detection files;
- SigLIP2 snapshots;
- Mega-ASR checkpoints;
- MuScriptor weights and config;
- MuSViT;
- vocal MIDI;
- LFM-VL processor and ONNX assets;
- first-party PEP 723 scripts that transitively import the adapter.

The final inventory is generated from the code scan and checked in. This prose
list is a minimum, not the source of truth.

## 9. Configuration Contract

### 9.1 Project settings

```text
MODEL_DOWNLOAD_SOURCE=huggingface|modelscope
QINGLONG_MODEL_BUNDLE_CACHE=<path>
```

Rules:

- source matching is case-insensitive after trimming;
- an empty source means `huggingface`;
- an unknown source fails before model initialization;
- a relative project-cache path resolves against the project root;
- the default project bundle-cache root is `<project>/modelscope/`;
- source selection is applied before optional framework imports.

`QINGLONG_MODEL_BUNDLE_CACHE` owns ready group directories. It is intentionally
not named `MODELSCOPE_CACHE`.

### 9.2 SDK-native ModelScope settings

The adapter honors the SDK's native settings:

```text
MODELSCOPE_API_TOKEN=<optional>
MODELSCOPE_ENDPOINT=<optional>
MODELSCOPE_CACHE=<optional SDK cache root>
```

`MODELSCOPE_CACHE` remains SDK-owned. It is never interpreted as the
qinglong-captions ready-group store.

The adapter downloads into an explicit unique `local_dir` under the project
staging root. SDK cache layout and project ready-manifest layout therefore do not
compete in one directory.

### 9.3 GUI

The GUI exposes:

- a source selector with Hugging Face selected by default;
- optional ModelScope endpoint;
- optional ModelScope token;
- optional project bundle-cache directory.

The token control is masked in the UI. Existing environment persistence is
plaintext and must not be described as encryption.

Changing the source requires launching a new task process. The GUI does not
hot-switch a process that may already own loaded models.

### 9.4 Provider mapping inventory

A project-owned mapping file records, for each canonical repository:

- repository type;
- ModelScope repository ID;
- verification status;
- exact revision mappings when required;
- provider default revision only when the caller omitted one;
- supported file selection;
- equivalence evidence and verification date;
- notes for unavailable entries;
- for upstream-managed entries, the dependency version constraint, exact
  project input name, upstream name transformation, expanded provider
  repository ID, required package-file metadata, source-order contract, and
  verification date.

Allowed statuses:

- `verified`;
- `unavailable`;
- `upstream-managed`.

An explicit caller revision is handled as follows:

1. use an exact verified revision mapping when present;
2. otherwise try the unchanged revision on ModelScope when that behavior was
   verified for the mapping;
3. otherwise mark the ModelScope attempt unavailable and fall back to Hugging
   Face with the original revision.

A scalar provider default never replaces an explicit caller revision.

### 9.5 Inventory completeness

Every shipped first-party canonical repository must be represented in the
inventory or explicitly covered by a registered upstream-managed mechanism.

CI fails for an unclassified repository. `unavailable` is acceptable and causes
group-level Hugging Face fallback without a doomed ModelScope download.

`upstream-managed` changes who owns resolution; it does not waive availability
evidence. Every upstream-managed project input must expand to concrete provider
repository IDs with a dated metadata result. A hoster's broad support list is
not evidence that the expanded repository exists.

Mapping completeness is checked per coherence group. If one required role in a
group is unavailable, the resolver does not download the other roles from
ModelScope.

## 10. Resolution State Machine

### 10.1 Explicit local group

A caller-supplied local group:

1. bypasses provider mapping and network;
2. must contain every required role and artifact;
3. receives origin `local-explicit`;
4. raises a direct configuration or content error if incomplete;
5. never silently fills missing roles from a remote provider.

### 10.2 Hugging Face mode

For each group:

1. preserve current Hugging Face resolution behavior;
2. pass the caller's canonical IDs and revisions unchanged;
3. do not inspect ModelScope mappings;
4. do not import `modelscope_hub`;
5. preserve existing loaded-cache key and residency behavior.

Where a first-party multi-component call site is migrated to local group
resolution, all roles receive paths from one Hugging Face snapshot identity.
Compatibility tests must prove no user-visible default-mode regression.

### 10.3 ModelScope mode

Resolve every coherence group independently:

1. accept a complete explicit local group;
2. locate, but do not yet accept, a ready project-managed ModelScope candidate;
3. accept an immutable ready candidate directly;
4. for a mutable ready candidate, apply the revalidation and stale-cache rules
   in Section 10.6;
5. otherwise reuse a complete Hugging Face local-only group, with no write;
6. if every role has a usable mapping, attempt the complete group on ModelScope;
7. validate the complete ModelScope group;
8. on a fallback-eligible failure, discard its staging directory and emit one
   warning;
9. retry the complete group through Hugging Face using original canonical IDs
   and caller revisions;
10. return one complete `ResolvedGroup` or an aggregate failure.

A Hugging Face local-only hit before ModelScope network access is reuse, not a
failed fallback, and does not emit a warning.

Other successfully resolved groups remain valid when one group falls back.

### 10.4 Upstream-managed group

An upstream-managed group follows the dependency's documented source order and
cache semantics.

The project records:

- requested preferred source;
- dependency and version;
- configuration applied before import;
- local-path bypass behavior;
- observed outcome when the upstream API exposes it.

Project-level warning cardinality and ready-manifest guarantees apply only when
the project owns the state machine. PaddleX warning and tertiary fallback
behavior are documented exceptions.

### 10.5 Force refresh

Force refresh is per group:

- bypass ready project-managed versions;
- create a new unique staging directory;
- request fresh provider metadata;
- never mutate or replace an existing ready version;
- on ModelScope failure, forward equivalent force behavior to Hugging Face.

Other coherence groups are unaffected.

Force refresh does not evict or hot-swap a runtime object already loaded in the
current process. CLI and GUI force-refresh flows apply before first load or in a
new task process.

### 10.6 Mutable revisions

Branch names such as `main` and `master` are mutable.

For a mutable revision, the resolver first identifies the newest valid ready
candidate without treating it as current. It then attempts ModelScope metadata
revalidation:

1. call the public file-list API;
2. compute the current remote version token;
3. reuse an existing ready version only when that token matches;
4. otherwise materialize a new immutable version.

The version token is computed from the sorted selected file set. Per file, it
uses provider `sha256` when available. `FileInfo.blob_id` is not assumed to be a
content hash because version 0.1.8 does not document that contract. When a
checksum is absent, an opaque metadata tuple may be used only after Phase 0
proves it is stable across repeated unchanged listings. Such a tuple is a change
detector, not content identity.

Revalidation has four distinct outcomes:

- `current`: metadata succeeded and the token matches; reuse with fresh
  provenance;
- `changed`: metadata succeeded and the token differs; do not use the old
  candidate as current;
- `indeterminate`: DNS, connection, timeout, rate-limit, server, authentication,
  or authorization failure prevented a freshness decision; reuse a valid ready
  candidate with `stale-revalidation-unavailable` provenance and one freshness
  warning;
- `definitively-invalid`: the repository or revision is confirmed absent, or
  local validation fails; do not use the old candidate.

An `indeterminate` result with a valid ready candidate is a successful degraded
cache hit. It does not trigger Hugging Face fallback. Only when no valid ready
candidate exists does that same provider failure enter the normal ModelScope to
Hugging Face fallback path.

For an immutable commit or verified immutable tag, the ready version can be used
without revalidation.

An explicit offline request reuses the most recent valid ready mutable version
with `stale-offline` provenance. If none exists, the resolver may try complete
Hugging Face local-only reuse, then fails without attempting either network.

## 11. Provider Adapter

### 11.1 Ownership

The adapter owns:

- source parsing;
- mapping and revision policy;
- coherence-group resolution;
- ModelScope lazy import;
- Hugging Face local-only probing in ModelScope mode;
- manifest validation and publication;
- fallback classification and warning aggregation;
- provenance records.

It does not own:

- Transformers object lifetime;
- ONNX session lifetime;
- PaddleX's official-model inventory;
- device, dtype, attention, or quantization fallback;
- application output caching.

### 11.2 Public operations

The implementation plan may refine names, but the public boundary must support:

```python
resolve_group(group_spec, *, source, force=False) -> ResolvedGroup
resolve_bundle(bundle_spec, *, source, force=False) -> ResolvedBundle
resolve_repo_file(...) -> Path
resolve_repo_file_set(...) -> Mapping[str, Path]
```

Existing ONNX helper names remain compatibility wrappers rather than being
duplicated in a second module.

Consumers do not contain `if source == "modelscope"` branches. Provider choice
stays inside the adapter or an upstream-management bridge.

### 11.3 Lazy imports and credentials

`modelscope_hub` is imported only when a project-managed ModelScope attempt is
needed.

Credentials are provider-specific:

- ModelScope receives only ModelScope token and endpoint settings;
- Hugging Face receives only existing Hugging Face settings;
- logs contain boolean token presence, never values, cookies, or signed URLs.

## 12. ModelScope SDK Boundary

### 12.1 Dependency

Project-managed ModelScope downloads use:

```toml
modelscope-hub==0.1.8
```

The exact pin limits alpha-SDK drift. Upgrading requires rerunning public
contract tests and reviewing release notes.

### 12.2 Public snapshot API

The adapter uses public APIs:

- `HubApi.list_repo_files`;
- `HubApi.download_repo`.

It does not reimplement the SDK's thread pool or depend on private download
internals.

For each repository role:

1. list the exact remote files for the chosen revision;
2. normalize and validate paths;
3. apply the declared selection;
4. download the selected snapshot into a new empty staging role directory;
5. compare the returned directory with the expected selected file set;
6. verify sizes and provider SHA-256 values when supplied;
7. compute and record local SHA-256 values for declared required artifacts;
8. reject an incomplete or corrupt snapshot.

`download_repo` in version 0.1.8 may log individual worker errors and still
return a directory. Returning a path is therefore not success; post-download
manifest validation is the success boundary.

### 12.3 Force behavior

The public Python snapshot API has no `force` parameter. The adapter implements
force safely by downloading into a fresh unique staging directory and bypassing
ready versions. It does not require a custom per-file executor.

### 12.4 Contract tests

Tests pin only public behavior required by the adapter:

- public method signatures used by the project;
- file metadata normalization;
- incomplete snapshot rejection;
- size and SHA-256 verification;
- mutable-revision version-token refresh and checksum-free degradation;
- stable repeated listings for the pinned metadata fields;
- cancellation propagation;
- authentication, not-found, and timeout normalization.

Tests do not assert private SDK implementation details.

An upstream issue should document that `download_repo` does not raise an
aggregate error for failed worker files. The project workaround remains
post-download validation.

## 13. Cache and Publication

### 13.1 Separate roots

Project-managed ModelScope groups use:

```text
<QINGLONG_MODEL_BUNDLE_CACHE>/
  groups/
    <group-key>/
      <manifest-fingerprint>/
```

Hugging Face keeps its existing cache and legacy project directories.

ModelScope cannot natively reuse Hugging Face Hub cache layout as its own SDK
cache. Reuse is implemented by asking Hugging Face Hub for a complete local-only
candidate and returning that path read-only.

### 13.2 Ready manifest

Every project-managed ready group contains `group-manifest.json` with:

- schema version;
- group key and compatibility-contract description;
- one origin;
- canonical and provider IDs;
- requested and resolved revisions;
- mapping inventory version;
- exact role and file list;
- sizes and available remote hashes;
- computed required-artifact hashes;
- manifest fingerprint;
- remote version token and whether it is checksum-backed or metadata-only;
- creation time;
- validation level.

No manifest means not ready. Directory existence alone is never a cache hit.
Freshness is a property of each resolution result. Revalidation does not mutate
the immutable ready directory or its manifest.

### 13.3 Publication on Windows

Publication never replaces a non-empty directory.

1. create a unique staging directory below the project cache filesystem;
2. download and validate every role in the group;
3. write and flush the complete ready manifest inside staging;
4. acquire a lock for the group and fingerprint;
5. if the immutable target already exists and validates, discard staging;
6. otherwise rename staging to the absent final version directory;
7. release the lock.

Readers inspect only final version directories with valid manifests. Staging
directories are never discoverable as candidates.

A crash before rename leaves an ignored staging directory. A later cleanup may
remove stale staging after verifying it is under the configured cache root.

### 13.4 Hugging Face reuse

Hugging Face reuse in ModelScope mode is all-or-nothing per group:

1. resolve every repository role with local-only Hugging Face APIs;
2. validate every declared artifact;
3. accept the group only when every role is complete;
4. otherwise leave all Hugging Face files untouched and continue to ModelScope.

No copy, move, hardlink, or ready manifest is written into the Hugging Face
cache.

### 13.5 Legacy directories

Existing project directories such as `huggingface/Mega-ASR` and current ONNX
tool directories may be accepted as `local-legacy` only when the entire group is
complete.

Legacy provenance is `unverified` unless metadata proves identity. An explicit
immutable revision cannot use a legacy directory without evidence that it
matches that revision.

## 14. Validation and Error Taxonomy

### 14.1 Validation levels

Project-managed resolution performs:

1. path and containment validation;
2. exact required-file validation;
3. size and digest validation where metadata exists;
4. format-specific structural validation that does not create a persistent
   runtime object.

Examples of structural validation:

- parse JSON, TOML, YAML, or CSV required by the consumer;
- validate safetensors headers;
- validate ONNX protobuf and declared external-data presence;
- verify label cardinality when the model output dimension is cheaply known.

### 14.2 Runtime construction boundary

The resolver does not execute arbitrary remote Python code or commit runtime
objects to global caches.

Consumer construction happens only after all groups resolve. A later device,
dependency, or remote-code failure is surfaced directly and does not
automatically switch providers in Revision 4.

A consumer may register a deterministic, side-effect-free structural validator.
Adding first-load runtime fallback or transactional object caching requires a
separate design because it changes memory and object-lifetime behavior.

### 14.3 Fallback-eligible errors

For a project-managed ModelScope group:

- repository, revision, or required file not found;
- authentication or authorization rejection while materializing a new group,
  or while revalidating when no valid ready candidate exists;
- provider timeout or exhausted transient network retry while materializing a
  new group, or while revalidating when no valid ready candidate exists;
- mapping status `unavailable`;
- incomplete returned snapshot;
- provider hash or size mismatch;
- invalid declared artifact structure.

These discard the complete tentative group and retry the complete group through
Hugging Face.

An indeterminate mutable-revision revalidation failure with a valid ready
candidate is not in this list. It returns that candidate with stale provenance
under Section 10.6.

### 14.4 Non-fallback errors

These propagate immediately:

- `KeyboardInterrupt`, `SystemExit`, or user cancellation;
- destination permission denial or invalid path;
- detected disk exhaustion;
- invalid project configuration or caller arguments;
- failure in an explicit local group;
- missing runtime dependency;
- CUDA or CPU out-of-memory;
- unsupported dtype, device, attention, or quantization configuration;
- consumer failures after resolution;
- unknown exceptions not classified as provider acquisition or declared
  structural-content errors.

Because ModelScope 0.1.8 can hide an individual worker exception, an incomplete
snapshot may not retain the exact root cause. The adapter performs local
writability and available-space checks where practical, but a hidden local I/O
failure may still cause one wasteful Hugging Face retry. This is a documented
SDK limitation, not a reason to own another downloader.

### 14.5 Warning and aggregate failure

One project warning includes:

- group key;
- canonical repository IDs;
- failure stage;
- sanitized reason;
- statement that the complete group is being retried through Hugging Face.

If Hugging Face also fails, the project error preserves both sanitized cause
chains and the identities checked.

PaddleX uses upstream warnings for upstream-managed groups.

A stale-cache freshness warning is distinct from a fallback warning. It states
that the local ModelScope group is being reused because freshness could not be
checked; it must not claim that Hugging Face fallback is starting.

## 15. Consumer Integration

### 15.1 Transformers

In ModelScope mode:

- resolve the model's coherence group before any `from_pretrained` call;
- pass local paths to processor, tokenizer, config, and model loads;
- use `local_files_only=True` to prevent hidden Hugging Face traffic;
- keep canonical model ID separate from the local load path for diagnostics and
  existing cache identity;
- do not change current global cache cardinality.

This separation requires an internal signature extension. The implementation
shape is equivalent to:

```python
load_processor(cache_id, ..., load_path=None)
load_model(cache_id, ..., load_path=None)
```

`load_path or cache_id` is passed to `from_pretrained`; only `cache_id`
participates in the existing cache key. In default Hugging Face mode,
`load_path` is absent and the exact existing key values are preserved.

The separation must propagate through `local_vlm_base.py`,
`local_alm_base.py`, and `local_llm_base.py`: their canonical `model_id` remains
the global cache identity while the resolved local path is carried separately
to construction. Their key expressions do not gain source or path components.

This is an internal signature and call-site change, not a cache-semantics
change. Focused tests compare the pre-change and post-change default-mode key
strings or tuples byte-for-byte and assert that the number of resident objects
is unchanged.

Default Hugging Face mode remains on its current code path unless a local-path
conversion is proven behaviorally equivalent by focused tests.

### 15.2 ONNX

One ONNX graph and its external-data files resolve as one group. Independent
models in a multi-model runtime may be separate groups when their session
contracts are independent.

Existing ONNX session construction and input/output validation remain consumer
responsibilities. Provider resolution does not change execution-provider
fallback.

### 15.3 Cross-runtime consumers

LFM-VL, WD Tagger, Gemma templates, audio models, and similar consumers define
groups from actual compatibility contracts rather than from module boundaries.

For example:

- LFM-VL processor and a model whose tensor schema it defines belong together;
- WD Tagger ONNX graph and index-ordered labels belong together;
- an optional hierarchy file may be separate when it does not define indices.

### 15.4 Process-local source stability

The selected source is immutable after the first model framework import.

CLI and GUI launch paths inject configuration before importing providers.
Tests fail when an optional framework is imported before source initialization.

## 16. Dependency and Process Boundaries

### 16.1 Project dependency

Add `modelscope-hub==0.1.8` to normal project dependencies for project-managed
downloads.

Hugging Face mode may have the package installed, but must not import it.

### 16.2 PEP 723 scripts

`module/rewardmodel.py` and `module/waterdetect.py` have inline PEP 723
dependencies and run through `uv run <script>`.

If they transitively import the project adapter, their inline dependency blocks
must also pin:

```toml
modelscope-hub==0.1.8
```

A test enumerates first-party PEP 723 scripts that import the adapter
transitively. Adding only the project dependency is a release-blocking failure.

### 16.3 Paddle dependency boundary

PaddleX's upstream implementation imports the `modelscope` package and owns its
hoster behavior. The project-managed `modelscope-hub` adapter must not be
monkey-patched into PaddleX.

Paddle source configuration is a bridge, not a second implementation.
`paddleocr` is constrained to `>=3.7.0,<3.8.0` until the 3.8 contract is
verified.

Installing `paddleocr-onnx` therefore places two ModelScope SDKs in one
environment:

- PaddleX uses the full `modelscope` package;
- the project adapter uses `modelscope-hub`.

Both ecosystems may recognize `MODELSCOPE_CACHE`. They must not share
qinglong-captions ready-group semantics. The project adapter always supplies its
unique staging `local_dir`; the verified PaddleX 3.7 hoster also supplies an
explicit `local_dir` under PaddleX's official-model cache. A dependency contract
test asserts both explicit-destination calls. If a future PaddleX version stops
passing `local_dir`, its upgrade is blocked pending a cache-isolation review.

## 17. Security and Path Safety

- Validate every provider path as relative.
- Reject drive-prefixed, absolute, empty, traversal, and containment-escaping
  paths.
- Reject normalized duplicates and Windows case-fold collisions.
- Reject symlink-like and special-file materialization.
- Resolve cache roots before creating or moving files.
- Never extract arbitrary provider archives in the project-managed adapter.
- Never execute remote model code during cache publication.
- A ready manifest proves completeness and integrity, not trustworthiness.
- Existing `trust_remote_code` behavior remains an explicit consumer security
  decision.
- Network-deny tests detect undeclared first-load repository traffic.
- Never print tokens, cookies, authorization headers, or signed URLs.

## 18. Delivery Stages

The stages are separate commits with independent rollback. Phase 0 is an
evidence gate, not production implementation. The feature is not declared
complete until Phase 0 and Stages A and B all pass.

### 18.1 Phase 0: dependency and repository evidence

Before production implementation:

1. lock the tested `modelscope-hub` and PaddleOCR/PaddleX versions;
2. call `HubApi.list_repo_files` twice for unchanged representative ModelScope
   repositories containing both regular and large/LFS files;
3. record whether `sha256` is present and stable;
4. treat `blob_id` as opaque unless upstream documentation establishes a
   stronger contract;
5. prove the chosen fallback metadata tuple is stable, or disable mutable ready
   cache matching when checksums are absent;
6. import the installed Paddle package and record the real expanded package
   names for every project-enabled Paddle model and engine;
7. issue metadata-only repository and required-file requests for every expanded
   ModelScope repository ID, including all six PP-OCRv6 ONNX packages and three
   supported PaddleOCR-VL IDs;
8. assert the installed hoster order for both selected sources without replacing
   the hoster classes with fakes;
9. assert both ModelScope SDK paths pass explicit `local_dir` destinations.

The evidence, dependency versions, request date, and exact expanded IDs become a
checked-in Phase 0 artifact consumed by the implementation plan. Any failed item
blocks the dependent design section from implementation; it is not deferred as
an ordinary coding surprise.

The 2026-07-31 design check observed stable `sha256` and `blob_id` values across
two unchanged listings of a representative PP-OCRv6 ONNX repository, with
`blob_id == sha256`, and successful metadata responses for the nine Paddle IDs
listed in Section 8.4. Phase 0 repeats and records this evidence because those
observations are not documented API guarantees.

### 18.2 Stage A: core project-managed resolution

- configuration and GUI launch injection;
- mapping inventory;
- coherence-group data model and state machine;
- public ModelScope snapshot adapter plus post-validation;
- existing ONNX helper integration;
- explicit single-file and snapshot outliers;
- Hugging Face default-mode compatibility tests;
- project-managed cache and manifests.

Stage A may be released behind the opt-in source flag, but documentation must
state its audited coverage.

### 18.3 Stage B: complete first-party coverage

- shared Transformers call-site migration;
- PaddleX source bridge;
- Eureka local-path bridge;
- See-Through outer-loader enforcement;
- direct auxiliary paths;
- PEP 723 dependency coverage;
- static inventory and architecture gates.

The user-facing claim "all first-party download paths support source selection"
is allowed only after Stage B.

### 18.4 Separate follow-up: runtime cache transactions

Partial processor/model object caching and first-load semantic fallback are
existing runtime-lifetime concerns.

They are not required for artifact-source correctness once every consumer loads
only from complete resolved groups. Any change must separately specify:

- eviction versus coexistence;
- RAM and VRAM impact;
- concurrency;
- failed-construction cleanup;
- cache-key compatibility.

## 19. Testing and CI

### 19.1 Core offline suite

Normal CI installs the existing `test` dependency group and runs without public
network.

Provider fakes cover:

- default source and lazy import;
- configuration validation;
- mapping and explicit-revision rules;
- one-origin-per-group construction;
- independent groups resolving from different origins;
- group-level fallback and one warning;
- no redownload of successful independent groups;
- complete Hugging Face local-only reuse;
- rejection of partial cross-source filling;
- public ModelScope snapshot post-validation;
- mutable-revision version-token refresh;
- mutable revalidation timeout plus ready candidate returns stale ModelScope
  provenance without Hugging Face access;
- the same timeout without a ready candidate enters Hugging Face fallback;
- explicit offline reuse and offline cache miss;
- `blob_id` is never treated as a content hash without a pinned contract;
- Windows publication and crash recovery;
- path traversal and collision rejection;
- sanitized aggregate errors.

### 19.2 Default Hugging Face regression

Tests prove that, without new configuration:

- existing canonical IDs and revisions are unchanged;
- the project-managed `modelscope_hub` adapter is not imported;
- current loaded-cache keys and object counts are unchanged;
- existing progress and error behavior remains compatible;
- the authoritative offline suite still passes.

### 19.3 Architecture enforcement

Static tests reject first-party direct download calls outside:

- the project adapter;
- existing ONNX compatibility wrappers;
- registered upstream-managed bridges;
- approved progress-reporting wrappers.

The check covers direct uses of:

- `hf_hub_download`;
- `snapshot_download`;
- `list_repo_files`;
- ModelScope SDK imports and calls;
- direct remote IDs passed to implicit-download wrappers.

Tests and vendored directories are excluded. First-party outer loaders are not
excluded.

### 19.4 Paddle optional-runtime CI

A dedicated contract job:

1. installs `qinglong-captions[paddleocr-onnx]`;
2. runs the focused Paddle source tests marked `optional_runtime`;
3. verifies environment injection occurs before PaddleX import;
4. imports the installed PaddleX classes with connectivity checks disabled and,
   without replacing hoster classes, asserts the actual order is ModelScope,
   Hugging Face, AIStudio, BOS when ModelScope is selected;
5. asserts the installed name-expansion logic produces the exact repository
   names recorded by Phase 0 for every project-enabled Paddle model and engine;
6. uses controlled hoster failures separately to verify warning and fallback
   execution;
7. verifies user `*_model_dir` values bypass source selection;
8. asserts the ModelScope hoster passes an explicit `local_dir`;
9. performs no public model download.

The production `PaddleOCR` constructor path must execute in this job. A skipped
test does not satisfy the gate.

### 19.5 Eureka optional-runtime CI

A dedicated contract job:

1. installs `qinglong-captions[eureka-audio-local]`;
2. runs the focused Eureka tests marked `optional_runtime`;
3. resolves a fake complete local group;
4. denies network;
5. verifies the production wrapper passes the local directory as `model_path`.

The job may monkeypatch model construction after importing the real optional
package, but it must exercise the production wrapper.

### 19.6 PEP 723 CI

Tests verify:

- every transitive adapter user with inline metadata contains the exact pin;
- isolated `uv run` import succeeds;
- Hugging Face mode does not import the project-managed `modelscope_hub`
  adapter.

### 19.7 Release metadata and optional download smoke

A metadata-only network job is required for Phase 0 and before a release that
changes mappings or the Paddle dependency:

```powershell
$env:RUN_MODELSCOPE_METADATA_CHECK = "1"
pytest tests/test_model_hub_metadata.py -q -m network
```

It verifies:

- every `verified` project-managed ModelScope repository;
- every exact upstream-managed Paddle repository ID generated by the installed
  dependency;
- required package-file metadata for each expanded Paddle repository;
- the revision or metadata surface needed by mutable-version tokens;
- the checked-in verification date and dependency version.

This job performs metadata requests only and downloads no weights. Normal
offline PR CI consumes its checked-in evidence and fails when the inventory or
dependency version no longer matches it.

A separate opt-in smoke downloads one deliberately small artifact:

```powershell
$env:RUN_MODELSCOPE_SMOKE = "1"
pytest tests/test_model_hub_smoke.py -q -m network
```

It verifies public snapshot download, manifest validation, ready-cache reuse,
and mutable-revision revalidation. Mirror measurements may also be collected,
but mirror performance is not a release gate for the explicit ModelScope
requirement.

## 20. Acceptance Criteria

Stage A is accepted when all Phase 0 and Stage A rows pass. Stage B and the
overall feature are accepted only when every Phase 0, Stage A, Stage B, and
shared row passes.

| ID | Stage | Criterion |
| --- | --- | --- |
| 1 | Phase 0 | Checked-in evidence verifies the pinned ModelScope metadata fields and never assumes undocumented `blob_id` content-hash semantics |
| 2 | Phase 0 | The installed Paddle dependency expands every enabled model to recorded repository IDs, all IDs pass metadata existence checks, hoster order is verified, and explicit `local_dir` behavior is confirmed |
| 3 | Stage A | Hugging Face remains the default and the project-managed `modelscope_hub` adapter is not imported by the default path |
| 4 | Stage A | Every Stage A path is classified project-managed or upstream-managed |
| 5 | Stage A | Every coherence group documents its compatibility contract, and uncertain boundaries default to one broader group |
| 6 | Stage A | A resolved group contains exactly one origin |
| 7 | Stage A | Independent groups may resolve from different origins |
| 8 | Stage A | A fallback-eligible ModelScope failure retries only the failed complete group through Hugging Face after one project fallback warning |
| 9 | Stage A | Other successful groups are not discarded or redownloaded |
| 10 | Stage A | A complete Hugging Face local-only group is reused without ModelScope or Hugging Face network access |
| 11 | Stage A | Partial caches cannot fill only some roles of a group |
| 12 | Stage A | ModelScope writes nothing under Hugging Face or legacy roots |
| 13 | Stage A | Explicit revisions are preserved unless an exact verified mapping exists |
| 14 | Stage A | A mutable ModelScope ready version is revalidated when metadata is reachable; an indeterminate revalidation reuses a valid ready version with stale provenance and does not contact Hugging Face; without a valid ready version, the same failure enters normal fallback |
| 15 | Stage A | Project-managed snapshots use public ModelScope APIs plus exact post-download validation |
| 16 | Stage A | Existing ONNX helper APIs and cache-directory ownership remain compatible |
| 17 | Stage B | Internal loader signatures may separate canonical cache identity from local load path, while default-mode cache key values remain byte-for-byte identical and resident object counts remain unchanged |
| 18 | Stage B | Every shipped first-party download is project-managed or registered upstream-managed |
| 19 | Stage B | PaddleX receives source configuration before import, keeps user local directories, generates Phase 0-verified repository IDs, and tries Hugging Face first after ModelScope |
| 20 | Stage B | Eureka Audio receives a resolved local `model_path` |
| 21 | Stage B | Every repository has `verified`, `unavailable`, or evidence-bearing `upstream-managed` inventory status |
| 22 | Stage B | `modelscope-hub==0.1.8`, both PaddleOCR dependency entries on `>=3.7.0,<3.8.0`, and every affected PEP 723 dependency are constrained as specified |
| 23 | Stage B | Paddle and Eureka optional-runtime gates execute rather than skip |
| 24 | Shared | The authoritative default Hugging Face test suite passes at the end of each implementation stage |
| 25 | Plan gate | No production code changes before a separately reviewed and approved implementation plan |

## 21. Independent Review Resolution

| Finding | Revision 4 resolution |
| --- | --- |
| F1: provider was treated as identity | Separates `ArtifactIdentity` from origin and replaces whole-bundle coherence with `CoherenceGroup`; one origin per group, groups independent |
| F2: PaddleX behavior was being reimplemented | Delegates production Paddle source selection to PaddleX 3.7 and preserves user local directories |
| F3: custom downloader duplicated an alpha SDK | Uses public `list_repo_files` plus `download_repo`, fresh staging, and exact post-download validation |
| F4: loaded-cache fingerprint could duplicate models | Removes loaded-cache changes and fixes source for the process |
| F5: one feature hid a whole-repository refactor | Splits core resolution, complete path coverage, and runtime cache transactions into separate stages |
| F6: existing ONNX helpers and cache owner were ignored | Extends existing helpers and preserves `resolve_model_cache_dir()` ownership |
| F7: Paddle and Eureka tests would be skipped | Adds dependency-complete optional-runtime CI jobs that execute focused production paths |
| F8: publication, cache-variable, mutable revision, and evidence gaps | Removes remote-code execution from publication, separates SDK and project caches, revalidates mutable revisions, requires metadata evidence, and keeps weight-download smoke optional |
| F9: mutable ready-cache rules conflicted and degraded in the wrong direction | Defines current, changed, indeterminate, and definitively-invalid outcomes; indeterminate revalidation reuses a valid stale ModelScope group without Hugging Face traffic |
| F10: upstream-managed status waived Paddle availability evidence | Requires exact expanded repository IDs, dated metadata evidence, and a release metadata gate |
| F11: uncertain coherence boundaries had no safe default | Defaults uncertainty to the broader group and requires evidence before splitting |
| F12: local load paths require internal loader API changes | Explicitly permits cache-ID/load-path signature separation while requiring identical default key values and object counts |
| F13: Paddle dependency and dual SDK behavior could drift | Adds a `<3.8.0` bound, installed-package contract tests, explicit-destination assertions, and a cache-isolation risk |
| F14: staged delivery had unassigned acceptance criteria | Assigns every criterion to Phase 0, Stage A, Stage B, Shared, or the plan gate |

## 22. Expected Implementation Touchpoints

Core:

- new provider adapter and mapping inventory under `utils/` and `config/`;
- `config/loader.py`;
- GUI environment settings and launch injection.

Existing download owners:

- `module/onnx_runtime/artifacts.py`;
- `module/onnx_runtime/config.py`;
- `module/onnx_runtime/single_model.py`;
- `module/onnx_runtime/multi_model.py`;
- `utils/transformer_loader.py`.

Cache-identity/load-path propagation:

- `module/providers/local_vlm_base.py`;
- `module/providers/local_alm_base.py`;
- `module/providers/local_llm_base.py`;
- every Stage B call site identified by the static scan.

Upstream bridges:

- `module/providers/ocr/paddle.py`;
- `module/providers/local_alm/eureka_audio_local.py`;
- first-party See-Through outer loaders.

Direct paths:

- audio separator;
- WD Tagger and tag highlighting;
- Gemma template loading;
- reward model and water detection;
- SigLIP2;
- Mega-ASR;
- MuScriptor and MuSViT;
- vocal MIDI;
- LFM-VL.

Packaging and CI:

- `pyproject.toml`;
- PEP 723 blocks;
- `.github/workflows/test.yml`;
- project `.gitignore`;
- focused core and optional-runtime tests.

The implementation plan must replace this category list with exact files,
symbols, tests, and stage ownership found by a fresh static scan.

## 23. Residual Risks

- A verified ModelScope mapping can become stale.
- Some providers do not expose immutable commit IDs or SHA-256 for every file.
- `FileInfo.blob_id` is opaque unless a pinned SDK contract proves otherwise;
  checksum-free mutable repositories may require conservative redownload.
- Online mutable-revision revalidation adds a metadata request.
- Reusing a stale ready group after an indeterminate revalidation intentionally
  favors availability over freshness.
- ModelScope 0.1.8 remains alpha and can hide individual worker exceptions.
- PaddleX may continue from Hugging Face to AIStudio or BOS; this is an accepted
  upstream-managed exception.
- PaddleX cache provenance is upstream-managed rather than project-manifest
  provenance.
- Full `modelscope` and `modelscope-hub` coexist in Paddle installations and may
  both observe SDK-native environment variables; explicit-destination contract
  tests guard the current isolation but must be rerun on upgrade.
- Dated repository-existence evidence can expire between releases.
- A third-party wrapper may add a new hidden download in a future release.
- Structural validation cannot prove arbitrary remote code will initialize.
- Strict group fallback can still redownload large data when a genuinely coupled
  role fails.
- A wrongly drawn coherence boundary can cause either skew or excessive
  fallback; the compatibility-contract review rule is therefore mandatory.

## 24. References

- [ModelScope Hub SDK repository](https://github.com/modelscope/modelscope_hub)
- [modelscope-hub 0.1.8](https://pypi.org/project/modelscope-hub/)
- [ModelScope 0.1.8 public API](https://github.com/modelscope/modelscope_hub/blob/v0.1.8/src/modelscope_hub/api.py)
- [ModelScope 0.1.8 download implementation](https://github.com/modelscope/modelscope_hub/blob/v0.1.8/src/modelscope_hub/_download.py)
- [PaddleX 3.7 source flags](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/utils/flags.py)
- [PaddleX 3.7 official model hosters](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/inference/utils/official_models.py)
- [PaddleOCR 3.7.0 package metadata](https://pypi.org/project/paddleocr/3.7.0/)
- [ModelScope PP-OCRv6 medium detection ONNX repository](https://modelscope.cn/models/PaddlePaddle/PP-OCRv6_medium_det_onnx)
- [uv inline script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)
- [Hugging Face cache reference](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)
