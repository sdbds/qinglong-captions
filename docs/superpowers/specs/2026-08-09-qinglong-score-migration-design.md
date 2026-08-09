# Qinglong Score Migration Design

**Status:** Approved in conversation; written specification pending user review

**Date:** 2026-08-09

**Scope:** Replace the image-scoring integration in `qinglong-captions` from
`imscore` to the public `qinglong-score` API, including CLI, GUI, configuration,
result provenance, and tests. This document does not authorize unrelated reward
model additions or changes to `qinglong-score` itself.

## 1. Decision Summary

`qinglong-captions` will switch directly from `imscore` to
`qinglong-score>=0.2.2`. There is no compatibility period for the old
`--repo_id` interface and no fallback to `imscore`.

The integration has two owners:

- `qinglong-score` owns scorer facts: registered scorer names, prompt
  requirements, dtypes, stability, registered checkpoints, default checkpoints,
  loading, and scoring contracts.
- `qinglong-captions` owns application policy: the default scorer, prompt source,
  per-scorer threshold profiles, GUI editing, dataset traversal, ranking,
  partition output, and run reports.

The default scorer is `aesthetic_predictor_v2_5`. It is image-only, so the
default workflow measures image aesthetics without requiring captions. Prompted
scorers remain available. When one is selected and no usable caption or global
prompt is available, the scorer receives an empty string as explicitly chosen
for this migration.

Scoring never partitions images by default. A scorer is partitioned only when
that scorer has an explicit, nonempty threshold list in `config/model.toml`.

## 2. Context

The current implementation imports many concrete `imscore` classes, maps Hugging
Face repository IDs to classes, applies model-specific dtype and HPSv3 input
branches, patches `torch.load`, and moves the loaded object after construction.
Those behaviors conflict with the `qinglong-score` contract:

- a scorer is selected by registered name, not inferred from a repository ID;
- checkpoints are selected separately from the scorer;
- images must already satisfy the public tensor, device, dtype, and range
  contract;
- prompt cardinality must match the batch for prompt-required scorers;
- `load_scorer()` owns device, compute dtype, attention selection, evaluation
  mode, and runtime freezing;
- a loaded scorer must not be moved or reconfigured with `.to()` or `.train()`.

Version 0.2.2 exposes the complete metadata surface required by the integration:

```python
list_scorers() -> tuple[str, ...]
get_scorer_spec(name: str) -> ScorerSpec
list_checkpoints(name: str) -> tuple[RemoteCheckpointSpec, ...]
load_scorer(name, checkpoint=None, *, ...) -> Scorer
```

`list_checkpoints()` is metadata-only and does not load an adapter, access the
network, inspect a cache, probe a device, or initialize CUDA.

## 3. Goals

1. Remove the runtime and packaging dependency on `imscore`.
2. Use only public `qinglong-score` APIs and public immutable metadata types.
3. Make the CLI and GUI use the same scorer and checkpoint vocabulary.
4. Make `aesthetic_predictor_v2_5` the single default scorer.
5. Preserve optional prompt-conditioned scoring, including the chosen empty
   prompt behavior.
6. Batch same-sized images while preserving deterministic output order.
7. Disable threshold partitioning unless the selected scorer has an explicit
   configuration.
8. Allow threshold profiles to be displayed and edited in the GUI.
9. Record enough package, checkpoint, runtime, prompt, and threshold provenance
   to audit a score later.
10. Preserve the repository's optional dependency isolation: opening the GUI
    must not force installation of the reward-model stack.

## 4. Non-Goals

- Retaining `--repo_id`, accepting it as an alias, or printing an extended
  deprecation warning.
- Retaining scorers that are absent from the current `qinglong-score` registry.
- Reimplementing or extending scorer adapters inside `qinglong-captions`.
- Reading `qinglong_score.registry._REGISTRATIONS` or any other private API.
- Calibrating universal score ranges across scorer families.
- Shipping active default thresholds for any scorer.
- Automatically deriving thresholds from a dataset or percentile distribution.
- Adding checkpoint-specific or data-domain-specific threshold profiles in this
  first migration.
- Loading local checkpoint paths through the CLI. The first CLI accepts only a
  registered remote checkpoint identifier or the scorer default.
- Downloading real multi-gigabyte model weights in the normal unit-test suite.

## 5. Dependency Contract

The `reward-model` optional dependency changes from `imscore` to:

```toml
"qinglong-score>=0.2.2"
```

There is no upper bound and no exact Qinglong Score patch pin. Each dependency
resolution may therefore select the latest available compatible release. The
minimum exists because this integration requires public `list_checkpoints()`.

The profile retains `qinglong-captions[torch-base]` so the existing CUDA wheel
source and ProcessRunner torch-backend selection continue to work. It also
retains direct dependencies that the script imports itself, such as Pillow.
Dependencies used only by the removed imscore integration, including the direct
`scipy` entry when no remaining reward-model code imports it, are removed.

The PEP 723 dependency block in `module/rewardmodel.py` is updated consistently.
It must not retain a different torch major/minor contract from `pyproject.toml`.

The dependency is intentionally not promoted to the base GUI environment.
`qinglong-score` requires PyTorch, TorchVision, Transformers, and NumPy 2; making
it a base dependency would defeat the existing per-tool environment boundary and
conflict with profiles that require NumPy 1.x.

## 6. Public CLI

The scoring command becomes:

```text
python -m module.rewardmodel TRAIN_DATA_DIR \
  [--scorer NAME] \
  [--checkpoint IDENTIFIER] \
  [--batch_size N] \
  [--prompt TEXT] \
  [--device auto|cpu|cuda|cuda:N] \
  [--dtype auto|float32|float16|bfloat16]
```

Rules:

- `--scorer` defaults to the configured `reward_model.default_scorer`, falling
  back in code to `aesthetic_predictor_v2_5` when configuration is missing or
  unreadable.
- `--checkpoint` defaults to `None`, which selects the registered default.
- A nonempty checkpoint value is passed unchanged to `load_scorer()` and must
  match one of `list_checkpoints(scorer)`.
- `--repo_id` is removed entirely. Passing it is an argparse error.
- `dtype=auto` is translated to `dtype=None`, allowing `load_scorer()` to select
  `ScorerSpec.default_compute_dtype`.
- Explicit dtype values are converted to `torch.dtype` before loading.
- `device=auto` resolves to `cuda:0` when CUDA is available and otherwise CPU.
- Explicit CUDA indices are validated before model loading.

Unknown scorers, unregistered checkpoints, unsupported dtypes, unavailable
devices, and load failures are fatal configuration/runtime errors. They are not
converted to `None` or hidden behind a generic success exit.

## 7. Runtime Components

The implementation should keep boundaries small rather than adding more policy
to the already large `main()` function.

Runtime and GUI code call `list_scorers()`, `get_scorer_spec()`, and
`list_checkpoints()` directly at their use sites. There is no application catalog
facade, copied registry, or wrapper around values that Qinglong Score already
returns as immutable objects. GUI imports remain lazy to preserve optional
dependency isolation.

### 7.1 Reward policy validation

The existing project config loader parses TOML. Reward-model code reads its
`[reward_model]` mapping and normalizes:

- `default_scorer`;
- zero or more threshold profiles keyed by scorer name.

Application-owned threshold rules are validated independently of model loading
so invalid folder names or duplicate thresholds fail before a checkpoint
download. This is policy validation over the parsed mapping, not a second TOML
parser.

### 7.2 Image conversion and batching

Each image is decoded through Pillow's RGB conversion to an unsigned 8-bit
array, then converted to a contiguous tensor by division by 255. This
construction guarantees:

- shape `[3, H, W]`;
- dtype `torch.float32`;
- RGB channels;
- values in `[0, 1]` before the scorer-requested dtype cast.

`qinglong-captions` does not add another full-tensor finite/range scan. The
public `scorer.score()` boundary in Qinglong Score performs the authoritative
finite/range validation once, after the tensor has reached the scorer's device
and `input_dtype`. Shape, channel count, and decode failures are rejected before
batching; value validation is not duplicated across both packages.

Images from one Lance scan batch are grouped by exact `(H, W)`. Each group is
stacked to `[B, 3, H, W]`, moved directly to `scorer.device`, and cast to
`scorer.input_dtype`. This satisfies Qinglong Score without resizing or padding
unrelated images merely to create a batch.

The code does not retain the old HPSv3 class-name branch or its silent aspect
ratio padding. Model-specific shape limits belong to the Qinglong Score adapter.

Each group result is paired with that group's paths and prompts and collected.
The final deterministic score/path sort defines report order regardless of
batch grouping.

### 7.3 Prompt construction

Prompt source precedence is:

1. nonempty CLI `--prompt`, repeated for each valid image;
2. that image's nonempty caption from the Lance batch;
3. an empty string.

For `ScorerSpec.requires_prompts=True`, the scorer receives exactly one string
per image. The explicit missing-prompt policy for this migration is `empty`:
missing, null, non-string, and blank captions become `""`. This is an intentional
scoring mode, not validation recovery and not a claim that empty text is
semantically equivalent to a real caption. The report records the exact empty
input as `prompt_source="empty"` rather than silently skipping the image or
inventing text.

For image-only scorers, the scorer receives `prompts=None`. The report records
both `prompt` and `prompt_source` as `null`; it does not pretend that ignored
captions participated in the score.

### 7.4 Scoring failures

If a same-shape group call fails, the application records one batch-scoped error
containing the ordered paths in that group. It does not silently retry the same
images through a singleton scoring path. The remaining shape groups continue;
the user can rerun with a smaller `--batch_size` to diagnose a batch-dependent
failure.

- Successfully scored groups remain in the result.
- A failed scoring group counts every path in that group as failed.
- Decode failures are also structured per-image errors.
- If at least one image succeeds, the report is written and the run completes
  with exit code zero and a visible partial-success warning.
- If no image succeeds, the command exits nonzero.
- Model discovery, model loading, invalid configuration, and output-write errors
  remain immediately fatal.

Exceptions are summarized without storing unbounded tracebacks in JSON. The
console retains the detailed diagnostic.

## 8. Scorer and Checkpoint Ownership

`qinglong-captions` must not maintain a second complete scorer/checkpoint table.
At runtime, the selected value is authoritative only after the public Qinglong
Score API accepts it.

All checkpoint rows returned by `list_checkpoints()` are eligible for explicit
selection, including registered rows whose `format` is `"imscore"`. These rows
are checkpoint formats supported by Qinglong Score; exposing them does not
restore the removed `imscore` Python dependency or its removed scorer classes.

Discovery rows with `tracks_updates=True` are presented as tracking checkpoints.
The report always stores the post-load `scorer.checkpoint_identity` once as
`checkpoint`. `tracking_source` is `null` for a pinned checkpoint and contains
the pre-load discovery row with the mutable reference for a tracking selection.
`requested_checkpoint` separately records whether the CLI used an explicit
identifier or the scorer default. Pinned checkpoints therefore do not duplicate
identical source and resolved structures, while tracking runs preserve both
debugging context and a reproducible commit.

## 9. TOML Configuration

The active split configuration moves reward-model policy from
`config/general.toml` to `config/model.toml`. The loader performs shallow
top-level merges, so `[reward_model]` must not remain active in both files.

The default checked-in configuration is:

```toml
[reward_model]
default_scorer = "aesthetic_predictor_v2_5"
```

There are no checked-in active thresholds. Documentation may include commented
examples, but merely installing or upgrading the project must not create quality
folders on the next run.

A user can enable scorer-specific partitioning with:

```toml
[[reward_model.scorers.aesthetic_predictor_v2_5.thresholds]]
name = "low_quality"
max_score = 4.5
color = "bold red"

[[reward_model.scorers.aesthetic_predictor_v2_5.thresholds]]
name = "normal_quality"
max_score = 6.5
color = "bold blue"

[[reward_model.scorers.aesthetic_predictor_v2_5.thresholds]]
name = "best_quality"
max_score = 10.0
color = "bold green"
```

The legacy monolithic `config/config.toml` receives the same default policy and
no active threshold rows so fallback behavior remains consistent.

### 9.1 Threshold validation

For each scorer profile:

- `thresholds` must be a list of tables;
- `name` must be unique and match `^[A-Za-z0-9][A-Za-z0-9_-]*$`;
- `max_score` must be finite and unique;
- `color`, when present, must be a nonempty string;
- entries are normalized into ascending `max_score` order;
- an absent or empty list means partitioning is disabled.

The config path and GUI apply the same validation before assignment or filesystem
changes.

The folder name is `name` with underscores converted to spaces, preserving the
existing visible naming convention while preventing path traversal.

### 9.2 Threshold assignment

For a successful score, the first ascending row satisfying
`score <= max_score` is selected. A score above every configured maximum is
assigned to the final row, preserving the existing upper-bound/catch-all
behavior.

Thresholds apply only to the exact scorer key selected for the run. They are not
borrowed from another scorer, normalized across models, or inferred from a
checkpoint name. They are explicit dataset policy; no default or percentile
inference enables them automatically.

When the selected scorer has no thresholds:

- no quality directory is created;
- no existing quality directory is cleaned;
- no symlink or copy is created;
- ranking and JSON output still run normally.

When thresholds exist, configured directories preserve source-relative paths.
The current symlink-first behavior remains; a permission or platform failure may
fall back to `shutil.copy2` and must be logged visibly.

Using a threshold profile with a tracking checkpoint emits one nonblocking
warning because the same scorer-level thresholds may drift when the tracked
repository changes.

## 10. Result Format

`reward_scores.json` becomes a structured document. It is no longer a nested
path tree whose leaf combines score and prompt into one delimiter-separated
string. No schema version is emitted until a real consumer must distinguish
coexisting formats; `qinglong-captions` does not read or migrate prior reports,
and each completed run atomically replaces its report.

For a directory input, the report remains `<input>/reward_scores.json`. For a
direct `.lance` input, it is written beside the dataset as
`<dataset-stem>.reward_scores.json`; this avoids treating the Lance path as an
output directory.

```json
{
  "run": {
    "qinglong_score_version": "0.2.2",
    "scorer": "aesthetic_predictor_v2_5",
    "requested_checkpoint": null,
    "tracking_source": null,
    "checkpoint": {
      "kind": "remote",
      "adapter": "aesthetic_predictor_v2_5",
      "identifier": "discus0434/aesthetic-predictor-v2-5",
      "format": "official",
      "is_default": true,
      "tracks_updates": false,
      "artifacts": []
    },
    "device": "cuda:0",
    "compute_dtype": "torch.float32",
    "input_dtype": "torch.float32",
    "attention_backend": null,
    "thresholds_enabled": false,
    "thresholds": []
  },
  "summary": {
    "scored": 100,
    "failed": 0,
    "empty_prompt_count": 0
  },
  "items": [
    {
      "rank": 1,
      "path": "images/example.png",
      "score": 7.42,
      "prompt": null,
      "prompt_source": null,
      "bucket": null
    }
  ],
  "errors": []
}
```

The actual remote `checkpoint.artifacts` array contains every public provenance
field from its resolved `RemoteCheckpointSpec`: provider, repository, revision,
filename, SHA-256, and role. It is shown empty above only to keep the example
short. Registry context such as `is_default`, `format`, and `tracks_updates` is
retained because it describes the selection at the recorded Qinglong Score
package version; it does not replace the resolved artifact identity. For a
tracking run, `tracking_source` uses the same public structure before resolution;
otherwise it is `null`.

Items are sorted by descending score. Ties are resolved by normalized relative
path in ascending order. `rank` is assigned after this deterministic sort.

For a prompt-required scorer, `prompt_source` is one of:

- `override`;
- `caption`;
- `empty`.

For an image-only scorer, both `prompt` and `prompt_source` are `null`.

`empty_prompt_count` counts scored items passed to a prompt-required scorer with
`prompt_source="empty"`. It is informational provenance, not an error or skipped
item count. Image-only items do not increment it.

Each error has the stable common fields `scope`, `stage`, `error_type`, and
`message`. A decode error has `scope="item"` and one `path`; a scoring-group
error has `scope="batch"` and a nonempty ordered `paths` array. `stage` is
`decode` or `score`; the JSON does not contain a Python traceback.
`summary.failed` counts affected image paths, not error objects.

The result is written atomically so interruption cannot replace a valid previous
report with partial JSON.

## 11. GUI Design

The existing Image Scoring tool remains in its current wizard tab and visual
language. The migration does not add another page or nest a card inside the
existing tool card.

### 11.1 Scorer and checkpoint controls

- Scorer uses a searchable, editable select.
- Checkpoint uses a searchable, editable select whose options follow the
  selected scorer.
- The first checkpoint option is a visible default choice mapped to `None`.
- Registered checkpoint labels distinguish the default and tracking rows without
  changing the identifier passed to the CLI.
- A refresh icon retries public API discovery and has a tooltip.

The configured default scorer is selected first; remaining discovered names are
shown in deterministic alphabetical order.

The GUI imports Qinglong Score lazily:

- when installed, it populates choices from `list_scorers()` and
  `list_checkpoints()`;
- when absent, scorer choices contain the configured default and scorer names
  already present under `reward_model.scorers`;
- the editable control accepts a newer or custom name even when discovery is
  unavailable;
- checkpoint falls back to the default choice plus editable input;
- runtime validation remains authoritative.

The refresh action retries discovery but does not silently install the full
reward-model profile merely to fill a menu.

### 11.2 Threshold editor

Below the existing scoring controls, the GUI displays an unframed threshold
grid for the selected scorer. Each stable row contains:

- a name input;
- a numeric `max_score` input;
- a color swatch control;
- a delete icon button with tooltip.

A plus button adds a row. A save button uses the familiar save icon. There are
no active rows by default; zero rows means partitioning is disabled.

The GUI keeps at most one visibly marked draft: the threshold profile currently
shown for the selected scorer. The save button is the only action that persists
it: save validates and sorts that profile, then updates only
`reward_model.scorers.<selected>.thresholds`. Switching scorer or leaving the
editor with unsaved changes offers save, discard, and cancel; there are no hidden
drafts for other scorers.

`tomlkit` is used for persistence so unrelated `model.toml` ordering, comments,
and formatting survive. The file replacement is atomic. Deleting the final row
removes or empties that scorer's thresholds and therefore disables partitioning.

Starting a scoring job never writes configuration. It is blocked only when the
selected scorer's displayed profile is dirty, because that profile would affect
the run's partitioning. The user must save or discard it; an invalid draft also
identifies the affected row. Because scorer switching first resolves the active
draft, unrelated hidden unsaved profiles cannot block a clean selected scorer.

The editor uses color swatches rather than requiring users to type Rich color
syntax. Persisted values remain valid Rich styles, for example `bold red` or
`bold #4caf50`.

### 11.3 GUI command construction

The GUI passes:

```text
--scorer=<selected scorer>
--batch_size=<value>
--device=<value>
--dtype=<value>
```

It adds `--checkpoint=<identifier>` only when the user selects or types a
non-default checkpoint. No GUI code emits `--repo_id`.

## 12. PowerShell and Documentation

`2.3.image_reward_model.ps1` changes its configuration keys to:

```powershell
scorer = "aesthetic_predictor_v2_5"
checkpoint = ""
```

It removes the old imscore repository comment inventory, passes `--scorer`, and
passes `--checkpoint` only when nonempty. The script's displayed dependency
profile should say `reward-model`, matching ProcessRunner and project metadata.

`docs/tools/image_scoring.md`, GUI parameter documentation, and relevant README
tables describe:

- scorer versus checkpoint;
- the default image-only scorer;
- empty-prompt behavior for prompted scorers;
- scorer-level TOML thresholds;
- default no-partition behavior;
- structured JSON output and checkpoint provenance;
- tracking-checkpoint reproducibility limits.

## 13. Compatibility and Migration

This is an intentional breaking migration.

- `--repo_id` stops working immediately.
- The old global `[reward_model].quality` list is not migrated automatically to
  every scorer because doing so would silently apply incompatible numeric scales.
- Existing global thresholds are removed from checked-in configuration.
- Existing quality directories are left untouched when no new scorer profile is
  configured; the application does not guess whether they are safe to delete.
- Existing `reward_scores.json` files are replaced by the new structured format
  only when a new run completes its atomic report write.
- Removed imscore-only scorer names are rejected by Qinglong Score with its
  normal `UnknownScorerError` guidance.

## 14. Test Strategy

### 14.1 Dependency tests

- `reward-model` contains `qinglong-score>=0.2.2` without an upper bound.
- `reward-model` and the script metadata contain no `imscore` dependency.
- dependency resolution remains compatible with torch 2.13, TorchVision 0.28,
  Transformers 4.57.x, and NumPy 2 in the isolated profile.
- OpenCLIP is absent from the resolved direct/runtime dependency closure.

### 14.2 Catalog and loading tests

- metadata smoke tests call `list_scorers()`, `get_scorer_spec()`, and
  `list_checkpoints()` without network or CUDA initialization;
- the default scorer is `aesthetic_predictor_v2_5`;
- default checkpoint becomes `None` at the loader boundary;
- explicit checkpoint, device, dtype, and attention defaults reach
  `load_scorer()` correctly;
- loaded scorers are not moved, cast, or switched to training/evaluation mode by
  the application.

### 14.3 Input and scoring tests

- NumPy, PIL, and tensor inputs become exact `[B,3,H,W]` float inputs in range;
- same-sized images batch together;
- output ranking remains deterministic across mixed-size batch grouping;
- a failed group produces one batch-scoped error and is not retried as
  singletons;
- partial image failures are reported without discarding successes;
- all-failed scoring exits nonzero;
- prompt-required scorers receive exactly `B` strings;
- image-only scorers receive `None`;
- override, caption, and empty prompt sources are counted correctly, while
  image-only prompt fields are null.

### 14.4 Threshold tests

- no profile creates no quality directories and performs no cleanup;
- profiles are isolated by scorer;
- thresholds sort by finite unique `max_score`;
- duplicate names, duplicate scores, invalid names, and nonfinite values fail;
- below, between, equal-boundary, and above-highest scores map deterministically;
- tracking checkpoint plus thresholds emits one warning;
- TOML updates preserve unrelated sections and comments.

### 14.5 Result tests

- the structured report contains required run, summary, item, and error fields;
- items sort by descending score and path tie-breaker;
- remote checkpoint provenance is serialized from public dataclasses;
- pinned checkpoints are stored once, while tracking checkpoints add their
  mutable source reference beside the resolved checkpoint;
- batch failure counts paths rather than error objects;
- JSON replacement is atomic;
- output contains no delimiter-encoded score/prompt leaf strings.

### 14.6 GUI and wrapper tests

- discovery-present and discovery-absent GUI states both render;
- editable fallback values reach the CLI;
- checkpoint choices change with scorer;
- switching scorer with a dirty displayed profile offers save, discard, and
  cancel without retaining hidden drafts;
- save persists only the selected profile, while start never persists and blocks
  only on a dirty selected profile;
- PowerShell and GUI emit `--scorer` and optional `--checkpoint`;
- no wrapper emits `--repo_id`.

Normal tests use a fake public scorer and fake immutable checkpoint metadata. One
small optional-runtime smoke test exercises the installed `qinglong-score>=0.2.2`
metadata API. Real checkpoints remain outside the normal suite.

## 15. Acceptance Criteria

The migration is complete when all of these are true:

1. `rg imscore` finds no active reward-model dependency, import, registry, or
   wrapper argument; historical documentation references may remain only when
   clearly labeled as history.
2. CLI and GUI default to `aesthetic_predictor_v2_5`.
3. CLI and GUI select scorer and checkpoint separately.
4. Every runtime model fact comes from a public Qinglong Score API.
5. `dtype=auto` delegates to Qinglong Score.
6. No code reconfigures a loaded scorer.
7. Mixed-size batches score in deterministic order.
8. Prompt-required scorers preserve the selected empty-string behavior.
9. No configured thresholds means no partitioning filesystem side effects.
10. GUI users can create, edit, color, delete, and save thresholds per scorer.
11. The JSON report records package version and one resolved checkpoint, adds
    source metadata only for tracking selections, and ranks scores
    deterministically.
12. A tracking checkpoint's requested reference and resolved revision are both
    auditable from the report.
13. Unit, GUI, wrapper, dependency, and metadata smoke tests pass without a real
    checkpoint download.

## 16. Implementation Boundaries

Expected production edits are limited to the reward-model integration and its
direct surfaces:

- `pyproject.toml`;
- `module/rewardmodel.py` and narrowly scoped reward helpers if extraction keeps
  responsibilities clearer;
- `config/model.toml`, `config/general.toml`, and legacy `config/config.toml`;
- `gui/wizard/step6_tools.py` and directly required i18n/config helpers;
- `2.3.image_reward_model.ps1`;
- image-scoring documentation;
- focused tests.

The implementation must not edit `E:\Code\qinglong-score`, vendor scorer code,
or unrelated dirty workspace files. Qinglong Score 0.2.2 is treated as the
published upstream contract.
