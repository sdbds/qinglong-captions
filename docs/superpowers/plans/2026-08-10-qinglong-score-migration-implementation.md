# Qinglong Score Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the complete `imscore` image-scoring surface with `qinglong-score>=0.2.2`, including scorer/checkpoint selection, deterministic batched scoring, opt-in thresholds, structured reports, GUI editing, wrappers, and documentation.

**Architecture:** Keep scorer/checkpoint facts in Qinglong Score and add one lightweight `module.reward_policy` module for application-owned TOML policy. Rewrite `module.rewardmodel` as a thin CLI plus testable image, batch, report, and partition functions; the GUI imports only reward policy at module load and discovers Qinglong Score lazily. Model loading remains a direct call to the public Qinglong Score API and loaded scorers are never reconfigured.

**Tech Stack:** Python 3.11+, PyTorch 2.13, Qinglong Score 0.2.2+, Pillow, Lance/PyArrow, TOML/tomlkit, Rich, NiceGUI, pytest, PowerShell.

## Global Constraints

- The dependency is `qinglong-score>=0.2.2` with no upper bound or exact patch pin.
- The default scorer is exactly `aesthetic_predictor_v2_5`.
- Remove `--repo_id` without an alias or compatibility path.
- `dtype=auto` passes `None` to `qinglong_score.load_scorer()`.
- The application never calls `.to()`, `.eval()`, or `.train()` on a loaded scorer.
- Missing prompts for prompt-required scorers are intentionally passed as `""` and reported with `prompt_source="empty"`.
- No configured thresholds means no quality-directory creation, cleanup, symlink, or copy.
- GUI import must not import Qinglong Score, PyTorch, or initialize CUDA.
- Do not edit `E:\Code\qinglong-score` or unrelated dirty workspace changes.

---

### Task 1: Reward Policy Contract

**Files:**
- Create: `module/reward_policy.py`
- Create: `tests/test_reward_policy.py`
- Modify: `config/model.toml`
- Modify: `config/general.toml`
- Modify: `config/config.toml`

**Interfaces:**
- Produces: `DEFAULT_SCORER`, `Threshold`, `RewardPolicy`, `normalize_thresholds()`, `parse_reward_policy()`, `load_reward_policy()`, `assign_threshold()`, and `save_threshold_profile()`.
- Consumes: the existing `config.loader.load_config()` TOML loader and `tomlkit` for comment-preserving writes.

- [ ] **Step 1: Write failing policy/default tests**

```python
def test_missing_reward_config_uses_default_without_thresholds():
    policy = parse_reward_policy({})
    assert policy.default_scorer == "aesthetic_predictor_v2_5"
    assert policy.thresholds_for("aesthetic_predictor_v2_5") == ()

def test_profiles_are_isolated_and_sorted():
    policy = parse_reward_policy({"reward_model": {"scorers": {
        "alpha": {"thresholds": [
            {"name": "high", "max_score": 8.0, "color": "bold green"},
            {"name": "low", "max_score": 3.0, "color": "bold red"},
        ]},
        "beta": {"thresholds": [{"name": "only", "max_score": 1.0}]},
    }}})
    assert [row.name for row in policy.thresholds_for("alpha")] == ["low", "high"]
    assert [row.name for row in policy.thresholds_for("beta")] == ["only"]
```

- [ ] **Step 2: Run policy tests and verify RED**

Run: `pytest -q tests/test_reward_policy.py`

Expected: collection fails because `module.reward_policy` does not exist.

- [ ] **Step 3: Implement immutable policy parsing and assignment**

```python
DEFAULT_SCORER = "aesthetic_predictor_v2_5"

@dataclass(frozen=True, slots=True)
class Threshold:
    name: str
    max_score: float
    color: str = "white"

    @property
    def folder_name(self) -> str:
        return self.name.replace("_", " ")

@dataclass(frozen=True, slots=True)
class RewardPolicy:
    default_scorer: str
    profiles: Mapping[str, tuple[Threshold, ...]]

    def thresholds_for(self, scorer: str) -> tuple[Threshold, ...]:
        return self.profiles.get(scorer, ())

def assign_threshold(score: float, rows: Sequence[Threshold]) -> Threshold | None:
    if not rows:
        return None
    return next((row for row in rows if score <= row.max_score), rows[-1])
```

Validation rejects malformed tables, unsafe/duplicate names, duplicate or nonfinite scores, and blank colors before model loading.

- [ ] **Step 4: Add failing persistence and config-placement tests**

```python
def test_save_profile_preserves_unrelated_toml(tmp_path):
    path = tmp_path / "model.toml"
    path.write_text("# keep\n[other]\nvalue = 7\n", encoding="utf-8")
    save_threshold_profile(path, "alpha", [{"name": "low", "max_score": 2.5, "color": "bold red"}])
    text = path.read_text(encoding="utf-8")
    assert "# keep" in text
    assert "value = 7" in text
    assert 'name = "low"' in text
```

Also parse checked-in `model.toml`, `general.toml`, and `config.toml`; assert only model/legacy config contain `[reward_model]`, the default scorer is present, and no active thresholds are checked in.

- [ ] **Step 5: Implement atomic tomlkit persistence and move checked-in policy**

Use `tomlkit.parse()`, update only `reward_model.scorers.<scorer>.thresholds`, write a sibling temporary file, then call `os.replace()`. Delete the old global `quality` section from `general.toml`; add only `default_scorer = "aesthetic_predictor_v2_5"` to `model.toml` and legacy `config.toml`.

- [ ] **Step 6: Run policy tests and commit**

Run: `pytest -q tests/test_reward_policy.py`

Expected: PASS.

```powershell
git add module/reward_policy.py tests/test_reward_policy.py config/model.toml config/general.toml config/config.toml
git commit -m "feat: add scorer threshold policy"
```

### Task 2: Qinglong Score Runtime and Batched Input

**Files:**
- Rewrite: `module/rewardmodel.py`
- Delete: `tests/test_rewardmodel_hpsv3_input.py`
- Create: `tests/test_rewardmodel_runtime.py`

**Interfaces:**
- Consumes: `RewardPolicy`, `Threshold`, `load_reward_policy()`, and `assign_threshold()` from Task 1; public `qinglong_score.list_checkpoints()`, `get_scorer_spec()`, and `load_scorer()`.
- Produces: `SourceImage`, `ScoredImage`, `RunError`, `resolve_device()`, `resolve_dtype()`, `decode_image()`, `select_prompt()`, `score_source_batch()`, `setup_parser()`, `run()`, and `main()`.

- [ ] **Step 1: Write failing CLI and prompt tests**

```python
def test_parser_replaces_repo_id_with_scorer_and_checkpoint():
    parser = setup_parser()
    args = parser.parse_args(["dataset", "--scorer=pickscore", "--checkpoint=repo/model"])
    assert args.scorer == "pickscore"
    assert args.checkpoint == "repo/model"
    with pytest.raises(SystemExit):
        parser.parse_args(["dataset", "--repo_id=old/model"])

def test_prompt_required_uses_override_caption_then_empty():
    assert select_prompt("caption", "override", True) == ("override", "override")
    assert select_prompt(["caption"], "", True) == ("caption", "caption")
    assert select_prompt([], "", True) == ("", "empty")
    assert select_prompt(["ignored"], "", False) == (None, None)
```

- [ ] **Step 2: Run runtime tests and verify RED**

Run: `pytest -q tests/test_rewardmodel_runtime.py`

Expected: FAIL because the new parser/runtime functions do not exist.

- [ ] **Step 3: Implement CLI normalization and exact tensor decode**

```python
def resolve_dtype(value: str) -> torch.dtype | None:
    return {"auto": None, "float32": torch.float32,
            "float16": torch.float16, "bfloat16": torch.bfloat16}[value]

def decode_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().float().div_(255)
```

`resolve_device()` validates explicit CUDA indices but does not change the global CUDA device. `setup_parser()` accepts only the approved CLI surface and leaves an omitted scorer as `None` so `run()` can apply the configured default.

- [ ] **Step 4: Add failing same-size grouping and batch-failure tests**

Use real temporary PNGs and a fake scorer whose `score()` records tensor shapes. Assert same sizes produce one `[B,3,H,W]` call, mixed sizes produce separate calls, image-only calls receive `None`, prompted calls receive exactly `B` strings, and one failing group produces one `scope="batch"` error without singleton retries.

- [ ] **Step 5: Implement `score_source_batch()`**

Decode records concurrently, retain per-item decode errors, group successful tensors by `(H, W)`, move/convert each stack directly to `scorer.device` and `scorer.input_dtype`, call `scorer.score()` once per group under inference mode, reject nonfinite or wrong-cardinality output, and collect successful groups without rebuilding input order.

- [ ] **Step 6: Add failing public-loader boundary test**

Install a complete fake `qinglong_score` module in `sys.modules` with immutable scorer/checkpoint metadata and a fake scorer whose `.to()`, `.eval()`, and `.train()` raise assertions. Run the CLI orchestration against a fake dataset and assert:

```python
assert load_calls == [{
    "name": "aesthetic_predictor_v2_5",
    "checkpoint": None,
    "device": "cpu",
    "dtype": None,
    "attention_backend": "auto",
}]
```

- [ ] **Step 7: Implement direct public API loading and Lance iteration**

Import `qinglong_score` only inside `run()`, call its public APIs directly, select the tracking discovery row before loading, and never mutate the returned scorer. Resolve directories or direct `.lance` inputs, request `captions` only when present, and convert each Arrow row to one `SourceImage`.

- [ ] **Step 8: Run runtime tests and commit**

Run: `pytest -q tests/test_rewardmodel_runtime.py`

Expected: PASS.

```powershell
git add module/rewardmodel.py tests/test_rewardmodel_runtime.py tests/test_rewardmodel_hpsv3_input.py
git commit -m "feat: score images with qinglong score"
```

### Task 3: Structured Report and Optional Partitioning

**Files:**
- Modify: `module/rewardmodel.py`
- Create: `tests/test_rewardmodel_outputs.py`

**Interfaces:**
- Consumes: `ScoredImage`, `RunError`, `Threshold`, and `assign_threshold()`.
- Produces: `serialize_checkpoint()`, `build_report()`, `result_path_for_input()`, `write_json_atomic()`, and `apply_thresholds()`.

- [ ] **Step 1: Write failing deterministic report tests**

```python
def test_report_sorts_scores_then_paths_and_assigns_rank():
    report = build_report(...items=[
        ScoredImage("b.png", None, None, 2.0),
        ScoredImage("a.png", None, None, 2.0),
        ScoredImage("c.png", None, None, 3.0),
    ])
    assert [(item["rank"], item["path"]) for item in report["items"]] == [
        (1, "c.png"), (2, "a.png"), (3, "b.png")]
    assert "schema_version" not in report
```

Add literal assertions for package/runtime fields, null image-only prompts, `empty_prompt_count`, batch path counting, one resolved checkpoint, and a non-null `tracking_source` only for tracking selections.

- [ ] **Step 2: Run output tests and verify RED**

Run: `pytest -q tests/test_rewardmodel_outputs.py`

Expected: FAIL because report functions are missing.

- [ ] **Step 3: Implement checkpoint serialization and report building**

Serialize public remote dataclasses with `dataclasses.asdict()`, add `kind="remote"`, stringify paths/dtypes, normalize report paths to POSIX separators, assign buckets, sort by `(-score, path)`, then assign 1-based ranks. Keep Python tracebacks out of error objects.

- [ ] **Step 4: Add failing atomic-write and no-threshold filesystem tests**

Create an existing report, attempt to write a payload containing NaN with `allow_nan=False`, and assert the existing report is unchanged. Create marker files in would-be quality directories, call `apply_thresholds(..., thresholds=())`, and assert no directory or marker changes occur.

- [ ] **Step 5: Implement atomic report writes and opt-in partitioning**

Write JSON to a sibling temporary file and replace with `os.replace()`. With thresholds, create configured roots, preserve source-relative paths, choose the first matching row/final catch-all, try `Path.symlink_to()`, and visibly fall back to `shutil.copy2()` on permission/platform failure. Emit one warning when thresholds are combined with a tracking checkpoint.

- [ ] **Step 6: Integrate reports/partitioning into `run()` and test exit codes**

Assert partial success writes a report and returns zero, all-failed scoring writes diagnostics and returns nonzero, directory input writes `<dir>/reward_scores.json`, and direct Lance input writes `<stem>.reward_scores.json` beside the dataset.

- [ ] **Step 7: Run output plus runtime tests and commit**

Run: `pytest -q tests/test_rewardmodel_outputs.py tests/test_rewardmodel_runtime.py`

Expected: PASS.

```powershell
git add module/rewardmodel.py tests/test_rewardmodel_outputs.py tests/test_rewardmodel_runtime.py
git commit -m "feat: report and partition reward scores"
```

### Task 4: Dependency Profile, Checked-in Configuration, and PowerShell

**Files:**
- Modify: `pyproject.toml`
- Modify: `module/rewardmodel.py`
- Modify: `2.3.image_reward_model.ps1`
- Create: `tests/test_rewardmodel_dependency_profile.py`

**Interfaces:**
- Consumes: the CLI created in Task 2.
- Produces: a resolvable `reward-model` extra and matching PEP 723/PowerShell entrypoints.

- [ ] **Step 1: Write failing dependency and wrapper tests**

Parse `pyproject.toml` and assert the reward extra contains `qinglong-score>=0.2.2`, contains no `imscore`, `open-clip-torch`, or direct `scipy`, and keeps `qinglong-captions[torch-base]`. Read the PEP 723 block and PowerShell wrapper and assert `torch==2.13.0`, `--scorer`, optional `--checkpoint`, `reward-model`, and no `--repo_id`.

- [ ] **Step 2: Run dependency tests and verify RED**

Run: `pytest -q tests/test_rewardmodel_dependency_profile.py`

Expected: FAIL on the current `imscore` dependency and wrapper arguments.

- [ ] **Step 3: Update all dependency and wrapper surfaces**

Replace only the reward-model entries in the dirty `pyproject.toml`. Update the PEP 723 block consistently. Change PowerShell configuration to:

```powershell
scorer = "aesthetic_predictor_v2_5"
checkpoint = ""
```

Pass `--scorer`, conditionally pass `--checkpoint`, print `reward-model`, and run the script with that dependency profile.

- [ ] **Step 4: Run tests and metadata checks, then commit**

Run: `pytest -q tests/test_rewardmodel_dependency_profile.py`

Run: `python -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); print('valid')"`

Expected: PASS and `valid`.

```powershell
git add pyproject.toml module/rewardmodel.py 2.3.image_reward_model.ps1 tests/test_rewardmodel_dependency_profile.py
git commit -m "build: switch reward profile to qinglong score"
```

### Task 5: GUI Scorer, Checkpoint, and Threshold Editing

**Files:**
- Modify: `gui/wizard/step6_tools.py`
- Modify: `gui/utils/i18n.py`
- Create: `tests/test_rewardmodel_gui.py`

**Interfaces:**
- Consumes: `DEFAULT_SCORER`, `load_reward_policy()`, and `save_threshold_profile()` without importing PyTorch or Qinglong Score at module import.
- Produces: searchable `reward_scorer`/`reward_checkpoint` controls, lazy public discovery, one active threshold draft, and approved CLI arguments.

- [ ] **Step 1: Write failing lazy-import and command tests**

In a subprocess, import `gui.wizard.step6_tools` and assert newly imported modules contain neither `qinglong_score` nor `torch`. Configure a `ToolsStep` with simple namespace controls, capture `run_job()`, and assert arguments include `--scorer=aesthetic_predictor_v2_5`, optional `--checkpoint=...`, batch/device/dtype, and no `--repo_id`.

- [ ] **Step 2: Run GUI tests and verify RED**

Run: `pytest -q tests/test_rewardmodel_gui.py`

Expected: FAIL because the GUI still exposes repository IDs.

- [ ] **Step 3: Implement lazy scorer/checkpoint discovery controls**

Replace `REWARD_MODELS` with the configured default and profile names. Inside render/refresh event handlers only, directly import `qinglong_score`, call `list_scorers()` and `list_checkpoints(selected)`, label default/tracking rows, and keep editable fallback controls when import/discovery fails. Add an icon-only refresh button with tooltip.

- [ ] **Step 4: Add failing active-draft state and persistence tests**

Assert loaded rows follow the selected scorer, editing marks only the displayed profile dirty, dirty scorer switching supports save/discard/cancel, saving calls `save_threshold_profile()` for the selected scorer, and `_start_reward()` blocks only while that displayed profile is dirty.

- [ ] **Step 5: Implement the threshold grid**

Render an unframed responsive row list with name input, numeric `max_score`, `ui.color_input` swatch, icon-only delete button/tooltips, plus button, and save button. Keep at most one active draft; switching or leaving resolves it through save/discard/cancel. Persist colors as `bold #RRGGBB` and show validation errors on the affected row.

- [ ] **Step 6: Add translations and GUI behavior tests**

Add scorer, checkpoint, default/tracking labels, thresholds, add/delete/save/discard/cancel, dirty-profile, validation, and discovery-refresh strings for English, Chinese, Japanese, and Korean. Extend tests to ensure every new key is nonempty in all four languages.

- [ ] **Step 7: Run GUI tests and commit**

Run: `pytest -q tests/test_rewardmodel_gui.py tests/test_gui_i18n.py`

Expected: PASS.

```powershell
git add gui/wizard/step6_tools.py gui/utils/i18n.py tests/test_rewardmodel_gui.py
git commit -m "feat: configure image scorers in gui"
```

### Task 6: User Documentation and Migration Cleanup

**Files:**
- Modify: `docs/tools/image_scoring.md`
- Modify: `docs/tools/image_scoring.en.md`
- Modify: `gui/PARAMETERS.md`
- Modify: `docs/superpowers/specs/2026-08-09-qinglong-score-migration-design.md`

**Interfaces:**
- Consumes: final CLI, configuration, GUI, and report behavior from Tasks 1-5.
- Produces: bilingual operational documentation and an approved/implemented design status.

- [ ] **Step 1: Update concise user documentation**

Document `--scorer` versus `--checkpoint`, the default image-only scorer, intentional empty prompts for prompted scorers, per-scorer TOML examples, default no-partition behavior, structured output/provenance, and tracking drift. Update GUI parameters without retaining imscore repository examples.

- [ ] **Step 2: Mark the approved specification implemented**

Change its status to `Approved and implemented` only after the production and focused tests pass.

- [ ] **Step 3: Search active surfaces for stale compatibility**

Run: `rg -n "imscore|--repo_id|RE-N-Y/hpsv3|RE-N-Y/pickscore" module/rewardmodel.py 2.3.image_reward_model.ps1 pyproject.toml gui/wizard/step6_tools.py docs/tools/image_scoring.md docs/tools/image_scoring.en.md gui/PARAMETERS.md`

Expected: no matches.

- [ ] **Step 4: Commit documentation**

```powershell
git add docs/tools/image_scoring.md docs/tools/image_scoring.en.md gui/PARAMETERS.md docs/superpowers/specs/2026-08-09-qinglong-score-migration-design.md
git commit -m "docs: document qinglong score workflow"
```

### Task 7: Full Verification and GUI Visual QA

**Files:**
- Verify only; modify a production file only through a new failing regression test if verification exposes a defect.

**Interfaces:**
- Consumes: every preceding task.
- Produces: fresh test, metadata, import-isolation, and visual evidence.

- [ ] **Step 1: Run the focused migration suite**

Run: `pytest -q tests/test_reward_policy.py tests/test_rewardmodel_runtime.py tests/test_rewardmodel_outputs.py tests/test_rewardmodel_dependency_profile.py tests/test_rewardmodel_gui.py tests/test_gui_i18n.py`

Expected: PASS.

- [ ] **Step 2: Run adjacent regression tests**

Run: `pytest -q tests/test_process_runner_uv_patch.py tests/test_process_runner_console_env.py tests/test_gui_main_lazy_import.py`

Expected: PASS.

- [ ] **Step 3: Run static and import checks**

Run: `ruff check module/rewardmodel.py module/reward_policy.py gui/wizard/step6_tools.py tests/test_reward_policy.py tests/test_rewardmodel_runtime.py tests/test_rewardmodel_outputs.py tests/test_rewardmodel_dependency_profile.py tests/test_rewardmodel_gui.py`

Run: `python -m module.rewardmodel --help`

Expected: lint clean and help lists `--scorer`/`--checkpoint` without `--repo_id`.

- [ ] **Step 4: Exercise published Qinglong Score metadata without weights**

Run the reward-model environment and print `qinglong_score.__version__`, `list_scorers()`, and `list_checkpoints("aesthetic_predictor_v2_5")`. Assert version is at least 0.2.2 and discovery performs no checkpoint download.

- [ ] **Step 5: Start the NiceGUI server and inspect the Tools reward panel**

Start `python -m gui.main` on an available local port, open `/tools`, select the Image Scoring tab, and verify desktop/mobile layouts show scorer, checkpoint, refresh, batch/device/dtype, and threshold controls without overlap. Exercise scorer switching and dirty save/discard/cancel behavior and capture one screenshot.

- [ ] **Step 6: Review final diff and commit any test-driven fixes**

Run: `git diff --check`

Run: `git status --short`

Confirm unrelated pre-existing dirty files remain untouched and all migration commits contain only their named files.
