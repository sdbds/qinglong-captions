# See-through Batch Tool Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 see-through 以“工具页目录批处理 + 独立 CLI/backend”的形态正式接入仓库，按 `LayerDiff(全目录) -> 卸载 -> Marigold(全目录) -> 卸载 -> Postprocess(全目录)` 执行；`output_dir` 直接作为本次任务的最终产物目录；断点续跑只根据当前 `output_dir` 内的 outputs 文件夹判断阶段，不再维护行级 merge state。

**Architecture:** `step6_tools.py` 只负责参数采集与发车；`module/see_through/cli.py` 负责 CLI 参数和配置装配；`module/see_through/runner.py` 负责目录扫描、配置指纹、`output_dir` 身份校验、规范化相对路径输出键、outputs 阶段判定和 phase-batch 调度；`module/see_through/model_manager.py` 负责 pipeline 生命周期和显存释放；`module/see_through/extracted/*` 与 `pipelines/*` 只保留最小算法子集，不引整仓。

**Tech Stack:** Python 3.11, NiceGUI, `torch`, `diffusers`, `transformers`, `psd-tools`, `huggingface_hub`, `opencv-python`, `pytest`, existing `process_runner` / `model.toml` config infrastructure.

---

## File Map

**Create**

- `E:\Code\qinglong-captions\module\see_through\__init__.py`
- `E:\Code\qinglong-captions\module\see_through\cli.py`
- `E:\Code\qinglong-captions\module\see_through\runtime.py`
- `E:\Code\qinglong-captions\module\see_through\model_manager.py`
- `E:\Code\qinglong-captions\module\see_through\runner.py`
- `E:\Code\qinglong-captions\module\see_through\postprocess.py`
- `E:\Code\qinglong-captions\module\see_through\extracted\layerdiff_core.py`
- `E:\Code\qinglong-captions\module\see_through\extracted\marigold_core.py`
- `E:\Code\qinglong-captions\module\see_through\extracted\postprocess_core.py`
- `E:\Code\qinglong-captions\module\see_through\pipelines\layerdiff.py`
- `E:\Code\qinglong-captions\module\see_through\pipelines\marigold.py`
- `E:\Code\qinglong-captions\tests\test_see_through_config.py`
- `E:\Code\qinglong-captions\tests\test_see_through_cli.py`
- `E:\Code\qinglong-captions\tests\test_see_through_runtime.py`
- `E:\Code\qinglong-captions\tests\test_see_through_model_manager.py`
- `E:\Code\qinglong-captions\tests\test_see_through_runner.py`
- `E:\Code\qinglong-captions\tests\test_see_through_tools_step.py`

**Modify**

- `E:\Code\qinglong-captions\config\model.toml`
- `E:\Code\qinglong-captions\pyproject.toml`
- `E:\Code\qinglong-captions\gui\utils\process_runner.py`
- `E:\Code\qinglong-captions\gui\utils\i18n.py`
- `E:\Code\qinglong-captions\gui\wizard\step6_tools.py`
- `E:\Code\qinglong-captions\tests\test_process_runner_uv_patch.py`

**Reference**

- `E:\Code\qinglong-captions\docs\superpowers\specs\2026-04-02-see-through-batch-tools-design.md`
- `C:\Users\QINGLO~1\AppData\Local\Temp\see-through-codex\inference\scripts\inference_psd.py`
- `C:\Users\QINGLO~1\AppData\Local\Temp\see-through-codex\common\utils\inference_utils.py`

### Responsibilities

- `config/model.toml`: 通过 `[see_through]` 承载 see-through 默认配置。
- `module/see_through/cli.py`: CLI 参数、配置覆盖、后端入口。
- `module/see_through/runtime.py`: `device/dtype/attention_backend` 解析与自动回退。
- `module/see_through/model_manager.py`: phase 内 pipeline 复用与 phase 边界显存卸载。
- `module/see_through/runner.py`: 输入扫描、`config_fingerprint`、`output_dir` 身份校验、规范化相对路径输出键、outputs 阶段判定、phase-batch 调度、失败继续、日志汇总。
- `module/see_through/extracted/*`: 上游必要算法子集，带来源标注，不保留上游全局 pipeline 状态。
- `module/see_through/pipelines/*`: 算法包装层，给 runner 提供稳定调用面。
- `gui/wizard/step6_tools.py`: 工具 tab 和参数映射。
- `gui/utils/process_runner.py`: `SCRIPT_REGISTRY` 和 `uv extra` 接线。

## Task 1: Add Config Surface And Launcher Wiring

**Files:**
- Modify: `E:\Code\qinglong-captions\config\model.toml`
- Modify: `E:\Code\qinglong-captions\pyproject.toml`
- Modify: `E:\Code\qinglong-captions\gui\utils\process_runner.py`
- Modify: `E:\Code\qinglong-captions\tests\test_process_runner_uv_patch.py`
- Test: `E:\Code\qinglong-captions\tests\test_see_through_config.py`

- [ ] **Step 1: Add a `[see_through]` section to `model.toml`**

```toml
[see_through]
repo_id_layerdiff = "layerdifforg/seethroughv0.0.2_layerdiff3d"
repo_id_depth = "24yearsold/seethroughv0.0.1_marigold"
resolution = 1280
dtype = "bfloat16"
offload_policy = "delete"
skip_completed = true
continue_on_error = true
save_to_psd = true
tblr_split = false
limit_images = 0
force_eager_attention = false
output_dir = "workspace/see_through_output"
```

- [ ] **Step 2: Keep config discovery unchanged**

Expected: no new split file, no `loader.py` change, and CLI/backend continue to read see-through defaults from the existing `model.toml` merge path.

- [ ] **Step 3: Add the dedicated `see-through` dependency profile and process runner entry**

```python
SCRIPT_REGISTRY["module.see_through.cli"] = ("./module/see_through/cli.py", "see-through")
```

Expected: GUI 发车时自动补 `see-through` extra，不要求 `step6_tools.py` 单独硬编码 `runner_kwargs`.

- [ ] **Step 4: Write failing tests for config loading and process-runner wiring**

```python
def test_split_loader_reads_seethrough_file(tmp_path):
    (tmp_path / "model.toml").write_text("[see_through]\nresolution=1024\n", encoding="utf-8")
    config = load_config(str(tmp_path))
    assert config["see_through"]["resolution"] == 1024

def test_see_through_uses_dedicated_uv_extra_by_default(monkeypatch):
    ...
    result = asyncio.run(runner.run_python_script("module.see_through.cli", ["--help"], native_console=False))
    assert captured == {"extras": ["see-through"], "groups": []}
```

- [ ] **Step 5: Run the focused config and launcher tests**

Run: `pytest tests/test_see_through_config.py tests/test_process_runner_uv_patch.py -q`

Expected: PASS, including the `model.toml` `[see_through]` load and `SCRIPT_REGISTRY` assertions.

- [ ] **Step 6: Review the diff before touching backend code**

Run: `git diff --stat -- config/model.toml pyproject.toml gui/utils/process_runner.py tests/test_process_runner_uv_patch.py tests/test_see_through_config.py`

## Task 2: Scaffold CLI Contract And Run Config

**Files:**
- Create: `E:\Code\qinglong-captions\module\see_through\__init__.py`
- Create: `E:\Code\qinglong-captions\module\see_through\cli.py`
- Create: `E:\Code\qinglong-captions\tests\test_see_through_cli.py`

- [ ] **Step 1: Write failing tests for the CLI contract before implementing parsing**

```python
def test_see_through_parser_defaults_from_config():
    parser = build_parser()
    args = parser.parse_args(["--input_dir=foo", "--output_dir=bar"])
    assert args.offload_policy == "delete"
    assert args.skip_completed is True

def test_see_through_script_help_runs_from_script_path():
    result = subprocess.run(
        [sys.executable, str(ROOT / "module" / "see_through" / "cli.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0
    assert "repo_id_layerdiff" in result.stdout
```

- [ ] **Step 2: Add a single normalized run-config surface**

```python
@dataclass(frozen=True)
class SeeThroughRunConfig:
    input_dir: Path
    output_dir: Path
    repo_id_layerdiff: str
    repo_id_depth: str
    resolution: int
    dtype: str
    offload_policy: str
    skip_completed: bool
    continue_on_error: bool
    save_to_psd: bool
    tblr_split: bool
    limit_images: int
    force_eager_attention: bool
```

- [ ] **Step 3: Keep CLI-to-backend mapping explicit and lossless**

```python
def main(argv: list[str] | None = None) -> int:
    config = build_run_config(parse_args(argv))
    return run_see_through_batch(config)
```

Expected: GUI 和 CLI 都走同一个 `SeeThroughRunConfig`，不再各自维护一份隐式参数语义。

- [ ] **Step 4: Make the CLI reject invalid directory mode early**

```python
if not config.input_dir.exists() or not config.input_dir.is_dir():
    raise SystemExit("input_dir must be an existing directory")
```

- [ ] **Step 5: Run the CLI tests**

Run: `pytest tests/test_see_through_cli.py -q`

Expected: PASS, including `--help` and default-value assertions.

- [ ] **Step 6: Review the diff and verify no backend logic leaked into CLI**

Run: `git diff --stat -- module/see_through/__init__.py module/see_through/cli.py tests/test_see_through_cli.py`

## Task 3: Implement Run Isolation And Output-Folder Resume In `runner.py`

**Files:**
- Modify: `E:\Code\qinglong-captions\module\see_through\runner.py`
- Modify: `E:\Code\qinglong-captions\tests\test_see_through_runner.py`

- [ ] **Step 1: Write failing tests for the real run/resume contract before implementing helpers**

```python
def test_runner_reuses_output_dir_when_run_meta_matches(tmp_path):
    ...
    assert prepared_output_dir == config.output_dir

def test_runner_rejects_output_dir_when_config_changes(tmp_path):
    ...
    with pytest.raises(ValueError):
        prepare_output_dir(config.output_dir, config.input_dir, changed_fingerprint)

def test_runner_rejects_output_dir_when_input_dir_changes(tmp_path):
    ...
    with pytest.raises(ValueError):
        prepare_output_dir(config.output_dir, other_input_dir, config_fingerprint)

def test_runner_uses_relative_path_as_output_key(tmp_path):
    ...
    assert item_dir.as_posix().endswith("foo/a.png")

def test_runner_duplicate_stems_do_not_collide(tmp_path):
    ...
    assert item_dir_a != item_dir_b

def test_runner_detects_stage_from_existing_outputs(tmp_path):
    ...
    assert detect_resume_stage(item_dir) == "marigold"
```

- [ ] **Step 2: Treat `output_dir` as the concrete artifact directory and validate run identity up front**

```python
def prepare_output_dir(output_dir: Path, input_dir: Path, config_fingerprint: str) -> dict[str, Any]: ...
```

Expected:

- empty `output_dir` initializes `run_meta.json` in place;
- same `input_dir` + same `config_fingerprint` resumes inside the same `output_dir`;
- any mismatch raises an explicit error telling the caller to choose a new `output_dir`.

- [ ] **Step 3: Keep item identity and artifact identity as the same key**

```python
def build_config_fingerprint(config: SeeThroughRunConfig) -> str: ...
def make_relative_output_key(input_dir: Path, source_path: Path) -> Path: ...
def make_item_dir(output_dir: Path, relative_key: Path) -> Path: ...
```

- [ ] **Step 4: Drive resume decisions only from output folders**

```python
if postprocess_outputs_complete(item_dir):
    return ResumeAction.skip()
if marigold_outputs_complete(item_dir):
    return ResumeAction.from_stage("postprocess")
if layerdiff_outputs_complete(item_dir):
    return ResumeAction.from_stage("marigold")
return ResumeAction.from_stage("layerdiff")
```

Rules to enforce:

- no row-level merge state;
- no `image_stem`-based output key;
- stage detection only inspects files under the current item directory.

- [ ] **Step 5: Make the single-run-directory contract explicit**

```python
run_meta = {
    "config_fingerprint": config_fingerprint,
    "input_dir": str(config.input_dir),
    "created_at": ...,
}
```

Expected: first release assumes inputs are immutable within one `output_dir`; if source files or effective config change, caller must provide a new `output_dir` instead of relying on hidden invalidation.

- [ ] **Step 6: Run the runner tests**

Run: `pytest tests/test_see_through_runner.py -q`

Expected: PASS, including mismatch rejection on reused `output_dir`, relative-path output keys, duplicate-stem isolation, and folder-based stage resume.

## Task 4: Implement Runtime Detection And Pipeline Lifecycle Management

**Files:**
- Create: `E:\Code\qinglong-captions\module\see_through\runtime.py`
- Create: `E:\Code\qinglong-captions\module\see_through\model_manager.py`
- Create: `E:\Code\qinglong-captions\tests\test_see_through_runtime.py`
- Create: `E:\Code\qinglong-captions\tests\test_see_through_model_manager.py`

- [ ] **Step 1: Write failing tests for `device/dtype/backend` resolution and offload policies**

```python
def test_resolve_attention_backend_prefers_flash_attn_then_sdpa_then_eager():
    ...

def test_release_layerdiff_delete_policy_clears_reference(monkeypatch):
    manager = SeeThroughModelManager(offload_policy="delete")
    manager._layerdiff = object()
    manager.release_layerdiff()
    assert manager._layerdiff is None
```

- [ ] **Step 2: Model runtime state as a small explicit object**

```python
@dataclass(frozen=True)
class RuntimeContext:
    device: str
    dtype: torch.dtype
    attention_backend: str
    reason: str
```

- [ ] **Step 3: Keep backend resolution runtime-driven, not user-API-driven**

```python
def resolve_attention_backend(*, force_eager_attention: bool = False) -> RuntimeContext:
    ...
```

Expected: 只保留 `force_eager_attention` 调试逃生口，不暴露 `flash_attention_2 | sdpa | eager` 正式枚举。

- [ ] **Step 4: Implement a phase-aware model manager with explicit release points**

```python
class SeeThroughModelManager:
    def get_layerdiff_pipeline(self): ...
    def get_marigold_pipeline(self): ...
    def release_layerdiff(self): ...
    def release_marigold(self): ...
    def release_all(self): ...
    def log_vram(self, stage_name: str): ...
```

- [ ] **Step 5: Run the runtime and lifecycle tests**

Run: `pytest tests/test_see_through_runtime.py tests/test_see_through_model_manager.py -q`

Expected: PASS, including fallback resolution and `cpu/delete` offload behaviors.

- [ ] **Step 6: Review the diff and verify state logic did not leak into lifecycle code**

Run: `git diff --stat -- module/see_through/runtime.py module/see_through/model_manager.py tests/test_see_through_runtime.py tests/test_see_through_model_manager.py`

## Task 5: Extract The Minimal Upstream Algorithm Subset And Wrap Phases

**Files:**
- Create: `E:\Code\qinglong-captions\module\see_through\extracted\layerdiff_core.py`
- Create: `E:\Code\qinglong-captions\module\see_through\extracted\marigold_core.py`
- Create: `E:\Code\qinglong-captions\module\see_through\extracted\postprocess_core.py`
- Create: `E:\Code\qinglong-captions\module\see_through\pipelines\layerdiff.py`
- Create: `E:\Code\qinglong-captions\module\see_through\pipelines\marigold.py`
- Create: `E:\Code\qinglong-captions\module\see_through\postprocess.py`

- [ ] **Step 1: Copy only the needed algorithm surface from upstream and label provenance**

Every extracted file must begin with a short provenance header:

```python
# Extracted and adapted from:
# - inference/scripts/inference_psd.py
# - common/utils/inference_utils.py
# Upstream repository: shitagaki-lab/see-through
# Notes: globals removed, project-local IO/state conventions applied.
```

- [ ] **Step 2: Delete upstream-style module globals during extraction**

Expected:

- No module-level cached pipelines in `extracted/*`.
- No direct dependency on upstream repo directory layout.
- No `sys.path` surgery.

- [ ] **Step 3: Wrap each phase behind a stable local interface**

```python
class LayerDiffPhase:
    def run_item(self, source_path: Path, output_dir: Path) -> dict[str, Any]: ...

class MarigoldPhase:
    def run_item(self, source_path: Path, output_dir: Path) -> dict[str, Any]: ...

def run_postprocess(*, source_path: Path, output_dir: Path, save_to_psd: bool, tblr_split: bool) -> dict[str, Any]: ...
```

- [ ] **Step 4: Keep output contracts explicit because the runner depends on them**

`layerdiff_outputs`, `marigold_outputs`, and `postprocess_outputs` should each return a deterministic dict of output paths that can later be validated by `runner.py` against the current item directory.

- [ ] **Step 5: Run a narrow import smoke test before runner integration**

Run: `pytest tests/test_see_through_runner.py -q -k "imports or contracts"`

Expected: phase wrappers import cleanly and expose stable output keys under mocked internals.

- [ ] **Step 6: Review the diff and confirm no whole-repo vendoring slipped in**

Run: `git diff --stat -- module/see_through/extracted module/see_through/pipelines module/see_through/postprocess.py`

## Task 6: Implement Phase-Batch Runner Orchestration

**Files:**
- Create: `E:\Code\qinglong-captions\module\see_through\runner.py`
- Create: `E:\Code\qinglong-captions\tests\test_see_through_runner.py`

- [ ] **Step 1: Write failing tests for the real orchestration contract**

```python
def test_runner_batches_layerdiff_then_releases_then_marigold_then_releases(monkeypatch, tmp_path):
    ...
    assert calls == [
        ("layerdiff", "a.png"),
        ("layerdiff", "b.png"),
        ("release", "layerdiff"),
        ("marigold", "a.png"),
        ("marigold", "b.png"),
        ("release", "marigold"),
        ("postprocess", "a.png"),
        ("postprocess", "b.png"),
    ]

def test_runner_continues_when_single_item_fails(monkeypatch, tmp_path):
    ...

def test_runner_honors_limit_images_before_phase_execution(monkeypatch, tmp_path):
    ...
```

- [ ] **Step 2: Implement input collection and deterministic item ordering**

```python
def collect_input_images(input_dir: Path, limit_images: int) -> list[Path]:
    ...
```

Expected: 目录扫描结果稳定、可测试，并在进入 phase 之前就截断 `limit_images`.

- [ ] **Step 3: Build the run plan from the current run directory before loading any GPU model**

```python
plan = build_execution_plan(config, config.output_dir, discovered_items)
```

This is important because:

- phase-batch scheduling depends on resume stage;
- loading models before knowing the plan just wastes VRAM and startup time.

- [ ] **Step 4: Execute phases in whole-directory batches and materialize outputs into the current `output_dir`**

```python
run_layerdiff_batch(planned_items_for_layerdiff)
model_manager.release_layerdiff()
run_marigold_batch(planned_items_for_marigold)
model_manager.release_marigold()
run_postprocess_batch(planned_items_for_postprocess)
```

- [ ] **Step 5: Ensure failure handling stays item-local**

Rules to enforce:

- `continue_on_error=true` keeps the batch moving.
- A failed item is marked with `last_error_stage/last_error`.
- Downstream phases only see items whose upstream outputs are valid.

- [ ] **Step 6: Run the runner tests**

Run: `pytest tests/test_see_through_runner.py -q`

Expected: PASS, including phase order, `continue_on_error`, `limit_images`, and folder-based resume-stage routing.

## Task 7: Add The GUI Tool Tab And Argument Mapping

**Files:**
- Modify: `E:\Code\qinglong-captions\gui\wizard\step6_tools.py`
- Modify: `E:\Code\qinglong-captions\gui\utils\i18n.py`
- Create: `E:\Code\qinglong-captions\tests\test_see_through_tools_step.py`

- [ ] **Step 1: Add a dedicated `See-through` tab near the existing visual tools**

Expected controls:

- `input_dir`
- `output_dir`
- `repo_id_layerdiff`
- `repo_id_depth`
- `resolution`
- `offload_policy`
- `dtype`
- `limit_images`
- `skip_completed`
- `continue_on_error`
- `save_to_psd`
- `tblr_split`
- advanced: `force_eager_attention`

- [ ] **Step 2: Follow the existing `ToolsStep` launch pattern instead of inventing a second GUI runtime**

```python
start_btn = ui.button(t("start_see_through"), on_click=self._start_see_through, icon="play_arrow")
await self.panel.run_job("module.see_through.cli", args, "See-through", ...)
```

- [ ] **Step 3: Write failing GUI arg-mapping tests before adding the tab logic**

```python
def test_tools_step_see_through_maps_args(monkeypatch, tmp_path):
    ...
    assert captured["script_key"] == "module.see_through.cli"
    assert "--repo_id_layerdiff=..." in captured["args"]
    assert "--skip_completed" in captured["args"]
```

- [ ] **Step 4: Add i18n keys only for the new public surface**

Expected: label/help/notify keys for the tab and parameters, but no internal stage terminology leaked into user-facing copy.

- [ ] **Step 5: Run the GUI tool tests**

Run: `pytest tests/test_see_through_tools_step.py -q`

Expected: PASS, including existing `ExecutionPanel`-style launch behavior and input validation warnings.

- [ ] **Step 6: Review the diff and confirm the GUI stays a thin caller**

Run: `git diff --stat -- gui/wizard/step6_tools.py gui/utils/i18n.py tests/test_see_through_tools_step.py`

## Task 8: End-To-End Verification And Smoke Coverage

**Files:**
- Verify all files touched above

- [ ] **Step 1: Run the focused backend suite**

Run: `pytest tests/test_see_through_config.py tests/test_see_through_cli.py tests/test_see_through_runtime.py tests/test_see_through_model_manager.py tests/test_see_through_runner.py tests/test_see_through_tools_step.py tests/test_process_runner_uv_patch.py -q`

Expected: PASS.

- [ ] **Step 2: Run the CLI help command from the real script path**

Run: `python module/see_through/cli.py --help`

Expected: exit code `0`, help includes `repo_id_layerdiff`, `repo_id_depth`, `offload_policy`, and `limit_images`.

- [ ] **Step 3: Perform one mocked smoke run with a tiny directory**

Run:

```powershell
python module/see_through/cli.py `
  --input_dir=tests/fixtures/images `
  --output_dir=tmp/see_through_smoke `
  --limit_images=2 `
  --continue_on_error `
  --skip_completed
```

Expected: target `output_dir` initialized or validated, summary logs printed, no attempt to load absent GPU models in mocked/test mode.

- [ ] **Step 4: Review the final diff for ownership drift**

Run: `git diff --stat`

Check:

- no unrelated provider wiring;
- no whole-upstream repo imports;
- no hidden row-level merge state;
- no per-image load/delete cycle in runner.

- [ ] **Step 5: Capture residual risks explicitly in the implementation note or PR body**

Required callouts:

- `flash-attn` runtime fallback still depends on environment correctness;
- extracted upstream subset may require follow-up sync when upstream算法变更;
- first release intentionally assumes inputs are immutable within one `output_dir`.
