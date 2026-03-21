# ONNX Runtime Unification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 统一仓库内 ONNX Runtime 相关配置与 session 装配逻辑，接入 `waterdetect` 和 `wdtagger` 到共享基础层，同时保留 `lfm_vl_local` 的多 artifact 形态与 legacy 配置兼容。

**Architecture:** 继续以 `module/onnx_runtime` 作为唯一的 runtime 基础层，扩展配置解析、session option/provider option 构建和 cache key 生成逻辑。新增单模型装配入口给 `waterdetect` / `wdtagger` 复用，但不改它们的预处理和后处理；`lfm_vl_local` 仍使用多文件 bundle，只把 runtime 配置来源切到共享 `onnx.toml` 并保留 legacy fallback。

**Tech Stack:** Python 3.11, `onnxruntime`, `huggingface_hub`, TOML split config, `pytest`, existing provider/runtime infrastructure.

---

## File Map

**Create**

- `E:\Code\qinglong-captions\config\onnx.toml`
- `E:\Code\qinglong-captions\module\onnx_runtime\single_model.py`
- `E:\Code\qinglong-captions\tests\test_onnx_single_model.py`
- `E:\Code\qinglong-captions\tests\test_waterdetect_onnx.py`
- `E:\Code\qinglong-captions\tests\test_wdtagger_onnx.py`

**Modify**

- `E:\Code\qinglong-captions\config\loader.py`
- `E:\Code\qinglong-captions\config\config.toml`
- `E:\Code\qinglong-captions\module\onnx_runtime\config.py`
- `E:\Code\qinglong-captions\module\onnx_runtime\session.py`
- `E:\Code\qinglong-captions\module\onnx_runtime\__init__.py`
- `E:\Code\qinglong-captions\module\waterdetect.py`
- `E:\Code\qinglong-captions\utils\wdtagger.py`
- `E:\Code\qinglong-captions\module\providers\local_vlm\lfm_vl_local.py`
- `E:\Code\qinglong-captions\tests\test_onnx_runtime.py`
- `E:\Code\qinglong-captions\tests\test_runtime_config.py`
- `E:\Code\qinglong-captions\tests\test_lfm_vl_local.py`

**Reference**

- `E:\Code\qinglong-captions\docs\superpowers\specs\2026-03-21-onnxruntime-unification-design.md`

### Responsibilities

- `config/onnx.toml`: ONNX Runtime defaults and per-tool runtime overrides.
- `config/loader.py`: 将 `onnx.toml` 纳入 split config 合并链路。
- `module/onnx_runtime/config.py`: 统一 runtime config 解析、legacy runtime 字段兼容读取、runtime fingerprint 输入规范化。
- `module/onnx_runtime/session.py`: 统一 session options、provider options、cache key 和 cache 复用。
- `module/onnx_runtime/single_model.py`: 单模型 ONNX artifact + session 装配入口。
- `module/waterdetect.py`: 改为通过共享单模型入口加载 ONNX session。
- `utils/wdtagger.py`: 改为通过共享单模型入口加载 ONNX session。
- `module/providers/local_vlm/lfm_vl_local.py`: runtime 配置优先从 `onnx_runtime.lfm_vl_local` 读取，兼容 legacy `model.toml` 字段。

### Task 1: Add `onnx.toml` And Config Compatibility

**Files:**
- Create: `E:\Code\qinglong-captions\config\onnx.toml`
- Modify: `E:\Code\qinglong-captions\config\loader.py`
- Modify: `E:\Code\qinglong-captions\config\config.toml`
- Test: `E:\Code\qinglong-captions\tests\test_runtime_config.py`

- [ ] **Step 1: Add the new split config file with runtime defaults**

```toml
[onnx_runtime.defaults]
execution_provider = "auto"
force_download = false
model_cache_dir = ""
session_cache_dir = ""

[onnx_runtime.defaults.session]
graph_optimization_level = "ORT_ENABLE_ALL"
enable_mem_pattern = true
enable_mem_reuse = true

[onnx_runtime.defaults.cuda]
arena_extend_strategy = "kSameAsRequested"
cudnn_conv_algo_search = "EXHAUSTIVE"
do_copy_in_default_stream = true

[onnx_runtime.defaults.tensorrt]
engine_cache_enable = true
timing_cache_enable = true
fp16_enable = true

[onnx_runtime.wdtagger]
[onnx_runtime.waterdetect]
[onnx_runtime.lfm_vl_local]
```

- [ ] **Step 2: Extend the split-config loader to merge `onnx.toml`**

```python
_CONFIG_FILES = ["prompts.toml", "model.toml", "general.toml", "onnx.toml"]
```

- [ ] **Step 3: Mirror the new ONNX runtime section into legacy `config.toml`**

```toml
[onnx_runtime.defaults]
execution_provider = "auto"
```

Expected: legacy single-file loading still sees `onnx_runtime`.

- [ ] **Step 4: Add failing tests for split loading and legacy fallback**

```python
def test_runtime_config_loads_onnx_split_file(tmp_path):
    (tmp_path / "onnx.toml").write_text("[onnx_runtime.defaults]\nexecution_provider='cpu'\n", encoding="utf-8")
    config = load_runtime_config(str(tmp_path))
    assert config["onnx_runtime"]["defaults"]["execution_provider"] == "cpu"
```

- [ ] **Step 5: Run the focused config tests**

Run: `pytest tests/test_runtime_config.py -q`
Expected: PASS, including new assertions for `onnx.toml`.

- [ ] **Step 6: Review the diff and verify scope**

Run: `git diff --stat -- config/loader.py config/config.toml config/onnx.toml tests/test_runtime_config.py`

### Task 2: Expand Shared ONNX Runtime Config And Cache Semantics

**Files:**
- Modify: `E:\Code\qinglong-captions\module\onnx_runtime\config.py`
- Modify: `E:\Code\qinglong-captions\module\onnx_runtime\session.py`
- Modify: `E:\Code\qinglong-captions\tests\test_onnx_runtime.py`

- [ ] **Step 1: Write failing tests for legacy runtime compatibility and runtime fingerprint-sensitive cache keys**

```python
def test_runtime_config_prefers_onnx_section_over_legacy_model_section():
    runtime = OnnxRuntimeConfig.from_runtime_sections(
        defaults={"execution_provider": "auto"},
        legacy={"execution_provider": "cpu"},
        override={"execution_provider": "cuda"},
    )
    assert runtime.execution_provider == "cuda"

def test_load_session_bundle_cache_key_includes_runtime_fingerprint(tmp_path):
    ...
    runtime_a = OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=1)
    runtime_b = OnnxRuntimeConfig(execution_provider="cpu", intra_op_num_threads=8)
    assert bundle_a is not bundle_b
```

- [ ] **Step 2: Extend `OnnxRuntimeConfig` to model session/provider settings and legacy merge behavior**

```python
@dataclass(frozen=True)
class OnnxRuntimeConfig:
    execution_provider: str = "auto"
    model_cache_dir: str = ""
    session_cache_dir: str = ""
    force_download: bool = False
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    enable_mem_pattern: bool = True
    enable_mem_reuse: bool = True
    execution_mode: str = ""
    inter_op_num_threads: int = 0
    intra_op_num_threads: int = 0
    provider_options: Mapping[str, Any] = field(default_factory=dict)
```

- [ ] **Step 3: Add a helper that merges defaults + legacy runtime fields + per-tool ONNX overrides**

```python
runtime = OnnxRuntimeConfig.from_runtime_sections(
    defaults=onnx_defaults,
    legacy=legacy_runtime_fields,
    override=tool_runtime_overrides,
)
```

Expected: `onnx.toml` wins, legacy fields still work, and deprecated fields can be logged once.

- [ ] **Step 4: Refactor `session.py` to build session options and provider descriptors from the normalized runtime config**

```python
sess_options = build_session_options(runtime)
providers = build_execution_providers(runtime, available_providers=...)
runtime_fingerprint = make_runtime_fingerprint(runtime, providers)
cache_key = make_cache_key(bundle_key, session_paths, providers, runtime_fingerprint)
```

- [ ] **Step 5: Run ONNX runtime unit tests**

Run: `pytest tests/test_onnx_runtime.py -q`
Expected: PASS, including the new cache-key and merge-order assertions.

- [ ] **Step 6: Review the diff and verify scope**

Run: `git diff --stat -- module/onnx_runtime/config.py module/onnx_runtime/session.py tests/test_onnx_runtime.py`

### Task 3: Add The Shared Single-Model Loader

**Files:**
- Create: `E:\Code\qinglong-captions\module\onnx_runtime\single_model.py`
- Modify: `E:\Code\qinglong-captions\module\onnx_runtime\__init__.py`
- Test: `E:\Code\qinglong-captions\tests\test_onnx_single_model.py`

- [ ] **Step 1: Write failing tests for the single-model loader contract**

```python
def test_load_single_model_bundle_downloads_artifact_and_builds_session(tmp_path):
    bundle = load_single_model_bundle(...)
    assert bundle.model_path.name == "model.onnx"
    assert bundle.session is fake_session
    assert bundle.providers == ("CPUExecutionProvider",)
```

- [ ] **Step 2: Add `OnnxModelSpec` and `SingleModelOnnxBundle`**

```python
@dataclass(frozen=True)
class OnnxModelSpec:
    repo_id: str
    onnx_filename: str
    local_dir: str | Path
    bundle_key: str

@dataclass(frozen=True)
class SingleModelOnnxBundle:
    model_path: Path
    session: Any
    providers: tuple[Any, ...]
    input_metas: tuple[Any, ...]
    runtime_config: OnnxRuntimeConfig
```

- [ ] **Step 3: Implement `load_single_model_bundle(...)` using existing artifact + session helpers**

```python
model_path = download_onnx_artifact(...)
session_bundle = load_session_bundle(
    bundle_key=spec.bundle_key,
    session_paths={"model": model_path},
    runtime_config=runtime_config,
)
session = session_bundle.sessions["model"]
```

- [ ] **Step 4: Export the new loader from `module/onnx_runtime/__init__.py`**

```python
from .single_model import OnnxModelSpec, SingleModelOnnxBundle, load_single_model_bundle
```

- [ ] **Step 5: Run the single-model tests**

Run: `pytest tests/test_onnx_single_model.py -q`
Expected: PASS.

- [ ] **Step 6: Review the diff and verify scope**

Run: `git diff --stat -- module/onnx_runtime/single_model.py module/onnx_runtime/__init__.py tests/test_onnx_single_model.py`

### Task 4: Migrate `waterdetect` To The Shared ONNX Loader

**Files:**
- Modify: `E:\Code\qinglong-captions\module\waterdetect.py`
- Test: `E:\Code\qinglong-captions\tests\test_waterdetect_onnx.py`

- [ ] **Step 1: Write a failing wiring test for `waterdetect.load_model`**

```python
def test_waterdetect_load_model_uses_single_model_bundle(monkeypatch):
    monkeypatch.setattr("module.waterdetect.load_single_model_bundle", fake_loader)
    session, input_name = load_model(args)
    assert session is fake_session
    assert input_name == "pixel_values"
```

- [ ] **Step 2: Replace local provider/session construction with `OnnxModelSpec` + `load_single_model_bundle(...)`**

```python
spec = OnnxModelSpec(
    repo_id=args.repo_id,
    onnx_filename="model.onnx",
    local_dir=Path(args.model_dir) / args.repo_id.replace("/", "_"),
    bundle_key=f"waterdetect:{args.repo_id}",
)
bundle = load_single_model_bundle(spec=spec, runtime_config=runtime)
```

- [ ] **Step 3: Keep processor loading and inference batch code unchanged except for reading from the bundle**

```python
ort_sess = bundle.session
input_name = ort_sess.get_inputs()[0].name
```

- [ ] **Step 4: Run the focused `waterdetect` wiring tests**

Run: `pytest tests/test_waterdetect_onnx.py -q`
Expected: PASS.

- [ ] **Step 5: Sanity-check existing module-level behavior**

Run: `pytest tests/test_onnx_runtime.py tests/test_waterdetect_onnx.py -q`
Expected: PASS.

- [ ] **Step 6: Review the diff and verify scope**

Run: `git diff --stat -- module/waterdetect.py tests/test_waterdetect_onnx.py`

### Task 5: Migrate `wdtagger` To The Shared ONNX Loader

**Files:**
- Modify: `E:\Code\qinglong-captions\utils\wdtagger.py`
- Test: `E:\Code\qinglong-captions\tests\test_wdtagger_onnx.py`

- [ ] **Step 1: Write a failing wiring test for the ONNX loading path in `wdtagger`**

```python
def test_wdtagger_load_model_and_tags_uses_single_model_bundle(monkeypatch):
    monkeypatch.setattr("utils.wdtagger.load_single_model_bundle", fake_loader)
    ort_sess, input_name, labels = load_model_and_tags(args)
    assert ort_sess is fake_session
```

- [ ] **Step 2: Refactor only the ONNX runtime portion out of `load_model_and_tags`**

```python
runtime = resolve_tool_runtime_config(...)
spec = OnnxModelSpec(
    repo_id=args.repo_id,
    onnx_filename="model.onnx",
    local_dir=Path(args.model_dir) / args.repo_id.replace("/", "_"),
    bundle_key=f"wdtagger:{args.repo_id}",
)
bundle = load_single_model_bundle(spec=spec, runtime_config=runtime)
```

- [ ] **Step 3: Leave tag CSV/JSON loading, category mapping, and downstream post-processing untouched**

Expected: only ONNX session construction moves; tag semantics do not change.

- [ ] **Step 4: Run the focused `wdtagger` wiring tests**

Run: `pytest tests/test_wdtagger_onnx.py -q`
Expected: PASS.

- [ ] **Step 5: Re-run the ONNX runtime tests that cover shared logic**

Run: `pytest tests/test_onnx_runtime.py tests/test_wdtagger_onnx.py -q`
Expected: PASS.

- [ ] **Step 6: Review the diff and verify scope**

Run: `git diff --stat -- utils/wdtagger.py tests/test_wdtagger_onnx.py`

### Task 6: Move `lfm_vl_local` To Shared Runtime Config Source And Run Final Regression

**Files:**
- Modify: `E:\Code\qinglong-captions\module\providers\local_vlm\lfm_vl_local.py`
- Modify: `E:\Code\qinglong-captions\tests\test_lfm_vl_local.py`
- Modify: `E:\Code\qinglong-captions\tests\test_runtime_config.py`

- [ ] **Step 1: Write a failing test that proves `lfm_vl_local` can read runtime config from `onnx_runtime.lfm_vl_local` and still honor legacy fallback**

```python
def test_lfm_runtime_config_prefers_onnx_section_over_model_section(...):
    ...
    assert captured_runtime.execution_provider == "cpu"
```

- [ ] **Step 2: Refactor `lfm_vl_local` to build runtime config from merged ONNX sections instead of only `self.model_config`**

```python
runtime_config = resolve_provider_runtime_config(
    self.ctx.config,
    tool_name="lfm_vl_local",
    legacy_section=self.model_config,
)
```

- [ ] **Step 3: Keep artifact selection and decode logic unchanged**

Expected: only runtime config sourcing changes; artifact filenames and generation semantics remain identical.

- [ ] **Step 4: Run the focused LFM tests**

Run: `pytest tests/test_lfm_vl_local.py -q`
Expected: PASS.

- [ ] **Step 5: Run the final regression slice for this refactor**

Run: `pytest tests/test_runtime_config.py tests/test_onnx_runtime.py tests/test_onnx_single_model.py tests/test_waterdetect_onnx.py tests/test_wdtagger_onnx.py tests/test_lfm_vl_local.py -q`
Expected: PASS.

- [ ] **Step 6: Review the diff and verify scope**

Run: `git diff --stat -- module/providers/local_vlm/lfm_vl_local.py tests/test_lfm_vl_local.py tests/test_runtime_config.py`

## Completion Checks

- [ ] `config/loader.py` merges `onnx.toml` in split-config mode.
- [ ] `config/config.toml` contains a legacy mirror for `onnx_runtime`.
- [ ] `OnnxRuntimeConfig` can merge defaults, legacy runtime fields, and per-tool ONNX overrides.
- [ ] Session cache keys include a runtime fingerprint and do not cross-reuse incompatible sessions.
- [ ] `waterdetect` no longer creates `onnxruntime.InferenceSession` directly.
- [ ] `wdtagger` no longer creates `onnxruntime.InferenceSession` directly.
- [ ] `lfm_vl_local` reads runtime config from the shared ONNX configuration source with legacy fallback.
- [ ] Focused regression suite passes.
