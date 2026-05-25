# WDTagger 模块化重构设计

## 背景

`utils/wdtagger.py` 当前约 1200 行，已经承担了太多互不相同的职责：

1. CLI 参数定义和默认值修正。
2. ONNX / SigLIP2 模型选择、下载、加载。
3. 图像读取、预处理、批处理推理。
4. tag taxonomy 解析、阈值过滤、父子 tag 去重、输出顺序组装。
5. Lance 数据集打开、候选行扫描、caption 状态判断、写回策略。
6. sidecar caption 读写、`tags.json` 汇总输出。
7. Lance 版本行为规避，包括避免 list filter 和规避 `merge_insert` 问题。

文件变长本身不是问题。真正的问题是这些职责有不同的变化原因：模型逻辑变化不应该牵动 Lance 写回；Lance 版本兼容不应该污染 tag 组装；CLI 参数不应该让纯函数测试变复杂。

这次重构的目标是先划清职责边界，再逐步搬迁代码。不要在一次改动里重写算法，也不要把可运行的行为换成未经验证的新路径。

## 第一性原理判断

`wdtagger` 的核心任务只有一句话：对一批图片运行 tagger 模型，并把模型输出转换成 caption 结果。

因此它的稳定结构应该围绕数据流切分：

```text
输入数据集/目录
  -> 解析候选图片
  -> 读取图片并预处理
  -> 模型推理
  -> tag 规则组装
  -> 写 sidecar / JSON / Lance
```

每一段都应该有明确输入输出，能单独测试，不能依赖全局 `args`、全局 `console` 或 Lance 的具体版本行为。

Lance 相关问题要分两类：

- 版本/API 兼容：Blob v2、`data_storage_version`、`pylance` 最低版本。这属于 `lanceImport` / `utils.lance_blob` / 依赖声明的责任。
- `wdtagger` 业务语义：哪些行需要打标、是否跳过已有 sidecar、是否 append tags、写回 captions 的策略。这属于 `wdtagger` 的责任，不应该塞进 `lanceImport`。

所以这次不应该把 `wdtagger` 的候选筛选和 sidecar 规则下沉到 `lanceImport`。`lanceImport` 应该保持“导入媒体为 Lance 数据集”的工具层定位。

## 已确认现状

1. `utils/wdtagger.py` 当前约 1191 行。
2. 当前已避免在 Lance scanner 中使用 `array_length(captions)` 过滤，改为 Lance 只过滤 `mime LIKE 'image/%'`，caption 是否为空在 Python 侧判断。
3. 当前默认新增了 `--lance_update_mode=rebuild`，会在打标后从 sidecar 重建 Lance 数据集。
4. `module.lanceImport.transform2lance()` 默认 `save_binary=False`，默认 `data_storage_version` 来自 `utils.lance_blob.DEFAULT_BLOB_DATA_STORAGE_VERSION`。
5. `utils.lance_blob.DEFAULT_BLOB_DATA_STORAGE_VERSION` 当前为 `"2.2"`，并依赖 `pylance` 暴露 `lance.blob_field` / `lance.blob_array`。
6. `pyproject.toml` 改动前的 `pylance` 下限低于 `6.0.1`，和 Blob v2 默认路径不匹配。
7. 其他模块仍有直接使用 `dataset.to_table()`、`dataset.scanner()`、`dataset.merge_insert()` 的代码；这些不应该由 `wdtagger` 重构一次性解决。

## 目标

1. 把 `utils/wdtagger.py` 缩成兼容入口和薄 orchestration 层，目标少于 250 行。
2. 把纯逻辑从 IO / Lance / CLI 中拆出，使 tag 规则和模型选择可独立测试。
3. 保留当前默认行为，除非现有行为本身是 Lance 版本规避造成的过度修复。
4. `pylance` 最低版本提升到 `>=6.0.1`，让默认 Blob v2 API 和项目依赖一致。
5. Lance 读取侧保留保守策略：不在 Lance SQL filter 中使用 `array_length(captions)`。
6. Lance 写回侧恢复清晰策略：sidecar 始终写；Lance `merge_insert` 是默认写回；全量 rebuild 只能是显式 fallback 或单独命令。
7. 测试覆盖拆出的纯函数、候选扫描、sidecar 跳过、Lance 写回策略和 CLI 参数兼容。

## 非目标

- 不重写 tagger 推理算法。
- 不改变默认 tag 阈值、输出顺序、分类规则。
- 不把 wdtagger 改成通用 caption pipeline。
- 不把所有 Lance 调用统一抽象成一个大型 ORM。
- 不在本次重构中修复 `waterdetect`、`rewardmodel`、`caption_pipeline` 等模块的 Lance 使用方式。
- 不默认全量重建 Lance 数据集来掩盖 `merge_insert` 的问题。
- 不删除 `utils/wdtagger.py` 这个脚本入口，避免破坏现有调用。

## 推荐模块边界

新增包：

```text
module/wdtagger/
  __init__.py
  cli.py
  constants.py
  model_loader.py
  preprocess.py
  taxonomy.py
  tag_assembly.py
  lance_io.py
  outputs.py
  runner.py
```

保留：

```text
utils/wdtagger.py
```

`utils/wdtagger.py` 只做兼容入口：

```python
from module.wdtagger.cli import finalize_args, setup_parser
from module.wdtagger.runner import main

if __name__ == "__main__":
    parser = setup_parser()
    main(finalize_args(parser.parse_args()))
```

短期可以从 `utils/wdtagger.py` re-export 老测试直接 import 的函数，但新代码不再往这个文件里加业务逻辑。

### `constants.py`

职责：

- `IMAGE_SIZE`
- 默认 repo / 文件名常量
- `IMAGE_MIME_FILTER`
- wdtagger 默认阈值相关常量

禁止事项：

- 不读取配置文件。
- 不 import Lance、ONNX、OpenCV。

### `taxonomy.py`

职责：

- `LabelData`
- CSV / JSON label 解析。
- parent-child map 加载。
- tag category 数据结构规范化。

公开 API：

```python
@dataclass
class LabelData:
    names: list[str]
    category_indices: dict[str, np.ndarray]
    tag_index_to_category: dict[int, str]

def load_label_data(...)
def load_parent_to_child_map(...)
```

### `model_loader.py`

职责：

- 根据 `repo_id` 选择 legacy WD14 ONNX 或 SigLIP2 / CL tagger 路径。
- 调用 `load_single_model_bundle()` / `load_cl_tagger_v2_bundle()`。
- 返回推理所需的 session、input name/context、label data、parent map。

公开 API：

```python
@dataclass
class WDTaggerModelBundle:
    session: Any
    input_name: Any
    label_data: LabelData
    parent_to_child_map: dict[str, list[str]]
    is_siglip2: bool

def load_model_bundle(args) -> WDTaggerModelBundle
```

### `preprocess.py`

职责：

- 图片读取。
- legacy ONNX 预处理。
- SigLIP2 RGB batch 加载。
- 批量推理包装。

公开 API：

```python
def preprocess_image(image, *, is_cl_tagger: bool = False) -> np.ndarray
def load_legacy_batch(uris: list[str], *, is_cl_tagger: bool) -> tuple[list[str], list[np.ndarray]]
def load_siglip2_batch(uris: list[str]) -> tuple[list[str], list[Image.Image]]
def run_inference(images, bundle: WDTaggerModelBundle) -> Any
```

注意：

- `load_legacy_batch()` 和 `load_siglip2_batch()` 都返回 `valid_uris`，避免坏图导致 URI 和预测结果错位。
- 不写 caption 文件。
- 不碰 Lance。

### `tag_assembly.py`

职责：

- `process_tags()`
- `get_tags_official()`
- `assemble_final_tags()`
- `assemble_tags_json()`
- tag replacement / threshold / ordering / parent removal。

公开 API：

```python
def process_tags(label_data: LabelData, options: WDTaggerOptions) -> list[str]
def get_tags_official(...)
def assemble_final_tags(...)
def assemble_tags_json(...)
```

要求：

- 这些函数必须是纯函数或接近纯函数。
- `tag_freq` 更新可以封装为显式返回值，避免隐藏副作用。
- `split_name_series()` 不应继续在 wdtagger 内维护一份重复实现。优先复用已有公共实现，或抽到公共模块。

### `lance_io.py`

职责：

- 打开或创建 Lance 数据集。
- 扫描 wdtagger 候选图片。
- 判断已有 caption / sidecar 是否应跳过。
- 执行 Lance captions 写回。

公开 API：

```python
@dataclass
class WDTaggerDatasetRef:
    dataset: Any
    dataset_path: Path | None
    source_dir: Path | None

@dataclass
class CandidateBatch:
    uris: list[str]

def resolve_dataset(train_data_dir: str | Any, *, output_name: str = "dataset") -> WDTaggerDatasetRef
def iter_candidate_batches(dataset, options) -> Iterator[CandidateBatch]
def count_candidate_rows(dataset, options) -> int
def merge_caption_updates(dataset, updates: Sequence[CaptionUpdate], *, batch_size: int) -> None
```

保留策略：

- Lance filter 只使用 `mime LIKE 'image/%'`。
- `captions IS NULL OR array_length(captions) = 0` 继续在 Python 侧判断。
- `late_materialization=False` 用于这条候选扫描路径，直到有明确测试证明新 Lance 版本在真实数据上完全稳定。

写回策略：

```text
sidecar: always
merge_insert: default
rebuild: explicit fallback only
none: explicit skip
```

`rebuild` 不应是默认值。它会全量重读原目录，并且当前实现强制 `save_binary=False`，可能改变用户已有 Lance 数据集的 blob 存储策略。

### `outputs.py`

职责：

- sidecar caption 写入。
- append / overwrite 语义。
- `tags.json` 汇总写入。

公开 API：

```python
@dataclass
class CaptionUpdate:
    uri: str
    captions: list[str]
    json_tags: dict[str, list[str]]

def read_sidecar_caption(uri: str, extension: str) -> list[str]
def has_sidecar_caption(uri: str, extension: str) -> bool
def write_sidecar_caption(update: CaptionUpdate, options) -> None
def write_tags_json(path: Path, tags: dict[str, dict[str, list[str]]]) -> None
```

### `runner.py`

职责：

- 串起数据流。
- 管理 progress。
- 调用 model / preprocess / tag assembly / outputs / lance_io。
- 捕获边界异常并给出用户可理解的错误。

`runner.py` 可以依赖其他 wdtagger 子模块，但子模块不能反向依赖 runner。

## Lance 兼容策略

### 依赖约束

`pyproject.toml` 应改为：

```toml
"pylance>=6.0.1",
```

原因：

1. 项目默认 `data_storage_version="2.2"`，需要 Blob v2 API。
2. 本项目已经使用 `lance.blob_field` / `lance.blob_array`。
3. 用户确认新版本 Lance 已修复当前 scanner 问题。
4. 依赖下限继续停在 `3.0.1` 会制造假兼容：安装能成功，运行到 Blob v2 或 scanner 路径才炸。

### 不应放进 `lanceImport` 的内容

- wdtagger 的 `append_tags` 语义。
- 跳过已有 sidecar caption。
- 只处理空 captions。
- `WDtagger` tag 名。
- `tags.json` 输出。

### 可以集中到 Lance 工具层的内容

如果后续多个模块都需要，可以新增通用 helper，而不是塞进 `lanceImport`：

```text
utils/lance_scan.py
utils/lance_update.py
```

候选能力：

- `count_rows_by_batches(scanner)`。
- `safe_merge_insert(dataset, table, on="uris")`。
- `image_mime_filter()`。
- `open_or_transform_dataset(...)`。

但首轮重构不要先抽通用层。先让 wdtagger 自己的边界清楚，再看重复是否真实存在。

## 迁移计划

### 阶段 1：依赖和测试基线

1. `pylance>=6.0.1`。
2. 跑现有 wdtagger 测试。
3. 增加或保留这些测试：
   - 不把 `array_length(captions)` 放进 Lance filter。
   - sidecar 已存在时默认跳过。
   - `overwrite=True` 时不读取 `captions` 列。
   - `append_tags=True` 时不会因为 sidecar 存在而跳过。
   - `lance_update_mode=merge` 会调用 `merge_insert`。
   - `lance_update_mode=none` 不调用 `merge_insert`。

### 阶段 2：抽纯函数

先迁移不依赖 Lance 的代码：

1. constants。
2. taxonomy。
3. tag_assembly。
4. preprocess。
5. model_loader。

每迁移一块都保持 `utils/wdtagger.py` re-export 兼容，测试逐步改到新模块。

### 阶段 3：抽 IO 和写回

1. 把 sidecar 和 `tags.json` 迁到 `outputs.py`。
2. 把 candidate scan 和 merge 写回迁到 `lance_io.py`。
3. 默认写回策略改回 `merge`。
4. `rebuild` 改为显式参数，仅在用户指定时运行。

### 阶段 4：收缩入口

1. `utils/wdtagger.py` 只保留 CLI shim。
2. `module/wdtagger/cli.py` 保留 parser 和 `finalize_args()`。
3. `module/wdtagger/runner.py` 保留主流程。
4. 删除已经迁移的旧函数或保留一轮兼容 re-export，并在测试里停止依赖旧路径。

## 验收标准

1. `utils/wdtagger.py` 少于 250 行。
2. `module/wdtagger/` 中没有单个文件超过 400 行。
3. wdtagger 单测覆盖核心纯函数和 Lance IO 策略。
4. `python -m pytest tests/test_wdtagger_onnx.py -q` 通过。
5. `python -m pytest tests/test_lance_blob.py -q` 通过。
6. 运行 wdtagger 时不再默认全量 rebuild Lance。
7. 默认依赖安装得到 `pylance>=6.0.1`。
8. 旧命令 `python utils/wdtagger.py ...` 仍可用。

## 风险和控制

### 风险：一次性搬太多导致行为漂移

控制：按阶段迁移，每阶段只移动代码，不改业务语义。重命名和行为修改分开提交。

### 风险：测试仍依赖 `utils.wdtagger`

控制：第一阶段保留 re-export。新测试写新模块路径，旧测试最后再迁。

### 风险：Lance 新版本修复不等于所有扫描路径都安全

控制：候选扫描仍避免 list filter。依赖升级解决 API 和已知 bug，不代表可以把复杂业务 filter 重新推给 Lance。

### 风险：rebuild 默认行为破坏数据集形态

控制：默认使用 `merge_insert`，rebuild 必须显式选择。若 merge 失败，提示用户使用 `--lance_update_mode=rebuild`，不要静默改写整个数据集。

### 风险：拆分后模块名和现有脚本冲突

控制：新包放在 `module/wdtagger/`，保留 `utils/wdtagger.py` 入口，避免 `utils.wdtagger` 文件和包同名问题。

## 推荐提交顺序

1. `Require pylance 6.0.1`
2. `Add wdtagger refactor design`
3. `Extract wdtagger tag assembly`
4. `Extract wdtagger preprocessing`
5. `Extract wdtagger model loading`
6. `Extract wdtagger Lance IO`
7. `Make wdtagger script a thin entrypoint`
8. `Restore merge Lance update as default`

每个提交都应该能独立跑通 wdtagger 相关测试。
