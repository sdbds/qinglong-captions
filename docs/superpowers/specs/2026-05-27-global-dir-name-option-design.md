# 全局 Dir Name 选项设计

## 背景

项目已经有 `--dir_name` / `$dir_name` 开关，当前语义是从输入媒体所在目录提取角色上下文，并把提示注入到 caption prompt：

```text
datasets/Alice (Wonderland)/sample.jpg
-> <Alice> from (Wonderland)
-> If there is a person/character or more in the image you must refer to them as <Alice> from (Wonderland).
```

现状不是“没有这个功能”，而是它还没有成为真正的全局能力：

1. CLI 和 PowerShell 入口已有开关，但 GUI / model panel 里容易把它当成某个模型的局部选项。
2. `Provider._get_character_prompt()` 已经提供了公共入口，但它仍然是 provider 私有方法，并且只按 `Path(uri).parent.name` 推断来源。
3. `Provider.resolve_prompts()` 会把角色上下文传给 `PromptResolver`，但重写 `resolve_prompts()` 的 provider 可能绕开它。
4. OCR、audio、translation 等模型也属于“模型”，但目录名 prompt 对它们不一定是正确语义。
5. 视频切片、pair image、Lance 重建路径会改变当前 `uri`，如果直接取当前路径父目录，可能拿到派生目录而不是原始数据集目录。

所以这次目标不是“在所有模型里复制一段目录名解析代码”，而是把 `dir_name` 变成运行时全局上下文能力：所有模型走同一个解析入口，所有 prompt 消费点走同一份契约，是否注入由任务语义决定。

## 第一性原理判断

`dir_name` 的本质不是文件系统功能，而是用户显式提供的外部语义先验。

模型真正需要的不是“目录名字符串”，而是一个稳定、可测试的上下文对象：

- 用户是否打开了目录名上下文。
- 目录名来自哪个源路径。
- 原始目录名是什么。
- 解析后的角色名是什么。
- 是否适合注入当前任务 prompt。
- 注入后最终 prompt 长什么样。

如果每个 provider 自己解析路径，就会出现三类坏味道：

- 新模型接入时默认漏掉 `dir_name`。
- 相同目录名在不同模型里格式化结果不一致。
- 视频 clip、pair image、OCR document image 这些边界在每个 provider 里重复猜。

正确切面是：目录名解析属于 pipeline/provider 入口的全局上下文构造，模型只消费已经解析好的 `PromptContext`。

## 目标

1. `dir_name` 成为全局运行参数，不再属于某个模型的局部配置。
2. 所有 provider 都能通过同一个 API 读取目录名上下文。
3. 所有 image/video caption provider 默认注入目录名 prompt。
4. 所有新接入的 image/video caption provider 默认继承该能力，不需要手写目录名逻辑。
5. OCR / PDF / ASR / translation 默认不注入目录名 prompt，避免污染任务输出。
6. 视频切片场景使用原始视频路径的父目录，不使用 clip 目录。
7. Pair image 场景使用主图路径的父目录，不使用 `pair_dir`。
8. CLI、PowerShell、GUI 只暴露一个全局 `dir_name` 开关。
9. 保持现有 `--dir_name` 名称和默认关闭行为。
10. 用 contract test 锁住 provider 覆盖面，防止后续回退。

## 非目标

- 不新增第二个用户可见开关来替代 `--dir_name`。
- 不把目录名 prompt 无条件注入所有媒体任务。
- 不改变 `split_name_series()` 的既有格式化语义。
- 不在本次改动中重写 prompt 模板系统。
- 不让 provider 在 attempt 阶段重新解析路径。
- 不为了“全局”把 OCR 结果、ASR transcript 或翻译结果强行带上角色名。

## 术语

`全局选项`：

用户只配置一次，所有 provider 都可以读到。

`全局能力`：

目录名解析、适用性判断和 prompt 注入走同一个共享模块。

`默认注入`：

对 image/video caption 任务自动前置目录名 prompt。

`可读取但不注入`：

对 OCR/audio/translation 等任务保留上下文对象，但默认不改 prompt。

## 推荐设计

### 1. 新增目录名上下文模块

新增：

```text
module/providers/directory_name_context.py
```

公开 API：

```python
@dataclass(frozen=True)
class DirectoryNameContext:
    enabled: bool
    applicable: bool
    source_uri: str = ""
    raw_directory_name: str = ""
    character_name: str = ""
    character_prompt: str = ""
    reason: str = ""

    @property
    def has_prompt(self) -> bool:
        return bool(self.character_prompt)


def resolve_directory_name_context(
    *,
    args: Any,
    uri: str,
    mime: str,
    provider_name: str = "",
    media: MediaContext | None = None,
    source_uri: str | None = None,
) -> DirectoryNameContext:
    ...


def apply_directory_name_context(
    prompts: PromptContext,
    context: DirectoryNameContext,
) -> PromptContext:
    ...
```

规则：

- `args.dir_name` 为 false 时返回 `enabled=False`。
- `source_uri` 优先级高于 `uri`。
- `args.directory_name_source_uri` 优先级高于 `uri`，用于视频 clip。
- `media.extras["directory_name_source_uri"]` 可作为等价来源。
- 默认只对 image/video caption 任务 `applicable=True`。
- `application/pdf`、audio、纯文本 translation 默认 `applicable=False`。
- 目录名为空、根路径、解析失败时不抛异常，返回空上下文并带 `reason`。

### 2. Provider 基类只做一次上下文构造

当前：

```python
char_name, char_prompt = self._get_character_prompt(uri)
```

目标：

```python
directory_context = resolve_directory_name_context(
    args=self.ctx.args,
    uri=uri,
    mime=mime,
    provider_name=self.name,
    media=media,
)
prompts = resolver.resolve(
    mime,
    self.ctx.args,
    character_prompt=directory_context.character_prompt,
    character_name=directory_context.character_name,
    media=media,
)
```

`Provider._get_character_prompt()` 可以保留为兼容 wrapper，但新代码禁止直接调用它。

### 3. PromptResolver 不解析路径

`PromptResolver` 的职责保持单纯：

- 选择 system/user prompt。
- 处理 provider-specific prompt fallback。
- 处理 pair image prompt。
- 前置已经给定的 `character_prompt`。

它不应该读取路径，也不应该知道 `split_name_series()`。

### 4. Provider 不再拥有目录名逻辑

provider contract：

1. `prepare_media()` 只准备媒体，不拼目录名 prompt。
2. `resolve_prompts()` 若重写，必须调用共享 helper 或委托基类。
3. `attempt()` 只消费 `PromptContext`，不再看 `Path(uri).parent.name`。
4. 需要角色名后验证的 provider 读取 `prompts.character_name`。
5. 新 provider 的测试必须证明它没有绕开目录名上下文。

### 5. 视频切片传递原始来源路径

视频切片时当前处理路径可能变成：

```text
datasets/Alice (Wonderland)/movie_clip/movie_000.mp4
```

不能从 `movie_clip` 推断角色名。

pipeline 在调用 provider 前应复制 args 并写入：

```python
clip_args.directory_name_source_uri = str(original_video_path)
```

目录名 helper 必须优先使用这个字段。

### 6. Pair image 使用主图路径

Pair image 的第二张图是对照或参考，不是当前样本身份来源。

规则：

```text
dataset/Alice (Wonderland)/sample.jpg
pair_dir/Bob/sample.jpg
```

`dir_name` 结果必须是：

```text
<Alice> from (Wonderland)
```

不能变成 `<Bob>`。

### 7. GUI / 配置层归位

`dir_name` 应出现在全局 caption 选项里，不应放在某个模型专属 panel 下。

推荐归属：

```text
Caption global settings
  - dir_name
  - mode
  - pair_dir
  - segment_time
  - retries / wait_time
```

模型 panel 只放模型选择、API key、model path、runtime backend 等模型私有配置。

## 适用性矩阵

| 任务 / provider 类型 | 默认读取上下文 | 默认注入 prompt | 原因 |
| --- | --- | --- | --- |
| Cloud VLM image caption | 是 | 是 | 目录名通常是角色先验 |
| Cloud VLM video caption | 是 | 是 | 视频 caption 同样可用角色先验 |
| Vision API image/video caption | 是 | 是 | 与 VLM caption 同语义 |
| Local VLM image/video caption | 是 | 是 | 与 VLM caption 同语义 |
| Codex subscription image caption | 是 | 是 | 结构化 caption 仍需角色先验 |
| OCR PDF/document | 是 | 否 | 角色 prompt 会污染转写 |
| OCR document image | 是 | 否 | 用户目标是还原文档，不是角色描述 |
| Audio caption / ASR | 是 | 否 | 音频目录名不是视觉角色上下文 |
| Local LLM translation | 是 | 否 | 翻译任务不应注入角色名 |

注意：这里的“所有模型”不是“所有模型都改 prompt”，而是“所有模型都通过统一上下文机制处理这个开关”。

## 涉及文件

新增：

```text
module/providers/directory_name_context.py
tests/test_directory_name_context.py
tests/test_directory_name_provider_contract.py
```

修改：

```text
module/providers/base.py
module/providers/ocr_base.py
module/providers/local_vlm/gemma4_local.py
module/caption_pipeline/orchestrator.py
gui/wizard/step4_caption.py
gui/main.py
tests/provider_v2_helpers.py
tests/test_provider_v2.py
tests/test_caption_pipeline.py
```

可选清理：

```text
utils/name_series.py
```

把 `utils.stream_util.split_name_series()` 和其他重复实现收敛为单一来源。首版可以暂不做，避免扩大行为变化。

## 测试计划

### 1. 纯 helper 测试

覆盖：

- `dir_name=False` 返回空上下文。
- image mime + `Alice (Wonderland)` 解析为 `<Alice> from (Wonderland)`。
- video mime 使用同样解析。
- audio mime 不注入。
- PDF mime 不注入。
- `directory_name_source_uri` 优先级高于 `uri`。
- pair image 不从 `pair_dir` 取目录名。
- 根路径、空目录名、非法路径不抛异常。
- 多角色目录名保持现有 `split_name_series()` 行为。

### 2. Provider contract 测试

新增轻量参数化测试，不真实调用模型：

1. 构造 provider。
2. patch `prepare_media()` 返回 image/video `MediaContext`。
3. patch `attempt()` 捕获 `PromptContext`。
4. 调 `execute()`。
5. 断言：

```python
prompts.character_name == "<Alice> from (Wonderland)"
prompts.character_prompt in prompts.user
prompts.user.startswith(prompts.character_prompt)
```

覆盖 provider 分组：

- `kimi_code`
- `kimi_vl`
- `mimo`
- `minimax_api`
- `minimax_code`
- `openai_compatible`
- `stepfun`
- `qwenvl`
- `ark`
- `glm`
- `codex_subscription`
- `gemini`
- `mistral_ocr` 的 image caption 路径
- `moondream`
- `qwen_vl_local`
- `step_vl_local`
- `penguin_vl_local`
- `reka_edge_local`
- `lfm_vl_local`
- `gemma4_local`
- `marlin_2b_local`

### 3. 不污染测试

覆盖：

- `ocr_model=deepseek_ocr` + `application/pdf` + `dir_name=True` 不前置角色 prompt。
- `document_image=True` 的 OCR image 不默认注入。
- `alm_model=music_flamingo_local` + audio 不注入。
- `acestep_transcriber_local` 这种 `consumes_prompts=False` 的 provider 不解析 prompt。

### 4. 视频来源测试

构造：

```text
datasets/Alice (Wonderland)/movie.mp4
datasets/Alice (Wonderland)/movie_clip/movie_000.mp4
```

断言 clip 调用时：

```python
args.directory_name_source_uri == str(original_movie_path)
```

provider 层再断言角色名来自 `Alice (Wonderland)`，不是 `movie_clip`。

## 验收标准

1. `dir_name` 在 GUI / CLI / PowerShell 中只有一个全局开关。
2. image/video caption provider 都通过 contract test。
3. provider 代码中不再新增 `Path(uri).parent.name` 风格的目录名解析。
4. 视频 clip 的角色名来自原始视频路径。
5. pair image 的角色名来自主图路径。
6. OCR / PDF / audio / translation 不被默认目录名 prompt 污染。
7. 现有 Kimi / Codex 目录名测试继续通过。
8. 新 provider 如果绕开统一 prompt 解析，测试会失败。

## 实施顺序

1. 加 `directory_name_context.py` 和纯 helper 测试。
2. 改 `Provider.resolve_prompts()` 使用 helper，保留 `_get_character_prompt()` wrapper。
3. 修正重写 `resolve_prompts()` 的 provider，使其调用共享 helper 或明确禁用注入。
4. 在视频切片路径写入 `directory_name_source_uri`。
5. GUI 把 `dir_name` 移到全局 caption 设置。
6. 加 provider contract 测试。
7. 加不污染测试。
8. 跑定向测试：

```powershell
uv run pytest tests/test_directory_name_context.py tests/test_directory_name_provider_contract.py tests/test_provider_v2.py tests/test_caption_pipeline.py
```

## Linus 风格审查点

- 不要把 `dir_name` 复制进每个 provider。复制是 bug 的温床。
- 不要为了“所有模型”污染 OCR 和 ASR。全局能力不等于全局注入。
- 不要让 prompt resolver 读文件路径。resolver 只负责 prompt 选择。
- 不要用当前 clip 路径猜原始语义。派生文件不是数据集身份。
- 不要新增第二套用户开关。旧的 `--dir_name` 已经够了。
- 不要把 GUI 的模型 panel 变成垃圾抽屉。模型无关选项必须上移。

## 与旧设计的关系

已有文档：

```text
docs/superpowers/specs/2026-05-21-directory-name-prompt-context-design.md
```

那份文档主要解决 image/video provider 的 prompt 上下文通用化。本 spec 是更上层的全局产品/工程契约，明确：

- `dir_name` 是全局选项，不是模型选项。
- 所有 provider 共享目录名上下文 API。
- 默认注入范围必须按任务语义收窄。
- GUI 配置归属必须从模型局部上移到 caption 全局设置。
