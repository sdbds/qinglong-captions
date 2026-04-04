# ALM 任务契约与 Cohere Transcribe 设计

## 背景

当前仓库已经有独立的 `alm_model` 路由，并且 `local_alm` 家族里已经同时存在两类能力：

- 音频理解 / 描述型模型
- 音频转录 / transcript 型模型

这本身没有问题。问题不在“ASR 算不算 ALM”，而在于当前代码没有把这两类任务差异显式建模。

现有实现默认把 `alm_model` 当成“本地音频 captioning” 路由来处理，但实际 provider 行为已经不一致：

- `music_flamingo_local` / `eureka_audio_local` 更接近 prompt 驱动的描述型任务
- `acestep_transcriber_local` 已经是 transcript 输出
- 新增的 `cohere_transcribe_local` 也是 transcript 输出，并且直接走 `model.transcribe(...)`

这说明现有系统真正缺的不是“再拆一个 ASR 家族”，而是：

1. `ALM` 内部的任务契约
2. 任务所需运行时参数的归属边界
3. 长音频在不同任务类型下的分段与合并策略

本次设计的目标不是重做路由拓扑，而是在保留 `alm_model` 入口的前提下，把任务契约补齐，并据此正式接入 `cohere_transcribe_local`。

## 已确认决策

- `cohere_transcribe_local` 继续留在 `local_alm` 家族，不新建独立 `asr_model` 顶级路由
- 公共入口继续使用 `alm_model`
- `alm_model` 的对外语义从“本地音频 captioning” 收敛为“本地音频任务模型”
- `ALM` 家族允许多个任务类型，但每个 provider 必须显式声明自己的任务契约
- `cohere_transcribe_local` 的任务类型固定为 `transcribe`
- `music_flamingo_local` / `eureka_audio_local` 的任务类型固定为 `caption`
- `acestep_transcriber_local` 需要一并迁移到新的 transcript 契约，避免家族内部出现半旧半新的双轨语义
- `language` 不再使用 provider 内硬编码默认值；对需要语言的任务，语言是显式运行时参数
- 长音频合并必须按任务类型处理，不能把 transcript 结果继续走“分段摘要拼接”
- `4、run.ps1`、GUI、测试和 provider 注册必须视为同一交付单元，不能只改部分入口

## 目标

1. 在不拆分 `alm_model` 路由的前提下，为 `ALM` provider 增加显式任务契约。
2. 按新契约接入 `cohere_transcribe_local`，并修复当前设计审查中暴露出的语言参数和长音频合并问题。
3. 让 transcript 型 provider 与 caption 型 provider 可以共存于同一条 `alm_model` 路由下，而不会互相污染行为。
4. 保持当前 Provider V2、Lance、CLI、GUI 和脚本入口结构基本不变。
5. 明确 `cohere_transcribe_local`、`acestep_transcriber_local` 与描述型 ALM 的边界，减少未来再接入音频模型时的路径依赖。

## 非目标

- 本次不新建 `asr_model` / `transcriber_model` 顶级路由
- 本次不引入 diarization、word-level timestamps 或强制对齐
- 本次不把所有 ALM 都统一成 transcript 输出
- 本次不重构 OCR / VLM 路由体系
- 本次不引入“自动语言识别后再转录”的二阶段流水线

## 现状问题

### 1. 路由语义和 provider 语义不一致

当前 `module/captioner.py` 中 `--alm_model` 的帮助文本仍然是“local audio captioning”，但 transcript 型 provider 实际并不做 caption 语义上的工作。

这不是命名吹毛求疵，而是一个真实兼容性问题：

- 用户以为自己在选“音频描述模型”
- 实际运行时却可能命中“纯转录模型”
- 编排层也因此沿用了不适合 transcript 的默认处理逻辑

### 2. `language` 被错误地固化成 provider 默认值

`cohere_transcribe_local` 当前原型里把 `language` 默认写成 `zh`。这会把一个任务上下文参数伪装成模型配置默认值。

后果：

- 非中文音频会静默走错语言
- GUI / CLI 没有显式表达“当前模型需要语言参数”
- provider 行为依赖本地 TOML 默认值，而不是当前任务输入

根因不是 Cohere 特殊，而是系统没有定义“ALM 运行时参数”和“模型配置”的边界。

### 3. 分段合并逻辑默认把 dict 输出当成摘要

当前 `module/caption_pipeline/orchestrator.py` 对分段后的 dict 输出统一走：

- 取 `description`
- 添加 `Segment N [start-end]`
- 合并成段落摘要文本

这对 caption 型任务可接受，但对 transcript 型任务是错误的。

对 transcript 任务来说，正确行为应该是：

- 保留原始转录文本语义
- 不强行加 “Segment N” 摘要头
- 允许按段拼接 transcript，而不是按摘要样式重写

### 4. prompt 是否参与任务没有显式声明

caption 型 ALM 需要 prompt。

`cohere_transcribe_local` 这种 transcript 型任务实际上不消费 prompt，但当前执行链仍然会统一构建 prompt，再由 provider 内部忽略。虽然功能上能跑，但结构上不诚实。

这会导致：

- route 级语义继续模糊
- provider 自己吞掉 prompt 差异
- 以后接入更多 transcript 型 provider 时，逻辑会继续散在各个实现里

### 5. 脚本与测试存在交付原子性风险

当前 GUI、registry、tests、`4、run.ps1` 都会参与 `alm_model` 的落地能力。如果脚本入口改动只停留在本地未跟踪文件，功能就不是完整交付。

这不是流程洁癖，而是实际兼容性风险：

- GUI 和测试会以为脚本支持已存在
- 用户从脚本入口运行时却可能拿不到对应依赖或参数透传

## 架构设计

### 1. 保留单一 `alm_model` 路由

本次不拆路由，继续保持：

- `ocr_model`
- `vlm_image_model`
- `alm_model`

但 `alm_model` 的对外语义更新为：

- “本地音频任务模型”

而不是：

- “本地音频 captioning 模型”

对应改动：

- `module/captioner.py`
- `gui/utils/i18n.py`
- `README.md`
- `gui/wizard/step4_caption.py`

### 2. 在 `LocalALMProvider` 层引入任务契约

新增一个集中声明差异的契约对象，而不是在 orchestrator、provider、GUI 里各自写一套 if/else。

建议新增：

- `ALMTaskContract`

建议位置：

- `module/providers/local_alm_base.py`

建议字段：

```python
@dataclass(frozen=True)
class ALMTaskContract:
    task_kind: Literal["caption", "transcribe"]
    consumes_prompts: bool
    requires_language: bool = False
    default_caption_extension: str = ".txt"
```

约束：

- 每个 `LocalALMProvider` 子类必须声明 `task_contract`
- transcript / caption 差异只能通过这一个契约对象表达
- 不允许 orchestrator 再靠 provider 名字判断 transcript 还是 caption

这样做的原因很简单：

- 不新增新路由
- 不引入一堆散落布尔开关
- 用一个小对象把真正的差异收拢

### 3. provider 分类

本次明确：

- `music_flamingo_local`
  - `task_kind = "caption"`
  - `consumes_prompts = True`
- `eureka_audio_local`
  - `task_kind = "caption"`
  - `consumes_prompts = True`
- `acestep_transcriber_local`
  - `task_kind = "transcribe"`
  - `consumes_prompts = True`
- `cohere_transcribe_local`
  - `task_kind = "transcribe"`
  - `consumes_prompts = False`
  - `requires_language = True`

这里特别说明：

- `acestep_transcriber_local` 虽然仍然消费 prompt，但其输出契约已经是 transcript，不应继续走摘要合并
- `cohere_transcribe_local` 不消费 prompt，这不是“掉队”，而是它的任务契约就是 direct ASR

### 4. Provider.execute 按契约决定是否构建 prompt

当前 `Provider.execute()` 会统一调用 `resolve_prompts()`。

本次建议改为：

- 如果 provider 没有显式任务契约，保持现状
- 如果 `task_contract.consumes_prompts = False`，则跳过 prompt 解析，构造空 `PromptContext`

建议实现位置：

- `module/providers/base.py`
- 或在 `LocalALMProvider` 中覆写执行前准备逻辑

要求：

- `cohere_transcribe_local` 不再依赖“拿到 prompt 再忽略”的隐式行为
- transcript 任务不必维护无意义的 provider 专属 prompt 配置

### 5. transcript 输出契约

caption 型任务继续输出：

- `description`
- `caption_extension = ".txt"`

transcript 型任务的 canonical 文本字段改为：

```python
{
    "task_kind": "transcribe",
    "transcript": "...",
    "caption_extension": ".txt",
    "provider": "cohere_transcribe_local"
}
```

要求：

- `transcript` 是 transcript 型任务的唯一真相源
- transcript 型 provider 不再重复写入同内容的 `description`
- transcript 型 provider 的 `post_validate()` 必须返回 `task_kind = "transcribe"`

兼容性放在读侧，而不是写侧复制字段。

需要修改以下共享读侧逻辑，使其统一按“`transcript` 优先，`description` 回退”处理：

- `utils/output_writer.py`
- `module/caption_pipeline/orchestrator.py`
- `module/providers/base.py` 中 `CaptionResult.description`

规则：

1. 如果 payload 含 `transcript`，优先使用 `transcript`
2. 否则回退到 `description`
3. 再回退到 `long_description` / `short_description`

### 6. 长音频合并按 `task_kind` 处理

当前 `_build_segment_summary_payload()` 只适用于 caption。

本次建议把分段 dict 合并拆成两条明确路径：

- `task_kind = "caption"`
  - 保持现有 `Segment N [start-end] + description` 的摘要拼接
- `task_kind = "transcribe"`
  - 按段拼接 transcript
  - 不自动加摘要头
  - 可保留 segments 元数据，但文本正文必须是 transcript 直连后的结果

建议新增：

- `_build_segment_transcript_payload()`

建议位置：

- `module/caption_pipeline/orchestrator.py`

行为：

- 对每个 chunk 的 payload，优先读取 `transcript`
- 若不存在 `transcript`，可回退到 `description`
- 最终合并结果写回：

```python
{
    "task_kind": "transcribe",
    "transcript": "...",
    "caption_extension": ".txt",
    "segments": [...]
}
```

关键约束：

- transcript 合并不再带 `Segment 1` 这样的摘要头
- 不允许 transcript provider 继续默认复用 `_build_segment_summary_payload()`

## 运行时参数设计

### 1. 新增显式 `alm_language`

对于 transcript 任务，特别是 `cohere_transcribe_local`，`language` 是运行时参数，不是模型默认。

建议新增 CLI 参数：

- `--alm_language`

建议位置：

- `module/captioner.py`

GUI 新增字段：

- `alm_language`

建议位置：

- `gui/wizard/step4_caption.py`
- `gui/utils/i18n.py`

脚本透传：

- `4、run.ps1`

### 2. 语言解析优先级

统一规则：

1. 如果用户显式传入 `--alm_language`，使用该值
2. 否则读取 provider config 中的 `language`
3. 如果任务契约 `requires_language = True` 且仍为空，则直接报错

明确要求：

- `cohere_transcribe_local` 不再在 provider 类里写死 `language = "zh"`
- `config/model.toml` 可以保留 `language` 作为用户级默认配置，但不应再以“代码常量”形式存在
- README 和 GUI 都必须明确告知：该 provider 需要显式语言

### 3. 是否做自动语言识别

本次不做自动语言识别。

原因：

- 这会引入新的模型 / 依赖 / 二阶段错误面
- 当前真实问题是参数归属不清，不是缺语言识别器

## 分段策略设计

### 1. `segment_time` 不能继续对 transcript provider 盲目套用 caption 默认值

当前 `normalize_runtime_args()` 对大多数 provider 使用通用 `600` 秒默认值。

这对 caption 型任务可以接受，但 transcript 型任务至少不能继续“默认 600 + 摘要合并”。

本次设计要求：

- transcript 合并策略修正是必须项
- `cohere_transcribe_local` 的默认分段策略改为“未显式传入 `segment_time` 时不强制外部分段”
- 如果用户显式传入 `--segment_time`，则允许 transcript provider 走切段 + transcript_concat

建议实现方式：

- `normalize_runtime_args()` 允许 `effective_segment_time = None`
- orchestrator 在 `effective_segment_time is None` 时跳过外部分段

规则：

- `music_flamingo_local`
  - 未显式传值时仍使用 `1200`
- 其他 caption 型 ALM
  - 未显式传值时维持现有默认策略
- `cohere_transcribe_local`
  - 未显式传值时不强制外部分段
- `acestep_transcriber_local`
  - 可在同一实现中迁移到 transcript 合并；是否保留固定默认分段由实现评估，但不能再走摘要合并

### 2. transcript 分段元数据

即使 transcript 正文不带 `Segment N` 标题，仍建议保留 segment 元数据，便于后续扩展：

- `index`
- `start_seconds`
- `end_seconds`
- `transcript`
- `provider`

要求：

- 元数据写入 `.json`
- 正文写入 `.txt`

## `cohere_transcribe_local` 专项设计

### 1. provider 位置与命名

保留：

- canonical provider: `cohere_transcribe_local`
- route value: `cohere_transcribe_local`
- provider package: `module/providers/local_alm/cohere_transcribe_local.py`

### 2. 推理路径

继续使用：

- `AutoProcessor`
- `AutoModelForSpeechSeq2Seq`
- `model.transcribe(...)`

首版不把它改造成 chat-template provider。

### 3. 输出契约

`cohere_transcribe_local.post_validate()` 必须至少返回：

```python
{
    "task_kind": "transcribe",
    "transcript": normalized_text,
    "description": normalized_text,
    "caption_extension": ".txt",
    "provider": "cohere_transcribe_local"
}
```

### 4. 配置项

在 `config/model.toml` 中保留：

- `model_id`
- `language`
- `punctuation`
- `compile`
- `pipeline_detokenization`
- `batch_size`

但要求变更如下：

- `language` 不再由 provider 默认常量补齐
- 注释要明确区分“可在配置中预设”与“必须显式存在”

### 5. GUI 与脚本

以下位置继续接入：

- `gui/wizard/step4_caption.py`
- `gui/utils/process_runner.py`
- `4、run.ps1`
- `pyproject.toml`
- `README.md`

新增要求：

- GUI 当选中 `cohere_transcribe_local` 时，展示 `alm_language`
- `4、run.ps1` 支持透传 `--alm_language`
- 依赖 extra 继续使用 `cohere-transcribe-local`

## `acestep_transcriber_local` 联动要求

本次 spec 不允许只把 `cohere_transcribe_local` 特判成 transcript，而让 `acestep_transcriber_local` 继续留在旧摘要路径。

要求：

- `acestep_transcriber_local` 迁移到 `task_kind = "transcribe"`
- 分段后结果按 transcript 任务路径合并
- 保留其 prompt 驱动生成逻辑，但输出契约与 transcript 型任务对齐

原因：

- 否则 `ALM` 家族内部仍然会有两套 transcript 处理逻辑
- 这会让这次 spec 只修 Cohere，不修结构

## 配置与文档设计

### 1. `module/captioner.py`

需要修改：

- `--alm_model` help 文案
- 新增 `--alm_language`
- 对 `segment_time` 的说明从“全局固定默认值”改为“provider-specific / task-aware default when unset”

### 2. `gui/wizard/step4_caption.py`

需要修改：

- `ALM_MODELS`
- `ALM_EXTRA_MAP`
- 参数收集逻辑
- 当 `alm_model` 需要语言时展示 / 收集 `alm_language`

### 3. `gui/utils/i18n.py`

需要新增：

- `alm_language`
- 对 `alm_model` 的更准确文案

### 4. `4、run.ps1`

需要修改：

- `alm_model` 依赖 extra 追加逻辑
- 新增 `alm_language` 参数透传
- 该文件必须作为正式 tracked 改动提交，不能只存在本地副本

### 5. `README.md`

需要补充：

- `alm_model` 现在包含 caption / transcribe 两类任务
- `cohere_transcribe_local` 需要接受 gated terms
- `cohere_transcribe_local` 需要显式语言
- transcript 型 ALM 输出 `.txt`，不是 `.srt`

## 测试设计

至少覆盖以下测试：

### 1. 任务契约测试

- 每个 `LocalALMProvider` 都声明 `task_contract`
- `cohere_transcribe_local.task_contract.task_kind == "transcribe"`
- `acestep_transcriber_local.task_contract.task_kind == "transcribe"`
- `music_flamingo_local` / `eureka_audio_local` 保持 `caption`

### 2. 语言参数测试

- `cohere_transcribe_local` 在 `args.alm_language="ja"` 时使用 `ja`
- `args.alm_language` 优先级高于 `model.toml`
- 对 `requires_language=True` 且未配置语言的 provider，直接报错
- 不再存在 provider 级硬编码 `zh`

### 3. prompt 参与测试

- `cohere_transcribe_local` 不消费 prompt 也能正常执行
- `music_flamingo_local` / `eureka_audio_local` 仍然消费 prompt
- `acestep_transcriber_local` 仍然消费 prompt

### 4. transcript 输出测试

- transcript provider 的 `post_validate()` 返回 `task_kind` 与 `transcript`
- transcript provider 不再重复返回同内容的 `description`
- transcript provider 写盘时生成 `.txt` 和 `.json`
- transcript 文本不被包装成 `Segment N` 摘要头

### 5. 分段合并测试

- caption provider 继续走 caption 合并路径
- transcript provider 走 transcript 合并路径
- transcript 分段合并后正文不包含 `Segment 1`
- transcript 分段元数据保留起止时间

### 6. CLI / GUI / 脚本测试

- `captioner.py` 暴露 `--alm_language`
- GUI 选中 `cohere_transcribe_local` 时会构造 `--alm_language=...`
- `4、run.ps1` 包含 `--alm_language`
- `test_penguin_dependencies.py` 继续覆盖 GUI 和脚本 extra 映射

## 风险

### 1. 契约引入风险

如果把 transcript / caption 差异拆成太多零散布尔字段，会产生新的复杂度。

缓解：

- 使用单一 `ALMTaskContract` 对象集中表达差异

### 2. 兼容性风险

修改 `Provider.execute()` 的 prompt 解析路径时，不能影响现有非 ALM provider。

缓解：

- 默认只对声明了 `task_contract` 的 `LocalALMProvider` 生效

### 3. 行为迁移风险

`acestep_transcriber_local` 从旧摘要合并迁移到 transcript 合并后，输出文本结构会变化。

这是允许的结构性修正，但必须：

- 更新测试
- 在 README / 变更说明中明确

### 4. 入口一致性风险

如果只改 provider 和 tests，不同步 `4、run.ps1` / GUI / i18n，用户仍然无法完整使用。

缓解：

- 把脚本、GUI、registry、tests 视为同一交付单元

## 实施顺序建议

1. 在 `local_alm_base.py` 新增 `ALMTaskContract`
2. 给现有 ALM provider 补齐 `task_contract`
3. 修改 `Provider.execute()` 或 `LocalALMProvider` 执行链，使 `consumes_prompts` 生效
4. 在 `captioner.py` / GUI / 脚本新增 `alm_language`
5. 修改 `cohere_transcribe_local`，移除硬编码 `zh`，按新优先级解析语言
6. 修改 `acestep_transcriber_local` 与 `cohere_transcribe_local` 的 `post_validate()`，统一 transcript payload
7. 修改共享读侧逻辑，使其优先读取 `transcript`
8. 在 `orchestrator.py` 新增 transcript 合并路径
9. 调整 `segment_time` 的 task-aware 默认策略
10. 补齐 CLI / GUI / registry / dependency / merge 路径测试
11. 最后更新 README 和脚本说明

## 结论

本设计不否认 “ASR 也是 ALM 的一个分支”，也不通过增加顶级路由来掩盖问题。真正的修正方式是在 `ALM` 家族内部引入显式任务契约，让 caption 型和 transcript 型 provider 可以共存于同一条 `alm_model` 路径下，同时把 `language` 的归属、prompt 的参与方式，以及长音频的合并策略都变成显式规则。

在这个前提下，`cohere_transcribe_local` 的接入才不是一次孤立的 provider 接线，而是对现有 `ALM` 结构缺口的正式修复。
