# Music Flamingo ALM 设计

## 背景

当前仓库的 Provider V2 体系已经覆盖：

- Cloud VLM
- Local VLM
- OCR
- Vision API

但本地音频语言模型还没有独立路由。现有 `audio/*` 流程可以消费音频并最终导出 `.srt`，但是在 Provider 层没有一个与图像/视频 VLM 解耦的本地 ALM 抽象。

本次设计的目标是在不破坏现有 CLI、GUI 和 Lance 编排流程的前提下，引入基于 Hugging Face 的本地音频语言模型路由，并以 `nvidia/music-flamingo-think-2601-hf` 作为首个落地模型。

## 已确认决策

- 首个 ALM 固定为 `nvidia/music-flamingo-think-2601-hf`
- 采用独立路由方案，不复用 `vlm_image_model`
- 默认音频 prompt 仍然做“描述型音频字幕”，不是严格逐字转录
- 真正的转录模型未来由独立 ASR 路线承担
- 音频 SRT 的分段、合并、后处理思路参考现有 Gemini/编排链路
- GUI 不新开页面，在现有 OCR / VLM 配置区旁新增 `ALM` 下拉框
- 对 `music_flamingo_local`，当用户未显式指定 `segment_time` 时，生效默认值使用模型卡声明的 20 分钟，即 `1200` 秒

## 目标

1. 为本地音频模型引入清晰的 Provider V2 路由和抽象层。
2. 接入 `music-flamingo-think-2601-hf`，支持对音频生成描述型 `.srt` 输出。
3. 保持当前 orchestrator、Lance 数据回写、字幕导出行为基本不变。
4. 将模型的特殊行为（`<think>` 输出、20 分钟上限、特殊依赖）约束在 ALM 路径内，不污染现有 VLM/OCR 逻辑。

## 非目标

- 本次不实现 ASR 逐字转录
- 本次不接入第二个 ALM 作为正式支持模型
- 本次不实现 OpenAI-compatible server backend 作为首发路径
- 本次不承诺 `music-flamingo-think-2601-hf` 产出可替代 ASR 精度的字幕

## 模型约束

`nvidia/music-flamingo-think-2601-hf` 当前有几个必须在设计中显式处理的约束：

- 许可是非商用研究用途，不适合作为默认“可商用”能力来描述
- 模型卡声明单样本最长输入为 20 分钟，超出部分不能直接整段送入
- 模型存在 `<think> ... </think>` 推理痕迹输出
- 当前依赖专用的 Transformers 分支，而不是仓库里其他 local provider 共享的稳定版本矩阵

## 架构设计

### 1. 新增独立 ALM 路由

在现有两条本地路由之外新增第三条：

- `ocr_model`
- `vlm_image_model`
- `alm_model`

`alm_model` 只服务 `audio/*` 媒体，不与图像、视频或文档路径混用。

对应改动：

- `module/captioner.py`
- `module/providers/catalog.py`
- `module/providers/registry.py`
- `module/providers/__init__.py`
- `tests/provider_v2_helpers.py`
- `tests/test_provider_routes.py`
- `tests/test_provider_catalog.py`

### 2. 新增本地音频 Provider 基类

新增 `LocalALMProvider`，定位与 `LocalVLMProvider` 平级，而不是其子类。

职责：

- 仅处理 `audio/*`
- 统一模型懒加载与缓存
- 封装音频媒体准备逻辑
- 统一 prompt 解析入口
- 统一 direct backend 推理入口
- 为后续其他 ALM 预留复用边界

建议文件：

- `module/providers/local_alm_base.py`
- `module/providers/local_alm/__init__.py`

### 3. 首个实现：`music_flamingo_local`

新增 `music_flamingo_local` provider，对应模型：

- `nvidia/music-flamingo-think-2601-hf`

职责：

- 模型与 processor 加载
- 按模型 chat template 组装音频消息
- 调用 `MusicFlamingoForConditionalGeneration`
- 提取最终回答，去掉 `<think> ... </think>`
- 返回给现有 postprocess / orchestrator

建议文件：

- `module/providers/local_alm/music_flamingo_local.py`

Provider 名称和路由值统一使用：

- canonical provider: `music_flamingo_local`
- route value: `music_flamingo_local`

### 4. Registry 与 discovery 集成

这部分不能省略。当前 `ProviderRegistry.discover()` 只扫描既有四类 package，新增 `local_alm` 子包后，必须同步更新 discovery 和 priority。

明确要求：

- `module/providers/registry.py` 需要把 `local_alm` 纳入 discover 扫描列表
- `ProviderRegistry._priority_order` 需要增加 `music_flamingo_local`
- `module/providers/__init__.py` 需要导出 `LocalALMProvider`

如果缺这一步，`alm_model` 只会停留在 catalog/CLI 层，运行时无法真正命中 provider。

## Prompt 设计

### 1. 默认语义

默认 prompt 仍然是“描述型音频字幕”，不是“逐字转录字幕”。

也就是说，模型需要生成适合 deaf/hearing-impaired 阅读的音频场景描述型 SRT，而不是以词级别转录为目标。

### 2. Provider 专属 prompt

在 `config/prompts.toml` 新增：

- `music_flamingo_audio_system_prompt`
- `music_flamingo_audio_prompt`

优先级：

1. `music_flamingo_audio_*`
2. 通用 `audio_system_prompt` / `audio_prompt`

这样既能给首个模型专门约束输出格式，也不会破坏现有 audio prompt 的 fallback。

### 3. Prompt 内容要求

provider 专属 prompt 必须显式要求：

- 输出 SRT
- 使用 markdown code block 包裹
- 时间戳格式为 `HH:MM:SS,mmm` 或至少可被现有规范化逻辑修复
- 每条结束时间大于开始时间
- 内容以音频描述为主，不要求逐字转录

## 数据流设计

### 1. 路由命中

当满足以下条件时命中 `music_flamingo_local`：

- `mime.startswith("audio")`
- `args.alm_model == "music_flamingo_local"`

这条路由优先级应独立于 `ocr_model` / `vlm_image_model`，避免误判。

### 2. 媒体准备

`prepare_media()` 保持轻量，不在此阶段把整段音频解码到内存。

建议只准备：

- `uri`
- `mime`
- `file_size`
- `duration_ms`
- `modality = AUDIO`

实际音频文件交给 processor / model 按路径读取或延迟加载。

### 3. 有效默认 `segment_time`

仓库现有 `segment_time` 是全局参数。本次不能只在文档层说“用户未显式指定时使用 1200”，因为当前参数流默认值会抹掉“是否显式指定”的信息。

首选设计：

- `module/captioner.py` 中 `--segment_time` 的 parser 默认值改为 `None`
- 在运行时归一化阶段生成 `effective_segment_time`
- 业务逻辑统一消费 `effective_segment_time`，而不是直接读取原始 `args.segment_time`

规则：

- 如果用户显式传入 `--segment_time`，始终使用用户值
- 如果用户未显式传入，且 `alm_model == "music_flamingo_local"`，则生效值为 `1200`
- 如果用户未显式传入，且不是 `music_flamingo_local`，则保持现有默认值 `600`

这样可以做到：

- 不影响现有视频 provider
- 不影响其他本地模型
- 符合模型卡的 20 分钟限制

建议把该逻辑集中放在运行时参数归一化或 orchestrator 入口附近，而不是散落在 provider 内。

对应联动要求：

- GUI 只有在用户实际修改 `segment_time` 后才追加 `--segment_time=...`
- `4、run.ps1` 只有在用户覆盖默认值时才透传 `--segment_time`
- 否则由后端统一计算 `effective_segment_time`

### 4. 长音频处理

长音频继续使用现有 orchestrator 分段与合并框架，不新增独立音频编排器。

行为：

- 音频长度小于等于有效 `segment_time`：单次调用 provider
- 音频长度大于有效 `segment_time`：按现有分段逻辑切片
- 每段结果产出 SRT，再通过现有偏移平移逻辑合并

## 输出与后处理

### 1. Provider 级清洗

`music_flamingo_local` 在返回前做最小必要清洗：

- 去掉 `<think> ... </think>`
- 如果存在明显的 final answer 区段，仅保留最终回答
- 不在 provider 内实现完整 SRT 解析器

Provider 只负责把“可能可消费的模型最终文本”交给下游。

### 2. Provider 级校验与重试挂钩

非法 SRT 的重试不能依赖 orchestrator 之后的后处理，因为当前 retry loop 只包裹 `Provider.attempt()`。

因此首版必须把“可重试的 SRT 合法性校验”挂到 provider 执行链内部，推荐位置：

- `music_flamingo_local.post_validate()`

行为：

- 对 provider 清洗后的输出执行 SRT 抽取与最小合法性校验
- 若结果不满足基本字幕约束，则抛出可重试错误
- 让现有 retry loop 生效

推荐做法：

- 把 SRT 抽取/校验逻辑下沉为共享 helper
- `post_validate()` 与 `postprocess_caption_content()` 复用同一套 helper
- 避免 provider 内和 postprocess 内各维护一份独立规则

### 3. Postprocess 级处理

继续复用当前 `postprocess_caption_content()` 的职责风格，对音频输出做统一处理：

- 从 markdown code block 中抽出 SRT 内容
- 对时间戳做规范化
- 尝试解析为合法字幕
- 对已经通过 provider 内最小校验的结果做最终归一化，不承担 retry 决策

必要时可新增辅助函数，例如：

- `_strip_reasoning_sections()`
- `_extract_srt_block()`
- `_validate_srt_content()`

### 4. 文件写出

音频输出仍然写为：

- `.srt`

由现有 `utils/output_writer.py` 保持该行为，不修改音频默认扩展名。

## 错误处理

### 1. 依赖错误

当缺少模型所需的特定 Transformers 分支或相关依赖时：

- 直接报清晰错误
- 不静默回退到其他 provider
- README、脚本和日志中都要说明此模型有额外依赖

### 2. 模型加载错误

如模型权重缺失、processor 初始化失败、设备不兼容：

- 直接失败当前文件
- 输出明确错误信息

### 3. OOM

对 CUDA OOM 不进行高次数盲重试。

策略参考当前本地 provider：

- 可分类为快速失败
- 提示用户减小分段长度、切换精度或改设备

### 4. 非法 SRT 输出

如果模型输出不是合法 SRT：

- 必须在 provider 执行链内部识别并抛出可重试错误
- 先走现有 retry 机制进行有限重试
- 重试仍失败时，显式报错
- 不应默默把原始自然语言文本保存成 `.srt`

这里的关键约束是：

- retry 触发点在 provider 内
- postprocess 只做最终规范化
- 不能把“校验失败后重试”写成纯 orchestrator 行为

## GUI 设计

GUI 不新增独立音频配置页面，而是在现有 OCR / VLM 配置区扩展为三组：

- `OCR`
- `VLM`
- `ALM`

其中：

- `ALM` 是独立下拉框
- 首版下拉值只有 `music_flamingo_local`
- 音频处理时使用 `ALM` 选择结果

建议改动：

- `gui/wizard/step4_caption.py`
- `gui/utils/i18n.py`
- `tests/test_penguin_dependencies.py`

必要时同步：

- 参数收集
- 依赖提示
- 与 `segment_time` 的联动展示
- `_has_local_route_config()` 需要把 `alm_model` 视为本地可执行 provider
- `_build_local_extra_args()` 需要追加 `music-flamingo-local`
- “至少配置一个 provider” 的提示文案需要从 OCR/VLM 扩展为 OCR/VLM/ALM
- 相关 GUI 测试需要覆盖新下拉框与 extra 映射

## 配置与文档

### 1. `pyproject.toml`

必须新增本地依赖 extra，例如：

- `music-flamingo-local`

用途：

- 安装 `music-flamingo-think-2601-hf` 所需的特定 Transformers 分支和相关依赖

这一步是正式接入的一部分，不是 README 附注。当前仓库的本地 OCR/VLM 都依赖 extra 机制，ALM 也必须进入同一条依赖链。

### 2. `config/model.toml`

新增 `music_flamingo_local` section，至少包含：

- `model_id = "nvidia/music-flamingo-think-2601-hf"`
- provider 所需的 generation 参数
- 对特殊依赖的注释

### 3. `config/prompts.toml`

新增 provider 专属 audio prompt。

### 4. `4、run.ps1`

新增：

- `alm_model` 变量
- 参数透传
- 对 `music_flamingo_local` 的依赖提示
- `Add-UvExtra "music-flamingo-local"`

脚本层要求：

- 仅在用户覆盖分段长度时透传 `--segment_time`
- 默认情况下让后端计算 `effective_segment_time`

### 5. `README.md`

补充：

- `music_flamingo_local` 的用途
- 非商用许可说明
- 安装与依赖说明
- 20 分钟限制
- 默认是描述型音频字幕，不是 ASR

### 6. `gui/wizard/step4_caption.py`

必须新增：

- `ALM_MODELS = list(route_choices("alm_model"))`
- `ALM_EXTRA_MAP`
- `alm_model` 下拉框
- 本地 provider 判定和参数拼装逻辑

### 7. `gui/utils/i18n.py`

必须新增或修改：

- `alm_model` 标签文案
- OCR/VLM/ALM 相关提示文案
- “至少配置一个 provider” 的说明文本

## 测试设计

至少覆盖以下测试：

### 1. 路由测试

- `audio/* + alm_model=music_flamingo_local` 命中 provider
- 非音频 mime 不会误命中
- `route_choices("alm_model")` 正确暴露 canonical route
- `normalize_runtime_args()` 正确规范化 `alm_model`

### 2. Prompt 测试

- provider 专属 prompt 优先于通用 prompt
- 未配置专属 prompt 时能正确 fallback

### 3. 输出清洗测试

- 含 `<think>` 的输出可正确提取最终 answer
- markdown code block 中的 SRT 可被抽取
- 裸 SRT 也能被消费

### 4. 后处理测试

- 时间戳可规范化
- 错序或非法时间戳会失败
- 非 SRT 文本不会被静默写出成 `.srt`
- provider 内 `post_validate()` 能把非法结果转换成可重试错误

### 5. 分段合并测试

- 多段音频结果在合并后索引连续
- 时间偏移平移正确

### 6. 依赖与 GUI 测试

- `pyproject.toml` 声明 `music-flamingo-local` extra
- `4、run.ps1` 包含 `alm_model` 与对应 extra
- GUI 的 `_build_local_extra_args()` 能追加 `music-flamingo-local`
- GUI 的 provider 校验逻辑接受 `alm_model`
- i18n 文案不再只提 OCR/VLM

## 风险

### 1. 模型用途风险

`music-flamingo-think-2601-hf` 偏音乐理解与推理，不是专业 ASR。即使 prompt 要求 SRT，它更擅长“描述型字幕”而非逐字转录。

### 2. 依赖风险

该模型依赖专用 Transformers 分支，可能与仓库现有 provider 的依赖矩阵形成冲突。

### 3. 输出稳定性风险

thinking 模型更容易生成多余解释、推理痕迹或非严格 SRT 格式，因此必须把清洗和校验作为首版必需能力，而不是可选优化。

## 实施顺序建议

1. 新增 `alm_model` 路由与 catalog
2. 接入 `local_alm` discovery、priority 与导出
3. 增加 `music-flamingo-local` extra 与脚本/GUI extra 映射
4. 实现 `LocalALMProvider`
5. 实现 `music_flamingo_local`
6. 打通 prompt 解析、`<think>` 清洗与 `post_validate()` 校验
7. 加强音频 SRT 后处理与校验
8. 更新脚本、GUI、i18n、README
9. 补齐路由、依赖与输出测试

## 结论

本设计采用独立 ALM 路由的方案，把 `music-flamingo-think-2601-hf` 作为首个本地音频语言模型接入到 Provider V2 架构中。默认输出目标是“描述型音频 SRT”，而不是逐字转录；超长音频继续复用当前编排链路，但对该模型施加 20 分钟有效默认分段。GUI 仅在现有 OCR / VLM 区域增加 `ALM` 下拉框，不新增独立页面。
