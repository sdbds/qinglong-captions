# Qianfan OCR Provider 设计

## 背景

当前仓库的 Provider V2 已经有一套稳定的 OCR 接入方式，现有本地 OCR provider 包括：

- `deepseek_ocr`
- `lighton_ocr`
- `dots_ocr`
- `hunyuan_ocr`
- `glm_ocr`
- `chandra_ocr`
- `olmocr`
- `paddle_ocr`
- `nanonets_ocr`
- `firered_ocr`

这些 provider 已经共享以下行为：

- 通过 `ocr_model` 路由选择 provider
- 对图片和 PDF 做 OCR 处理
- 输出目录默认使用输入文件去后缀后的同名目录
- 支持本地 `transformers` 直连推理
- 大多支持复用本地 OpenAI-compatible server backend
- GUI 与 `4、run.ps1` 通过 `uv --extra` 追加本地依赖

本次设计的目标是在不破坏现有 OCR 主链路的前提下，把 Hugging Face 上的 `baidu/Qianfan-OCR` 接入为新的本地 OCR provider。

参考模型页：

- `https://huggingface.co/baidu/Qianfan-OCR`

该模型相较于现有 OCR provider 的关键差异有两点：

- 本地直连调用入口是 `model.chat(..., question=...)`
- `Layout-as-Thought` 通过在 query 尾部追加 `<think>` 触发

## 已确认决策

- provider 名称固定为 `qianfan_ocr`
- 主路由继续走 `ocr_model`
- 默认任务是 “Parse this document to Markdown.”
- 默认开启 thinking 模式，即默认在最终 query 末尾追加 `<think>`
- 是否开启 thinking 由配置项控制，不拆第二个 provider
- 支持两种 prompt 组合策略：
  - `append`
  - `replace`
- 用户自定义 prompt 默认支持 `append` 和 `replace` 两种模式，由配置项决定
- direct backend 与 OpenAI-compatible server backend 共用同一套 question 组装逻辑
- v1 不新增专门的 KIE / JSON / HTML 子路由，先通过自定义 prompt 承载扩展任务
- 输出契约维持现有 OCR 习惯，不像 `dots_ocr` 那样引入 SVG 等特殊文件格式

## 目标

1. 在现有 OCR Provider V2 体系中新增 `qianfan_ocr` 路由。
2. 接入 `baidu/Qianfan-OCR` 作为默认本地模型。
3. 通过 TOML 配置支持：
   - thinking 开关
   - 自定义 prompt
   - `append` / `replace` prompt 策略
4. 保持图片 / PDF 输入、结果目录、Markdown 写盘和 GUI / 脚本集成方式与现有 OCR provider 一致。
5. 保证 direct backend 与 OpenAI-compatible server backend 在 question 拼装上不漂移。

## 非目标

- 本次不新增 `qianfan_ocr_think` 等子 provider
- 本次不改造 `OCRProvider` 基类为通用 prompt 组合框架
- 本次不为 `qianfan_ocr` 增加 GUI 专属配置页
- 本次不承诺覆盖模型卡中的全部任务类型
- 本次不引入 HTML / JSON / SVG 等额外输出格式

## 模型约束

根据 Hugging Face 模型页，`Qianfan-OCR` 的设计约束需要显式处理：

- 依赖 `trust_remote_code=True`
- 本地调用入口是 `AutoModel.from_pretrained(...).chat(tokenizer, pixel_values=..., question=...)`
- 默认文档解析 prompt 是 `Parse this document to Markdown.`
- thinking 模式通过在 query 末尾追加 `<think>` 触发
- 模型页明确指出：
  - 复杂版面建议开启 thinking
  - 简单单栏文档关闭 thinking 可能更快且更稳定

本次设计选择默认开启 thinking，但允许用户通过配置关闭。

## 架构设计

### 1. 新增 `qianfan_ocr` provider

新增文件：

- `module/providers/ocr/qianfan.py`

该 provider 与 `lighton_ocr`、`deepseek_ocr` 同级，继承 `OCRProvider`。

统一命名：

- canonical provider: `qianfan_ocr`
- route value: `qianfan_ocr`

### 2. 路由与 registry 集成

以下位置需要增加 `qianfan_ocr`：

- `module/providers/catalog.py`
- `module/providers/registry.py`
- `module/providers/ocr/__init__.py`
- `tests/test_provider_catalog.py`
- `tests/test_provider_registry.py`
- `tests/test_provider_routes.py`

要求：

- `route_choices("ocr_model")` 暴露 `qianfan_ocr`
- `ProviderRegistry._priority_order` 包含 `qianfan_ocr`
- `find_provider()` 在 `ocr_model="qianfan_ocr"` 时可以正确命中图片和 PDF OCR 路由

### 3. GUI 与脚本集成

以下位置需要增加 `qianfan_ocr` 的本地依赖映射：

- `gui/wizard/step4_caption.py`
- `4、run.ps1`
- `tests/test_penguin_dependencies.py`

要求：

- GUI OCR 下拉框暴露 `qianfan_ocr`
- GUI 的 local extra 映射返回 `--extra qianfan-ocr`
- `4、run.ps1` 支持 `ocr_model="qianfan_ocr"` 并追加 `Add-UvExtra "qianfan-ocr"`

## 配置设计

### 1. Provider 配置

在 `config/model.toml` 和 `config/config.toml` 中新增：

```toml
[qianfan_ocr]
model_id = "baidu/Qianfan-OCR"
base_prompt = "Parse this document to Markdown."
custom_prompt = ""
prompt_strategy = "append"
think_enabled = true
max_new_tokens = 16384
```

字段职责：

- `model_id`: 默认模型权重
- `base_prompt`: provider 内建基础 prompt
- `custom_prompt`: 用户自定义 prompt
- `prompt_strategy`: `append` 或 `replace`
- `think_enabled`: 是否在最终 question 尾部追加 `<think>`
- `max_new_tokens`: 推理上限

### 2. 兼容性 prompt 配置

在 `config/prompts.toml` 保留：

- `qianfan_ocr_prompt = ""`

它的作用不是主配置入口，而是兼容仓库现有 OCR provider 的 prompts 读取模式。

推荐优先级：

1. `[qianfan_ocr].custom_prompt`
2. `[prompts].qianfan_ocr_prompt`
3. 空字符串

也就是说，provider 级配置优先，旧式 prompts 配置作为兜底。

### 3. Question 组装规则

provider 内新增一个统一 helper，例如：

- `_compose_question()`

规则如下：

1. 先读取 `base_prompt`
2. 再确定生效的自定义 prompt：
   - 优先 `[qianfan_ocr].custom_prompt`
   - 其次 `[prompts].qianfan_ocr_prompt`
3. 根据 `prompt_strategy` 组装：
   - `append`: `base_prompt + "\n" + custom_prompt`
   - `replace`: 直接使用自定义 prompt
4. 如果没有自定义 prompt，则直接使用 `base_prompt`
5. 如果 `think_enabled = true`，且最终 question 末尾还没有 `<think>`，则追加 `<think>`

这个 helper 是本次设计的核心边界。direct backend 和 server backend 必须复用同一个 helper，不能各自拼 query。

## 推理流程设计

### 1. 本地 direct backend

`qianfan_ocr` 的首发主路径是本地 `transformers` 推理。

建议实现方式：

- `AutoModel.from_pretrained(..., trust_remote_code=True, device_map="auto")`
- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- 使用 provider 内的图像预处理 helper，把单图转换为 `pixel_values`
- 调用：

```python
model.chat(
    tokenizer,
    pixel_values=pixel_values,
    question=question,
    generation_config={"max_new_tokens": max_new_tokens},
)
```

### 2. OpenAI-compatible server backend

`qianfan_ocr` 同时支持复用现有本地 OpenAI-compatible server backend。

但它不能只“借用 OCR 基类默认 prompt”，而必须复用同一个 `_compose_question()`。

server backend 的职责：

- 使用与 direct backend 相同的最终 question
- 走现有 `build_vision_messages()` / runtime completion 流程
- 输出仍然落到 Markdown 文件，不新增特殊格式

因此它和 `dots_ocr` 不同，不需要自定义复杂 output contract；只需要保证 prompt 一致。

### 3. 图片输入

图片输入行为与现有 OCR provider 保持一致：

- 输出目录：`Path(uri).with_suffix("")`
- 最终文本写入 `result.md`
- `CaptionResult.raw` 为 OCR 返回的 markdown 字符串

### 4. PDF 输入

PDF 复用现有高质量分页逻辑：

- `pdf_to_images_high_quality()`

每页行为：

- 保存 `page_xxxx/page_xxxx.png`
- 单页 OCR 结果写 `page_xxxx/result.md`

根目录行为：

- 汇总成功页面内容
- 写出根目录 `result.md`

汇总格式沿用现有 OCR provider：

```text
page1-content
<--- Page Split --->
page2-content
```

页面级容错策略：

- 某页转图失败或保存失败时跳过该页
- 继续处理后续页面
- 根目录结果仅包含成功页

## 输出设计

`qianfan_ocr` v1 只输出 Markdown 文本：

- 单图：`result.md`
- PDF：`page_xxxx/result.md` + 根目录 `result.md`

不引入：

- `result.svg`
- `result.html`
- `result.json`

原因：

- 这次目标是尽快把 `Qianfan-OCR` 接入现有 OCR 主路径
- 当前仓库对 OCR 下游的消费默认是 markdown 文本
- 与 `dots_ocr` 不同，本次没有额外的强约束要求引入特殊输出契约

## 错误处理

### 1. 缺少依赖

如果缺少 `transformers`、模型类不可导入或 `trust_remote_code` 路径无法加载：

- 直接报清晰错误
- 错误信息明确提示安装：

```bash
uv sync --extra qianfan-ocr
```

### 2. 非法 `prompt_strategy`

如果 `prompt_strategy` 不是：

- `append`
- `replace`

则直接抛错，不静默回退。

### 3. 页面级失败

PDF 某页处理失败：

- 跳过该页
- 继续后续页面
- 最终结果只聚合成功页

### 4. server backend 配置错误

如果用户选择走 OpenAI-compatible server backend，但未配置可用服务：

- 继续沿用现有 runtime backend 的报错
- `qianfan_ocr` 本身不单独吞错或做静默降级

## 依赖设计

### 1. `pyproject.toml`

新增 optional extra：

- `qianfan-ocr`

依赖策略按官方 quick start 的最小可运行集合，而不是复制 `dots_ocr` 的重依赖栈。建议至少包括：

- `torch`
- `torchvision`
- `transformers`
- `huggingface_hub`
- `safetensors`
- `Pillow`
- `PyMuPDF`
- `img2pdf`
- `accelerate`

如果在实现验证时发现必须依赖特定版本，再以测试结果为准补精确 pin。

### 2. `4、run.ps1`

新增：

- `ocr_model` 注释选项包含 `qianfan_ocr`
- `qianfan_ocr` -> `Add-UvExtra "qianfan-ocr"`

### 3. GUI

`gui/wizard/step4_caption.py` 需要：

- OCR 模型列表暴露 `qianfan_ocr`
- local extra map 增加 `qianfan_ocr -> qianfan-ocr`

## 测试设计

至少覆盖以下测试：

### 1. Catalog / registry / route

- `tests/test_provider_catalog.py`
  - `route_choices("ocr_model")` 包含 `qianfan_ocr`
- `tests/test_provider_registry.py`
  - registry 列表包含 `qianfan_ocr`
  - priority order 包含 `qianfan_ocr`
- `tests/test_provider_routes.py`
  - `ocr_model="qianfan_ocr"` 时图片和 PDF 路由正确

### 2. Provider 行为测试

新增：

- `tests/test_qianfan_ocr_provider.py`

至少覆盖：

- `think_enabled = true` 时最终 question 追加 `<think>`
- `think_enabled = false` 时不追加 `<think>`
- `prompt_strategy = "append"` 时正确保留基础 prompt 并追加自定义 prompt
- `prompt_strategy = "replace"` 时仅使用自定义 prompt
- 当 `[qianfan_ocr].custom_prompt` 为空时，`[prompts].qianfan_ocr_prompt` 作为兜底
- direct backend 与 server backend 使用相同的最终 question
- 单图写 `result.md`
- PDF 写 `page_xxxx/result.md` 和根目录 `result.md`

### 3. 依赖与脚本测试

- `tests/test_penguin_dependencies.py`
  - `pyproject.toml` 声明 `qianfan-ocr` extra
  - GUI 的 `_build_local_extra_args()` 返回 `--extra qianfan-ocr`
  - `4、run.ps1` 包含 `qianfan_ocr` 和 `Add-UvExtra "qianfan-ocr"`

## 风险

### 1. thinking 默认开启的稳定性风险

模型页建议复杂文档开启 `<think>`，但简单文档关闭可能更快且更稳。本次选择默认开启，会提高复杂版面鲁棒性，但可能在简单文档上增加延迟或生成额外推理文本。

缓解方式：

- `think_enabled` 做显式配置项
- provider 内只负责 query 拼装，不把 `<think>` 行为散落到多处

### 2. direct / server prompt 漂移风险

如果 direct backend 和 server backend 各自拼 question，后续很容易出现：

- 一个路径追加 `<think>`
- 另一个路径没追加
- 一个路径用 `append`
- 另一个路径误用了 `replace`

缓解方式：

- 所有 prompt 解析都集中到 `_compose_question()`

### 3. 依赖版本风险

`Qianfan-OCR` 使用 `trust_remote_code`，且模型发布较新。实际运行时可能对 `transformers`、`torchvision` 或远程代码依赖有更严格的版本要求。

缓解方式：

- 先以最小 extra 落地
- 用 focused tests 和一次真实导入验证来收敛最终版本约束

## 实施顺序建议

1. 在 catalog、registry、GUI、`run.ps1` 中接入 `qianfan_ocr` 路由名
2. 在 `pyproject.toml` 中增加 `qianfan-ocr` extra
3. 新增 `config/model.toml`、`config/config.toml`、`config/prompts.toml` 的 `qianfan_ocr` 配置
4. 先写 `tests/test_qianfan_ocr_provider.py`，固定：
   - `append` / `replace`
   - `think_enabled`
   - direct / server 共用 question 逻辑
   - 图片 / PDF 输出约定
5. 实现 `module/providers/ocr/qianfan.py`
6. 补充依赖、脚本和 GUI 测试
7. 跑 focused tests 验证路由、配置和依赖映射

## 结论

本设计采用单 provider 方案，把 `baidu/Qianfan-OCR` 作为新的 `qianfan_ocr` OCR provider 接入现有 Provider V2 架构。与 `dots_ocr` 不同，本次不引入新的输出格式或任务路由，而是把变化收敛在 query 组装规则上：默认基础 prompt 为 `Parse this document to Markdown.`，默认开启 `<think>`，并通过 `append` / `replace` 策略支持用户自定义 prompt。direct backend 与 OpenAI-compatible server backend 统一复用同一套 `_compose_question()`，从而在保持现有 OCR 输出契约不变的前提下，引入 `Qianfan-OCR` 的核心能力。
