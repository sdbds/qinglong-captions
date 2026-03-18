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
- 用户自定义 prompt 只保留一个配置入口
- 支持两种 prompt 组合策略：
  - `append`
  - `replace`
- direct backend 是首发必做路径
- 保留 OpenAI-compatible server backend 扩展点，但不把 generic server parity 作为当前实现承诺
- v1 不新增专门的 KIE / JSON / HTML 子路由，先通过自定义 prompt 承载扩展任务
- 输出契约维持现有 OCR 习惯，不像 `dots_ocr` 那样引入 SVG 等特殊文件格式

## 目标

1. 在现有 OCR Provider V2 体系中新增 `qianfan_ocr` 路由。
2. 接入 `baidu/Qianfan-OCR` 作为默认本地模型。
3. 通过 TOML 配置支持：
   - thinking 开关
   - 单一自定义 prompt
   - `append` / `replace` prompt 策略
4. 保持图片 / PDF 输入、结果目录、Markdown 写盘和 GUI / 脚本集成方式与现有 OCR provider 一致。
5. 明确定义 thinking 输出的清洗契约，保证写入 `result.md` 的内容不包含 reasoning 段。

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
prompt = ""
prompt_strategy = "append"
think_enabled = true
max_new_tokens = 16384
```

字段职责：

- `model_id`: 默认模型权重
- `prompt`: 唯一的用户自定义 prompt 配置入口
- `prompt_strategy`: `append` 或 `replace`
- `think_enabled`: 是否在最终 question 尾部追加 `<think>`
- `max_new_tokens`: 推理上限

### 2. Question 组装规则

provider 内新增一个统一 helper，例如：

- `_compose_question()`

规则如下：

1. provider 内建基础 prompt 固定为 `Parse this document to Markdown.`
2. 读取 `[qianfan_ocr].prompt`
3. 根据 `prompt_strategy` 组装：
   - `append`: `base_prompt + "\n" + prompt`
   - `replace`: 直接使用 `prompt`
4. 如果 `prompt` 为空，则直接使用内建 `base_prompt`
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

### 2. thinking 输出清洗

默认开启 `<think>` 的前提是必须定义清楚输出清洗责任。

`qianfan_ocr` 必须在 provider 内拥有一个统一的清洗 helper，例如：

- `_clean_reasoning_output()`

职责：

- 去掉 `<think>...</think>`
- 去掉明显的 reasoning 包裹段
- 返回最终要展示、写盘和放入 `CaptionResult.raw` 的 markdown 文本

关键约束：

- 清洗必须发生在任何 `write_markdown_output(...)` 之前
- 如果后续补 server backend，direct backend 和 server backend 都必须复用同一个清洗 helper
- 如果清洗后结果为空，provider 必须抛出显式错误，而不是写出空的 `result.md`

### 3. OpenAI-compatible server backend

当前 spec 只把 server backend 作为扩展点，不把 generic OpenAI-compatible 协议兼容性写成正式承诺。

原因：

- 仓库当前 OCR server 路径默认走 generic OpenAI multimodal message shape
- 现阶段没有已验证的证据表明 Qianfan served model 与这个协议完全等价

因此当前设计要求是：

- direct backend 必须可用
- server backend 不是当前实现的阻塞项
- 只有在验证过具体 serving target 后，才把 server backend 升级为正式支持路径

如果后续要接 server backend，要求如下：

- 使用与 direct backend 相同的最终 question
- 不能直接原样复用 `OCRProvider.attempt_via_openai_backend()`
- 必须走 provider 自己的 server path，以便在写盘前完成 reasoning 清洗

### 4. 图片输入

图片输入行为与现有 OCR provider 保持一致：

- 输出目录：`Path(uri).with_suffix("")`
- 最终文本写入 `result.md`
- `CaptionResult.raw` 为 OCR 返回的 markdown 字符串

### 5. PDF 输入

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
- 默认开启 thinking 后，最终写盘内容必须是清洗后的 markdown，而不是原始 reasoning 输出

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

### 3. reasoning 清洗失败

如果模型返回了带有 reasoning 的内容，但清洗后结果为空或明显无效：

- 直接抛错
- 不写出污染后的 `result.md`

### 4. 页面级失败

PDF 某页处理失败：

- 跳过该页
- 继续后续页面
- 最终结果只聚合成功页

### 5. server backend 配置错误

如果用户选择走 OpenAI-compatible server backend，但未配置可用服务：

- 继续沿用现有 runtime backend 的报错
- `qianfan_ocr` 本身不单独吞错或做静默降级
- server backend 只有在协议验证完成后才进入正式支持范围

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
- `prompt_strategy = "append"` 时正确保留基础 prompt 并追加 `[qianfan_ocr].prompt`
- `prompt_strategy = "replace"` 时仅使用 `[qianfan_ocr].prompt`
- reasoning 清洗发生在写盘前，而不是写盘后
- `<think>...</think>` 不会出现在最终 `result.md`
- 单图写 `result.md`
- PDF 写 `page_xxxx/result.md` 和根目录 `result.md`
- 如果后续开启 server backend，再补它与 direct backend 共用 question / cleanup 逻辑的测试

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
- provider 内统一负责 query 拼装和 reasoning 清洗，不把 `<think>` 行为散落到多处

### 2. direct / server prompt 漂移风险

如果后续补 server backend，而 direct backend 和 server backend 各自拼 question / 各自清洗输出，后续很容易出现：

- 一个路径追加 `<think>`
- 另一个路径没追加
- 一个路径用 `append`
- 另一个路径误用了 `replace`
- 一个路径写盘前清洗
- 另一个路径把 reasoning 原样写入 `result.md`

缓解方式：

- 所有 prompt 解析都集中到 `_compose_question()`
- 所有 reasoning 清洗都集中到 `_clean_reasoning_output()`

### 3. 依赖版本风险

`Qianfan-OCR` 使用 `trust_remote_code`，且模型发布较新。实际运行时可能对 `transformers`、`torchvision` 或远程代码依赖有更严格的版本要求。

缓解方式：

- 先以最小 extra 落地
- 用 focused tests 和一次真实导入验证来收敛最终版本约束

## 实施顺序建议

1. 在 catalog、registry、GUI、`run.ps1` 中接入 `qianfan_ocr` 路由名
2. 在 `pyproject.toml` 中增加 `qianfan-ocr` extra
3. 新增 `config/model.toml`、`config/config.toml` 的 `qianfan_ocr` 配置
4. 先写 `tests/test_qianfan_ocr_provider.py`，固定：
   - `append` / `replace`
   - `think_enabled`
   - reasoning 清洗先于写盘
   - 图片 / PDF 输出约定
5. 实现 `module/providers/ocr/qianfan.py`
6. 补充依赖、脚本和 GUI 测试
7. 跑 focused tests 验证路由、配置和依赖映射
8. 如果后续需要 server backend，再单独补协议验证和 provider 专属 server path

## 结论

本设计采用单 provider 方案，把 `baidu/Qianfan-OCR` 作为新的 `qianfan_ocr` OCR provider 接入现有 Provider V2 架构。与 `dots_ocr` 不同，本次不引入新的输出格式或任务路由，而是把变化收敛在两个明确的 provider 责任上：一是统一的 `_compose_question()`，固定基础 prompt 并支持单一自定义 prompt 的 `append` / `replace` 组合；二是统一的 `_clean_reasoning_output()`，在默认开启 `<think>` 的前提下，保证写入 `result.md` 的仍然是干净的 markdown。direct backend 是当前必做路径；server backend 只保留扩展点，待具体协议验证后再升级为正式支持能力。
