# dots.ocr-1.5 OCR Provider 设计

## 背景

当前仓库的 Provider V2 已经具备较完整的 OCR 路由体系，现有本地 OCR provider 包括：

- `deepseek_ocr`
- `lighton_ocr`
- `hunyuan_ocr`
- `glm_ocr`
- `chandra_ocr`
- `olmocr`
- `paddle_ocr`
- `nanonets_ocr`
- `firered_ocr`

这些 provider 已经打通了以下共性能力：

- `ocr_model` 路由选择
- 图片 / PDF 的 OCR 处理
- `output_dir` 输出目录约定
- 本地 `transformers` 直连推理
- 可选 OpenAI-compatible server backend
- GUI 与 `4、run.ps1` 的 local extra 安装链路

本次设计的目标是在不破坏现有 OCR 路由结构的前提下，把 Hugging Face 上的 `davanstrien/dots.ocr-1.5` 接入为一个新的 OCR provider，并支持其扩展任务模式：

- 官方 demo 暴露的 `prompt_mode`
- 重点覆盖文档解析、Web Parsing、Scene Spotting、Image to SVG

## 已确认决策

- 采用单 provider 方案，不拆 `dots_ocr_svg` 或其他子 provider
- provider 名称固定为 `dots_ocr`
- 主路由仍然走 `ocr_model`
- 不再自造 `task_mode`，统一使用官方 demo 的 `prompt_mode`
- 默认 `prompt_mode` 使用官方文档解析模式
- 所有扩展任务通过 TOML 配置切换，不新增新的 CLI 主路由
- `trust_remote_code` 不做用户配置项，默认始终启用
- flash attention 不做用户配置项，默认优先启用，失败时自动回退
- `image_to_svg` 允许使用单独的 `svg_model_id`
- direct / OpenAI-compatible server backend 共享同一套 `prompt_mode -> prompt` 解析规则
- `prompt_mode` 决定任务语义、模型选择、输出类型和 backend 资格
- 自定义 `prompt` 只覆盖最终提示词文本，不改变任务类型

## 目标

1. 在现有 OCR Provider V2 架构中新增 `dots_ocr` 路由。
2. 接入 `davanstrien/dots.ocr-1.5` 作为默认权重。
3. 通过 TOML 配置支持官方 demo 的 `prompt_mode`。
4. 保持当前图片 / PDF 输入、输出目录、结果写盘和 GUI / 脚本集成方式一致。
5. 通过测试覆盖 provider 路由、配置优先级、模型切换和 backend 约束。

## 非目标

- 本次不拆出第二个 `dots_ocr_svg` provider
- 本次不重构 `OCRProvider` 基类为“多任务 OCR 基类”
- 本次不新增单独的 CLI 参数来切换 `dots_ocr` 的任务模式以外的新路由
- 本次不把 “PDF -> Markdown + 图替换成 SVG” 作为正式行为写死
- 本次不为 `dots_ocr` 增加 GUI 专属配置页面

## 模型约束

`dots.ocr-1.5` 是一个 prompt-driven 的多任务模型族，至少覆盖：

- 文档 OCR / parsing
- 网页截图解析
- 场景内容识别
- 图像转 SVG

设计上需要显式处理以下约束：

- 模型依赖 `trust_remote_code=True`
- 任务语义由官方 `prompt_mode` 和模型选择共同决定
- `image_to_svg` 适合保留单独的模型权重入口
- 文本类输出和 SVG 输出的文件格式不同，不能继续完全沿用统一的 `result.md`
- 上游已经提供 HF 和 vLLM 两种 demo 路径，设计不应再发明第三套任务命名

## 架构设计

### 1. 新增 `dots_ocr` provider

新增文件：

- `module/providers/ocr/dots.py`

该 provider 与现有 `lighton_ocr`、`deepseek_ocr` 同级，继承 `OCRProvider`。

Provider 名称与 route value 统一使用：

- canonical provider: `dots_ocr`
- route value: `dots_ocr`

### 2. 路由与 catalog 集成

以下位置需要增加 `dots_ocr`：

- `module/providers/catalog.py`
- `module/providers/registry.py`
- `tests/test_provider_catalog.py`
- `tests/test_provider_registry.py`
- `tests/test_provider_routes.py`

要求：

- `route_choices("ocr_model")` 暴露 `dots_ocr`
- `ProviderRegistry._priority_order` 包含 `dots_ocr`
- `find_provider()` 在 `ocr_model="dots_ocr"` 时能正确命中

### 3. GUI 与脚本集成

以下位置需要增加 `dots_ocr` 的本地依赖映射：

- `gui/wizard/step4_caption.py`
- `4、run.ps1`
- `tests/test_penguin_dependencies.py`

要求：

- GUI 的 OCR 模型列表包含 `dots_ocr`
- GUI 的 local extra 映射可返回 `--extra dots-ocr`
- `4、run.ps1` 支持 `ocr_model="dots_ocr"` 并追加对应 extra

## 配置设计

### 1. Provider 配置

在 `config/model.toml` 和 `config/config.toml` 新增：

```toml
[dots_ocr]
model_id = "davanstrien/dots.ocr-1.5"
svg_model_id = "davanstrien/dots.ocr-1.5-svg"
prompt_mode = "prompt_layout_all_en"
max_new_tokens = 4096
prompt = ""
```

字段说明：

- `model_id`: 默认文本类任务模型
- `svg_model_id`: `image_to_svg` 专用模型
- `prompt_mode`: 官方 demo / 官方 `dict_promptmode_to_prompt` 中的模式名
- `max_new_tokens`: 最大生成长度
- `prompt`: provider 级 prompt 覆盖

允许值不维护本地硬编码副本，而是以官方 `dict_promptmode_to_prompt` 的 key 集合作为单一事实来源。

v1 至少覆盖官方 demo 中实际演示的模式：

- `prompt_layout_all_en`
- `prompt_web_parsing`
- `prompt_scene_spotting`
- `prompt_image_to_svg`

如果 `prompt_mode` 不在官方映射中，provider 必须直接报错，不静默回退。

### 2. Prompt 配置

在 `config/prompts.toml` 和 `config/config.toml` 的 `[prompts]` 下新增：

- `dots_ocr_prompt`

Prompt 优先级：

1. 由 `prompt_mode` 从官方 `dict_promptmode_to_prompt` 解析出默认 prompt
2. 如果 `[dots_ocr].prompt` 非空，则覆盖默认 prompt
3. 否则如果 `[prompts].dots_ocr_prompt` 非空，则覆盖默认 prompt

这保证：

- 任务语义跟随官方 demo
- 用户仍可以针对 `dots_ocr` 做 provider 级覆盖
- direct / server 都能复用同一个 prompt 解析结果

### 3. `prompt_mode` 与模型映射

默认模型选择规则：

- 非 `prompt_image_to_svg` -> `[dots_ocr].model_id`
- `prompt_image_to_svg` -> `[dots_ocr].svg_model_id`，未配置时使用内置默认值

职责边界：

- `prompt_mode` 决定任务语义、默认 prompt、模型选择、输出类型和 backend 路径
- `prompt` 只覆盖最终发送给模型的文本，不改变任务类型

## 推理流程设计

### 1. 模型加载

`dots_ocr` 走本地 `transformers` 推理路径，和其他本地 OCR provider 保持一致。

固定行为：

- `trust_remote_code=True`
- 自动选择设备与 dtype
- 优先启用 flash attention
- flash attention 不可用时自动回退到仓库现有注意力实现策略

### 2. 图片输入

图片文件直接送入模型推理。

根据 `prompt_mode`：

- 文本类任务输出字符串
- `prompt_image_to_svg` 输出 SVG 字符串

### 3. PDF 输入

PDF 复用现有高质量分页逻辑：

- `pdf_to_images_high_quality()`

每一页转成 PNG 后逐页调用同一任务模式推理。

行为：

- 文本类任务：每页写 `page_xxxx/result.md`

PDF 某页转换或保存失败时：

- 跳过该页
- 继续处理后续页面

`prompt_image_to_svg` 对 PDF 采用明确的混合输出契约：

- 每页生成 `page_xxxx/result.svg`
- 根目录生成聚合 `result.md`
- `result.md` 不嵌入原始 SVG 字符串，而是按页使用相对路径引用对应的 `result.svg`
- `result.md` 作为 PDF + SVG 模式的统一消费入口

这样做的原因是：

- 每页 SVG 仍然保持独立文件，便于后处理和人工检查
- 根目录仍然保留一个稳定的文档入口，兼容现有 OCR “结果目录 + 主结果文件”的使用方式
- 不把多页 SVG 直接拼成一个无效的大 SVG 文件

### 4. 输出写盘

文本类任务：

- 单图输出 `result.md`
- PDF 按页输出 `page_xxxx/result.md`
- 根目录写合并后的 `result.md`

SVG 任务：

- 单图输出 `result.svg`
- PDF 按页输出 `page_xxxx/result.svg`
- PDF 根目录额外输出聚合 `result.md`，按页引用对应 SVG

原因：

- 不能把 SVG 内容继续伪装成 markdown
- 但 PDF 场景仍然需要一个稳定的主入口文件来串起多页结果

### 5. Provider 返回值

Provider 统一返回 `CaptionResult`：

- 文本类任务：`raw` 为生成文本
- 单图 SVG：`raw` 为 SVG 字符串
- PDF + `prompt_image_to_svg`：`raw` 为聚合 `result.md` 的文本内容

这样 `raw` 在 PDF + SVG 场景下表达的是“主入口文档”，而不是多页 SVG 的索引或拼接产物。

## Backend 设计

### 1. Direct backend

`dots_ocr` 的首发主路径是 direct backend。

覆盖任务：

- 所有已支持的官方 `prompt_mode`
- v1 重点验证 `prompt_layout_all_en`、`prompt_web_parsing`、`prompt_scene_spotting`、`prompt_image_to_svg`

### 2. OpenAI-compatible server backend

server backend 参考官方 vLLM demo 路径，要求与 direct backend 共用同一套 `prompt_mode -> prompt` 解析规则。

也就是说：

- 先解析 `prompt_mode`
- 再决定模型选择和输出类型
- 最后由 direct / server 走各自推理实现

不能出现：

- direct 走官方 prompt
- server 走另一套自造 prompt

对于 `prompt_image_to_svg`：

- 如果 server backend 指向兼容的 served model，则允许使用
- provider 不能直接复用通用 `OCRProvider.attempt_via_openai_backend()` 后就把结果一律写成 markdown
- `dots_ocr` 需要在必要时 override server 路径，以便根据 `prompt_mode` 正确写出单图 `result.svg` 或 PDF 场景下的 `result.md + page_xxxx/result.svg`

## Prompt 设计

Provider 不维护本地手写 prompt 文案副本，而是复用官方 `dict_promptmode_to_prompt` 作为默认 prompt 来源。

建议逻辑：

- `prompt_layout_all_en`: 文档解析
- `prompt_web_parsing`: 网页截图解析
- `prompt_scene_spotting`: 场景或界面内容识别
- `prompt_image_to_svg`: 图像转 SVG

设计约束：

- prompt 默认值应与 `prompt_mode` 强绑定
- 不再复制一份上游 prompt 文案到本仓库作为“第二真相源”
- `prompt` 覆盖逻辑不能影响 `prompt_mode -> model_id` 的选择

## 错误处理

### 1. 非法 `prompt_mode`

如果 `prompt_mode` 不在官方映射中：

- 直接抛错
- 错误信息包含非法值和允许值

### 2. 缺少依赖

如果缺少 `transformers` 或模型类不可导入：

- 直接报清晰错误
- 错误文案明确提示安装 `dots-ocr` extra

### 3. 页面级失败

PDF 某页处理失败：

- 跳过该页
- 继续后续页面
- 最终结果只包含成功页

## 依赖设计

### 1. `pyproject.toml`

新增一个 optional extra：

- `dots-ocr`

依赖策略应尽量贴近当前本地 OCR provider 的公共栈，至少包括：

- `torch`
- `torchvision`
- `accelerate`
- `transformers[serving]`
- `huggingface_hub[hf_xet]`
- `safetensors`
- `Pillow`
- `PyMuPDF`
- `img2pdf`

是否声明额外冲突取决于真实版本矩阵，不做预设冲突。

### 2. `4、run.ps1`

新增：

- `ocr_model` 的注释选项包含 `dots_ocr`
- `dots_ocr` -> `Add-UvExtra "dots-ocr"`

### 3. GUI

`gui/wizard/step4_caption.py` 需要：

- 在 OCR 模型列表中暴露 `dots_ocr`
- 在 local extra 映射中加入 `dots_ocr -> dots-ocr`

## 测试设计

至少覆盖以下测试：

### 1. Catalog / registry / route

- `tests/test_provider_catalog.py`
  - `route_choices("ocr_model")` 包含 `dots_ocr`
- `tests/test_provider_registry.py`
  - registry 列表包含 `dots_ocr`
  - priority order 包含 `dots_ocr`
- `tests/test_provider_routes.py`
  - `ocr_model="dots_ocr"` 时图片和 PDF 路由正确

### 2. Provider 行为测试

新增 `dots_ocr` provider 单测，至少覆盖：

- `prompt_mode` 从官方映射正确解析 prompt
- `prompt_image_to_svg` 使用 `svg_model_id`
- prompt 优先级按 `官方映射默认值 -> [dots_ocr].prompt 覆盖 -> [prompts].dots_ocr_prompt 兜底覆盖`
- direct / server 共享同一套 prompt 解析逻辑
- 非法 `prompt_mode` 直接报错

### 3. 依赖与脚本测试

- `tests/test_penguin_dependencies.py`
  - `pyproject.toml` 声明 `dots-ocr` extra
  - GUI 的 `_build_local_extra_args()` 返回 `--extra dots-ocr`
  - `4、run.ps1` 包含 `dots_ocr` 和 `Add-UvExtra "dots-ocr"`

### 4. 输出文件测试

可选增加 provider 单测，验证：

- 文本任务写 `result.md`
- SVG 任务写 `result.svg`
- `prompt_image_to_svg` + PDF 会生成 `page_xxxx/result.svg`
- `prompt_image_to_svg` + PDF 的根目录会生成引用这些 SVG 的 `result.md`

## 风险

### 1. 任务 prompt 漂移

`dots_ocr` 的多个模式本质上依赖 prompt 触发。如果不直接复用上游 `dict_promptmode_to_prompt`，本地复制版很容易和官方 demo 漂移。

### 2. SVG 输出稳定性

`image_to_svg` 属于结构化生成，输出可能出现：

- 非法 SVG
- 混入说明文字
- 服务端部署模型和本地模型权重不一致

因此 `prompt_image_to_svg` 的模型选择和输出写盘规则必须由 `prompt_mode` 统一控制，不能让 backend 自己决定；尤其是 PDF 场景必须稳定落到 “根目录 `result.md` + 逐页 `result.svg`” 这一契约上。

### 3. 依赖版本风险

`dots-ocr` 可能与仓库中其他 local OCR extra 存在依赖张力，尤其是 `transformers` 版本。如果真实安装矩阵冲突，再通过测试和冲突声明补齐。

## 实施顺序建议

1. 在 catalog、registry、GUI、`run.ps1` 中接入 `dots_ocr` 路由名
2. 在 `pyproject.toml` 中增加 `dots-ocr` extra
3. 新增 `config/model.toml`、`config/config.toml`、`config/prompts.toml` 的 `dots_ocr` 配置，并把 `task_mode` 改为 `prompt_mode`
4. 先写 provider 单测，固定 `prompt_mode`、prompt 优先级和 output contract
5. 实现 `module/providers/ocr/dots.py`
6. 补充依赖、脚本和 GUI 测试
7. 跑最小测试集验证路由、配置和依赖映射

## 结论

本设计采用单 provider 方案，把 `dots.ocr-1.5` 作为新的 `dots_ocr` OCR provider 接入现有 Provider V2 架构。任务语义不再由本仓库发明，而是直接对齐官方 demo 的 `prompt_mode` 和官方 `dict_promptmode_to_prompt`。文本与 SVG 的模型选择、输出类型和 direct / server 行为都由 `prompt_mode` 统一驱动；其中 PDF + `prompt_image_to_svg` 明确采用 “根目录 `result.md` + 逐页 `page_xxxx/result.svg`” 的混合契约。整个接入以减少双真相源、并保持现有 OCR 路由兼容为原则。
