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

- 基础文档 OCR
- Web Parsing
- Scene Spotting
- Image to SVG

## 已确认决策

- 采用单 provider 方案，不拆 `dots_ocr_svg` 或其他子 provider
- provider 名称固定为 `dots_ocr`
- 主路由仍然走 `ocr_model`
- 默认任务模式是基础文档 OCR
- `web_parsing`、`scene_spotting`、`image_to_svg` 作为同一 provider 下的扩展任务
- 所有扩展任务通过 TOML 配置切换，不新增新的 CLI 主路由
- `trust_remote_code` 不做用户配置项，默认始终启用
- flash attention 不做用户配置项，默认优先启用，失败时自动回退
- `image_to_svg` 允许使用单独的 `svg_model_id`
- 文本类任务可复用 OpenAI-compatible server backend
- `image_to_svg` 首发仅支持 direct backend，不支持 server backend

## 目标

1. 在现有 OCR Provider V2 架构中新增 `dots_ocr` 路由。
2. 接入 `davanstrien/dots.ocr-1.5` 作为默认权重。
3. 通过 TOML 配置支持 `document_ocr`、`web_parsing`、`scene_spotting`、`image_to_svg` 四种任务模式。
4. 保持当前图片 / PDF 输入、输出目录、结果写盘和 GUI / 脚本集成方式一致。
5. 通过测试覆盖 provider 路由、配置优先级、模型切换和 backend 约束。

## 非目标

- 本次不拆出第二个 `dots_ocr_svg` provider
- 本次不重构 `OCRProvider` 基类为“多任务 OCR 基类”
- 本次不新增单独的 CLI 参数来切换 `dots_ocr` 的任务模式
- 本次不承诺 OpenAI-compatible server backend 支持 SVG 生成
- 本次不为 `dots_ocr` 增加 GUI 专属配置页面

## 模型约束

`dots.ocr-1.5` 是一个 prompt-driven 的多任务模型族，至少覆盖：

- 文档 OCR / parsing
- 网页截图解析
- 场景内容识别
- 图像转 SVG

设计上需要显式处理以下约束：

- 模型依赖 `trust_remote_code=True`
- 任务模式本质上由 prompt 和模型选择共同决定
- `image_to_svg` 适合保留单独的模型权重入口
- 文本类输出和 SVG 输出的文件格式不同，不能继续完全沿用统一的 `result.md`
- OpenAI-compatible server backend 对文本类任务可行，但 SVG 输出格式不稳定，不适合作为首发路径

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
task_mode = "document_ocr"
max_new_tokens = 4096
prompt = ""
```

字段说明：

- `model_id`: 默认文本类任务模型
- `svg_model_id`: `image_to_svg` 专用模型
- `task_mode`: provider 任务模式
- `max_new_tokens`: 最大生成长度
- `prompt`: provider 级 prompt 覆盖

允许的 `task_mode` 值：

- `document_ocr`
- `web_parsing`
- `scene_spotting`
- `image_to_svg`

如果 `task_mode` 不在约定值内，provider 必须直接报错，不静默回退。

### 2. Prompt 配置

在 `config/prompts.toml` 和 `config/config.toml` 的 `[prompts]` 下新增：

- `dots_ocr_prompt`

Prompt 优先级：

1. `[dots_ocr].prompt`
2. `[prompts].dots_ocr_prompt`
3. provider 内置的任务默认 prompt

这保证：

- 用户可以针对 `dots_ocr` 做 provider 级覆盖
- 保持与现有 prompt 配置习惯一致
- 任务模式切换时仍然有可预测的默认行为

### 3. 任务模式与模型映射

默认模型选择规则：

- `document_ocr` -> `[dots_ocr].model_id`
- `web_parsing` -> `[dots_ocr].model_id`
- `scene_spotting` -> `[dots_ocr].model_id`
- `image_to_svg` -> `[dots_ocr].svg_model_id`，未配置时使用内置默认值

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

根据任务模式：

- 文本类任务输出字符串
- `image_to_svg` 输出 SVG 字符串

### 3. PDF 输入

PDF 复用现有高质量分页逻辑：

- `pdf_to_images_high_quality()`

每一页转成 PNG 后逐页调用同一任务模式推理。

行为：

- 文本类任务：每页写 `page_xxxx/result.md`
- `image_to_svg`：每页写 `page_xxxx/result.svg`

PDF 某页转换或保存失败时：

- 跳过该页
- 继续处理后续页面

### 4. 输出写盘

文本类任务：

- 单图输出 `result.md`
- PDF 按页输出 `page_xxxx/result.md`
- 根目录可额外写合并后的 `result.md`

SVG 任务：

- 单图输出 `result.svg`
- PDF 按页输出 `page_xxxx/result.svg`
- 根目录写一个简短索引文本，列出已生成的 SVG 页面路径

原因：

- 不能把 SVG 内容继续伪装成 markdown
- 又要保持现有 `CaptionResult.raw` / output_dir 约定不被彻底打破

### 5. Provider 返回值

Provider 统一返回 `CaptionResult`：

- 文本类任务：`raw` 为生成文本
- 单图 SVG：`raw` 为 SVG 字符串
- PDF + SVG：`raw` 为汇总索引文本，而不是把多页 SVG 直接拼成一个无效大字符串

## Backend 设计

### 1. Direct backend

`dots_ocr` 的首发主路径是 direct backend。

覆盖任务：

- `document_ocr`
- `web_parsing`
- `scene_spotting`
- `image_to_svg`

### 2. OpenAI-compatible server backend

文本类任务允许复用 `OCRProvider.attempt_via_openai_backend()` 的能力：

- `document_ocr`
- `web_parsing`
- `scene_spotting`

`image_to_svg` 不支持 server backend。

如果用户在 `image_to_svg` 场景强制走 server backend，provider 必须直接报错，例如：

- `dots_ocr image_to_svg only supports direct runtime backend`

这样可以避免：

- server 返回普通自然语言而不是 SVG
- 写盘格式和用户预期不一致
- 产生“半支持”的不稳定行为

## Prompt 设计

Provider 内部维护 `task_mode -> default_prompt` 映射。

建议逻辑：

- `document_ocr`: 文档 OCR / markdown 解析
- `web_parsing`: 网页截图解析
- `scene_spotting`: 场景或界面内容识别
- `image_to_svg`: 图像转 SVG

设计约束：

- prompt 默认值应与任务模式强绑定
- 不能把四个任务都退化成同一个通用 prompt
- prompt 覆盖逻辑不能影响 `task_mode -> model_id` 的选择

## 错误处理

### 1. 非法任务模式

如果 `task_mode` 不是约定值：

- 直接抛错
- 错误信息包含非法值和允许值

### 2. 缺少依赖

如果缺少 `transformers` 或模型类不可导入：

- 直接报清晰错误
- 错误文案明确提示安装 `dots-ocr` extra

### 3. SVG backend 冲突

如果 `task_mode="image_to_svg"` 且命中了 OpenAI-compatible server backend：

- 直接报错
- 不做降级

### 4. 页面级失败

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

- `task_mode` 到默认 prompt 的映射
- `image_to_svg` 使用 `svg_model_id`
- prompt 优先级按 `[dots_ocr].prompt > [prompts].dots_ocr_prompt > 内置默认`
- `image_to_svg` + server backend 直接报错
- 非法 `task_mode` 直接报错

### 3. 依赖与脚本测试

- `tests/test_penguin_dependencies.py`
  - `pyproject.toml` 声明 `dots-ocr` extra
  - GUI 的 `_build_local_extra_args()` 返回 `--extra dots-ocr`
  - `4、run.ps1` 包含 `dots_ocr` 和 `Add-UvExtra "dots-ocr"`

### 4. 输出文件测试

可选增加 provider 单测，验证：

- 文本任务写 `result.md`
- SVG 任务写 `result.svg`
- PDF + SVG 会创建 `page_xxxx/result.svg`

## 风险

### 1. 任务 prompt 漂移

`dots_ocr` 的多个模式本质上依赖 prompt 触发。如果默认 prompt 设计太弱，不同模式可能退化成近似输出。

### 2. SVG 输出稳定性

`image_to_svg` 属于结构化生成，输出可能出现：

- 非法 SVG
- 混入说明文字
- server backend 返回自由文本

因此 SVG 路径必须限制在 direct backend。

### 3. 依赖版本风险

`dots-ocr` 可能与仓库中其他 local OCR extra 存在依赖张力，尤其是 `transformers` 版本。如果真实安装矩阵冲突，再通过测试和冲突声明补齐。

## 实施顺序建议

1. 在 catalog、registry、GUI、`run.ps1` 中接入 `dots_ocr` 路由名
2. 在 `pyproject.toml` 中增加 `dots-ocr` extra
3. 新增 `config/model.toml`、`config/config.toml`、`config/prompts.toml` 的 `dots_ocr` 配置
4. 先写 provider 单测，固定任务模式、prompt 优先级和 backend 限制
5. 实现 `module/providers/ocr/dots.py`
6. 补充依赖、脚本和 GUI 测试
7. 跑最小测试集验证路由、配置和依赖映射

## 结论

本设计采用单 provider 方案，把 `dots.ocr-1.5` 作为新的 `dots_ocr` OCR provider 接入现有 Provider V2 架构。基础文档 OCR 是默认行为，`web_parsing`、`scene_spotting`、`image_to_svg` 作为 TOML 可配置的扩展任务。文本类任务允许 direct / OpenAI-compatible 双路径，SVG 首发仅支持 direct backend。整个接入以最小破坏现有 OCR 路由、输出结构和本地依赖链路为原则。
