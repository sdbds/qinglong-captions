# Infinity Parser2 OCR Provider 设计

## 背景

当前仓库的 Provider V2 已经有稳定的 OCR 接入方式，现有本地 OCR provider 包括：

- `chandra_ocr`
- `dots_ocr`
- `lighton_ocr`
- `olmocr`
- `paddle_ocr`
- `qianfan_ocr`
- `deepseek_ocr`
- `glm_ocr`
- `firered_ocr`
- `nanonets_ocr`
- `logics_ocr`
- `hunyuan_ocr`

这些 provider 已经共享以下行为：

- 通过 `ocr_model` 路由选择 provider
- 对图片和 PDF 做 OCR 处理
- 输出目录默认使用输入文件去后缀后的同名目录
- 输出 Markdown 到 `result.md`
- GUI 和 PowerShell 脚本通过 `uv --extra` 追加本地依赖
- Provider registry 通过显式模块清单发现 provider

本次需求是新增一个 Infinity Parser2 OCR provider。仓库当前没有已注册的 `Infinity-Parser` / `infly` / `infinity_parser` provider，也没有旧模型需要迁移。

因此本设计按“全新 provider 接入”处理，不做旧 provider 升级，不做旧配置自动迁移。

参考来源：

- Flash 模型页：`https://huggingface.co/infly/Infinity-Parser2-Flash`
- Pro 模型页：`https://huggingface.co/infly/Infinity-Parser2-Pro`
- 官方指南：`https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser2`

## 模型事实

Infinity Parser2 当前有两个本次关心的公开权重：

| 角色 | 模型 ID | 规模 | 官方定位 | olmOCR-bench | ParseBench |
| --- | --- | --- | --- | --- | --- |
| 默认模型 | `infly/Infinity-Parser2-Flash` | 2B BF16 | 低延迟、高吞吐 | 86.0 | 72.2 |
| 可选模型 | `infly/Infinity-Parser2-Pro` | 35B BF16 | 高精度、精度关键场景 | 87.6 | 74.3 |

设计结论：

- 默认模型必须使用 `Infinity-Parser2-Flash`。
- `Infinity-Parser2-Pro` 只作为 `model_list` 中的可选项。
- 用户只有显式把 `[infinity_parser2_ocr].model_id` 改成 Pro 时才加载 35B 模型。

官方 native Transformers 路径使用：

- `AutoModelForImageTextToText`
- `AutoProcessor`
- `qwen_vl_utils.process_vision_info`
- `processor.apply_chat_template(..., enable_thinking=False)`
- `min_pixels = 2048`
- `max_pixels = 16777216`
- `image_patch_size = 16`
- `max_new_tokens = 32768`
- `temperature = 0.0`
- `top_p = 1.0`

官方 `infinity_parser2` wrapper 支持三种 backend：

- `vllm-engine`
- `transformers`
- `vllm-server`

官方限制：

- 主要支持英文和中文文档
- 多语言内容性能下降
- 复杂图表和多角度旋转元素准确率下降
- 不保留 bold / italic / strikethrough 等细粒度文本样式
- 复杂多步视觉指令跟随能力一般

## 已确认决策

- 新增 provider，而不是改写现有 provider。
- canonical provider 使用 `infinity_parser2_ocr`。
- `ocr_model` 主路由值使用 `infinity_parser2_ocr`。
- 默认模型使用 `infly/Infinity-Parser2-Flash`。
- `infly/Infinity-Parser2-Pro` 只作为可选模型出现在配置的 `model_list` 中。
- 首发实现只承诺本地 direct Transformers 路径。
- `infinity_parser2` wrapper 和 vLLM server 只作为后续扩展点，不作为首发阻塞项。
- 首发默认任务是 `doc2md`，输出继续遵守现有 OCR provider 的 Markdown 契约。
- `enable_thinking=False` 是硬约束，不作为普通用户配置项暴露。
- 不自动下载任何 Infinity Parser2 模型；只在用户显式选择该 OCR provider 时加载。
- 本 provider 是可选 OCR extra，不进入默认依赖集。

## 命名决策

本次不使用 `infinity_parser_ocr`。

原因：

- 仓库没有旧 Infinity provider，没必要为不存在的旧路由保留家族名。
- 新 provider 明确接入的是 Infinity Parser2 模型族。
- `infinity_parser2_ocr` 能同时承载 Flash 和 Pro 两个版本，版本边界清楚。

本次不设计 `infinity_parser2_pro` route alias。

原因：

- 当前 `RouteSpec.aliases` 只能把 route value canonicalize 到 provider，不能同时切换 `model_id`。
- 如果 `ocr_model="infinity_parser2_pro"` 仍然加载默认 Flash，会制造配置幻觉。
- Pro 应通过 `[infinity_parser2_ocr].model_id` 或 GUI 模型配置面板选择。

## 目标

1. 在 Provider V2 中新增 `infinity_parser2_ocr` OCR provider。
2. 默认接入 `infly/Infinity-Parser2-Flash`。
3. 在配置中暴露 `infly/Infinity-Parser2-Pro` 作为可选模型。
4. 支持图片和 PDF 输入。
5. 对 PDF 复用现有高质量分页与逐页输出约定。
6. 输出继续写入 `result.md`，`CaptionResult.raw` 返回最终 Markdown。
7. 将模型依赖放入独立 optional extra。
8. 在 GUI、脚本、catalog、registry、tests 中完整暴露该 provider。

## 非目标

- 本次不做旧 Infinity provider 迁移，因为仓库没有旧 provider。
- 本次不把 `Infinity-Parser2-Pro` 设为默认模型。
- 本次不把 `infinity_parser2` wrapper 作为首发主路径。
- 本次不承诺 vLLM engine / vLLM server parity。
- 本次不新增独立 GUI 配置页。
- 本次不新增 OCR 通用 backend 抽象层。
- 本次不把 layout JSON 作为默认输出格式。
- 本次不改造 `OCRProvider` 基类。
- 本次不处理 `third_party/dots.ocr` benchmark 中旧 `Infinity-Parser 7B` 文本。
- 本次不承诺样式级还原、多语言高准确率、复杂旋转表格稳定解析。

## 架构设计

### 1. 新增 provider 文件

新增文件：

- `module/providers/ocr/infinity_parser2.py`

类名：

- `InfinityParser2OCRProvider`

注册名：

```python
@register_provider("infinity_parser2_ocr")
```

类属性：

```python
default_model_id = "infly/Infinity-Parser2-Flash"
default_prompt = "Please transform the document's contents into Markdown format."
```

### 2. 路由与 catalog 集成

`module/providers/catalog.py` 增加：

```python
"infinity_parser2_ocr": ProviderSpec(
    canonical_name="infinity_parser2_ocr",
    config_sections=("infinity_parser2_ocr",),
    prompt_prefixes=("infinity_parser2",),
)
```

`ROUTE_SPECS["ocr_model"]` 增加：

```python
RouteSpec("infinity_parser2_ocr", "infinity_parser2_ocr")
```

排序建议：

- 在 OCR route choice 中排到 `chandra_ocr` 之前。
- 理由：Flash 官方 `olmOCR-bench = 86.0` 高于当前 catalog 注释中的 Chandra 85.9，同时 Flash 是低延迟默认模型。

### 3. Registry 集成

`module/providers/registry.py` 增加模块清单：

```python
"infinity_parser2_ocr": "module.providers.ocr.infinity_parser2",
```

`_priority_order` 增加：

```python
"infinity_parser2_ocr",
```

### 4. GUI 与脚本集成

以下位置需要增加 `infinity_parser2_ocr`：

- `gui/wizard/step4_caption.py`
- `4.captioner.ps1`
- `4、run.ps1`

要求：

- GUI OCR 下拉框暴露 `infinity_parser2_ocr`
- GUI local extra 映射返回 `--extra infinity-parser2-ocr`
- PowerShell 脚本支持 `ocr_model="infinity_parser2_ocr"` 并追加 `Add-UvExtra "infinity-parser2-ocr"`
- Pro 不做单独 route；通过模型配置选择 Pro

## 配置设计

在 `config/model.toml` 和 `config/config.toml` 新增：

```toml
# Infinity Parser2 OCR configuration
# Flash: https://huggingface.co/infly/Infinity-Parser2-Flash
# Pro: https://huggingface.co/infly/Infinity-Parser2-Pro
[infinity_parser2_ocr]
model_list."Infinity Parser2 Flash" = { model_id = "infly/Infinity-Parser2-Flash", meta = { min_vram_gb = 8 } }
model_list."Infinity Parser2 Pro" = { model_id = "infly/Infinity-Parser2-Pro", meta = { min_vram_gb = 80 } }
model_id = "infly/Infinity-Parser2-Flash"
task_type = "doc2md"
prompt = ""
max_new_tokens = 32768
temperature = 0.0
top_p = 1.0
min_pixels = 2048
max_pixels = 16777216
image_patch_size = 16
```

字段说明：

- `model_list`: GUI 模型选择来源，Flash 是默认项，Pro 是可选项。
- `model_id`: 实际加载的 Hugging Face 权重，默认 Flash。
- `task_type`: 首发仅支持 `doc2md` 和 `custom`；`doc2json` 留作后续。
- `prompt`: provider 级 prompt 覆盖；为空时使用内置 Markdown prompt。
- `max_new_tokens`: 生成上限，默认沿用官方 native Transformers 示例。
- `temperature`: 默认 0，确保 OCR 输出确定性。
- `top_p`: 默认 1.0。
- `min_pixels`: 官方默认最小像素数。
- `max_pixels`: 官方默认最大像素数。
- `image_patch_size`: 官方 `process_vision_info(..., image_patch_size=16)` 默认值。
- `meta.min_vram_gb`: 只作为 GUI fit hint；真实显存需求取决于 GPU、device map、量化和 offload，不能当硬性保证。

在 `config/prompts.toml` 的 `[prompts]` 下新增：

```toml
infinity_parser2_ocr_prompt = """
"""
```

Prompt 优先级：

1. `[infinity_parser2_ocr].prompt`
2. `[prompts].infinity_parser2_ocr_prompt`
3. provider 内置 `default_prompt`

`task_type="custom"` 时：

- 必须存在非空 prompt。
- provider 直接发送该 prompt。
- 输出仍按 Markdown 文本写入，除非后续明确新增 JSON 输出契约。

## 推理流程设计

### 1. 本地 direct Transformers 路径

首发实现使用官方 native Transformers 形态，而不是 wrapper。

理由：

- 仓库已有多个 OCR provider 使用 direct Transformers 模式，集成成本低。
- wrapper 默认牵涉 vLLM engine、server 依赖和 Python 3.12+ 要求。
- 仓库当前 extras 需要可组合，不应让一个 OCR provider 污染默认安装或低版本 Python 环境。

核心流程：

1. 读取 `model_id`、`prompt`、`max_new_tokens`、`temperature`、`top_p`、`min_pixels`、`max_pixels`、`image_patch_size`。
2. 加载：
   - `AutoModelForImageTextToText`
   - `AutoProcessor`
3. 构造官方消息格式：

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": pil_image,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            {"type": "text", "text": prompt},
        ],
    }
]
```

4. 调用：

```python
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
image_inputs, _ = process_vision_info(messages, image_patch_size=image_patch_size)
inputs = processor(
    text=text,
    images=image_inputs,
    do_resize=False,
    padding=True,
    return_tensors="pt",
)
```

5. 将 tensor 移动到模型 device。
6. 调用 `model.generate(...)`。
7. 截掉 input token，只 decode 新生成 token。
8. 清理首尾空白和常见 Markdown fence 包裹。
9. 如果清理后为空，抛出显式错误，不写空 `result.md`。

### 2. Thinking 约束

`enable_thinking=False` 必须硬编码在 provider 内。

原因：

- OCR provider 的输出契约是最终 OCR 内容，不是 reasoning trace。
- 后处理 reasoning 比禁用 thinking 更脆弱。
- 官方 native 示例明确使用 `enable_thinking=False`。

首发不暴露 `think_enabled` 配置。后续若要支持 thinking，必须先定义 reasoning 清洗契约和测试。

### 3. 图片输入

图片处理行为：

- `OCRProvider.prepare_media()` 仍负责输出目录与 preview pixels。
- provider 内重新用 PIL 打开图片并转 RGB。
- 单图 OCR 输出写入 `Path(uri).with_suffix("") / "result.md"`。
- `CaptionResult.raw` 返回清理后的 Markdown。

### 4. PDF 输入

PDF 复用现有分页逻辑：

- `utils.stream_util.pdf_to_images_high_quality()`

逐页行为：

- 保存 `page_xxxx/page_xxxx.png`
- 对该页调用同一单图推理函数
- 写入 `page_xxxx/result.md`

根目录行为：

- 成功页面按顺序使用 `\n<--- Page Split --->\n` 拼接
- 写入根目录 `result.md`
- `CaptionResult.raw` 返回拼接内容

失败策略：

- 单页失败时记录异常并继续下一页。
- 如果所有页面都失败，抛出显式错误。

### 5. Wrapper / vLLM 后续扩展点

`infinity_parser2` wrapper 后续可以作为第二阶段接入，但不能阻塞首发。

后续扩展条件：

- 明确项目支持 Python 3.12+ 的运行链路。
- 明确 vLLM / CUDA / FlashAttention 版本矩阵。
- 明确 wrapper 返回内容和仓库 `result.md` / `CaptionResult.raw` 契约完全一致。
- 明确 `output_dir` 模式下 wrapper 的写盘目录不会和仓库现有输出目录约定冲突。

vLLM server 后续接入要求：

- 不能直接无验证复用 `OCRProvider.attempt_via_openai_backend()`。
- 必须验证 `vllm serve <selected-model-id> --reasoning-parser qwen3` 与仓库当前 OpenAI-compatible message shape 兼容。
- 必须复用同一 prompt 解析逻辑和输出清理逻辑。

## 依赖设计

在 `pyproject.toml` 新增独立 optional extra：

```toml
infinity-parser2-ocr = [
    "qinglong-captions[torch-base]",
    "transformers[serving]>=5.3.0",
    "tokenizers>=0.22.2",
    "qwen-vl-utils>=0.0.14",
    "Pillow>=12",
    "PyMuPDF",
    "img2pdf",
    "flash-attn==2.8.4; sys_platform == 'linux'",
    "triton-windows; sys_platform == 'win32'",
    "flash-attn @ https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.4/flash_attn-2.8.4+cu130torch2.11.0cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.11'",
]
```

说明：

- 不把 `infinity_parser2` PyPI package 放进首发 extra。
- 不把 vLLM 放进首发 extra。
- 如果后续接 wrapper，可新增单独 extra，例如 `infinity-parser2-wrapper`，避免污染 direct Transformers 用户。
- Windows FlashAttention wheel 是否可用必须沿用仓库现有策略；不可用时 provider 应允许回退到非 flash attention 路径，或给出明确错误。

## 输出契约

首发输出只承诺 Markdown 文本。

要求：

- 图片：`output_dir/result.md`
- PDF 单页：`output_dir/page_xxxx/result.md`
- PDF 汇总：`output_dir/result.md`
- `CaptionResult.raw`: Markdown 字符串
- `CaptionResult.metadata.provider`: `infinity_parser2_ocr`
- `CaptionResult.metadata.model_id`: 实际使用的模型 ID
- `CaptionResult.metadata.output_dir`: 输出目录

不承诺：

- 保留字体粗细、斜体、删除线
- 保留所有复杂表格结构
- 保留旋转元素的准确阅读顺序
- 默认输出 layout JSON

## 错误处理

必须提供清晰错误：

- 缺 `transformers`：
  - 提示安装 `--extra infinity-parser2-ocr`
- 缺 `qwen_vl_utils`：
  - 提示安装 `--extra infinity-parser2-ocr`
- 显存不足：
  - 保留原始 CUDA OOM 信息，并提示当前选择的模型 ID 与建议切换 Flash / 降低像素范围
- `task_type` 非法：
  - 直接报 `Unsupported infinity_parser2_ocr task_type`
- `task_type="custom"` 但 prompt 为空：
  - 直接报 `custom task requires a non-empty prompt`
- 生成结果为空：
  - 直接报 `Infinity Parser2 OCR returned empty output`

## 测试计划

新增测试文件：

- `tests/test_infinity_parser2_ocr_provider.py`

覆盖：

- provider 默认 prompt。
- provider 默认 model ID 是 `infly/Infinity-Parser2-Flash`。
- `model_list` 包含 Flash 和 Pro。
- prompt 优先级。
- `task_type` 校验。
- `custom` 空 prompt 报错。
- `enable_thinking=False` 被传给 `apply_chat_template`。
- `min_pixels` / `max_pixels` / `image_patch_size` 默认值可覆盖。
- 输出清理不会写空结果。

更新现有测试：

- `tests/test_provider_catalog.py`
  - `route_choices("ocr_model")` 包含 `infinity_parser2_ocr`
- `tests/test_provider_registry.py`
  - registry 能发现 provider
- `tests/test_provider_routes.py`
  - PDF mime 显式选择 `infinity_parser2_ocr` 能命中
- `tests/test_penguin_dependencies.py`
  - pyproject 声明 `infinity-parser2-ocr`
  - GUI local extra 映射返回 `--extra infinity-parser2-ocr`
  - PowerShell 脚本包含 `Add-UvExtra "infinity-parser2-ocr"`
- `tests/test_model_toml_helpers.py`
  - Flash / Pro 的 `model_list` fit hint 可被 GUI 读取

慢测，不进默认 CI：

- 使用一张中英文文档截图跑 Flash 真实推理，验证输出非空 Markdown。
- 使用一个短 PDF 跑 Flash 逐页推理，验证根目录和页目录写盘。
- 可选手动把 `model_id` 改成 Pro 跑同一输入，确认 provider 不需要改代码。

## 实施顺序

1. 增加 provider spec 和配置项。
2. 增加 `pyproject.toml` optional extra。
3. 增加 `module/providers/ocr/infinity_parser2.py`。
4. 接入 catalog / registry。
5. 接入 GUI / PowerShell extra 映射。
6. 增加 provider 单元测试和路由测试。
7. 跑不加载真实模型的快速测试。
8. 手动慢测 Flash 图片和 PDF。
9. 可选手动慢测 Pro。

## 验收标准

- `ocr_model="infinity_parser2_ocr"` 对 PDF 和 document image 均能命中 provider。
- 默认模型 ID 是 `infly/Infinity-Parser2-Flash`。
- 配置 `model_list` 同时包含 Flash 和 Pro。
- `--extra infinity-parser2-ocr` 能被 GUI 和 PowerShell 脚本正确追加。
- provider 构造 native Transformers 消息时包含图片、prompt、官方像素范围和 `enable_thinking=False`。
- 不安装 extra 时错误信息能指导用户安装正确 extra。
- 不加载真实模型的单元测试通过。
- 真实慢测至少验证 Flash 单图输出非空。
- 将 `model_id` 改为 `infly/Infinity-Parser2-Pro` 后，provider 仍走同一推理路径。

## 风险与缓解

### 1. Pro 35B 成本高

风险：

- 用户可能误把 Pro 作为默认模型，导致显存不足和安装失败。

缓解：

- 默认使用 Flash。
- Pro 只放入 `model_list` 可选项。
- GUI fit hint 标注 Pro 高显存需求。
- OOM 错误中提示当前模型 ID，并建议切回 Flash。

### 2. 官方 wrapper 依赖过重

风险：

- wrapper 的 Python 3.12+ 和 vLLM 依赖可能破坏当前环境。

缓解：

- 首发不引入 wrapper。
- wrapper 后续独立 extra 接入。

### 3. 输出格式误解

风险：

- 官方默认 doc2json 能力强，但仓库 OCR 用户期望 Markdown。

缓解：

- 首发默认 `doc2md`。
- `doc2json` 等 layout JSON 输出等单独设计，不混入当前 OCR 契约。

### 4. route alias 造成模型选择幻觉

风险：

- 如果增加 `infinity_parser2_pro` route alias，用户会以为 route 能直接选择 Pro。

缓解：

- 不增加 Pro route alias。
- 模型版本只通过 `[infinity_parser2_ocr].model_id` / GUI 模型配置选择。

## 更低成本替代方案

如果目标只是获得更强 OCR，而不是必须使用 Infinity Parser2，当前仓库已经有多个低成本 provider：

- `chandra_ocr`
- `dots_ocr`
- `lighton_ocr`
- `qianfan_ocr`

从第一性原理看，`Infinity-Parser2-Flash` 是更合理的默认值：它接近 Pro 的公开 OCR 指标，但只有 2B BF16，失败成本远低于 35B Pro。Pro 应留给明确需要最高精度并且有足够硬件的场景。
