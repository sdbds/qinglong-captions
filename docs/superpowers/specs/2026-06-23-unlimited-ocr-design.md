# Unlimited-OCR Provider 设计

## 背景

仓库 Provider V2 已经有稳定的本地 OCR 接入方式。现有本地 OCR provider：

- `infinity_parser2_ocr`
- `deepseek_ocr`
- `logics_ocr`
- `dots_ocr`
- `qianfan_ocr`
- `lighton_ocr`
- `hunyuan_ocr`
- `glm_ocr`
- `chandra_ocr`
- `olmocr`
- `paddle_ocr`
- `nanonets_ocr`
- `firered_ocr`

这些 provider 已经共享以下行为：

- 通过 `ocr_model` 路由选择 provider（`OCRProvider.can_handle`：`ocr_model == cls.name`，PDF 恒处理，图片需 `document_image=True`）。
- `prepare_media` 默认输出目录使用输入文件去后缀同名目录。
- 输出 Markdown 到 `result.md`，`CaptionResult.raw` 返回最终 Markdown。
- provider 身份、路由、config section、extra 名称由 `module/providers/declarations.py` 的 `ProviderDeclaration` 统一声明，`catalog.py` 据此构建路由表。
- GUI / PowerShell 通过 `uv --extra` 追加本地依赖，每个 OCR extra 独立 pin transformers。
- registry 通过显式模块清单（declarations）发现 provider。

本次需求：新增 Baidu Unlimited-OCR 本地 OCR provider。

参考来源：

- 代码仓库：`https://github.com/baidu/Unlimited-OCR`
- 权重：`https://huggingface.co/baidu/Unlimited-OCR`（MIT 许可）

仓库当前没有任何 `unlimited` / `baidu` OCR provider，没有旧 provider 需要迁移。因此本设计按“全新 provider 接入”处理，不做旧 provider 升级，不做旧配置自动迁移。

## 模型事实

下列事实来自官方 README（已通过抓取确认）：

- Unlimited-OCR 是 Baidu 的文档解析模型，定位“one-shot long-horizon parsing”，官方说法是“把 DeepSeek-OCR 再推进一步”，并致谢 DeepSeek-OCR / DeepSeek-OCR-2 / PaddleOCR。即它是 DeepSeek-OCR 系建模代码的衍生，推理接口与 `deepseek_ocr` 同形。
- 提供两条推理路径：
  1. **native Transformers**（HF，`trust_remote_code=True`）。README 测试环境 Python 3.12.3 + CUDA 12.9，torch 2.10.0、torchvision 0.25.0、transformers 4.57.1、pymupdf 1.27.2.2，外加 Pillow / matplotlib / einops / addict / easydict / psutil。
  2. **SGLang HTTP server**（捆绑 wheel + `kernels==0.11.7`，OpenAI 兼容 `/v1/chat/completions`，监听 `0.0.0.0` 无鉴权）。仓库 `infer.py` 走的是这条路。
- native Transformers 加载方式：`AutoModel.from_pretrained(model_id, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16)` 然后 `model.eval().cuda()`。
- 单图推理：`model.infer(...)`，prompt `"<image>document parsing."`，关键参数 `max_length=32768`、`no_repeat_ngram_size=35`、`ngram_window=128`、`save_results=True`、`output_path=...`。
- 多页 / PDF 推理：`model.infer_multi(...)`，prompt `"<image>Multi page parsing."`，接收 `image_files` 列表、`image_size=1024`、`max_length=32768`、`no_repeat_ngram_size=35`、`ngram_window=1024`、`save_results=True`。PDF 先用 PyMuPDF 转 PNG 再传入。
- 图像配置：
  - gundam：`base_size=1024, image_size=640, crop_mode=True`
  - base：`base_size=1024, image_size=1024, crop_mode=False`
  - 多页 / PDF 只用 base（`image_size=1024, crop_mode=False`）。

### 未验证项（实现期必须落实，不得凭空写死）

README 没有给出、当前无法确认的事实，全部标注为风险，不在 spec 里假装确定：

1. `model.infer` / `model.infer_multi` 的**确切关键字参数名**（tokenizer/processor 的传入位置、`image_file` vs `image_files`、是否还需 `prompt=`）。trust_remote_code 模型签名会随版本漂移，必须对照 HF 上的 `modeling_*.py` 核对。
2. **输出落盘的确切文件名**。`deepseek_ocr` 的 `infer` 写的是 `result.mmd`，provider 再 rename 成 `result.md`。Unlimited-OCR 作为同系 fork 很可能也是 `result.mmd`，但未证实；也可能 `infer` 直接返回字符串。
3. `infer_multi` 的返回形态：是返回一段合并文本，还是按页写多个文件、返回值另算。
4. 与仓库当前 `torch-base`（近期已升到 torch 2.12.x）的兼容性。README 只测过 torch 2.10.0 / CUDA 12.9。
5. olmOCR-bench / ParseBench 分数、支持语言列表 —— README 未给出，**不得在 GUI 里编造排名分数**。

## 三个问题（先回答再动手）

1. **是真问题还是臆想复杂度？** 真问题。用户要新增一个 OCR 模型能力，仓库已有 13 个同构 OCR provider，再加一个是有边界的常规接入工作。

2. **有没有更简单的做法？** 有，而且这是本 spec 的核心：Unlimited-OCR 提供 native Transformers 与 SGLang 两条路。仓库现成的 `deepseek_ocr` provider 已经在用 native `model.infer()` 路径，而 Unlimited-OCR 与它**同接口**。所以最简做法 = 以 `deepseek_ocr` 为模板克隆，换 `model_id` / prompt，PDF 改用 `infer_multi`。**首发不引入 SGLang / vLLM**（重依赖、Python 3.12 wheel、`0.0.0.0` 无鉴权监听）—— 这正是 `infinity_parser2` spec 已经踩过并记录的结论。

3. **会破坏什么？** 作为全新 provider，正确实现不破坏任何现有行为。唯一真实风险是**依赖冲突矩阵**：每个 OCR extra 各自 pin 不兼容的 transformers（`deepseek-ocr` pin `transformers==4.46.3`，Unlimited README 用 4.57.1）。因此 Unlimited 必须有**独立 extra + 独立 pin**，并加入 `[tool.uv]` 的冲突矩阵，尤其与 `deepseek-ocr` 互斥。

## 已确认决策

- 新增 provider，不改写现有 provider，不动 `OCRProvider` 基类。
- canonical provider 名 `unlimited_ocr`；`ocr_model` 路由值 `unlimited_ocr`；config section `[unlimited_ocr]`；extra `unlimited-ocr`；prompt key `unlimited_ocr_prompt`。命名沿用 `deepseek_ocr` 家族风格。
- 首发只承诺 **native Transformers** 路径；SGLang / vLLM server 作为后续扩展点，不阻塞首发。
- 默认 `model_id = "baidu/Unlimited-OCR"`。
- 单图默认 **gundam** 模式（`base_size=1024, image_size=640, crop_mode=True`），对应官方 `infer.py` 默认 `--image_mode gundam`。
- PDF / 多页强制 **base** 模式（`image_size=1024, crop_mode=False`），用 `model.infer_multi` 一次性长程解析 —— 这是该模型相对 deepseek 的真正差异点，也是它作为独立 provider 存在的理由。**不**把 PDF 退化成逐页 `infer` 循环。
- 输出契约不变：图片 `output_dir/result.md`；PDF 汇总 `output_dir/result.md`；`CaptionResult.raw` 返回 Markdown 字符串。
- 不自动下载权重；只在用户显式选 `ocr_model="unlimited_ocr"` 时加载。
- 该 provider 是可选 OCR extra，不进默认依赖集，加入冲突矩阵。
- **不复制 `deepseek.py` 里的 `self` 引用 bug**（见“预先存在的 bug”一节）。

## 命名决策

不使用 `baidu_ocr`，用 `unlimited_ocr`。

- 仓库已有 `qianfan_ocr`（同为 Baidu 系），用厂商名 `baidu_ocr` 会和 qianfan 语义撞车。
- `unlimited_ocr` 直接对应模型族名 Unlimited-OCR，边界清楚。

不设计 SGLang 的 route alias。理由同 `infinity_parser2` spec：`RouteSpec.aliases` 只能把 route value canonicalize 到 provider，不能切换运行后端；增加 alias 只会制造“能直接选 server 模式”的配置幻觉。后端选择应通过 `[unlimited_ocr]` 配置（后续若接 server，复用 `get_runtime_backend` 的 `is_openai` 判定）。

## 目标

1. Provider V2 新增 `unlimited_ocr` 本地 OCR provider。
2. native Transformers 路径接入 `baidu/Unlimited-OCR`。
3. 单图走 `model.infer`（gundam），PDF/多页走 `model.infer_multi`（base, 一次性长程）。
4. 输出继续写 `result.md`，`CaptionResult.raw` 返回 Markdown。
5. 模型依赖放入独立 optional extra `unlimited-ocr` 并登记冲突矩阵。
6. 在 declarations / config / prompts / GUI / PowerShell / tests 完整暴露该 provider。

## 非目标

- 不做旧 provider 迁移（仓库无旧 provider）。
- 首发不引入 SGLang / vLLM server 路径。
- 不新增独立 GUI 配置页。
- 不新增 OCR 通用 backend 抽象层。
- 不改造 `OCRProvider` 基类。
- 不承诺样式级还原、多语言高准确率、复杂旋转表格稳定解析（README 未承诺）。
- 不在 GUI 排名表里给 Unlimited-OCR 编造 bench 分数。

## 架构设计

### 1. 新增 provider 文件

新增 `module/providers/ocr/unlimited.py`，类 `UnlimitedOCRProvider`，注册名 `unlimited_ocr`。

以 `module/providers/ocr/deepseek.py` 为模板（同接口、同 `result.mmd→result.md` 处理、同 `transformerLoader`），差异：

- `default_model_id = "baidu/Unlimited-OCR"`
- `default_prompt`：单图 `"<image>document parsing."`
- PDF 分支用 `model.infer_multi`（base 模式），prompt `"<image>Multi page parsing."`，而非逐页 `infer`。
- 不复制 deepseek 的 `self` 引用 bug。

### 2. declarations 集成（路由 / catalog 的真正来源）

`module/providers/declarations.py` 的 `PROVIDER_DECLARATIONS` 新增一项。`catalog.py` / `registry.py` 全自动从 declarations 派生，**不需要**手改 catalog 或 registry：

```python
ProviderDeclaration(
    name="unlimited_ocr",
    module_path="module.providers.ocr.unlimited",
    priority=125,  # 介于 infinity_parser2_ocr(120) 与 deepseek_ocr(130) 之间
    routes=(route("ocr_model", order=85),),  # 紧邻 deepseek_ocr(order=80)
    config_sections=("unlimited_ocr",),
    prompt_prefixes=("unlimited_ocr",),
    capabilities=capabilities(supports_images=True, supports_documents=True),
),
```

route `order` 决定 GUI 下拉框排序。bench 分数未知，故排在 deepseek 附近而非榜首；**不抢 `infinity_parser2_ocr` 的 #1 位置**。

### 3. GUI 集成（`gui/wizard/step4_caption.py`）

- `OCR_MODELS` 来自 `route_choices("ocr_model")`，自动包含 `unlimited_ocr`，无需手改。
- `OCR_EXTRA_MAP` 新增 `"unlimited_ocr": "unlimited-ocr"`（这是真正驱动 `_build_local_extra_args` 的映射，必须手加）。
- `_OCR_RANK_SCORES`：**不加** Unlimited 条目（无可信分数）。未命中条目时 `_build_ocr_labels` 已回退为裸名 `m`，行为正确。若后续拿到官方 bench，再补。

### 4. PowerShell 集成（`4.captioner.ps1` 与 `4、run.ps1`）

在两个脚本的 OCR extra 分发块（紧邻现有 `infinity_parser2_ocr` 判断）追加：

```powershell
if ($ocr_model -eq "unlimited_ocr") {
  Add-UvExtra "unlimited-ocr"
}
```

## 配置设计

`config/model.toml` 与 `config/config.toml` 新增（紧邻 `[deepseek_ocr]`）：

```toml
# Unlimited-OCR configuration
# Model: https://huggingface.co/baidu/Unlimited-OCR
[unlimited_ocr]
model_list."Unlimited OCR" = { model_id = "baidu/Unlimited-OCR", meta = { min_vram_gb = 12 } }
model_id = "baidu/Unlimited-OCR"
image_mode = "gundam"   # gundam | base，仅作用于单图；PDF/多页恒用 base
base_size = 1024
image_size = 640        # gundam 默认；base 模式下 provider 内部覆盖为 1024
crop_mode = true        # gundam=true, base=false
max_length = 32768
no_repeat_ngram_size = 35
```

字段说明：

- `image_mode`：单图模式选择。`gundam`（裁剪、640）精度优先；`base`（不裁剪、1024）速度优先。PDF/多页忽略此项，强制 base。
- `base_size` / `image_size` / `crop_mode`：直接映射官方 `infer` 参数。**provider 必须从 config section 读取这些值（`self._get_model_config(...)`），不要学 deepseek 从 `args` 读** —— 见下一节。
- `max_length` / `no_repeat_ngram_size`：官方解码默认值，作为 long-horizon 解析的关键参数暴露但不鼓励改。
- `meta.min_vram_gb`：仅 GUI fit hint，非硬保证（实际取决于 GPU / device_map / 量化 / offload）。

`config/prompts.toml` 的 `[prompts]` 下新增：

```toml
unlimited_ocr_prompt = """
<image>document parsing.
"""
```

Prompt 优先级（沿用 `OCRProvider.get_prompts` 既有逻辑）：

1. `[prompts].unlimited_ocr_prompt`
2. provider 类属性 `default_prompt`

注意：单图与 PDF 的官方 prompt 不同（`document parsing.` vs `Multi page parsing.`）。`unlimited_ocr_prompt` 覆盖的是**单图** prompt；PDF 的 `Multi page parsing.` 由 provider 内部固定，不走 prompts 覆盖（避免长程解析 prompt 被误改导致 `infer_multi` 行为异常）。若要可配置 PDF prompt，单独再设计，不混入首发。

## 预先存在的 bug（必须避免复制，不在本次修）

`module/providers/ocr/deepseek.py:69` 在模块级函数 `attempt_deepseek_ocr` 内引用了 `self`：

```python
supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False)),
```

该函数没有 `self` 形参，这行只要被执行就会 `NameError`。它能侥幸不炸，仅因为 `_TRANS_LOADER` 是模块级单例，第二次调用起 `_TRANS_LOADER is not None` 直接短路跳过。首次冷调用就会崩。

**要求**：`unlimited.py` 克隆时**不要**带上这一行；`transformerLoader(attn_kw=..., device_map="auto")` 即可，`supports_flex_attn` 用默认 `False`。

> 顺带：deepseek 的这个 bug 应另开任务修，不在本 spec 范围。

## 推理流程设计

### 1. 运行时加载

```python
from transformers import AutoModel, AutoTokenizer
_, dtype, attn_impl = resolve_device_dtype()
# 模块级单例 _TRANS_LOADER，attn_kw="_attn_implementation"（与 deepseek 一致，trust_remote_code 模型用下划线名）
tokenizer = _TRANS_LOADER.get_or_load_processor(model_id, AutoTokenizer, console=console)
model = _TRANS_LOADER.get_or_load_model(
    model_id, AutoModel,
    dtype=dtype, attn_impl=attn_impl,
    trust_remote_code=True, low_cpu_mem_usage=True,
    device_map="auto", use_safetensors=True, console=console,
)
```

### 2. 单图路径（gundam 默认）

1. 解析 `image_mode`：gundam → `(base_size=1024, image_size=640, crop_mode=True)`；base → `(1024, 1024, False)`。config 显式给出的 `base_size/image_size/crop_mode` 优先于模式默认。
2. 调 `model.infer(tokenizer, prompt=prompt_text, image_file=str(p), output_path=output_dir, base_size=..., image_size=..., crop_mode=..., save_results=True)`。
   - **实现期校对**：HF `modeling_*.py` 里 `infer` 的确切签名（关键字名、是否还接受 `max_length`/`no_repeat_ngram_size`/`ngram_window`）。README 单图示例含 `max_length=32768, no_repeat_ngram_size=35, ngram_window=128`，若 `infer` 接受则一并传入。
3. 取结果：优先读 `output_dir/result.mmd`（存在则 rename 为 `result.md` 并读回），否则用 `infer` 返回值 `str(res)`。与 deepseek 完全一致。
4. `write_markdown_output(output_dir, content)`，`display_markdown(...)`。

### 3. PDF / 多页路径（base，一次性长程）

这是 Unlimited-OCR 的核心能力，**不退化为逐页循环**：

1. `images = pdf_to_images_high_quality(pdf_path)`。
2. 每页存 `output_dir/page_xxxx/page_xxxx.png`（沿用现有命名），收集路径列表 `image_files`。
3. 调 `model.infer_multi(tokenizer, prompt="<image>Multi page parsing.", image_files=image_files, output_path=output_dir, image_size=1024, save_results=True, max_length=32768, no_repeat_ngram_size=35, ngram_window=1024)`。
   - **实现期校对**：`infer_multi` 的确切参数名（`image_files` vs `images`）、返回形态、落盘文件名。
4. 取汇总结果：
   - 若 `infer_multi` 返回合并文本 → 直接作为 `content`。
   - 若它按页写盘 → 读回并以 `"\n<--- Page Split --->\n"` 拼接（沿用仓库分页约定）。
   - 两种都兜底，谁有内容用谁。
5. `write_markdown_output(output_dir, content)`。

失败策略：

- `infer_multi` 是单次长程调用，无法逐页 try/except 续跑。整体失败时抛显式错误 `"Unlimited-OCR multi-page parsing failed"`，附原始异常。
- 清理后内容为空 → 抛 `"Unlimited-OCR returned empty output"`，不写空 `result.md`。

#### 页数守卫（page_budget，2026-06-25 补丁）

实测确认 arXiv 论文的硬限制：R-SWA 一次性长程解析受 32K 上下文 + prefill 长度约束，实测上限 ~40-50 页，**模型本身没有 chunking**。原实现单次 `infer_multi` 喂全部页面，超限会**静默截断**尾部页面（输出停在 32K token，不报错）——对数据集/标注工具等于静默污染。

补丁把"页数超限"从静默 landmine 变成普通的分批循环（good taste：消除特殊分支）：

- 新增 `[unlimited_ocr].page_budget`（默认 30，`<=0` 关闭分批）。
- `len(image_files) <= page_budget`：**单次** `infer_multi`，行为与补丁前逐字节一致（never break userspace）。
- 超过：按 `page_budget` 分批，每批独立 `output_dir/chunk_xxxx/`（避免各批 `result.md` 互相覆盖），批间用 `"\n<--- Page Split --->\n"` 拼接。批内仍是一次性长程，只在批边界断连贯性——保住 95%+ 的跨页收益，消除截断。
- 失败仍 fail loud：任一批抛 `RuntimeError` 直接上抛，不静默丢批（不完整 markdown 比报错更坏）。
- 单图路径不受影响。
- 抽出 `_run_infer_multi(...)` 单批 helper，单次与分批共用，无重复。

明确**不做**：128K 上下文、prefill pool、KV chunking —— 百度自己都没做，是研究不是集成，ROI 极差。

### 4. OpenAI / SGLang 后端

首发 `attempt` 中 `if self.get_runtime_backend().is_openai:` 直接 `raise NotImplementedError("unlimited_ocr does not support the OpenAI-compatible runtime backend yet")`，与 `infinity_parser2` 首发一致。

后续接 SGLang server 的前置条件：

- 明确支持 Python 3.12 + 捆绑 wheel + `kernels==0.11.7` 的安装链路（独立 extra，如 `unlimited-ocr-sglang`）。
- 复用 `OCRProvider.attempt_via_openai_backend` 前，必须验证 `DeepseekOCRNoRepeatNGramLogitProcessor`（`ngram_size=35`）能否经 OpenAI 兼容请求传递；不能就不复用。
- server 默认 `0.0.0.0` 无鉴权，文档须警告仅限可信网络。

## 依赖设计

`pyproject.toml` 新增独立 optional extra，pin 对齐 README 测试环境，并与 `deepseek-ocr` 区隔（后者 pin `transformers==4.46.3`，二者不可共存）：

```toml
unlimited-ocr = [
    "qinglong-captions[torch-base]",
    "transformers[serving]==4.57.1",
    "matplotlib",
    "einops",
    "addict",
    "easydict",
    "psutil",
    "flash-attn==2.8.4; sys_platform == 'linux'",
    "triton-windows; sys_platform == 'win32'",
    "flash-attn==2.8.4; sys_platform == 'win32' and python_version == '3.11'",
    "PyMuPDF",
    "img2pdf",
]
```

说明：

- **不**放 SGLang 捆绑 wheel / `kernels` / vLLM。
- transformers pin 为 README 实测的 `4.57.1`；实现期若发现 trust_remote_code 代码要求更高/更低版本，以实跑为准修正，并记录原因。
- `[tool.uv]` conflicts 矩阵新增 `unlimited-ocr` 与以下互斥（与 `infinity-parser2-ocr` 已有矩阵对齐）：
  - `deepseek-ocr`（transformers pin 直接冲突，最关键）
  - `paddleocr-native`、`dots-ocr`、`translate`
  - `penguin-vl-local`、`eureka-audio-local`、`cohere-transcribe-local`
  - `reward-model`、`music-flamingo-local`
  - 具体集合以 `tests/test_pyproject_uv_conflicts.py` 现有 `infinity-parser2-ocr` 列表为准平移。

## 输出契约

首发只承诺 Markdown 文本：

- 图片：`output_dir/result.md`
- PDF 汇总：`output_dir/result.md`（页图存于 `output_dir/page_xxxx/page_xxxx.png`）
- `CaptionResult.raw`：Markdown 字符串
- `CaptionResult.metadata.provider`：`unlimited_ocr`
- `CaptionResult.metadata.model_id`：实际模型 ID
- `CaptionResult.metadata.output_dir`：输出目录

不承诺：字体样式还原、复杂表格结构、旋转元素阅读顺序、layout JSON。

## 错误处理

- 缺 `transformers` / `einops` / `addict` 等：提示安装 `--extra unlimited-ocr`。
- 显存不足：保留原始 CUDA OOM，提示当前 model_id 与降低 `image_size` / 改用 base 模式的建议。
- `image_mode` 非法（非 gundam/base）：报 `"Unsupported unlimited_ocr image_mode: <x>"`。
- `infer_multi` 整体失败：报 `"Unlimited-OCR multi-page parsing failed"` 附原始异常。
- 生成结果为空：报 `"Unlimited-OCR returned empty output"`，不写空文件。
- 选了 OpenAI backend：`NotImplementedError`，提示首发仅支持本地 Transformers。

## 测试计划

新增 `tests/test_unlimited_ocr_provider.py`，覆盖（全部 mock 模型，不下载权重）：

- provider 默认 model ID 是 `baidu/Unlimited-OCR`。
- 默认单图 prompt 为 `<image>document parsing.`，PDF 内部 prompt 为 `<image>Multi page parsing.`。
- `image_mode="gundam"` 解析出 `(1024, 640, True)`；`"base"` 解析出 `(1024, 1024, False)`；非法值报错。
- 单图调 `infer`、PDF 调 `infer_multi`（用 mock 断言调用的方法与参数）。
- 空输出抛错，不写空 `result.md`。
- OpenAI backend 抛 `NotImplementedError`。
- prompt 覆盖优先级（`unlimited_ocr_prompt` > `default_prompt`）。

更新现有测试（与 `infinity_parser2_ocr` 已有断言平移）：

- `tests/test_provider_catalog.py`：`route_choices("ocr_model")` 含 `unlimited_ocr`；`route_requires_remote_config(...)` 为 `False`。
- `tests/test_provider_registry.py`：registry 能发现 `unlimited_ocr`。
- `tests/test_provider_routes.py`：PDF mime 与 `document_image=True` 图片显式选择 `unlimited_ocr` 能命中。
- `tests/test_penguin_dependencies.py`：
  - pyproject 声明 `unlimited-ocr` extra。
  - GUI `_build_local_extra_args` 对 `unlimited_ocr` 返回 `["--extra", "unlimited-ocr"]`。
  - 两个 PS1 脚本含 `"unlimited_ocr"` 与 `Add-UvExtra "unlimited-ocr"`。
  - `config/model.toml`、`config/config.toml` 含 `[unlimited_ocr]`；`config/prompts.toml` 含 `unlimited_ocr_prompt`。
- `tests/test_pyproject_uv_conflicts.py`：`unlimited-ocr` 与 `deepseek-ocr` 等互斥对存在。

慢测（不进默认 CI，需真实权重）：

- 一张中英文文档截图跑 gundam，验证非空 Markdown。
- 一个多页 PDF 跑 `infer_multi`，验证根目录 `result.md` 与页图落盘。

## 实施顺序

1. `declarations.py` 增加 `unlimited_ocr` 声明。
2. `pyproject.toml` 增加 `unlimited-ocr` extra 与冲突矩阵。
3. `config/model.toml`、`config/config.toml`、`config/prompts.toml` 增加配置。
4. 新增 `module/providers/ocr/unlimited.py`（以 deepseek 为模板，避开 `self` bug，PDF 用 `infer_multi`）。
5. GUI `OCR_EXTRA_MAP` 增加映射。
6. 两个 PS1 脚本增加 extra 分发分支。
7. 新增 provider 单测 + 更新现有路由/依赖/冲突测试。
8. 跑不加载真实模型的快测。
9. **实现期对照 HF `modeling_*.py` 核对 `infer`/`infer_multi` 签名与落盘文件名**，修正 step 4 中的占位调用。
10. 手动慢测 gundam 单图与多页 PDF。

## 验收标准

- `ocr_model="unlimited_ocr"` 对 PDF 与 document image 均命中 provider。
- 默认 model ID 是 `baidu/Unlimited-OCR`。
- 单图走 `model.infer`（gundam），PDF 走 `model.infer_multi`（base）。
- `--extra unlimited-ocr` 能被 GUI 与两个 PowerShell 脚本正确追加。
- `unlimited-ocr` 与 `deepseek-ocr` 在 uv 冲突矩阵中互斥。
- 未安装 extra 时错误信息指导用户安装正确 extra。
- 不加载真实模型的单测通过。
- `unlimited.py` 不含 deepseek 的 `self` 引用 bug。
- 真实慢测至少验证 gundam 单图输出非空、多页 PDF 汇总 `result.md` 非空。

## 风险与缓解

1. **`infer`/`infer_multi` 签名漂移**（最高风险）：trust_remote_code 模型签名随版本变。缓解：实施顺序 step 9 强制对照 HF modeling 代码核对，慢测用真实权重验证；spec 中所有签名标为“占位，以实跑为准”。
2. **输出落盘文件名未知**：缓解：复用 deepseek 的“先试 `result.mmd`、否则用返回值”双兜底，对 `infer_multi` 同样双兜底。
3. **依赖与 deepseek 冲突**：transformers pin 不同。缓解：独立 extra + 冲突矩阵显式互斥，测试断言。
4. **torch-base 已升 2.12，README 只测 2.10**：缓解：慢测实跑确认；若 trust_remote_code 代码对 torch 版本敏感，记录并在 extra 内 pin。
5. **GUI 编造分数**：缓解：明确不进 `_OCR_RANK_SCORES`，回退裸名。

## 更低成本替代方案

如果目标只是“有一个能跑长文档的强 OCR”，而非必须用 Unlimited-OCR：仓库已有 `infinity_parser2_ocr`（2B Flash，bench #1 86.0）、`deepseek_ocr`、`chandra_ocr`、`dots_ocr` 等多个低成本本地 provider，且 `deepseek_ocr` 已是同系上游。

从第一性原理看，Unlimited-OCR 值得单独接入的唯一硬理由是它的 **`infer_multi` 一次性长程多页解析**（“one-shot long-horizon parsing”）—— 这是现有逐页 OCR provider 给不了的能力。如果接入后慢测显示多页效果并不优于 deepseek 逐页，则这个 provider 没有存在价值，应直接放弃，不要为了凑数留一个和 deepseek 几乎一样的壳。
