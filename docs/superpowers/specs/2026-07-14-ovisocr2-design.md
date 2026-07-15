# OvisOCR2 双后端适配规格

## 目标

为项目增加 `ovis_ocr2` OCR Provider，支持图片和 PDF，并提供两条推理路径：

- Direct Transformers：进程内加载 `ATH-MaaS/OvisOCR2`，作为默认路径。
- OpenAI-compatible vLLM：连接外部 Linux/WSL 服务。

两条路径只负责“输入单页 RGB PIL 图像，返回单页文本”。图片/PDF 逐页处理、失败隔离、官方重复尾清理、视觉区域处理、路径重写和 Markdown 写盘由 Provider 共享，避免后端产生不同的文件契约。Direct 路径允许在生成器确认陷入稳定周期重复时提前停止，并只归一化触发停止的精确重复后缀；这是推理运行时的终止保护，不分叉 Provider 的共享语义后处理。

“输出一致”指相同输入协议、后处理和目录结构，不承诺不同推理引擎逐字生成相同文本。

## 架构边界

不修改 `OCRProvider` 抽象，不复用会缩放图像或插入空 system 消息的通用视觉消息辅助函数，也不使用基类的 OpenAI 后端快捷路径。

```text
OvisOCR2Provider
  -> RGB page iterator
  -> DirectPageInferencer
       -> greedy generation
       -> repeated-tail stopping and exact trigger normalization
     | OpenAIPageInferencer
  -> non-empty validation
  -> official repeated-tail cleanup
  -> bbox crop/drop
  -> page result.md
  -> PDF root path rewrite and aggregation
```

## Provider 契约

- Provider 名称：`ovis_ocr2`
- 默认模型：`ATH-MaaS/OvisOCR2`
- 支持：`application/pdf`，以及启用 `document_image` 的图片。
- 不把其他 `application/*` MIME 当成 PDF。
- 所有页面在推理前转换为 RGB PIL 图像。
- PDF 使用 `iter_pdf_pages_high_quality()`，页面目录为 `page_0001`、`page_0002` 等。
- PDF 单页失败时记录警告和 `failed_pages`，继续处理其他页面；所有页面失败时抛错。
- 根结果只聚合成功页面，分隔符沿用 `<--- Page Split --->`。
- 页面 PNG 是调试快照；保存失败不阻断内存中的推理与裁图。

结果元数据包含 `provider`、`output_dir`、`runtime_backend`、`runtime_model_id`、`failed_pages` 和 `visual_region_mode`。

## 已复现的 Direct 性能故障

2026-07-15 在项目 `.venv`、RTX 4090、PyTorch `2.13.0+cu130`、Transformers `5.13.1` 上复现了 Direct 路径首图超过 20 分钟而进度仍为 `0/116` 的问题。测试输入是 116 张图片组成的目录；进度条只会在整张图片完成后递增，因此表象不是 PDF 卡页，也不是媒体读取死锁。

对首图 `001.jpg`（1357x1920）做受控测试得到：

- Processor 约 0.383 秒，输入 2647 tokens，其中合并后视觉 tokens 为 2520。
- Prefill 加 1 个生成 token 约 28.8 秒，排除预处理或 prefill 无限阻塞。
- 生成 512 tokens 约 53.7 秒，恰好撞到上限；末 token 不是 EOS，输出尾部持续重复 `1`。
- 当前约 9.5 token/s；若保持 `max_new_tokens=16384`，单页最坏生成时间约 29 分钟。
- GPU 利用率约 19% 到 30%，显存约 5.9 GiB，未发现 OOM、CPU offload 或设备错配。

同一批输入的 `003.jpg` 进一步排除了“只是短周期重复”的假设：约 25 分钟后输出达到 26,440 字符、4,988 行，正确目录之后出现模型生成的空 `<think>...</think>`，随后从 `## 1` 单调递增到 `## 2470` 并撞上生成上限。单调计数没有稳定字符周期，因此任何只扩大周期窗口的方案仍然无法终止它。`001.jpg` 和 `002.jpg` 的额外 `1`、重复标题同样都出现在首个模型生成的 `<think>` 之后。

本地 Processor 的受控模板渲染确认：`enable_thinking=False` 会在输入提示末尾预填一个空 `<think>...</think>`。Direct 解码已经裁掉全部输入 tokens，所以最终结果里的标签不是误解码的提示词，而是模型在正确答案后重新生成的控制标记。三个表象因此共享一个可靠边界：**新生成文本中出现独占行且内容为空的 `<think>...</think>` 控制块时，答案已经结束。**

故障由三个因素叠加：

1. **控制标记退化：** 模型会在正确结果后重新生成 `<think>`，其后可能是孤立数字、重复正文，也可能是无法由周期算法发现的单调计数。
2. **周期重复退化：** 模型进入稳定周期重复后不发 EOS，现有模型卡算法只在 `generate()` 完成后清理文本，无法阻止已浪费的生成。[官方模型卡](https://huggingface.co/ATH-MaaS/OvisOCR2)的 vLLM 示例同样包含 `_clean_truncated_repeats()`，说明重复尾不是 Torch fallback 独有现象。
3. **生成吞吐：** 当前环境未安装 `flash-linear-attention` 和 `causal-conv1d`，Transformers 回退到 Torch 实现；回退路径中的逐步计算使低利用率和长尾时间更明显，但不是不发 EOS 的根因。

因此，安装快速内核只能缩短每个 token 的耗时，不能替代生成期终止保护。反过来，终止保护避免无效 token，却不承诺正常长页面的吞吐会达到 vLLM 水平。

## Direct Transformers

- 使用 `AutoProcessor` 和 `Qwen3_5ForConditionalGeneration`。
- `trust_remote_code=False`，依赖 `transformers[serving]>=5.7.0` 的原生 Qwen3.5 实现。
- 消息只有一条 user，内容顺序严格为 image、text，不增加 system 消息。
- `apply_chat_template()` 参数：
  - `tokenize=True`
  - `add_generation_prompt=True`
  - `return_dict=True`
  - `return_tensors="pt"`
  - `enable_thinking=False`
  - `processor_kwargs={"images_kwargs": {"min_pixels": 448*448, "max_pixels": 2880*2880}}`
- 像素参数必须放入 `processor_kwargs`，不得作为 `apply_chat_template()` 的顶层额外参数，以兼容 Transformers 5.x 并消除回退警告。
- 生成使用 `do_sample=False`、`max_new_tokens=16384` 和下述重复尾 StoppingCriteria，裁掉输入 token 后解码。
- 模型和 Processor 通过现有 `transformerLoader` 缓存。

## OpenAI-compatible vLLM

- `OpenAIChatRuntime.complete()` 增加可选 `extra_body`，既有调用保持兼容。
- 原始 RGB 页面编码为不缩放的 PNG data URL。
- 消息只有一条 user，内容顺序为 image_url、text。
- 请求附带：

```json
{
  "mm_processor_kwargs": {
    "images_kwargs": {
      "min_pixels": 200704,
      "max_pixels": 8294400
    }
  },
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

服务部署示例：

```bash
pip install "vllm==0.22.1" pillow
vllm serve ATH-MaaS/OvisOCR2 --gdn-prefill-backend triton --gpu-memory-utilization 0.8
```

项目 extra 不安装 vLLM；vLLM 不原生支持 Windows，服务部署在外部 Linux/WSL 环境。

## 提示词

代码中只保存一份官方默认提示词。覆盖优先级：

1. 非空 `[ovis_ocr2].prompt`
2. 非空 legacy `prompts.ovis_ocr2_prompt`
3. 代码中的官方默认提示词

空字符串和纯空白不会覆盖默认值。

## 生成终止与退化尾清理

处理分成两层，职责不可混合。

### Direct 生成期终止保护

每次 `infer_page()` 创建新的、仅用于 batch size 1 greedy generation 的 StoppingCriteria。它只观察 prompt 之后的新 tokens；若未来调用方传入其他 batch size，立即报错而不是静默返回错误形状。检测采用以下内部常量：

- 至少生成 128 tokens 后才开始检测。
- 每新增 32 tokens 检测一次，避免每 token 都触发 GPU 到 CPU 同步。
- 每次只解码末尾最多 768 个新 tokens，检测成本不随页面长度增长。
- 字符周期长度为 1 到 200。
- 精确后缀至少重复 8 次，且整个重复区至少 200 个字符，两个条件必须同时满足。

每次定期解码尾窗后，必须先查找独占行且内容为空的完整 `<think>...</think>` 控制块。该控制块只可能来自新生成 tokens，不能与输入模板中的预填块混淆；命中后记录 `thinking_marker` 原因并立即停止，不等待后续文本。行内字面量、未闭合开标签或包含非空正文的标签块都不构成边界。若未命中控制块，再执行周期重复检测。

重复识别必须抽成一个无模型状态的纯函数，并由生成期检测和 Provider 官方清理共同复用；两层只传入不同阈值，不各写一套周期算法。匹配结果至少记录周期字符长度、重复区字符长度、重复次数、末尾不完整周期长度，以及用于最终复验的精确后缀指纹。多个周期都满足时，确定性选择最短周期，与模型卡从短到长扫描的顺序一致。

确认任一停止条件后，StoppingCriteria 返回与 `input_ids` 同设备、形状为 `(1,)` 的布尔张量。它不在逐 token 回调中写日志，只保存触发信息；`generate()` 返回后统一记录停止原因和新生成 token 数，周期重复额外记录周期长度和重复次数。

最终全文解码后，Direct 推理器必须用记录的固定周期和精确后缀指纹重新验证，只把已确认的重复后缀缩为一个完整周期并保留原有的不完整周期尾片段，与模型卡算法的截断语义一致。不得在这里重新自由搜索另一个周期。若 token 边界或解码差异导致无法复验，则保留原文并记录警告，不做猜测性裁剪。该步骤只清理由本次提前停止直接产生的尾部，不处理其他文本模式。

以下行为明确排除：

- 不降低全局 `max_new_tokens=16384`，避免截断合法的长页面。
- 不增加 `repetition_penalty` 或 `no_repeat_ngram_size`，它们会改变正常 greedy 输出，并可能破坏表格、公式和目录中的合法重复。
- 不默认使用 `MaxTimeCriteria`；墙钟超时受设备、首次加载和页面复杂度影响，只能作为未来独立的兜底策略。
- 不修改 vLLM 请求路径。服务端仍负责自己的生成调度，返回文本进入同一共享后处理。

### Provider 共享清理

无论后端，Provider 先从首个独占行的空 `<think>...</think>` 控制块处截断并去掉其前方尾随空白，再严格采用模型卡算法做第二层清理：文本长度至少 8000；周期范围 1 到 200；重复总字符至少 100；重复次数至少 5。只删除检测到的截断式重复尾并保留一个完整周期。截断后为空按空模型输出处理，不写入伪结果。

共享清理是跨后端的输出规范；Direct 生成期保护是有界生成保障。OpenAI-compatible 客户端只能在服务返回后执行共享截断，服务端生成期保护属于 vLLM 部署配置，不在本 Provider 内伪装成客户端能力。即使未来快速内核可用，也必须保留这两层设计。

## 视觉区域

`visual_region_mode` 仅接受 `crop` 或 `drop`，默认 `crop`，其他值立即报错。

只识别以下官方标签：

```html
<img src="images/bbox_L_T_R_B.jpg" />
```

`crop` 模式按 `round(coord * dimension / 1000)` 映射坐标并夹紧到页面边界。有效区域保存为完全相同的文件名并保留标签。非法 bbox 或保存失败时删除对应标签并记录警告，不留下损坏链接。重复 bbox 只需确定性生成一个资源文件。

`drop` 模式只移除匹配标签，保留同一段落的其他文本，不创建 `images` 目录。

PDF 页面 `result.md` 使用 `images/...`；根 `result.md` 只给匹配的 OvisOCR2 标签添加 `page_0001/` 前缀，其他 Markdown 图片和 HTML 图片保持原样。

## 配置与依赖

声明路由 `ovis_ocr2`，`priority=122`、`order=15`。GUI 通过 catalog 自动获得路由并映射到 `ovis-ocr2` extra，但不加入版本不可比的 OCR 排名表。

默认配置：

```toml
[ovis_ocr2]
model_id = "ATH-MaaS/OvisOCR2"
runtime_backend = "direct"
prompt = ""
max_new_tokens = 16384
temperature = 0.0
top_p = 1.0
min_pixels = 200704
max_pixels = 8294400
visual_region_mode = "crop"
```

`ovis-ocr2` extra 只包含 `torch-base`、`transformers[serving]>=5.7.0` 和 `PyMuPDF`。不加入 vLLM、`qwen-vl-utils` 或 `img2pdf`。与旧 Transformers 固定版本及 fork runtime 的 extra 声明冲突隔离。

`flash-linear-attention` 和 `causal-conv1d` 是可选性能依赖，不加入默认 extra，也不成为 Direct 正确性的前置条件。项目不承诺其 Windows 原生安装链路；用户即使自行安装且两个模块都能成功导入，重复尾 StoppingCriteria 仍保持启用。快速内核负责吞吐，停止条件负责终止，不能用前者替换后者。

## 验证

- 单元测试覆盖重复尾边界、crop/drop、非法及重复 bbox、裁图失败和 PDF 路径重写。
- 生成期检测测试覆盖：prompt tokens 被排除、空 `<think>...</think>` 控制边界、行内字面量及非空标签块保留、128/32/768 边界、1 到 200 字符周期、8 次和 200 字符双阈值、未达到阈值不触发，以及常见 Markdown/HTML 合法重复不被截断。
- StoppingCriteria 测试断言 batch size 1、返回张量的形状和设备、每页状态隔离、触发元数据，以及完整解码只能压缩已复验的触发后缀。
- Direct 契约测试断言 `generate()` 收到 StoppingCriteria；vLLM 契约及输出保持不变。
- 两个后端分别验证消息顺序、像素参数、thinking、greedy、未缩放 PNG 和 `extra_body`。
- 共享管线覆盖两个后端的 `<think>` 及后续计数截断、单图、双页 PDF、部分失败、全部失败、空输出和调试快照失败。
- 接入测试覆盖 Provider catalog、配置镜像、GUI extra、PowerShell 路由和依赖声明。
- 完成前运行相关 pytest、ruff、`git diff --check`，并在可用 GPU 上做一次真实 Direct 单页冒烟。只有仓库已跟踪 `uv.lock` 时才运行 `uv lock --check`；当前仓库没有锁文件，本次无依赖改动，不为该修复新建锁文件。
- GPU 验收沿用已复现的 `001.jpg` 和 `003.jpg` 及 `max_new_tokens=16384`：必须在远低于上限时因稳定重复或新生成 `<think>` 停止；最终 Markdown 不含控制标签、额外 `1` 或单调编号尾。冷启动时间只作为当前机器的人工基准记录，不写成跨机器自动化断言。

2026-07-15 修复后在同一 `.venv`、同一 `003.jpg`、Torch fallback 和 `max_new_tokens=16384` 下完成真实 Direct 冒烟：含模型冷加载共 71.44 秒，在 224 个生成 tokens 时命中空 thinking 控制块；原始输出 366 字符，共享截断后为 306 字符，断言不含 `<think>` 或 `</think>`。该结果验证了 25 分钟单调计数退化已在生成早期被阻断。
