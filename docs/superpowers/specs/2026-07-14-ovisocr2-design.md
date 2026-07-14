# OvisOCR2 双后端适配规格

## 目标

为项目增加 `ovis_ocr2` OCR Provider，支持图片和 PDF，并提供两条推理路径：

- Direct Transformers：进程内加载 `ATH-MaaS/OvisOCR2`，作为默认路径。
- OpenAI-compatible vLLM：连接外部 Linux/WSL 服务。

两条路径只负责“输入单页 RGB PIL 图像，返回原始文本”。图片/PDF 逐页处理、失败隔离、重复尾清理、视觉区域处理、路径重写和 Markdown 写盘由 Provider 共享，避免后端产生不同的文件契约。

“输出一致”指相同输入协议、后处理和目录结构，不承诺不同推理引擎逐字生成相同文本。

## 架构边界

不修改 `OCRProvider` 抽象，不复用会缩放图像或插入空 system 消息的通用视觉消息辅助函数，也不使用基类的 OpenAI 后端快捷路径。

```text
OvisOCR2Provider
  -> RGB page iterator
  -> DirectPageInferencer | OpenAIPageInferencer
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
  - `images_kwargs={"min_pixels": 448*448, "max_pixels": 2880*2880}`
- 生成使用 `do_sample=False`、`max_new_tokens=16384`，裁掉输入 token 后解码。
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

## 重复尾清理

严格采用模型卡算法：文本长度至少 8000；周期范围 1 到 200；重复总字符至少 100；重复次数至少 5。只删除检测到的截断式重复尾并保留一个完整周期。

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

## 验证

- 单元测试覆盖重复尾边界、crop/drop、非法及重复 bbox、裁图失败和 PDF 路径重写。
- 两个后端分别验证消息顺序、像素参数、thinking、greedy、未缩放 PNG 和 `extra_body`。
- 共享管线覆盖单图、双页 PDF、部分失败、全部失败、空输出和调试快照失败。
- 接入测试覆盖 Provider catalog、配置镜像、GUI extra、PowerShell 路由和依赖声明。
- 完成前运行相关 pytest、ruff、`git diff --check`、`uv lock --check`，并在可用 GPU 上做一次真实 Direct 单页冒烟。
