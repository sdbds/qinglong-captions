# 字幕与多模态描述

Caption 是项目的核心页面。它读取媒体目录或 Lance 数据集，按输入类型选择云端 API、OpenAI-compatible 服务、本地 OCR、本地 VLM 或本地 ALM，并把结果写回数据集或字幕文件。

## 入口

```powershell
.\4.captioner.ps1
```

Python 入口：`python -m module.captioner --help`。云端 Provider 使用基础依赖；本地模型会按路由安装对应 extra。

## 推荐流程

1. 视频先运行 Split，图片可直接导入。
2. 需要标签上下文时先运行 Tagger。
3. 在 Caption 页面确认输入、Provider、模型、prompt 和输出模式。
4. 用小批量验证格式、速率和成本，再执行全量任务。
5. 完成后从 Export 页面选择正确 version/tag 导出。

## 输入与调度

- `dataset_path`：Lance 数据集或媒体目录。
- `pair_dir`：可选配对图片目录。
- `mode`：`all`、`short` 或 `long`。
- `wait_time` / `max_retries`：请求节流与失败重试。
- `segment_time`：音频或视频分段长度。
- `scene_detector`、`scene_threshold`、`scene_min_len`：Caption 内部场景检测设置。

指定 `.lance` 时会直接打开该库。传入父目录时优先打开 `dataset.lance`，否则打开唯一的 Lance 库；有多个候选库则要求明确选择。只有没有库时才首次导入。目录里后来新增的素材不会自动混入旧快照；要处理新素材，请在 Import 中明确导入到新库。若原素材缺失，或内容和库中记录的 SHA-256 不同，任务会在调用模型前停止。

分段字幕可以合并手工提供的同名 SRT。没有被项目索引认领的原 SRT 不会被覆盖，新结果会写到另一个文件，重跑不会把上一次生成的字幕再加一遍。OCR 中间文件位于源文件旁的 `.qinglong-ocr-artifacts`，不会被当作新素材；保留旧批次的目录是为了不弄断旧文档的图片链接。

批次结束后会列出写入、策略跳过和失败的数量。退出码 `0` 表示没有失败（可包含策略跳过），`3` 表示有成功项已保存但也有失败项，其他非零值表示失败。失败或跳过的素材不会用库内的旧字幕重新覆盖旁边的手工文件。

## Provider 路由

- 云端 API：在 GUI 填写 Key、模型和 Base URL。
- OpenAI-compatible：使用 `openai_api_key`、`openai_base_url`、`openai_model_name`，详见 [专用指南](../openai_compatible.md)。
- 本地 OCR / VLM / ALM：GUI 根据路由安装 profile，模型配置来自 `config/model.toml`。
- 标准 DeepSeek OCR 2 使用 Transformers 5.x 原生实现；旧模型、非标准裁块参数和显式旧版入口见 [DeepSeek OCR 2 切换与验收](deepseek_ocr2.md)。
- 显式选择本地 `ocr_model` 或 `vlm_image_model` 时，图像请求优先走本地路由。

选择 `vlm_image_model = "moondream"` 默认使用 [Moondream 3.1 9B A2B](https://huggingface.co/moondream/moondream3.1-9B-A2B)。本地 Photon 不需要 API Key；首次运行会缓存约 10.5 GB 的权重和分词器。图片 OCR 使用 `ocr_model = "moondream"` 和 `document_image = true`。要继续使用旧模型，在模型列表选回 **Moondream 3 Preview**，仍由 Transformers 加载。

Windows 版 Photon 的预编译内核使用 CUDA 12。本地运行 3.1 时，默认 GUI 和 PowerShell 启动器会准备独立的 `.runtime_venvs/moondream-photon` 环境（PyTorch 2.11 / CUDA 12.8），不需要降级共享环境或显卡驱动。手动运行 3.1 时也请使用这个环境的解释器。Preview 继续使用原来的依赖环境；通过 OpenAI 兼容服务生成描述、问答或图片 OCR 不会安装 Photon。定位和检测仍需本地推理。

使用本地 Photon 时，图片、提示词和生成结果不会上传。Photon 默认会尝试上报主机名、GPU 信息和汇总用量；`HF_HUB_OFFLINE=1` 只禁止从 Hugging Face 下载，不会关闭这项[运行统计](https://github.com/m87-labs/kestrel#telemetry)。在 GUI 中显式指定其他虚拟环境会把依赖安装到该环境，请使用专用环境。

选择 `vlm_image_model = "lfm_vl_local"` 可用 [LFM3-VL-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-VL-2B-ONNX) 生成单图或配对图片描述。首次运行下载分词器、预处理配置，以及所选精度的两个 ONNX 图和权重，不会下载其他精度。默认在 CPU 上用 `q4`，在 CUDA 上用 `q4f16`；可在 `config/model.toml` 中用 `decoder_variant` 指定其他精度。原生 ONNX 路径关闭思考输出并使用贪心解码，不处理视频。需要复现旧批次时，在模型列表中选回 **LFM 2.5 VL 1.6B ONNX**，并沿用原来的精度和提示词。

LFM3 的所选精度缓存完整后，新进程也能离线启动。先安装或更新 `lfm-vl-local` 依赖，保留分词器、聊天模板、预处理配置、两个图及它们引用的全部权重分片；设置 `HF_HUB_OFFLINE=1` 可禁止联网。切换到未缓存的精度或启用 `force_download` 前需要联网。旧配置中的 `float32`、`base`、`default` 仍指 FP32，不等同于 `auto`。

API Key 当前仍可能进入子进程命令行与任务日志。不要分享完整命令、进程列表或未脱敏日志；`--cloud` 也没有内置鉴权。
