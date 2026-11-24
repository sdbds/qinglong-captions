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

选择 `vlm_image_model = "lfm_vl_local"` 可用 [LFM3-VL-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-VL-2B-ONNX) 生成单图或配对图片描述。首次运行下载分词器、预处理配置，以及所选精度的两个 ONNX 图和权重，不会下载其他精度。默认在 CPU 上用 `q4`，在 CUDA 上用 `q4f16`；可在 `config/model.toml` 中用 `decoder_variant` 指定其他精度。原生 ONNX 路径关闭思考输出并使用贪心解码，不处理视频。需要复现旧批次时，在模型列表中选回 **LFM 2.5 VL 1.6B ONNX**，并沿用原来的精度和提示词。

API Key 当前仍可能进入子进程命令行与任务日志。不要分享完整命令、进程列表或未脱敏日志；`--cloud` 也没有内置鉴权。
