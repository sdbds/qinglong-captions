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

## Provider 路由

- 云端 API：在 GUI 填写 Key、模型和 Base URL。
- OpenAI-compatible：使用 `openai_api_key`、`openai_base_url`、`openai_model_name`，详见 [专用指南](../openai_compatible.md)。
- 本地 OCR / VLM / ALM：GUI 根据路由安装 profile，模型配置来自 `config/model.toml`。
- 显式选择本地 `ocr_model` 或 `vlm_image_model` 时，图像请求优先走本地路由。

API Key 当前仍可能进入子进程命令行与任务日志。不要分享完整命令、进程列表或未脱敏日志；`--cloud` 也没有内置鉴权。
