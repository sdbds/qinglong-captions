# 配置指南

本项目有两条配置路径：GUI 配置和 PowerShell 脚本顶部的 `Configuration` 区域。两者最终都会把参数传给同一组 Python CLI。日常使用优先用 GUI；需要可复现批处理时，再把同一组值写入脚本。

## 配置文件

| 文件 | 作用 | 建议 |
| --- | --- | --- |
| `config/config.toml` | Provider、生成参数和工具默认值 | 修改前先备份；按功能查找对应 section |
| `config/model.toml` | Provider 路由、模型 ID、runtime 默认值 | 本地模型路由的主要来源 |
| `config/general.toml` | 通用路径、目录名、批处理默认值 | 适合设置项目级默认值 |
| `config/onnx.toml` | ONNX Runtime、execution provider、缓存和 session | 只在确认工具需要时调整 |
| `config/task_tabs.toml` | GUI 任务标签页的 runtime / 并发设置 | 影响后台任务隔离和资源占用 |
| `config/prompts.toml` | Caption prompt 和模板 | 修改后先用小数据集验证 |
| `config/env_vars.json` | GUI 保存的环境变量 | 明文本地状态，不提交、不上传 |

配置路径使用仓库相对路径时，请从仓库根目录启动脚本；GUI 会自行切换到项目根目录。

## 环境变量

GUI `Setup` 页面可管理以下常用变量：

| 变量 | 用途 |
| --- | --- |
| `HF_HOME` | Hugging Face 模型缓存目录，默认 `huggingface` |
| `HF_ENDPOINT` | Hugging Face 镜像或自建 endpoint |
| `HF_TOKEN` | gated / 私有模型访问令牌 |
| `CUDA_VISIBLE_DEVICES` | 限制可见 GPU，例如 `0` 或 `0,1`；`-1` 表示 CPU |
| `UV_INDEX_URL` | uv 主 PyPI 索引 |
| `UV_EXTRA_INDEX_URL` | PyTorch / ONNX 等额外索引 |
| `UV_CACHE_DIR` | uv 下载缓存目录 |
| `HTTP_PROXY` / `HTTPS_PROXY` | 下载和 API 请求代理 |

`config/env_vars.json` 由 GUI 写入 JSON。它没有加密能力，令牌泄露后应立即在对应服务撤销并重新生成。更安全的做法是通过当前 shell 的环境变量注入令牌，而不是把令牌写进脚本或配置文件。

## Provider 配置

### 云端 API

在 GUI `Caption` 页面填写 API Key、模型和 Base URL。脚本模式则修改 `4.captioner.ps1` 顶部变量，例如：

```powershell
$gemini_api_key = $env:GEMINI_API_KEY
$gemini_model_path = "gemini-3-pro-preview"
```

不要把真实 Key 写入提交的脚本。当前部分兼容入口仍会把 Key 作为子进程参数传递，因此也不要分享完整命令行或任务日志。

### OpenAI-compatible

通用配置使用：

```text
openai_base_url  = http://127.0.0.1:8000/v1
openai_model_name = <server-side model name>
openai_api_key   = <local placeholder or server key>
```

支持 vLLM、SGLang、Ollama、LM Studio 和其他兼容服务。完整启动与请求示例见 [OpenAI-compatible 指南](openai_compatible.md)。本地服务如果只绑定 `127.0.0.1`，不要把 GUI 或服务改成公网监听。

### 本地 OCR / VLM / ALM

在 GUI 选择路由后，任务页会根据 `config/model.toml` 的模型 ID 和路由映射补齐 `uv extra`。常用类别：

- OCR：`logics_ocr`、`lighton_ocr`、`dots_ocr`、`paddle_ocr`、`unlimited_ocr` 等
- VLM：`qwen_vl_local`、`step_vl_local`、`gemma4_local`、`marlin_2b_local` 等
- ALM：`music_flamingo_local`、`eureka_audio_local`、`acestep_transcriber_local`、`cohere_transcribe_local`、`mega_asr_local`

不同 extra 可能存在依赖冲突。不要一次性安装所有可选 profile；让 GUI 按当前任务安装，或只在隔离虚拟环境中手工测试。

## Lance 与标签

- `lanceImport.ps1` 顶部的 `data_storage_version` 控制新建数据集格式；当前默认是 `2.2`。
- `tag` 用于区分导入或字幕版本；导出前确认 `lanceExport.ps1` 的 `version` 与目标标签一致。
- 翻译链路会生成 `raw.import.*`、`norm.docling.*` 和 `tr.<model>.<lang>.*` 标签，并默认写出语言后缀 Markdown。
- 只在确认备份和版本号后执行批量更新；Lance 文件夹不要与正在运行的任务共享写入目录。

## 依赖 profile

基础安装只处理 `pyproject.toml` 的默认依赖。脚本和 GUI 会按任务增量安装下列 profile：

| Profile | 主要用途 |
| --- | --- |
| `video-split` | 视频场景检测 |
| `wdtagger` | WDTagger / CL Tagger |
| `image-align` | 图片预处理和特征对齐 |
| `reward-model` | 图像质量评分 |
| `translate` | 文本规范化和翻译 |
| `vocal-midi` | ONNX 音频分轨及 MIDI 辅助路径 |
| `psdexport` | PSD 图层读取和导出 |
| `musvit-onnx` | MuSViT ONNX 乐谱扫描 |
| `see-through` | Image2PSD / See-through |
| `onnx-base` | ONNX Runtime GPU 基础依赖 |

日常运行直接从 `pyproject.toml` 解析 profile，不要求仓库存在 `uv.lock`。要复现 CI 或发行环境，应在独立环境中固定 Python、CUDA、PyTorch 和 profile 版本。
