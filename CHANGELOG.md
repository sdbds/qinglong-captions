# 更新日志

版本历史从 README 拆出，避免发行说明和操作手册互相遮挡。当前操作入口请看 [README.md](README.md)。

## 4.6.0 - MuScriptor 全分轨 MIDI 与试听

1. 将 README 的发行说明、配置和故障排查拆分到独立文档，并重写 Windows/Linux、GUI、脚本和 Python CLI 的使用路径。
2. 补充可选依赖 profile、Linux x86_64 PowerShell 安装限制、OpenAI-compatible 参数边界和 GUI/脚本配置优先级说明。
3. PowerShell 入口在 Linux/无 `LOCALAPPDATA` 环境下使用 `HOME` 或仓库本地目录作为 `UV_CACHE_DIR`；Linux PowerShell helper 改用可移植的 Bash/POSIX 检测，并忽略 GUI 生成的明文 `config/env_vars.json`。
4. 音频分轨新增 MuScriptor 全分轨转 MIDI 二级选项：按分轨标签约束音色家族，支持为 `other` 自动识别或手动限定音色，并复用一次模型加载处理六轨；可为每轨生成纯 MIDI 或左右声道对照试听。
5. 修复 MuScriptor 试听首次下载官方 SoundFont 时缺少 SOCKS 代理支持的问题，并在分轨 MIDI 失败汇总中直接显示共同根因。
6. 补充 MuScriptor 官方论文 [MuScriptor: An Open Model for Multi-Instrument Music Transcription](https://arxiv.org/abs/2607.08168) 的上游说明与 BibTeX 引用。

## 4.5 - 新增 OCR Provider、Grok Build 订阅与 GUI 任务标签页

1. 新增 `unlimited_ocr`，基于 `baidu/Unlimited-OCR` 3B 模型，支持图片与 PDF OCR，并提供 `gundam` / `base` 图像模式。
2. 新增 `infinity_parser2` OCR Provider。
3. 新增 `grok_build_subscription` 订阅 Provider，通过无头浏览器自动化使用 Grok Build 生成字幕。
4. PaddleOCR 迁移到 PP-OCRv6 ONNX 后端，统一走 ONNX Runtime 配置。
5. GUI 新增任务标签页，支持隔离运行时并发执行多个任务。
6. `gemma4_local` 新增 12B 本地模型支持；Kimi Code 新增 thinking mode。
7. 云端 Provider 新增 image-only 并发模式；图片预处理新增 resize 限制并优化 OpenCV GPU 初始化。
8. Lance 数据集导入改为流式处理，降低内存占用。
9. Codex 订阅 Provider 增强超时清理、客户端重试、MCP 启动隔离、传输和评分默认值。
10. Grok Build 支持 image MIME 转码；新增全局目录名上下文。
11. 依赖安装默认使用阿里云 PyPI 镜像；Provider 声明重构并统一 `CaptionResult` 返回类型。
12. 修复 Unlimited-OCR Markdown 缺少硬换行导致预览不换行的问题。

## 4.4 - 本地模型、GUI 与 WDTagger 更新

1. 新增 `marlin_2b_local`、`mega_asr_local` 和 `codex_subscription` Provider。
2. 本地运行时更新到显式 CUDA 13 安装源；`onnx-base` 增加 TensorRT CUDA 13；`translate` 升级 Transformers 5.6+ 并补齐 `compressed-tensors`。
3. `translate` 默认模型更新为 `tencent/Hy-MT2-7B`，保留轻量 FP8 变体。
4. CL Tagger 默认版本更新为 `v2_01a`，支持动态 tag category，并修复 gated 模型访问与 Lance caption 扫描路径。
5. Lance Blob v2 读写逻辑重做；默认导入不再保存原始二进制 blob，以降低数据集体积。
6. GUI 改为从 `pyproject.toml` 读取版本号，新增本地模型显存适配提示、多 GPU 摘要、see-through 显存推荐、共享标签本地化、日志视图和任务列表更新。
7. PowerShell 入口改为按当前 profile 从 `pyproject.toml` 增量安装，不再强依赖旧的 `uv.lock`。
8. Reward model 独立为 `reward-model` extra；新增全局 `image_quality` 配置；修复 Qwen 本地模型、Kimi Code、Marlin-2B、see-through 等路径的稳定性问题。
9. 测试覆盖扩展到 Codex subscription、Marlin-2B、Mega-ASR、WDTagger、GPU profile、运行时依赖冲突、Caption pipeline、Lance Blob、文本翻译和 GUI i18n。

## 4.3 - Image2PSD / See-through

1. 新增 `module/see_through/cli.py` 和 `2.6.image2psd.ps1`。
2. 主流程为 `LayerDiff 透明分层 -> Marigold 深度估计 -> PSD 导出`。
3. GUI 与 CLI 支持透传 `inference_steps_depth`、`seed` 等关键推理参数。
4. README 增加 See-through、Marigold 和 GAME 上游引用说明。

## 4.2 - ONNX Runtime 统一与音频工具

1. 新增 `audio_separator.py` / `2.5.audio_separator.ps1`，支持单文件、目录和 `.lance` 输入，默认输出 BS-RoFormer ONNX 六 stems，可选 harmony 二次分离。
2. 新增 `config/onnx.toml`，统一 WDTagger、WaterDetect 和本地 VLM 的 runtime、session 与缓存逻辑。
3. 新增 `logics_ocr`，支持图片和 PDF OCR，并将结构化 HTML 规整为 Markdown。
4. GUI 工具箱补齐音频分轨和文本翻译入口；依赖 profile 做收敛，音频工具统一走 `vocal-midi` profile。

## 4.1 - 文档 / 纯文本翻译工具

1. 新增 `texttranslate.py`，支持使用本地 Hugging Face 翻译模型处理文本和文档。
2. Lance 数据集新增 `chunk_offsets`，记录规范化 Markdown 的分块边界。
3. `.txt` / `.md` 可作为 standalone 文本资产导入，同时继续支持媒体 sidecar。
4. 新增 `raw.import.*`、`norm.docling.*` 和 `tr.<model>.<lang>.*` 版本标签。
5. 新增 `5.translate.ps1`，输出带语言后缀的 Markdown，例如 `foo_zh_cn.md`，不会覆盖原文件。
6. 新增 `reka_edge_local`、`lighton_ocr`、`eureka_audio_local`、`acestep_transcriber_local`、`cohere_transcribe_local` 和 `mega_asr_local` 路径。
7. 运行时脚本直接从 `pyproject.toml` 解析当前 profile 并增量安装，不再强依赖全局 `uv.lock`。

## 4.0 - Provider V2 架构重构

1. 重构 Provider V2，统一 Cloud VLM、Local VLM、OCR 和 Vision API 抽象，支持自动发现与优先级路由。
2. 统一 `CaptionResult` 返回类型，减少 Provider 返回值多态。
3. 新增 OpenAI-compatible Provider，支持 vLLM、SGLang、Ollama、LM Studio 和其他兼容服务。
4. JSON 模式不支持时可自动降级为文本模式；详细配置见 [docs/openai_compatible.md](docs/openai_compatible.md)。

## 3.x 及更早版本

- **3.9**：新增 MiniMax API 与 MiniMax Code 多模态 Provider。
- **3.8**：新增 FireRed-OCR。
- **3.7**：新增 GLM-OCR 和 Nanonets-OCR2-3B。
- **3.6**：支持 Kimi 2.5，并更新多模型与 OCR/VLM 参数示例。
- **3.5**：支持 Step3-VL 10B，更新 short/long 模板。
- **3.4**：支持 PSD exporter。
- **3.3**：支持 HunyuanOCR。
- **3.2**：支持 DeepSeek OCR 和 PaddleOCR。
- **3.1**：增加 third_party SongPrep，用于音乐字幕 / 描述。
- **3.0**：支持 tagger JSON、分类 tags.json、image reward model 和 nano banana 多输入多输出任务。
