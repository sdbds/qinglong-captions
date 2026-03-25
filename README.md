[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N1NOO2K)

<details>
<summary>中文说明（点击展开）</summary>

# 青龙字幕工具 (4.2.0)

## 更新日志

### 4.2 - ONNX Runtime 统一与音频工具

1. 新增独立音频分轨工具链 `audio_separator.py` / `2.5.audio_separator.ps1`：
   - 支持单文件、目录和 `.lance` 数据集输入
   - 默认使用 BS-RoFormer ONNX 模型输出 6 stems
   - 可选二次 harmony 分离，并已接入 GUI 工具箱
2. 统一仓库内 ONNX Runtime 配置：
   - 新增 `config/onnx.toml`
   - `wdtagger`、`waterdetect` 与 `lfm_vl_local` 共享 runtime / session / cache 逻辑
   - 补齐 artifact 下载日志与 legacy 配置兼容
3. 新增本地 OCR `logics_ocr`：
   - 默认模型：`Logics-MLLM/Logics-Parsing-v2`
   - 支持图片与 PDF OCR
   - 自动把结构化 HTML 输出规整为 Markdown
4. GUI 工具箱补齐音频分轨与文本翻译入口，文档同步到当前脚本命名（`1.install-uv-qinglong.ps1`、`2.0.video_spliter.ps1`、`5.translate.ps1`）。
5. 依赖 profile 做了一轮收敛：
   - `onnx-base` 现在直接包含 `torch-base`，避免 ONNX GPU 路径单独漏装 `torch`
   - `wdtagger` extra 去掉了未使用的 `transformers`，Windows 上的 OpenCV 轮子改为运行时判定
   - `lfm-vl-local` 继续保留 `transformers`（当前 `AutoProcessor` 仍需要），但移除了额外的 Windows `flash-attn` 依赖
   - `2.5.audio_separator.ps1` 与 GUI 工具箱现在统一走 `vocal-midi` dependency profile

### 4.1 - 文档 / 纯文本翻译工具

1. 新增独立的 `texttranslate.py` 工具链，支持使用本地 Hugging Face 翻译模型处理文本和文档。
2. Lance 数据集新增 `chunk_offsets` 列，用于记录规范化 Markdown 的分块边界，便于复现与重跑。
3. 文本导入新增 standalone 文本资产判定：
   - `.txt/.md` 可作为主资产导入
   - `.txt/.md/.srt` 仍可作为同 stem 媒体 / 文档的 sidecar
4. 新增文档规范化与翻译导出流程：
   - 原始版本 tag：`raw.import.*`
   - 规范化版本 tag：`norm.docling.*`
   - 翻译版本 tag：`tr.<model>.<lang>.*`
5. 新增 `5.translate.ps1`，翻译结果导出为语言后缀 Markdown，例如 `foo_zh_cn.md`，不会覆盖原文件。
6. 新增本地 VLM `reka_edge_local`：
   - 默认模型：`RekaAI/reka-edge-2603`
   - 支持图像和视频输入
   - 支持直接 Transformers 推理，或复用 OpenAI-compatible 本地服务路径
7. 新增本地 OCR `lighton_ocr`：
   - 默认模型：`lightonai/LightOnOCR-2-1B`
   - 支持图片与 PDF OCR
   - 依赖 `transformers>=5`
8. 运行时脚本现在直接从 `pyproject.toml` 解析当前所选 `extra` / `group` 并增量安装，不再强依赖全局 `uv.lock`。
9. 新增可选本地 ALM `eureka_audio_local`：
   - 默认模型：`cslys1999/Eureka-Audio-Instruct`
   - 支持 `audio/*` 输入
   - 基于官方 `AutoModelForCausalLM + AutoProcessor` 推理路径
   - 当前 extra 固定依赖 `transformers==5.2.0`
10. 新增可选本地 ALM `acestep_transcriber_local`：
   - 默认模型：`ACE-Step/acestep-transcriber`
   - 支持 `audio/*` 输入
   - 默认输出结构化 `.txt` 转写文本

### 4.0 - Provider V2 架构重构

1. **全新 Provider V2 架构** - 完全重构的模块化 Provider 系统
   - 统一抽象的 `Provider` 基类，支持 Cloud VLM、Local VLM、OCR、Vision API 四大类别
   - 自动发现机制，通过装饰器自动注册 Provider
   - 统一的 `CaptionResult` 返回类型，解决返回值多态问题
   - 基于优先级的 Provider 路由，自动选择最佳 Provider

2. **新增 OpenAI Compatible Provider** - 通用 OpenAI API 兼容接口
   - 支持对接任何 OpenAI 兼容服务：vLLM、SGLang、Ollama、LM Studio
   - 统一配置参数：`openai_base_url`、`openai_model_name` 等
   - 自动降级：JSON 模式不支持时自动切换到文本模式
   - 支持本地 GPU 部署（Qwen2-VL、LLaVA、MiniCPM-V 等）
   - 查看 [docs/openai_compatible.md](docs/openai_compatible.md) 获取详细使用指南

<details>
<summary>更早更新日志（&lt; 4.0）</summary>

### 3.9

1. **新增 MiniMax API 支持** - 集成 MiniMax 开放平台多模态能力
   - 支持模型：MiniMax-M2.5、MiniMax-M2.5-highspeed、MiniMax-M2.1、MiniMax-M2.1-highspeed、MiniMax-M2
   - 支持图像和视频理解
   - 支持 Tags 自动加载和高亮显示（与 Kimi VL 相同的体验）
   - 配置参数：`--minimax_api_key`、`--minimax_model_path`、`--minimax_api_base_url`

2. **新增 MiniMax Code 支持** - 针对代码和结构化输出优化的 Provider
   - 默认使用 MiniMax-M2 模型，专为代码理解和 Agent 工作流优化
   - 支持 reasoning_split 分离推理过程
   - 更强的 JSON 结构化输出能力
   - 支持 Tags 自动加载和高亮显示
   - 配置参数：`--minimax_code_api_key`、`--minimax_code_model_path`、`--minimax_code_base_url`

### 3.8

1. 新增 FireRed-OCR (FireRedTeam/FireRed-OCR) 支持，基于 Qwen3-VL 的高性能文档解析 OCR 模型。

### 3.7

1. 新增 GLM-OCR (zai-org/GLM-OCR) 支持，用于从图像和文档中识别文本。
2. 新增 Nanonets-OCR2-3B 支持，用于文档转换为 Markdown。

### 3.6

1. 支持 Kimi 2.5 作为图像描述模型。
2. 更新脚本参数示例，补充多模型与 OCR/VLM 选项。

### 3.5

1. 支持 Step3-VL 10B。
2. 更新 short/long 模板。

### 3.4

1. 支持 PSD exporter。
2. 更新 short/long 模板。

### 3.3

1. 支持 HunyuanOCR。
2. 更新 processor 配置（use fast false）。

### 3.2

1. 支持 DeepSeek OCR 和 PaddleOCR。
2. 补齐缺失依赖。

### 3.1

1. 增加 third_party SongPrep，用于音乐字幕/描述。
2. 更新 submodule。

### 3.0

1. 支持 tagger JSON 格式输出，并生成分类后的 tags.json。
2. 新增 image_reward_model（imscore）脚本。
3. 支持 nano banana 图片编辑/处理任务（多输入多输出）。

</details>

基于 Lance 数据库的多模态数据处理与字幕工具，支持 GUI 驱动的视频 / 图像 / 音频描述、OCR、翻译、音频分轨，以及云端 / 本地 Provider 路由。

## 功能特点
- **Provider V2 架构** - 模块化、可扩展的 Provider 系统，支持自动发现和统一接口
- **OpenAI 兼容 API** - 通用接口支持 vLLM、SGLang、Ollama、LM Studio 本地 GPU 推理
- **GUI 优先工作流** - NiceGUI 图形界面统一承载导入、分镜、打标、字幕、导出和工具箱
- 支持使用云端或本地 Provider 进行视频 / 图像 / 音频理解
- 导出 SRT 格式字幕文件
- 支持多种视频格式
- 批量处理并显示进度
- 保持原始目录结构
- 通过 TOML 文件配置
- 集成 Lance 数据库实现高效数据管理
- 新增 ONNX 音频分轨工具，支持 6 stems 分离与 harmony 二次分离
- 统一 ONNX runtime 配置，支持共享缓存和 provider 选项
- 新增独立文本 / 文档翻译链路，支持 txt、md、json、pdf、doc/docx、xls/xlsx、ppt/pptx、rtf、epub
## 模块说明

### 数据集导入 (`lanceImport.py`)
- 将视频导入 Lance 数据库格式
- 保持原始目录结构
- 支持单目录和配对目录结构

### 数据集导出 (`lanceexport.py`)
- 从 Lance 数据集中提取视频和字幕
- 保持原有文件结构
- 在源视频所在目录导出 SRT 格式字幕

### 自动字幕生成 (`captioner.py` & `api_handler_v2.py`)
- **Provider V2 架构**，支持 20+ 种 Provider（Cloud VLM、Local VLM、OCR、Vision API）
- **OpenAI 兼容 Provider**，支持本地推理（vLLM、SGLang、Ollama、LM Studio）
- 支持远程 API、本地 OCR / VLM / ALM 与 OpenAI-compatible 路由
- 支持批量处理
- 生成带时间戳的 SRT 格式字幕
- 健壮的错误处理和重试机制
- 批处理进度跟踪

### 文本 / 文档翻译 (`texttranslate.py`)
- 使用 Lance 版本控制保存原始导入、规范化 Markdown、翻译结果
- standalone `.txt/.md` 可直接导入为主资产
- 文档先规范化为 Markdown，再按 `chunk_offsets` 分块喂给本地翻译模型
- 默认本地模型：`tencent/HY-MT1.5-7B`
- 导出结果统一为 `*_lang.md`，避免覆盖原文件

### 音频分轨 (`audio_separator.py`)
- 支持音频文件、目录与 `.lance` 数据集输入
- 默认输出 6 stems，可选追加 harmony 二次分离
- 支持 `wav` / `flac` / `mp3` 导出，适合伴奏、人声和和声拆分

### 配置模块 (`config.py`、`config.toml` & `config/onnx.toml`)
- API 配置管理
- 可自定义批处理参数
- 支持 ONNX runtime 默认值与按工具覆写
- 默认结构包含文件路径和元数据

## 安装方法

### Windows 系统
运行以下 PowerShell 脚本：
```powershell
./1.install-uv-qinglong.ps1
```
### Linux 系统
1. 首先安装 PowerShell：
```bash
sudo sh ./0、install pwsh.sh
```
2. 然后使用 PowerShell 运行安装脚本：
```powershell
pwsh ./1.install-uv-qinglong.ps1
```
## 使用方法

### 推荐流程：先用 GUI

1. 运行安装脚本：
   ```powershell
   ./1.install-uv-qinglong.ps1
   ```
2. 启动 GUI：
   ```powershell
   ./start_gui.ps1
   ```
3. 默认会在浏览器打开 `http://127.0.0.1:7899`。
4. 进入 GUI 后，建议先打开 `Setup` 页面检查 Python / PyTorch / CUDA，然后按 `Import -> Split -> Tagger -> Caption -> Export / Tools` 的顺序使用。

### GUI 启动方式

#### 方式 1：推荐，使用项目脚本

```powershell
./start_gui.ps1
```

- 这个脚本会自动切到项目根目录，补齐 `PYTHONPATH`，优先复用 `.venv` / `venv`
- 默认参数来自 `start_gui.ps1` 内的 `$Config`，当前默认地址是 `127.0.0.1:7899`
- 默认以浏览器模式启动；如果端口占用，GUI 会自动尝试后续端口并打印实际 URL

#### 方式 2：直接调用 Python 入口

```powershell
python -m gui.launch --port 7899
```

常用参数：

```powershell
python -m gui.launch --cloud --port 7899 --no-browser
python -m gui.launch --native --port 7899
```

- `--cloud` 会绑定到 `0.0.0.0`
- `--native` 会使用原生窗口模式（需要 `pywebview`）
- `--no-browser` 不自动打开浏览器

更多 GUI 页面说明可见 [gui/README.md](gui/README.md)。

### GUI 下的模型 / Provider 使用说明

- 远程 API、OCR、本地 VLM、本地 ALM 现在都统一从 GUI 配置
- 在 `Caption` 页面选择本地 OCR / VLM / ALM 后，GUI 会根据所选路由自动补对应的 `uv extra`
- 不再推荐手动执行一长串 `uv sync --extra xxx` 来安装不同本地 VLM / ALM
- 如果你确实走脚本模式，再去修改对应 `.ps1` 或 `config/*.toml`

### 脚本模式（高级 / 批处理）

如果你已经熟悉当前工程，仍然可以直接运行脚本：

```powershell
./lanceImport.ps1
./4、run.ps1
./lanceExport.ps1
./2.2.preprocess_images.ps1
./2.5.audio_separator.ps1
./5.translate.ps1
```

说明：

- `4、run.ps1` 用于批量字幕生成
- `2.2.preprocess_images.ps1` 用于图片预处理与可选图像对齐；`--matcher-backend=auto` 会在 CUDA 上优先 `affine_steerers`，否则优先 `xfeat`，失败时回退 ORB
- `2.5.audio_separator.ps1` 用于 ONNX 音频分轨
- `2.5.audio_separator.ps1` 默认会安装并复用 `vocal-midi` profile，不需要再手动补 `--extra vocal-midi`
- `5.translate.ps1` 用于文档规范化和翻译，输出如 `*_zh_cn.md`
- 日常运行会按当前所选 profile 增量安装依赖；`uv.lock` 主要留给 CI / 发版流程维护

</details>

# qinglong-captioner (4.2.0)

A multimodal toolkit built on Lance for GUI-driven captioning, OCR, translation, audio separation, and cloud / local provider routing.

## Changelog

### 4.2 - Unified ONNX Runtime And Audio Tools

1. Added a dedicated audio separation pipeline: `audio_separator.py` / `2.5.audio_separator.ps1`.
   - Accepts a single file, a directory, or a `.lance` dataset
   - Uses the BS-RoFormer ONNX export for 6-stem separation by default
   - Can optionally run a second harmony split and is now exposed in the GUI toolbox
2. Unified ONNX Runtime configuration across the repository.
   - Added `config/onnx.toml`
   - `wdtagger`, `waterdetect`, and `lfm_vl_local` now share runtime / session / cache logic
   - Added artifact download logging and legacy config fallback
3. Added local OCR `logics_ocr`.
   - Default model: `Logics-MLLM/Logics-Parsing-v2`
   - Supports image and PDF OCR
   - Converts the model's structured HTML output into project-friendly Markdown
4. The GUI toolbox now includes audio separation and text translation, and the docs were updated to match the current script names (`1.install-uv-qinglong.ps1`, `2.0.video_spliter.ps1`, `5.translate.ps1`).
5. Dependency profiles were tightened up:
   - `onnx-base` now includes `torch-base`, so the ONNX GPU path no longer misses `torch`
   - the `wdtagger` extra no longer pulls an unused `transformers` dependency, and Windows OpenCV selection is now resolved at runtime
   - `lfm-vl-local` still keeps `transformers` because the current implementation uses `AutoProcessor`, but the extra Windows `flash-attn` wheel was removed
   - `2.5.audio_separator.ps1` and the GUI toolbox now use the `vocal-midi` dependency profile by default

### 4.1 - Text / Document Translation Tool

1. Added a standalone `texttranslate.py` pipeline for text and document translation with local Hugging Face models.
2. Lance datasets now carry a `chunk_offsets` column for reproducible markdown chunk boundaries.
3. Standalone `.txt/.md` assets can be imported as primary assets, while `.txt/.md/.srt` still work as same-stem sidecars.
4. Added `5.translate.ps1` and suffix-based markdown export such as `foo_zh_cn.md`.
5. Added local VLM `reka_edge_local`:
   - Default model: `RekaAI/reka-edge-2603`
   - Accepts both image and video inputs
   - Works with direct Transformers inference or an OpenAI-compatible local server
6. Added local OCR `lighton_ocr`:
   - Default model: `lightonai/LightOnOCR-2-1B`
   - Supports image and PDF OCR
   - Requires `transformers>=5`
7. Runtime scripts now install only the selected `extra` / `group` directly from `pyproject.toml`, while `uv.lock` is reserved for CI or release workflows.
8. Added optional local ALM `eureka_audio_local`:
   - Default model: `cslys1999/Eureka-Audio-Instruct`
   - Supports `audio/*` inputs
   - Uses the official `AutoModelForCausalLM + AutoProcessor` inference path
   - The current extra pins `transformers==5.2.0`
9. Added optional local ALM `acestep_transcriber_local`:
   - Default model: `ACE-Step/acestep-transcriber`
   - Supports `audio/*` inputs
   - Defaults to structured `.txt` transcript output

### 4.0 - Provider V2 Architecture Refactoring

1. **Brand New Provider V2 Architecture** - Fully refactored modular Provider system
   - Unified abstract `Provider` base class supporting Cloud VLM, Local VLM, OCR, and Vision API
   - Auto-discovery mechanism with decorator-based Provider registration
   - Unified `CaptionResult` return type resolving polymorphic return value issues
   - Priority-based Provider routing with automatic best Provider selection

2. **New OpenAI Compatible Provider** - Universal OpenAI API compatible interface
   - Support any OpenAI-compatible service: vLLM, SGLang, Ollama, LM Studio
   - Unified configuration: `openai_base_url`, `openai_model_name`, etc.
   - Auto-fallback: Automatically switches to text mode when JSON mode is unsupported
   - Support local GPU deployment (Qwen2-VL, LLaVA, MiniCPM-V, etc.)
   - See [docs/openai_compatible.md](docs/openai_compatible.md) for detailed usage guide

<details>
<summary>Older changelog (&lt; 4.0)</summary>

### 3.9

1. **Added MiniMax API Support** - Integrated MiniMax platform multimodal capabilities
   - Supported models: MiniMax-M2.5, MiniMax-M2.5-highspeed, MiniMax-M2.1, MiniMax-M2.1-highspeed, MiniMax-M2
   - Support for image and video understanding
   - Tags auto-loading and highlight display (same experience as Kimi VL)
   - Configuration: `--minimax_api_key`, `--minimax_model_path`, `--minimax_api_base_url`

2. **Added MiniMax Code Support** - Provider optimized for coding and structured output
   - Default MiniMax-M2 model optimized for code understanding and Agent workflows
   - Support reasoning_split to separate reasoning process
   - Enhanced JSON structured output capability
   - Tags auto-loading and highlight display
   - Configuration: `--minimax_code_api_key`, `--minimax_code_model_path`, `--minimax_code_base_url`

### 3.8

1. Added support for FireRed-OCR (FireRedTeam/FireRed-OCR), a high-performance document parsing OCR model based on Qwen3-VL.

### 3.7

1. Added support for GLM-OCR (zai-org/GLM-OCR) for text recognition from images and documents.
2. Added support for Nanonets-OCR2-3B for document-to-markdown conversion.

### 3.6

1. Support Kimi 2.5 for image captioning.

### 3.5

1. Support Step3-VL 10B.
2. Update for short and long template.

### 3.4

1. Support PSD exporter.
2. Update for short and long template.

### 3.3

1. Day 0 support HunyuanOCR.
2. Update for processor use fast false.

### 3.2

1. Support DeepSeek OCR and PaddleOCR.
2. Update missing deps.

### 3.1

1. Add third_party SongPrep for music captions.
2. Commit submodule changes.

### 3.0

1. We support tagger JSON format files, and now a tags.json file will be generated in the root directory of the data after marking, which will be classified according to tag categories
 
<img width="681" height="694" alt="image" src="https://github.com/user-attachments/assets/beb6a383-5144-49d4-b128-ba516525b55c" />

2. We have added image_deward_madel.ps1! Using [imscore](https://github.com/RE-N-Y/imscore) as an interface to call many aesthetic and performance models!

<img width="1006" height="882" alt="image" src="https://github.com/user-attachments/assets/7e8d0aba-d677-482a-91fe-1f89d3603210" />


3. We have supported nano banana as a new image editing and processing task, and it supports multiple inputs and outputs.

<img width="714" height="231" alt="image" src="https://github.com/user-attachments/assets/c4b64472-d364-469b-bc25-258bc68ea073" />

If the prompt indicates outputting multiple images, they will also be saved separately and the corresponding text content will be saved.
If you add pair-dir, you can input more images for multimodal context interleaving!

### 2.9

We now support and use CL_tagger as the default best tagger model.

What is cl_tagger?

CL EVA02 Tagger model (ONNX), fine-tuned from SmilingWolf/wd-eva02-large-tagger-v3 by cella.

Compared to wd-eva02-large-tagger-v3, cl tagger expands the total number of tags from 20,000 to over 40,000.

Added quality tags, meta tags, model tags and support for photos (cosplay) recognition.

<img width="901" height="839" alt="image" src="https://github.com/user-attachments/assets/f29312b6-ade0-499e-a2e5-5ecdb4b1022a" />

<img width="800" height="1200" alt="image" src="https://github.com/user-attachments/assets/f5b5dfc9-28f9-4168-8786-66383c4b607e" />


offical code:
https://github.com/celll1/tagutl

HF Space demo:
https://huggingface.co/spaces/cella110n/cl_tagger

### 2.8
<img width="1078" height="708" alt="image" src="https://github.com/user-attachments/assets/8374d156-1221-41e4-8d72-925a54782dfc" />

We have added support for the `gemini-2.5-pro` model for pair image captions. This allows for more accurate and detailed descriptions of pair of images.

**How to use:**
1. Open the `4、run.ps1` script.
2. Set your Gemini API key in the `$gemini_api_key` variable.
3. Set the model path to `gemini-2.5-pro`: `$gemini_model_path = "gemini-2.5-pro"`(pro can do NSFW images,flash only sfw images.)
4. Place the edited images you want to caption in the folder specified by `$dataset_path`.
5. Place the original images you want to caption in the folder specified by `$pair_dir`(with same image name).
6. Run the script: `./4、run.ps1`



### 2.7

We've added a script for batch image datasets! It includes pre-scaling resolution and alignment functionality for image pairs!
If only one path is entered, it will only process the size of the images, and you can set the maximum values for the longest and shortest edges to scale!

If two paths are entered, it will process image pairs, used for training matching of some editing models.

### 2.6
![image](https://github.com/user-attachments/assets/34f8150b-3414-4e0c-9ade-b9406cd1602b)

A new watermark detection script has been added, initially supporting two watermark detection models, which can quickly classify images in the dataset into watermarked/unwatermarked categories.
It will generate two folders, and data separation is done through symbolic links. If needed, you can copy the corresponding folder to transfer data without deleting it, and it does not occupy additional space.
(As symbolic links require permissions, you must run PowerShell as admin.)

Finally, it will generate a JSON file report listing the watermark detection results for all images in the original path, including detection values and results.
The watermark threshold can be modified in the script to correspondingly change the detection results.


### 2.5
![image](https://github.com/user-attachments/assets/bffd2120-6868-4a6e-894b-05c4ff5fd98f)

We officially support the tags highlight captions feature! Currently unlocked in the pixtral model, and we are considering adding it to other models such as gemini in the future.

What are tags highlight?

As is well known, non-state-of-the-art VLMs have some inaccuracies, so first use wdtagger for tags annotation, and then input the tags annotation to the VLM for assistance, which can improve accuracy.

Currently, the tags have been categorized, and it is also possible to quickly check the annotation quality (e.g., purple is for character names and copyright, red is for clothing, brown is for body features, light yellow is for actions, etc.)

The annotation quality obtained in the end is comparable to some closed-source models!

Additionally, we have added check parameters, which can specify the parent folder as the character name to designate the character's name, as well as specify the check for the tags highlight rate. Generally, good captions should have a highlight rate of over 35%.

You can also specify different highlight rates to change the default standard.

How to use？ just use 3、tagger.ps1 first for generate tags for your image datasets,

then use 4、run.ps1 with pixtral apikey

### 2.4

We support Gemini image caption and rating.
It also supports gemini2.5-flash-preview-04-17.

However, after testing, the flash version has poor effects and image review, it is recommended to use the pro version

![image](https://github.com/user-attachments/assets/6ae9ed38-e67a-41d2-aa1d-4caf0e0db394)
flash↑

![image](https://github.com/user-attachments/assets/c83682aa-3a37-4198-b117-ffe7f74ff812)
pro ↑

### 2.3

Well, we forgot to release version 2.2, so we directly released version 2.3!

Version 2.3 updated the GLM4V model for video captions

### 2.2

Version 2.2 has updated TensorRT for accelerating local ONNX model WDtagger.

After testing, it takes 30 minutes to mark 10,000 samples with the standard CUDA tag,

while using TensorRT, it can be completed in just 15 to 20 minutes.

However, the first time using it will take a longer time to compile.

If TensorRT fails, it will automatically revert to CUDA without worry.

If it prompts that TensorRT librarys are missing, it may be missing some parts

Please install version 10.7.x manually from [here](https://developer.nvidia.com/tensorrt/download/10x)

### 2.1

Added support for Gemini 2.5 Pro Exp. Now uses 600 seconds cut video by default.

### 2.0 Big Update！

Now we support video segmentation! A new video segmentation module has been added, which detects key timestamps based on scene changes and then outputs the corresponding images and video clips!
Export an HTML for reference, the effect is very significant!
![image](https://github.com/user-attachments/assets/94407fec-92af-4a34-a15e-bc02bf45d2cd)

We have also added subtitle alignment algorithms, which automatically align Gemini's timestamp subtitles to the millisecond level after detecting scene change frames (there are still some errors, but the effect is much better).

Finally, we added the image output feature of the latest gemini-2.0-flash-exp model!

You can customize the task, add the task name in the [`config.toml`](https://github.com/sdbds/qinglong-captions/blob/main/config/config.toml), which will automatically handle the corresponding images (and then label them)

Currently, some simple task descriptions are as follows: Welcome the community to continuously optimize these task prompts and provide contributions!
https://github.com/sdbds/qinglong-captions/blob/12b7750ee0bca7e41168e98775cd95c7b9c57173/config/config.toml#L239-L249

![image](https://github.com/user-attachments/assets/7e5ae1a9-b635-4705-b664-1c20934d12bc)

![image](https://github.com/user-attachments/assets/58527298-34f8-496d-8c4e-1a1c1c965b73)


### 1.9

Now with Mistral OCR functionality!
Utilizing Mistral's advanced OCR capabilities to extract text information from videos and images.

This feature is particularly useful when processing media files containing subtitles, signs, or other text elements, enhancing the accuracy and completeness of captions.

The OCR functionality is integrated into the existing workflow and can be used without additional configuration.

### 1.8

Now added WDtagger！
Even if you cannot use the GPU, you can also use the CPU for labeling.

It has multi-threading and various optimizations, processing large-scale data quickly.

Using ONNX processing, model acceleration.

Code reference@kohya-ss 
https://github.com/sdbds/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py

Version 2.0 will add dual caption functionality, input wdtagger's taggers, then output natural language
![image](https://github.com/user-attachments/assets/f14d4a69-9c79-4ffb-aff7-84d103dfeff4)


### 1.7

Now we support the qwen-VL series video caption model!

- qwen-vl-max-latest
- qwen2.5-vl-72b-instruct 
- qwen2.5-vl-7b-instruct
- qwen2.5-vl-3b-instruct

qwen2.5-vl has 2 seconds ~ 10 mins, qwen-vl-max-latest has 1 min limit.
These models are not good at capturing timestamps; it is recommended to use segmented video clips for captions and to modify the prompts.

Video upload feature requires an application to be submitted to the official, please submit the application [here](https://smartservice.console.aliyun.com/service/create-ticket?spm=a2c4g.11186623.0.0.3489b0a8Ql486b).

We consider adding local model inference in the future, such as qwen2.5-vl-7b-instruct, etc.

Additionally, now using streaming inference to output logs, you can see the model's real-time output before the complete output is displayed.

### 1.6

Now the Google gemini SDK has been updated, and the new version of the SDK is suitable for the new model of gemini 2.0!

The new SDK is more powerful and mainly supports the function of verifying uploaded videos.

If you want to repeatedly tag the same video and no longer need to upload it repeatedly, the video name and file size/hash will be automatically verified.

At the same time, the millisecond-level alignment function has been updated. After the subtitles of long video segmentation are merged, the timeline is automatically aligned to milliseconds, which is very neat!

</details>

## Features

- **Provider V2 Architecture** - Modular, extensible provider system with auto-discovery and unified interfaces
- **OpenAI Compatible API** - Universal interface supporting vLLM, SGLang, Ollama, LM Studio for local GPU inference
- **GUI-first workflow** via NiceGUI for import, split, tag, caption, export, and toolbox tasks
- Supports cloud APIs and local OCR / VLM / ALM providers through a unified routing layer
- Export captions in SRT format
- Support for multiple video formats
- Batch processing with progress tracking
- Maintains original directory structure
- Configurable through TOML files
- Lance database integration for efficient data management
- ONNX audio separation with 6-stem export and optional harmony split
- Shared ONNX runtime configuration with reusable cache and provider options
- Standalone text / document translation for txt, md, json, pdf, doc/docx, xls/xlsx, ppt/pptx, rtf, and epub

## Modules

### Dataset Import (`lanceImport.py`)
- Import videos into Lance database format
- Preserve original directory structure
- Support for both single directory and paired directory structures

### Dataset Export (`lanceexport.py`)
- Extract videos and captions from Lance datasets
- Maintains original file structure
- Exports captions as SRT files in the same directory as source videos
- Auto Clip with SRT timestamps

### Auto Captioning (`captioner.py` & `api_handler_v2.py`)
- **Provider V2 Architecture** with 20+ providers (Cloud VLM, Local VLM, OCR, Vision API)
- **OpenAI Compatible Provider** for local inference (vLLM, SGLang, Ollama, LM Studio)
- Remote APIs, local OCR / VLM / ALM, and OpenAI-compatible runtime routing
- Batch processing support
- SRT format output with timestamps
- Robust error handling and retry mechanisms
- Progress tracking for batch operations

### Text / Document Translation (`texttranslate.py`)
- Uses Lance version tags to store imported raw assets, normalized markdown, and translated output
- Supports standalone `.txt/.md` assets as primary inputs
- Normalizes documents into Markdown before chunked local-model translation
- Default local model: `tencent/HY-MT1.5-7B`
- Exports translated files as `*_lang.md` instead of overwriting the source

### Audio Separation (`audio_separator.py`)
- Accepts audio files, folders, and `.lance` datasets
- Produces 6 stems by default, with optional harmony re-splitting
- Supports `wav`, `flac`, and `mp3` export for vocal / backing-track workflows

### Configuration (`config.py`, `config.toml` & `config/onnx.toml`)
- API prompt configuration management
- Customizable batch processing parameters
- ONNX runtime defaults and per-tool overrides
- Default schema includes file paths and metadata

## Installation

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type Set-ExecutionPolicy Unrestricted and answer A
- Close admin powershell window

![Video Preview](https://files.catbox.moe/jr5n3e.gif)

### Windows
Run the following PowerShell script:
```powershell
./1.install-uv-qinglong.ps1
```

### Linux
1. First install PowerShell:
```bash
sudo sh ./0、install pwsh.sh
```
2. Then run the installation script using PowerShell:
```powershell
sudo pwsh ./1.install-uv-qinglong.ps1
```
use sudo pwsh if you in Linux.

### TensorRT (Optional)
windows need to install TensorRT-libs manually from [here](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/zip/TensorRT-10.9.0.34.Windows.win10.cuda-12.8.zip).
TensorRT can faster use WD14Tagger (not effect API part)
Now we use 10.9 version

## Usage

### Recommended flow: use the GUI first

1. Install dependencies:
   ```powershell
   ./1.install-uv-qinglong.ps1
   ```
2. Start the GUI:
   ```powershell
   ./start_gui.ps1
   ```
3. The browser will open `http://127.0.0.1:7899` by default.
4. In the GUI, start with the `Setup` page to check Python / PyTorch / CUDA, then use `Import -> Split -> Tagger -> Caption -> Export / Tools`.

### GUI startup methods

#### Method 1: recommended project wrapper

```powershell
./start_gui.ps1
```

- Switches to the project root, fixes `PYTHONPATH`, and reuses `.venv` / `venv` when present
- Uses the `$Config` block in `start_gui.ps1`; the current default endpoint is `127.0.0.1:7899`
- Starts in browser mode by default; if the port is busy, the launcher probes the next available ports and prints the actual URL

#### Method 2: call the Python entrypoint directly

```powershell
python -m gui.launch --port 7899
```

Common variants:

```powershell
python -m gui.launch --cloud --port 7899 --no-browser
python -m gui.launch --native --port 7899
```

- `--cloud` binds to `0.0.0.0`
- `--native` uses native window mode (requires `pywebview`)
- `--no-browser` keeps the browser closed

See [gui/README.md](gui/README.md) for the page layout and GUI-specific notes.

### Model / provider usage in the GUI

- Remote APIs, OCR, local VLMs, and local ALMs are now configured from the GUI
- When you select a local OCR / VLM / ALM route in the `Caption` page, the GUI automatically adds the matching `uv extra`
- Manual `uv sync --extra xxx` steps for each local VLM / ALM are no longer the recommended workflow
- If you intentionally stay on the script path, edit the corresponding `.ps1` file or `config/*.toml`

### Script mode (advanced / batch workflows)

If you already know the project and want direct scripting, these entry points are still available:

```powershell
./lanceImport.ps1
./4、run.ps1
./lanceExport.ps1
./2.2.preprocess_images.ps1
./2.5.audio_separator.ps1
./5.translate.ps1
```

Notes:

- `4、run.ps1` runs batch captioning
- `2.2.preprocess_images.ps1` handles image preprocessing and optional image alignment; `--matcher-backend=auto` prefers `affine_steerers` on CUDA and `xfeat` otherwise, then falls back to ORB
- `2.5.audio_separator.ps1` runs the ONNX audio separator
- `2.5.audio_separator.ps1` now installs and reuses the `vocal-midi` profile by default, so no extra manual `--extra vocal-midi` step is needed
- `5.translate.ps1` normalizes and translates documents, exporting files such as `*_zh_cn.md`
- Day-to-day runs install the selected dependency profile incrementally; `uv.lock` is mainly kept for CI / release maintenance
