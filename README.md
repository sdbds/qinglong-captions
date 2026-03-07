[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N1NOO2K)

<details>
<summary>中文说明（点击展开）</summary>

# 青龙字幕工具 (3.6)

## 更新日志

### 4.0 - Provider V2 架构重构

1. **全新 Provider V2 架构** - 完全重构的模块化 Provider 系统
   - 统一抽象的 `Provider` 基类，支持 Cloud VLM、Local VLM、OCR、Vision API 四大类别
   - 自动发现机制，通过装饰器自动注册 Provider
   - 统一的 `CaptionResult` 返回类型，解决返回值多态问题
   - 基于优先级的 Provider 路由，自动选择最佳 Provider
   - 向后兼容：设置 `$env:QINGLONG_API_V2="0"` 可回退到旧架构

2. **新增 OpenAI Compatible Provider** - 通用 OpenAI API 兼容接口
   - 支持对接任何 OpenAI 兼容服务：vLLM、SGLang、Ollama、LM Studio
   - 统一配置参数：`openai_base_url`、`openai_model_name` 等
   - 自动降级：JSON 模式不支持时自动切换到文本模式
   - 支持本地 GPU 部署（Qwen2-VL、LLaVA、MiniCPM-V 等）
   - 查看 [docs/openai_compatible.md](docs/openai_compatible.md) 获取详细使用指南

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

基于 Lance 数据库格式的视频自动字幕生成工具，使用 Gemini API 进行场景描述生成。

## 功能特点
- **Provider V2 架构** - 模块化、可扩展的 Provider 系统，支持自动发现和统一接口
- **OpenAI 兼容 API** - 通用接口支持 vLLM、SGLang、Ollama、LM Studio 本地 GPU 推理
- 使用 Google Gemini API 进行视频场景自动字幕生成
- 导出 SRT 格式字幕文件
- 支持多种视频格式
- 批量处理并显示进度
- 保持原始目录结构
- 通过 TOML 文件配置
- 集成 Lance 数据库实现高效数据管理
## 模块说明

### 数据集导入 (`lanceImport.py`)
- 将视频导入 Lance 数据库格式
- 保持原始目录结构
- 支持单目录和配对目录结构

### 数据集导出 (`lanceexport.py`)
- 从 Lance 数据集中提取视频和字幕
- 保持原有文件结构
- 在源视频所在目录导出 SRT 格式字幕

### 自动字幕生成 (`captioner.py` & `api_handler.py`)
- **Provider V2 架构**，支持 19+ 种 Provider（Cloud VLM、Local VLM、OCR、Vision API）
- **OpenAI 兼容 Provider**，支持本地推理（vLLM、SGLang、Ollama、LM Studio）
- 使用 Gemini API 进行视频场景描述
- 支持批量处理
- 生成带时间戳的 SRT 格式字幕
- 健壮的错误处理和重试机制
- 批处理进度跟踪

### 配置模块 (`config.py` & `config.toml`)
- API 配置管理
- 可自定义批处理参数
- 默认结构包含文件路径和元数据

## 安装方法

### Windows 系统
运行以下 PowerShell 脚本：
```powershell
./1、install-uv-qinglong.ps1
```
### Linux 系统
1. 首先安装 PowerShell：
```bash
sudo sh ./0、install pwsh.sh
```
2. 然后使用 PowerShell 运行安装脚本：
```powershell
pwsh ./1、install-uv-qinglong.ps1
```
## 使用方法

### 把媒体文件放到datasets文件夹下

### 导入视频
使用 PowerShell 脚本导入视频：
```powershell
./lanceImport.ps1
```

### 导出数据
使用 PowerShell 脚本从 Lance 格式导出数据：
```powershell
./lanceExport.ps1
```

### 自动字幕生成
使用 PowerShell 脚本为视频生成字幕：
```powershell
./4、run.ps1
```
注意：使用自动字幕生成功能前，需要在 `run.ps1` 中配置 [Gemini API 密钥](https://aistudio.google.com/apikey)。
[Pixtral API 秘钥](https://console.mistral.ai/api-keys/) 可选为图片打标。
现在我们支持使用[阶跃星辰](https://platform.stepfun.com/)的视频模型进行视频标注。
现在我们支持使用[通义千问VL](https://bailian.console.aliyun.com/#/model-market)的视频模型进行视频标注。
现在我们支持使用[Mistral OCR](https://console.mistral.ai/api-keys/)的OCR功能进行图片字幕生成。
现在我们支持使用[智谱GLM](https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys)的视频模型进行视频标注。
config_prompt

```
$dataset_path = "./datasets"
$pair_dir = ""
$gemini_api_key = ""
$gemini_model_path = "gemini-3-pro-preview"
$gemini_task = ""
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$step_api_key = ""
$step_model_path = "step-1.5v-mini"
$kimi_api_key = ""
$kimi_model_path = "moonshotai/kimi-k2.5"
$kimi_base_url = "https://integrate.api.nvidia.com/v1"
$qwenVL_api_key = ""
$qwenVL_model_path = "qwen-vl-max-latest" # qwen2.5-vl-72b-instruct<10mins qwen-vl-max-latest <1min
$glm_api_key = ""
$glm_model_path = "GLM-4V-Plus-0111"
$ark_api_key = ""
$ark_model_path = "doubao-seed-1-6"

# OpenAI Compatible API 配置（支持 vLLM、SGLang、Ollama、LM Studio）
# 当配置了 openai_base_url 时，会优先使用此接口进行本地 GPU 推理
$openai_api_key = ""           # API 密钥（本地服务可填任意值）
$openai_base_url = ""          # 服务器地址，如 http://localhost:8000/v1
$openai_model_name = ""        # 模型名称，如 Qwen2-VL-7B-Instruct
$openai_temperature = 0.7      # 生成温度
$openai_max_tokens = 2048      # 最大 token 数
$openai_json_mode = $true      # 是否使用 JSON 模式

$dir_name = $false
$mode = "long"
$not_clip_with_caption = $true              # Not clip with caption | 不根据caption裁剪
$wait_time = 1
$max_retries = 10
$segment_time = 600
# OCR model configuration
$ocr_model = ""  # Options: "pixtral_ocr", "deepseek_ocr", "hunyuan_ocr", "olmocr", "paddle_ocr", "moondream", "firered_ocr", ""
$document_image = $true

# VLM model configuration for image tasks
$vlm_image_model = ""  # Options: "moondream", "qwen_vl_local", "step_vl_local", "" 

$scene_detector = "AdaptiveDetector" # from ["ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"]
$scene_threshold = 0.0 # default value ["ContentDetector": 27.0, "AdaptiveDetector": 3.0, "HashDetector": 0.395, "HistogramDetector": 0.05, "ThresholdDetector": 12]
$scene_min_len = 1
$scene_luma_only = $false
$tags_highlightrate = 0.3
```

</details>

# qinglong-captioner (3.6)

A Python toolkit for generating video captions using the Lance database format and Gemini API for automatic captioning.

## Changelog

### 4.0 - Provider V2 Architecture Refactoring

1. **Brand New Provider V2 Architecture** - Fully refactored modular Provider system
   - Unified abstract `Provider` base class supporting Cloud VLM, Local VLM, OCR, and Vision API
   - Auto-discovery mechanism with decorator-based Provider registration
   - Unified `CaptionResult` return type resolving polymorphic return value issues
   - Priority-based Provider routing with automatic best Provider selection
   - Backward compatible: Set `$env:QINGLONG_API_V2="0"` to fallback to legacy architecture

2. **New OpenAI Compatible Provider** - Universal OpenAI API compatible interface
   - Support any OpenAI-compatible service: vLLM, SGLang, Ollama, LM Studio
   - Unified configuration: `openai_base_url`, `openai_model_name`, etc.
   - Auto-fallback: Automatically switches to text mode when JSON mode is unsupported
   - Support local GPU deployment (Qwen2-VL, LLaVA, MiniCPM-V, etc.)
   - See [docs/openai_compatible.md](docs/openai_compatible.md) for detailed usage guide

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

<details>
<summary>Older changelog (&lt; 3.0)</summary>

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
- Automatic video/audio/image description using Google's Gemini API or only image with pixtral-large 124B
- Export captions in SRT format
- Support for multiple video formats
- Batch processing with progress tracking
- Maintains original directory structure
- Configurable through TOML files
- Lance database integration for efficient data management

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

### Auto Captioning (`captioner.py` & `api_handler.py`)
- **Provider V2 Architecture** with 19+ providers (Cloud VLM, Local VLM, OCR, Vision API)
- **OpenAI Compatible Provider** for local inference (vLLM, SGLang, Ollama, LM Studio)
- Automatic video scene description using Gemini API or Pixtral API
- Batch processing support
- SRT format output with timestamps
- Robust error handling and retry mechanisms
- Progress tracking for batch operations

### Configuration (`config.py` & `config.toml`)
- API prompt configuration management
- Customizable batch processing parameters
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
./1、install-uv-qinglong.ps1
```

### Linux
1. First install PowerShell:
```bash
sudo sh ./0、install pwsh.sh
```
2. Then run the installation script using PowerShell:
```powershell
sudo pwsh ./1、install-uv-qinglong.ps1
```
use sudo pwsh if you in Linux.

### TensorRT (Optional)
windows need to install TensorRT-libs manually from [here](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/zip/TensorRT-10.9.0.34.Windows.win10.cuda-12.8.zip).
TensorRT can faster use WD14Tagger (not effect API part)
Now we use 10.9 version

## Usage

video example:
https://files.catbox.moe/8fudnf.mp4

### Just put Video or audio files into datasets folders

### Importing Media
Use the PowerShell script to import your videos:
```powershell
./lanceImport.ps1
```

### Exporting Media
Use the PowerShell script to export data from Lance format:
```powershell
./lanceExport.ps1
```

### Auto Captioning
Use the PowerShell script to generate captions for your videos:

```powershell
./run.ps1
```

Note: You'll need to configure your [Gemini API key](https://aistudio.google.com/apikey) in `run.ps1` before using the auto-captioning feature.
[Pixtral API key](https://console.mistral.ai/api-keys/) optional for image caption.

Now we support [step-1.5v-mini](https://platform.stepfun.com/) optional for video captioner.

Now we support [qwen-VL](https://bailian.console.aliyun.com/#/model-market) series optional for video captioner.

Now we support [Mistral OCR](https://console.mistral.ai/api-keys/) optional for PDF and image OCR.

Now we support [GLM](https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys) series optional for video captioner.

```
$dataset_path = "./datasets"
$pair_dir = ""
$gemini_api_key = ""
$gemini_model_path = "gemini-3-pro-preview"
$gemini_task = ""
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$step_api_key = ""
$step_model_path = "step-1.5v-mini"
$kimi_api_key = ""
$kimi_model_path = "moonshotai/kimi-k2.5"
$kimi_base_url = "https://integrate.api.nvidia.com/v1"
$qwenVL_api_key = ""
$qwenVL_model_path = "qwen-vl-max-latest" # qwen2.5-vl-72b-instruct<10mins qwen-vl-max-latest <1min
$glm_api_key = ""
$glm_model_path = "GLM-4V-Plus-0111"
$ark_api_key = ""
$ark_model_path = "doubao-seed-1-6"
$dir_name = $false
$mode = "long"
$not_clip_with_caption = $true              # Not clip with caption | 不根据caption裁剪
$wait_time = 1
$max_retries = 10
$segment_time = 600
# OCR model configuration
$ocr_model = ""  # Options: "pixtral_ocr", "deepseek_ocr", "hunyuan_ocr", "olmocr", "paddle_ocr", "moondream", ""
$document_image = $true

# VLM model configuration for image tasks
$vlm_image_model = ""  # Options: "moondream", "qwen_vl_local", "step_vl_local", "" 

$scene_detector = "AdaptiveDetector" # from ["ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"]
$scene_threshold = 0.0 # default value ["ContentDetector": 27.0, "AdaptiveDetector": 3.0, "HashDetector": 0.395, "HistogramDetector": 0.05, "ThresholdDetector": 12]
$scene_min_len = 1
$scene_luma_only = $false
$tags_highlightrate = 0.3
```
