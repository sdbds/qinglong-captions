[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N1NOO2K)

# qinglong-captioner (2.0)

A Python toolkit for generating video captions using the Lance database format and Gemini API for automatic captioning.

## Changlog

### 2.3

Well, we forgot to release version 2.2, so we directly released version 2.3!

Version 2.3 updated the GLM4V model for video captions

### 2.2

Version 2.2 has updated TensorRT for accelerating local ONNX model WDtagger.

After testing, it takes 30 minutes to mark 10,000 samples with the standard CUDA tag,

while using TensorRT, it can be completed in just 15 to 20 minutes.

However, the first time using it will take a longer time to compile.

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

## Features

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
./0、install pwsh.sh
```
2. Then run the installation script using PowerShell:
```powershell
sudo pwsh ./1、install-uv-qinglong.ps1
```
use sudo pwsh if you in Linux.

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
$gemini_api_key = ""
$gemini_model_path = "gemini-2.0-pro-exp-02-05"
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$step_api_key = ""
$step_model_path = "step-1.5v-mini"
$qwenVL_api_key = ""
$qwenVL_model_path = "qwen-vl-max-latest" # qwen2.5-vl-72b-instruct<10mins qwen-vl-max-latest <1min
$glm_api_key = ""
$glm_model_path = "GLM-4V-Plus-0111"
$dir_name = $true
$mode = "long"
$not_clip_with_caption = $false              # Not clip with caption | 不根据caption裁剪
$wait_time = 1
$max_retries = 100
$segment_time = 600
$ocr = $false
$document_image = $true
$scene_detector = "AdaptiveDetector" # from ["ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"]
$scene_threshold = 0.0 # default value ["ContentDetector": 27.0, "AdaptiveDetector": 3.0, "HashDetector": 0.395, "HistogramDetector": 0.05, "ThresholdDetector": 12]
$scene_min_len = 15
$scene_luma_only = $false
```
---

# 青龙数据集工具 (1.0)

基于 Lance 数据库格式的视频自动字幕生成工具，使用 Gemini API 进行场景描述生成。

## 功能特点

- 使用 Google Gemini API 进行视频场景自动描述
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
./0、install pwsh.sh
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
./run.ps1
```

注意：使用自动字幕生成功能前，需要在 `run.ps1` 中配置 [Gemini API 密钥](https://aistudio.google.com/apikey)。
[Pixtral API 秘钥](https://console.mistral.ai/api-keys/) 可选为图片打标。

现在我们支持使用[阶跃星辰](https://platform.stepfun.com/)的视频模型进行视频标注。

现在我们支持使用[通义千问VL](https://bailian.console.aliyun.com/#/model-market)的视频模型进行视频标注。

现在我们支持使用[Mistral OCR](https://console.mistral.ai/api-keys/)的OCR功能进行图片字幕生成。

现在我们支持使用[智谱GLM](https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys)的视频模型进行视频标注。

```
$dataset_path = "./datasets"
$gemini_api_key = ""
$gemini_model_path = "gemini-2.0-pro-exp-02-05"
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$step_api_key = ""
$step_model_path = "step-1.5v-mini"
$qwenVL_api_key = ""
$qwenVL_model_path = "qwen-vl-max-latest" # qwen2.5-vl-72b-instruct<10mins qwen-vl-max-latest <1min
$glm_api_key = ""
$glm_model_path = "GLM-4V-Plus-0111"
$dir_name = $true
$mode = "long"
$not_clip_with_caption = $false              # Not clip with caption | 不根据caption裁剪
$wait_time = 1
$max_retries = 100
$segment_time = 600
$ocr = $false
$document_image = $true
$scene_detector = "AdaptiveDetector" # from ["ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"]
$scene_threshold = 0.0 # default value ["ContentDetector": 27.0, "AdaptiveDetector": 3.0, "HashDetector": 0.395, "HistogramDetector": 0.05, "ThresholdDetector": 12]
$scene_min_len = 15
$scene_luma_only = $false
```
