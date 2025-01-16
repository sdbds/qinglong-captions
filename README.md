[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N1NOO2K)

# qinglong-captioner (1.4)

A Python toolkit for generating video captions using the Lance database format and Gemini API for automatic captioning.

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

<video width="720" height="480" controls>
  <source src="https://files.catbox.moe/8fudnf.mp4" type="video/mp4">
</video>

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

```
$dataset_path = "./datasets"
$gemini_api_key = ""
$gemini_model_path = "gemini-exp-1206"
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$dir_name = $true
$mode = "long"
$not_clip_with_caption = $false              # Not clip with caption | 不根据caption裁剪
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
```
$dataset_path = "./datasets"
$gemini_api_key = ""
$gemini_model_path = "gemini-exp-1206"
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$dir_name = $true
$mode = "long"
$not_clip_with_caption = $false              # Not clip with caption | 不根据caption裁剪
```
