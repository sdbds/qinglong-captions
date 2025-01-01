[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N1NOO2K)

# qinglong-captioner (WIP)

A Python toolkit for managing and processing image-caption datasets using the Lance database format and Gemini API for automatic captioning.

## Features

- Convert image-caption pairs to Lance database format
- Export Lance datasets back to image and caption files
- Support for multiple image formats (PNG, JPG, JPEG, WEBP, BMP)
- Optional support for advanced formats (AVIF, JPEG-XL) when dependencies are available
- Automatic image captioning using Google's Gemini API
- Configurable dataset schema through TOML files
- PyTorch Dataset integration for machine learning workflows
- Maintains directory structure during import and export

## Modules

### Dataset Import (`lanceImport.py`)
- Convert raw image-caption data to Lance format
- Preserve original directory structure
- Support for both single directory and paired directory structures

### Dataset Export (`lanceexport.py`)
- Extract images and captions from Lance datasets
- Maintains original file structure
- Supports exporting captions as SRT or TXT files

### Auto Captioning (`captioner.py` & `api_handler.py`)
- Automatic image captioning using Gemini API
- Batch processing support
- SRT format output for video/animation files
- Robust error handling and retry mechanisms
- Progress tracking for batch operations

### Configuration (`config.py` & `config.toml`)
- Extensible image format support
- Customizable dataset schema through TOML files
- API configuration management
- Default schema includes file paths, image metadata, and captions

## Installation

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
pwsh ./1、install-uv-qinglong.ps1
```

## Usage

### Importing Data
Use the PowerShell script to import your image-caption dataset:
```powershell
./lanceImport.ps1
```

### Exporting Data
Use the PowerShell script to export data from Lance format:
```powershell
./lanceExport.ps1
```

### Auto Captioning
Use the PowerShell script to generate captions for your images:
```powershell
./run.ps1
```

Note: You'll need to configure your Gemini API key in `config.toml` before using the auto-captioning feature.

---

# 青龙数据集工具 (开发中)

基于 Lance 数据库格式的图像-文字对数据集管理工具，支持使用 Gemini API 进行自动字幕生成。

## 功能特点

- 将图像-文字对转换为 Lance 数据库格式
- 支持将 Lance 数据集导出回图像和文字文件
- 支持多种图像格式 (PNG, JPG, JPEG, WEBP, BMP)
- 可选支持高级格式 (AVIF, JPEG-XL)（需要额外依赖）
- 使用 Google Gemini API 进行自动图像描述生成
- 通过 TOML 文件配置数据集结构
- 集成 PyTorch Dataset 接口，方便机器学习应用
- 导入导出时保持原始目录结构

## 模块说明

### 数据集导入 (`lanceImport.py`)
- 将原始图像-文字数据转换为 Lance 格式
- 保持原始目录结构
- 支持单目录和配对目录结构

### 数据集导出 (`lanceexport.py`)
- 从 Lance 数据集中提取图像和文字说明
- 保持原有文件结构
- 支持导出为 SRT 或 TXT 格式的字幕文件

### 自动字幕生成 (`captioner.py` & `api_handler.py`)
- 使用 Gemini API 进行自动图像描述
- 支持批量处理
- 为视频/动画文件生成 SRT 格式字幕
- 健壮的错误处理和重试机制
- 批处理进度跟踪

### 配置模块 (`config.py` & `config.toml`)
- 可扩展的图像格式支持
- 通过 TOML 文件自定义数据集结构
- API 配置管理
- 默认结构包含文件路径、图像元数据和文字说明

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

### 导入数据
使用 PowerShell 脚本导入图像-文字数据集：
```powershell
./lanceImport.ps1
```

### 导出数据
使用 PowerShell 脚本从 Lance 格式导出数据：
```powershell
./lanceExport.ps1
```

### 自动字幕生成
使用 PowerShell 脚本为图像生成描述：
```powershell
./run.ps1
```

注意：使用自动字幕生成功能前，需要在 `config.toml` 中配置 Gemini API 密钥。
