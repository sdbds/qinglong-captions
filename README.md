# qinglong-captioner (WIP)

A Python toolkit for managing and processing image-caption datasets using the Lance database format.

## Features

- Convert image-caption pairs to Lance database format
- Export Lance datasets back to image and caption files
- Support for multiple image formats (PNG, JPG, JPEG, WEBP, BMP)
- Optional support for advanced formats (AVIF, JPEG-XL) when dependencies are available
- Configurable dataset schema through TOML files
- PyTorch Dataset integration for machine learning workflows

## Modules

### Dataset Import (`lancedatasets.py`)
- `ImageProcessor`: Handles image loading and metadata extraction
- `QingLongDataset`: PyTorch Dataset implementation for Lance format data
- `transform2lance()`: Convert raw image-caption data to Lance format

### Dataset Export (`lanceexport.py`)
- Extract images and captions from Lance datasets
- Maintains original file structure
- Quality control for image saving

### Configuration (`config.py`)
- Extensible image format support
- Customizable dataset schema through TOML files
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

---

# 青龙数据集工具 (开发中)

基于 Lance 数据库格式的图像-文字对数据集管理工具。

## 功能特点

- 将图像-文字对转换为 Lance 数据库格式
- 支持将 Lance 数据集导出回图像和文字文件
- 支持多种图像格式 (PNG, JPG, JPEG, WEBP, BMP)
- 可选支持高级格式 (AVIF, JPEG-XL)（需要额外依赖）
- 通过 TOML 文件配置数据集结构
- 集成 PyTorch Dataset 接口，方便机器学习应用

## 模块说明

### 数据集导入 (`lancedatasets.py`)
- `ImageProcessor`: 处理图像加载和元数据提取
- `QingLongDataset`: Lance 格式数据的 PyTorch Dataset 实现
- `transform2lance()`: 将原始图像-文字数据转换为 Lance 格式

### 数据集导出 (`lanceexport.py`)
- 从 Lance 数据集中提取图像和文字说明
- 保持原有文件结构
- 支持图像保存质量控制

### 配置模块 (`config.py`)
- 可扩展的图像格式支持
- 通过 TOML 文件自定义数据集结构
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
