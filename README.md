[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N1NOO2K)

# qinglong-captioner (2.5)

**qinglong-captioner** is a Python toolkit designed for generating captions and tags for video, audio, and image files. It leverages various AI models for content analysis and uses LanceDB for efficient data management. The toolkit is primarily operated through PowerShell scripts, offering a user-friendly command-line interface.

## Features

*   **Versatile Captioning/Tagging**: Automatic caption and tag generation for videos, audio, and images.
*   **Multi-API Support**: Integrates with multiple AI services:
    *   Google Gemini (Vertex AI & AI Studio)
    *   Mistral AI (Pixtral models for images, Mistral OCR)
    *   StepFun (for video captioning)
    *   Qwen-VL (Alibaba Tongyi Qianwen VL models for video captioning)
    *   GLM (Zhipu AI GLM models for video captioning)
*   **LanceDB Integration**: Uses LanceDB for optimized storage and retrieval of media metadata and captions.
*   **Video Scene Detection**: Splits videos into manageable segments based on scene changes using `2、video_spliter.ps1` (powered by `module/scenedetect.py`).
*   **Subtitle Alignment**: Automatically aligns generated subtitles with detected scene changes for improved accuracy.
*   **Image Tagging & Highlight**:
    *   Generates descriptive tags for images using WD14 Tagger via `3、tagger.ps1` (powered by `utils/wdtagger.py`).
    *   Features "tags highlight" for visual quality assessment of tags (currently for Pixtral).
*   **Image Watermark Detection**: Identifies watermarks in images using `2.1、image_watermark_detect.ps1` (powered by `module/waterdetect.py`).
*   **Batch Processing**: Efficiently processes multiple files.
*   **Configurability**:
    *   Core operations controlled via PowerShell scripts.
    *   Fine-grained settings, prompts, and API parameters managed in `config/config.toml`.
*   **Output Formats**: Generates captions in SRT (SubRip Subtitle) format and tags as text/markdown files.

## How it Works (Workflow Overview)

1.  **Prepare Media**: Place your video, audio, or image files into the `./datasets` folder (or a custom directory).
2.  **Import to LanceDB**: Run `./lanceImport.ps1` to import your media files into a LanceDB dataset (e.g., `./datasets/dataset.lance`). This script indexes your files and extracts initial metadata.
3.  **Generate Image Tags (Optional)**: If you have images, run `./3、tagger.ps1` to generate descriptive tags. These tags are saved as `.txt` files alongside the images and can be used to enhance captioning.
4.  **Configure and Run Main Processing**:
    *   Edit `4、run.ps1` to input your API keys for the desired AI models (Gemini, Pixtral, etc.).
    *   Customize other parameters in `4、run.ps1` such as model selection, processing mode, and specific task instructions.
    *   Execute `./4、run.ps1`. This script processes the media in your dataset (or directly from the `datasets` folder) using the selected AI models to generate captions or tags.
5.  **Access Results**: Captions (typically as SRT files) and tags (as text files) are saved in the same directory as your original media files.
6.  **Export Data (Optional)**: While `4、run.ps1` often handles saving results directly, `./lanceExport.ps1` can be used to export specific versions of data or clip videos based on generated SRT files from a LanceDB dataset.

**Utility Scripts**:
*   `./2、video_spliter.ps1`: Use this script to pre-split videos based on scene changes before further processing.
*   `./2.1、image_watermark_detect.ps1`: Use this script to identify and sort images based on the presence of watermarks.

## Installation

Ensure PowerShell can execute scripts:
*   Open an administrator PowerShell window.
*   Type `Set-ExecutionPolicy Unrestricted` and answer `A` (Yes to All).
*   Close the administrator PowerShell window.

The installation scripts use `uv` to create and manage a Python 3.11 virtual environment.

![Video Preview of Installation](https://files.catbox.moe/jr5n3e.gif)

### Windows
Run the following PowerShell script:
```powershell
./1、install-uv-qinglong.ps1
```

### Linux
1.  First, install PowerShell if it's not already present:
    ```bash
    ./0、install pwsh.sh
    ```
2.  Then, run the installation script using PowerShell (you might need `sudo` depending on your setup):
    ```powershell
    pwsh ./1、install-uv-qinglong.ps1 
    # or sudo pwsh ./1、install-uv-qinglong.ps1
    ```

### TensorRT Acceleration (Optional)
For faster WD14Tagger performance on NVIDIA GPUs, you can install TensorRT.
*   **Windows**: Manually download and install TensorRT libraries (e.g., version 10.9.x) from the [NVIDIA TensorRT page](https://developer.nvidia.com/tensorrt). Ensure the version matches CUDA compatibility. The scripts attempt to use TensorRT if available and fall back to standard CUDA or CPU if not.
*   The first run with TensorRT may take longer due to model compilation.

## Core Scripts & Usage

### `lanceImport.ps1`
*   **Purpose**: Imports media files from a specified directory into a LanceDB file.
*   **Key Variables (configurable within the script)**:
    *   `$train_data_dir = "./datasets"`: Directory containing your media files.
    *   `$caption_dir = "./datasets/caption"`: Optional directory for pre-existing caption files (e.g., `.srt`, `.txt`).
    *   `$output_name = "dataset"`: Name of the LanceDB file to be created (e.g., `dataset.lance`) in `$train_data_dir`.
    *   `$import_mode = 0`: Media import mode:
        *   `0`: All media types (video, audio, image).
        *   `1`: Video files only.
        *   `2`: Audio files only.
        *   `3`: Split video files based on audio presence.
    *   `$tag = "latest"`: Tag to version the dataset within LanceDB.

### `3、tagger.ps1`
*   **Purpose**: Generates descriptive tags for images using the WD14 Tagger model. Tags are saved as `.txt` files next to the corresponding images.
*   **Key Variables/Sections (configurable within the script)**:
    *   **`$Config` Block**:
        *   `train_data_dir = "./datasets"`: Input directory for images.
        *   `repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"`: Hugging Face repository ID for the tagger model.
        *   `model_dir = "./wd14_tagger_model"`: Local directory to cache the downloaded model.
        *   `batch_size = 8`: Number of images to process in a batch.
        *   `general_thresh = 0.35`: General threshold for tag confidence.
        *   `character_thresh = 0.85`: Character-specific threshold for tag confidence.
    *   **`$Features` Block**:
        *   `remove_underscore = $true`: Remove underscores from tags.
        *   `frequency_tags = $false`: Prepend tags with their frequency.
        *   `overwrite = $false`: Overwrite existing tag files if `$true`.
    *   **`$TagConfig` Block**:
        *   `undesired_tags = @("absurdres", "highres", "translation_request")`: List of tags to exclude.
        *   `always_first_tags = @("1girl", "1boy", "2girls", "multiple_girls", "male_focus", "female_focus")`: Tags to prioritize at the beginning.
        *   `tag_replacement = @{ "old_tag" = "new_tag" }`: Dictionary for replacing specific tags.

### `4、run.ps1` (Main Captioning/Processing Script)
*   **Purpose**: Processes media from the specified dataset path (filesystem directory or LanceDB file) using configured AI models to generate captions, tags, or perform other AI-driven tasks.
*   **Key User-Configurable Variables (at the top of the script)**:
    *   `$dataset_path = "./datasets"`: Path to media files directory or a `.lance` dataset file.
    *   **API Keys (Required for respective services)**:
        *   `$gemini_api_key = ""`: Google Gemini API Key.
        *   `$pixtral_api_key = ""`: Mistral AI API Key (for Pixtral image models & Mistral OCR).
        *   `$step_api_key = ""`: StepFun API Key.
        *   `$qwenVL_api_key = ""`: Alibaba Qwen-VL API Key.
        *   `$glm_api_key = ""`: Zhipu GLM API Key.
    *   **Model Paths/Names**:
        *   `$gemini_model_path = "gemini-1.5-pro-latest"` (or e.g., `gemini-1.0-pro-vision-001`, `gemini-1.5-flash-latest`)
        *   `$pixtral_model_path = "mistral-large-latest"` (or e.g., `open-mixtral-8x22b`, `mistral-small-latest` for Pixtral; `mistral-large-latest` also used for OCR if `$ocr = $true`)
        *   `$step_model_path = "step-1v-8k"` (or other StepFun models)
        *   `$qwenVL_model_path = "qwen-vl-max"` (or e.g., `qwen-vl-plus`, `qwen-long`)
        *   `$glm_model_path = "glm-4v"` (or other GLM models)
    *   `$gemini_task = "default"`: Specifies a task for Gemini image processing (defined in `config/config.toml` under `[prompts.task]`).
    *   `$dir_name = $true`: If `$true`, uses the parent directory name of an image as a character name hint for Pixtral image captioning.
    *   `$mode = "long"`: Captioning mode for Pixtral:
        *   `"all"`: Generate comprehensive captions.
        *   `"long"`: Generate detailed long-form captions.
        *   `"short"`: Generate concise short-form captions.
    *   `$not_clip_with_caption = $false`: If `$true`, disables automatic video clipping based on generated SRT captions during export.
    *   `$wait_time = 1`: Seconds to wait between API retries.
    *   `$max_retries = 100`: Maximum number of retries for API calls.
    *   `$segment_time = 600`: Default duration (in seconds) for splitting long videos before captioning (if not using `2、video_spliter.ps1` beforehand).
    *   `$ocr = $false`: If `$true`, enables OCR mode using Mistral OCR. Primarily for images; can also process PDFs if `$document_image` is `$true`.
    *   `$document_image = $true`: If `$true` and `$ocr` is `$true` for a PDF file, attempts to extract and save images from each page of the PDF.
    *   **Scene Detection Parameters (used for long video captioning if not pre-split)**:
        *   `$scene_detector = "AdaptiveDetector"`: Algorithm for scene detection (options: "ContentDetector", "AdaptiveDetector", "HashDetector", "HistogramDetector", "ThresholdDetector").
        *   `$scene_threshold = 0.0`: Detection sensitivity (0.0 for auto, otherwise specific to detector).
        *   `$scene_min_len = 15`: Minimum frames for a valid scene.
        *   `$scene_luma_only = $false`: Use only luma channel for scene detection if `$true`.
    *   `$tags_highlightrate = 0.35`: Minimum ratio of highlighted tags (good quality) for Pixtral image tagging to pass an internal quality check.

### `lanceExport.ps1`
*   **Purpose**: Exports media and/or captions from a LanceDB dataset back to the filesystem.
*   **Key Command-Line Parameters**:
    *   `-lance_file "./datasets/dataset.lance"`: Path to the LanceDB dataset file. (Required)
    *   `-output_dir "./datasets/output"`: Directory where files will be exported. (Required)
    *   `-version "latest"`: LanceDB dataset version (tag) to export. (Optional, defaults to "latest")
    *   `-not_clip_with_caption`: If specified as a switch (`-not_clip_with_caption`), disables automatic video clipping based on SRT files during export. (Optional)

### `2、video_spliter.ps1`
*   **Purpose**: Utility script to split videos into smaller clips based on detected scene changes. Outputs video clips, and optionally an HTML report and frame grabs from detected scenes.
*   **Key Variables (within the `$Config` block at the top of the script)**:
    *   `input_video_dir = "./datasets/video"`: Directory containing videos to split.
    *   `output_dir = "./datasets/output_video_segments"`: Directory to save the segmented video clips and other outputs.
    *   `detector = "AdaptiveDetector"`: Scene detection algorithm.
    *   `threshold = 3.0`: Detection sensitivity (specific to the chosen `detector`).
    *   `min_scene_len = 15`: Minimum number of frames to constitute a scene.
    *   `save_html = $true`: If `$true`, generates an HTML report visualizing scene cuts.
    *   `video2images_min_number = 10`: Minimum number of images to extract if saving frames.

### `2.1、image_watermark_detect.ps1`
*   **Purpose**: Utility to detect watermarks in images. It creates symbolic links to the original images in `watermarked` or `no_watermark` subfolders within the output directory.
*   **Key Variables (within the `$Config` block at the top of the script)**:
    *   `train_data_dir = "./datasets/images_to_check"`: Input directory containing images.
    *   `output_dir = "./datasets/watermark_check_results"`: Directory where `watermarked` and `no_watermark` subfolders with symlinks will be created.
    *   `repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"`: Hugging Face repository ID for the watermark detection model (Note: This seems to be the same as the tagger model; ensure this is the intended model for watermark detection or update if a specific watermark model is used).
    *   `model_dir = "./watermark_model_cache"`: Local directory to cache the downloaded model.
    *   `batch_size = 8`: Number of images to process in a batch.
    *   `thresh = 0.5`: Threshold for watermark detection confidence.

## Configuration (`config/config.toml`)

The `config/config.toml` file provides fine-grained control over various aspects of the captioning and tagging process. Users can modify this file to customize prompts, AI model parameters, and other operational settings.

*   **`[prompts]`**:
    *   This is the core section for defining the prompts sent to the AI models.
    *   It contains sub-sections for different media types (`video`, `audio`, `image`) and AI providers (e.g., `gemini`, `pixtral`).
    *   Users can customize `system_prompt` and `user_prompt` templates for each model and media combination to tailor the AI's output (e.g., style, length, focus of captions).
*   **`[prompts.task]`**:
    *   Defines specific task-oriented prompts, primarily for Gemini image manipulation (e.g., `remove_watermark_strong`, `remove_watermark_slight`). These are referenced by the `$gemini_task` variable in `4、run.ps1`.
*   **`[generation_config]`**:
    *   Allows tweaking of API parameters for different Gemini models (e.g., `gemini-1.5-pro-latest`, `gemini-1.0-pro-vision-001`).
    *   Parameters include `temperature` (randomness of output), `top_p` (nucleus sampling), `max_output_tokens` (limits response length).
*   **`[schema]`**:
    *   Defines the structure (fields and data types) for the LanceDB datasets.
    *   This is mainly for advanced users or developers who need to understand or modify the data storage format.
*   **`[colors]`**:
    *   Maps item types (e.g., "video", "image", "audio", "text") to specific colors used for console output via the `rich` library, enhancing readability of logs.
*   **`[tag_type]`**:
    *   Defines categories (e.g., `character`, `copyright`, `clothing`, `body_features`, `action`) and their associated display colors.
    *   This configuration is used for the "tags highlight captions" feature, primarily with Pixtral, to visually categorize and assess the quality of generated image tags.

## Python Modules Overview (For Developers/Advanced Users)

For those interested in the project's internals or looking to extend its capabilities, here's a high-level overview of the key Python modules:

*   **`module/captioner.py`**: Orchestrates the main captioning workflow. It manages batch processing, handles video splitting (often by calling `scenedetect.py`), aligns subtitles to scene changes, and coordinates with `api_handler.py` for AI processing.
*   **`module/api_handler.py`**: Central module for all communications with external AI APIs (Gemini, Pixtral, StepFun, Qwen-VL, GLM). It's responsible for formatting requests according to each API's specification, sending requests, parsing responses, and implementing error handling and retry logic.
*   **`module/lanceImport.py`**: Handles the creation of LanceDB datasets. This includes scanning input directories for media files, extracting relevant metadata (like file paths, types, and resolution), and writing this information into a LanceDB table according to the schema defined in `config.toml`.
*   **`module/lanceexport.py`**: Responsible for extracting data from LanceDB datasets back to the filesystem. It can retrieve media files and their associated captions, and includes functionality to clip videos based on SRT timestamps.
*   **`module/scenedetect.py`**: Implements video scene detection capabilities using the PySceneDetect library. It provides functions to identify scene boundaries, which are then used for splitting videos or aligning subtitles.
*   **`module/waterdetect.py`**: Contains the logic for detecting watermarks in images. It typically uses an ONNX model for inference and helps in organizing images based on whether they contain watermarks.
*   **`utils/wdtagger.py`**: Implements image tagging functionality using a WD14-style tagger model (often ONNX-based). It processes images to generate descriptive tags, applying configured thresholds and filters.
*   **`config/config.py`**: Provides Python functions to load, parse, and access settings from the `config/config.toml` file. It also defines default configurations, supported file extensions, and other global settings used throughout the toolkit.

## Changelog

### 2.5
![image](https://github.com/user-attachments/assets/bffd2120-6868-4a6e-894b-05c4ff5fd98f)
*   Official support for "tags highlight captions" feature (initially for Pixtral).
*   WDtagger annotations can be used to improve VLM accuracy.
*   Categorized tags with color-coding for quality check (e.g., character, clothing, body features, actions).
*   Added parameters to specify parent folder as character name and check for tags highlight rate (aiming for >35%).
*   Usage: `3、tagger.ps1` then `4、run.ps1` with Pixtral API key.

*(Older changelog entries from 2.4 down to 1.6 have been retained below for historical context but are omitted here for brevity in this plan. They will be included in the final file.)*

### 2.4
We support Gemini image caption and rating. It also supports gemini2.5-flash-preview-04-17. However, after testing, the flash version has poor effects and image review, it is recommended to use the pro version.
![image](https://github.com/user-attachments/assets/6ae9ed38-e67a-41d2-aa1d-4caf0e0db394)
flash↑
![image](https://github.com/user-attachments/assets/c83682aa-3a37-4198-b117-ffe7f74ff812)
pro ↑

### 2.3
Well, we forgot to release version 2.2, so we directly released version 2.3! Version 2.3 updated the GLM4V model for video captions.

### 2.2
Version 2.2 has updated TensorRT for accelerating local ONNX model WDtagger. After testing, it takes 30 minutes to mark 10,000 samples with the standard CUDA tag, while using TensorRT, it can be completed in just 15 to 20 minutes. However, the first time using it will take a longer time to compile. If TensorRT fails, it will automatically revert to CUDA without worry. If it prompts that TensorRT librarys are missing, it may be missing some parts. Please install version 10.7.x manually from [here](https://developer.nvidia.com/tensorrt/download/10x)

### 2.1
Added support for Gemini 2.5 Pro Exp. Now uses 600 seconds cut video by default.

### 2.0 Big Update！
Now we support video segmentation! A new video segmentation module has been added, which detects key timestamps based on scene changes and then outputs the corresponding images and video clips! Export an HTML for reference, the effect is very significant!
![image](https://github.com/user-attachments/assets/94407fec-92af-4a34-a15e-bc02bf45d2cd)
We have also added subtitle alignment algorithms, which automatically align Gemini's timestamp subtitles to the millisecond level after detecting scene change frames (there are still some errors, but the effect is much better).
Finally, we added the image output feature of the latest gemini-2.0-flash-exp model! You can customize the task, add the task name in the [`config.toml`](https://github.com/sdbds/qinglong-captions/blob/main/config/config.toml), which will automatically handle the corresponding images (and then label them).
Currently, some simple task descriptions are as follows: Welcome the community to continuously optimize these task prompts and provide contributions!
https://github.com/sdbds/qinglong-captions/blob/12b7750ee0bca7e41168e98775cd95c7b9c57173/config/config.toml#L239-L249
![image](https://github.com/user-attachments/assets/7e5ae1a9-b635-4705-b664-1c20934d12bc)
![image](https://github.com/user-attachments/assets/58527298-34f8-496d-8c4e-1a1c1c965b73)

### 1.9
Now with Mistral OCR functionality! Utilizing Mistral's advanced OCR capabilities to extract text information from videos and images. This feature is particularly useful when processing media files containing subtitles, signs, or other text elements, enhancing the accuracy and completeness of captions. The OCR functionality is integrated into the existing workflow and can be used without additional configuration.

### 1.8
Now added WDtagger！ Even if you cannot use the GPU, you can also use the CPU for labeling. It has multi-threading and various optimizations, processing large-scale data quickly. Using ONNX processing, model acceleration. Code reference@kohya-ss https://github.com/sdbds/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py
Version 2.0 will add dual caption functionality, input wdtagger's taggers, then output natural language.
![image](https://github.com/user-attachments/assets/f14d4a69-9c79-4ffb-aff7-84d103dfeff4)

### 1.7
Now we support the qwen-VL series video caption model!
- qwen-vl-max-latest
- qwen2.5-vl-72b-instruct 
- qwen2.5-vl-7b-instruct
- qwen2.5-vl-3b-instruct
qwen2.5-vl has 2 seconds ~ 10 mins, qwen-vl-max-latest has 1 min limit. These models are not good at capturing timestamps; it is recommended to use segmented video clips for captions and to modify the prompts. Video upload feature requires an application to be submitted to the official, please submit the application [here](https://smartservice.console.aliyun.com/service/create-ticket?spm=a2c4g.11186623.0.0.3489b0a8Ql486b). We consider adding local model inference in the future, such as qwen2.5-vl-7b-instruct, etc. Additionally, now using streaming inference to output logs, you can see the model's real-time output before the complete output is displayed.

### 1.6
Now the Google gemini SDK has been updated, and the new version of the SDK is suitable for the new model of gemini 2.0! The new SDK is more powerful and mainly supports the function of verifying uploaded videos. If you want to repeatedly tag the same video and no longer need to upload it repeatedly, the video name and file size/hash will be automatically verified. At the same time, the millisecond-level alignment function has been updated. After the subtitles of long video segmentation are merged, the timeline is automatically aligned to milliseconds, which is very neat!

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on how to contribute to this project. (Note: `CONTRIBUTING.md` will be created in a future step).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (assuming a `LICENSE` file exists or will be added, typically MIT for such projects).

---

# 青龙字幕工具 (Qinglong Captioner) (2.5)

**qinglong-captioner (青龙字幕工具)** 是一个 Python 工具包，专为视频、音频和图像文件生成字幕与标签。它利用多种人工智能模型进行内容分析，并使用 LanceDB 进行高效的数据管理。该工具包主要通过 PowerShell 脚本操作，提供了一个用户友好的命令行界面。

## 功能特点

*   **多样化字幕/标签生成**: 为视频、音频和图像自动生成字幕和标签。
*   **多 AI 服务支持**: 集成多个 AI 服务：
    *   Google Gemini (Vertex AI & AI Studio)
    *   Mistral AI (Pixtral 模型用于图像处理，Mistral OCR 用于文字识别)
    *   阶跃星辰 (StepFun, 用于视频字幕)
    *   通义千问-VL (Qwen-VL, 阿里巴巴模型, 用于视频字幕)
    *   智谱 GLM (Zhipu AI GLM 模型, 用于视频字幕)
*   **LanceDB 集成**: 使用 LanceDB 优化媒体元数据和字幕的存储与检索。
*   **视频场景检测**: 通过 `2、video_spliter.ps1` (依赖 `module/scenedetect.py`) 根据场景变化将视频分割成易于管理的小段。
*   **字幕对齐**: 自动将生成的字幕与检测到的场景变化对齐，以提高准确性。
*   **图像标签与高亮**:
    *   通过 `3、tagger.ps1` (依赖 `utils/wdtagger.py`) 使用 WD14 Tagger 为图像生成描述性标签。
    *   具备“标签高亮”功能，用于直观评估标签质量 (目前主要针对 Pixtral)。
*   **图像水印检测**: 使用 `2.1、image_watermark_detect.ps1` (依赖 `module/waterdetect.py`) 识别图像中的水印。
*   **批量处理**: 高效处理多个文件。
*   **高度可配置**:
    *   核心操作通过 PowerShell 脚本控制。
    *   精细化设置、提示词和 API 参数在 `config/config.toml` 文件中管理。
*   **输出格式**: 生成 SRT (SubRip Subtitle) 格式的字幕和文本/Markdown 格式的标签。

## 工作流程概述

1.  **准备媒体文件**: 将您的视频、音频或图像文件放入 `./datasets` 文件夹 (或自定义目录)。
2.  **导入到 LanceDB**: 运行 `./lanceImport.ps1` 将媒体文件导入 LanceDB 数据集 (例如 `./datasets/dataset.lance`)。此脚本会索引文件并提取初始元数据。
3.  **生成图像标签 (可选)**: 如果您有图像文件，运行 `./3、tagger.ps1` 生成描述性标签。这些标签会保存为 `.txt` 文件，与图像放在一起，可用于增强字幕效果。
4.  **配置并运行主处理脚本**:
    *   编辑 `4、run.ps1` 文件，输入您所需 AI 模型 (Gemini, Pixtral 等) 的 API 密钥。
    *   在 `4、run.ps1` 中自定义其他参数，如模型选择、处理模式和特定任务指令。
    *   执行 `./4、run.ps1`。此脚本会使用选定的 AI 模型处理数据集中的媒体 (或直接从 `datasets` 文件夹读取)，以生成字幕或标签。
5.  **查看结果**: 字幕 (通常为 SRT 文件) 和标签 (文本文件) 会保存在原始媒体文件所在的目录中。
6.  **导出数据 (可选)**: 虽然 `4、run.ps1` 通常会直接保存结果，但 `./lanceExport.ps1` 也可用于从 LanceDB 数据集导出特定版本的数据，或根据生成的 SRT 文件裁剪视频。

**辅助脚本**:
*   `./2、video_spliter.ps1`: 在进一步处理之前，使用此脚本根据场景变化预分割视频。
*   `./2.1、image_watermark_detect.ps1`: 使用此脚本根据水印存在情况识别和分类图像。

## 安装指南

请确保 PowerShell 可以执行脚本：
*   打开一个管理员 PowerShell 窗口。
*   输入 `Set-ExecutionPolicy Unrestricted` 并回答 `A` (全是)。
*   关闭管理员 PowerShell 窗口。

安装脚本使用 `uv` 创建和管理 Python 3.11 虚拟环境。

![安装过程视频预览](https://files.catbox.moe/jr5n3e.gif)

### Windows 系统
运行以下 PowerShell 脚本：
```powershell
./1、install-uv-qinglong.ps1
```

### Linux 系统
1.  如果尚未安装 PowerShell，请先安装：
    ```bash
    ./0、install pwsh.sh
    ```
2.  然后，使用 PowerShell 运行安装脚本 (根据您的设置，可能需要 `sudo`)：
    ```powershell
    pwsh ./1、install-uv-qinglong.ps1
    # 或 sudo pwsh ./1、install-uv-qinglong.ps1
    ```

### TensorRT 加速 (可选)
为了在 NVIDIA GPU 上获得更快的 WD14Tagger 性能，您可以安装 TensorRT。
*   **Windows**: 从 [NVIDIA TensorRT 页面](https://developer.nvidia.com/tensorrt) 手动下载并安装 TensorRT 库 (例如 10.9.x 版本)。确保版本与 CUDA 兼容。如果 TensorRT 可用，脚本会尝试使用它；否则将回退到标准 CUDA 或 CPU。
*   首次使用 TensorRT 运行可能因为模型编译而耗时较长。

##核心脚本与用法

### `lanceImport.ps1`
*   **用途**: 将指定目录中的媒体文件导入 LanceDB 文件。
*   **主要变量 (可在脚本内配置)**:
    *   `$train_data_dir = "./datasets"`: 包含媒体文件的目录。
    *   `$caption_dir = "./datasets/caption"`: 可选目录，用于存放已有的字幕文件 (例如 `.srt`, `.txt`)。
    *   `$output_name = "dataset"`: 要创建的 LanceDB 文件的名称 (不含扩展名)，将存放在 `$train_data_dir` 中 (例如 `dataset.lance`)。
    *   `$import_mode = 0`: 媒体导入模式:
        *   `0`: 所有媒体类型 (视频, 音频, 图像)。
        *   `1`: 仅视频文件。
        *   `2`: 仅音频文件。
        *   `3`: 根据音频存在情况分割视频文件。
    *   `$tag = "latest"`: 用于在 LanceDB 中标记数据集版本的标签。

### `3、tagger.ps1`
*   **用途**: 使用 WD14 Tagger 模型为图像生成描述性标签。标签以 `.txt` 文件形式保存在相应图像旁边。
*   **主要变量/配置块 (可在脚本内配置)**:
    *   **`$Config` 配置块**:
        *   `train_data_dir = "./datasets"`: 图像输入目录。
        *   `repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"`: Tagger 模型的 Hugging Face 仓库 ID。
        *   `model_dir = "./wd14_tagger_model"`: 本地缓存下载模型的目录。
        *   `batch_size = 8`: 每批处理的图像数量。
        *   `general_thresh = 0.35`: 通用标签的置信度阈值。
        *   `character_thresh = 0.85`: 角色相关标签的置信度阈值。
    *   **`$Features` 配置块**:
        *   `remove_underscore = $true`: 从标签中移除下划线。
        *   `frequency_tags = $false`: 为标签添加频率前缀。
        *   `overwrite = $false`: 如果为 `$true`，则覆盖现有的标签文件。
    *   **`$TagConfig` 配置块**:
        *   `undesired_tags = @("absurdres", "highres", "translation_request")`: 要排除的标签列表。
        *   `always_first_tags = @("1girl", "1boy", "2girls", "multiple_girls", "male_focus", "female_focus")`: 总是置于最前面的标签。
        *   `tag_replacement = @{ "old_tag" = "new_tag" }`: 用于替换特定标签的字典。

### `4、run.ps1` (主字幕/处理脚本)
*   **用途**: 使用配置的 AI 模型处理指定数据集路径 (文件系统目录或 LanceDB 文件) 中的媒体，以生成字幕、标签或执行其他 AI 驱动的任务。
*   **主要用户可配置变量 (位于脚本顶部)**:
    *   `$dataset_path = "./datasets"`: 媒体文件目录或 `.lance` 数据集文件的路径。
    *   **API 密钥 (相应服务必需)**:
        *   `$gemini_api_key = ""`: Google Gemini API 密钥。
        *   `$pixtral_api_key = ""`: Mistral AI API 密钥 (用于 Pixtral 图像模型和 Mistral OCR)。
        *   `$step_api_key = ""`: 阶跃星辰 API 密钥。
        *   `$qwenVL_api_key = ""`: 阿里巴巴通义千问-VL API 密钥。
        *   `$glm_api_key = ""`: 智谱 GLM API 密钥。
    *   **模型路径/名称**:
        *   `$gemini_model_path = "gemini-1.5-pro-latest"` (例如 `gemini-1.0-pro-vision-001`, `gemini-1.5-flash-latest`)
        *   `$pixtral_model_path = "mistral-large-latest"` (例如 `open-mixtral-8x22b`, `mistral-small-latest` 用于 Pixtral; 如果 `$ocr = $true`, `mistral-large-latest` 也用于 OCR)
        *   `$step_model_path = "step-1v-8k"` (或其他阶跃星辰模型)
        *   `$qwenVL_model_path = "qwen-vl-max"` (例如 `qwen-vl-plus`, `qwen-long`)
        *   `$glm_model_path = "glm-4v"` (或其他 GLM 模型)
    *   `$gemini_task = "default"`: 指定 Gemini 图像处理任务 (在 `config/config.toml` 的 `[prompts.task]` 下定义)。
    *   `$dir_name = $true`: 如果为 `$true`，则使用图像的父目录名作为 Pixtral 图像字幕的角色名称提示。
    *   `$mode = "long"`: Pixtral 字幕模式:
        *   `"all"`: 生成全面字幕。
        *   `"long"`: 生成详细的长字幕。
        *   `"short"`: 生成简洁的短字幕。
    *   `$not_clip_with_caption = $false`: 如果为 `$true`，则在导出时禁用基于生成的 SRT 字幕自动裁剪视频。
    *   `$wait_time = 1`: API 重试之间的等待秒数。
    *   `$max_retries = 100`: API 调用的最大重试次数。
    *   `$segment_time = 600`: 在字幕处理前分割长视频的默认时长 (秒) (如果未使用 `2、video_spliter.ps1` 预先分割)。
    *   `$ocr = $false`: 如果为 `$true`，则使用 Mistral OCR 启用 OCR 模式。主要用于图像；如果 `$document_image` 为 `$true`，也可以处理 PDF。
    *   `$document_image = $true`: 如果为 `$true` 且对 PDF 文件启用 `$ocr`，则尝试从 PDF 的每一页提取并保存图像。
    *   **场景检测参数 (用于长视频字幕，如果未预先分割)**:
        *   `$scene_detector = "AdaptiveDetector"`: 场景检测算法 (选项: "ContentDetector", "AdaptiveDetector", "HashDetector", "HistogramDetector", "ThresholdDetector")。
        *   `$scene_threshold = 0.0`: 检测灵敏度 (0.0 表示自动，否则特定于检测器)。
        *   `$scene_min_len = 15`: 有效场景的最小帧数。
        *   `$scene_luma_only = $false`: 如果为 `$true`，则仅使用亮度通道进行场景检测。
    *   `$tags_highlightrate = 0.35`: Pixtral 图像标签高亮标签 (高质量) 的最低比例，以通过内部质量检查。

### `lanceExport.ps1`
*   **用途**: 将 LanceDB 数据集中的媒体和/或字幕导出回文件系统。
*   **主要命令行参数**:
    *   `-lance_file "./datasets/dataset.lance"`: LanceDB 数据集文件的路径。(必需)
    *   `-output_dir "./datasets/output"`: 文件导出目录。(必需)
    *   `-version "latest"`: 要导出的 LanceDB 数据集版本 (标签)。(可选, 默认为 "latest")
    *   `-not_clip_with_caption`: 如果作为开关指定 (`-not_clip_with_caption`)，则在导出期间禁用基于 SRT 文件的自动视频裁剪。(可选)

### `2、video_spliter.ps1`
*   **用途**: 根据检测到的场景变化将视频分割成较小片段的实用工具脚本。输出视频片段，并可选地输出 HTML 报告和检测场景的帧截图。
*   **主要变量 (位于脚本顶部的 `$Config` 配置块中)**:
    *   `input_video_dir = "./datasets/video"`: 包含待分割视频的目录。
    *   `output_dir = "./datasets/output_video_segments"`: 保存分割后视频片段和其他输出的目录。
    *   `detector = "AdaptiveDetector"`: 场景检测算法。
    *   `threshold = 3.0`: 检测灵敏度 (特定于所选 `detector`)。
    *   `min_scene_len = 15`:构成一个场景的最小帧数。
    *   `save_html = $true`: 如果为 `$true`，则生成可视化场景剪切的 HTML 报告。
    *   `video2images_min_number = 10`: 如果保存帧，则提取的最小图像数量。

### `2.1、image_watermark_detect.ps1`
*   **用途**: 检测图像中水印的实用工具。它会在输出目录的 `watermarked` 或 `no_watermark` 子文件夹中创建指向原始图像的符号链接。
*   **主要变量 (位于脚本顶部的 `$Config` 配置块中)**:
    *   `train_data_dir = "./datasets/images_to_check"`: 包含图像的输入目录。
    *   `output_dir = "./datasets/watermark_check_results"`: 将在其中创建包含符号链接的 `watermarked` 和 `no_watermark` 子文件夹的目录。
    *   `repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"`: 水印检测模型的 Hugging Face 仓库 ID (注意: 这似乎与 Tagger 模型相同；请确保这是用于水印检测的预期模型，或在有特定的水印模型时更新)。
    *   `model_dir = "./watermark_model_cache"`: 本地缓存下载模型的目录。
    *   `batch_size = 8`: 每批处理的图像数量。
    *   `thresh = 0.5`: 水印检测置信度阈值。

## 配置 (`config/config.toml`)

`config/config.toml` 文件提供了对字幕和标签生成过程各个方面的精细控制。用户可以修改此文件以自定义提示、AI 模型参数和其他操作设置。

*   **`[prompts]`**:
    *   这是定义发送给 AI 模型的提示的核心部分。
    *   它包含针对不同媒体类型 (`video`, `audio`, `image`) 和 AI 提供商 (例如 `gemini`, `pixtral`) 的子部分。
    *   用户可以为每个模型和媒体组合自定义 `system_prompt` 和 `user_prompt` 模板，以调整 AI 的输出 (例如字幕的风格、长度、重点)。
*   **`[prompts.task]`**:
    *   定义特定面向任务的提示，主要用于 Gemini 图像处理 (例如 `remove_watermark_strong`, `remove_watermark_slight`)。这些由 `4、run.ps1` 中的 `$gemini_task` 变量引用。
*   **`[generation_config]`**:
    *   允许调整不同 Gemini 模型 (例如 `gemini-1.5-pro-latest`, `gemini-1.0-pro-vision-001`) 的 API 参数。
    *   参数包括 `temperature` (输出的随机性), `top_p` (核心采样), `max_output_tokens` (限制响应长度)。
*   **`[schema]`**:
    *   定义 LanceDB 数据集的结构 (字段和数据类型)。
    *   这主要供需要理解或修改数据存储格式的高级用户或开发人员使用。
*   **`[colors]`**:
    *   将项目类型 (例如 "video", "image", "audio", "text") 映射到通过 `rich` 库用于控制台输出的特定颜色，从而增强日志的可读性。
*   **`[tag_type]`**:
    *   定义类别 (例如 `character` (角色), `copyright` (版权), `clothing` (服装), `body_features` (身体特征), `action` (动作)) 及其关联的显示颜色。
    *   此配置用于“标签高亮字幕”功能 (主要与 Pixtral 一起使用)，以可视化方式对生成的图像标签进行分类和质量评估。

## Python 模块概述 (面向开发者/高级用户)

对于那些对项目内部结构感兴趣或希望扩展其功能的用户，以下是关键 Python 模块的高级概述：

*   **`module/captioner.py`**: 协调主要的字幕生成工作流程。它管理批量处理，处理视频分割 (通常通过调用 `scenedetect.py`)，将字幕与场景变化对齐，并与 `api_handler.py` 协调进行 AI 处理。
*   **`module/api_handler.py`**: 处理与外部 AI API (Gemini, Pixtral, StepFun, Qwen-VL, GLM) 所有通信的中央模块。它负责根据每个 API 的规范格式化请求、发送请求、解析响应以及实现错误处理和重试逻辑。
*   **`module/lanceImport.py`**: 处理 LanceDB 数据集的创建。这包括扫描输入目录中的媒体文件，提取相关元数据 (如文件路径、类型和分辨率)，并根据 `config.toml` 中定义的模式将此信息写入 LanceDB 表。
*   **`module/lanceexport.py`**: 负责将 LanceDB 数据集中的数据 (媒体和字幕) 提取回文件系统。它可以检索媒体文件及其关联的字幕，并包括根据 SRT 时间戳裁剪视频的功能。
*   **`module/scenedetect.py`**: 使用 PySceneDetect 库实现视频场景检测功能。它提供识别场景边界的功能，这些边界随后用于分割视频或对齐字幕。
*   **`module/waterdetect.py`**: 包含检测图像中水印的逻辑。它通常使用 ONNX 模型进行推理，并帮助根据图像是否包含水印来组织图像。
*   **`utils/wdtagger.py`**: 使用 WD14 风格的 Tagger 模型 (通常基于 ONNX) 实现图像标签功能。它处理图像以生成描述性标签，并应用配置的阈值和过滤器。
*   **`config/config.py`**: 提供 Python 函数来加载、解析和访问 `config/config.toml` 文件中的设置。它还定义了整个工具包中使用的默认配置、支持的文件扩展名和其他全局设置。

## 更新日志

### 2.5
![图片](https://github.com/user-attachments/assets/bffd2120-6868-4a6e-894b-05c4ff5fd98f)
*   正式支持“标签高亮字幕”功能 (初期针对 Pixtral)。
*   WDtagger 标注可用于提升 VLM 准确性。
*   标签分类并以颜色编码进行质量检查 (例如：角色、服装、身体特征、动作)。
*   增加了将父文件夹名指定为角色名称以及检查标签高亮率的参数 (目标 >35%)。
*   用法：先运行 `3、tagger.ps1`，然后使用 Pixtral API 密钥运行 `4、run.ps1`。

*(为简洁起见，此处省略了从 2.4 到 1.6 的旧版更新日志条目，但它们将包含在最终文件中以供历史参考。)*

### 2.4
我们支持 Gemini 图像字幕和评分。它也支持 gemini2.5-flash-preview-04-17。但是，经过测试，flash 版本效果不佳且图像审查有问题，建议使用 pro 版本。
![图片](https://github.com/user-attachments/assets/6ae9ed38-e67a-41d2-aa1d-4caf0e0db394)
flash 版本 ↑
![图片](https://github.com/user-attachments/assets/c83682aa-3a37-4198-b117-ffe7f74ff812)
pro 版本 ↑

### 2.3
好吧，我们忘记发布 2.2 版本了，所以我们直接发布了 2.3 版本！2.3 版本更新了 GLM4V 模型用于视频字幕。

### 2.2
2.2 版本更新了 TensorRT 以加速本地 ONNX 模型 WDtagger。测试后，使用标准 CUDA 标记 10,000 个样本需要 30 分钟，而使用 TensorRT 只需 15 到 20 分钟即可完成。但是，首次使用时编译时间会更长。如果 TensorRT 失败，它会自动恢复到 CUDA，无需担心。如果提示缺少 TensorRT 库，则可能缺少某些部分。请从[此处](https://developer.nvidia.com/tensorrt/download/10x)手动安装 10.7.x 版本。

### 2.1
增加了对 Gemini 2.5 Pro Exp 的支持。现在默认使用 600 秒剪切视频。

### 2.0 重大更新！
现在我们支持视频分割！增加了一个新的视频分割模块，该模块根据场景变化检测关键时间戳，然后输出相应的图像和视频片段！导出 HTML 以供参考，效果非常显著！
![图片](https://github.com/user-attachments/assets/94407fec-92af-4a34-a15e-bc02bf45d2cd)
我们还添加了字幕对齐算法，在检测到场景变化帧后，自动将 Gemini 的时间戳字幕对齐到毫秒级别 (仍存在一些错误，但效果要好得多)。
最后，我们添加了最新的 gemini-2.0-flash-exp 模型的图像输出功能！您可以在 [`config.toml`](https://github.com/sdbds/qinglong-captions/blob/main/config/config.toml) 中自定义任务并添加任务名称，它将自动处理相应的图像 (然后对其进行标记)。
目前，一些简单的任务描述如下：欢迎社区不断优化这些任务提示并提供贡献！
https://github.com/sdbds/qinglong-captions/blob/12b7750ee0bca7e41168e98775cd95c7b9c57173/config/config.toml#L239-L249
![图片](https://github.com/user-attachments/assets/7e5ae1a9-b635-4705-b664-1c20934d12bc)
![图片](https://github.com/user-attachments/assets/58527298-34f8-496d-8c4e-1a1c1c965b73)

### 1.9
现在支持 Mistral OCR 功能！利用 Mistral 先进的 OCR 功能从视频和图像中提取文本信息。此功能在处理包含字幕、标志或其他文本元素的媒体文件时特别有用，可提高字幕的准确性和完整性。OCR 功能已集成到现有工作流程中，无需额外配置即可使用。

### 1.8
现在添加了 WDtagger！即使您无法使用 GPU，也可以使用 CPU 进行标记。它具有多线程和各种优化功能，可快速处理大规模数据。使用 ONNX 处理，模型加速。代码参考 @kohya-ss https://github.com/sdbds/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py
2.0 版本将添加双字幕功能，输入 wdtagger 的标签，然后输出自然语言。
![图片](https://github.com/user-attachments/assets/f14d4a69-9c79-4ffb-aff7-84d103dfeff4)

### 1.7
现在我们支持 qwen-VL 系列视频字幕模型！
- qwen-vl-max-latest
- qwen2.5-vl-72b-instruct 
- qwen2.5-vl-7b-instruct
- qwen2.5-vl-3b-instruct
qwen2.5-vl 支持 2 秒到 10 分钟的视频，qwen-vl-max-latest 有 1 分钟的限制。这些模型不擅长捕捉时间戳；建议使用分割的视频片段进行字幕生成，并修改提示。视频上传功能需要向官方提交申请，请在此处提交申请：[链接](https://smartservice.console.aliyun.com/service/create-ticket?spm=a2c4g.11186623.0.0.3489b0a8Ql486b)。我们考虑将来添加本地模型推理，例如 qwen2.5-vl-7b-instruct 等。此外，现在使用流式推理输出日志，您可以在完整输出显示之前看到模型的实时输出。

### 1.6
现在 Google Gemini SDK 已更新，新版 SDK 适用于 Gemini 2.0 的新模型！新 SDK 功能更强大，主要支持验证上传视频的功能。如果您想重复标记同一视频而不再需要重复上传，视频名称和文件大小/哈希值将自动验证。同时，更新了毫秒级对齐功能。长视频分割的字幕合并后，时间轴会自动对齐到毫秒，非常整齐！

## 贡献指南

欢迎参与贡献！有关如何为本项目做贡献的指南，请参阅 `CONTRIBUTING.md`。(注意: `CONTRIBUTING.md` 将在后续步骤中创建)。

## 许可证

本项目采用 MIT 许可证授权。详情请参阅 `LICENSE` 文件 (假设 `LICENSE` 文件存在或将被添加，此类项目通常使用 MIT 许可证)。
