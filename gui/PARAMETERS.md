# GUI 参数映射

本文档只记录当前 GUI 页面和仓库入口的参数边界。它不是 `config/*.toml` 的完整副本；GUI 默认值以页面实现和 TOML 配置为准，脚本默认值以 wrapper 顶部、TOML 配置和 `--help` 输出为准。

## 参数来源边界

GUI 和 PowerShell wrapper 不是一条全局覆盖链：GUI 不会读取 `4.captioner.ps1` 等脚本顶部的 `Configuration` 区域。

GUI 运行时：

1. 页面当前值转换为 Python CLI 参数
2. `config/env_vars.json` 注入环境变量
3. `config/*.toml` 提供路由、模型和 runtime 默认值
4. Python CLI 的 argparse / Provider 默认值兜底

脚本运行时：

1. wrapper 顶部 `Configuration` 生成显式 CLI 参数
2. Python CLI 读取 `config/*.toml` 和 Provider 默认值
3. argparse 默认值兜底

配置文件和日志都可能包含本地路径；`config/env_vars.json` 还可能包含令牌，禁止提交或分享。

## Setup 页面

| 变量 | 类型 | 说明 |
| --- | --- | --- |
| `HF_HOME` | path | Hugging Face 模型缓存目录 |
| `HF_ENDPOINT` | URL | Hugging Face endpoint / 镜像 |
| `HF_TOKEN` | secret | gated / 私有模型访问令牌 |
| `CUDA_VISIBLE_DEVICES` | text | 可见 GPU，例如 `0` 或 `0,1` |
| `UV_INDEX_URL` | URL | uv 主索引 |
| `UV_EXTRA_INDEX_URL` | text | 额外包索引，多个地址以空格分隔 |
| `UV_CACHE_DIR` | path | uv 下载缓存目录 |
| `HTTP_PROXY` / `HTTPS_PROXY` | URL | 网络代理 |

## Import 页面 / `lanceImport.ps1`

| 参数 | 说明 |
| --- | --- |
| `train_data_dir` / `dataset_dir` | 输入数据目录 |
| `caption_dir` | 可选 caption sidecar 目录 |
| `output_name` | 输出 Lance 名称 |
| `no_save_binary` | 不保存原始二进制 blob |
| `not_save_disk` | 仅在内存中加载 |
| `import_mode` | `0` 全部、`1` 视频、`2` 音频、`3` 分割 |
| `tag` | 导入版本标签 |
| `data_storage_version` | 新建 Lance 的存储格式，脚本默认 `2.2` |
| `include_text_assets` | 包含 standalone 文本资产 |

## Split 页面 / `2.0.video_spliter.ps1`

| 参数 | 说明 |
| --- | --- |
| `input_video_dir` | 输入视频目录 |
| `output_dir` | 输出目录；为空时使用输入目录 |
| `detector` | `ContentDetector`、`AdaptiveDetector`、`HashDetector`、`HistogramDetector` 或 `ThresholdDetector` |
| `threshold` | 场景检测阈值 |
| `min_scene_len` | 最短场景长度 |
| `luma_only` | 只使用亮度变化 |
| `save_html` | 保存 HTML 报告 |
| `video2images_min_number` | 每个场景保存的图像数；`0` 表示不保存 |
| `recursive` | 递归扫描子目录 |

## Tagger 页面 / `3.tagger.ps1`

| 参数 | 说明 |
| --- | --- |
| `train_data_dir` | 图片目录 |
| `repo_id` | Hugging Face 模型仓库 |
| `cl_tagger_v2_version` | CL Tagger 版本，例如 `v2_01a` |
| `model_dir` | 本地模型目录 |
| `batch_size` | 推理批大小 |
| `thresh` | 概念标签阈值 |
| `general_threshold` | general 分类阈值 |
| `character_threshold` | character 分类阈值 |
| `overwrite` | 是否覆盖已有 sidecar |

## Caption 页面 / `4.captioner.ps1`

### 输入与调度

| 参数 | 说明 |
| --- | --- |
| `dataset_path` | Lance 或媒体输入目录 |
| `pair_dir` | 可选配对图片目录 |
| `mode` | `all`、`short` 或 `long` |
| `wait_time` | 请求间等待时间 |
| `max_retries` | Provider 重试次数 |
| `segment_time` | 音频 / 视频分段长度；空值使用 Provider 默认值 |
| `dir_name` | 是否把输入目录名加入 prompt 上下文 |
| `not_clip_with_caption` | 不根据 caption 裁剪 |
| `scene_detector` / `scene_threshold` / `scene_min_len` | 场景检测参数 |

### 路由选择

| 参数 | 典型值 |
| --- | --- |
| `ocr_model` | `pixtral_ocr`、`logics_ocr`、`dots_ocr`、`paddle_ocr`、`unlimited_ocr` 等 |
| `vlm_image_model` | `moondream`、`qwen_vl_local`、`step_vl_local`、`gemma4_local`、`marlin_2b_local` 等 |
| `alm_model` | `music_flamingo_local`、`eureka_audio_local`、`acestep_transcriber_local`、`cohere_transcribe_local`、`mega_asr_local` |
| `document_image` | OCR 文档是否转为图像输入 |
| `alm_language` | 可选 ISO 639-1 语言提示 |

云端 Provider 的 API Key、模型和 Base URL 由对应字段传递。OpenAI-compatible 使用 `openai_api_key`、`openai_base_url` 和 `openai_model_name`。当前部分路径会把 Key 作为命令参数传给子进程，避免分享日志和进程列表。

## Export 页面 / `lanceExport.ps1`

| 参数 | 说明 |
| --- | --- |
| `lance_file` | Lance 数据集路径 |
| `output_dir` | 导出目录 |
| `version` | 读取的 dataset tag / version |
| `caption_suffix` | 插入 caption 文件名前的后缀，例如 `_zh_cn` |
| `caption_extension` | 可选扩展名覆盖，例如 `.md` |
| `allowed_caption_types` | 允许导出的 caption media types，逗号分隔 |
| `not_clip_with_caption` | 不根据 caption 裁剪媒体 |

## Tools 页面

工具页不共享一套固定参数。页面会把配置转换为以下入口：

| 工具 | 入口 | 依赖 profile |
| --- | --- | --- |
| 水印检测 | `module/waterdetect.py` | PEP 723 inline dependencies；使用 `uv run module/waterdetect.py` |
| 图片预处理 | `utils/preprocess_datasets.py` | `image-align` |
| 图像评分 | `module/rewardmodel.py` | `reward-model` |
| 音频分轨 | `module/audio_separator.py` | `vocal-midi` |
| 音乐转录 | `module/muscriptor_tool/cli.py` | `muscriptor-local` |
| 乐谱扫描 | `module/sheet_music_musvit.py` | `musvit-onnx` |
| 文档翻译 | `module/texttranslate.py` | `translate` |
| Image2PSD | `module/see_through/cli.py` | `see-through` |

Python 入口提供 argparse 帮助，但可选工具必须先安装对应 profile；基础安装不保证所有入口都能在导入阶段成功：

```powershell
python -m <module> --help
```

WaterDetect 使用脚本内的 PEP 723 依赖声明，应运行 `uv run module/waterdetect.py --help`。`2.4.psdexport.ps1` 是脚本专用入口，当前不属于 GUI Tools 页面。

### MuScriptor 音乐转录

| 参数 | 说明 |
| --- | --- |
| 输入路径 | 单个音频文件或目录，自动判断；目录固定递归扫描 WAV、FLAC、MP3、M4A、OGG、AAC |
| 输出位置 | 不单独配置；默认写入输入位置下的 `muscriptor_output` |
| MuScriptor 模型 | 下拉显示 `MuScriptor/muscriptor-small`、`MuScriptor/muscriptor-medium`、`MuScriptor/muscriptor-large`，默认 large；没有自定义权重来源 |
| Device | `auto`、`cpu`、`cuda` 或 `cuda:N`；显式 CUDA 不会静默回退 CPU |
| 推理批大小 | 每次前向处理的 5 秒音频块数量，不是文件数；`0` 表示运行时自动选择 |
| 音色模式 | 自动识别，或从绑定 `muscriptor==0.2.1` 的官方 35 项目录快照中手动多选；无需先安装模型 runtime |
| 解码 | `greedy`、`sampling` 或 `beam`；temperature 只作用于 sampling，beam width 必须至少为 2 |
| 输出格式 | MIDI、JSON、JSONL，可多选但至少保留一种；单输入只执行一次推理 |
| 试听 | 关闭、纯 MIDI，或左原音 / 右合成 MIDI 对照；默认 MP3，WAV 为显式回退 |
| 批处理与恢复 | GUI 只保留 skip completed；默认递归并遇错继续，关闭 skip completed 即重跑；strict EOS 和 notes 仍是推理选项 |

依赖 profile 固定 `muscriptor==0.2.1`。首次使用需要接受官方 Hugging Face 权重条款并运行 `hf auth login`；权重为 CC BY-NC 4.0。试听要求 FluidSynth，官方 `MuseScore_General.sf2` SoundFont 自动解析和缓存，不能改成系统或自定义 SoundFont。此批处理页不实现上游 WebUI、钢琴卷帘或单文件 Demo。

## 修改建议

- 常规任务：只改页面值，让 GUI 管理 profile。
- 可复现批处理：修改对应 `.ps1` 顶部的配置区，并把模型、路径和输出目录记录在项目外部。
- 新增参数：同时更新 Python argparse、GUI 页面、PowerShell wrapper、测试和本文件。
- 不要把 `config/env_vars.json`、模型缓存、数据集或日志加入 Git。
