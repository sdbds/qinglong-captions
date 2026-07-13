# 青龙字幕工具

AI 多模态媒体处理与文档翻译工具，围绕 Lance 数据集提供视频、图像、音频描述、OCR、标签生成、翻译、音频分轨和 Image2PSD 工作流。

当前版本：`4.6.0` · [English README](README.md) · [更新日志](CHANGELOG.md)

## 4.6.0 版本重点

- 音频分轨页新增“全部分轨转 MIDI”二级选项：人声、鼓、贝斯、吉他和钢琴使用对应音色家族，`other` 支持自动识别或手动指定。
- 同一设置支持纯 MIDI 试听，或“左原分轨 / 右合成 MIDI”对照试听，格式可选 MP3 或 WAV，且不会重复执行 MuScriptor 推理。
- `muscriptor-local` profile 补齐 SOCKS 代理支持，官方 SoundFont 下载失败时，任务最终日志会直接显示共同根因。

## 先看这里

| 你的目标 | 推荐入口 | 结果 |
| --- | --- | --- |
| 视频 / 图像批量字幕 | GUI：`Import -> Split -> Tagger -> Caption -> Export` | Lance 数据集与字幕文件 |
| 视频镜头切分 | GUI `Split` 或 `2.0.video_spliter.ps1` | 场景区间、代表帧和报告 |
| 图片内容打标 | GUI `Tagger` 或 `3.tagger.ps1` | WDTagger / CL Tagger 标签 |
| OCR、VLM、ALM 字幕 | GUI `Caption` 或 `4.captioner.ps1` | 描述、OCR、音频转写或字幕 |
| 其他独立工具 | [工具文档索引](docs/tools/README.md) | 预处理、评分、音频、PSD、翻译等 |

详细页面说明见 [GUI 使用手册](gui/README.md)，配置见 [配置指南](docs/configuration.md)，故障排查见 [故障排查](docs/troubleshooting.md)。

## 能做什么

- 将长视频按镜头切分并提取代表帧
- 使用 WDTagger / CL Tagger 批量生成图片标签
- 使用云端或本地模型生成画面描述、OCR、音频转写和字幕
- 用 Lance 保存处理中间结果，再导出媒体、标签和字幕文件
- 通过独立工具完成水印检测、图片预处理、质量评分、音频分轨、翻译和 PSD 导出

## 环境要求

- Windows 或 Linux（仓库内的 Linux PowerShell 安装脚本目前只提供 x86_64 下载；ARM64 或其他架构请手动安装 `pwsh` 7+）
- Python `>=3.10,<3.13`；安装脚本默认创建 Python 3.11 环境
- `uv`：安装脚本会尝试自动安装；若终端找不到，请重新打开终端后再运行
- Windows 使用 PowerShell 5.1+；Linux 需要 `pwsh`
- GPU 不是所有功能的必需条件，但本地 VLM、OCR、翻译和 Image2PSD 通常需要显存与较大的磁盘空间
- 第一次使用某个本地路由时会下载依赖和模型，模型默认放在 `huggingface/`

## 安装

### Windows

在仓库根目录执行：

```powershell
.\1.install-uv-qinglong.ps1
```

基础安装只安装 `pyproject.toml` 的默认依赖。GUI 或工具第一次选中某个本地 Provider 时，会按路由增量安装对应的 `extra`。

### Linux

仓库文件名包含空格，命令必须保留引号：

```bash
chmod +x "./0.install pwsh.sh"
sudo bash "./0.install pwsh.sh"
pwsh ./1.install-uv-qinglong.ps1
```

安装脚本会创建或复用仓库内的 `.venv` 或 `venv`。脚本会从 `pyproject.toml` 解析依赖，不依赖仓库内的 `uv.lock`。仓库内的辅助脚本固定下载 Linux x86_64 版 PowerShell；ARM64/其他架构请跳过该脚本，按发行版文档安装 `pwsh` 7+。下文命令以 `.venv` 为例；如果实际使用的是 `venv`，请将路径中的 `.venv` 替换为 `venv`。

### 安装后自检

```powershell
uv --version
uv run gui/launch.py --help
```

如果 `uv` 刚刚安装但当前 shell 找不到它，请重启 PowerShell / 终端后重试。

## 五步完成一次任务

### 1. 启动 GUI

```powershell
.\start_gui.ps1
```

默认浏览器地址为 `http://127.0.0.1:7899`。端口被占用时，GUI 会自动尝试后续端口并在终端打印实际地址。

`start_gui.ps1` 实际执行 `uv run gui/launch.py`。`gui/launch.py` 带有 PEP 723 依赖声明，会使用 GUI 自己的隔离运行时；不要把它理解为自动复用项目 `.venv` 的入口。

如果要直接运行 Python 入口，请先在当前终端激活已经安装好 NiceGUI 的项目环境（安装脚本子进程不会改变当前终端）：

```powershell
. .\.venv\Scripts\Activate.ps1
python -m gui.launch --port 7899
```

Linux Bash 使用 `source .venv/bin/activate` 后执行同一条 `python` 命令。

常用参数：

```powershell
uv run gui/launch.py --port 7899 --no-browser
uv run gui/launch.py --native --port 7899
```

- `--no-browser`：不自动打开浏览器
- `--native`：使用原生窗口模式，需要 `pywebview`
- `--cloud`：绑定 `0.0.0.0`，仅适合受信任的内网或已由外部网关保护的环境

GUI 本身没有登录鉴权。不要把 `--cloud` 直接暴露到公网，也不要在不可信网络中使用它。

### 2. 导入素材

打开 `Import`，选择视频、图片或已有数据目录，设置输出 Lance 路径后开始导入。第一次使用建议保留默认数据版本和存储选项。Lance 是后续分割、打标、字幕和导出的共同数据集。

### 3. 分割视频

打开 `Split` 并选择刚导入的数据集。普通视频先使用默认场景检测器；切分过碎时提高阈值或最短场景长度，漏掉镜头时降低阈值。只处理图片时可以跳过此步。

### 4. 生成标签和字幕

在 `Tagger` 中选择 WDTagger 或 CL Tagger 生成标签。随后进入 `Caption`：

1. 选择云端 API、OpenAI-compatible 服务或本地模型。
2. 填写模型、API Key 和 Base URL；本地模型按界面提示安装依赖。
3. 选择提示词与输出字段，先用少量样本确认结果。
4. 确认后运行完整任务。切换页面不会停止后台任务。

只需要字幕时可以跳过 Tagger；只需要标签时也不必运行 Caption。

### 5. 导出结果

打开 `Export`，选择数据集版本、字幕后缀和文件格式，再导出媒体与 sidecar 字幕。覆盖已有文件前先用少量样本检查文件名和编码。详细说明见 [Import / Export 文档](docs/tools/import_export.md)。

## 核心功能

青龙字幕工具的主流程是 `Import -> Split -> Tagger -> Caption -> Export`。任务会显示在 GUI 的任务标签页中；切换页面不会停止后台任务。

### 视频分割

Split 使用场景检测把长视频拆成镜头区间，并按需提取代表帧。支持 Content、Adaptive、Hash、Histogram 和 Threshold 检测器，可调整阈值、最短场景长度、递归扫描和 HTML 报告。完整输入输出和调参说明见 [视频分割文档](docs/tools/video_split.md)。

### 图像打标

Tagger 使用 WDTagger 或 CL Tagger 为图片与 Lance 数据集生成内容标签。可分别设置通用、角色和概念阈值；gated 模型需要先接受 Hugging Face 条款并配置 `HF_TOKEN`。完整说明见 [图像打标文档](docs/tools/tagging.md)。

### 字幕与多模态描述

Caption 负责核心推理：它可连接云端 API、OpenAI-compatible 服务、本地 OCR、本地 VLM 和本地 ALM，处理图片、视频、音频或文档，并把结果写回 Lance 或字幕文件。Provider、分段、重试、prompt 和输出模式见 [字幕生成文档](docs/tools/captioning.md)。

数据导入、版本选择与字幕导出见 [Import / Export 文档](docs/tools/import_export.md)。

## 其他工具

| 功能 | 独立文档 | 入口 |
| --- | --- | --- |
| 水印检测 | [WaterDetect](docs/tools/watermark_detection.md) | `2.1.image_watermark_detect.ps1` |
| 图片预处理与对齐 | [Preprocess](docs/tools/image_preprocessing.md) | `2.2.preprocess_images.ps1` |
| 图像质量评分 | [Reward Model](docs/tools/image_scoring.md) | `2.3.image_reward_model.ps1` |
| PSD 图层导出 | [PSD Export](docs/tools/psd_export.md) | `2.4.psdexport.ps1` |
| 音频分轨 | [Audio Separation](docs/tools/audio_separation.md) | `2.5.audio_separator.ps1` |
| Image2PSD | [See-through](docs/tools/image2psd.md) | `2.6.image2psd.ps1` |
| 音频转 MIDI | [MuScriptor](docs/tools/muscriptor.md) | `2.7.music_transcription.ps1` |
| 乐谱扫描 embedding | [MuSViT](docs/tools/sheet_music.md) | GUI Tools |
| 文本与文档翻译 | [Translation](docs/tools/text_translation.md) | `5.translate.ps1` |

MuScriptor 通过 `muscriptor-local` profile 安装，支持官方 `small`、`medium`、`large` 模型。只导出 MIDI、JSON 或 JSONL 时不需要音频合成器。启用试听后，运行时会在模型推理前执行预检；FluidSynth 或官方 SoundFont 不可用会终止整批任务，因此需要关闭试听才能在没有试听运行时的情况下只导出符号结果。该 profile 已包含首次下载 SoundFont 所需的 SOCKS 代理支持。

可选的纯 MIDI 试听或“左声道原音、右声道合成 MIDI”对照试听，无论选择 MP3 还是 WAV，都要求系统 `PATH` 中存在原生 [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) 可执行文件。改用 WAV 不能绕过该要求；MP3 还要求当前 `soundfile/libsndfile` 支持 MP3 编码。Windows 请使用 x64 版本，把解压后的 `bin` 目录加入 `PATH`，重启终端和 GUI 后用 `fluidsynth --version` 验证。官方 `MuseScore_General.sf2` SoundFont 会自动解析，不需要系统音源或自定义 SoundFont。

全部入口见 [工具文档索引](docs/tools/README.md)。

## 选择字幕模型

GUI 配置入口优先于手工修改脚本。配置原则：

- 云端 Provider：在 `Caption` 页面填写对应 API Key、模型和 Base URL。
- OpenAI-compatible：填写 `openai_base_url`、`openai_model_name`；本地服务可使用占位 API Key。
- 本地 OCR / VLM / ALM：选择路由后，GUI 会为该路由补齐所需 `uv extra`，并显示显存适配提示。
- Hugging Face gated / 私有模型：在环境变量中配置 `HF_TOKEN`，不要把令牌写入提交的 `.ps1` 或 Markdown。
- 详细 OpenAI-compatible 示例见 [docs/openai_compatible.md](docs/openai_compatible.md)。

GUI 环境变量设置保存在 `config/env_vars.json`。该文件是明文本地状态，可能包含令牌或代理信息，**不要提交、分享或上传**；提交前请确认它不在 Git 暂存区。

## 批量脚本

PowerShell 入口把配置集中放在文件顶部。先编辑脚本顶部的 `Configuration` 区域，再从仓库根目录运行：

| 脚本 | 用途 | 典型输入 / 输出 |
| --- | --- | --- |
| `lanceImport.ps1` | 导入 Lance | 图片、视频或数据目录 -> `.lance` |
| `2.0.video_spliter.ps1` | 场景检测和视频切分 | 视频目录 -> 场景图像 / 报告 |
| `3.tagger.ps1` | WDTagger / CL Tagger | 图片目录 -> tags sidecar / Lance 更新 |
| `4.captioner.ps1` | 批量字幕生成 | Lance 或媒体目录 -> caption / SRT |
| `lanceExport.ps1` | 导出 Lance | `.lance` -> 媒体与字幕文件 |

例如：

```powershell
.\lanceImport.ps1
.\4.captioner.ps1
.\lanceExport.ps1
.\5.translate.ps1
```

脚本会在需要时按 profile 增量安装依赖。不要手工拼接一长串 `uv sync --extra`；如果需要精确控制 profile，参见 [配置指南](docs/configuration.md)。

## 输入、输出与数据安全

- Lance 是导入、字幕、标签和翻译的主要中间格式；导出前请确认当前 version/tag。
- 默认导入配置不保存原始二进制 blob，具体行为以 `lanceImport.ps1` 和 GUI 选项为准。
- 翻译不会覆盖原文件，默认写出带语言后缀的 Markdown，例如 `foo_zh_cn.md`。
- 脚本和 GUI 可能把输入绝对路径、模型名和错误信息写入任务日志；分享日志前请脱敏。
- 当前部分字幕入口仍通过命令行参数传递 API Key。不要把任务日志、进程列表截图或完整命令行复制到公开渠道；长期使用建议改用环境变量或受控凭据存储。

## 故障排查

1. **找不到 `uv`**：重启终端；确认 `uv --version` 可执行，再重新运行安装脚本。
2. **GUI 启动失败**：优先使用 `start_gui.ps1`，不要从 `gui/` 子目录猜入口；检查 `uv run gui/launch.py --help`。
3. **端口冲突**：指定 `--port`，或使用终端输出的自动切换地址。
4. **本地模型缺依赖**：回到 `Caption` / `Tools` 页面重新选择路由，让 GUI 补齐 profile；不要混装互相冲突的 OCR / CUDA extra。
5. **Hugging Face 403**：确认模型访问条款已接受，`HF_TOKEN` 已注入当前运行环境。
6. **显存不足**：降低批大小、分辨率或并发数，启用 CPU/offload 选项，并清理不再使用的模型缓存。
7. **翻译或 Lance 更新失败**：确认输入是有效目录或 `.lance`，先运行 `--normalize_only` 验证规范化链路，再查看任务日志。
8. **云端模式风险**：`--cloud` 没有内置鉴权；停止服务并改回本机绑定，或放在已有认证与 TLS 的反向代理后面。

更多处理步骤见 [docs/troubleshooting.md](docs/troubleshooting.md)。

## 上游与引用

- `Image2PSD / see-through` 对接上游仓库：[shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through)。当前仓库中的 `module/see_through/` 是针对青龙工具链做的抽取与批处理适配，不是逐文件镜像。
- `vocal-midi` 路径使用的音高与分段模型参考项目：[openvpi/GAME](https://github.com/openvpi/GAME)。GAME 当前未在 README 提供官方 BibTeX；下方保留旧版仓库级引用，上游发布正式 citation 后应以上游为准。

```bibtex
@article{lin2026seethrough,
  title={See-through: Single-image Layer Decomposition for Anime Characters},
  author={Lin, Jian and Li, Chengze and Qin, Haoyun and Chan, Kwun Wang and Jin, Yanghua and Liu, Hanyuan and Choy, Stephen Chun Wang and Liu, Xueting},
  journal={arXiv preprint arXiv:2602.03749},
  year={2026}
}
```

```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

```bibtex
@software{openvpi_game,
  title={GAME: Generative Adaptive MIDI Extractor},
  author={{OpenVPI}},
  url={https://github.com/openvpi/GAME}
}
```

其他 Provider 和模型的来源、许可证与限制请以对应上游仓库为准。完整历史见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

本项目使用仓库根目录的 [LICENSE](LICENSE)。第三方目录和模型仍受各自许可证、使用条款与访问协议约束。
