# 青龙字幕工具

AI 多模态媒体处理与文档翻译工具，围绕 Lance 数据集提供视频、图像、音频描述、OCR、标签生成、翻译、音频分轨和 Image2PSD 工作流。

当前版本：`4.5.0` · [English README](README.en.md) · [更新日志](CHANGELOG.md)

## 先看这里

| 你的目标 | 推荐入口 | 结果 |
| --- | --- | --- |
| 视频 / 图像批量字幕 | GUI：`Import -> Split -> Tagger -> Caption -> Export` | Lance 数据集与字幕文件 |
| OCR 或本地 VLM / ALM | GUI 的 `Caption` 页面 | OCR、图像描述或音频转写 |
| 文本 / 文档翻译 | GUI 的 `Tools` 页面或 `5.translate.ps1` | `*_zh_cn.md` 等语言后缀文件 |
| 音频分轨 | GUI 的 `Tools` 页面或 `2.5.audio_separator.ps1` | 6 stems，可选 harmony 二次分离 |
| 乐谱扫描 embedding | GUI 的 `Tools` 页面或 `module.sheet_music_musvit` | MuSViT ONNX embedding 与 manifest |
| 单图转分层 PSD | GUI 的 `Tools` 页面或 `2.6.image2psd.ps1` | 分层 PSD 与中间结果 |
| 批处理 / 自动化 | 对应 `.ps1` 或 Python CLI | 可脚本化的离线流程 |

详细页面说明见 [GUI 使用手册](gui/README.md)，配置见 [配置指南](docs/configuration.md)，故障排查见 [故障排查](docs/troubleshooting.md)。

## 功能范围

- GUI 优先的导入、场景分割、WDTagger、字幕、导出与工具箱工作流
- Provider V2：云端 VLM、OpenAI-compatible、本地 VLM、OCR 与本地 ALM
- Lance 数据集导入、版本标签、Blob v2 读写与字幕导出
- 文本和文档规范化、分块翻译与语言后缀 Markdown 导出
- ONNX 音频分轨、WDTagger、WaterDetect、MusViT 等工具
- Image2PSD / See-through：LayerDiff 透明分层、Marigold 深度估计和 PSD 导出

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

## 使用方法

### 1. GUI 快速开始（推荐）

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

### 2. 推荐工作流

1. **Setup**：检查 Python、PyTorch、CUDA、模型缓存和环境变量。
2. **Import**：选择输入目录，导入 Lance 数据集；需要时配置 caption sidecar、tag 和导入模式。
3. **Split**：对视频目录执行场景检测并生成场景图像；不处理视频时可跳过。
4. **Tagger**：使用 WDTagger / CL Tagger 生成标签；首次使用 gated 模型前，先在 Hugging Face 接受条款并设置 `HF_TOKEN`。
5. **Caption**：选择云端 API、OpenAI-compatible、本地 OCR、VLM 或 ALM 路由，确认输入类型和模型配置后执行。
6. **Export**：从 Lance 数据集导出图片、媒体和字幕，必要时指定版本 tag 或语言后缀。
7. **Tools**：按需运行预处理、评分、WaterDetect、音频分轨、文本翻译或 Image2PSD。

任务会显示在 GUI 的任务标签页中。切换页面不会停止后台任务；停止任务请使用对应任务的 Stop 控件。

### 3. 配置 Provider 与模型

GUI 配置入口优先于手工修改脚本。配置原则：

- 云端 Provider：在 `Caption` 页面填写对应 API Key、模型和 Base URL。
- OpenAI-compatible：填写 `openai_base_url`、`openai_model_name`；本地服务可使用占位 API Key。
- 本地 OCR / VLM / ALM：选择路由后，GUI 会为该路由补齐所需 `uv extra`，并显示显存适配提示。
- Hugging Face gated / 私有模型：在环境变量中配置 `HF_TOKEN`，不要把令牌写入提交的 `.ps1` 或 Markdown。
- 详细 OpenAI-compatible 示例见 [docs/openai_compatible.md](docs/openai_compatible.md)。

GUI 环境变量设置保存在 `config/env_vars.json`。该文件是明文本地状态，可能包含令牌或代理信息，**不要提交、分享或上传**；提交前请确认它不在 Git 暂存区。

### 4. 脚本模式（批处理）

PowerShell 入口把配置集中放在文件顶部。先编辑脚本顶部的 `Configuration` 区域，再从仓库根目录运行：

| 脚本 | 用途 | 典型输入 / 输出 |
| --- | --- | --- |
| `lanceImport.ps1` | 导入 Lance | 图片、视频或数据目录 -> `.lance` |
| `2.0.video_spliter.ps1` | 场景检测和视频切分 | 视频目录 -> 场景图像 / 报告 |
| `3.tagger.ps1` | WDTagger / CL Tagger | 图片目录 -> tags sidecar / Lance 更新 |
| `4.captioner.ps1` | 批量字幕生成 | Lance 或媒体目录 -> caption / SRT |
| `lanceExport.ps1` | 导出 Lance | `.lance` -> 媒体与字幕文件 |
| `2.1.image_watermark_detect.ps1` | 水印检测 | 图片目录 -> 检测结果 |
| `2.2.preprocess_images.ps1` | 缩放、裁剪和可选对齐 | 图片目录 -> 预处理图片 |
| `2.3.image_reward_model.ps1` | 图像质量评分 | 图片目录 -> 评分结果 |
| `2.4.psdexport.ps1` | PSD 图层导出 | PSD 目录 -> 图层图片 / 可选 Lance |
| `2.5.audio_separator.ps1` | ONNX 音频分轨 | 音频 / 目录 / `.lance` -> wav/flac/mp3 stems |
| `2.6.image2psd.ps1` | See-through 分层 | 图片目录 -> 分层 PSD |
| `5.translate.ps1` | 文本规范化和翻译 | 文档目录 / `.lance` -> `*_lang.md` |

例如：

```powershell
.\lanceImport.ps1
.\4.captioner.ps1
.\lanceExport.ps1
.\5.translate.ps1
```

脚本会在需要时按 profile 增量安装依赖。不要手工拼接一长串 `uv sync --extra`；如果需要精确控制 profile，参见 [配置指南](docs/configuration.md)。

### 5. Python CLI（高级）

安装脚本在子 PowerShell 进程中激活 `.venv`，不会改变当前终端。直接运行 Python CLI 前，先在当前终端激活项目环境：

```powershell
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1

# Linux Bash
source .venv/bin/activate
```

基础入口可以直接查看帮助；可选工具会在 argparse 之前导入模型依赖，必须先运行对应 wrapper 安装 profile。WaterDetect 使用自己的 PEP 723 `uv run` 环境：

| 入口 | 依赖准备 | 帮助命令 |
| --- | --- | --- |
| `module.lanceImport` | 基础依赖 | `python -m module.lanceImport --help` |
| `module.lanceexport` | 基础依赖 | `python -m module.lanceexport --help` |
| `module.captioner` | 基础依赖；本地路由按配置安装 extra | `python -m module.captioner --help` |
| `module/waterdetect.py` | PEP 723 脚本依赖 | `uv run module/waterdetect.py --help` |
| `module.texttranslate` | 先运行 `5.translate.ps1`（`translate`） | `python -m module.texttranslate --help` |
| `utils.psd_dataset_pipeline` | 先运行 `2.4.psdexport.ps1`（`psdexport`） | `python -m utils.psd_dataset_pipeline --help` |
| `module.audio_separator` | 先运行 `2.5.audio_separator.ps1`（`vocal-midi`） | `python -m module.audio_separator --help` |
| `module.muscriptor_tool.cli` | 先运行 `2.7.music_transcription.ps1`（`muscriptor-local`） | `python -m module.muscriptor_tool.cli --help` |
| `module.sheet_music_musvit` | GUI Tools 或先安装 `musvit-onnx` | `python -m module.sheet_music_musvit --help` |
| `module.see_through.cli` | 先运行 `2.6.image2psd.ps1`（`see-through`） | `python -m module.see_through.cli --help` |
| `utils.preprocess_datasets` | 先运行 `2.2.preprocess_images.ps1`（`image-align`） | `python -m utils.preprocess_datasets --help` |

关键行为：

- `module.lanceImport` 的 `--data_storage_version` 控制新建数据集格式，默认脚本配置为 `2.2`。
- `module.lanceexport` 可用 `--version`、`--caption_suffix` 和 `--caption_extension` 控制导出版本与文件名。
- `module.texttranslate` 支持 `--normalize_only`、`--skip_normalize`、`--no_export` 和 `--runtime_backend openai`。
- `module.audio_separator` 默认输出 6 stems，可追加 `--harmony_separation`。
- `module.see_through.cli` 适合批量图片；模型和中间文件会占用大量磁盘空间。

## MuScriptor 音频转 MIDI

项目集成固定版本 `muscriptor==0.2.1`，只接受三个官方模型：
[small](https://huggingface.co/MuScriptor/muscriptor-small)、
[medium](https://huggingface.co/MuScriptor/muscriptor-medium) 和
[large](https://huggingface.co/MuScriptor/muscriptor-large)，CLI、GUI 和 runtime 默认使用 `large`。不接受本地权重、URL 或自定义仓库。
首次使用前须在 Hugging Face 模型页接受条款并运行 `hf auth login`。模型权重采用 CC BY-NC 4.0，且模型页另有输入音乐权利要求；代码和 SoundFont 分别按各自许可证分发。

PowerShell 批处理入口会按项目现有方式把独立 profile 增量安装到当前 Python 环境：

```powershell
.\2.7.music_transcription.ps1 .\audio --format midi --format jsonl
```

也可以先安装 profile，再直接使用 CLI：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r pyproject.toml --extra muscriptor-local
python -m module.muscriptor_tool.cli transcribe song.wav --format midi
python -m module.muscriptor_tool.cli batch .\album --model large --device cuda:0 --format midi --format json --format jsonl
python -m module.muscriptor_tool.cli list-instruments --format json
```

批处理可组合导出 MIDI、JSON 和 JSONL，并按输入相对路径建立项目目录；未传 `--output-dir` 时，结果写入输入位置下的 `muscriptor_output`。每项文件名为 `transcription.mid`、`events.json`、`events.jsonl`、`metadata.json`，根目录另有 `manifest.json`。同一输入只推理一次；完成签名、原子写入和输出锁支持中断后恢复。默认递归处理目录；关闭“跳过已完成”即可重跑，不另设覆盖开关。

事件只包含 onset、offset、pitch 和 instrument。模型不转录原始力度，MIDI 使用上游固定力度；同一乐器同一音高的同时重叠音符无法表示，鼓按 onset-only 与上游最小时长规则输出。密集混音、少见音色、重处理音频和部分合唱材料可能降低准确度，工具不会用后处理猜测缺失信息。

试听是符号输出之外的可选项：`--preview-mode midi` 导出纯合成 MIDI 音频，`comparison` 导出左声道原音、右声道合成 MIDI 的对照音频。默认格式为 MP3，仅在当前 `soundfile/libsndfile` 通过实际读写探测时开放；WAV 可作为显式回退。试听需要 PATH 中的 FluidSynth，并自动使用 MuScriptor 官方 `MuseScore_General.sf2` SoundFont；本项目没有系统音源或自定义 SoundFont 选项。当前范围不包含上游 WebUI、钢琴卷帘或单文件 Demo 页面。

Linux 可用发行版包管理器安装，例如 `sudo apt install fluidsynth`。Windows 可从 [FluidSynth 官方 Releases](https://github.com/FluidSynth/fluidsynth/releases) 安装并把可执行文件目录加入 PATH。两端都用 `fluidsynth --version` 检测；检测失败时 MIDI、JSON、JSONL 仍可使用，只需关闭试听或先选择 WAV 排除 MP3 编码问题。

## 输入、输出与数据安全

- Lance 是导入、字幕、标签和翻译的主要中间格式；导出前请确认当前 version/tag。
- 默认导入配置不保存原始二进制 blob，具体行为以 `lanceImport.ps1` 和 GUI 选项为准。
- 翻译不会覆盖原文件，默认写出带语言后缀的 Markdown，例如 `foo_zh_cn.md`。
- 脚本和 GUI 可能把输入绝对路径、模型名和错误信息写入任务日志；分享日志前请脱敏。
- 当前部分字幕入口仍通过命令行参数传递 API Key。不要把任务日志、进程列表截图或完整命令行复制到公开渠道；长期使用建议改用环境变量或受控凭据存储。

## 目录与配置速查

| 路径 | 作用 |
| --- | --- |
| `gui/launch.py` | GUI 实际 Python 入口 |
| `gui/README.md` | GUI 页面、启动参数和故障排查 |
| `module/providers/` | Provider V2 实现 |
| `module/caption_pipeline/` | 字幕任务编排与 Lance 同步 |
| `config/model.toml` | Provider / 本地模型路由默认值 |
| `config/config.toml` | 任务和模型运行参数 |
| `config/general.toml` | 通用路径、批处理和界面默认值 |
| `config/onnx.toml` | ONNX Runtime、缓存和执行 provider |
| `config/task_tabs.toml` | GUI 任务标签页运行时设置 |
| `config/env_vars.json` | GUI 生成的本地环境变量（明文，不提交） |
| `tests/` | 单元、集成和跨平台兼容测试 |

完整配置说明见 [docs/configuration.md](docs/configuration.md)。

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

## 开发与验证

测试工具属于 `test` dependency group，不在基础安装中。先从仓库根目录把测试依赖安装到项目环境，再在当前终端激活 `.venv`（Windows 使用 `. .\.venv\Scripts\Activate.ps1`，Linux 使用 `source .venv/bin/activate`）：

```powershell
# Windows PowerShell
uv pip install --python .\.venv\Scripts\python.exe --group test

# Linux Bash
uv pip install --python .venv/bin/python --group test
```

然后运行非网络、非 GPU、非可选运行时测试：

```shell
python -m pytest tests -q --strict-markers -m "not optional_runtime and not gpu and not network"
```

CI 当前还会运行：

```shell
python -m ruff check module gui utils config tests --select F821,F823
```

提交前请不要把 `config/env_vars.json`、模型缓存、数据集、日志或本地凭据加入 Git。

## 上游与引用

- [See-through](https://github.com/shitagaki-lab/see-through)：Image2PSD 上游
- [GAME](https://github.com/openvpi/GAME)：`vocal-midi` 使用的音高与分段模型参考
- 其他 Provider 和模型的来源、许可证与限制请以对应上游仓库为准
- 完整历史见 [CHANGELOG.md](CHANGELOG.md)

## 许可证

本项目使用仓库根目录的 [LICENSE](LICENSE)。第三方目录和模型仍受各自许可证、使用条款与访问协议约束。
