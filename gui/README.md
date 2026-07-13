# 青龙字幕工具 GUI

GUI 基于 NiceGUI，负责把导入、场景分割、打标、字幕、导出和工具箱统一到任务标签页中。日常启动入口是仓库根目录的 `start_gui.ps1`。

## 快速启动

先完成根目录 [中文使用说明](../README.zh-CN.md) 的安装步骤，然后从仓库根目录运行：

```powershell
.\start_gui.ps1
```

默认地址：`http://127.0.0.1:7899`。

`start_gui.ps1` 会执行：

```text
uv run gui/launch.py
```

`gui/launch.py` 带有 PEP 723 依赖声明，因此该入口使用 GUI 自己的隔离运行时。它不承诺复用仓库的 `.venv` / `venv`。需要显式指定 Python 环境时，安装脚本子进程不会改变当前终端，请先激活项目环境并确认 NiceGUI 已安装：

```powershell
. .\.venv\Scripts\Activate.ps1
python -m gui.launch --port 7899
```

Linux Bash 使用 `source .venv/bin/activate`，然后执行同一条 `python` 命令。

以上命令假定环境目录是 `.venv`；如果安装脚本复用了 `venv`，请将路径中的 `.venv` 替换为 `venv`。

## 启动参数

```powershell
uv run gui/launch.py --help
uv run gui/launch.py --host 127.0.0.1 --port 7899 --no-browser
uv run gui/launch.py --native --port 7899
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--host` | 绑定地址 | `127.0.0.1` |
| `--port` | 监听端口 | `8080`；`start_gui.ps1` 包装后为 `7899` |
| `--cloud` | 绑定 `0.0.0.0` | 关闭 |
| `--native` | 使用原生窗口，需要 `pywebview` | 关闭 |
| `--no-browser` | 不自动打开浏览器 | 关闭 |

### 远程访问安全

`--cloud` 只改变监听地址，不会增加登录鉴权。不要把它直接暴露到公网；如必须远程访问，请放在已配置认证、TLS 和访问控制的反向代理后面。默认使用 `127.0.0.1`。

## 页面与任务

### Setup

检查 Python、PyTorch、CUDA、GPU、模型缓存和环境变量。这里可以管理 `HF_HOME`、`HF_TOKEN`、代理和 uv 索引等运行时变量。

### Import

把输入目录导入 Lance 数据集，配置 caption sidecar、tag、导入模式和是否保存二进制。详见 [Import / Export 文档](../docs/tools/import_export.md)。

### Split

对视频目录执行场景检测、切分和场景图像生成。详见 [视频分割文档](../docs/tools/video_split.md)。

### Tagger

运行 WDTagger / CL Tagger，配置模型、批大小和标签阈值。详见 [图像打标文档](../docs/tools/tagging.md)。

### Caption

选择云端 API、OpenAI-compatible、本地 OCR、本地 VLM 或本地 ALM。详见 [字幕生成文档](../docs/tools/captioning.md)。

### Export

从 Lance 数据集导出媒体和字幕，可指定 version tag、caption suffix 和 caption extension。详见 [Import / Export 文档](../docs/tools/import_export.md)。

### Tools

工具箱提供：

- [水印检测](../docs/tools/watermark_detection.md)
- [图片缩放、裁剪和可选对齐](../docs/tools/image_preprocessing.md)
- [图像质量评分](../docs/tools/image_scoring.md)
- [ONNX 音频分轨与可选 harmony、GAME 人声 MIDI、MuScriptor 全分轨 MIDI / 试听](../docs/tools/audio_separation.md)
- [MuScriptor 官方模型音频转 MIDI](../docs/tools/muscriptor.md)
- [MuSViT ONNX 乐谱扫描 embedding](../docs/tools/sheet_music.md)
- [文本 / 文档规范化与翻译](../docs/tools/text_translation.md)
- [Image2PSD / See-through 分层](../docs/tools/image2psd.md)

脚本专用的 [PSD 图层导出](../docs/tools/psd_export.md) 当前不属于 GUI Tools 页面。

## GUI 到脚本的对应关系

| GUI 页面 / 工具 | PowerShell 入口 | Python 入口 | 依赖 profile |
| --- | --- | --- | --- |
| Import | `lanceImport.ps1` | `python -m module.lanceImport` | 基础依赖 |
| Split | `2.0.video_spliter.ps1` | `python -m module.videospilter` | `video-split` |
| Tagger | `3.tagger.ps1` | `python utils/wdtagger.py --help` | `wdtagger` 或 `wdtagger-cl-tagger-v2` |
| Caption | `4.captioner.ps1` | `python -m module.captioner` | 基础依赖；本地路由按配置安装 |
| Export | `lanceExport.ps1` | `python -m module.lanceexport` | 基础依赖 |
| WaterDetect | `2.1.image_watermark_detect.ps1` | `uv run module/waterdetect.py --help` | PEP 723 inline dependencies |
| Preprocess | `2.2.preprocess_images.ps1` | `python -m utils.preprocess_datasets` | `image-align` |
| Reward model | `2.3.image_reward_model.ps1` | `python -m module.rewardmodel` | `reward-model` |
| Audio separation | `2.5.audio_separator.ps1` | `python -m module.audio_separator` | `vocal-midi`；MuScriptor 分轨 MIDI 追加 `muscriptor-local` |
| Music transcription | `2.7.music_transcription.ps1` | `python -m module.muscriptor_tool.cli batch` | `muscriptor-local` |
| Sheet music | GUI Tools（无独立 PowerShell wrapper） | `python -m module.sheet_music_musvit --help` | `musvit-onnx` |
| Image2PSD | `2.6.image2psd.ps1` | `python -m module.see_through.cli` | `see-through` |
| Translation | `5.translate.ps1` | `python -m module.texttranslate` | `translate` |

Python 入口提供 `--help`，但可选工具会在 argparse 之前导入模型依赖；基础安装不保证所有入口都能直接启动。先运行对应 PowerShell 入口或安装目标 profile，再执行 Python 命令；WaterDetect 例外，直接使用其 PEP 723 `uv run` 入口。

`2.4.psdexport.ps1` 是脚本专用的 PSD 图层导出流程，当前没有接入 GUI Tools 页面。

## MuScriptor 批处理工具

模型、输出、恢复、试听与 FluidSynth 要求见 [MuScriptor 专用文档](../docs/tools/muscriptor.md)。

## 运行时与日志

- 每个任务拥有独立的状态、进度和日志缓冲区。
- 任务标签页支持并发运行；停止一个任务不会主动终止其他标签页。
- 日志可能包含输入路径、模型名、代理地址和子进程错误。
- 当前部分 Caption 路径会把 API Key 作为命令行参数传给子进程。不要分享完整命令行、进程列表截图或未经脱敏的日志。
- GUI 环境变量保存到 `config/env_vars.json`，该文件是明文，不能提交到 Git。

## 开发调试

直接运行 `gui/main.py` 仅适合页面开发，不是日常入口：

```powershell
Set-Location gui
python main.py
```

开发 GUI 组件时，优先在仓库根目录运行测试；不要为了绕过导入问题修改 `PYTHONPATH` 或从 `gui/` 目录提交运行时生成文件。

## 常见问题

### 页面打不开

```powershell
uv run gui/launch.py --help
uv run gui/launch.py --port 7900 --no-browser
```

先确认 `uv`、NiceGUI 依赖和端口；使用终端打印的实际 URL。

### 本地 Provider 缺依赖

回到 `Caption` / `Tools` 页面重新选择路由，让 GUI 增量安装对应 profile。不要在同一环境中一次性安装所有互相冲突的 OCR、CUDA 或 Transformers extra。

### 任务失败但页面仍在

打开任务标签页日志，确认输入路径、模型缓存、Hugging Face 权限和 GPU 显存。停止任务应使用任务自己的 Stop 控件。

### 主题或语言切换

主题偏好保存在浏览器本地；语言切换会刷新页面。未保存的表单值可能丢失。当前支持中文、英文、日文和韩文。

## 相关文档

- [中文使用说明](../README.zh-CN.md)
- [配置指南](../docs/configuration.md)
- [故障排查](../docs/troubleshooting.md)
- [OpenAI-compatible Provider](../docs/openai_compatible.md)
