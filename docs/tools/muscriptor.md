# MuScriptor 音频转 MIDI

项目绑定 `muscriptor==0.2.1`，把音乐音频转为 MIDI、JSON 或 JSONL 事件。支持官方 small、medium、large 三个模型，默认使用 large；不接受本地权重、URL 或自定义模型仓库。

## 使用前准备

首次使用前须在 Hugging Face 模型页接受条款并运行 `hf auth login`。模型权重采用 CC BY-NC 4.0，且模型页另有输入音乐权利要求。

## 入口

```powershell
.\2.7.music_transcription.ps1 .\audio --format midi --format jsonl
```

直接使用 CLI：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r pyproject.toml --extra muscriptor-local
python -m module.muscriptor_tool.cli transcribe song.wav --format midi
python -m module.muscriptor_tool.cli batch .\album --model large --device cuda:0 --format midi --format json --format jsonl
python -m module.muscriptor_tool.cli list-instruments --format json
```

体验 MuScriptor 官方 WebUI 时运行：

```powershell
.\2.7.1.muscriptor_webui.ps1
.\2.7.1.muscriptor_webui.ps1 -Model small -Device cuda:0 -Port 8222
```

脚本把 `muscriptor-local` 安装到项目共享 `.venv`，然后从同一环境启动官方 `muscriptor serve`，访问终端显示的地址即可。`-Model`（别名 `-ModelSize`）支持 `small`、`medium`、`large`，默认使用 `large`；设备默认 `auto`。不要使用 `uvx muscriptor serve`：`uvx` 会解析独立工具环境，不复用项目 `.venv`，可能重复安装另一套 Torch/CUDA 依赖。

## 输出与恢复

批处理按输入相对路径建立项目目录；默认输出到输入位置下的 `muscriptor_output`。每项包含以源文件名为 stem 的 `<source-stem>.mid`、`events.json`、`events.jsonl`、`metadata.json`，根目录包含 `manifest.json`。`metadata.json` 分别记录手动限制的 `instruments` 和模型实际识别到的 `detected_instruments`。同一输入只推理一次，完成签名、原子写入和输出锁支持中断恢复。

## 试听

`preview-mode midi` 导出合成 MIDI 音频；`comparison` 输出左声道原音、右声道合成 MIDI。试听需要 PATH 中的 FluidSynth，并使用官方 `MuseScore_General.sf2` SoundFont。FluidSynth 不可用时，MIDI、JSON 和 JSONL 仍可正常生成。

事件只包含 onset、offset、pitch 和 instrument。模型不恢复原始力度；密集混音、少见音色、重处理音频和部分合唱材料可能降低准确度。
