# SheetSage2 音乐转录

[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) 从整首混音识别人声和器乐主旋律、和弦、节拍、调号与段落。它生成的是旋律和和弦谱，不是逐乐器分轨；需要多乐器音符事件时用 [MuScriptor](muscriptor.md)。

在 GUI 的「工具 → 音乐转录」选择 **SheetSage2**，输入单个音频文件或文件夹。命令行也可使用：

```powershell
.\2.7.music_transcription.ps1 -Backend sheetsage2 .\audio --device=cuda:0
.\2.7.music_transcription.ps1 -Backend sheetsage2 .\song.wav --melody-only
```

也可运行 `uv run --script module/sheetsage2_tool.py --help` 查看选项。不指定 `-Backend` 仍使用 MuScriptor。

- 将 FFmpeg 加到 `PATH`。`uv` 为 SheetSage2 创建独立的 Python 3.11 环境，不改项目 `.venv`。首次运行还会下载 SheetSage2 与其 MERT-v2-FullSong 基座；两者的权重均采用 CC BY-NC 4.0。
- 默认设备为 `auto`（有 CUDA 用 CUDA，否则用 CPU）；也可指定 `cpu`、`cuda` 或 `cuda:N`。Windows 与 Linux x64 的 GPU 环境使用 PyTorch CUDA 12.8 wheel，需要兼容的 NVIDIA 驱动。CPU 转录整首歌曲可能较慢。
- 每首歌保存到 `sheetsage2_output/<输入相对路径，含扩展名>/`。其中有 `score.abc`、`transcription.mid`、旋律及和弦分轨 MIDI、`events.json` 与节拍、和弦等 `.lab` 标注。使用 `--output-dir` 可更改输出根目录。
- `--melody-only` 从 ABC 和合成 MIDI 中移除和弦伴奏，但保留原始事件和标注。未能生成 ABC 时保留 MIDI 和标注，并报告该首未完整转录；已有乐谱则将这次的结果写入 `partial/`，不覆盖上次结果。
- 默认跳过输入、模型版本和选项都匹配的完整结果。用 `--no-skip-completed` 强制重跑。

## 前置分轨

在「工具 → 音乐转录 → SheetSage2」打开「前置分轨」，先用现有 ONNX 模型分离并保存 WAV，再逐条转录非静音主分轨。也可在「音频分轨 → 全部分轨转 MIDI」选择 SheetSage2。两个入口都在每首歌的 `04_stem_midi/sheetsage2/<分轨文件名>/` 保存 MIDI、标注及可生成的 ABC。

匹配的音频分轨会被复用。某轨缺少生成 ABC 所需的信息时，仍保存本次 MIDI 和标注。如果该轨变为静音，或新转录没有 ABC 而已有旧谱，历史结果会归档到对应目录下的 `.previous-*`，不当作当前产物。鼓轨不会因此变成逐击鼓谱；要逐乐器音符，使用 MuScriptor。不开启前置分轨时，仍直接从原始混音识别和弦和节拍。

在已安装 `vocal-midi` 依赖的项目环境里，可用 `python -m module.audio_separator .\audio --audio_files_only --sheetsage2_midi` 运行同一流程。
