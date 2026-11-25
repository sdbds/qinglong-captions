# Dettato 多乐器转 MIDI

在 **工具 → 音乐转录 → Dettato** 选择原始混音文件或文件夹。每首歌生成一个 MIDI，识别出的乐器各占一轨，检测到的打击乐写入鼓轨。它不输出分离音频，也不把演奏量化成乐谱。

```powershell
.\2.7.music_transcription.ps1 -Backend dettato ".\我的歌.wav"
.\2.7.music_transcription.ps1 -Backend dettato .\album --device=cuda:0 --output-dir .\midi
```

默认递归扫描 WAV、FLAC、MP3、M4A、OGG、AAC，按相对路径保存。例如 `album/disc1/song.flac` 生成 `album/dettato_output/disc1/song.flac/song.mid`，旁边的 `metadata.json` 记录音色、各轨音符数、调音估计和鼓模型结果。同目录下的 `song.wav` 不会覆盖它。重复运行跳过已完成的相同输入；用 `--no-skip-completed` 强制重算。

- 启动器把 `dettato-local` 安装到项目共享环境，复用现有 PyTorch。不指定 `-Backend` 时仍使用 MuScriptor。已安装该依赖时也可直接运行 `python -m module.dettato_tool --help`。
- 首次转录会下载[四个官方模型](https://huggingface.co/collections/kobimusic/dettato)，同一批文件只加载一次。依赖和权重均已缓存时，设置 `UV_OFFLINE=1` 和 `HF_HUB_OFFLINE=1` 可离线运行。已有独立权重目录时，也可将 `DETTATO_WEIGHTS` 指向四个模型子目录的父目录；更新权重后用 `--no-skip-completed` 重算。设备默认自动选择，CLI 也支持 `cpu`、`cuda`、`cuda:N` 或 `mps`。
- 鼓模型只处理采样率至少 40 kHz 的输入。低采样率仍可导出旋律乐器，跳过鼓的原因会写入日志和 JSON。优先使用原始 44.1 / 48 kHz 音频，不要为了触发鼓模型而升采样。
- 部分压缩格式需要 `ffmpeg` 和 `ffprobe` 在 PATH 中；导出 MIDI 不需要音源或 FluidSynth。密集混音可能出现音符被分到错误乐器的情况，详见[上游说明](https://github.com/kobimusic/dettato#limits)。
