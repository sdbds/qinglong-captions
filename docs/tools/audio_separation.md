# 音频分轨

音频分轨工具使用 ONNX 模型把音乐拆为六个 stems，并可继续进行 harmony 分离、GAME 人声转 MIDI，或 MuScriptor 全分轨转 MIDI。

## 入口

```powershell
.\2.5.audio_separator.ps1
```

Python 入口：`python -m module.audio_separator --help`。基础依赖 profile 为 `vocal-midi`；启用 MuScriptor 分轨 MIDI 时还会加载 `muscriptor-local`。

## 输入与输出

- 输入：单个音频文件、目录或 `.lance` 数据集。
- 输出：`wav`、`flac` 或 `mp3` stems。
- 默认输出六轨；`harmony_separation` 开启后增加二次人声/和声处理。
- `muscriptor_midi` 开启后，本次生成的六个主分轨会写入每首歌的 `04_stem_midi` 目录。
- MuScriptor 二级设置可为每个分轨同时生成纯 MIDI 试听，或“左声道原分轨 / 右声道 MIDI 合成”的对照试听；格式支持 MP3 和 WAV。
- `segment_size`、`overlap` 和 `batch_size` 控制推理速度、边界质量和显存占用。

## MuScriptor 分轨音色

分轨标签只在信息可靠的范围内约束 MuScriptor：

- 人声固定为 `voice`，鼓轨固定为 `drums`。
- 贝斯在原声/电贝斯中识别，吉他在原声/清音电吉他/失真电吉他中识别，钢琴在原声/电钢琴中识别。
- `other` 默认自动识别，也可以在二级设置中手动限定一个或多个音色。

每轨只执行一次转录；试听直接复用这次转录生成的 MIDI 事件，不会再次运行模型。MIDI、可选的 `.preview.mp3` / `.preview.wav` 和 `.metadata.json` 写入同一目录，metadata 记录约束音色、实际识别音色、试听配置和缓存签名。如果当前模型及输出格式对应的主分轨已经存在，工具会直接复用这些确定路径，不重新执行分轨推理。

试听需要系统 `PATH` 中可用的 FluidSynth，并使用 MuScriptor 官方默认 `MuseScore_General.sf2` SoundFont。开启试听时会在模型加载前统一检查依赖；检查失败不会删除已经存在的 MIDI。MuScriptor 官方权重需要先接受 Hugging Face 条款并运行 `hf auth login`。

首次运行需要下载相应模型。排查问题时先关闭 harmony、GAME 人声 MIDI 与 MuScriptor 分轨 MIDI，只验证基础六轨输出。
