# 音频分轨

音频分轨工具使用 ONNX 模型把音乐拆为六个 stems，并可继续进行 harmony 分离或 vocal-to-MIDI 辅助处理。

## 入口

```powershell
.\2.5.audio_separator.ps1
```

Python 入口：`python -m module.audio_separator --help`。依赖 profile 为 `vocal-midi`。

## 输入与输出

- 输入：单个音频文件、目录或 `.lance` 数据集。
- 输出：`wav`、`flac` 或 `mp3` stems。
- 默认输出六轨；`harmony_separation` 开启后增加二次人声/和声处理。
- `segment_size`、`overlap` 和 `batch_size` 控制推理速度、边界质量和显存占用。

首次运行需要下载 ONNX 模型。排查问题时先关闭 harmony 与 vocal MIDI，只验证基础六轨输出。
