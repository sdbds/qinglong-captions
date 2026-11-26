# MuScriptor 音频转 MIDI

项目绑定 [MuScriptor 0.4.0](https://github.com/muscriptor/muscriptor/releases/tag/v0.4.0)。升级的是程序，仍使用原有的 small、medium、large 三档主转录权重，默认 large；已缓存的主权重无需重新下载。

## 使用前准备

首次使用前须在 Hugging Face 模型页接受条款并运行 `hf auth login`。模型权重采用 CC BY-NC 4.0，且模型页另有输入音乐权利要求。

## 入口

```powershell
.\2.7.music_transcription.ps1 .\audio --format midi --format jsonl
.\2.7.music_transcription.ps1 .\audio --format midi --format quantized-midi --format sheets
```

直接使用 CLI：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r pyproject.toml --extra muscriptor-local
python -m module.muscriptor_tool.cli transcribe song.wav --format midi
python -m module.muscriptor_tool.cli transcribe song.wav --quantize --grid 4 --detect-tempo false --bpm 96
python -m module.muscriptor_tool.cli batch .\album --model large --device cuda:0 --format midi --format json --format jsonl
python -m module.muscriptor_tool.cli list-instruments --format json
```

体验 MuScriptor 官方 WebUI 时运行：

```powershell
.\2.7.1.muscriptor_webui.ps1
.\2.7.1.muscriptor_webui.ps1 -Model small -Device cuda:0 -BatchSize 4 -Port 8222
.\2.7.1.muscriptor_webui.ps1 -NoBrowser
```

脚本把 `muscriptor-local` 安装到项目共享 `.venv`，使用上游内置的优化 SDPA 加载模型，同时继续使用官方 WebUI 和 HTTP 服务；服务健康检查通过后自动打开浏览器。`-NoBrowser` 可关闭自动打开。`-Model`（别名 `-ModelSize`）支持 `small`、`medium`、`large`，默认使用 `large`；设备默认 `auto`。`-BatchSize` 默认 `0`：启动时直接读取已记录的模型显存曲线，按总显存选择偶数 batch，不再重新执行 BS1/BS2 校准。OOM 时会自动改用较小批次；Windows 下检测到当前进程新增共享 GPU 显存占用后，后续 batch 会减 2。每次请求完成或取消后都会释放临时张量和空闲 CUDA cache，但保留模型权重。CPU 默认使用 BS1。传入正整数可跳过 profile 选择，但仍保留显存保护。BS1 保留上游的跨块延音约束；更大的 batch 关闭该约束以换取吞吐量，块边界的音符可能变化。不要使用 `uvx muscriptor serve`：`uvx` 会解析独立工具环境，不复用项目 `.venv`，可能重复安装另一套 Torch/CUDA 依赖。

CLI、官方 WebUI 启动器、批量转录与音频分离共用 `config/muscriptor_batch_profiles.toml` 的模型显存曲线和自适应 CUDA runtime；加载权重前，显存不足的 `auto` 会回退 CPU，显式 CUDA 则直接阻止任务。CUDA 分配器预算会扣除分段预留和当前设备上 PyTorch 之外的显存占用。打开任一 GUI 页面时，还会把用户所选 GPU 的总显存代入 BS1/BS2 与完整 batch 验证后的边际显存公式来选择偶数 batch，参考 GPU 与 0.2.1 版本号只标明实测来源，并非在 0.4.0 上重测的数据。在不超过 16 GiB 时使用 1 GiB 预留，small、medium、large 的最低总显存约为 1.90、3.25、10.28 GiB。用户手动修改前，切换模型或设备会重新计算。

## 输出与恢复

批处理默认输出到输入位置下的 `muscriptor_output`，按输入相对路径分目录。每首歌只转录一次；以下输出可在 GUI 多选，也可在 `batch` 子命令中重复传入 `--format`。单曲需要多种格式时同样使用 `batch song.wav`；`transcribe` 一次只选一种。

| 格式 | 每首歌的文件 | 用途 |
| --- | --- | --- |
| `midi`（默认） | `<歌名>.mid` | 保留演奏的节奏变化；检测到速度不稳时，MIDI 的速度图跟随演奏。 |
| `quantized-midi` | `<歌名>_quantized.mid` | 将音符吸附到节拍网格，适合 DAW 编辑。`--grid 4` 指每拍四格。 |
| `sheets` | `<歌名>_sheets.zip` | 含 `score.mid`、`score.musicxml`、总谱和分谱 PDF。复用量化 MIDI，不再转录。 |
| `json` / `jsonl` | `events.json` / `events.jsonl` | 原始音符事件，时间单位为秒；不生成 MIDI、谱面或试听时不运行节拍检测。 |

GUI 的「MIDI 节拍」可选择自动、跟随演奏或固定速度。在离线环境或短片段中，选择「手动 BPM」并填写每拍格数即可量化，不需要节拍模型。试听优先使用所选的普通 MIDI；只选量化 MIDI 或谱面时，播放与导出一致的量化版本。

节拍检测使用独立的 Beat This! `final0` 权重，首次使用按需下载。默认 `--detect-tempo best-effort`：无法估拍时，普通 MIDI 使用 120 BPM 占位值；没有可用网格时不会强行量化，警告写入 `metadata.json`。设为 `true` 可要求检测成功，设为 `false` 可跳过检测；离线量化可手动指定 `--bpm` 和 `--grid`。`--tempo-map` 强制跟随已检测到的节拍，`--fixed-tempo` 使用单一速度。分轨 MIDI 不逐轨估拍，以便各轨叠回原混音。

仅 `sheets` 需要另装 **MuseScore 4 或更高版本**。若不在 PATH 中，将 `MUSCRIPTOR_MUSESCORE` 指向可执行文件。批处理在加载转录模型前检查它。每项的 `metadata.json` 记录输出、警告和实际音色；根目录的 `manifest.json` 汇总状态。修改格式或节拍参数会使旧结果重新生成。

切换格式时，批处理只清理上次 `metadata.json` 为该输入登记的旧产物。如果新目标与未登记的文件同名，本项报错并保留原件；可移动该文件，或用 `--output-dir` 选择其他目录。

## 试听

`preview-mode midi` 导出合成 MIDI 音频；`comparison` 输出左声道原音、右声道合成 MIDI。试听需要 PATH 中的 FluidSynth，并使用官方 `MuseScore_General.sf2` SoundFont。FluidSynth 不可用时，MIDI、JSON 和 JSONL 仍可正常生成。

事件只包含 onset、offset、pitch 和 instrument。模型不恢复原始力度；密集混音、少见音色、重处理音频和部分合唱材料可能降低准确度。
