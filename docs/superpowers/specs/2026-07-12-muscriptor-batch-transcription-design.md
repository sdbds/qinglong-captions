# MuScriptor 多乐器 MIDI 批处理设计

## 结论

本设计接入 [muscriptor/muscriptor](https://github.com/muscriptor/muscriptor) 的完整模型推理能力，并在本项目 GUI 的 `Tools` 页面新增独立的“音乐转录”批处理工具。

本轮只做两件事：

1. 完整后端能力，包括全部已发布模型、设备选择、乐器条件、解码参数、MIDI/JSON/JSONL、试听音频渲染、单文件 CLI 和目录批处理 CLI。
2. GUI 批处理页面，包括输入发现、模型与设备配置、乐器条件、解码参数、输出选择、批处理策略和现有 Job 执行面板接入。

上游 WebUI 的实时钢琴卷帘、播放、拖动定位、混音、立体声、乐器静音/独奏和单文件 Demo 页面不在本设计内。它们必须在后续独立 spec 中处理，不能借本次批处理接入顺手塞入。

## Linus 三问

### 1. 这是实际问题吗？

是。

本项目现有 `vocal-midi` 使用 GAME，面向人声或单旋律音高与分段；MuScriptor 面向真实混音中的多乐器复调转录。两者的输入虽然都是音频，输出虽然都可包含 MIDI，但任务语义、模型结构和参数完全不同。

### 2. 有更简单的办法吗？

有。直接调用 MuScriptor 的 Python API，一次转录事件流同时生成所有选中输出。

不能为 MIDI、JSON、JSONL 分别调用一次上游 CLI。那会重复加载模型或重复推理，GPU 成本直接乘以输出格式数。

### 3. 会破坏什么？

如果边界不清楚，会破坏以下行为：

- 把 MuScriptor 塞进 Provider V2 会让“字幕文本”和“符号音乐事件”共用错误的输出契约。
- 把它塞进 `audio_separator` 会混淆音源分离、人声 MIDI 和多乐器转录。
- 在 GUI 主进程导入模型会破坏当前任务 tab 的依赖隔离，并可能让显存错误拖垮整个 GUI。
- 重用 `batch_size` 但不说明含义，会让用户误以为它是文件并发数；MuScriptor 的 `batch_size` 实际是每次前向处理的 5 秒音频块数量。
- 静默回退 CPU、静默跳过输出或静默忽略无效参数会产生不可验证结果。

因此采用独立工具、独立依赖 extra、独立进程和显式输出契约。

## 已确认范围

### 目标

1. 支持 `small`、`medium`、`large` 三个官方模型。
2. 支持本地 `.safetensors`、`hf://` 和 `http(s)://` 自定义权重来源。
3. 支持 `auto`、`cpu`、`cuda`、`cuda:N` 设备选择。
4. 支持 greedy、temperature sampling 和 beam search。
5. 支持 temperature、CFG、5 秒块 batch size、strict EOS。
6. 支持全部官方乐器组条件输入，并复用上游的大小写、缩写和歧义校验行为。
7. 支持 MIDI、JSON、JSONL 输出。
8. 支持只合成转录 MIDI 的音频，以及“左声道原音、右声道合成 MIDI”的对照音频。
9. 支持 WAV 和运行时可用时的 MP3 音频渲染。
10. 支持单文件 CLI 的 stdout/file 行为，以及文件或目录批处理。
11. 批处理中模型只加载一次，每个输入文件只推理一次。
12. GUI 提供独立批处理工具页，并接入现有任务 tab、Start/Stop、进度和日志系统。
13. 保持现有 GAME 人声 MIDI、音频分轨、MuSViT 乐谱扫描和 ALM provider 行为不变。

### 非目标

- 不实现或复制上游 WebUI。
- 不实现钢琴卷帘、实时音符绘制、浏览器 SoundFont 播放、原音/MIDI 混音控制、乐器静音或独奏。
- 不新增公开的 `serve` HTTP 服务。HTTP/SSE 是上游 Web 产品入口，不是模型推理能力；未来单文件 Demo 如需服务协议，应由其独立 spec 决定。
- 不接入 Provider V2，不生成字幕，不写 Lance caption 字段。
- 不替换 GAME 人声 MIDI。
- 不做音源分离、节拍校正、量化、MusicXML 或乐谱排版。
- 不做 ONNX、量化、`torch.compile` 或第三方推理引擎改造。
- 不分发 MuScriptor 权重或 SoundFont 文件。
- 不承诺手工乐谱级准确度，也不伪造模型不提供的力度信息。

## 上游基线与兼容策略

### 版本基线

首版依赖固定为：

```text
muscriptor==0.2.1
```

审查基线：

- release tag: `v0.2.1`
- commit: `964e2350d5677eb3c3ca4d29e0e03286671e910a`
- 当前主分支审查 commit: `7eacca9d481acc35738f42628cb8327fb6486538`

选择已发布版本而不是 `main` 上的 `0.2.2a1`，避免 GUI 功能绑定未发布 API。上游后续升级必须单独验证 CLI、事件结构、模型参数和输出字节兼容性。

### 复用边界

直接复用以下上游能力：

- `TranscriptionModel.load_model(...)`
- `TranscriptionModel.transcribe(...)`
- `TranscriptionModel.events_to_midi_bytes(...)`
- `resolve_instrument_names(...)`
- `synthesize(...)`
- `auralize(...)`

不复制模型、tokenizer、Transformer、音符清理或 MIDI 序列化实现。本项目只负责：

- 规范化和校验 CLI/GUI 参数；
- 批量发现输入；
- 将一次事件流扇出为多个输出；
- 原子写文件；
- 跳过、覆盖、错误隔离和 manifest；
- GUI 到 CLI 的参数映射；
- 把上游错误转换成可操作的用户消息。

## 模型能力契约

### 官方模型

| CLI 值 | 规模 | 层数 | 维度 | 用途 |
| --- | ---: | ---: | ---: | --- |
| `small` | 约 103M | 14 | 768 | CPU 或资源受限环境 |
| `medium` | 约 307M | 24 | 1024 | 默认速度/质量平衡 |
| `large` | 约 1.4B | 48 | 1536 | 质量优先，建议 CUDA |

默认值固定为 `medium`，与上游一致。

模型字段同时接受：

- `small`、`medium`、`large`；
- 本地 `.safetensors` 文件；
- `hf://ORG/REPO/path/to/model.safetensors`；
- `https://.../model.safetensors` 或 `http://...`。

GUI 用模型下拉框展示三种官方模型，并提供 `Custom`。只有选择 `Custom` 时才显示权重来源输入框。

### 设备

支持：

```text
auto
cpu
cuda
cuda:0
cuda:1
...
```

规则：

- `auto` 在 CUDA 可用时使用 CUDA，否则使用 CPU。
- 用户显式指定 `cuda` 或 `cuda:N` 时，CUDA 不可用或索引越界必须直接失败，不能静默回退 CPU。
- 日志和 metadata 必须同时记录请求值与解析后的实际设备。
- GUI 从运行时检测结果构建 `auto`、`cpu` 和可用 CUDA 设备列表，同时允许手工输入 `cuda:N`。
- 不暴露 `dtype`。官方权重为 FP32，上游在 CUDA 前向中自行使用 autocast；增加 dtype 开关会承诺上游没有公开保证的行为。

### 输入预处理

保留上游语义：

- 解码为单声道 float waveform；
- 重采样为 16 kHz；
- 按 5 秒切块；
- 最后一块补零；
- 按时间顺序产生事件；
- 文件级处理保持串行，`batch_size` 只控制 5 秒块的前向 batch。

目录发现复用本项目音频扩展名集合，但必须排除 `.mid` 和 `.midi`。发现到但运行时无法解码的文件按单项失败处理，不能让整个目录无结果退出。

### 乐器条件

MuScriptor 使用固定 `MT3_FULL_PLUS` 乐器组。`v0.2.1` 的公开名称映射当前暴露 35 个可选名称；后端必须从上游映射动态读取，不能把数量或名称在 GUI 再维护一份会漂移的常量。

支持：

- 不指定：让模型自行判断；
- 指定一个或多个官方乐器组；
- CLI 大小写不敏感；
- CLI 支持无歧义缩写；
- 未知值给出最多三个接近建议；
- 歧义缩写列出候选并失败。

乐器条件是“预期乐器提示”，不能在 UI 文案中伪装成严格过滤器。

### 解码模式

GUI 使用单选分段控件，三种模式互斥：

| 模式 | 参数 |
| --- | --- |
| Greedy | `sampling=false`, `beam_size=1` |
| Sampling | `sampling=true`, `beam_size=1`, 使用 `temperature` |
| Beam | `sampling=false`, `beam_size>=2` |

后端校验：

- 内部只保存一个 `DecodingMode = greedy|sampling|beam`，再集中转换为上游的 `use_sampling` 和 `beam_size`，不让互相矛盾的布尔标志流入 runtime。
- 单文件兼容 CLI 的 `--sampling` / `--beam-size` 先归一化为 `DecodingMode`。
- `beam_size` 必须为正整数。
- `sampling=true` 与 `beam_size>=2` 同时出现时直接报参数错误。上游 beam 路径会忽略 sampling，静默接受会误导用户。
- 非 Sampling 模式若传入非默认 temperature，直接报参数错误，不静默忽略。
- temperature 必须为有限正数。
- CFG 必须为有限数，默认 `1.0`。GUI 提供常用范围控制，但文本输入不应偷偷裁剪合法浮点值。
- `strict_eos=false` 时，无 EOS 的块保留已生成结果并记录 warning。
- `strict_eos=true` 时，该输入文件失败；批处理默认继续下一个文件。

### 上游表示限制

metadata 和文档必须明确：

- 输出包含 onset、offset、pitch 和 instrument；
- 不包含原始力度，MIDI 使用上游固定力度；
- 同一乐器、同一音高的同时重叠音符无法表示；
- 鼓是 onset-only，结束时间由上游最小时长规则生成；
- 密集混音、少见音色、重处理音频和部分合唱材料可能明显降低精度。

这些是模型边界，不能由后处理猜测补齐。

## 后端结构

新增独立 package，避免创建 `module/muscriptor.py` 后遮蔽已安装的 `muscriptor` 包：

```text
module/muscriptor_tool/
  __init__.py
  cli.py
  options.py
  runtime.py
  events.py
  outputs.py
  batch.py
  manifest.py
  auralization.py
```

职责：

- `cli.py`: Typer 命令、stdout/stderr 规则和退出码。
- `options.py`: `DecodingMode`、`TranscriptionOptions`、`BatchOptions`、参数归一化和互斥校验。
- `runtime.py`: 延迟导入 MuScriptor、设备解析、模型加载和文件级转录。
- `events.py`: 上游事件到稳定 JSON schema 的转换。
- `outputs.py`: 一次事件流扇出、MIDI/JSON/JSONL 写入和原子替换。
- `batch.py`: 输入发现、模型单次加载、逐文件执行、跳过与失败隔离。
- `manifest.py`: 运行签名、单项 metadata、总 manifest。
- `auralization.py`: FluidSynth、SoundFont 和音频编码预检；调用上游合成函数。

GUI 不导入这些模块中的 torch 或 MuScriptor。所有重依赖只在任务子进程中延迟导入。

## CLI 契约

统一入口：

```text
python -m module.muscriptor_tool.cli COMMAND ...
```

### `transcribe`: 单文件兼容入口

```text
python -m module.muscriptor_tool.cli transcribe INPUT_AUDIO [OPTIONS]
```

必须支持：

```text
-o, --output PATH|-
-f, --format midi|json|jsonl
--notes
--sampling
-t, --temperature FLOAT
--cfg-coef FLOAT
-m, --model small|medium|large|PATH|URL
-d, --device auto|cpu|cuda|cuda:N
-b, --batch-size INT
--strict-eos
--beam-size INT
--auralize PATH
--auralize-mode synth|stereo
--soundfont PATH
--instruments NAME[,NAME...]
```

默认值与上游一致：

```text
format=midi
model=medium
device=auto
batch_size=auto, CPU=1, CUDA=4
sampling=false
temperature=1.0
cfg_coef=1.0
strict_eos=false
beam_size=1
auralize_mode=stereo
```

输出规则：

- 未指定 `--output` 时，在输入文件旁生成对应扩展名。
- `--output -` 时，stdout 只包含目标内容；日志、计时、notes 和错误全部写 stderr。
- MIDI stdout 写二进制 bytes。
- JSON stdout 写一个完整 JSON array 并以换行结束。
- JSONL stdout 每行一个完整 JSON object，并逐行 flush。
- `--auralize` 只允许 MIDI 格式且 `--output` 不能为 `-`。
- `--auralize-mode synth` 只渲染转录音符。
- `--auralize-mode stereo` 左声道原音、右声道合成 MIDI。
- `--notes` 输出可读事件到 stderr，不能污染机器可读 stdout。

### `batch`: 文件和目录批处理

```text
python -m module.muscriptor_tool.cli batch INPUT_PATH [OPTIONS]
```

模型与解码参数与 `transcribe` 共用同一个 options 类型。批处理另加：

```text
--output-dir PATH
--format midi|json|jsonl              # 可重复
--audio-render synth|stereo           # 可重复
--render-format wav|mp3
--decode-mode greedy|sampling|beam
--recursive / --no-recursive
--skip-completed / --no-skip-completed
--overwrite
--fail-fast
```

规则：

- 默认输出目录为 `workspace/muscriptor_output`。
- 没有显式传入 `--format` 时默认 MIDI；一旦显式传入，输出集合只由出现的 `--format` 值决定，不把默认 MIDI 偷偷并入。
- 至少选择一种符号输出或一种音频渲染。
- 多个 `--format` 和 `--audio-render` 仍只运行一次模型推理。
- 输入发现和完成签名检查先于模型加载；存在待处理文件时才加载模型，整批最多加载一次。
- 所有项目都可合法跳过时，不加载模型、不访问权重，直接写 manifest 并退出 `0`。
- 待处理输入文件逐个执行，不做文件级并发。
- 默认递归、默认跳过已完成、默认遇错继续。
- `--fail-fast` 在首个文件失败或请求输出失败后停止。
- `--overwrite` 优先于 `--skip-completed`。
- GUI 不提供文件级并发，因为同一 GPU 上并发模型前向只会放大显存峰值和失败面。
- batch 在 output directory 上持有跨进程 `.muscriptor.lock`；另一个 CLI 或任务 tab 使用同一输出目录时，在模型加载前直接失败。

### `list-instruments`

```text
python -m module.muscriptor_tool.cli list-instruments
```

每行输出一个上游 canonical 乐器名，保持上游顺序。GUI 的多选列表和 CLI 校验必须由同一读取函数驱动。

### 退出码

| 退出码 | 含义 |
| ---: | --- |
| `0` | 所有请求成功，或项目被合法跳过 |
| `1` | 模型、输入、转录或任一请求输出失败 |
| `2` | CLI 参数错误，由 Typer/Click 产生 |
| `130` | 用户中断 |

批处理中存在 `partial` 项时退出 `1`，即使 MIDI 已成功但请求的试听音频失败。

## 稳定事件输出

### NoteStart

```json
{
  "type": "start",
  "pitch": 60,
  "start_time": 1.25,
  "index": 42,
  "instrument": "acoustic_piano"
}
```

### NoteEnd

```json
{
  "type": "end",
  "end_time": 1.75,
  "start_event_index": 42
}
```

兼容规则：

- JSON 是上述对象的单一 array。
- JSONL 每行一个对象。
- `ProgressEvent` 不写入 JSON/JSONL，保持上游 CLI 兼容；它只用于终端进度和 manifest 统计。
- 字段名、类型和时间单位不得由 GUI 单独改写。
- 空转录是合法结果：MIDI 为空轨、JSON 为 `[]`、JSONL 为空文件，同时 metadata 记录 `EMPTY_TRANSCRIPTION` warning。

## 单次推理、多输出数据流

每个输入文件只执行以下流程一次：

1. 启动 JSONL `.part` writer，仅当请求 JSONL 时启用。
2. 遍历 `model.transcribe(...)`。
3. Progress 事件更新终端进度与统计，不进入公开事件文件。
4. NoteStart/NoteEnd 立即写入 JSONL `.part`；只有请求 JSON、MIDI 或音频渲染时才追加到内存事件列表。
5. 推理结束后，用同一事件列表生成 JSON 和 MIDI。
6. 如果请求音频渲染而未请求保存 MIDI，使用临时 MIDI 文件完成 FluidSynth 渲染，随后删除临时文件。
7. 每个成功输出先写 `.part`，完成后原子替换目标文件。
8. 最后写 metadata。metadata 存在且签名匹配，才代表该项完整提交。

不能调用 `transcribe_to_midi()` 后再调用 `transcribe()` 生成 JSON，因为这会重复推理。

## 批处理输入与输出

### 输入发现

- 文件输入：只处理该文件。
- 目录输入：按标准化相对路径排序，保证输出顺序稳定。
- `--recursive` 控制子目录。
- 大小写不敏感匹配扩展名。
- 排除 `.mid` 和 `.midi`，防止把符号输出当音频重新输入。
- 解析并排除 output directory 的完整子树；即使输出目录位于输入目录内，也不能在后续运行中重新转录 `transcription.wav` 或 `comparison.wav`。
- output directory 与输入目录完全相同时直接拒绝，避免既无法安全发现输入又污染 source tree。
- 不跟随目录符号链接，避免循环和越界遍历。
- 没有发现输入时退出 `1` 并写明路径。

### 输出布局

使用包含原始扩展名的每项目录，消除同名不同格式碰撞：

```text
workspace/muscriptor_output/
  album/
    song.wav/
      transcription.mid
      events.json
      events.jsonl
      transcription.wav
      comparison.wav
      metadata.json
    song.flac/
      transcription.mid
      metadata.json
  manifest.json
```

名称固定：

| 请求 | 文件名 |
| --- | --- |
| MIDI | `transcription.mid` |
| JSON | `events.json` |
| JSONL | `events.jsonl` |
| synth WAV/MP3 | `transcription.wav` / `transcription.mp3` |
| stereo WAV/MP3 | `comparison.wav` / `comparison.mp3` |
| 单项元数据 | `metadata.json` |

### 跳过与覆盖

单项运行签名由以下内容生成 SHA-256：

- source relative path；
- source size；
- source `mtime_ns`；
- MuScriptor package version；
- model source；
- resolved device；
- instruments；
- normalized decode mode、temperature、CFG、batch size、strict EOS、beam size；
- 请求输出集合；
- 音频渲染格式和 SoundFont 标识。

`skip_completed=true` 只有在以下条件全部满足时才跳过：

- `metadata.json` 可解析；
- 签名相同；
- metadata 状态为 `ok`；
- 所有本次请求输出都存在。

仅凭 `transcription.mid` 存在不能跳过。参数变化后必须重新处理。

### metadata

每项至少记录：

```json
{
  "schema_version": 1,
  "source_path": "album/song.wav",
  "source_size": 12345678,
  "source_mtime_ns": 1783800000000000000,
  "status": "ok",
  "run_signature": "sha256:...",
  "muscriptor_version": "0.2.1",
  "model_source": "medium",
  "requested_device": "auto",
  "resolved_device": "cuda:0",
  "instruments": [],
  "options": {
    "decode_mode": "greedy",
    "temperature": 1.0,
    "cfg_coef": 1.0,
    "batch_size": 4,
    "strict_eos": false,
    "beam_size": 1
  },
  "note_count": 1024,
  "event_count": 2048,
  "chunk_count": 15,
  "outputs": {
    "midi": "transcription.mid",
    "json": "events.json"
  },
  "warnings": [],
  "elapsed_seconds": 12.34,
  "error": null
}
```

状态值：

- `ok`: 所有请求输出成功。
- `skipped`: manifest 中使用，单项原 metadata 不重写。
- `partial`: 转录成功，但至少一个请求输出失败。
- `failed`: 没有得到可提交的转录结果。

### manifest

`manifest.json` 必须在正常结束、部分失败和 fail-fast 结束时写出，至少包含：

- schema version；
- 开始和结束时间；
- input/output 路径；
- 完整规范化参数；
- package/model/device 信息；
- discovered、processed、skipped、partial、failed 计数；
- 每项 source、status、metadata path 和 error 摘要；
- 总耗时。

manifest 使用临时文件加原子替换。被强制终止时遗留的 `.part` 文件不算完成，并在下次处理同一项目时先清理。

批处理从发现输入到写完 manifest 的整个期间持有 output directory 文件锁。锁由操作系统在进程退出时释放，不能把“锁文件还存在”等同于任务仍存活。

## 音频渲染

音频渲染不是 MIDI 成功的前置条件，但如果用户明确请求，它就是验收输出。

### 模式

- `synth`: 只渲染转录 MIDI，单声道。
- `stereo`: 左声道原音，右声道转录合成音，合成侧按上游规则做 RMS 匹配。

### 依赖与预检

请求任一音频渲染时，在模型加载前检查：

1. `fluidsynth` 可执行文件存在并可启动。
2. 自定义 `.sf2` 存在，或默认 MuseScore General SF2 可以从上游地址访问缓存。
3. `soundfile` 支持目标格式。

默认 SoundFont 约 215 MB，只按需下载并缓存，不打包进项目。

若预检失败：

- 单文件 CLI 在推理前失败。
- GUI 在启动前显示明确错误。
- 批处理命令行在推理前失败，避免处理完整目录后才发现无法生成请求输出。

不增加隐式 ffmpeg MP3 回退。当前依赖不支持 MP3 写出时，要求用户选择 WAV，避免两套编码路径。

## 模型访问与许可

MuScriptor 代码是 MIT。官方权重是 gated `CC BY-NC 4.0`，并附带输入音乐和生成结果的权利条件。

要求：

- 不把权重提交、打包或镜像到本项目。
- 官方模型首次使用前，用户必须在对应 Hugging Face 页面接受条件并完成本机登录。
- 认证只读取 Hugging Face 本机凭据或 `HF_TOKEN`，GUI 不保存 token。
- 遇到 401/403/GatedRepoError 时，错误必须包含对应模型页和 `hf auth login` 操作，不显示长 traceback 代替说明。
- GUI 模型区显示 `CC BY-NC 4.0` 状态链接，但不自建虚假的“接受许可”复选框；真正授权由 Hugging Face gate 负责。
- README 增加上游代码、模型许可和输入权利限制链接。

## GUI 批处理页

### 页面位置

在 `gui/wizard/step6_tools.py` 的 `TOOL_TABS` 新增独立标签：

```python
("music_transcription", "music_transcription", "piano")
```

顺序放在 `audio_separator` 之后、`sheet_music` 之前：

```text
Audio Separator -> Music Transcription -> Sheet Music
```

它不能成为 Audio Separator 内部的高级开关，因为用户可能直接转录原始混音，不需要先分轨。

### 页面结构

保持现有 Tools 页视觉系统和共享执行面板，不复制 MuScriptor 黑色品牌页。表单分为六个紧凑区域，不嵌套卡片：

1. 输入与输出路径。
2. 模型与运行设备。
3. 乐器条件。
4. 解码设置。
5. 输出选择与音频渲染。
6. 批处理行为。

页面底部继续使用现有 `ExecutionPanel` 的任务 tab、Start、Stop、进度和日志，不增加第二套 Job 系统。

### 输入与输出

- 输入模式使用 File/Directory 分段控件。
- 路径选择器随模式选择文件或目录，并仍允许手工输入路径。
- 文件过滤器显示项目支持的音频扩展名，但排除 MIDI。
- 输出目录默认 `workspace/muscriptor_output`。
- Start 前检查输入存在、输出目录可创建。

### 模型与设备

- 模型下拉：Small、Medium、Large、Custom。
- Custom 模式显示权重来源输入。
- 设备下拉：Auto、CPU、检测到的 CUDA 设备。
- 提供刷新设备图标按钮和 tooltip，不增加说明型大段文案。
- batch size 使用 `Auto` 或正整数输入，标签明确为“5 秒块 batch size”。
- Large + CPU 只提示性能 warning，不阻止运行。

### 乐器条件

- 模式：Auto Detect / Specify。
- Specify 时显示可搜索多选控件。
- 选项使用友好显示名，提交值保持 canonical snake_case。
- 支持清空全部。
- GUI 不接受缩写；缩写只属于 CLI 文本输入。GUI 必须提交精确 canonical 值。

### 解码设置

- Greedy / Sampling / Beam 分段控件。
- Sampling 时显示 temperature。
- Beam 时显示 beam size，最小值 2。
- CFG 始终可编辑，默认 1.0。
- Strict EOS 使用 toggle。
- Print note events 使用高级 toggle，默认关闭，避免海量日志。

隐藏控件不能残留旧值影响提交。模式改变时参数构造必须由当前模式统一生成，而不是在按钮回调中堆叠互相冲突的 if 分支。

### 输出

- MIDI、JSON、JSONL 使用多选 checkbox，默认 MIDI。
- Audio Render 使用 Off、Synth、Stereo、Both 选项集。
- 请求音频渲染时显示 WAV/MP3 选择和 SoundFont 路径。
- SoundFont 为空表示使用上游默认缓存。
- 至少选择一个符号输出或音频渲染，否则 Start 不可用。
- GUI 允许一次选择多个输出，后端保证一次推理。

### 批处理行为

- Recursive toggle，目录模式可见。
- Skip completed toggle，默认开。
- Overwrite toggle，开启后 GUI 禁用 Skip completed；后端仍以 overwrite 优先。
- Fail fast toggle，默认关。
- 不增加文件并发数控件。

### 执行与日志

GUI 调用：

```text
module.muscriptor_tool.cli batch ...
```

ProcessRunner 注册使用 module 模式：

```text
"module.muscriptor_tool.cli": ("-m:module.muscriptor_tool.cli", "muscriptor-local")
```

要求：

- 运行在当前任务 tab 的 venv。
- Stop 终止完整进程树。
- 日志显示模型加载、resolved device、输入计数、当前文件、块进度、输出和汇总。
- 不把每个 NoteStart/NoteEnd 默认打印到日志。
- GUI 渲染时不得导入 torch 或 muscriptor。
- 切换到该标签页不能触发模型下载或模型加载。

## 配置

在 `config/model.toml` 新增：

```toml
[muscriptor]
model = "medium"
custom_model_source = ""
device = "auto"
batch_size = 0
instruments = []
decode_mode = "greedy"
temperature = 1.0
cfg_coef = 1.0
strict_eos = false
beam_size = 1
output_dir = "workspace/muscriptor_output"
output_formats = ["midi"]
audio_render_modes = []
render_format = "wav"
soundfont_path = ""
recursive = true
skip_completed = true
overwrite = false
fail_fast = false
print_notes = false
```

`batch_size = 0` 表示 auto。GUI 和 CLI 的默认值必须来自共享常量或 options 类型，不能在 Python、TOML 和测试中各写一套不同默认值。

## 依赖与运行时

在 `pyproject.toml` 新增独立 extra：

```toml
muscriptor-local = [
    "qinglong-captions[torch-base]",
    "muscriptor==0.2.1",
    "filelock>=3.16",
]
```

MuScriptor 已声明 `numpy`、`einops`、`mido`、`safetensors`、`huggingface_hub`、`typer`、`soundfile` 等运行依赖，不重复列出，除非 uv 锁定验证证明需要本项目覆盖版本。

FluidSynth 是可选系统依赖：

- MIDI/JSON/JSONL 不依赖 FluidSynth。
- 只有请求音频渲染时才要求 FluidSynth。
- 安装脚本和 README 必须给出 Windows/Linux 的检测与安装说明，但不能因为 FluidSynth 缺失而阻止安装基础 `muscriptor-local` extra。

必须验证：

```text
uv lock --check
uv sync --extra muscriptor-local
python -m module.muscriptor_tool.cli --help
```

## PowerShell 与文档入口

新增批处理包装脚本：

```text
2.7.music_transcription.ps1
```

脚本只负责：

- 安装/补齐 `muscriptor-local` extra；
- 把配置映射为 `python -m module.muscriptor_tool.cli batch` 参数；
- 保持 stdout/stderr 和退出码；
- 不复制 Python 的参数校验和批处理逻辑。

同步更新：

- `README.md`
- `README.en.md`
- `gui/README.md`
- `gui/PARAMETERS.md`
- `gui/utils/i18n.py` 的英文、简体中文、日文、韩文词条
- 上游与许可引用

## 错误处理

### 启动前失败

以下错误在模型加载或推理前返回：

- 输入不存在；
- 没有支持的输入；
- 输出目录不可创建；
- 参数组合无效；
- 自定义本地权重不存在；
- 显式 CUDA 不可用或索引无效；
- 请求音频渲染但 FluidSynth、SoundFont 或编码格式不可用；
- 官方 gated 模型未授权。

### 文件级失败

以下错误只影响当前文件，除非开启 fail-fast：

- 音频无法解码；
- strict EOS 失败；
- 模型运行时异常；
- 输出写入失败；
- 音频渲染失败。

成功的符号输出不能因后续音频渲染失败而删除；该项记录 `partial`，批处理最终退出 `1`。

### 模型级失败

模型加载失败后不能为每个输入重复同一失败。写 run-level manifest 错误并立即退出。

错误日志必须包含异常类型、当前 source、当前阶段和可操作信息。不能只显示“任务失败”。

## 兼容性

以下现有接口必须保持不变：

- `module.audio_separator`
- `module.vocal_midi`
- `vocal-midi` extra
- `module.sheet_music_musvit`
- `musvit-onnx` extra
- Provider V2 的 ALM/ASR 路由
- Tools 页其他标签与共享执行面板
- 现有 task tab venv 与 Job 并发规则

新增名称使用 `music_transcription` 和 `muscriptor-local`，避免把三种不同能力都叫 `midi` 或 `music`。

## 测试设计

### CLI 合同测试

- `transcribe --help` 暴露全部上游推理参数和输出参数。
- `batch --help` 暴露全部批处理参数。
- `list-instruments` 与上游映射完全一致。
- MIDI/JSON/JSONL 的 stdout 不含日志。
- `--notes` 只写 stderr。
- 默认输出扩展名正确。
- sampling + beam 被拒绝。
- 非 sampling 的非默认 temperature 被拒绝。
- 无效 CUDA 和 batch size 被拒绝。
- auralize + 非 MIDI 或 stdout 被拒绝。

### 假模型单元测试

- 两个输入只加载一次模型。
- 每个输入只调用一次 `transcribe()`，即使同时请求 MIDI、JSON、JSONL 和两种音频渲染。
- ProgressEvent 不进入公开事件文件。
- JSON/JSONL schema 与上游一致。
- 空事件流产生合法空输出并记录 warning。
- missing EOS warning 写入 metadata。
- strict EOS 使当前项目失败。
- 某一文件失败后默认继续下一文件。
- fail-fast 在首错停止。

### 输出与恢复测试

- 保留输入相对目录和原始扩展名目录。
- 同名 `.wav`/`.flac` 不碰撞。
- 输出目录位于输入目录内时，发现器会剪除整个输出子树。
- 输出目录与输入目录相同会在模型加载前失败。
- `.part` 只在完成后替换。
- metadata 最后提交。
- 签名一致且输出完整时跳过。
- 参数或 mtime 变化后不跳过。
- overwrite 始终重跑。
- 音频渲染失败保留 MIDI/JSON 并标记 partial。
- manifest 在全成功、部分失败、fail-fast 下都可解析。
- 两个进程使用同一输出目录时，后启动者在模型加载前失败；不同输出目录可并行。

### GUI 测试

- Tools 页存在独立 `music_transcription` tab。
- action 指向 `module.muscriptor_tool.cli` 的 `batch` 子命令。
- ProcessRunner registry 使用 `muscriptor-local`。
- File/Directory 模式映射正确。
- 三种官方模型和 custom source 映射正确。
- auto/cpu/cuda:N 映射正确。
- 乐器多选提交 canonical 名称。
- Greedy/Sampling/Beam 只生成合法参数组合。
- 多格式、多渲染、SoundFont、recursive、skip、overwrite、fail-fast 全部映射。
- 未选择任何输出时 Start 禁用。
- GUI import/render 测试不需要安装 torch 或 muscriptor。
- 英文、中文、日文、韩文新增 key 完整。

### 依赖测试

- `muscriptor-local` 存在并固定 `muscriptor==0.2.1`。
- extra 依赖 `torch-base`。
- extra 显式声明批处理直接使用的 `filelock`，不依赖偶然的传递依赖。
- registry、PowerShell 和 GUI 使用同一 extra 名。
- uv lock 无冲突。
- 未请求音频渲染时，不检查 FluidSynth。

### 真实 smoke test

真实模型测试不进入默认 CI，因为权重 gated 且体积较大。提供显式环境门禁：

```text
MUSCRIPTOR_SMOKE=1
```

至少手工验证：

```text
python -m module.muscriptor_tool.cli transcribe sample.wav --model small --device cpu --format midi
python -m module.muscriptor_tool.cli transcribe sample.wav --model small --device cuda:0 --format jsonl
python -m module.muscriptor_tool.cli batch samples --model medium --device cuda:0 --format midi --format json
```

请求音频渲染时另做 FluidSynth smoke test。

## 验收标准

- 三个官方模型和上游支持的全部自定义权重来源形式可从 CLI 使用。
- auto、CPU、CUDA 和指定 CUDA index 行为明确且可验证。
- 全部上游推理参数可从单文件 CLI 使用。
- 单文件 CLI 完整支持 MIDI、JSON、JSONL、stdout 和 auralization。
- 批处理支持文件和目录，模型每批只加载一次，每文件只推理一次。
- 一次推理可同时写 MIDI、JSON、JSONL 和请求的试听音频。
- 输出目录无同名碰撞，metadata 和 manifest 可用于恢复与审计。
- GUI 有独立音乐转录 tab，并暴露全部批处理相关模型、设备、条件、解码、输出和恢复选项。
- GUI 使用现有任务 tab 和执行面板，不实现第二套 Job 系统。
- GUI 渲染不加载模型、不下载权重、不要求 torch 已安装。
- 没有授权、CUDA 不可用、参数冲突、解码失败、FluidSynth 缺失和部分输出失败都有明确错误。
- 现有音频分轨、GAME 人声 MIDI、MuSViT 和 Provider V2 测试保持通过。
- 本轮没有钢琴卷帘、实时试听或上游 WebUI 代码。

## 主要风险

### 权重许可和输入权利

官方权重不可用于违反 `CC BY-NC 4.0` 和附加条件的场景。项目只能提供接入能力，不能替用户取得音乐输入权利，也不能把 gated 权重重新分发。

### 模型与 SoundFont 体积

模型和默认 SF2 会占用明显磁盘空间。下载必须沿用缓存，不能每个任务 tab 重复保存一套文件。

### GPU 显存

Large 模型、beam search、较大块 batch 和 CFG 都可能提高显存峰值。首版不添加未经验证的自动 VRAM 推荐值；CUDA OOM 必须原样归因并建议降低模型、beam 或 batch size。

### JSON 体积

长音频的 JSON 会将全部事件保存在内存。JSONL-only 路径保持流式且不保留完整事件列表；MIDI、JSON 和音频渲染仍需要事件集合。首版接受该成本，不为尚未证明的超长输入增加磁盘事件数据库。

### 音频编码环境差异

FluidSynth 和 libsndfile 的 MP3 写入能力依系统而异。预检必须在推理前完成，且 WAV 永远是受支持的推荐渲染格式。

## 被否决的方案

### 为每种输出调用一次上游 CLI

否决。重复模型加载或重复推理，性能随输出数线性恶化。

### 把 MuScriptor 接进 ALM Provider V2

否决。Provider V2 的主契约是 caption/transcript，MuScriptor 输出是带引用关系的符号事件和 MIDI bytes。

### 把功能放进 Audio Separator

否决。音源分离和多乐器转录是可独立运行的任务，强制父子关系会让直接转录变得困难。

### 复制上游 WebUI 或启动 iframe

否决并延期。它不解决批处理，反而引入第二套前端、服务生命周期和状态管理；单文件交互页面由后续独立 spec 设计。

## 建议实施顺序

1. 建立 options、事件 schema 和 CLI 合同测试。
2. 实现单文件 direct-library adapter 和 stdout/file 输出。
3. 实现一次事件流的多输出扇出。
4. 实现批处理发现、输出布局、metadata、manifest、skip/overwrite。
5. 接入可选 FluidSynth 渲染和预检。
6. 增加 dependency extra、ProcessRunner registry 和 PowerShell 包装脚本。
7. 增加 GUI 独立 tab 和全部参数映射。
8. 补齐 i18n、README 和许可说明。
9. 运行 focused tests、全量 tests 和 gated real smoke test。

## 参考

- MuScriptor repository: https://github.com/muscriptor/muscriptor
- MuScriptor v0.2.1: https://github.com/muscriptor/muscriptor/releases/tag/v0.2.1
- MuScriptor PyPI package: https://pypi.org/project/muscriptor/0.2.1/
- MuScriptor medium model and license: https://huggingface.co/MuScriptor/muscriptor-medium
- MuScriptor small model: https://huggingface.co/MuScriptor/muscriptor-small
- MuScriptor large model: https://huggingface.co/MuScriptor/muscriptor-large
- 当前 Tools 页面: `gui/wizard/step6_tools.py`
- 当前进程注册: `gui/utils/process_runner.py`
- 当前任务管理: `gui/utils/job_manager.py`
- 当前 GAME 人声 MIDI: `module/vocal_midi.py`
- 当前 MuSViT 工具: `module/sheet_music_musvit.py`
