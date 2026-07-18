# 通用音乐时序分析、增强 MIDI 与 MusicXML 导出设计

- 日期：2026-07-19
- 状态：已确认，等待实现计划
- 影响范围：MuScriptor、本地 GAME 人声转 MIDI、音频分离后的 MIDI 工作流、MuScriptor WebUI、音频预览

## 1. 背景与问题定义

当前项目有两条音频转 MIDI 路径：

1. MuScriptor 将乐器事件导出为 MIDI、JSON 和 JSONL。
2. GAME 人声转 MIDI 将人声音高轨迹导出为 MIDI。

两条路径都没有从输入音乐中建立可信的音乐时序：

- MuScriptor 上游导出器固定使用 120 BPM，并把所有音符力度固定为 100。
- GAME 人声 MIDI 显式写入 120 BPM，秒到 tick 的换算也依赖 120 BPM；其 note_on 未设置力度，实际使用 mido 默认值 64。
- JSON 和 JSONL 只保存秒级起止事件，缺少 BPM、拍号、节拍位置、力度来源和分析 provenance。
- 上游 MuScriptor WebUI 的最终 MIDI 与 auralize 路由绕过项目导出层，因此也保留 120 BPM、固定力度和系统 PATH 中的 fluidsynth 依赖。
- 项目没有 MusicXML 导出，无法直接得到面向 MuseScore 编辑的谱面。
- FluidSynth 只做可执行文件预检，Windows 用户仍需手动安装和配置 PATH。

问题的本质不是单独替换一个 MIDI tempo 字段。MIDI 中音符位置以 tick 表示，tempo map 决定 tick 与真实秒数的映射。只修改固定 120 BPM 的 set_tempo 而不重映射音符，会改变播放时长并破坏音频对齐。因此需要建立一个通用的“音乐时序 + 增强音符”中间层，让所有输出从同一份事实生成。

## 2. 目标

### 2.1 必须实现

1. 从原始完整混音中检测节拍、下拍、全局 BPM、可选 tempo map 和拍号。
2. MuScriptor MIDI、JSON、JSONL 和 MusicXML 使用同一份时序分析。
3. GAME 人声 MIDI 复用同一通用时序组件，不再默认固定 120 BPM。
4. 音频分离工作流对每首原始音乐只分析一次，所有人声与乐器 stem 复用结果。
5. 为 MuScriptor 音符估算自然的相对力度，并在 MIDI、JSON、JSONL 和 MusicXML 中保持一致。
6. 新增面向 MuseScore 阅读和编辑的 MusicXML 导出。
7. Windows x64 在需要音频预览时可按需安装项目私有 FluidSynth，不修改全局 PATH。
8. 分析失败时明确回退到 120 BPM、4/4，并记录来源和警告；不得静默假装检测成功。
9. 支持手动 BPM 和拍号覆盖；两者都提供时跳过 Beat This，允许完全离线工作。
10. 所有新输出均原子写入，批处理能够识别旧结果并重新生成。

### 2.2 非目标

1. 不声称恢复演奏设备或乐器真实 MIDI velocity；只提供可听、稳定的相对动态。
2. 本轮不改变 GAME 人声 MIDI 的力度，继续使用固定 64。
3. 不自动推断调号、钢琴左右手、连奏/断奏等演奏法、渐强发夹或出版级制谱。
4. 不重写 MuScriptor 上游编译后的 Web 前端，因此上游 WebUI 不新增 MusicXML 按钮。
5. 不在 Linux 或 macOS 上模拟不可靠的私有 FluidSynth 二进制安装。
6. 不引入 madmom、DBN 或完整的节拍后处理栈。

## 3. 已确认的产品决策

1. MusicXML 优先“可读、可编辑”，允许节奏量化，不追求无量化的毫秒级还原。
2. 节拍模型使用 Beat This 1.1.0 的 small0 checkpoint，首次使用允许下载约 8.1 MB 权重。
3. Beat This 默认在 CPU 运行，避免与主转录模型争用 GPU。
4. FluidSynth 优先使用已验证的系统安装，其次使用项目管理缓存；仅在请求音频预览时触发检查和下载。
5. JSON 升级为 schema version 2 的对象；JSONL 第一行是 metadata，后续每行是事件。
6. 低质量或失败的时序分析显式回退到 120 BPM 和 4/4，并允许用户手动覆盖。
7. MuScriptor 采用统一增强乐谱中间层，而不是事后修补上游 MIDI。
8. GAME 人声 MIDI 只复用通用时序，本轮不复用 MuScriptor 力度算法。

## 4. 总体架构

### 4.1 模块边界

新增通用包：

    module/music_analysis/
        __init__.py
        types.py
        timing.py
        beat_this_runtime.py
        midi_timing.py
        cache.py

职责：

- types.py：定义 MusicTiming、TempoPoint、Meter、AnalysisQuality 和 provenance。
- beat_this_runtime.py：懒加载并复用 Beat This small0，管理设备、权重缓存和推理。
- timing.py：将 beat/downbeat logits 后处理为节拍、下拍、BPM、tempo map、拍号和质量判断。
- midi_timing.py：统一 seconds、beats、ticks 的双向映射和 MIDI tempo meta 生成。
- cache.py：按源音频状态、算法版本、checkpoint 和手动覆盖生成缓存/批处理签名。

扩展 MuScriptor 包：

    module/muscriptor_tool/
        score.py
        dynamics.py
        midi_export.py
        structured_export.py
        musicxml_export.py
        fluidsynth_runtime.py

职责：

- score.py：配对 NoteStartEvent/NoteEndEvent，复刻上游音符清理语义，生成统一 AnalyzedScore。
- dynamics.py：从完整混音估算相对力度。
- midi_export.py：用 mido 直接写 Type 1 MIDI，不调用上游固定 120/100 的导出器。
- structured_export.py：写 JSON v2 和 JSONL。
- musicxml_export.py：从 AnalyzedScore 直接构建 music21 Score。
- fluidsynth_runtime.py：发现、验证、安装和解析 FluidSynth 可执行文件。

### 4.2 依赖方向

    原始音频
       |
       +--> 通用 MusicTiming
       |
       +--> MuScriptor 原始事件
                |
                +--> EnrichedNote + 相对力度
                         |
                         +--> MIDI
                         +--> JSON v2
                         +--> JSONL
                         +--> MusicXML

GAME 人声转 MIDI 只依赖 MusicTiming 和 midi_timing，不依赖 MuScriptor 的 score、dynamics、MusicXML 或 FluidSynth。

### 4.3 为什么不事后修补 MIDI

事后仅替换 set_tempo 会改变播放速度。若再反向解析上游 MIDI 重映射 tick，还必须同时恢复乐器、事件索引和秒级时间，容易让 JSON、MusicXML 与 MIDI 分叉。统一中间层只进行一次事件配对和时间映射，输出器成为纯投影，减少不一致。

## 5. 核心数据模型

### 5.1 MusicTiming

建议的不可变数据结构：

~~~python
MusicTiming(
    duration_seconds: float,
    beats_seconds: tuple[float, ...],
    downbeats_seconds: tuple[float, ...],
    global_bpm: float,
    tempo_map: tuple[TempoPoint, ...],
    numerator: int,
    denominator: int,
    tempo_source: Literal["detected", "manual", "fallback"],
    meter_source: Literal["detected", "manual", "fallback"],
    quality_score: float,
    usable: bool,
    warnings: tuple[str, ...],
    algorithm: str,
    algorithm_version: str,
    checkpoint: str | None,
)
~~~

TempoPoint：

~~~python
TempoPoint(
    time_seconds: float,
    beat_position: float,
    bpm: float,
)
~~~

约束：

- tempo_map 至少有一个点，首点的 beat_position 为 0。
- bpm 必须有限且在 30 到 300 之间。
- tempo_map 的 time_seconds 和 beat_position 严格递增。
- 手动 BPM 产生单点 tempo map。
- fallback 固定产生 120 BPM 单点 tempo map。
- tempo_source 与 meter_source 分开记录，因为 BPM 可检测成功而拍号回退。

### 5.2 AnalysisQuality

Beat This 的 sigmoid 输出不是经过校准的置信概率，不对外命名为 confidence。quality_score 是工程质量分，用于展示和诊断，来源包括：

- 有效节拍数量。
- 相邻节拍间隔的有限性和局部连续性。
- 异常间隔比例。
- 下拍构成完整小节后的众数一致性。

quality_score 范围为 0 到 1，但只表达本算法版本中的相对质量，不作为唯一可用性门槛。第一版定义为：

- beat_count_score = clamp((beat_count - 3) / 13, 0, 1)。
- valid_interval_score = 清洗后 interval 数 / 原始 interval 数。
- continuity_score = 1 - clamp(relative_robust_sigma / 0.15, 0, 1)。
- bar_score = 拍号众数支持率；无法形成完整小节时取 0.5。
- quality_score = 0.25 × beat_count_score + 0.30 × valid_interval_score + 0.25 × continuity_score + 0.20 × bar_score。

usable 由第 6.2 节的硬条件决定，避免把人为加权分数误用成模型概率。所有阈值属于版本化默认值，不是科学意义上的置信概率。

### 5.3 EnrichedNote

~~~python
EnrichedNote(
    index: int,
    pitch: int,
    instrument: str,
    gm_program: int | None,
    is_drum: bool,
    start_seconds: float,
    end_seconds: float,
    start_beat: float,
    end_beat: float,
    velocity: int,
    velocity_source: Literal["spectral_relative", "default"],
)
~~~

约束：

- pitch 范围 0 到 127。
- 清理后保留的音符必须满足 end_seconds 严格大于 start_seconds。
- velocity 范围 1 到 127，本设计正常输出范围为 24 到 120，fallback 为 80。
- start_beat/end_beat 由同一 MusicTiming 映射，不能由各导出器自行计算。

### 5.4 AnalyzedScore

~~~python
AnalyzedScore(
    timing: MusicTiming,
    notes: tuple[EnrichedNote, ...],
    source_audio: str,
    duration_seconds: float,
    warnings: tuple[str, ...],
    schema_version: int = 2,
)
~~~

所有符号输出接收 AnalyzedScore，不再接收裸事件和隐含默认 BPM。

### 5.5 Canonical note 清理

统一乐谱层替代 loaded.midi_bytes 后，必须保留 MuScriptor 0.2.1 的既有清理语义，而不是只做朴素事件配对：

1. 只有收到对应 NoteEndEvent 的 start note 才进入 canonical notes；流结束时仍未闭合的 start note 丢弃并记录 warning。
2. end 引用不存在的 start_event_index 视为损坏事件流并终止该次转录，保持上游当前的失败语义。
3. onset 大于 offset 时，把 offset 修正为 onset + 0.01 秒。
4. 非鼓音符短于 0.01 秒时扩展到 0.01 秒。
5. 按 (gm_program, pitch, is_drum) 分组；同组相邻音符重叠时，把前一音符截断到后一音符 onset。
6. 截断后 onset 不小于 offset 的音符丢弃。
7. 最终按 (onset, is_drum, gm_program, pitch, offset) 排序。

以上规则在力度计算和四种导出之前执行。回归测试使用相同事件同时调用上游 MuScriptor 0.2.1 清理路径与本地 canonical builder，比较清理后的 onset、offset、program、pitch、is_drum，而不是要求增强 MIDI 的字节完全相同。

## 6. 时序分析算法

### 6.1 模型与前处理

- 固定依赖 beat-this==1.1.0。
- 使用 small0 checkpoint。
- 复用项目现有音频解码路径，避免每个模型各自处理格式。
- 使用 Beat This 的 Audio2Frames 得到 beat/downbeat logits。
- 使用官方 minimal postprocessor：局部峰值且 logit 大于 0，对应 sigmoid 0.5。
- 不使用 DBN，不额外依赖 madmom。
- 模型在单进程批处理中只加载一次。

### 6.2 BPM 可用性

检测结果满足以下条件才可用于 tempo：

1. 至少 4 个 beat。
2. 相邻 beat interval 均可计算，清洗后仍有足够样本。
3. 由 interval 得到的 BPM 有限并落在 30 到 300。

间隔清洗的第一版规则：

1. 丢弃非有限、非正值以及小于 0.2 秒或大于 2.0 秒的 interval。
2. 对每个 interval 取最多前后各 2 个 interval 的局部窗口。
3. 计算局部中位数和 MAD；偏差大于 max(3 × 1.4826 × MAD, 0.35 × 局部中位数) 的点视为漏拍/重拍离群值。
4. 清洗后至少保留 3 个 interval，且保留比例至少 60%。

全局 BPM 为清洗后 60 / interval 的中位数。不得为了得到“常见 BPM”而自动做二倍或二分折叠，因为这会隐藏模型的拍层级判断并可能破坏音符网格。

### 6.3 稳定速度与变速

- 计算 relative_robust_sigma = 1.4826 × MAD(local_bpm) / median(local_bpm)。
- 若 relative_robust_sigma 不超过 3%，且 local_bpm 的第 90 与第 10 百分位之差不超过 8 BPM，输出单点 tempo map。
- 若波动超过阈值，使用 5-beat 中位数窗口平滑局部 BPM。
- 相邻 tempo point 差值小于 1 BPM 时合并，避免 MIDI 中产生密集且无意义的 tempo 事件。
- 每个 tempo point 同时记录 seconds 和 beat position。

MIDI tick 映射使用 tempo map 的分段积分。这样变速音乐仍保持音符与原音频的秒级对齐。

坐标原点固定为音频时间 0 秒：

- tempo_map 的首点固定为 time_seconds=0、beat_position=0，BPM 使用首个有效平滑值。
- 后续点放在对应 interval 的时间中点，beat_position 由前一段 tempo 积分得到。
- EnrichedNote 的 start_beat/end_beat 使用这个非负 transport beat 坐标。
- MusicXML 另以第一个可靠 downbeat 的 transport beat 为小节原点；此前内容成为 measure 0 弱起。
- 若没有可靠 downbeat，音频时间 0 即谱面小节原点，并使用 fallback/manual meter。

### 6.4 拍号

拍号通过相邻 downbeat 之间的 beat 数量推断：

- 至少 3 个完整小节。
- 众数支持率至少 75%。
- 自动接受 2/4、3/4、4/4。
- 6/8 和其他复合拍不做猜测；返回 4/4 fallback 并警告用户手动指定。

拍号检测失败不应丢弃已可用的 tempo。此时 tempo_source 可以是 detected，而 meter_source 是 fallback。

### 6.5 手动覆盖与回退

- --bpm 接受 30 到 300 的有限浮点数。
- --time-signature 接受规范的 numerator/denominator，初期支持常见二进制分母。
- 同时提供 BPM 和拍号时完全跳过 Beat This。
- 只覆盖其中一个时，另一个仍尝试检测；失败时单独 fallback。
- 模型下载、解码或推理失败时，未覆盖项回退到 120 BPM、4/4。
- 回退必须进入 warnings、CLI 日志、GUI 状态和结构化输出 metadata。

### 6.6 缓存

分析缓存键至少包含：

- 原始音频规范化路径、大小和 mtime_ns。
- 解码/分析 schema 版本。
- Beat This 包版本。
- checkpoint 名称。
- 手动 BPM 和拍号。
- 影响结果的阈值版本。

缓存内容只保存 MusicTiming，不保存模型 logits 或大型谱图。

## 7. MuScriptor 相对力度

### 7.1 原则

全频短窗 RMS 容易把同时发生的鼓点、和弦或伴奏能量错误分配给目标音符。准确的演奏力度恢复依赖源分离、谱成分分解和特定音源映射，当前项目没有足够信息做绝对恢复。因此采用“音高条件攻击能量 + RMS”的稳健相对估计。

### 7.2 分块公共频谱

不得保留整首音乐的 STFT 矩阵。以 1 小时、12 ms hop、2049 个 float32 频率 bin 估算，完整功率谱约占 2.46 GB，违背批处理和 JSONL 的低内存目标。

力度估计采用固定内存的分块扫描：

- 每个 core chunk 为 30 秒。
- chunk 两侧各读取 250 ms overlap，足以覆盖 80 ms 前后窗口和约 93 ms STFT 半窗。
- 窗长约 93 ms，hop 约 12 ms。
- 每个 chunk 单独计算 torch.stft 功率谱，处理完立即释放。
- onset 落在当前 core chunk 的音符由当前 chunk 唯一负责，overlap 只提供上下文，避免重复统计。
- 只保留每个音符的 raw intensity 标量和必要的 instrument 分组统计，不保存 STFT frame。
- 频率 bin 与音符基频、谐波索引预计算。

独立 MuScriptor 对一个 AnalyzedScore 扫描一次音频。音频分离工作流先收集同一首歌全部 MuScriptor candidate 的 canonical notes，再把它们合并交给一次分块扫描；扫描结果按 candidate/note index 分发，避免每个 stem 重复分析完整混音。

峰值内存因此为 O(30 秒频谱 + 音符数)，而不是 O(音频时长 × 频率 bin)。精确样本数根据输入采样率换算，并作为算法版本的一部分写入 metadata。

### 7.3 非打击乐

对每个音符：

1. 取 onset 前约 80 ms 作为背景窗口。
2. 取 onset 后约 80 ms 作为攻击窗口。
3. 累计基频和前 4 个谐波附近的功率。
4. 计算攻击相对背景的增量。
5. 与 onset 附近短窗 RMS 混合，得到 raw intensity。

频带宽度随 STFT 分辨率确定，避免只取单 bin 导致微小调音偏差。

### 7.4 打击乐

鼓类没有稳定谐波基频，使用 onset 附近的正向 spectral flux 与短窗 RMS 混合。

### 7.5 归一化与 fallback

- 按 instrument 分组，避免鼓、贝斯和高音乐器相互压缩动态范围。
- 每组使用第 5 到第 95 百分位稳健归一化。
- 映射到 MIDI velocity 24 到 120。
- 少于 4 个音符，或估算动态范围小于 3 dB 时，该组全部使用 80。
- 任意 STFT/特征计算失败时，所有受影响音符使用 80，记录 warning，但不阻断导出。

velocity_source 为 spectral_relative 或 default。不得使用 recovered、true 等暗示绝对还原的名称。

## 8. 输出契约

### 8.1 MIDI

MuScriptor 写 Type 1 MIDI：

- conductor track 写 tempo map 和 time signature。
- 每个乐器一个 track。
- melodic instrument 写代表性的 GM program change。
- drums 使用 MIDI channel 10。
- note_on 显式写入 EnrichedNote.velocity。
- 音符 tick 由 MusicTiming 分段映射，不使用固定 BPM 公式。
- 同 tick 的事件排序保证 note_off 在重复音高的 note_on 之前，避免粘音。

GAME 人声 MIDI：

- conductor track 使用同一 tempo map 和 time signature。
- 人声音符用同一 seconds-to-ticks 映射。
- velocity 本轮固定显式写 64。
- 生成 metadata sidecar，记录时序来源、算法版本、warnings 和输入签名。

### 8.2 JSON v2

顶层从事件数组升级为对象：

~~~json
{
  "schema_version": 2,
  "analysis": {
    "global_bpm": 123.4,
    "tempo_source": "detected",
    "time_signature": "4/4",
    "meter_source": "detected",
    "tempo_map": [
      {"time_seconds": 0.0, "beat_position": 0.0, "bpm": 123.4}
    ],
    "beats_seconds": [0.0, 0.486],
    "downbeats_seconds": [0.0],
    "quality_score": 0.86,
    "usable": true,
    "algorithm": "beat_this",
    "algorithm_version": "1.1.0",
    "checkpoint": "small0",
    "warnings": []
  },
  "events": []
}
~~~

start 事件增加：

- start_beat
- velocity
- velocity_source
- gm_program

end 事件增加 end_beat。

不写入原始 logits、STFT、RMS 数组或其他大体积中间特征。

### 8.3 JSONL

- 第一行固定是 record_type=metadata，内容与 JSON v2 的 analysis 等价。
- 后续每行是 record_type=event。
- JSON 与 JSONL 由同一序列化函数生成字段，避免 schema 漂移。

最终力度需要整首歌内的百分位统计，因此不能把刚生成的原始事件直接当作最终 JSONL 流。实现采用轻量临时 spool 或第二遍读取：

1. 流式记录最小事件/音符信息。
2. 完成全曲统计和力度映射。
3. 原子写出最终 JSONL。

spool 在成功或失败后清理，不能被批处理当作完成输出。

### 8.4 MusicXML

- 新格式名 musicxml。
- 文件名为 source-stem.musicxml。
- CLI 单文件模式支持写 stdout。
- Python 3.10 固定使用 music21==9.9.1；Python 3.11 和 3.12 固定使用 music21==10.5.0。
- 直接从 AnalyzedScore 构建，不经由临时 MIDI。

谱面规则：

- 每个 instrument 一个 Part，使用规范名称和 GM program。
- 打击乐使用 percussion clef 和 GM drum mapping。
- 第一个可靠 downbeat 之前的内容写入 measure 0 作为弱起。
- 节奏量化到十六分音符或三连音八分音符，选取误差更小的网格。
- 同 onset、同 duration 的音符合并为 chord。
- 重叠声部使用 makeVoices。
- 跨小节时值使用 makeTies。
- 生成 rests、beams 和 accidentals。
- tempo map 转为 MetronomeMark。
- 每个 Part 每小节取 velocity 中位数，映射 pp、p、mp、mf、f、ff；只在持续变化时写 dynamics，避免标记噪声。
- 空转录仍输出合法的一小节全休止，并记录 warning。

导出验证：

1. 调用 isWellFormedNotation。
2. 写出后用 music21 重新读取。
3. 验证 Part、Measure、音符/和弦、tempo 标记的基本数量。
4. 解析失败时仅标记 MusicXML 输出失败，不删除已成功的 MIDI/JSON/JSONL。

## 9. 各工作流数据流

### 9.1 独立 MuScriptor

    解码输入
      -> 获取/分析 MusicTiming
      -> MuScriptor 转录一次
      -> 配对事件
      -> 计算一次 STFT 与力度
      -> 构建 AnalyzedScore
      -> 按请求分别导出 MIDI/JSON/JSONL/MusicXML
      -> 可选预览

### 9.2 MuScriptor 批处理

- Beat This 模型每个进程加载一次。
- MuScriptor 模型维持现有生命周期。
- 每首输入只做一次 timing 和 dynamics 分析。
- 多格式输出共享 AnalyzedScore。
- 单格式失败不阻断其他格式，最终状态为 partial 并报告失败格式。

### 9.3 音频分离

    原始完整混音
      -> MusicTiming 一次
      -> vocal stem -> GAME notes -> vocal MIDI
      -> dry vocal -> GAME notes -> vocal MIDI
      -> instrument stems -> MuScriptor -> EnrichedScore -> 各格式

时序分析必须针对 source_path 的原始完整混音，而不是 stem input_path。原因是鼓点和下拍在单一人声/乐器 stem 中可能缺失，各 stem 独立分析也会产生互相矛盾的 tempo map。MuScriptor stems 先完成 raw event 转录和 canonical 清理，再对同一 source_path 做一次分块力度扫描，最后逐 stem 导出。

### 9.4 MuScriptor 上游 WebUI

上游 /transcribe 当前直接返回上游固定 120/100 的 MIDI，/auralize 直接调用命令名 fluidsynth。上游 create_app 没有 MIDI exporter 或 FluidSynth executable 的注入点，不能依靠模糊 monkey patch。

项目新增本地 web app factory：

1. 创建父 FastAPI app，并在 mount 之前注册本地 /transcribe 和 /auralize，因此这两个路由拥有最高匹配优先级。
2. 把上游 create_app 返回的 app mount 到 /，继续提供 /health、/instruments、/soundfonts 和编译静态前端。
3. 本地 /transcribe 保留上游 0.2.1 的 multipart 参数、SSE start/end/progress/final-midi 协议、单转录锁、取消和 no-cache header。
4. 上传音频写入请求级临时文件；原始事件继续实时发送，转录结束后从同一临时音频构建 MusicTiming、canonical notes、relative velocity 和最终 MIDI。
5. 本地 /auralize 保留 mode=mix/synth 契约，但调用项目本地 renderer，不导入上游 synthesize/auralize。
6. 路由契约测试固定上游 0.2.1 的请求字段、SSE record shape、状态码和响应 media type，升级 MuScriptor 时显式审查差异。

MusicXML 不加入上游编译 WebUI，仍由项目 CLI 与 NiceGUI 提供。

## 10. FluidSynth 按需运行时

### 10.1 触发边界

只有用户请求 auralization/preview 时才解析 FluidSynth。单纯导出 MIDI、JSON、JSONL 或 MusicXML：

- 不检查 FluidSynth。
- 不访问网络。
- 不创建工具缓存。

预览前先完成音频 codec probe。若输入无法解码，不应下载 FluidSynth。

### 10.2 解析顺序

1. 系统 PATH 中可执行文件，且 --version 验证成功。
2. 项目管理缓存中的已验证版本。
3. Windows x64 自动下载固定版本。
4. 其他平台返回可操作的安装说明。

PreviewRuntime 必须保存已经验证的 fluidsynth_executable 绝对路径。项目本地 renderer 直接执行：

    <absolute-fluidsynth> -ni -F <temporary.wav> -r 44100 <soundfont> <midi>

本地 renderer 负责读取合成 WAV，并复用当前预览的 mono synth 或 original-left/synth-right 混音、RMS 对齐和目标格式写出逻辑。不得再调用上游 synthesize/auralize，因为它们把命令硬编码为 fluidsynth，无法接收私有安装路径。

### 10.3 Windows x64 固定资产

- FluidSynth 版本：2.5.6
- 资产：fluidsynth-v2.5.6-win10-x64-cpp11.zip
- URL：https://github.com/FluidSynth/fluidsynth/releases/download/v2.5.6/fluidsynth-v2.5.6-win10-x64-cpp11.zip
- SHA256：a4b8bd4f133b7b6770537f6c18b2b2b93579338d51e26f777d025e40e15a7e81
- 缓存：%LOCALAPPDATA%/qinglong-captions/tools/fluidsynth/2.5.6

安装步骤：

1. 使用 filelock 对版本目录加锁。
2. 下载到 .part。
3. 校验完整 SHA256。
4. 拒绝绝对路径、盘符路径和包含 .. 的 ZIP member，防止路径穿越。
5. 解压到 staging 目录。
6. 验证 fluidsynth --version。
7. 写包含版本、资产和 hash 的 manifest。
8. 原子 rename 到最终版本目录。
9. subprocess 始终使用绝对 exe 路径，不修改当前或永久 PATH。

并发进程等待同一锁；锁内二次检查，避免重复下载。损坏缓存隔离后重装，不在原目录上覆盖半成品。

### 10.4 平台与退出机制

- Linux/macOS 只接受已安装的系统 fluidsynth，并给出对应包管理器提示。
- 提供环境变量关闭自动下载，服务端、离线和受管环境可以强制 fail-fast。
- 下载/安装失败要报告根因、目标缓存路径和恢复建议。
- 预览失败不能影响已完成的符号输出。

## 11. CLI、GUI 与兼容性

### 11.1 MuScriptor CLI

新增：

- --format musicxml，或把 musicxml 加入多格式 choices。
- --bpm FLOAT。
- --time-signature N/D。
- 输出摘要显示 detected/manual/fallback、BPM、拍号和 warnings。

### 11.2 GAME 人声 CLI

新增相同的 --bpm 与 --time-signature。生成 MIDI 时显式写 tempo map、time signature 和 velocity 64。

### 11.3 NiceGUI 与音频分离 GUI

- MuScriptor 输出格式增加 MusicXML。
- 时序设置提供 Auto/Manual 模式；手动模式显示 BPM 和拍号输入。
- 音频分离的 vocal MIDI 与 stem MIDI 共用一组 timing 控件。
- fallback 和自动安装错误使用现有任务状态/错误区域展示，不新增解释型营销文案。

### 11.4 JSON 兼容

JSON v2 是有意的结构升级。项目内部消费者必须同时识别：

- v1：顶层事件数组，无 analysis。
- v2：顶层对象，包含 schema_version、analysis、events。

新导出只写 v2。不得根据“文件存在”把旧 v1 当作当前任务的完成结果。

## 12. 批处理签名、原子性与清理

完成签名至少包含：

- 输入路径、大小和 mtime_ns。
- 原始完整混音状态。
- timing schema、Beat This 版本/checkpoint、阈值版本。
- BPM/拍号覆盖。
- dynamics 算法版本。
- 输出 schema 和请求格式。

规则：

- 旧 MIDI 或无 timing metadata 的 GAME 输出不算完成，需重建。
- 临时文件、spool、.part 和 staging 不算正式输出。
- MIDI、JSON、JSONL、MusicXML 和 sidecar 分别写临时文件后原子 replace。
- manifest/清理逻辑认识 .musicxml 和新增 sidecar。
- 清理只删除项目声明拥有的产物，不删除未知文件。

## 13. 错误处理

| 故障 | 行为 |
| --- | --- |
| Beat This 下载/加载/推理失败 | 未手动覆盖项回退 120 BPM、4/4，写 warning |
| tempo 可用但 meter 不可用 | 保留检测 tempo，meter 回退 4/4 |
| dynamics 计算失败 | MuScriptor velocity 使用 80，其他导出继续 |
| 单个格式序列化失败 | 保留其他成功格式，任务为 partial |
| MusicXML 回读失败 | 删除该格式临时产物，不影响其他格式 |
| FluidSynth 缺失且不请求预览 | 完全忽略 |
| FluidSynth 安装或预览失败 | 报告预览失败，符号输出保持成功 |
| 临时 spool 中断 | 下次运行清理并重建 |
| 手动参数非法 | 在模型加载和转录前立即报参数错误 |

日志和结构化 warning 必须包含可操作原因，不能只记录异常类名。

## 14. 依赖与许可

建议依赖：

- muscriptor-local extra 增加 beat-this==1.1.0。
- muscriptor-local 在 Python 3.10 增加 music21==9.9.1，在 Python 3.11/3.12 增加 music21==10.5.0，使用 PEP 508 python_version marker。
- vocal-midi extra 增加 beat-this==1.1.0，不增加 music21。
- 继续使用已有 torch、mido、numpy、filelock。

Beat This 代码与权重采用 MIT 许可；实现时在依赖文档/第三方声明中保留来源。FluidSynth 为 LGPL；项目下载原始官方二进制，不修改或静态链接，缓存 manifest 记录官方资产来源。

## 15. 测试策略

### 15.1 时序单元测试

- 合成稳定 click：全局 BPM、单点 tempo map。
- 合成变速 click：tempo map 变化方向和秒级对齐。
- 2/4、3/4、4/4 downbeat 序列。
- 6/8/含糊 meter 回退并警告。
- 少于 4 beats、非法间隔、模型异常回退。
- BPM/拍号单独覆盖和同时覆盖；同时覆盖不调用模型。
- seconds -> beat -> ticks -> seconds round-trip 在规定误差内。

模型单元测试使用注入的 logits/beat times，不触发真实下载。

### 15.2 力度测试

- 合成同音高不同振幅，velocity 单调增加。
- 加入无关频带冲击，音高条件能量比纯 RMS 更稳定。
- 不同 instrument 分组独立归一化。
- 鼓 spectral flux 路径。
- 少于 4 notes、动态范围小于 3 dB、STFT 异常使用 80。
- 跨 30 秒 chunk 边界的 onset 与单块参考结果一致。
- 1 小时合成输入通过 fake chunk reader 验证同时存活的频谱不超过一个 core chunk 加 overlap。

### 15.3 输出测试

MIDI 用 mido 回读并验证：

- conductor tempo/time signature。
- 变速 tempo event。
- program change、drum channel。
- velocity 范围与 EnrichedNote 一致。
- 音符秒级位置在容差内。

JSON/JSONL 验证：

- schema v2。
- metadata 等价。
- 事件顺序、数量、beat 和 velocity 等价。
- v1 reader 兼容。

MusicXML 用 music21 回读并验证：

- well-formed notation。
- part、measure、note/chord 数量。
- voice、tie、tempo、dynamic 和空谱。
- MuseScore 可读性保留一项人工 smoke checklist，但自动测试不依赖 MuseScore 安装。
- Python 3.10/music21 9.9.1 与 Python 3.11-3.12/music21 10.5.0 运行相同的 exporter contract tests。

### 15.4 FluidSynth 测试

- 假 ZIP + 假 executable 的成功安装。
- SHA256 不匹配。
- ZIP path traversal。
- 并发锁和锁内二次检查。
- staging 原子替换与中断恢复。
- 自动下载 opt-out。
- 系统版本优先于缓存。
- 未请求预览时断言无网络和无 probe。

### 15.5 集成测试

- MuScriptor CLI 四格式共享 timing/velocity。
- canonical builder 与 MuScriptor 0.2.1 的 validate_notes + trim_overlapping_notes 结果一致。
- GAME vocal MIDI 不再含固定 120 公式，velocity 仍为 64。
- 音频分离每首原曲只调用一次 timing analyzer。
- NiceGUI 参数传递和 MusicXML 选项。
- MuScriptor WebUI 最终 MIDI 走本地 exporter，auralize 使用绝对 FluidSynth 路径。
- 旧输出因签名缺失被升级。
- 默认测试套件不下载真实模型或二进制；真实 Beat This/FluidSynth smoke test 通过显式环境开关运行。

## 16. 验收标准

1. 稳定 tempo 测试音频导出的 MIDI tempo 不再固定为 120，且播放总时长与原事件秒数对齐。
2. 变速音频包含 tempo map，MIDI、JSON、JSONL 和 MusicXML 使用同一时序。
3. 分析失败时四种格式明确记录 fallback 120 BPM、4/4 和 warning。
4. 手动 BPM + 拍号无需下载/加载 Beat This。
5. 同一 MuScriptor 乐器内更强攻击的音符通常具有更高 velocity，低信息场景稳定回退 80。
6. JSON v2 与 JSONL metadata/events 可机械比较为等价内容。
7. MusicXML 可被 music21 回读、通过基本 notation 检查，并在 MuseScore 中可编辑。
8. GAME 人声 MIDI 复用通用时序，不再使用 onset * 120 * 8 的固定公式。
9. 音频分离对原始完整混音只分析一次，并将结果复用于所有候选。
10. Windows x64 首次请求预览时可私有安装已校验的 FluidSynth；不修改 PATH。
11. 不请求预览时无 FluidSynth 检查或下载。
12. MuScriptor 上游 WebUI 导出的 MIDI 也具有检测 tempo 和相对力度。
13. 所有新增测试通过，默认测试过程无外部网络依赖。
14. 一小时输入的力度分析内存上界由固定 chunk 大小决定，不随音频时长线性增长。
15. Python 3.10、3.11、3.12 均能解析 muscriptor-local 依赖并运行 MusicXML contract tests。

## 17. 被否决的方案

### 17.1 仅修补上游 MIDI

改动小，但 tempo 与 tick 强耦合；容易改变时长，而且 JSON、MusicXML 会形成各自的推导逻辑。否决。

### 17.2 维护 MuScriptor 私有 fork

可以从源头改导出，但会承担上游同步、模型 API 漂移和发布维护成本。项目包装层已经存在，更适合本地接管输出。否决。

### 17.3 Beat This 完整模型

主模型约 78 MB，论文结果相对 small 模型只提升约 0.3 F1；对当前 BPM/拍号用途不值得增加约十倍下载。否决。

### 17.4 librosa 传统 beat tracker

依赖和模型下载更少，但对跨风格、弱打击乐和变速音乐的稳健性较低，且项目最终仍需要可靠 downbeat。否决。

### 17.5 纯全频 RMS 力度

实现简单，但同时发声和鼓点会明显污染单音力度。采用音高条件攻击能量与 RMS 混合。否决。

### 17.6 MIDI 转 MusicXML

会丢失原始 instrument/event 语义，并把 MIDI tick 量化误差带入制谱。MusicXML 应直接从 AnalyzedScore 构建。否决。

### 17.7 使用系统包管理器自动安装 FluidSynth

需要权限，平台行为不一致，会改变用户系统并产生版本漂移。Windows 使用项目私有固定资产，其他平台提供明确安装提示。否决。

## 18. 版本化默认值

以下数值是本设计第一版的工程默认值：

- tempo 有效范围 30 到 300 BPM。
- 至少 4 beats。
- tempo 平滑窗口 5 beats。
- 相邻 tempo 合并阈值 1 BPM。
- meter 至少 3 个完整小节、众数支持率 75%。
- STFT core chunk 30 秒、两侧 overlap 各 250 ms。
- STFT 窗长约 93 ms、hop 约 12 ms。
- onset 前后窗口各约 80 ms。
- 基频加前 4 个谐波。
- velocity 第 5 到第 95 百分位映射到 24 到 120。
- 少于 4 notes 或动态范围小于 3 dB 时 fallback 80。

实现时这些值集中在带算法版本的配置对象中，并由测试固定。调整它们必须更新算法版本、批处理签名和相应测试，不能散落为魔法数字。

## 19. 参考资料

- Beat This 论文：https://arxiv.org/abs/2407.21658
- Beat This 官方仓库与 minimal postprocessor：https://github.com/CPJKU/beat_this
- music21 项目：https://pypi.org/project/music21/
- music21 MIDI/量化文档：https://music21.org/music21docs/moduleReference/moduleConverter.html
- 音符强度估计研究：https://mac.kaist.ac.kr/pubs/JeongNam-aes2017.pdf
- FluidSynth 官方下载：https://www.fluidsynth.org/download/
- FluidSynth 安装说明：https://www.fluidsynth.org/wiki/Download/
