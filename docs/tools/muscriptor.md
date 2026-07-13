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
.\2.7.1.muscriptor_webui.ps1 -Model small -Device cuda:0 -BatchSize 4 -Port 8222
.\2.7.1.muscriptor_webui.ps1 -NoBrowser
```

脚本把 `muscriptor-local` 安装到项目共享 `.venv`，通过项目优化后的 SDPA runtime 加载模型，同时继续使用官方 WebUI 和 HTTP 服务；服务健康检查通过后自动打开浏览器。`-NoBrowser` 可关闭自动打开。`-Model`（别名 `-ModelSize`）支持 `small`、`medium`、`large`，默认使用 `large`；设备默认 `auto`。`-BatchSize` 默认 `0`：启动时直接读取已记录的模型显存曲线，按总显存选择偶数 batch，不再重新执行 BS1/BS2 校准。OOM 时会自动改用较小批次；Windows 下检测到当前进程新增共享 GPU 显存占用后，后续 batch 会减 2。每次请求完成或取消后都会释放临时张量和空闲 CUDA cache，但保留模型权重。CPU 使用 BS1。传入正整数可跳过 profile 选择，但仍保留显存保护。batch 大于 1 会提高吞吐，但同批后续 5 秒块的页面事件会延后出现。不要使用 `uvx muscriptor serve`：`uvx` 会解析独立工具环境，不复用项目 `.venv`，可能重复安装另一套 Torch/CUDA 依赖。

CLI、官方 WebUI 启动器、批量转录与音频分离共用 `config/muscriptor_batch_profiles.toml` 的模型显存曲线和自适应 CUDA runtime；加载权重前，显存不足的 `auto` 会回退 CPU，显式 CUDA 则直接阻止任务。CUDA 分配器预算会扣除分段预留和当前设备上 PyTorch 之外的显存占用。打开任一 GUI 页面时，还会把用户所选 GPU 的总显存代入 BS1/BS2 与完整 batch 验证后的边际显存公式来选择偶数 batch，参考 GPU 只标明测量来源。在不超过 16 GiB 时使用 1 GiB 预留，small、medium、large 的最低总显存约为 1.90、3.25、10.28 GiB。用户手动修改前，切换模型或设备会重新计算。

## 输出与恢复

批处理按输入相对路径建立项目目录；默认输出到输入位置下的 `muscriptor_output`。每项包含以源文件名为 stem 的 `<source-stem>.mid`、`events.json`、`events.jsonl`、`metadata.json`，根目录包含 `manifest.json`。`metadata.json` 分别记录手动限制的 `instruments` 和模型实际识别到的 `detected_instruments`。同一输入只推理一次，完成签名、原子写入和输出锁支持中断恢复。

## 试听

`preview-mode midi` 导出合成 MIDI 音频；`comparison` 输出左声道原音、右声道合成 MIDI。试听需要 PATH 中的 FluidSynth，并使用官方 `MuseScore_General.sf2` SoundFont。FluidSynth 不可用时，MIDI、JSON 和 JSONL 仍可正常生成。

事件只包含 onset、offset、pitch 和 instrument。模型不恢复原始力度；密集混音、少见音色、重处理音频和部分合唱材料可能降低准确度。
