# MuSViT 乐谱 OMR

乐谱工具使用固定版本的 MuSViT ONNX 模型转录乐谱图片和 PDF 页面，可输出
MusicXML、MIDI 或两者，并为每页保留 Kern 与 token 诊断文件。

主要入口位于 GUI 的 Tools 页面。等价 CLI：

```powershell
.\.venv\Scripts\python.exe -m module.sheet_music_musvit INPUT_PATH `
  --output_format both
```

`INPUT_PATH` 可以是图片、PDF 或目录，目录支持递归扫描。PDF 按页流式渲染：
当前页完成推理和导出并释放图像后，才会请求下一页。

不指定 `--output_dir` 时，文件输入默认写入输入文件旁的
`musvit_omr_output`，目录输入默认写入该目录内的 `musvit_omr_output`。

每张图片使用独立输出目录；PDF 使用 `page_0001`、`page_0002` 等分页目录。
成功页面包含：

- `tokens.json`
- `score.krn`
- `metadata.json`
- `score.musicxml` 和/或 `score.mid`

根目录的 `manifest.json` 记录成功、跳过和失败状态。Kern 无效或生成被截断时，
工具保留诊断文件，但不会把该页伪装成成功的 MusicXML/MIDI。PDF 的所有页面
都成功后，PDF 输出目录根部还会生成一份按页序合并的 `score.musicxml` 和/或
`score.mid`；任一页面或聚合步骤失败时不会保留整本结果，分页诊断仍然可用。
整本聚合以 Kern 的公共小节线对齐声部；模型明确输出为 `.` 的静默小节会使用
隐藏休止补齐，不会因 music21 省略空小节而把后续页面错位。
谱号、调号、拍号和常用/二分拍符号按 Kern 中的位置重新锚定；合并时会省略
下一页重复的相同页头上下文，但不会改写模型实际输出的上下文变化。

依赖 profile 为 `musvit-onnx`，默认模型 revision 固定在
`config/model.toml`。PDF DPI 只控制模型固定 1024 x 1024 bilinear 缩放前的
栅格化；提高 DPI 不保证识别率一定提升。
