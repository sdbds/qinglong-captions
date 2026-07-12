# MuSViT 乐谱 embedding

乐谱扫描工具使用 MuSViT ONNX 从扫描乐谱图片或 PDF 页面提取 embedding，并生成 manifest。它不是音频转 MIDI；符号音乐转录请看 [MuScriptor](muscriptor.md)。

## 入口

当前主要入口是 GUI Tools 页面。Python CLI：

```powershell
python -m module.sheet_music_musvit --help
```

依赖 profile 为 `musvit-onnx`。

## 输入与输出

- 输入：图片、PDF 或目录。
- 输出：逐页 embedding 和根目录 `manifest.json`。
- `preprocess_mode` 支持整页缩放或补白方图。
- `pdf_dpi` 控制 PDF 渲染分辨率。
- `recursive`、`skip_completed`、`overwrite` 控制批处理与恢复。

提高 PDF DPI 会增加细节，也会增加内存和处理时间。先用单页确认预处理模式与模型输出。
