# 视频分割

Split 页面用场景检测把视频切成镜头区间，并按需为每个场景提取代表帧。它通常是视频打标和字幕生成前的第一步。

## 入口

```powershell
.\2.0.video_spliter.ps1
```

Python 入口：`python -m module.videospilter --help`。依赖 profile 为 `video-split`。

## 输入与输出

- 输入：单个视频目录，可递归扫描子目录。
- 输出：场景切分结果、代表帧，以及可选 HTML 报告。
- `output_dir` 为空时使用输入目录下的默认输出位置。
- `video2images_min_number=0` 时只计算场景，不保存代表帧。

## 检测器

支持 `ContentDetector`、`AdaptiveDetector`、`HashDetector`、`HistogramDetector` 和 `ThresholdDetector`。默认先从 Content/Adaptive 开始：切分过密时提高 `threshold` 或 `min_scene_len`，漏切时降低阈值。

长视频先用少量样本确定阈值，再批量运行。快速闪烁、MV、演唱会灯光和动画特效容易造成误切，应配合更长的 `min_scene_len`。
