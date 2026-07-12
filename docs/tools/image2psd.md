# Image2PSD / See-through

Image2PSD 把单张角色图或图片目录转换为可继续编辑的分层 PSD。主流程为 LayerDiff 透明分层、Marigold 深度估计和 PSD 导出。

## 入口

```powershell
.\2.6.image2psd.ps1
```

Python 入口：`python -m module.see_through.cli --help`。依赖 profile 为 `see-through`。

## 使用提示

- 输入为图片目录，输出包括分层结果、深度信息、中间文件和 PSD。
- `seed` 控制可复现性。
- 深度与分层推理步数影响速度和细节。
- 模型下载和中间结果会占用较多磁盘；低显存环境应启用 CPU/offload 策略。

该工具更适合主体清晰、遮挡关系明确的角色插画。复杂背景、透明特效和细碎装饰可能需要手工修层。
