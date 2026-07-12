# PSD 图层导出

PSD Export 把 PSD 文件夹中的图层导出为 PNG 数据集，并可选构建或回导 Lance。它是脚本专用流程，当前没有接入 GUI Tools 页面。

## 入口

```powershell
.\2.4.psdexport.ps1
```

Python 入口：`python -m utils.psd_dataset_pipeline --help`。依赖 profile 为 `psdexport`。

## 主要能力

- 批量读取 PSD 并导出可见图层。
- 控制直接导出的最大图层数。
- 可选合并线稿、固定七层布局和包含隐藏图层。
- 可选缩放导出，并构建或回导 Lance 数据集。

复杂混合模式、蒙版和智能对象的合成结果可能与 Photoshop 不完全一致。批量处理前先检查几个代表 PSD。
