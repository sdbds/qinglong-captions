# 水印检测

WaterDetect 对图片目录执行水印概率检测，可用于数据清洗或筛选。GUI Tools 页面和 PowerShell wrapper 都提供入口。

## 入口

```powershell
.\2.1.image_watermark_detect.ps1
uv run module/waterdetect.py --help
```

该脚本使用 PEP 723 隔离依赖，不应把它当作普通基础环境模块运行。

## 主要参数

- 输入图片目录与输出位置。
- `repo_id` / `model_dir`：模型仓库与缓存目录。
- `batch_size`：批处理大小。
- `thresh`：水印判定阈值。

首次运行会下载模型。显存不足时降低批大小；误报过多时提高阈值，并用人工抽样确认结果。
