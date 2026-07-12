# 数据导入与导出

Import 和 Export 把普通媒体目录与 Lance 数据集连接起来。推荐先导入，再执行分割、打标或字幕任务，最后按目标版本导出媒体和字幕。

## 入口

```powershell
.\lanceImport.ps1
.\lanceExport.ps1
```

Python 入口：

```powershell
python -m module.lanceImport --help
python -m module.lanceexport --help
```

## 导入

- 输入可以是图片、视频、音频、文本目录或已有数据目录。
- `caption_dir` 可指定独立 sidecar 字幕目录。
- `import_mode` 用于限制导入媒体类型。
- `tag` 标记本次导入版本；`data_storage_version` 控制新建 Lance 格式。
- 默认配置不保存原始二进制 blob，运行前应确认脚本或 GUI 的保存选项。

## 导出

- `lance_file` 指向目标 `.lance` 数据集。
- `version` 选择需要读取的 dataset tag。
- `caption_suffix` 和 `caption_extension` 控制字幕文件名和格式。
- `allowed_caption_types` 可限制导出的 caption media type。

同一个 Lance 数据集不要被多个写任务同时修改。批量处理前保留备份，并在导出前确认 version/tag。
