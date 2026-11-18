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

独立导出默认把媒体和字幕复制到 `./exports` 或选定的输出目录，不改源文件旁的字幕。可以用 `--caption_dir` 为字幕单独选目录。库保存了媒体二进制时使用该版本的内容，否则从原文件复制。目标内有无法确认归属的同名文件时，请选空目录；不要为了让导出通过而删除自己的文件。只有明确使用 `--in_place` 时才会回写源文件旁的字幕。

分页 Markdown 在独立导出时生成。原地发布只写主文档，避免占用素材旁边的同名目录；音视频仍可按字幕裁剪。程序不会覆盖归属不明的裁剪目录。可验证且未变化的派生文件可在重跑时复用。

同一个 Lance 数据集不要被多个写任务同时修改。批量处理前保留备份，并在导出前确认 version/tag。
