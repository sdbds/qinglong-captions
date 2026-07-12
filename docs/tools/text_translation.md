# 文本与文档翻译

翻译工具把文本或文档规范化为 Markdown，记录分块边界，再使用本地模型或 OpenAI-compatible backend 翻译。结果默认写为语言后缀文件，不覆盖原文。

## 入口

```powershell
.\5.translate.ps1
```

Python 入口：`python -m module.texttranslate --help`。依赖 profile 为 `translate`。

## 处理流程

1. 导入原始文本或文档。
2. 规范化为 Markdown，并保存 `chunk_offsets`。
3. 按目标语言执行翻译。
4. 写入 Lance 版本并导出 `*_lang.md`。

常用参数包括 `normalize_only`、`skip_normalize`、`no_export`、目标语言、分块大小和 `runtime_backend openai`。首次排查时先运行 `--normalize_only`，确认文档解析和输出目录正常。
