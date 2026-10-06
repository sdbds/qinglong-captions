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

## Index-Translate

GUI 的翻译模型列表和 `5.translate.ps1` 的 `$model_id` 均可选择：

- `IndexTeam/Index-Translate-9B`
- `IndexTeam/Index-Translate-2B`

默认模型仍为 Hy-MT2-7B。两款 Index 模型共用适配器，使用官方中文约束提示格式，关闭思考模式，保留 Markdown、占位符、术语表和分块上下文。`source_lang=auto` 时不在提示中指定源语言；`zh_cn` 和 `zh_tw` 分别指定简体和繁体中文。

本地推理使用 Qwen3.5 加载类，依赖现有 `translate` profile 中的 Transformers 5.6+，不需要安装 vLLM。首次运行会下载所选模型权重：

```powershell
python -m module.texttranslate ./datasets --model_id IndexTeam/Index-Translate-2B --target_lang zh_cn
```

连接已启动的 OpenAI-compatible 服务：

```powershell
python -m module.texttranslate ./datasets --model_id IndexTeam/Index-Translate-9B --runtime_backend openai --openai_base_url http://127.0.0.1:8000/v1 --openai_model_name served-index --target_lang zh_cn
```

`--model_id` 保留上述官方 ID，用于选择 Index 适配器；服务端自定义模型别名放在 `--openai_model_name` 中，未设置时默认使用 `--model_id`。API 请求会发送 `chat_template_kwargs.enable_thinking=false`，服务端需支持该参数。

默认使用贪心解码，沿用项目每块 `max_new_tokens=4096` 的预算。若返回未结束的思考块或空译文，会报错而不写入无效翻译；长文本可减小 `--max_chars` 或增大 `--max_new_tokens`，同时留意服务端上下文限制和显存占用。

官方模型与部署说明：[9B](https://huggingface.co/IndexTeam/Index-Translate-9B)、[2B](https://huggingface.co/IndexTeam/Index-Translate-2B)。
