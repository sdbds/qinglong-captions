# 图像打标

Tagger 页面为图片或 Lance 数据集生成内容标签，主要支持 WDTagger 与 CL Tagger。标签可作为筛选条件，也可进入后续 caption prompt。

## 入口

```powershell
.\3.tagger.ps1
```

兼容 Python 入口：`python utils/wdtagger.py --help`。

## 模型与依赖

- WDTagger 使用 `wdtagger` profile。
- CL Tagger v2 使用 `wdtagger-cl-tagger-v2` profile。
- Gated 模型需要先在 Hugging Face 接受条款，并通过环境变量提供 `HF_TOKEN`。
- `repo_id` 选择模型；`model_dir` 指定本地缓存目录。

## 关键参数

- `batch_size`：推理批大小，显存不足时先减小。
- `thresh`：概念标签总阈值。
- `general_threshold` / `character_threshold`：通用与角色标签阈值。
- `overwrite`：是否覆盖已有 sidecar 或 Lance caption。

阈值越低，召回越高，但噪声标签也会增加。建议先在几十张代表图片上比较结果，再决定全量阈值。
