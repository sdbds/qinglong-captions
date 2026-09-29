# 图像打标

Tagger 页面为图片或 Lance 数据集生成内容标签，支持 WDTagger、CL Tagger 与 PixAI Tagger。标签可作为筛选条件，也可进入后续 caption prompt。

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

## PixAI Tagger v1.0 ONNX

直接使用已转换的 [bdsqlsz/pixai-tagger-v1.0-ONNX](https://huggingface.co/bdsqlsz/pixai-tagger-v1.0-ONNX)，不需要先运行转换脚本。GUI 选择该仓库，或执行下面的命令。

```powershell
python utils/wdtagger.py ./datasets --repo_id=bdsqlsz/pixai-tagger-v1.0-ONNX --batch_size=1
```

首次下载 `model.onnx` 和官方格式的 `config.json` 到 `wd14_tagger_model/bdsqlsz_pixai-tagger-v1.0-ONNX/`。标签顺序、分类和输入尺寸直接读取 `config.json`，不需要 `model.json`。不计算或校验模型文件哈希，也不固定远端提交版本。

默认复用本地缓存。仓库更新后，添加 `--force_download` 重新下载模型与配置，并刷新 ONNX 会话。旧地址 `pixai-labs/pixai-tagger-v1.0` 仍兼容，保留原缓存目录，缺少文件或强制刷新时改从新的 ONNX 仓库下载。

GUI 与 PowerShell 启动器自动选择 `wdtagger-pixai` 依赖组，日常 ONNX 推理不加载 Transformers 或远程模型代码。默认批大小为 1，默认阈值读取下载的 `config.json.category_best_threshold`，随配置刷新。当前版本为 general 0.17、character 0.27、style 0.15、copyright 0.24、meta 0.17、rating 0.41。GUI 默认启用“使用模型默认阈值”，关闭后可手动设置；命令行通过对应的 `--类别_threshold` 调整。显式 `--thresh` 覆盖所有类别默认值，单类参数优先，支持零阈值。分级标签仍由 `--use_rating_tags` 控制。

模型输入为 `[batch, 3, 1008, 1008]` RGB 浮点张量，输出为 `[batch, 30877]` logits，由打标器执行一次 sigmoid。预处理沿用官方张量缩放、黑色补边、归一化和透明区域白底合成。运行时 `auto` 使用 CUDA/CPU，默认关闭 TF32；显式 provider 配置仍然生效，TensorRT/FP16 需另行验证。

需要自行重新转换时，仍可使用下面的命令。默认下载上游 `main`，也可通过 `--revision` 自选版本；只导出 ONNX，不再生成 `model.json` 或哈希清单。

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[pixai-onnx-export]"
python -m utils.onnx_export pixai-tagger --download --device cuda --verify
```

`--model-dir` 指定源模型目录，`--output-path` 指定 ONNX 导出路径，`--overwrite` 允许替换已有导出。导出与验证成功后才替换旧文件，失败或取消时清理本次临时产物。迁移 ONNX 时一并保留官方 `config.json`。代理环境下 Xet 下载停滞时，可设置 `$env:HF_HUB_DISABLE_XET="1"` 后重试。
