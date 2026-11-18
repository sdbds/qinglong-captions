# DeepSeek OCR 2：本地后端和回归

标准的 `--ocr_model deepseek_ocr` 不需要改模型配置。对于官方 V2 和默认的 1024/768 动态裁块参数，它会使用 Transformers 5.x [内置实现](https://huggingface.co/docs/transformers/v5.23.0/en/model_doc/deepseek_ocr2)，把原来的官方模型名映射到固定提交的原生格式。项目安装项固定为此次 GPU 验收的 `5.23.0`，不会自动换到未经这套样本验证的新版本。使用 safetensors，不执行模型仓库的 Python 代码；沿用 `deepseek_ocr_prompt` 和原有的远端服务设置。原生推理报错不会偷偷改用旧实现。

| 场景 | OCR 路由 | 安装项 |
| --- | --- | --- |
| 标准 DeepSeek OCR 2，或本地 `model_type=deepseek_ocr2` 的原生检查点 | `deepseek_ocr` | `deepseek-ocr` |
| 显式只用原生实现 | `deepseek_ocr_native` | `deepseek-ocr-native` |
| 老模型、原版自定义检查点、旧裁块尺寸，或显式回滚 | `deepseek_ocr` 根据配置选择旧版，或 `deepseek_ocr_legacy` | `deepseek-ocr-legacy`，独立环境 |

GUI 和 PowerShell 默认把两个本地栈分别安装在 `.runtime_venvs/deepseek-ocr-native` 和 `.runtime_venvs/deepseek-ocr-legacy`，不轮流覆盖项目的共享环境。GUI 中显式指定的独立解释器或虚拟环境仍会被尊重。首次创建需要 uv 和可用的 Python 3.11；全离线时还需预先缓存依赖和 pip、setuptools、wheel。手动运行时也请为新旧版分别指定解释器。旧版会运行模型代码，仅使用可信模型和输入。原生裁图使用无损 PNG；它与旧版 JPEG、Markdown 图片说明、空白并非字节级兼容。

原生每页产出 `result.md`、未清理的 `raw.txt`、包含原始与像素坐标的 `grounding.json`，以及 Markdown 实际引用的裁图。定位坐标范围为 0 到 999，横纵方向分别按该页的实际宽、高换算。PDF 按 `page_0001` 等目录存放，同时生成汇总 Markdown。正式任务每次保留独立产物目录，旧字幕的链接在重跑后仍然有效。

## 重做真实对照

1. 准备三个独立的 CUDA Python 环境：旧后端使用 Transformers 4.46.3、tokenizers 0.20.3 和本项目对应的 FlashAttention；两套原生环境分别固定上一次验收的 5.17.0 和候选版 5.23.0，其他包保持一致。三套环境安装 PyMuPDF，使用同一块 GPU 和同一份提示词，先后运行。不要在 5.17.0 参考环境安装当前版本的项目 extra，否则它会被升到候选版。
2. 将[旧仓库](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/tree/aaa02f3811945a91062062994c5c4a3f4c0af2b0)和[原生仓库](https://huggingface.co/deepseek-community/DeepSeek-OCR-2/tree/838dc37567aa97106fcd590e2be6a8f279881813)的固定版本分别下载到新的本地目录。两仓库的权重 SHA-256 相同，运行器会校验权重和各自的代码、配置、分词器。全部推理离线执行。
3. 从仓库根目录执行下面的 PowerShell 示例。使用空的工作目录；`$OldPython`、`$ReferencePython`、`$NewPython` 分别指向 4.46.3、5.17.0、5.23.0，`$OldModel` 和 `$NewModel` 是本地快照。两个原生版本使用同一个 `$NewModel`。样本由程序生成，不读取已有素材。缺少中文字体时给 `prepare` 指定 `--font`。

```powershell
& $NewPython tools/deepseek_ocr2_compare.py prepare --out "$Work/fixture"
if ($LASTEXITCODE -ne 0) { throw "Fixture preparation failed" }
foreach ($i in 1,2) {
    & $OldPython tools/deepseek_ocr2_run.py --backend legacy --fixture "$Work/fixture" --model-dir $OldModel --out "$Work/old-$i"
    if ($LASTEXITCODE -ne 0) { throw "Legacy inference failed" }
    & $ReferencePython tools/deepseek_ocr2_run.py --backend native --fixture "$Work/fixture" --model-dir $NewModel --out "$Work/ref-$i"
    if ($LASTEXITCODE -ne 0) { throw "Reference inference failed" }
    & $NewPython tools/deepseek_ocr2_run.py --backend native --fixture "$Work/fixture" --model-dir $NewModel --out "$Work/new-$i"
    if ($LASTEXITCODE -ne 0) { throw "Native inference failed" }
}
& $NewPython tools/deepseek_ocr2_compare.py compare --fixture "$Work/fixture" --legacy "$Work/old-1" --legacy "$Work/old-2" --reference-native "$Work/ref-1" --reference-native "$Work/ref-2" --reference-native-version 5.17.0 --native "$Work/new-1" --native "$Work/new-2" --native-version 5.23.0 --out "$Work/report.json"
if ($LASTEXITCODE -ne 0) { throw "Do not promote the candidate: inspect report.json" }
```

2026-11-18 在 RTX 4090、PyTorch 2.13.0+cu130 上复测：两张不同的中文图及由它们组成的双页 PDF，三个版本各跑两次，共 24 次页级观测。全部页序正确，归一化字符错误率为 0。各版本还分别与已知文字墨迹、插图位置和原图像素比对，并非只要求两版结果相同。

| 比较对象 | 最低文字框 IoU | 最低插图框 IoU | 最低裁图像素相似度 |
| --- | ---: | ---: | ---: |
| 5.23.0 对旧后端 4.46.3 | 0.9672 | 0.9952 | 0.9944 |
| 5.23.0 对原生参考 5.17.0 | 1.0000 | 1.0000 | 1.0000 |
| 同版本重复运行 | 1.0000 | 1.0000 | 1.0000 |

两个原生版本每次产生的 19 个文档、坐标与图像文件逐字节一致（不计记录软件版本和耗时的 `run.json`）。这份成绩只覆盖此次中文单栏图文样本，不代表复杂表格、多栏排版或手写稿已经验证。
