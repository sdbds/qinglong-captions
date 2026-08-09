# 图像质量评分

图像评分工具通过 Qinglong Score 为图片生成模型偏好分数，适合排序、抽样和辅助清洗。分数不是跨模型、跨数据域通用的绝对质量标准。

## 运行

依赖 profile 为 `reward-model`。可使用 GUI 的“工具 / 图像评分”，也可运行：

```powershell
.\2.3.image_reward_model.ps1
uv run --extra reward-model python -m module.rewardmodel .\datasets `
  --scorer aesthetic_predictor_v2_5 `
  --batch_size 4 `
  --device auto `
  --dtype auto
```

`--scorer` 选择评分算法，`--checkpoint` 选择该评分器注册的权重。省略 `--checkpoint` 时使用注册表默认权重。默认评分器是仅使用图片的 `aesthetic_predictor_v2_5`。

需要文本的评分器按以下顺序构造每张图片的 prompt：非空 `--prompt`、该图片在 Lance 中的非空 caption、空字符串。空字符串是明确支持的无文本评分模式，并在报告中记录为 `prompt_source="empty"`；仅图片评分器收到 `prompts=None`。

## 分档

默认不分档，也不会创建、清理或改动任何质量目录。只有在 `config/model.toml` 为当前评分器显式配置阈值后才启用：

```toml
[[reward_model.scorers.aesthetic_predictor_v2_5.thresholds]]
name = "low_quality"
max_score = 4.5
color = "bold red"

[[reward_model.scorers.aesthetic_predictor_v2_5.thresholds]]
name = "best_quality"
max_score = 10.0
color = "bold green"
```

阈值按 `max_score` 升序应用，分数进入第一个满足 `score <= max_score` 的档位；超过全部上限时进入最后一档。GUI 可以为每个评分器添加、排序验证、着色并保存这些阈值。目录保留源文件的相对路径，优先创建符号链接，平台不支持时复制文件。

## 结果

目录输入写入 `<目录>/reward_scores.json`；直接输入 `.lance` 时写入同级 `<名称>.reward_scores.json`。报告按分数降序和路径升序稳定排名，并记录 Qinglong Score 版本、实际解析后的 checkpoint、设备、dtype、prompt 来源、分档、失败项和汇总。文件采用原子替换；部分图片失败但至少一张成功时仍返回成功并保留诊断，全部失败则返回非零状态。

跟踪更新的 checkpoint 会同时记录可变来源和本次解析到的固定 revision。它便于追踪最新权重，但同一组 scorer 阈值可能随权重更新发生漂移，正式数据处理更适合选择固定 checkpoint。
