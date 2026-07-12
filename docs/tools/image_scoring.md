# 图像质量评分

图像评分工具使用 reward model 为图片生成质量分数，适合排序、抽样和清洗数据集。分数只反映所选模型的偏好，不等于绝对质量标准。

## 入口

```powershell
.\2.3.image_reward_model.ps1
```

Python 入口：`python -m module.rewardmodel --help`。依赖 profile 为 `reward-model`。

## 使用建议

- 输入图片目录或 GUI 选择的目标数据集。
- 根据显存调整 `batch_size` 与 device。
- 保留原始分数，再按项目需求设置筛选阈值。
- 对人物、插画、摄影等不同数据域分别抽样校验。

不要用单一阈值直接删除唯一数据。更稳妥的做法是先排序和人工抽检，再决定保留范围。
