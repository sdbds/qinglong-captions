# Image Quality Scoring

The reward-model tool scores images for sorting, sampling, and dataset cleaning. Scores represent the selected model's preference, not an absolute quality measure.

```powershell
.\2.3.image_reward_model.ps1
python -m module.rewardmodel --help
```

The profile is `reward-model`. Adjust device and batch size for available memory. Keep raw scores and manually inspect different data domains before applying deletion thresholds.
