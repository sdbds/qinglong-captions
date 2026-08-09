# Image Quality Scoring

Image scoring uses Qinglong Score to produce model-preference scores for ranking, sampling, and assisted dataset cleaning. Scores are not an absolute quality scale shared across scorers or data domains.

## Run

The dependency profile is `reward-model`. Use Tools / Image Scoring in the GUI, or choose one command:

```powershell
# Recommended: PowerShell wrapper
.\2.3.image_reward_model.ps1

# Or run the Python script with its PEP 723 dependencies
$env:PYTHONPATH = (Get-Location).Path
uv run --no-project .\module\rewardmodel.py .\datasets `
  --scorer aesthetic_predictor_v2_5 `
  --batch_size 4 `
  --device auto `
  --dtype auto
```

`--scorer` selects the scoring algorithm. `--checkpoint` selects registered weights for that scorer; omitting it uses the registry default. The default scorer is the image-only `aesthetic_predictor_v2_5`.

Prompt-required scorers use the first available source for each image: a nonempty `--prompt`, that image's nonempty Lance caption, or an empty string. The empty string is an intentional no-text scoring mode and is reported as `prompt_source="empty"`. Image-only scorers receive `prompts=None`.

## Thresholds

Partitioning is disabled by default, with no quality-directory creation, cleanup, links, or copies. Enable it explicitly for one scorer in `config/model.toml`:

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

Rows are applied by ascending `max_score`. A score enters the first row where `score <= max_score`; scores above every bound enter the final row. The GUI can add, validate, color, and save a separate profile for each scorer. Output folders preserve source-relative paths and use symlinks first, with a visible copy fallback when the platform disallows them.

## Results

Directory input writes `<directory>/reward_scores.json`. Direct `.lance` input writes a sibling `<name>.reward_scores.json`. The structured report ranks by descending score then ascending path, and records the Qinglong Score version, resolved checkpoint, device, dtypes, prompt provenance, buckets, failures, and summary. It is replaced atomically. Partial image failures return success when at least one image scores; an all-failed run returns nonzero after writing diagnostics.

A tracking checkpoint records both its mutable source and the pinned revision resolved for that run. Tracking is convenient for following new weights, but scorer-level thresholds can drift when those weights change; prefer a pinned checkpoint for repeatable production runs.
