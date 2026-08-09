from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_reward_extra_uses_qinglong_score_without_legacy_direct_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["optional-dependencies"]["reward-model"]
    normalized = [dependency.lower() for dependency in dependencies]

    assert "qinglong-captions[torch-base]" in normalized
    assert "qinglong-score>=0.2.2" in normalized
    assert not any(dependency.startswith("imscore") for dependency in normalized)
    assert not any(dependency.startswith("open-clip-torch") for dependency in normalized)
    assert not any(dependency.startswith("scipy") for dependency in normalized)


def test_inline_script_dependencies_match_reward_runtime_contract():
    source = (ROOT / "module" / "rewardmodel.py").read_text(encoding="utf-8")
    match = re.search(r"# /// script\n(?P<metadata>.*?)# ///", source, re.DOTALL)
    assert match is not None
    metadata = match.group("metadata").lower()

    assert '"torch==2.13.0"' in metadata
    assert '"qinglong-score>=0.2.2"' in metadata
    assert '"tomlkit"' in metadata
    assert "imscore" not in metadata
    assert "open-clip-torch" not in metadata
    assert '"scipy' not in metadata


def test_powershell_wrapper_uses_scorer_checkpoint_and_reward_extra():
    wrapper = (ROOT / "2.3.image_reward_model.ps1").read_text(encoding="utf-8")

    assert 'scorer             = "aesthetic_predictor_v2_5"' in wrapper
    assert 'checkpoint         = ""' in wrapper
    assert '--scorer=$($Config.scorer)' in wrapper
    assert 'if ($Config.checkpoint)' in wrapper
    assert '--checkpoint=$($Config.checkpoint)' in wrapper
    assert "uv dependency profile: reward-model" in wrapper
    assert 'uv run --no-project "./module/rewardmodel.py"' in wrapper
    assert "uv run --extra reward-model" not in wrapper
    assert "--repo_id" not in wrapper
