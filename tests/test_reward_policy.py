from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from module.reward_policy import (
    DEFAULT_SCORER,
    Threshold,
    assign_threshold,
    normalize_thresholds,
    parse_reward_policy,
    save_threshold_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def test_missing_reward_config_uses_default_without_thresholds():
    policy = parse_reward_policy({})

    assert policy.default_scorer == "aesthetic_predictor_v2_5"
    assert policy.thresholds_for("aesthetic_predictor_v2_5") == ()
    assert policy.scorer_names == ()


def test_profiles_are_isolated_and_sorted():
    policy = parse_reward_policy(
        {
            "reward_model": {
                "default_scorer": "alpha",
                "scorers": {
                    "alpha": {
                        "thresholds": [
                            {
                                "name": "high",
                                "max_score": 8.0,
                                "color": "bold green",
                            },
                            {
                                "name": "low",
                                "max_score": 3.0,
                                "color": "bold red",
                            },
                        ]
                    },
                    "beta": {
                        "thresholds": [{"name": "only", "max_score": 1.0}]
                    },
                },
            }
        }
    )

    assert policy.default_scorer == "alpha"
    assert policy.scorer_names == ("alpha", "beta")
    assert [row.name for row in policy.thresholds_for("alpha")] == ["low", "high"]
    assert [row.name for row in policy.thresholds_for("beta")] == ["only"]
    assert policy.thresholds_for("missing") == ()
    assert policy.thresholds_for("beta")[0].color == "white"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("not-a-list", "must be a list"),
        ([{"name": "../escape", "max_score": 1.0}], "invalid threshold name"),
        (
            [
                {"name": "same", "max_score": 1.0},
                {"name": "same", "max_score": 2.0},
            ],
            "duplicate threshold name",
        ),
        (
            [
                {"name": "one", "max_score": 1.0},
                {"name": "two", "max_score": 1.0},
            ],
            "duplicate max_score",
        ),
        ([{"name": "nan", "max_score": float("nan")}], "must be finite"),
        ([{"name": "blank", "max_score": 1.0, "color": "  "}], "color"),
    ],
)
def test_invalid_threshold_rows_fail_before_runtime(rows, message):
    with pytest.raises(ValueError, match=message):
        normalize_thresholds(rows, scorer="alpha")


def test_threshold_assignment_uses_upper_bound_and_final_catch_all():
    rows = (
        Threshold("low", 3.0, "red"),
        Threshold("mid", 7.0, "blue"),
        Threshold("high", 10.0, "green"),
    )

    assert assign_threshold(3.0, rows).name == "low"
    assert assign_threshold(3.1, rows).name == "mid"
    assert assign_threshold(7.0, rows).name == "mid"
    assert assign_threshold(99.0, rows).name == "high"
    assert assign_threshold(1.0, ()) is None


def test_threshold_folder_name_is_safe_visible_name():
    assert Threshold("low_quality", 1.0, "red").folder_name == "low quality"


def test_save_profile_preserves_unrelated_toml(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text("# keep this comment\n[other]\nvalue = 7\n", encoding="utf-8")

    saved = save_threshold_profile(
        path,
        "alpha",
        [{"name": "low", "max_score": 2.5, "color": "bold red"}],
    )

    text = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert "# keep this comment" in text
    assert parsed["other"]["value"] == 7
    assert parsed["reward_model"]["scorers"]["alpha"]["thresholds"] == [
        {"name": "low", "max_score": 2.5, "color": "bold red"}
    ]
    assert saved == (Threshold("low", 2.5, "bold red"),)


def test_saving_empty_profile_removes_only_selected_thresholds(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text(
        """
[reward_model]
default_scorer = "alpha"

[[reward_model.scorers.alpha.thresholds]]
name = "old"
max_score = 1.0

[[reward_model.scorers.beta.thresholds]]
name = "keep"
max_score = 2.0
""".lstrip(),
        encoding="utf-8",
    )

    assert save_threshold_profile(path, "alpha", []) == ()

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert "thresholds" not in parsed["reward_model"]["scorers"].get("alpha", {})
    assert parsed["reward_model"]["scorers"]["beta"]["thresholds"][0]["name"] == "keep"


def test_checked_in_reward_policy_has_one_active_owner_and_no_thresholds():
    model = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))
    general = tomllib.loads((ROOT / "config" / "general.toml").read_text(encoding="utf-8"))
    legacy = tomllib.loads((ROOT / "config" / "config.toml").read_text(encoding="utf-8"))

    assert DEFAULT_SCORER == "aesthetic_predictor_v2_5"
    assert model["reward_model"] == {"default_scorer": DEFAULT_SCORER}
    assert "reward_model" not in general
    assert legacy["reward_model"] == {"default_scorer": DEFAULT_SCORER}
