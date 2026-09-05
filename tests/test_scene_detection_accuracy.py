import hashlib
import json
from pathlib import Path

import pytest

from tests.scene_accuracy import score_cut_frames

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scene_detection_yugioh.json"


def test_score_cut_frames_matches_each_detection_at_most_once():
    score = score_cut_frames(
        reference_frames=[100, 200, 300],
        detected_frames=[99, 100, 201, 450],
        tolerance_frames=1,
    )

    assert score == {
        "true_positives": 2,
        "false_positives": 2,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(4 / 7),
    }


def test_score_cut_frames_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="tolerance_frames"):
        score_cut_frames([], [], tolerance_frames=-1)


def test_score_cut_frames_handles_empty_inputs():
    assert score_cut_frames([], [], tolerance_frames=1) == {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_yugioh_fixture_freezes_manual_truth_and_profile_metrics():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["schema_version"] == 1
    assert fixture["source"]["start_seconds"] == 132.0
    assert fixture["source"]["end_seconds"] == 222.0
    assert fixture["source"]["fps"] == "24000/1001"
    assert fixture["tolerance_frames"] == 1
    assert fixture["reference_cut_frames"] == sorted(
        set(fixture["reference_cut_frames"])
    )

    for profile in fixture["profiles"].values():
        assert score_cut_frames(
            fixture["reference_cut_frames"],
            profile["detected_cut_frames"],
            fixture["tolerance_frames"],
        ) == pytest.approx(profile["metrics"])


def _find_annotated_video(fixture):
    matches = list((Path(__file__).parent.parent / "datasets" / "video").glob(
        fixture["source"]["filename_glob"]
    ))
    if not matches:
        pytest.skip("annotated Yu-Gi-Oh video is not available locally")

    video_path = matches[0]
    if video_path.stat().st_size != fixture["source"]["size_bytes"]:
        pytest.skip("local Yu-Gi-Oh video does not match the annotated fixture")

    digest = hashlib.sha256()
    with video_path.open("rb") as video_file:
        for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != fixture["source"]["sha256"]:
        pytest.skip("local Yu-Gi-Oh video does not match the annotated fixture")
    return video_path


@pytest.mark.integration
@pytest.mark.parametrize(
    "profile_name",
    ["adaptive_project_recommended", "adaptive_legacy_parameters"],
)
def test_yugioh_scene_detection_matches_frozen_f1_at_1(profile_name):
    from scenedetect import detect
    from scenedetect.detectors import AdaptiveDetector

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    video_path = _find_annotated_video(fixture)
    profile = fixture["profiles"][profile_name]
    settings = profile["settings"]
    min_scene_len = settings.get(
        "min_scene_len_seconds", settings.get("min_scene_len_frames")
    )
    scenes = detect(
        video_path,
        AdaptiveDetector(
            adaptive_threshold=settings["adaptive_threshold"],
            min_scene_len=min_scene_len,
            window_width=settings["window_width"],
        ),
        start_time=fixture["source"]["start_seconds"],
        end_time=fixture["source"]["end_seconds"],
        start_in_scene=True,
        backend=settings["backend"],
        show_progress=False,
    )
    detected_cut_frames = [start.frame_num for start, _end in scenes[1:]]

    assert detected_cut_frames == profile["detected_cut_frames"]
    assert score_cut_frames(
        fixture["reference_cut_frames"],
        detected_cut_frames,
        fixture["tolerance_frames"],
    ) == pytest.approx(profile["metrics"])
