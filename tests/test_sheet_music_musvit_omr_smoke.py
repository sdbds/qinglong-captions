from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

DATASET_REVISION = "b3170c8b8f322885b566efe9e264af9328b5603f"
VALIDATION_IMAGE_SHA256 = (
    "53189d713563d87da723544abf0a8585ff2b32f3b0bf642e03f2185c3fe10fa5"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MUSVIT_OMR_SMOKE") != "1",
    reason="set RUN_MUSVIT_OMR_SMOKE=1 to download and run the pinned model",
)


def _download_pinned_validation_page(target: Path) -> None:
    query = urllib.parse.urlencode(
        {
            "dataset": "PRAIG/polish-scores",
            "config": "default",
            "split": "val",
            "offset": 0,
            "length": 1,
            "revision": DATASET_REVISION,
        }
    )
    with urllib.request.urlopen(
        f"https://datasets-server.huggingface.co/rows?{query}",
        timeout=60,
    ) as response:
        row_payload = json.load(response)
    image_url = row_payload["rows"][0]["row"]["image"]["src"]
    assert f"/{DATASET_REVISION}/" in urllib.parse.urlparse(image_url).path
    with urllib.request.urlopen(image_url, timeout=60) as response:
        image_bytes = response.read()
    assert hashlib.sha256(image_bytes).hexdigest() == VALIDATION_IMAGE_SHA256
    target.write_bytes(image_bytes)


def test_pinned_model_transcribes_known_page_to_both_formats(tmp_path: Path):
    from music21 import converter

    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        normalize_page_score,
    )
    from module.sheet_music_omr.decode import parse_kern_score
    from module.sheet_music_omr.model import (
        DEFAULT_MODEL_REPO_ID,
        DEFAULT_MODEL_REVISION,
        MuSViTOnnxRecognizer,
    )
    from module.sheet_music_omr.pipeline import MuSViTOmrPipeline

    image_path = tmp_path / "polish-scores-val-0.jpg"
    _download_pinned_validation_page(image_path)
    recognizer = MuSViTOnnxRecognizer()

    result = MuSViTOmrPipeline(
        recognizer=recognizer,
        output_dir=tmp_path / "output",
        output_format="both",
    ).run(image_path)

    page_dir = tmp_path / "output" / image_path.name
    metadata = json.loads(
        (page_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert result.ok is True
    assert metadata["model_repo_id"] == DEFAULT_MODEL_REPO_ID
    assert metadata["model_revision"] == DEFAULT_MODEL_REVISION
    assert metadata["terminated_by_eos"] is True
    assert metadata["truncated"] is False
    assert metadata["kern_status"] == "ok"
    assert metadata["providers"]
    assert (page_dir / "score.musicxml").is_file()
    assert (page_dir / "score.mid").is_file()

    parsed_kern = parse_kern_score(
        (page_dir / "score.krn").read_text(encoding="utf-8")
    )
    normalized_kern = normalize_page_score(
        KernPageScore(
            score=parsed_kern,
            kern_text=(page_dir / "score.krn").read_text(encoding="utf-8"),
        )
    ).score
    parsed_musicxml = converter.parse(page_dir / "score.musicxml")

    def note_sequence(score):
        return tuple(
            tuple(
                (
                    tuple(
                        pitch.midi
                        for pitch in (
                            event.pitches
                            if hasattr(event, "pitches")
                            else (event.pitch,)
                        )
                    ),
                    event.quarterLength,
                    event.duration.isGrace,
                )
                for event in part.recurse().notes
            )
            for part in score.parts
        )

    assert note_sequence(parsed_musicxml) == note_sequence(normalized_kern)
    assert tuple(
        Counter(part_notes)
        for part_notes in note_sequence(parsed_musicxml)
    ) == tuple(
        Counter(part_notes)
        for part_notes in note_sequence(parsed_kern)
    )
