from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pytest
import torch

from module.reward_policy import Threshold


@pytest.fixture()
def rewardmodel():
    sys.modules.pop("module.rewardmodel", None)
    import module.rewardmodel as rewardmodel_module

    return rewardmodel_module


@dataclass(frozen=True, slots=True)
class FakeArtifact:
    provider: str
    repository: str
    revision: str
    filename: str | None
    sha256: str | None
    role: str


@dataclass(frozen=True, slots=True)
class FakeCheckpoint:
    adapter: str
    identifier: str
    format: str
    artifacts: tuple[FakeArtifact, ...]
    is_default: bool
    tracks_updates: bool = False


ARTIFACT = FakeArtifact(
    provider="huggingface",
    repository="owner/model",
    revision="a" * 40,
    filename="model.safetensors",
    sha256="b" * 64,
    role="weights",
)
RESOLVED_CHECKPOINT = FakeCheckpoint(
    adapter="aesthetic_predictor_v2_5",
    identifier="owner/model",
    format="official",
    artifacts=(ARTIFACT,),
    is_default=True,
)


def test_report_is_deterministic_and_records_runtime_provenance(
    rewardmodel, tmp_path: Path
):
    source_root = tmp_path / "images"
    source_root.mkdir()
    tracking_source = FakeCheckpoint(
        adapter="aesthetic_predictor_v2_5",
        identifier="owner/tracking",
        format="official",
        artifacts=(
            FakeArtifact(
                provider="huggingface",
                repository="owner/model",
                revision="main",
                filename=None,
                sha256=None,
                role="snapshot",
            ),
        ),
        is_default=False,
        tracks_updates=True,
    )
    items = [
        rewardmodel.ScoredImage(str(source_root / "b.png"), None, None, 2.0),
        rewardmodel.ScoredImage(str(source_root / "a.png"), None, None, 2.0),
        rewardmodel.ScoredImage(str(source_root / "c.png"), "", "empty", 3.0),
    ]
    errors = [
        rewardmodel.RunError.item(
            str(source_root / "broken.png"), ValueError("cannot decode")
        ),
        rewardmodel.RunError.batch(
            [str(source_root / "d.png"), str(source_root / "e.png")],
            RuntimeError("score failed"),
        ),
    ]
    thresholds = (
        Threshold("low_quality", 2.0, "red"),
        Threshold("best_quality", 10.0, "green"),
    )

    report = rewardmodel.build_report(
        qinglong_score_version="0.2.2",
        scorer="aesthetic_predictor_v2_5",
        requested_checkpoint="owner/tracking",
        tracking_source=tracking_source,
        checkpoint=RESOLVED_CHECKPOINT,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        input_dtype=torch.float32,
        attention_backend=None,
        thresholds=thresholds,
        items=items,
        errors=errors,
        source_root=source_root,
        buckets={
            str(source_root / "a.png"): "low_quality",
            str(source_root / "b.png"): "low_quality",
            str(source_root / "c.png"): "best_quality",
        },
    )

    assert "schema_version" not in report
    assert report["run"] == {
        "qinglong_score_version": "0.2.2",
        "scorer": "aesthetic_predictor_v2_5",
        "requested_checkpoint": "owner/tracking",
        "tracking_source": {
            "kind": "remote",
            "adapter": "aesthetic_predictor_v2_5",
            "identifier": "owner/tracking",
            "format": "official",
            "artifacts": [
                {
                    "provider": "huggingface",
                    "repository": "owner/model",
                    "revision": "main",
                    "filename": None,
                    "sha256": None,
                    "role": "snapshot",
                }
            ],
            "is_default": False,
            "tracks_updates": True,
        },
        "checkpoint": {
            "kind": "remote",
            "adapter": "aesthetic_predictor_v2_5",
            "identifier": "owner/model",
            "format": "official",
            "artifacts": [
                {
                    "provider": "huggingface",
                    "repository": "owner/model",
                    "revision": "a" * 40,
                    "filename": "model.safetensors",
                    "sha256": "b" * 64,
                    "role": "weights",
                }
            ],
            "is_default": True,
            "tracks_updates": False,
        },
        "device": "cpu",
        "compute_dtype": "torch.float32",
        "input_dtype": "torch.float32",
        "attention_backend": None,
        "thresholds_enabled": True,
        "thresholds": [
            {"name": "low_quality", "max_score": 2.0, "color": "red"},
            {"name": "best_quality", "max_score": 10.0, "color": "green"},
        ],
    }
    assert report["summary"] == {
        "scored": 3,
        "failed": 3,
        "empty_prompt_count": 1,
    }
    assert [
        (row["rank"], row["path"], row["bucket"]) for row in report["items"]
    ] == [
        (1, "c.png", "best_quality"),
        (2, "a.png", "low_quality"),
        (3, "b.png", "low_quality"),
    ]
    assert report["errors"] == [
        {
            "scope": "item",
            "stage": "decode",
            "error_type": "ValueError",
            "message": "cannot decode",
            "path": "broken.png",
        },
        {
            "scope": "batch",
            "stage": "score",
            "error_type": "RuntimeError",
            "message": "score failed",
            "paths": ["d.png", "e.png"],
        },
    ]


def test_pinned_checkpoint_has_no_tracking_source_and_no_thresholds(
    rewardmodel, tmp_path: Path
):
    report = rewardmodel.build_report(
        qinglong_score_version="0.2.2",
        scorer="aesthetic_predictor_v2_5",
        requested_checkpoint=None,
        tracking_source=None,
        checkpoint=RESOLVED_CHECKPOINT,
        device="cpu",
        compute_dtype=torch.float32,
        input_dtype=torch.float32,
        attention_backend=None,
        thresholds=(),
        items=[],
        errors=[],
        source_root=tmp_path,
        buckets={},
    )

    assert report["run"]["tracking_source"] is None
    assert report["run"]["thresholds_enabled"] is False
    assert report["run"]["thresholds"] == []


def test_result_path_depends_on_directory_or_direct_lance_input(
    rewardmodel, tmp_path: Path
):
    directory = tmp_path / "images"
    directory.mkdir()
    lance_path = tmp_path / "catalog.lance"

    assert rewardmodel.result_path_for_input(directory) == directory / "reward_scores.json"
    assert rewardmodel.result_path_for_input(lance_path) == tmp_path / "catalog.reward_scores.json"


def test_atomic_json_write_preserves_previous_report_on_encoding_failure(
    rewardmodel, tmp_path: Path
):
    path = tmp_path / "reward_scores.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        rewardmodel.write_json_atomic(path, {"score": float("nan")})

    assert path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob("*.tmp")) == []


def test_no_thresholds_have_zero_filesystem_side_effects(
    rewardmodel, tmp_path: Path
):
    source = tmp_path / "nested" / "a.png"
    source.parent.mkdir()
    source.write_bytes(b"image")
    old_quality = tmp_path / "low quality"
    old_quality.mkdir()
    marker = old_quality / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    buckets = rewardmodel.apply_thresholds(
        [rewardmodel.ScoredImage(str(source), None, None, 1.0)],
        thresholds=(),
        source_root=tmp_path,
    )

    assert buckets == {str(source): None}
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["low quality", "nested"]


def test_threshold_partition_preserves_paths_and_logs_copy_fallback_once(
    rewardmodel, monkeypatch, tmp_path: Path
):
    source = tmp_path / "source" / "nested" / "a.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image data")
    messages = []
    monkeypatch.setattr(
        Path,
        "symlink_to",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    buckets = rewardmodel.apply_thresholds(
        [rewardmodel.ScoredImage(str(source), None, None, 1.0)],
        thresholds=(
            Threshold("low_quality", 2.0, "red"),
            Threshold("best_quality", 10.0, "green"),
        ),
        source_root=tmp_path / "source",
        tracking_checkpoint=True,
        warn=messages.append,
    )

    assert buckets == {str(source): "low_quality"}
    copied = tmp_path / "source" / "low quality" / "nested" / "a.png"
    assert copied.read_bytes() == b"image data"
    assert sum("tracking checkpoint" in message for message in messages) == 1
    assert sum("copied" in message for message in messages) == 1


def _install_run_fakes(rewardmodel, monkeypatch, image_paths: list[str]) -> None:
    checkpoint = RESOLVED_CHECKPOINT

    @dataclass(frozen=True, slots=True)
    class FakeSpec:
        name: str = "aesthetic_predictor_v2_5"
        requires_prompts: bool = False
        default_compute_dtype: torch.dtype = torch.float32

    class FakeScorer:
        device = torch.device("cpu")
        input_dtype = torch.float32
        attention_backend = None
        checkpoint_identity = checkpoint

        def score(self, images, prompts=None):
            assert prompts is None
            return torch.ones(images.shape[0], dtype=torch.float32)

    qscore = types.ModuleType("qinglong_score")
    qscore.__version__ = "0.2.2"
    qscore.get_scorer_spec = lambda _name: FakeSpec()
    qscore.list_checkpoints = lambda _name: (checkpoint,)
    qscore.load_scorer = lambda **_kwargs: FakeScorer()
    monkeypatch.setitem(sys.modules, "qinglong_score", qscore)

    class FakeScanner:
        def to_batches(self):
            return [
                pa.record_batch(
                    [
                        pa.array(image_paths),
                        pa.array(["image/png"] * len(image_paths)),
                    ],
                    names=["uris", "mime"],
                )
            ]

    class FakeDataset:
        schema = pa.schema([("uris", pa.string()), ("mime", pa.string())])

        def scanner(self, **_kwargs):
            return FakeScanner()

    monkeypatch.setattr(rewardmodel, "_resolve_dataset", lambda _path: FakeDataset())
    monkeypatch.setattr(
        rewardmodel,
        "load_reward_policy",
        lambda _path: types.SimpleNamespace(
            default_scorer="aesthetic_predictor_v2_5",
            thresholds_for=lambda _scorer: (),
        ),
    )


def test_run_writes_partial_success_report_and_returns_zero(
    rewardmodel, monkeypatch, tmp_path: Path
):
    from PIL import Image

    image_path = tmp_path / "ok.png"
    Image.new("RGB", (3, 3), "white").save(image_path)
    missing_path = tmp_path / "missing.png"
    _install_run_fakes(
        rewardmodel,
        monkeypatch,
        [str(image_path), str(missing_path)],
    )

    args = rewardmodel.setup_parser().parse_args([str(tmp_path), "--device=cpu"])
    assert rewardmodel.run(args) == 0

    report = json.loads((tmp_path / "reward_scores.json").read_text(encoding="utf-8"))
    assert report["summary"] == {
        "scored": 1,
        "failed": 1,
        "empty_prompt_count": 0,
    }
    assert report["items"][0]["path"] == "ok.png"
    assert report["errors"][0]["path"] == "missing.png"


def test_all_failed_direct_lance_run_writes_sibling_report_and_returns_nonzero(
    rewardmodel, monkeypatch, tmp_path: Path
):
    missing_path = tmp_path / "missing.png"
    lance_path = tmp_path / "catalog.lance"
    _install_run_fakes(rewardmodel, monkeypatch, [str(missing_path)])

    args = rewardmodel.setup_parser().parse_args([str(lance_path), "--device=cpu"])
    assert rewardmodel.run(args) == 1

    output = tmp_path / "catalog.reward_scores.json"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["scored"] == 0
    assert report["summary"]["failed"] == 1

