from __future__ import annotations

import hashlib
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


def test_report_is_a_deterministic_path_tree_with_numeric_scores(
    rewardmodel, tmp_path: Path
):
    source_root = tmp_path / "images"
    items = [
        rewardmodel.ScoredImage(str(source_root / "b.png"), None, None, 2.0),
        rewardmodel.ScoredImage(str(source_root / "a.png"), None, None, 2.0),
        rewardmodel.ScoredImage(
            str(source_root / "nested" / "c.png"), "", "empty", 3.25
        ),
    ]

    report = rewardmodel.build_report(
        items=items,
        source_root=source_root,
    )

    assert list(report) == ["a.png", "b.png", "nested"]
    assert report == {
        "a.png": 2.0,
        "b.png": 2.0,
        "nested": {"c.png": 3.25},
    }


def test_report_with_no_successful_items_is_empty(rewardmodel, tmp_path: Path):
    assert rewardmodel.build_report(items=[], source_root=tmp_path) == {}


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


@pytest.fixture()
def copy_partition(monkeypatch, tmp_path):
    source = tmp_path / "nested" / "a.png"
    source.parent.mkdir()
    source.write_bytes(b"original image")
    monkeypatch.setattr(
        Path, "symlink_to",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    return source, (Threshold("low_quality", 2.0, "red"), Threshold("best_quality", 10.0, "green"))


@pytest.mark.parametrize("next_run", ["moved", "skipped", "changed_source"])
def test_threshold_rerun_removes_owned_copies(rewardmodel, copy_partition, tmp_path, next_run):
    source, thresholds = copy_partition
    item = rewardmodel.ScoredImage(str(source), None, None, 1.0)
    rewardmodel.apply_thresholds([item], thresholds=thresholds, source_root=tmp_path)
    previous = tmp_path / "low quality" / "nested" / "a.png"
    assert previous.read_bytes() == b"original image"
    if next_run == "changed_source":
        source.write_bytes(b"updated image")
    items = [] if next_run == "skipped" else [rewardmodel.ScoredImage(str(source), None, None, 5.0)]

    rewardmodel.apply_thresholds(items, thresholds=thresholds, source_root=tmp_path)

    assert not previous.exists()
    current = tmp_path / "best quality" / "nested" / "a.png"
    if next_run == "skipped":
        assert not current.exists()
    else:
        assert current.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("content", [b"original image", b"unrelated image"])
def test_threshold_partition_preserves_unowned_files(rewardmodel, copy_partition, tmp_path, content):
    source, thresholds = copy_partition
    existing = tmp_path / "low quality" / "nested" / "a.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(content)

    with pytest.raises(ValueError, match="unowned"):
        rewardmodel.apply_thresholds(
            [rewardmodel.ScoredImage(str(source), None, None, 1.0)],
            thresholds=thresholds, source_root=tmp_path,
        )

    assert existing.read_bytes() == content
    assert source.read_bytes() == b"original image"


def test_threshold_partition_preserves_modified_owned_copy(rewardmodel, copy_partition, tmp_path):
    source, thresholds = copy_partition
    item = rewardmodel.ScoredImage(str(source), None, None, 1.0)
    rewardmodel.apply_thresholds([item], thresholds=thresholds, source_root=tmp_path)
    existing = tmp_path / "low quality" / "nested" / "a.png"
    existing.write_bytes(b"user edit")

    with pytest.raises(ValueError, match="modified"):
        rewardmodel.apply_thresholds([item], thresholds=thresholds, source_root=tmp_path)

    assert existing.read_bytes() == b"user edit"


@pytest.mark.parametrize("payload", ["not json", '{"../outside.png": {}}'])
def test_threshold_partition_rejects_invalid_manifest(rewardmodel, copy_partition, tmp_path, payload):
    source, thresholds = copy_partition
    (tmp_path / ".reward_partition.json").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        rewardmodel.apply_thresholds(
            [rewardmodel.ScoredImage(str(source), None, None, 1.0)],
            thresholds=thresholds, source_root=tmp_path,
        )

    assert source.read_bytes() == b"original image"


def test_partition_manifest_cannot_escape_with_double_slash(rewardmodel, tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    relative = "//" + outside.as_posix().split(":", 1)[-1].lstrip("/")
    payload = {relative: {"kind": "copy", "sha256": hashlib.sha256(b"outside").hexdigest()}}
    (root / ".reward_partition.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        rewardmodel.apply_thresholds([], thresholds=(Threshold("low_quality", 2.0, "red"),), source_root=root)

    assert outside.read_bytes() == b"outside"


@pytest.mark.parametrize("operation,error", [("copyfileobj", OSError), ("copystat", OSError), ("copyfileobj", KeyboardInterrupt)])
def test_partition_copy_failure_can_be_retried(rewardmodel, copy_partition, tmp_path, monkeypatch, operation, error):
    source, thresholds = copy_partition
    item = rewardmodel.ScoredImage(str(source), None, None, 1.0)
    with monkeypatch.context() as patch:
        patch.setattr(rewardmodel.shutil, operation, lambda *_args: (_ for _ in ()).throw(error("copy failed")))
        with pytest.raises(error, match="copy failed"):
            rewardmodel.apply_thresholds([item], thresholds=thresholds, source_root=tmp_path)

    rewardmodel.apply_thresholds([item], thresholds=thresholds, source_root=tmp_path)
    assert (tmp_path / "low quality" / "nested" / "a.png").read_bytes() == source.read_bytes()


def test_partition_preflight_preserves_old_outputs_on_parent_file_collision(rewardmodel, copy_partition, tmp_path):
    source, thresholds = copy_partition
    rewardmodel.apply_thresholds(
        [rewardmodel.ScoredImage(str(source), None, None, 1.0)], thresholds=thresholds, source_root=tmp_path,
    )
    collision = tmp_path / "best quality" / "nested"
    collision.parent.mkdir()
    collision.write_bytes(b"user file")

    with pytest.raises(ValueError, match="directory"):
        rewardmodel.apply_thresholds(
            [rewardmodel.ScoredImage(str(source), None, None, 5.0)], thresholds=thresholds, source_root=tmp_path,
        )

    assert (tmp_path / "low quality" / "nested" / "a.png").read_bytes() == source.read_bytes()
    assert collision.read_bytes() == b"user file"


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


def _capture_run_console(rewardmodel, monkeypatch) -> list[str]:
    messages = []
    console = types.SimpleNamespace(
        print=lambda message, **_kwargs: messages.append(str(message))
    )
    monkeypatch.setattr(rewardmodel, "resolve_rich_console", lambda: console)
    return messages


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
    messages = _capture_run_console(rewardmodel, monkeypatch)

    args = rewardmodel.setup_parser().parse_args([str(tmp_path), "--device=cpu"])
    assert rewardmodel.run(args) == 0

    report = json.loads((tmp_path / "reward_scores.json").read_text(encoding="utf-8"))
    assert report == {"ok.png": 1.0}
    log = "\n".join(messages)
    assert "Qinglong Score: 0.2.2" in log
    assert "Scorer: aesthetic_predictor_v2_5" in log
    assert "missing.png" in log
    assert "decode failed" in log
    assert "Scoring completed with errors: scored=1, failed=1" in log


def test_all_failed_direct_lance_run_writes_sibling_report_and_returns_nonzero(
    rewardmodel, monkeypatch, tmp_path: Path
):
    missing_path = tmp_path / "missing.png"
    lance_path = tmp_path / "catalog.lance"
    _install_run_fakes(rewardmodel, monkeypatch, [str(missing_path)])
    messages = _capture_run_console(rewardmodel, monkeypatch)

    args = rewardmodel.setup_parser().parse_args([str(lance_path), "--device=cpu"])
    assert rewardmodel.run(args) == 1

    output = tmp_path / "catalog.reward_scores.json"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == {}
    log = "\n".join(messages)
    assert "missing.png" in log
    assert "Scoring failed: scored=0, failed=1" in log
