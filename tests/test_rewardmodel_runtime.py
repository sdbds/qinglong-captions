from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch
from PIL import Image


@pytest.fixture()
def rewardmodel():
    sys.modules.pop("module.rewardmodel", None)
    import module.rewardmodel as rewardmodel_module

    return rewardmodel_module


def _write_image(path: Path, *, size: tuple[int, int], value: int = 0) -> None:
    width, height = size
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


class FakeScorer:
    def __init__(self, *, fail_width: int | None = None, nonfinite: bool = False):
        self.device = torch.device("cpu")
        self.input_dtype = torch.float32
        self.fail_width = fail_width
        self.nonfinite = nonfinite
        self.calls: list[tuple[tuple[int, ...], object]] = []

    def score(self, images, prompts=None):
        self.calls.append((tuple(images.shape), prompts))
        if self.fail_width == images.shape[-1]:
            raise RuntimeError(f"width {self.fail_width} rejected")
        if self.nonfinite:
            return torch.full((images.shape[0],), float("nan"))
        return torch.arange(images.shape[0], device=images.device, dtype=torch.float32)


def test_parser_replaces_repo_id_with_scorer_and_checkpoint(rewardmodel):
    parser = rewardmodel.setup_parser()
    args = parser.parse_args(
        ["dataset", "--scorer=pickscore", "--checkpoint=repo/model"]
    )

    assert args.scorer == "pickscore"
    assert args.checkpoint == "repo/model"
    with pytest.raises(SystemExit):
        parser.parse_args(["dataset", "--repo_id=old/model"])


def test_dtype_auto_delegates_and_explicit_values_map_to_torch(rewardmodel):
    assert rewardmodel.resolve_dtype("auto") is None
    assert rewardmodel.resolve_dtype("float32") is torch.float32
    assert rewardmodel.resolve_dtype("float16") is torch.float16
    assert rewardmodel.resolve_dtype("bfloat16") is torch.bfloat16


def test_device_auto_and_index_validation_do_not_set_global_device(
    rewardmodel, monkeypatch
):
    set_device_calls = []
    monkeypatch.setattr(rewardmodel.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rewardmodel.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        rewardmodel.torch.cuda,
        "set_device",
        lambda index: set_device_calls.append(index),
    )

    assert rewardmodel.resolve_device("auto") == torch.device("cuda:0")
    assert rewardmodel.resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(ValueError, match="out of range"):
        rewardmodel.resolve_device("cuda:2")
    assert set_device_calls == []


def test_prompt_required_uses_override_caption_then_intentional_empty(rewardmodel):
    assert rewardmodel.select_prompt("caption", "override", True) == (
        "override",
        "override",
    )
    assert rewardmodel.select_prompt(["first", "second"], "", True) == (
        "first\nsecond",
        "caption",
    )
    assert rewardmodel.select_prompt([], "", True) == ("", "empty")
    assert rewardmodel.select_prompt(["ignored"], "", False) == (None, None)


def test_decode_image_constructs_contiguous_chw_float_in_unit_range(
    rewardmodel, tmp_path: Path
):
    path = tmp_path / "sample.png"
    _write_image(path, size=(5, 4), value=128)

    tensor = rewardmodel.decode_image(path)

    assert tuple(tensor.shape) == (3, 4, 5)
    assert tensor.dtype is torch.float32
    assert tensor.is_contiguous()
    assert tensor.min().item() == pytest.approx(128 / 255)
    assert tensor.max().item() == pytest.approx(128 / 255)


def test_same_sizes_batch_together_and_mixed_sizes_form_separate_calls(
    rewardmodel, tmp_path: Path
):
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    _write_image(paths[0], size=(8, 6), value=10)
    _write_image(paths[1], size=(8, 6), value=20)
    _write_image(paths[2], size=(9, 6), value=30)
    scorer = FakeScorer()
    records = [
        rewardmodel.SourceImage(str(path), [f"caption-{index}"])
        for index, path in enumerate(paths)
    ]

    items, errors = rewardmodel.score_source_batch(
        records,
        scorer=scorer,
        requires_prompts=True,
        prompt_override="",
        max_workers=1,
    )

    assert errors == []
    assert len(items) == 3
    assert [call[0] for call in scorer.calls] == [(2, 3, 6, 8), (1, 3, 6, 9)]
    assert scorer.calls[0][1] == ["caption-0", "caption-1"]
    assert scorer.calls[1][1] == ["caption-2"]


def test_image_only_scorer_receives_none_prompts(rewardmodel, tmp_path: Path):
    path = tmp_path / "a.png"
    _write_image(path, size=(4, 4))
    scorer = FakeScorer()

    items, errors = rewardmodel.score_source_batch(
        [rewardmodel.SourceImage(str(path), ["ignored"])],
        scorer=scorer,
        requires_prompts=False,
        prompt_override="override is ignored",
        max_workers=1,
    )

    assert errors == []
    assert items[0].prompt is None
    assert items[0].prompt_source is None
    assert scorer.calls == [((1, 3, 4, 4), None)]


def test_failed_shape_group_is_reported_once_without_singleton_retry(
    rewardmodel, tmp_path: Path
):
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    _write_image(paths[0], size=(8, 6))
    _write_image(paths[1], size=(9, 6))
    _write_image(paths[2], size=(9, 6))
    scorer = FakeScorer(fail_width=9)

    items, errors = rewardmodel.score_source_batch(
        [rewardmodel.SourceImage(str(path), []) for path in paths],
        scorer=scorer,
        requires_prompts=True,
        prompt_override="",
        max_workers=1,
    )

    assert len(items) == 1
    assert len(scorer.calls) == 2
    assert len(errors) == 1
    assert errors[0].scope == "batch"
    assert errors[0].stage == "score"
    assert errors[0].paths == (str(paths[1]), str(paths[2]))
    assert "width 9 rejected" in errors[0].message


def test_decode_failure_is_item_scoped(rewardmodel, tmp_path: Path):
    missing = tmp_path / "missing.png"

    items, errors = rewardmodel.score_source_batch(
        [rewardmodel.SourceImage(str(missing), None)],
        scorer=FakeScorer(),
        requires_prompts=False,
        prompt_override="",
        max_workers=1,
    )

    assert items == []
    assert len(errors) == 1
    assert errors[0].scope == "item"
    assert errors[0].stage == "decode"
    assert errors[0].path == str(missing)


def test_nonfinite_score_fails_the_group(rewardmodel, tmp_path: Path):
    path = tmp_path / "a.png"
    _write_image(path, size=(4, 4))

    items, errors = rewardmodel.score_source_batch(
        [rewardmodel.SourceImage(str(path), None)],
        scorer=FakeScorer(nonfinite=True),
        requires_prompts=False,
        prompt_override="",
        max_workers=1,
    )

    assert items == []
    assert len(errors) == 1
    assert errors[0].scope == "batch"
    assert "finite" in errors[0].message


def test_source_images_from_arrow_batch_preserves_per_row_captions(rewardmodel):
    batch = pa.record_batch(
        [
            pa.array(["a.png", "b.png"]),
            pa.array([["caption a"], ["line one", "line two"]]),
        ],
        names=["uris", "captions"],
    )

    assert rewardmodel.source_images_from_batch(batch) == [
        rewardmodel.SourceImage("a.png", ["caption a"]),
        rewardmodel.SourceImage("b.png", ["line one", "line two"]),
    ]


def test_source_images_from_arrow_batch_uses_missing_caption_values(rewardmodel):
    batch = pa.record_batch([pa.array(["a.png", "b.png"])], names=["uris"])

    assert rewardmodel.source_images_from_batch(batch) == [
        rewardmodel.SourceImage("a.png", None),
        rewardmodel.SourceImage("b.png", None),
    ]


def test_run_uses_public_loader_without_mutating_the_scorer(
    rewardmodel, monkeypatch, tmp_path: Path
):
    image_path = tmp_path / "sample.png"
    _write_image(image_path, size=(4, 3), value=64)
    load_calls = []

    @dataclass(frozen=True, slots=True)
    class FakeScorerSpec:
        name: str
        requires_prompts: bool
        default_compute_dtype: torch.dtype

    @dataclass(frozen=True, slots=True)
    class FakeCheckpoint:
        adapter: str
        identifier: str
        format: str
        artifacts: tuple[object, ...]
        is_default: bool
        tracks_updates: bool = False

    class ImmutableScorer(FakeScorer):
        checkpoint_identity = FakeCheckpoint(
            adapter="aesthetic_predictor_v2_5",
            identifier="owner/resolved",
            format="safetensors",
            artifacts=(object(),),
            is_default=True,
        )
        attention_backend = None

        def to(self, *_args, **_kwargs):
            raise AssertionError("run must not reconfigure a loaded scorer")

        def eval(self):
            raise AssertionError("run must not change scorer mode")

        def train(self, _mode=True):
            raise AssertionError("run must not change scorer mode")

    checkpoint = FakeCheckpoint(
        adapter="aesthetic_predictor_v2_5",
        identifier="owner/default",
        format="safetensors",
        artifacts=(object(),),
        is_default=True,
    )
    fake_module = types.ModuleType("qinglong_score")
    fake_module.__version__ = "0.2.2"
    fake_module.get_scorer_spec = lambda name: FakeScorerSpec(
        name=name,
        requires_prompts=False,
        default_compute_dtype=torch.float32,
    )
    fake_module.list_checkpoints = lambda _name: (checkpoint,)

    def load_scorer(**kwargs):
        load_calls.append(kwargs)
        return ImmutableScorer()

    fake_module.load_scorer = load_scorer
    monkeypatch.setitem(sys.modules, "qinglong_score", fake_module)

    class FakeScanner:
        def to_batches(self):
            return [
                pa.record_batch(
                    [pa.array([str(image_path)]), pa.array(["image/png"])],
                    names=["uris", "mime"],
                )
            ]

    class FakeDataset:
        schema = pa.schema([("uris", pa.string()), ("mime", pa.string())])

        def scanner(self, **kwargs):
            assert kwargs["columns"] == ["uris", "mime"]
            assert kwargs["filter"] == "mime LIKE 'image/%'"
            assert kwargs["batch_size"] == 2
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

    args = rewardmodel.setup_parser().parse_args(
        [str(tmp_path), "--batch_size=2", "--device=cpu"]
    )
    assert rewardmodel.run(args) == 0
    assert load_calls == [
        {
            "name": "aesthetic_predictor_v2_5",
            "checkpoint": None,
            "device": "cpu",
            "dtype": None,
            "attention_backend": "auto",
        }
    ]
