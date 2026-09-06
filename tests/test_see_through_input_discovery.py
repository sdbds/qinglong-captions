from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console

import module.see_through.runner as runner


@pytest.mark.parametrize("same_directory", [False, True])
def test_public_batch_does_not_rediscover_its_own_outputs(monkeypatch, tmp_path, same_directory):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    Image.new("RGBA", (4, 4), "red").save(inputs / "a.png")
    processed = []
    backups = []

    class Manager:
        def __init__(self, **kwargs):
            pass

        def log_vram(self, stage):
            return {"stage": stage, "device": "cpu"}

        def release_layerdiff(self):
            pass

        def release_marigold(self):
            pass

        def release_all(self):
            pass

    class LayerDiff:
        def __init__(self, *args, **kwargs):
            pass

        def run_item(self, source_path, item_dir):
            processed.append(source_path.relative_to(inputs).as_posix())
            Image.new("RGBA", (4, 4), "red").save(item_dir / "src_img.png")
            (item_dir / "layerdiff").mkdir(exist_ok=True)
            (item_dir / "layerdiff" / "manifest.json").write_text("{}", encoding="utf-8")

    class Marigold:
        def __init__(self, *args, **kwargs):
            pass

        def run_item(self, source_path, item_dir):
            (item_dir / "depth").mkdir(exist_ok=True)
            Image.new("L", (4, 4), 128).save(item_dir / "depth" / "depth.png")

    def postprocess(*, output_dir, **kwargs):
        (output_dir / "optimized").mkdir(exist_ok=True)
        (output_dir / "optimized" / "manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "final.psd").write_bytes(b"psd")

    def backup(**kwargs):
        backups.append([path.relative_to(inputs).as_posix() for path in kwargs["source_paths"]])
        return {"dataset_path": str(inputs / "dataset.lance"), "tag": "review", "version": 1}

    monkeypatch.setattr(runner, "SeeThroughModelManager", Manager)
    monkeypatch.setattr(runner, "LayerDiffPhase", LayerDiff)
    monkeypatch.setattr(runner, "MarigoldPhase", Marigold)
    monkeypatch.setattr(runner, "run_postprocess", postprocess)
    monkeypatch.setattr(runner, "backup_input_dataset_to_lance", backup)
    monkeypatch.setattr(runner, "resolve_attention_backend", lambda **kwargs: SimpleNamespace(attention_backend="eager"))
    config = SimpleNamespace(
        input_dir=inputs, output_dir=inputs if same_directory else inputs / "outputs",
        repo_id_layerdiff="review/layerdiff", repo_id_depth="review/depth",
        resolution=1280, dtype="bfloat16", offload_policy="delete",
        skip_completed=True, continue_on_error=True, save_to_psd=True,
        tblr_split=False, limit_images=0, force_eager_attention=False,
    )
    output = Console(file=io.StringIO(), force_terminal=False, color_system=None)
    assert runner.run_see_through_batch(config, console_obj=output) == 0
    assert runner.run_see_through_batch(config, console_obj=output) == 0
    assert processed == ["a.png"]
    assert backups == [["a.png"], ["a.png"]]


def test_discovery_excludes_managed_versions_before_limit_not_directory_names(tmp_path):
    for name in ("00-old", "01-older", "outputs"):
        (tmp_path / name).mkdir()
        Image.new("RGB", (2, 2)).save(tmp_path / name / "source.png")
    for name in ("00-old", "01-older"):
        (tmp_path / name / "run_meta.json").write_text(json.dumps({
            "input_dir": str(tmp_path), "config_fingerprint": "older-config", "created_at": 1.0,
        }), encoding="utf-8")
    assert runner.collect_input_images(tmp_path, 1) == [tmp_path / "outputs" / "source.png"]


def test_explicit_output_directory_is_excluded_without_a_marker(tmp_path):
    output = tmp_path / "00-output"
    output.mkdir()
    Image.new("RGB", (2, 2)).save(output / "generated.png")
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(source)
    assert runner.collect_input_images(tmp_path, 1, output_dir=output) == [source]


@pytest.mark.parametrize("directory_link", [False, True])
def test_discovery_does_not_follow_output_aliases(tmp_path, directory_link):
    output = tmp_path / "outputs"
    output.mkdir()
    generated = output / "generated.png"
    Image.new("RGB", (2, 2)).save(generated)
    (output / "run_meta.json").write_text(json.dumps({
        "input_dir": str(tmp_path), "config_fingerprint": "config", "created_at": 1.0,
    }), encoding="utf-8")
    alias = tmp_path / ("alias" if directory_link else "alias.png")
    try:
        alias.symlink_to(output if directory_link else generated, target_is_directory=directory_link)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    assert runner.collect_input_images(tmp_path, 0) == []
