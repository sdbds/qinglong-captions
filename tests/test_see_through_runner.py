import io
from pathlib import Path
import sys
from types import SimpleNamespace

import lance
from PIL import Image
from rich.console import Console
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import module.see_through.runner as runner_module
from module.see_through.runner import (
    backup_input_dataset_to_lance,
    detect_resume_stage,
    make_item_dir,
    make_relative_output_key,
    prepare_output_dir,
)


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(path)


def test_runner_reuses_output_dir_when_run_meta_matches(tmp_path):
    output_dir = tmp_path / "outputs"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    prepared = prepare_output_dir(output_dir, input_dir, "fp-1")

    assert prepared["output_dir"] == output_dir
    assert (output_dir / "run_meta.json").exists()

    reused = prepare_output_dir(output_dir, input_dir, "fp-1")

    assert reused["output_dir"] == output_dir


def test_runner_rejects_output_dir_when_config_changes(tmp_path):
    output_dir = tmp_path / "outputs"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    prepare_output_dir(output_dir, input_dir, "fp-1")

    with pytest.raises(ValueError):
        prepare_output_dir(output_dir, input_dir, "fp-2")


def test_runner_rejects_output_dir_when_input_dir_changes(tmp_path):
    output_dir = tmp_path / "outputs"
    input_dir = tmp_path / "inputs"
    other_input_dir = tmp_path / "other"
    input_dir.mkdir()
    other_input_dir.mkdir()
    prepare_output_dir(output_dir, input_dir, "fp-1")

    with pytest.raises(ValueError):
        prepare_output_dir(output_dir, other_input_dir, "fp-1")


def test_runner_uses_outputs_child_when_base_dir_is_non_empty(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(b"png")

    prepared = prepare_output_dir(input_dir, input_dir, "fp-1")

    assert prepared["output_dir"] == input_dir / "outputs"
    assert (input_dir / "outputs" / "run_meta.json").exists()


def test_runner_uses_next_outputs_child_when_default_outputs_is_occupied(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(b"png")
    occupied = input_dir / "outputs"
    occupied.mkdir()
    (occupied / "stale.txt").write_text("occupied", encoding="utf-8")

    prepared = prepare_output_dir(input_dir, input_dir, "fp-1")

    assert prepared["output_dir"] == input_dir / "outputs-2"
    assert (input_dir / "outputs-2" / "run_meta.json").exists()


def test_runner_uses_relative_path_as_output_key(tmp_path):
    input_dir = tmp_path / "inputs"
    source_path = input_dir / "foo" / "a.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"png")

    relative_key = make_relative_output_key(input_dir, source_path)
    item_dir = make_item_dir(tmp_path / "outputs", relative_key)

    assert relative_key.as_posix() == "foo/a.png"
    assert item_dir.as_posix().endswith("foo/a.png")


def test_runner_duplicate_stems_do_not_collide(tmp_path):
    output_dir = tmp_path / "outputs"

    item_dir_a = make_item_dir(output_dir, Path("foo/a.png"))
    item_dir_b = make_item_dir(output_dir, Path("bar/a.png"))

    assert item_dir_a != item_dir_b


def test_runner_detects_stage_from_existing_outputs(tmp_path):
    item_dir = tmp_path / "outputs" / "foo" / "a.png"
    (item_dir / "layerdiff").mkdir(parents=True)
    (item_dir / "depth").mkdir(parents=True)
    (item_dir / "src_img.png").write_bytes(b"png")
    (item_dir / "layerdiff" / "manifest.json").write_text("{}", encoding="utf-8")
    (item_dir / "depth" / "depth.png").write_bytes(b"png")

    assert detect_resume_stage(item_dir) == "postprocess"


def test_backup_input_dataset_to_lance_writes_versioned_snapshot(tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()
    image_path = input_dir / "a.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(image_path)

    buffer = io.StringIO()
    rich_console = Console(file=buffer, force_terminal=False, color_system=None)

    backup = backup_input_dataset_to_lance(
        input_dir=input_dir,
        output_dir=output_dir,
        source_paths=[image_path],
        console_obj=rich_console,
    )

    dataset_path = Path(backup["dataset_path"])
    assert dataset_path.exists()
    assert dataset_path == input_dir / "dataset.lance"
    assert backup["tag"].startswith("raw.see_through.backup.")
    ds = lance.dataset(str(dataset_path), version=backup["tag"])
    assert ds.count_rows() == 1

    logs = buffer.getvalue()
    assert "Creating Lance backup" in logs
    assert "Lance backup ready" in logs


def test_runner_batches_phase_order_and_releases_models(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    write_png(input_dir / "a.png")
    write_png(input_dir / "b.png")

    events: list[tuple[str, str]] = []

    class FakeManager:
        def __init__(self, *, offload_policy, runtime_context):
            self.offload_policy = offload_policy
            self.runtime_context = runtime_context

        def log_vram(self, stage_name):
            return {"stage": stage_name, "device": "cpu"}

        def release_layerdiff(self):
            events.append(("release", "layerdiff"))

        def release_marigold(self):
            events.append(("release", "marigold"))

        def release_all(self):
            return None

    class FakeLayerDiffPhase:
        def __init__(self, model_manager, config, runtime_context, console_obj=None):
            self.model_manager = model_manager
            self.config = config
            self.runtime_context = runtime_context
            self.console_obj = console_obj

        def run_item(self, source_path, item_dir):
            events.append(("layerdiff", source_path.name))
            (item_dir / "src_img.png").write_bytes(b"png")
            (item_dir / "layerdiff").mkdir(parents=True, exist_ok=True)
            (item_dir / "layerdiff" / "manifest.json").write_text("{}", encoding="utf-8")

    class FakeMarigoldPhase:
        def __init__(self, model_manager, config, runtime_context, console_obj=None):
            self.model_manager = model_manager
            self.config = config
            self.runtime_context = runtime_context
            self.console_obj = console_obj

        def run_item(self, source_path, item_dir):
            events.append(("marigold", source_path.name))
            (item_dir / "depth").mkdir(parents=True, exist_ok=True)
            (item_dir / "depth" / "depth.png").write_bytes(b"png")

    def fake_postprocess(*, source_path, output_dir, save_to_psd, tblr_split):
        events.append(("postprocess", source_path.name))
        (output_dir / "optimized").mkdir(parents=True, exist_ok=True)
        (output_dir / "optimized" / "manifest.json").write_text("{}", encoding="utf-8")
        if save_to_psd:
            (output_dir / "final.psd").write_bytes(b"psd")
        return {"optimized_manifest": output_dir / "optimized" / "manifest.json"}

    monkeypatch.setattr(runner_module, "resolve_attention_backend", lambda **kwargs: SimpleNamespace(attention_backend="eager"))
    monkeypatch.setattr(runner_module, "SeeThroughModelManager", FakeManager)
    monkeypatch.setattr(runner_module, "LayerDiffPhase", FakeLayerDiffPhase)
    monkeypatch.setattr(runner_module, "MarigoldPhase", FakeMarigoldPhase)
    monkeypatch.setattr(runner_module, "run_postprocess", fake_postprocess)

    config = SimpleNamespace(
        input_dir=input_dir,
        output_dir=output_dir,
        repo_id_layerdiff="layerdiff/repo",
        repo_id_depth="marigold/repo",
        resolution=1280,
        dtype="bfloat16",
        offload_policy="delete",
        skip_completed=False,
        continue_on_error=True,
        save_to_psd=True,
        tblr_split=False,
        limit_images=0,
        force_eager_attention=False,
        vae_ckpt=None,
        unet_ckpt=None,
    )

    exit_code = runner_module.run_see_through_batch(config)

    assert exit_code == 0
    assert events == [
        ("layerdiff", "a.png"),
        ("layerdiff", "b.png"),
        ("release", "layerdiff"),
        ("marigold", "a.png"),
        ("marigold", "b.png"),
        ("release", "marigold"),
        ("postprocess", "a.png"),
        ("postprocess", "b.png"),
    ]


def test_runner_logs_backup_runtime_and_item_successes(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    write_png(input_dir / "a.png")

    buffer = io.StringIO()
    rich_console = Console(file=buffer, force_terminal=False, color_system=None)

    class FakeManager:
        def __init__(self, *, offload_policy, runtime_context):
            self.offload_policy = offload_policy
            self.runtime_context = runtime_context

        def release_layerdiff(self):
            return None

        def release_marigold(self):
            return None

        def release_all(self):
            return None

        def log_vram(self, stage_name):
            return {"stage": stage_name, "device": "cpu"}

    class FakeLayerDiffPhase:
        def __init__(self, model_manager, config, runtime_context, console_obj=None):
            self.console_obj = console_obj

        def run_item(self, source_path, item_dir):
            (item_dir / "src_img.png").write_bytes(b"png")
            (item_dir / "layerdiff").mkdir(parents=True, exist_ok=True)
            (item_dir / "layerdiff" / "manifest.json").write_text("{}", encoding="utf-8")

    class FakeMarigoldPhase:
        def __init__(self, model_manager, config, runtime_context, console_obj=None):
            self.console_obj = console_obj

        def run_item(self, source_path, item_dir):
            (item_dir / "depth").mkdir(parents=True, exist_ok=True)
            (item_dir / "depth" / "depth.png").write_bytes(b"png")

    def fake_postprocess(*, source_path, output_dir, save_to_psd, tblr_split):
        (output_dir / "optimized").mkdir(parents=True, exist_ok=True)
        (output_dir / "optimized" / "manifest.json").write_text("{}", encoding="utf-8")
        if save_to_psd:
            (output_dir / "final.psd").write_bytes(b"psd")
        return {"optimized_manifest": output_dir / "optimized" / "manifest.json"}

    monkeypatch.setattr(runner_module, "console", rich_console)
    monkeypatch.setattr(
        runner_module,
        "backup_input_dataset_to_lance",
        lambda **kwargs: {
            "dataset_path": str(input_dir / "dataset.lance"),
            "tag": "raw.see_through.backup.test",
            "version": 3,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_attention_backend",
        lambda **kwargs: SimpleNamespace(
            attention_backend="flash_attn",
            reason="flash-attn import probe succeeded",
            device="cuda",
            dtype="bfloat16",
        ),
    )
    monkeypatch.setattr(runner_module, "SeeThroughModelManager", FakeManager)
    monkeypatch.setattr(runner_module, "LayerDiffPhase", FakeLayerDiffPhase)
    monkeypatch.setattr(runner_module, "MarigoldPhase", FakeMarigoldPhase)
    monkeypatch.setattr(runner_module, "run_postprocess", fake_postprocess)

    config = SimpleNamespace(
        input_dir=input_dir,
        output_dir=output_dir,
        repo_id_layerdiff="layerdiff/repo",
        repo_id_depth="marigold/repo",
        resolution=1280,
        dtype="bfloat16",
        offload_policy="delete",
        skip_completed=False,
        continue_on_error=True,
        save_to_psd=True,
        tblr_split=False,
        limit_images=0,
        force_eager_attention=False,
        vae_ckpt=None,
        unet_ckpt=None,
    )

    exit_code = runner_module.run_see_through_batch(config)

    assert exit_code == 0
    logs = buffer.getvalue()
    assert "Lance backup ready" in logs
    assert "Attention backend" in logs
    assert "LayerDiff succeeded" in logs
    assert "Marigold succeeded" in logs
    assert "Postprocess succeeded" in logs


def test_runner_logs_item_failures_with_rich_exception(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    write_png(input_dir / "a.png")
    write_png(input_dir / "b.png")

    buffer = io.StringIO()
    rich_console = Console(file=buffer, force_terminal=False, color_system=None)

    class FakeManager:
        def __init__(self, *, offload_policy, runtime_context):
            self.offload_policy = offload_policy
            self.runtime_context = runtime_context

        def release_layerdiff(self):
            return None

        def release_marigold(self):
            return None

        def release_all(self):
            return None

        def log_vram(self, stage_name):
            return {"stage": stage_name, "device": "cpu"}

    class FailingLayerDiffPhase:
        def __init__(self, model_manager, config, runtime_context, console_obj=None):
            self.console_obj = console_obj

        def run_item(self, source_path, item_dir):
            if source_path.name == "b.png":
                raise RuntimeError("boom")
            (item_dir / "src_img.png").write_bytes(b"png")
            (item_dir / "layerdiff").mkdir(parents=True, exist_ok=True)
            (item_dir / "layerdiff" / "manifest.json").write_text("{}", encoding="utf-8")

    class FakeMarigoldPhase:
        def __init__(self, model_manager, config, runtime_context, console_obj=None):
            self.console_obj = console_obj

        def run_item(self, source_path, item_dir):
            (item_dir / "depth").mkdir(parents=True, exist_ok=True)
            (item_dir / "depth" / "depth.png").write_bytes(b"png")

    def fake_postprocess(*, source_path, output_dir, save_to_psd, tblr_split):
        (output_dir / "optimized").mkdir(parents=True, exist_ok=True)
        (output_dir / "optimized" / "manifest.json").write_text("{}", encoding="utf-8")
        if save_to_psd:
            (output_dir / "final.psd").write_bytes(b"psd")
        return {"optimized_manifest": output_dir / "optimized" / "manifest.json"}

    monkeypatch.setattr(runner_module, "console", rich_console)
    monkeypatch.setattr(
        runner_module,
        "backup_input_dataset_to_lance",
        lambda **kwargs: {
            "dataset_path": str(input_dir / "dataset.lance"),
            "tag": "raw.see_through.backup.test",
            "version": 3,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_attention_backend",
        lambda **kwargs: SimpleNamespace(
            attention_backend="sdpa",
            reason="PyTorch SDPA available",
            device="cuda",
            dtype="bfloat16",
        ),
    )
    monkeypatch.setattr(runner_module, "SeeThroughModelManager", FakeManager)
    monkeypatch.setattr(runner_module, "LayerDiffPhase", FailingLayerDiffPhase)
    monkeypatch.setattr(runner_module, "MarigoldPhase", FakeMarigoldPhase)
    monkeypatch.setattr(runner_module, "run_postprocess", fake_postprocess)

    config = SimpleNamespace(
        input_dir=input_dir,
        output_dir=output_dir,
        repo_id_layerdiff="layerdiff/repo",
        repo_id_depth="marigold/repo",
        resolution=1280,
        dtype="bfloat16",
        offload_policy="delete",
        skip_completed=False,
        continue_on_error=True,
        save_to_psd=True,
        tblr_split=False,
        limit_images=0,
        force_eager_attention=False,
        vae_ckpt=None,
        unet_ckpt=None,
    )

    exit_code = runner_module.run_see_through_batch(config)

    assert exit_code == 1
    logs = buffer.getvalue()
    assert "LayerDiff failed for" in logs
    assert "RuntimeError: boom" in logs
