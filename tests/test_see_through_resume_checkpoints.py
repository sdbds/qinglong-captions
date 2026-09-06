import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from module.see_through import runner


pytestmark = pytest.mark.compat

PHASE_OUTPUTS = {
    "layerdiff": ("src_img.png", "layerdiff/manifest.json"),
    "marigold": ("depth/depth.png",),
    "postprocess": ("optimized/manifest.json", "final.psd"),
}


def _write_outputs(item_dir, phase, content=b"output"):
    for name in PHASE_OUTPUTS[phase]:
        path = item_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _run_phase(item, phase, handler=None):
    return runner._process_phase_items(
        phase_name=phase,
        items=[item],
        handler=handler or (lambda value: _write_outputs(value.item_dir, phase)),
        continue_on_error=True,
        console_obj=Console(file=io.StringIO(), force_terminal=False),
    )


@pytest.fixture
def planned_image(tmp_path):
    source = tmp_path / "input.png"
    source.write_bytes(b"abc")
    config = SimpleNamespace(input_dir=tmp_path, skip_completed=True, save_to_psd=True)
    output = tmp_path / "out"
    item = runner.build_execution_plan(config, output, [source])[0]
    return config, output, source, item


@pytest.mark.parametrize("finished_phase, expected", [
    ("layerdiff", "marigold"),
    ("marigold", "postprocess"),
])
def test_completed_intermediate_phases_are_resumable(planned_image, finished_phase, expected):
    config, output, source, item = planned_image
    for phase in PHASE_OUTPUTS:
        successes, failures = _run_phase(item, phase)
        assert successes == [item] and failures == 0
        if phase == finished_phase:
            break

    resumed = runner.build_execution_plan(config, output, [source])[0]
    assert resumed.resume_stage == expected
    assert runner._select_auto_rig_items([resumed]) == []


def test_changed_source_cannot_reuse_stale_downstream_outputs(planned_image):
    config, output, source, item = planned_image
    for phase in PHASE_OUTPUTS:
        _run_phase(item, phase)
    assert runner.build_execution_plan(config, output, [source])[0].resume_stage == "completed"
    source.write_bytes(b"changed source")
    changed = runner.build_execution_plan(config, output, [source])[0]
    assert changed.resume_stage == "layerdiff"
    for phase, expected in (("layerdiff", "marigold"), ("marigold", "postprocess")):
        _run_phase(changed, phase)
        resumed = runner.build_execution_plan(config, output, [source])[0]
        assert resumed.resume_stage == expected
        assert (item.item_dir / "final.psd").read_bytes() == b"output"
        assert runner._select_auto_rig_items([resumed]) == []
    _run_phase(changed, "postprocess")
    completed = runner.build_execution_plan(config, output, [source])[0]
    assert completed.resume_stage == "completed"
    assert runner._select_auto_rig_items([completed]) == [completed]


@pytest.mark.parametrize("failed_phase", ["layerdiff", "marigold", "postprocess"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_rerun_failure_invalidates_only_current_and_downstream_phases(planned_image, failed_phase, error_type):
    config, output, source, item = planned_image
    for phase in PHASE_OUTPUTS:
        _run_phase(item, phase)
    force_config = SimpleNamespace(**{**vars(config), "skip_completed": False})
    rerun = runner.build_execution_plan(force_config, output, [source])[0]
    for phase in PHASE_OUTPUTS:
        if phase == failed_phase:
            break
        _run_phase(rerun, phase)

    def interrupted(value):
        _write_outputs(value.item_dir, failed_phase, b"partial overwrite")
        raise error_type("phase interrupted")

    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            _run_phase(rerun, failed_phase, interrupted)
    else:
        successes, failures = _run_phase(rerun, failed_phase, interrupted)
        assert successes == [] and failures == 1
    resumed = runner.build_execution_plan(config, output, [source])[0]
    assert resumed.resume_stage == failed_phase
    assert runner._select_auto_rig_items([resumed]) == []


@pytest.mark.parametrize("missing, expected", [
    (None, "completed"),
    ("depth/depth.png", "marigold"),
    ("final.psd", "postprocess"),
])
def test_legacy_completed_source_marker_remains_usable(planned_image, missing, expected):
    config, output, source, item = planned_image
    for phase in PHASE_OUTPUTS:
        _write_outputs(item.item_dir, phase)
    (item.item_dir / runner.SOURCE_METADATA).write_text(json.dumps({
        "schema_version": 1,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }), encoding="utf-8")
    if missing is not None:
        (item.item_dir / missing).unlink()
    resumed = runner.build_execution_plan(config, output, [source])[0]
    assert resumed.resume_stage == expected
    assert runner._select_auto_rig_items([resumed]) == ([resumed] if missing is None else [])


@pytest.mark.parametrize("metadata", [
    {"schema_version": 999, "completed_phase": "postprocess"},
    {"schema_version": 2, "completed_phase": "unknown"},
    {"schema_version": 2, "completed_phase": []},
    "not a metadata object",
])
def test_unrecognized_checkpoint_cannot_reuse_existing_outputs(planned_image, metadata):
    config, output, source, item = planned_image
    for phase in PHASE_OUTPUTS:
        _write_outputs(item.item_dir, phase)
    if isinstance(metadata, dict):
        metadata = {**metadata, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
    (item.item_dir / runner.SOURCE_METADATA).write_text(json.dumps(metadata), encoding="utf-8")
    resumed = runner.build_execution_plan(config, output, [source])[0]
    assert resumed.resume_stage == "layerdiff"
    assert runner._select_auto_rig_items([resumed]) == []
    assert (item.item_dir / "final.psd").read_bytes() == b"output"


def test_checkpoint_commit_failure_does_not_publish_completion(planned_image, monkeypatch):
    from module.music_export import service

    config, output, source, item = planned_image
    for phase in ("layerdiff", "marigold"):
        _run_phase(item, phase)
    original_replace = service.os.replace
    output_written = False

    def write_postprocess(value):
        nonlocal output_written
        _write_outputs(value.item_dir, "postprocess")
        output_written = True

    def fail_commit(source_path, target):
        if output_written and str(target).endswith(runner.SOURCE_METADATA):
            raise OSError("checkpoint commit failed")
        return original_replace(source_path, target)

    monkeypatch.setattr(service.os, "replace", fail_commit)
    successes, failures = _run_phase(item, "postprocess", write_postprocess)
    assert successes == [] and failures == 1
    resumed = runner.build_execution_plan(config, output, [source])[0]
    assert resumed.resume_stage == "postprocess"
    assert runner._select_auto_rig_items([resumed]) == []


def test_failed_invalidation_never_enters_handler_or_changes_verified_outputs(planned_image, monkeypatch):
    config, output, source, item = planned_image
    for phase in PHASE_OUTPUTS:
        _run_phase(item, phase)
    metadata_path = item.item_dir / runner.SOURCE_METADATA
    previous_metadata = metadata_path.read_bytes()
    original_unlink = Path.unlink
    called = False

    def prevent_invalidation(path, *args, **kwargs):
        if path == metadata_path:
            raise PermissionError("checkpoint is locked")
        return original_unlink(path, *args, **kwargs)

    def overwrite(value):
        nonlocal called
        called = True
        _write_outputs(value.item_dir, "layerdiff", b"new generation")

    monkeypatch.setattr(Path, "unlink", prevent_invalidation)
    successes, failures = _run_phase(item, "layerdiff", overwrite)
    assert successes == [] and failures == 1
    assert called is False
    assert metadata_path.read_bytes() == previous_metadata
    for names in PHASE_OUTPUTS.values():
        assert all((item.item_dir / name).read_bytes() == b"output" for name in names)
    assert runner.build_execution_plan(config, output, [source])[0].resume_stage == "completed"


def test_batch_retries_only_failed_postprocess(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "image.png").write_bytes(b"source")
    config = SimpleNamespace(
        input_dir=inputs, output_dir=tmp_path / "outputs", skip_completed=True,
        repo_id_layerdiff="layerdiff", repo_id_depth="marigold", resolution=1024,
        resolution_depth=768, dtype="float16", offload_policy="delete",
        save_to_psd=True, tblr_split=False, limit_images=0, force_eager_attention=False,
        continue_on_error=True,
    )
    events = []

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

    def infer(phase, item_dir):
        events.append(phase)
        _write_outputs(item_dir, phase)
        if events == ["layerdiff", "marigold", "postprocess"]:
            raise RuntimeError("postprocess interrupted")

    monkeypatch.setattr(runner, "SeeThroughModelManager", Manager)
    monkeypatch.setattr(runner, "resolve_attention_backend", lambda **kwargs: SimpleNamespace(attention_backend="eager", device="cpu"))
    monkeypatch.setattr(runner, "backup_input_dataset_to_lance", lambda **kwargs: {
        "dataset_path": str(tmp_path / "backup.lance"), "tag": "test", "version": 1,
    })
    monkeypatch.setattr(runner, "LayerDiffPhase", lambda *args, **kwargs: SimpleNamespace(
        run_item=lambda source, item_dir: infer("layerdiff", item_dir),
    ))
    monkeypatch.setattr(runner, "MarigoldPhase", lambda *args, **kwargs: SimpleNamespace(
        run_item=lambda source, item_dir: infer("marigold", item_dir),
    ))
    monkeypatch.setattr(runner, "run_postprocess", lambda **kwargs: infer("postprocess", kwargs["output_dir"]))
    console = Console(file=io.StringIO(), force_terminal=False)
    assert runner.run_see_through_batch(config, console_obj=console) == 1
    assert runner.run_see_through_batch(config, console_obj=console) == 0
    assert runner.run_see_through_batch(config, console_obj=console) == 0
    assert events == ["layerdiff", "marigold", "postprocess", "postprocess"]
