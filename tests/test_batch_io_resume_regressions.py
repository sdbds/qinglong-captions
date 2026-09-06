from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest
from rich.console import Console

from module import texttranslate
from module.muscriptor_tool.batch import run_batch
from module.muscriptor_tool.options import BatchOptions, TranscriptionOptions
from module.muscriptor_tool.outputs import TranscriptionResult
from module.muscriptor_tool import manifest
from module.muscriptor_tool.stems import StemMidiCandidate, transcribe_stem_candidates
from module.see_through import runner


class RecordingTranslator:
    model_id = "test-model"
    backend = "direct"
    max_new_tokens = 4096
    temperature = 0.0

    def __init__(self):
        self.calls = []

    def translate(self, text, **kwargs):
        self.calls.append(text)
        return "translated " + text


def translate(path, version, translator, root, **changes):
    kwargs = dict(source_lang="en", target_lang="zh_cn", max_chars=2200,
                  context_chars=300, glossary="", export_root=root)
    kwargs.update(changes)
    texttranslate.translate_dataset(path, version, "translated", translator, **kwargs)


def dataset_with_documents(root, names, captions):
    paths = [root / name for name in names]
    for path, caption in zip(paths, captions):
        path.write_text(caption, encoding="utf-8")
    path = root / "input.lance"
    ds = lance.write_dataset(pa.table({"uris": [str(p) for p in paths],
                                      "captions": [[c] for c in captions]}), str(path))
    return path, ds.version


def test_translation_same_stem_documents_keep_distinct_results(tmp_path):
    path, version = dataset_with_documents(tmp_path, ["report.txt", "report.md"], ["first", "second"])
    translator = RecordingTranslator()
    translate(path, version, translator, tmp_path)
    assert translator.calls == ["first", "second"]
    rows = lance.dataset(str(path)).to_table().column("captions").to_pylist()
    assert rows == [["translated first\n"], ["translated second\n"]]


@pytest.mark.parametrize("change", ["source", "glossary", "model", "chunks", "language"])
def test_translation_resume_checks_source_and_configuration(tmp_path, change):
    path, version = dataset_with_documents(tmp_path, ["report.txt"], ["first"])
    translator = RecordingTranslator()
    translate(path, version, translator, tmp_path)
    kwargs = {}
    if change == "source":
        source = "second"
        ds = lance.write_dataset(pa.table({"uris": [str(tmp_path / "report.txt")],
                                          "captions": [[source]]}), str(path), mode="overwrite")
        version = ds.version
    elif change == "glossary":
        kwargs["glossary"] = "new glossary"
    elif change == "model":
        translator.model_id = "different-model"
    elif change == "chunks":
        kwargs["max_chars"] = 2201
    elif change == "language":
        kwargs["source_lang"] = "fr"
    translate(path, version, translator, tmp_path, **kwargs)
    assert len(translator.calls) == 2


def test_translation_resume_ignores_version_only_rewrite(tmp_path):
    path, version = dataset_with_documents(tmp_path, ["report.txt"], ["first"])
    translator = RecordingTranslator()
    translate(path, version, translator, tmp_path)
    rewritten = lance.write_dataset(
        pa.table({"uris": [str(tmp_path / "report.txt")], "captions": [["first"]]}),
        str(path),
        mode="overwrite",
    )

    translate(path, rewritten.version, translator, tmp_path)

    assert translator.calls == ["first"]


def test_translation_valid_resume_ignores_transient_translator_state(tmp_path):
    path, version = dataset_with_documents(tmp_path, ["report.txt"], ["first"])
    translator = RecordingTranslator()
    translate(path, version, translator, tmp_path)
    translate(path, version, translator, tmp_path)
    assert translator.calls == ["first"]


def test_translation_does_not_trust_or_overwrite_unowned_exports(tmp_path):
    path, version = dataset_with_documents(tmp_path, ["report.txt"], ["first"])
    legacy = tmp_path / "report_zh_cn.md"
    legacy.write_text("user legacy content", encoding="utf-8")
    new_name = tmp_path / "report.txt_zh_cn.md"
    new_name.write_text("user new content", encoding="utf-8")
    translator = RecordingTranslator()
    translate(path, version, translator, tmp_path)
    assert translator.calls == ["first"]
    assert legacy.read_text(encoding="utf-8") == "user legacy content"
    assert new_name.read_text(encoding="utf-8") == "user new content"
    assert lance.dataset(str(path)).to_table().column("captions").to_pylist() == [["translated first\n"]]


def successful_transcriber(_loaded, _source, _options, targets, **kwargs):
    for path in targets.requested_paths().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old success")
    return TranscriptionResult(1, 1, 1, 1,
                               {k: str(v) for k, v in targets.requested_paths().items()}, ())


def failed_transcriber(_loaded, _source, _options, targets, **kwargs):
    targets.midi.write_bytes(b"new incomplete output")
    raise RuntimeError("inference failed")


def replacement_transcriber(loaded, source, options, targets, **kwargs):
    result = successful_transcriber(loaded, source, options, targets, **kwargs)
    targets.midi.write_bytes(b"new success")
    return result


def test_failed_batch_rerun_keeps_old_outputs_and_their_metadata(tmp_path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    common = dict(input_path=source, output_dir=tmp_path / "out", package_version="test",
                  resolved_device="cpu", model_loader=lambda _: object())
    run_batch(**common, transcriber=successful_transcriber)
    item_dir = tmp_path / "out" / "song.wav"
    old_metadata = (item_dir / "metadata.json").read_bytes()
    result = run_batch(**common, options=BatchOptions(transcription=TranscriptionOptions(instruments=("voice",))),
                       transcriber=failed_transcriber)
    assert result.exit_code == 1
    assert (item_dir / "song.mid").read_bytes() == b"old success"
    assert (item_dir / "metadata.json").read_bytes() == old_metadata
    assert Path(result.items[0].metadata_path).name == "last_attempt.json"


def test_failed_stem_rerun_keeps_old_outputs_and_their_metadata(tmp_path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    candidate = StemMidiCandidate(source, tmp_path / "out", "vocals", source)
    common = dict(candidates=[candidate], base_options=TranscriptionOptions(),
                  device_resolver=lambda _: "cpu", model_loader=lambda _: object())
    first = transcribe_stem_candidates(**common, transcriber=successful_transcriber)
    paths = first.items[0]
    old_metadata = paths.metadata_path.read_bytes()
    result = transcribe_stem_candidates(**common, overwrite=True, transcriber=failed_transcriber)
    assert result.failed == 1
    assert paths.midi_path.read_bytes() == b"old success"
    assert paths.metadata_path.read_bytes() == old_metadata


def completed_image_outputs(item_dir):
    for name in ["src_img.png", "layerdiff/manifest.json", "depth/depth.png", "optimized/manifest.json", "final.psd"]:
        path = item_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old success")


def test_see_through_unverified_source_restarts_without_destroying_old_outputs(tmp_path):
    source = tmp_path / "art.png"
    source.write_bytes(b"new source")
    output = tmp_path / "out"
    completed_image_outputs(output / "art.png")
    config = SimpleNamespace(input_dir=tmp_path, skip_completed=True, save_to_psd=True)
    plan = runner.build_execution_plan(config, output, [source])
    assert plan[0].resume_stage == "layerdiff"
    assert (output / "art.png" / "final.psd").read_bytes() == b"old success"
    assert runner._select_auto_rig_items(plan) == []


def fail_one_metadata_publish(monkeypatch, destination):
    original = manifest.os.replace
    failed = False

    def replace(source, target):
        nonlocal failed
        if not failed and Path(target) == destination and Path(source).parent.name.startswith(".attempt-"):
            failed = True
            raise OSError("simulated metadata commit failure")
        return original(source, target)

    monkeypatch.setattr(manifest.os, "replace", replace)


def test_batch_publish_failure_rolls_back_outputs_and_reports_failure(tmp_path, monkeypatch):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    common = dict(input_path=source, output_dir=tmp_path / "out", package_version="test",
                  resolved_device="cpu", model_loader=lambda _: object())
    run_batch(**common, transcriber=successful_transcriber)
    item_dir = tmp_path / "out" / "song.wav"
    old_metadata = (item_dir / "metadata.json").read_bytes()
    fail_one_metadata_publish(monkeypatch, item_dir / "metadata.json")
    summary = run_batch(**common, options=BatchOptions(skip_completed=False), transcriber=replacement_transcriber)
    assert summary.failed == 1
    assert summary.processed == 0
    assert (item_dir / "song.mid").read_bytes() == b"old success"
    assert (item_dir / "metadata.json").read_bytes() == old_metadata


def test_stem_publish_failure_rolls_back_outputs_and_reports_failure(tmp_path, monkeypatch):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    candidate = StemMidiCandidate(source, tmp_path / "out", "vocals", source)
    common = dict(candidates=[candidate], base_options=TranscriptionOptions(),
                  device_resolver=lambda _: "cpu", model_loader=lambda _: object())
    first = transcribe_stem_candidates(**common, transcriber=successful_transcriber)
    paths = first.items[0]
    old_metadata = paths.metadata_path.read_bytes()
    fail_one_metadata_publish(monkeypatch, paths.metadata_path)
    summary = transcribe_stem_candidates(**common, overwrite=True, transcriber=replacement_transcriber)
    assert summary.failed == 1
    assert summary.processed == 0
    assert paths.midi_path.read_bytes() == b"old success"
    assert paths.metadata_path.read_bytes() == old_metadata


def test_batch_staging_metadata_failure_keeps_old_generation(tmp_path, monkeypatch):
    import module.muscriptor_tool.batch as batch_module

    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    common = dict(input_path=source, output_dir=tmp_path / "out", package_version="test",
                  resolved_device="cpu", model_loader=lambda _: object())
    run_batch(**common, transcriber=successful_transcriber)
    item_dir = tmp_path / "out" / "song.wav"
    old_metadata = (item_dir / "metadata.json").read_bytes()
    original = batch_module.atomic_write_json

    def write_json(path, payload):
        if Path(path).parent.name.startswith(".attempt-"):
            raise OSError("metadata staging failed")
        return original(path, payload)

    monkeypatch.setattr(batch_module, "atomic_write_json", write_json)
    summary = run_batch(**common, options=BatchOptions(skip_completed=False), transcriber=replacement_transcriber)
    assert summary.failed == 1
    assert summary.processed == 0
    assert (item_dir / "metadata.json").read_bytes() == old_metadata
    assert (item_dir / "song.mid").read_bytes() == b"old success"


def test_see_through_commits_fingerprint_only_after_successful_postprocess(tmp_path):
    source = tmp_path / "art.png"
    source.write_bytes(b"first source")
    output = tmp_path / "out"
    completed_image_outputs(output / "art.png")
    config = SimpleNamespace(input_dir=tmp_path, skip_completed=True, save_to_psd=True)
    item = runner.build_execution_plan(config, output, [source])[0]
    runner._process_phase_items(phase_name="postprocess", items=[item], handler=lambda _: None, continue_on_error=True)
    assert runner.build_execution_plan(config, output, [source])[0].resume_stage == "completed"
    source.write_bytes(b"other source")
    new_item = runner.build_execution_plan(config, output, [source])[0]
    assert new_item.resume_stage == "layerdiff"

    def fail(_):
        raise RuntimeError("processing failed")

    runner._process_phase_items(phase_name="postprocess", items=[new_item], handler=fail, continue_on_error=True)
    assert runner.build_execution_plan(config, output, [source])[0].resume_stage == "layerdiff"
    assert runner._select_auto_rig_items([new_item]) == []
    assert (item.item_dir / "final.psd").read_bytes() == b"old success"


def test_see_through_failed_same_source_rerun_invalidates_completed_marker(tmp_path):
    source = tmp_path / "art.png"
    source.write_bytes(b"same source")
    output = tmp_path / "out"
    item_dir = output / "art.png"
    completed_image_outputs(item_dir)
    force_config = SimpleNamespace(input_dir=tmp_path, skip_completed=False, save_to_psd=True)
    item = runner.build_execution_plan(force_config, output, [source])[0]
    runner._commit_item_source(item)

    def partially_overwrite_then_fail(_):
        (item_dir / "src_img.png").write_bytes(b"new partial output")
        raise RuntimeError("processing failed")

    runner._process_phase_items(
        phase_name="layerdiff",
        items=[item],
        handler=partially_overwrite_then_fail,
        continue_on_error=True,
        console_obj=Console(file=StringIO(), force_terminal=False),
    )

    resume_config = SimpleNamespace(input_dir=tmp_path, skip_completed=True, save_to_psd=True)
    assert runner.build_execution_plan(resume_config, output, [source])[0].resume_stage == "layerdiff"
    assert (item_dir / "final.psd").read_bytes() == b"old success"


def test_translation_user_edit_is_preserved_and_not_resumed(tmp_path):
    path, version = dataset_with_documents(tmp_path, ["report.txt"], ["first"])
    translator = RecordingTranslator()
    translate(path, version, translator, tmp_path)
    output = tmp_path / "report.txt_zh_cn.md"
    output.write_text("edited by user", encoding="utf-8")
    translate(path, version, translator, tmp_path)
    translate(path, version, translator, tmp_path)
    assert translator.calls == ["first", "first"]
    assert output.read_text(encoding="utf-8") == "edited by user"
