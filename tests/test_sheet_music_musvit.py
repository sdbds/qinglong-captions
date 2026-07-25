import argparse
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console


def test_parser_defaults_to_pinned_omr_model_and_symbolic_output():
    from module.sheet_music_musvit import (
        DEFAULT_MODEL_REPO_ID,
        DEFAULT_MODEL_REVISION,
        build_parser,
    )

    args = build_parser(
        {
            "musvit": {
                "repo_id": DEFAULT_MODEL_REPO_ID,
                "revision": DEFAULT_MODEL_REVISION,
                "model_dir": "huggingface",
                "output_dir": "workspace/musvit_omr_output",
                "output_format": "musicxml",
                "pdf_dpi": 144,
                "recursive": True,
                "skip_completed": True,
                "overwrite": False,
                "force_download": False,
            }
        }
    ).parse_args(["input.png"])

    assert args.repo_id == DEFAULT_MODEL_REPO_ID
    assert args.revision == DEFAULT_MODEL_REVISION
    assert args.output_dir is None
    assert args.output_format == "musicxml"
    assert args.pdf_dpi == 144
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "preprocess_mode")


def test_default_output_directory_is_input_local(tmp_path: Path):
    from module.sheet_music_musvit import default_output_dir

    source_file = tmp_path / "score.pdf"
    source_file.write_bytes(b"%PDF")
    source_dir = tmp_path / "scores"
    source_dir.mkdir()

    assert default_output_dir(source_file) == tmp_path / "musvit_omr_output"
    assert default_output_dir(source_dir) == source_dir / "musvit_omr_output"


def test_run_cli_resolves_omitted_output_directory_next_to_input(tmp_path: Path):
    from module.sheet_music_musvit import run_sheet_music_musvit

    input_path = tmp_path / "score.png"
    Image.new("RGB", (2, 2), color="white").save(input_path)
    captured = {}

    class FakeRecognizer:
        providers = ("CPUExecutionProvider",)

        def __init__(self, **kwargs):
            pass

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @staticmethod
        def run(path):
            return SimpleNamespace(
                ok=True,
                manifest_path=tmp_path / "musvit_omr_output" / "manifest.json",
                pages=(SimpleNamespace(skipped=False, ok=True),),
                documents=(),
                source_failures={},
                processed=1,
                skipped=0,
                failed=0,
            )

    args = argparse.Namespace(
        input_path=str(input_path),
        output_dir=None,
        output_format="musicxml",
        repo_id="custom/repo",
        revision="custom-revision",
        model_dir=str(tmp_path / "models"),
        pdf_dpi=144,
        recursive=True,
        skip_completed=True,
        overwrite=False,
        force_download=False,
        max_tokens=None,
    )

    assert (
        run_sheet_music_musvit(
            args,
            recognizer_factory=FakeRecognizer,
            pipeline_factory=FakePipeline,
        )
        == 0
    )
    assert captured["output_dir"] == tmp_path / "musvit_omr_output"


def test_run_cli_forwards_omr_options_and_returns_pipeline_status(
    monkeypatch,
    tmp_path: Path,
):
    import module.sheet_music_musvit as sheet_music_musvit

    input_path = tmp_path / "score.png"
    Image.new("RGB", (2, 2), color="white").save(input_path)
    output_dir = tmp_path / "out"
    captured = {}
    output_bytes = io.BytesIO()
    output_stream = io.TextIOWrapper(output_bytes, encoding="gbk")
    monkeypatch.setattr(
        sheet_music_musvit,
        "console",
        Console(
            file=output_stream,
            color_system="truecolor",
            force_terminal=True,
        ),
    )

    class FakeRecognizer:
        providers = ("CPUExecutionProvider",)

        def __init__(self, **kwargs):
            captured["recognizer"] = kwargs

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

        def run(self, path):
            captured["input_path"] = Path(path)
            return SimpleNamespace(
                ok=True,
                manifest_path=output_dir / "manifest.json",
                pages=(SimpleNamespace(skipped=False, ok=True),),
                documents=(),
                source_failures={},
                processed=1,
                skipped=0,
                failed=0,
            )

    args = argparse.Namespace(
        input_path=str(input_path),
        output_dir=str(output_dir),
        output_format="both",
        repo_id="custom/repo",
        revision="custom-revision",
        model_dir=str(tmp_path / "models"),
        pdf_dpi=200,
        recursive=False,
        skip_completed=False,
        overwrite=True,
        force_download=True,
        max_tokens=123,
    )

    exit_code = sheet_music_musvit.run_sheet_music_musvit(
        args,
        recognizer_factory=FakeRecognizer,
        pipeline_factory=FakePipeline,
    )
    output_stream.flush()

    assert exit_code == 0
    assert captured["recognizer"]["repo_id"] == "custom/repo"
    assert captured["recognizer"]["revision"] == "custom-revision"
    assert captured["pipeline"]["output_format"] == "both"
    assert captured["pipeline"]["pdf_dpi"] == 200
    assert captured["pipeline"]["max_tokens"] == 123
    assert captured["input_path"] == input_path


def test_run_cli_escapes_manifest_path_outside_console_encoding(
    monkeypatch,
    tmp_path: Path,
):
    import module.sheet_music_musvit as sheet_music_musvit

    input_path = tmp_path / "score.png"
    Image.new("RGB", (2, 2), color="white").save(input_path)
    output_bytes = io.BytesIO()
    output_stream = io.TextIOWrapper(output_bytes, encoding="gbk")
    monkeypatch.setattr(
        sheet_music_musvit,
        "console",
        Console(
            file=output_stream,
            color_system=None,
            force_terminal=False,
        ),
    )

    class FakeRecognizer:
        providers = ("CPUExecutionProvider",)

        def __init__(self, **kwargs):
            pass

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def run(path):
            return SimpleNamespace(
                ok=True,
                manifest_path=tmp_path / "manifest_\u3099.json",
                pages=(SimpleNamespace(skipped=True, ok=True),),
                documents=(),
                source_failures={},
                processed=0,
                skipped=1,
                failed=0,
            )

    args = argparse.Namespace(
        input_path=str(input_path),
        output_dir="",
        output_format="musicxml",
        repo_id="custom/repo",
        revision="custom-revision",
        model_dir=str(tmp_path / "models"),
        pdf_dpi=144,
        recursive=True,
        skip_completed=True,
        overwrite=False,
        force_download=False,
        max_tokens=None,
    )

    exit_code = sheet_music_musvit.run_sheet_music_musvit(
        args,
        recognizer_factory=FakeRecognizer,
        pipeline_factory=FakePipeline,
    )
    output_stream.flush()

    assert exit_code == 0
    assert b"manifest_" in output_bytes.getvalue()


def test_run_cli_returns_nonzero_when_any_page_fails(tmp_path: Path):
    from module.sheet_music_musvit import run_sheet_music_musvit

    input_path = tmp_path / "score.png"
    Image.new("RGB", (2, 2), color="white").save(input_path)

    class FakeRecognizer:
        providers = ("CPUExecutionProvider",)

        def __init__(self, **kwargs):
            pass

    class FailingPipeline:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def run(path):
            return SimpleNamespace(
                ok=False,
                manifest_path=tmp_path / "manifest.json",
                pages=(SimpleNamespace(skipped=False, ok=False),),
                documents=(),
                source_failures={},
                processed=1,
                skipped=0,
                failed=1,
            )

    args = argparse.Namespace(
        input_path=str(input_path),
        output_dir=str(tmp_path / "out"),
        output_format="musicxml",
        repo_id="custom/repo",
        revision="custom-revision",
        model_dir=str(tmp_path / "models"),
        pdf_dpi=144,
        recursive=True,
        skip_completed=True,
        overwrite=False,
        force_download=False,
        max_tokens=None,
    )

    assert (
        run_sheet_music_musvit(
            args,
            recognizer_factory=FakeRecognizer,
            pipeline_factory=FailingPipeline,
        )
        == 1
    )


def test_run_cli_reports_document_only_aggregation_failure(
    monkeypatch,
    tmp_path: Path,
):
    import module.sheet_music_musvit as sheet_music_musvit

    input_path = tmp_path / "score.pdf"
    input_path.write_bytes(b"%PDF")
    output_bytes = io.BytesIO()
    output_stream = io.TextIOWrapper(output_bytes, encoding="gbk")
    monkeypatch.setattr(
        sheet_music_musvit,
        "console",
        Console(
            file=output_stream,
            color_system=None,
            force_terminal=False,
        ),
    )

    class FakeRecognizer:
        providers = ("CPUExecutionProvider",)

        def __init__(self, **kwargs):
            pass

    class FailingPipeline:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def run(path):
            return SimpleNamespace(
                ok=False,
                manifest_path=tmp_path / "manifest.json",
                pages=(SimpleNamespace(skipped=False, ok=True),),
                documents=(SimpleNamespace(ok=False),),
                source_failures={},
                processed=1,
                skipped=0,
                failed=1,
            )

    args = argparse.Namespace(
        input_path=str(input_path),
        output_dir=str(tmp_path / "out"),
        output_format="both",
        repo_id="custom/repo",
        revision="custom-revision",
        model_dir=str(tmp_path / "models"),
        pdf_dpi=144,
        recursive=True,
        skip_completed=True,
        overwrite=False,
        force_download=False,
        max_tokens=None,
    )

    assert (
        sheet_music_musvit.run_sheet_music_musvit(
            args,
            recognizer_factory=FakeRecognizer,
            pipeline_factory=FailingPipeline,
        )
        == 1
    )
    output_stream.flush()
    assert "processed=1 skipped=0 failed=1" in output_bytes.getvalue().decode(
        "gbk"
    )


@pytest.mark.parametrize(
    ("pdf_dpi", "max_tokens"),
    (
        (0, None),
        (144, 1),
        (144, 7513),
    ),
)
def test_invalid_runtime_args_fail_before_loading_model(
    tmp_path: Path,
    pdf_dpi: int,
    max_tokens: int | None,
):
    from module.sheet_music_musvit import run_sheet_music_musvit

    input_path = tmp_path / "score.png"
    Image.new("RGB", (2, 2), color="white").save(input_path)
    recognizer_calls = []

    class MustNotLoadRecognizer:
        def __init__(self, **kwargs):
            recognizer_calls.append(kwargs)

    args = argparse.Namespace(
        input_path=str(input_path),
        output_dir=str(tmp_path / "out"),
        output_format="musicxml",
        repo_id="custom/repo",
        revision="custom-revision",
        model_dir=str(tmp_path / "models"),
        pdf_dpi=pdf_dpi,
        recursive=True,
        skip_completed=True,
        overwrite=False,
        force_download=False,
        max_tokens=max_tokens,
    )

    assert (
        run_sheet_music_musvit(
            args,
            recognizer_factory=MustNotLoadRecognizer,
        )
        == 1
    )
    assert recognizer_calls == []


def test_pipeline_construction_errors_are_reported_as_cli_failures(tmp_path: Path):
    from module.sheet_music_musvit import run_sheet_music_musvit

    input_path = tmp_path / "score.png"
    Image.new("RGB", (2, 2), color="white").save(input_path)

    class FakeRecognizer:
        providers = ("CPUExecutionProvider",)

        def __init__(self, **kwargs):
            pass

    class FailingPipeline:
        def __init__(self, **kwargs):
            raise ValueError("invalid pipeline options")

    args = argparse.Namespace(
        input_path=str(input_path),
        output_dir=str(tmp_path / "out"),
        output_format="musicxml",
        repo_id="custom/repo",
        revision="custom-revision",
        model_dir=str(tmp_path / "models"),
        pdf_dpi=144,
        recursive=True,
        skip_completed=True,
        overwrite=False,
        force_download=False,
        max_tokens=None,
    )

    assert (
        run_sheet_music_musvit(
            args,
            recognizer_factory=FakeRecognizer,
            pipeline_factory=FailingPipeline,
        )
        == 1
    )
