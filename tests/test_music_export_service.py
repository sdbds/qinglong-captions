from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_atomic_output_path_preserves_suffix_and_replaces_target(tmp_path: Path):
    from module.music_export.service import atomic_output_path

    target = tmp_path / "score.musicxml"
    target.write_text("old", encoding="utf-8")

    with atomic_output_path(target) as temporary:
        assert temporary.parent == target.parent
        assert temporary.suffix == ".musicxml"
        assert temporary != target
        temporary.write_text("new", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob("*.part*")) == []


def test_failed_validation_preserves_existing_target_and_continues(tmp_path: Path):
    from module.music_export.service import ExportJob, run_export_jobs

    invalid_target = tmp_path / "score.musicxml"
    valid_target = tmp_path / "score.mid"
    invalid_target.write_text("previous", encoding="utf-8")

    def reject(_path: Path) -> None:
        raise ValueError("invalid score")

    statuses = run_export_jobs(
        (
            ExportJob(
                format="musicxml",
                target=invalid_target,
                writer=lambda path: path.write_text("broken", encoding="utf-8"),
                validator=reject,
            ),
            ExportJob(
                format="midi",
                target=valid_target,
                writer=lambda path: path.write_bytes(b"MThd"),
                validator=lambda path: None,
            ),
        )
    )

    assert invalid_target.read_text(encoding="utf-8") == "previous"
    assert valid_target.read_bytes() == b"MThd"
    assert [(status.format, status.ok) for status in statuses] == [
        ("musicxml", False),
        ("midi", True),
    ]
    assert "ValueError: invalid score" == statuses[0].error
    assert list(tmp_path.glob("*.part*")) == []


def test_export_status_preserves_writer_warning(tmp_path: Path):
    from module.music_export.service import ExportJob, run_export_jobs

    target = tmp_path / "score.mid"
    statuses = run_export_jobs(
        (
            ExportJob(
                format="midi",
                target=target,
                writer=lambda path: (
                    path.write_bytes(b"MThd"),
                    "linearized malformed repeats",
                )[1],
                validator=lambda path: None,
            ),
        )
    )

    assert statuses[0].ok is True
    assert statuses[0].warning == "linearized malformed repeats"


def test_music_export_package_does_not_eagerly_import_music21():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import module.music_export; "
                "assert 'music21' not in sys.modules; "
                "from module.music_export import ExportJob, run_export_jobs"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
