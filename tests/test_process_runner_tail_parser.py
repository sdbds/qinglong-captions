import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.utils.process_runner import ProcessRunner


def create_runner() -> ProcessRunner:
    runner = ProcessRunner()
    runner._reset_tail_state()
    return runner


def test_plain_newlines_are_forwarded():
    runner = create_runner()

    lines = runner._consume_native_log_chunk("alpha\nbeta\n")

    assert lines == ["alpha", "beta"]


def test_spinner_frames_with_carriage_returns_are_ignored():
    runner = create_runner()

    chunk = "\x1b[2K\r⠋ syncing\r\x1b[2K\r⠙ syncing"

    assert runner._consume_native_log_chunk(chunk) == []
    assert runner._consume_native_log_chunk("", final_flush=True) == []


def test_overwritten_spinner_can_yield_final_real_log_line():
    runner = create_runner()

    chunk = "\r⠋ syncing\r\x1b[2K\r完成\n"

    assert runner._consume_native_log_chunk(chunk) == ["完成"]


def test_crlf_split_across_chunks_keeps_real_line():
    runner = create_runner()

    assert runner._consume_native_log_chunk("stable line\r") == []
    assert runner._consume_native_log_chunk("\nnext line\n") == ["stable line", "next line"]


def test_unterminated_plain_line_flushes_on_finalize():
    runner = create_runner()

    assert runner._consume_native_log_chunk("tail line") == []
    assert runner._consume_native_log_chunk("", final_flush=True) == ["tail line"]
