import io
import sys
from pathlib import Path

from rich.console import Console


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def _raise_runtime_error():
    raise RuntimeError("boom")


def test_print_exception_is_exposed():
    import utils.console_util as console_util

    assert hasattr(console_util, "print_exception")


def test_print_exception_includes_prefix_type_message_and_stack():
    import utils.console_util as console_util

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)

    try:
        _raise_runtime_error()
    except RuntimeError as exc:
        console_util.print_exception(console, exc, prefix="provider failed")

    output = buf.getvalue()
    assert "provider failed" in output
    assert "RuntimeError: boom" in output
    assert "Traceback" in output
    assert "_raise_runtime_error" in output


def test_print_exception_handles_exception_without_traceback():
    import utils.console_util as console_util

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)

    exc = RuntimeError("detached")
    console_util.print_exception(console, exc, prefix="detached error")

    output = buf.getvalue()
    assert "detached error" in output
    assert "RuntimeError: detached" in output


def test_plain_console_output_is_unchanged():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)

    console.print("plain text")

    assert buf.getvalue().strip() == "plain text"
