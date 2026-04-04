from __future__ import annotations

from typing import Any, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def resolve_rich_console(console: Optional[Any] = None) -> Console:
    if isinstance(console, Console):
        return console

    try:
        from utils.console_util import console as shared_console

        return shared_console
    except Exception:
        return Console(color_system="truecolor", force_terminal=True)


def create_caption_progress(
    console: Optional[Any] = None,
    *,
    transient: bool = False,
    expand: bool = True,
) -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        MofNCompleteColumn(separator="/"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("•"),
        TaskProgressColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=resolve_rich_console(console),
        transient=transient,
        expand=expand,
    )


def create_download_progress(
    console: Optional[Any] = None,
    *,
    transient: bool = False,
    expand: bool = True,
) -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("•"),
        TaskProgressColumn(),
        TextColumn("•"),
        DownloadColumn(binary_units=True),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=resolve_rich_console(console),
        transient=transient,
        expand=expand,
    )
