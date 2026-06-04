from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional


SIDECAR_CAPTION_EXTENSIONS = (".txt", ".md", ".srt")


def resolve_lance_rebuild_source(train_data_dir: Any, dataset_path: Optional[Path] = None) -> Optional[Path]:
    if not isinstance(train_data_dir, str):
        return None

    input_path = Path(train_data_dir)
    if input_path.suffix == ".lance":
        return input_path.parent
    if input_path.is_dir():
        return input_path
    if dataset_path is not None:
        return dataset_path.parent
    return None


def _read_dataset_uri_rebuild_data(
    dataset: Any,
    *,
    caption_extension: Optional[str],
    read_sidecar_caption_fn: Optional[Callable[[str, str], list[str]]],
) -> list[dict[str, Any]]:
    if dataset is None:
        return []

    data: list[dict[str, Any]] = []
    scanner = dataset.scanner(
        columns=["uris"],
        scan_in_order=True,
        batch_size=1024,
        late_materialization=False,
    )
    for batch in scanner.to_batches():
        for uri in batch["uris"].to_pylist():
            if caption_extension is None or read_sidecar_caption_fn is None:
                caption = read_detected_sidecar_caption(str(uri))
            else:
                caption = read_sidecar_caption_fn(str(uri), caption_extension)
            data.append(
                {
                    "file_path": str(uri),
                    "caption": caption,
                    "chunk_offsets": [],
                }
            )
    return data


def read_detected_sidecar_caption(uri: str) -> list[str]:
    sidecar_base = Path(uri).with_suffix("")
    for extension in SIDECAR_CAPTION_EXTENSIONS:
        caption_path = sidecar_base.with_suffix(extension)
        if not caption_path.exists() or caption_path == Path(uri):
            continue
        try:
            content = caption_path.read_text(encoding="utf-8")
        except OSError:
            continue
        return content.splitlines() if extension == ".txt" else [content]
    return []


def load_lance_rebuild_data(
    source_dir: Optional[Path],
    dataset: Any,
    *,
    load_data_fn: Callable[[str], list[dict[str, Any]]],
    read_sidecar_caption_fn: Optional[Callable[[str, str], list[str]]] = None,
    caption_extension: Optional[str] = None,
) -> list[dict[str, Any]]:
    if source_dir is not None:
        data = load_data_fn(str(source_dir))
        if data:
            return data

    return _read_dataset_uri_rebuild_data(
        dataset,
        caption_extension=caption_extension,
        read_sidecar_caption_fn=read_sidecar_caption_fn,
    )


def rebuild_lance_from_sidecars(
    source_dir: Optional[Path],
    *,
    output_name: str,
    dataset: Any,
    tag: str,
    transform2lance_fn: Callable[..., Any],
    load_data_fn: Callable[[str], list[dict[str, Any]]],
    console: Any,
    caption_extension: Optional[str] = None,
    read_sidecar_caption_fn: Optional[Callable[[str, str], list[str]]] = None,
) -> Optional[Any]:
    if source_dir is None:
        console.print("[yellow]Skipping Lance rebuild: source directory is unavailable.[/yellow]")
        return None

    data = load_lance_rebuild_data(
        source_dir,
        dataset,
        load_data_fn=load_data_fn,
        read_sidecar_caption_fn=read_sidecar_caption_fn,
        caption_extension=caption_extension,
    )
    if not data:
        console.print("[yellow]Skipping Lance rebuild: no source media rows were found.[/yellow]")
        return None

    console.print("[yellow]Rebuilding Lance dataset from sidecar caption files...[/yellow]")
    rebuilt_dataset = transform2lance_fn(
        str(source_dir),
        output_name=output_name,
        save_binary=False,
        not_save_disk=False,
        tag=tag,
        load_condition=lambda *_args, **_kwargs: data,
    )
    if rebuilt_dataset is None:
        console.print("[yellow]Lance rebuild did not return a dataset; sidecar captions were still written.[/yellow]")
    else:
        console.print("[green]Lance dataset rebuilt from sidecar caption files[/green]")
    return rebuilt_dataset
