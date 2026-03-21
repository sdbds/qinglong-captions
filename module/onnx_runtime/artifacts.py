"""ONNX artifact discovery and download helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


def _emit_log(logger: Callable[..., Any] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _maybe_enable_hf_progress_bars() -> None:
    try:
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()
    except Exception:
        # Progress bar support is best-effort only; downloads still work without it.
        return


def _normalize_variant(variant: str) -> str:
    value = str(variant or "").strip().lower()
    if value in {"", "fp32", "float32", "base", "default"}:
        return ""
    return value


def build_component_filename(component: str, variant: str = "") -> str:
    normalized_variant = _normalize_variant(variant)
    suffix = f"_{normalized_variant}" if normalized_variant else ""
    return f"onnx/{component}{suffix}.onnx"


def collect_external_data_files(repo_files: Iterable[str], onnx_filename: str) -> tuple[str, ...]:
    data_prefix = f"{onnx_filename}_data"
    matches = sorted(
        file_name
        for file_name in repo_files
        if file_name == data_prefix or file_name.startswith(f"{data_prefix}_")
    )
    return tuple(matches)


def list_required_artifact_files(repo_files: Iterable[str], onnx_filename: str) -> tuple[str, ...]:
    return (onnx_filename, *collect_external_data_files(repo_files, onnx_filename))


def download_onnx_artifact(
    repo_id: str,
    onnx_filename: str,
    *,
    local_dir: str | Path | None = None,
    force_download: bool = False,
    repo_files: Iterable[str] | None = None,
    downloader: Callable[..., str] | None = None,
    repo_file_lister: Callable[[str], Iterable[str]] | None = None,
    logger: Callable[..., Any] | None = None,
) -> Path:
    if repo_files is None:
        if repo_file_lister is None:
            from huggingface_hub import list_repo_files

            repo_file_lister = list_repo_files
        repo_files = tuple(repo_file_lister(repo_id))
    else:
        repo_files = tuple(repo_files)

    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    download_dir = None if local_dir is None else str(Path(local_dir))
    downloaded_model: Path | None = None

    for file_name in list_required_artifact_files(repo_files, onnx_filename):
        existing_target = Path(download_dir) / file_name if download_dir is not None else None
        if existing_target is not None and existing_target.exists() and not force_download:
            target = existing_target
            _emit_log(logger, f"[green]Using existing ONNX artifact[/green] {target}")
        else:
            _emit_log(logger, f"[cyan]Downloading ONNX artifact[/cyan] {repo_id}:{file_name}")
            _maybe_enable_hf_progress_bars()
            target = Path(
                downloader(
                    repo_id=repo_id,
                    filename=file_name,
                    local_dir=download_dir,
                    force_download=force_download,
                )
            )
            _emit_log(logger, f"[green]Downloaded ONNX artifact[/green] {target}")
        if file_name == onnx_filename:
            downloaded_model = target

    if downloaded_model is None:
        raise FileNotFoundError(f"Failed to download ONNX artifact: {onnx_filename}")

    return downloaded_model


def download_onnx_artifact_set(
    repo_id: str,
    artifacts: Mapping[str, str],
    *,
    local_dir: str | Path | None = None,
    force_download: bool = False,
    repo_files: Iterable[str] | None = None,
    downloader: Callable[..., str] | None = None,
    repo_file_lister: Callable[[str], Iterable[str]] | None = None,
    logger: Callable[..., Any] | None = None,
) -> dict[str, Path]:
    if repo_files is None:
        if repo_file_lister is None:
            from huggingface_hub import list_repo_files

            repo_file_lister = list_repo_files
        repo_files = tuple(repo_file_lister(repo_id))
    else:
        repo_files = tuple(repo_files)
    return {
        name: download_onnx_artifact(
            repo_id,
            onnx_filename,
            local_dir=local_dir,
            force_download=force_download,
            repo_files=repo_files,
            downloader=downloader,
            repo_file_lister=repo_file_lister,
            logger=logger,
        )
        for name, onnx_filename in artifacts.items()
    }
