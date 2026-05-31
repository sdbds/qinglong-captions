from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import pysrt
from rich.console import Console
from rich.progress import (
    Progress,
)

from config.config import APPLICATION_EXTENSIONS_SET, AUDIO_EXTENSIONS_SET, VIDEO_EXTENSIONS_SET
from module.caption_pipeline.dataset_sync import update_dataset_captions
from module.caption_pipeline.postprocess import postprocess_caption_content
from module.caption_pipeline.scene_alignment import align_subtitles_with_scenes, create_scene_detector
from utils.output_writer import write_caption_output
from utils.rich_progress import create_caption_progress
from utils.stream_util import (
    get_video_duration,
    split_media_stream_clips,
    split_video_with_imageio_ffmpeg,
)


@dataclass(frozen=True)
class CaptionJob:
    index: int
    filepath: str
    mime: str
    duration: int
    sha256hash: str


@dataclass
class CaptionJobResult:
    index: int
    filepath: str
    mime: str
    output: Any
    log_text: str = ""


def _resolve_dataset(args, transform2lance_fn):
    dataset_type = getattr(lance, "LanceDataset", None)
    if dataset_type is not None and isinstance(args.dataset_dir, dataset_type):
        return args.dataset_dir

    if args.gemini_api_key == "" and args.mistral_api_key == "":
        return transform2lance_fn(dataset_dir=args.dataset_dir)
    return transform2lance_fn(dataset_dir=args.dataset_dir, save_binary=False)


def _serialize_subtitles(subs: pysrt.SubRipFile) -> str:
    return "".join(str(sub) for sub in subs)


def _structured_description(payload: dict) -> str:
    task_kind = str(payload.get("task_kind") or "").strip().lower()
    if task_kind == "ast":
        return str(payload.get("translation_srt") or payload.get("transcript") or "").strip()
    return str(
        payload.get("long_description")
        or payload.get("transcript")
        or payload.get("description")
        or payload.get("short_description")
        or ""
    ).strip()


def _format_segment_timestamp(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _build_segment_summary_payload(segment_outputs: list[dict]) -> dict:
    description_parts = []
    for segment in segment_outputs:
        description_parts.append(
            f"Segment {segment['index']} [{_format_segment_timestamp(segment['start_seconds'])} - {_format_segment_timestamp(segment['end_seconds'])}]\n"
            f"{segment['description']}"
        )

    merged = {
        "description": "\n\n".join(description_parts).strip(),
        "caption_extension": ".txt",
        "segments": segment_outputs,
    }
    provider = next((segment.get("provider") for segment in segment_outputs if segment.get("provider")), None)
    if provider:
        merged["provider"] = provider
    return merged


def _build_segment_transcript_payload(segment_outputs: list[dict]) -> dict:
    transcript_parts = [segment["text"] for segment in segment_outputs if segment.get("text")]
    merged = {
        "task_kind": "transcribe",
        "transcript": "\n\n".join(transcript_parts).strip(),
        "caption_extension": ".txt",
        "segments": segment_outputs,
    }
    provider = next((segment.get("provider") for segment in segment_outputs if segment.get("provider")), None)
    if provider:
        merged["provider"] = provider
    return merged


def _build_segment_ast_payload(segment_outputs: list[dict]) -> dict:
    translation_parts = [str(segment["text"]).strip() for segment in segment_outputs if str(segment.get("text", "")).strip()]
    merged = {
        "task_kind": "ast",
        "translation_srt": "\n\n".join(translation_parts),
        "caption_extension": ".srt",
        "subtitle_format": "srt",
        "segments": segment_outputs,
    }
    provider = next((segment.get("provider") for segment in segment_outputs if segment.get("provider")), None)
    if provider:
        merged["provider"] = provider
    return merged


def _should_bypass_segmentation(args, mime: str) -> bool:
    if mime.startswith("video"):
        return getattr(args, "vlm_image_model", "") == "gemma4_local"
    if mime.startswith("audio"):
        return getattr(args, "alm_model", "") == "gemma4_local"
    return False


def _coerce_float(value, default: float) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _section(config, *names: str) -> dict:
    if not config or not hasattr(config, "get"):
        return {}
    for name in names:
        value = config.get(name)
        if (isinstance(value, dict) or hasattr(value, "get")) and value:
            return value
    return {}


def _marlin_video_max_seconds(config) -> float:
    section = _section(config, "marlin_2b_local", "marlin")
    return _coerce_float(section.get("video_max_seconds", 120.0), 120.0)


def _marlin_safe_segment_seconds(config) -> int | None:
    max_seconds = _marlin_video_max_seconds(config)
    if max_seconds <= 0:
        return None
    return max(1, int(max_seconds) - 1)


def _resolve_media_segment_time(args, mime: str, config):
    segment_time = getattr(args, "segment_time", None)
    if not mime.startswith("video") or getattr(args, "vlm_image_model", "") != "marlin_2b_local":
        return segment_time

    safe_segment_time = _marlin_safe_segment_seconds(config)
    if safe_segment_time is None:
        return segment_time
    if segment_time in (None, ""):
        return safe_segment_time
    try:
        requested_segment_time = int(segment_time)
    except (TypeError, ValueError):
        return safe_segment_time
    if requested_segment_time <= 0:
        return safe_segment_time
    return min(requested_segment_time, safe_segment_time)


def _direct_duration_limit_ms(args, mime: str, config) -> int | None:
    if mime.startswith("video") and getattr(args, "vlm_image_model", "") == "marlin_2b_local":
        max_seconds = _marlin_video_max_seconds(config)
        if max_seconds > 0:
            return int(max_seconds * 1000)
    return None


def _should_process_without_segmentation(args, mime: str, duration: int, segment_time, config) -> bool:
    if mime.startswith("image") or _should_bypass_segmentation(args, mime) or segment_time is None:
        return True

    direct_limit = _direct_duration_limit_ms(args, mime, config)
    if direct_limit is not None:
        return duration <= direct_limit

    return duration <= (segment_time + 1) * 1000


def _with_media_segment_time(args, segment_time):
    if getattr(args, "segment_time", None) == segment_time:
        return args
    media_args = copy(args)
    setattr(media_args, "segment_time", segment_time)
    setattr(media_args, "effective_segment_time", segment_time)
    return media_args


def _with_directory_name_source_uri(args, source_uri):
    source_uri = str(source_uri)
    if getattr(args, "directory_name_source_uri", "") == source_uri:
        return args
    media_args = copy(args)
    setattr(media_args, "directory_name_source_uri", source_uri)
    return media_args


def _process_segmented_media(filepath, mime, duration, sha256hash, args, config, progress, task_id, api_process_batch_fn, console):
    console.print(f"[blue]{filepath} video > {args.segment_time} seconds[/blue]")
    console.print("[blue]split video[/blue]")

    subs = pysrt.SubRipFile()
    duration_seconds = duration / 1000
    chunk_duration = args.segment_time
    num_chunks = int((duration_seconds + chunk_duration - 1) // chunk_duration)

    for index in range(num_chunks):
        start_time = index * chunk_duration
        end_time = min((index + 1) * chunk_duration, duration_seconds)
        subs.append(
            pysrt.SubRipItem(
                index=index,
                start=pysrt.SubRipTime(seconds=start_time),
                end=pysrt.SubRipTime(seconds=end_time),
                text=f"Chunk {index + 1}",
            )
        )

    try:
        split_video_with_imageio_ffmpeg(
            Path(filepath),
            subs,
            save_caption_func=None,
            segment_time=args.segment_time,
        )
    except Exception as exc:
        meta_type = "video" if mime.startswith("video") else "audio"
        console.print(f"[red]Error splitting video with imageio-ffmpeg: {exc}[/red]")
        split_media_stream_clips(Path(filepath), meta_type, subs)

    pathfile = Path(filepath)
    clip_dir = pathfile.parent / f"{pathfile.stem}_clip"
    files = sorted(clip_dir.glob(f"*{pathfile.suffix}"))

    merged_subs = pysrt.SubRipFile()
    segment_outputs: list[dict] = []
    subtitle_mode: bool | None = None
    structured_task_kind: str | None = None
    total_duration = 0
    clip_task = progress.add_task("[cyan]Processing clips...", total=num_chunks)
    clip_args = _with_directory_name_source_uri(args, filepath)

    for index in range(num_chunks):
        start_time = index * chunk_duration
        end_time = min((index + 1) * chunk_duration, duration_seconds)
        uri = files[index]
        chunk_output = api_process_batch_fn(
            uri=uri,
            mime=mime,
            config=config,
            args=clip_args,
            sha256hash=sha256hash,
            progress=progress,
            task_id=task_id,
        )
        chunk_output = postprocess_caption_content(chunk_output, uri, clip_args, console)

        if isinstance(chunk_output, dict):
            if subtitle_mode is None:
                subtitle_mode = False
            task_kind = str(chunk_output.get("task_kind") or "caption").strip().lower()
            structured_task_kind = structured_task_kind or task_kind
            if task_kind == "ast":
                text = str(chunk_output.get("translation_srt") or chunk_output.get("transcript") or "")
            else:
                text = _structured_description(chunk_output)
            if str(text).strip():
                segment_outputs.append(
                    {
                        "index": index + 1,
                        "start_seconds": start_time,
                        "end_seconds": end_time,
                        "text": text,
                        "provider": chunk_output.get("provider"),
                    }
                )
            progress.update(clip_task, advance=1, refresh=True, description="[yellow]merged summary chunk[/yellow]")
            continue

        if subtitle_mode is None:
            subtitle_mode = True
            sub_path = Path(filepath).with_suffix(".srt")
            if sub_path.exists():
                merged_subs.extend(pysrt.open(sub_path, encoding="utf-8"))

        chunk_subs = pysrt.from_string(chunk_output)

        for sub in list(chunk_subs):
            if sub.start.ordinal > args.segment_time * 1000:
                chunk_subs.remove(sub)

        if index > 0:
            total_duration += int(float(get_video_duration(files[index - 1])))
            chunk_subs.shift(
                minutes=int(total_duration / 60000),
                seconds=int((total_duration % 60000) / 1000),
                milliseconds=total_duration % 1000,
            )

        merged_subs.extend(chunk_subs)
        progress.update(clip_task, advance=1, refresh=True, description="[yellow]merging complete for chunk [/yellow]")

    progress.update(clip_task, completed=num_chunks, visible=False)
    if segment_outputs:
        for file in files:
            file.unlink(missing_ok=True)
        if structured_task_kind == "transcribe":
            return _build_segment_transcript_payload(segment_outputs)
        if structured_task_kind == "ast":
            return _build_segment_ast_payload(segment_outputs)
        for segment in segment_outputs:
            segment["description"] = segment.pop("text")
        return _build_segment_summary_payload(segment_outputs)

    merged_subs.clean_indexes()

    for file in files:
        file.unlink(missing_ok=True)

    return _serialize_subtitles(merged_subs)


def _positive_int(value, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _collect_caption_jobs(scanner) -> list[CaptionJob]:
    jobs: list[CaptionJob] = []
    for batch in scanner.to_batches():
        filepaths = batch["uris"].to_pylist()
        mimes = batch["mime"].to_pylist()
        durations = batch["duration"].to_pylist()
        sha256hashes = batch["hash"].to_pylist()

        for filepath, mime, duration, sha256hash in zip(filepaths, mimes, durations, sha256hashes):
            jobs.append(
                CaptionJob(
                    index=len(jobs),
                    filepath=filepath,
                    mime=mime,
                    duration=duration,
                    sha256hash=sha256hash,
                )
            )
    return jobs


def _resolve_cloud_concurrency_provider(args, mime: str):
    if not str(mime).startswith("image/"):
        return None

    from module.providers import get_registry

    provider_class = get_registry().find_provider(args, mime)
    if provider_class is None:
        return None
    capabilities = getattr(provider_class, "capabilities", None)
    if not bool(getattr(capabilities, "supports_cloud_concurrency", False)):
        return None
    return provider_class


def _effective_cloud_concurrency(args, provider_class) -> int:
    cloud_max = _positive_int(getattr(args, "cloud_max_concurrency", 1), 1)
    if cloud_max <= 1:
        return 1

    if getattr(provider_class, "name", "") == "codex_subscription":
        codex_max = _positive_int(getattr(args, "codex_max_concurrency", 1), 1)
        return min(cloud_max, codex_max)

    return cloud_max


def _resolve_batch_cloud_concurrency(args, jobs: list[CaptionJob], console_obj) -> tuple[Any | None, int]:
    eligible_providers = {
        provider
        for job in jobs
        if (provider := _resolve_cloud_concurrency_provider(args, job.mime)) is not None
    }
    if len(eligible_providers) != 1:
        return None, 1

    provider_class = next(iter(eligible_providers))
    if getattr(provider_class, "name", "") == "codex_subscription":
        cloud_max = _positive_int(getattr(args, "cloud_max_concurrency", 1), 1)
        codex_max = _positive_int(getattr(args, "codex_max_concurrency", 1), 1)
        if cloud_max == 1 and codex_max > 1:
            console_obj.print(
                "[yellow]codex_max_concurrency > 1 requires --cloud_max_concurrency > 1; Codex image jobs remain serial.[/yellow]"
            )

    max_workers = _effective_cloud_concurrency(args, provider_class)
    if max_workers <= 1:
        return None, 1
    return provider_class, max_workers


def _process_single_caption_job(
    job: CaptionJob,
    args,
    config,
    *,
    api_process_batch_fn,
    console_obj,
    progress=None,
    task_id=None,
) -> CaptionJobResult:
    scene_detector = create_scene_detector(args, job.mime, progress)
    if scene_detector is not None:
        scene_detector.start_async_detection(job.filepath)

    segment_time = _resolve_media_segment_time(args, job.mime, config)
    media_args = _with_media_segment_time(args, segment_time)
    if _should_process_without_segmentation(args, job.mime, job.duration, segment_time, config):
        output = api_process_batch_fn(
            uri=job.filepath,
            mime=job.mime,
            config=config,
            args=media_args,
            sha256hash=job.sha256hash,
            progress=progress,
            task_id=task_id,
        )
        output = postprocess_caption_content(output, job.filepath, media_args, console_obj)
    else:
        output = _process_segmented_media(
            job.filepath,
            job.mime,
            job.duration,
            job.sha256hash,
            media_args,
            config,
            progress,
            task_id,
            api_process_batch_fn,
            console_obj,
        )

    if isinstance(output, str) and job.mime.startswith(("video", "audio")) and output:
        try:
            subs = pysrt.from_string(output)
            subs = align_subtitles_with_scenes(
                subs,
                scene_detector,
                filepath=job.filepath,
                segment_time=media_args.segment_time,
                console=console_obj,
                timeout=getattr(args, "scene_detection_timeout", None),
            )
            output = _serialize_subtitles(subs)
        except Exception as exc:
            console_obj.print(f"[yellow]Subtitle validation failed for {job.filepath}: {exc}[/yellow]")

    if output:
        text_path, _ = write_caption_output(Path(job.filepath), output, job.mime)
        console_obj.print(f"[green]Saved captions to {text_path}[/green]")

    return CaptionJobResult(index=job.index, filepath=job.filepath, mime=job.mime, output=output)


def _process_single_caption_job_buffered(job: CaptionJob, args, config, *, api_process_batch_fn) -> CaptionJobResult:
    log_buffer = io.StringIO()
    worker_console = Console(file=log_buffer, force_terminal=False, color_system=None)
    worker_args = copy(args)
    result = _process_single_caption_job(
        job,
        worker_args,
        config,
        api_process_batch_fn=api_process_batch_fn,
        console_obj=worker_console,
        progress=None,
        task_id=None,
    )
    result.log_text = log_buffer.getvalue()
    return result


def _run_caption_jobs_concurrently(
    jobs: list[CaptionJob],
    args,
    config,
    *,
    api_process_batch_fn,
    console_obj,
    progress,
    task_id,
    max_workers: int,
    provider_class,
) -> dict[int, CaptionJobResult]:
    results: dict[int, CaptionJobResult] = {}
    futures = {}
    provider_name = getattr(provider_class, "name", "unknown")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for job in jobs:
            future = executor.submit(
                _process_single_caption_job_buffered,
                job,
                args,
                config,
                api_process_batch_fn=api_process_batch_fn,
            )
            futures[future] = job

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"Cloud caption job failed for {job.filepath} via {provider_name}: {exc}"
                ) from exc
            if result.log_text:
                console_obj.print(result.log_text, end="")
            results[result.index] = result
            progress.update(task_id, advance=1)

    return results


def process_batch(
    args,
    config,
    *,
    api_process_batch_fn,
    transform2lance_fn,
    extract_from_lance_fn,
    console_obj,
):
    setattr(args, "cloud_max_concurrency", _positive_int(getattr(args, "cloud_max_concurrency", 1), 1))
    dataset = _resolve_dataset(args, transform2lance_fn)
    scanner = dataset.scanner(
        columns=["uris", "blob", "mime", "captions", "duration", "hash"],
        scan_in_order=True,
        late_materialization=["blob"],
        batch_size=1,
    )
    jobs = _collect_caption_jobs(scanner)
    concurrent_provider, concurrent_max_workers = _resolve_batch_cloud_concurrency(args, jobs, console_obj)

    with create_caption_progress(console_obj) as progress:
        task = progress.add_task("[bold cyan]Processing media...", total=len(jobs))
        results_by_index: dict[int, CaptionJobResult] = {}
        index = 0

        while index < len(jobs):
            job = jobs[index]
            if (
                concurrent_provider is not None
                and _resolve_cloud_concurrency_provider(args, job.mime) is concurrent_provider
            ):
                block = [job]
                index += 1
                while index < len(jobs) and _resolve_cloud_concurrency_provider(args, jobs[index].mime) is concurrent_provider:
                    block.append(jobs[index])
                    index += 1

                if len(block) > 1:
                    results_by_index.update(
                        _run_caption_jobs_concurrently(
                            block,
                            args,
                            config,
                            api_process_batch_fn=api_process_batch_fn,
                            console_obj=console_obj,
                            progress=progress,
                            task_id=task,
                            max_workers=concurrent_max_workers,
                            provider_class=concurrent_provider,
                        )
                    )
                    continue

                result = _process_single_caption_job(
                    block[0],
                    args,
                    config,
                    api_process_batch_fn=api_process_batch_fn,
                    console_obj=console_obj,
                    progress=progress,
                    task_id=task,
                )
                results_by_index[result.index] = result
                progress.update(task, advance=1)
                continue

            result = _process_single_caption_job(
                job,
                args,
                config,
                api_process_batch_fn=api_process_batch_fn,
                console_obj=console_obj,
                progress=progress,
                task_id=task,
            )
            results_by_index[result.index] = result
            progress.update(task, advance=1)
            index += 1

        progress.update(task, visible=False)

    ordered_results = [results_by_index[job.index] for job in jobs]
    processed_filepaths = [result.filepath for result in ordered_results]
    results = [result.output for result in ordered_results]

    update_dataset_captions(
        dataset,
        processed_filepaths,
        results,
        merge_batch_size=getattr(args, "merge_batch_size", 1000),
        console=console_obj,
    )
    extract_from_lance_fn(dataset, args.dataset_dir, clip_with_caption=not args.not_clip_with_caption)
