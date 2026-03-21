from __future__ import annotations

from pathlib import Path

import lance
import pysrt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from config.config import APPLICATION_EXTENSIONS_SET, AUDIO_EXTENSIONS_SET, VIDEO_EXTENSIONS_SET
from module.caption_pipeline.dataset_sync import update_dataset_captions
from module.caption_pipeline.postprocess import postprocess_caption_content
from module.caption_pipeline.scene_alignment import align_subtitles_with_scenes, create_scene_detector
from utils.output_writer import write_caption_output
from utils.stream_util import (
    get_video_duration,
    split_media_stream_clips,
    split_video_with_imageio_ffmpeg,
)


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
    return str(
        payload.get("long_description")
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
    total_duration = 0
    clip_task = progress.add_task("[cyan]Processing clips...", total=num_chunks)

    for index in range(num_chunks):
        start_time = index * chunk_duration
        end_time = min((index + 1) * chunk_duration, duration_seconds)
        uri = files[index]
        chunk_output = api_process_batch_fn(
            uri=uri,
            mime=mime,
            config=config,
            args=args,
            sha256hash=sha256hash,
            progress=progress,
            task_id=task_id,
        )
        chunk_output = postprocess_caption_content(chunk_output, uri, args, console)

        if isinstance(chunk_output, dict):
            if subtitle_mode is None:
                subtitle_mode = False
            description = _structured_description(chunk_output)
            if description:
                segment_outputs.append(
                    {
                        "index": index + 1,
                        "start_seconds": start_time,
                        "end_seconds": end_time,
                        "description": description,
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
        return _build_segment_summary_payload(segment_outputs)

    merged_subs.clean_indexes()

    for file in files:
        file.unlink(missing_ok=True)

    return _serialize_subtitles(merged_subs)


def process_batch(
    args,
    config,
    *,
    api_process_batch_fn,
    transform2lance_fn,
    extract_from_lance_fn,
    console_obj,
):
    dataset = _resolve_dataset(args, transform2lance_fn)
    scanner = dataset.scanner(
        columns=["uris", "blob", "mime", "captions", "duration", "hash"],
        scan_in_order=True,
        late_materialization=["blob"],
        batch_size=1,
    )

    with Progress(
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
        expand=True,
    ) as progress:
        task = progress.add_task("[bold cyan]Processing media...", total=dataset.count_rows())
        results = []
        processed_filepaths = []

        for batch in scanner.to_batches():
            filepaths = batch["uris"].to_pylist()
            mimes = batch["mime"].to_pylist()
            durations = batch["duration"].to_pylist()
            sha256hashes = batch["hash"].to_pylist()

            for filepath, mime, duration, sha256hash in zip(filepaths, mimes, durations, sha256hashes):
                scene_detector = create_scene_detector(args, mime, progress)
                if scene_detector is not None:
                    scene_detector.start_async_detection(filepath)

                if mime.startswith("image") or duration <= (args.segment_time + 1) * 1000:
                    output = api_process_batch_fn(
                        uri=filepath,
                        mime=mime,
                        config=config,
                        args=args,
                        sha256hash=sha256hash,
                        progress=progress,
                        task_id=task,
                    )
                    output = postprocess_caption_content(output, filepath, args, console_obj)
                else:
                    output = _process_segmented_media(
                        filepath,
                        mime,
                        duration,
                        sha256hash,
                        args,
                        config,
                        progress,
                        task,
                        api_process_batch_fn,
                        console_obj,
                    )

                if isinstance(output, str) and mime.startswith(("video", "audio")) and output:
                    try:
                        subs = pysrt.from_string(output)
                        subs = align_subtitles_with_scenes(
                            subs,
                            scene_detector,
                            filepath=filepath,
                            segment_time=args.segment_time,
                            console=console_obj,
                            timeout=getattr(args, "scene_detection_timeout", None),
                        )
                        output = _serialize_subtitles(subs)
                    except Exception as exc:
                        console_obj.print(f"[yellow]Subtitle validation failed for {filepath}: {exc}[/yellow]")

                if output:
                    text_path, _ = write_caption_output(Path(filepath), output, mime)
                    console_obj.print(f"[green]Saved captions to {text_path}[/green]")

                results.append(output)
                processed_filepaths.append(filepath)
                progress.update(task, advance=1)

        progress.update(task, visible=False)

    update_dataset_captions(
        dataset,
        processed_filepaths,
        results,
        merge_batch_size=getattr(args, "merge_batch_size", 1000),
        console=console_obj,
    )
    extract_from_lance_fn(dataset, args.dataset_dir, clip_with_caption=not args.not_clip_with_caption)
