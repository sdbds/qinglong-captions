from __future__ import annotations


def create_scene_detector(args, mime: str, progress):
    if not (getattr(args, "scene_threshold", 0) > 0 and getattr(args, "scene_min_len", 0) > 0 and mime.startswith("video")):
        return None

    from module.videospilter import SceneDetector

    detector = SceneDetector(
        detector=args.scene_detector,
        threshold=args.scene_threshold,
        min_scene_len=args.scene_min_len,
        luma_only=args.scene_luma_only,
        console=progress,
    )
    return detector


def align_subtitles_with_scenes(subs, scene_detector, filepath: str, segment_time: int, console, timeout=None):
    if scene_detector is None:
        return subs

    try:
        scene_list = scene_detector.wait_for_detection(filepath, timeout=timeout)
    except Exception as exc:
        console.print(f"[yellow]Scene detection unavailable for {filepath}: {exc}[/yellow]")
        return subs

    if not scene_list:
        return subs

    try:
        console.print("[bold cyan]Aligning subtitles with scene changes...[/bold cyan]")
        return scene_detector.align_subtitle(
            subs,
            scene_list=scene_list,
            console=console,
            segment_time=segment_time,
        )
    except Exception as exc:
        console.print(f"[yellow]Scene alignment skipped for {filepath}: {exc}[/yellow]")
        return subs
