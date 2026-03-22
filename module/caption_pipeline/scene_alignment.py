from __future__ import annotations


def _load_scene_defaults() -> dict:
    try:
        import toml
        from pathlib import Path
        cfg = toml.load(Path(__file__).resolve().parent.parent.parent / "config" / "model.toml")
        return cfg.get("scene_detection", {})
    except Exception:
        return {}


def create_scene_detector(args, mime: str, progress):
    scene_threshold = getattr(args, "scene_threshold", 0)
    scene_min_len = getattr(args, "scene_min_len", 0)

    # If CLI args are at default/unset values, read TOML defaults
    if scene_threshold == 0 or scene_min_len == 0:
        defaults = _load_scene_defaults()
        if scene_threshold == 0:
            scene_threshold = defaults.get("threshold", 0.0)
        if scene_min_len == 0:
            scene_min_len = defaults.get("min_scene_len", 0)

    if not (scene_threshold > 0 and scene_min_len > 0 and mime.startswith("video")):
        return None

    defaults = _load_scene_defaults()
    scene_detector = getattr(args, "scene_detector", None) or defaults.get("detector", "AdaptiveDetector")
    scene_luma_only = getattr(args, "scene_luma_only", None)
    if scene_luma_only is None:
        scene_luma_only = defaults.get("luma_only", False)

    from module.videospilter import SceneDetector

    detector = SceneDetector(
        detector=scene_detector,
        threshold=scene_threshold,
        min_scene_len=scene_min_len,
        luma_only=scene_luma_only,
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
