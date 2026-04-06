from __future__ import annotations

import hashlib
import json
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from rich.console import Console

from utils.console_util import print_exception
from utils.lance_utils import build_version_tag, update_or_create_tag
from utils.rich_progress import create_caption_progress
from .model_manager import SeeThroughModelManager
from .pipelines.layerdiff import LayerDiffPhase
from .pipelines.marigold import MarigoldPhase
from .postprocess import run_postprocess
from .runtime import resolve_attention_backend

if TYPE_CHECKING:
    from .cli import SeeThroughRunConfig


RUN_META_NAME = "run_meta.json"
ERROR_RECORD_NAME = "error.json"
LAYERDIFF_MANIFEST = Path("layerdiff") / "manifest.json"
DEPTH_FILE = Path("depth") / "depth.png"
OPTIMIZED_MANIFEST = Path("optimized") / "manifest.json"
PSD_FILE = Path("final.psd")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BACKUP_DATASET_NAME = "dataset"
PHASE_DISPLAY_NAMES = {
    "layerdiff": "LayerDiff",
    "marigold": "Marigold",
    "postprocess": "Postprocess",
}

console = Console(color_system="truecolor", force_terminal=True)


@dataclass(frozen=True)
class ExecutionItem:
    source_path: Path
    relative_key: Path
    item_dir: Path
    resume_stage: str


@dataclass(frozen=True)
class ResumeAction:
    stage: str
    skip: bool = False

    @classmethod
    def skip_item(cls) -> "ResumeAction":
        return cls(stage="completed", skip=True)

    @classmethod
    def from_stage(cls, stage: str) -> "ResumeAction":
        return cls(stage=stage, skip=False)


def build_config_fingerprint(config: "SeeThroughRunConfig") -> str:
    payload = {
        "repo_id_layerdiff": config.repo_id_layerdiff,
        "repo_id_depth": config.repo_id_depth,
        "resolution": int(config.resolution),
        "resolution_depth": int(getattr(config, "resolution_depth", 720)),
        "inference_steps_depth": int(getattr(config, "inference_steps_depth", -1)),
        "seed": int(getattr(config, "seed", 1026)),
        "dtype": str(config.dtype),
        "quant_mode": str(getattr(config, "quant_mode", "none")),
        "save_to_psd": bool(config.save_to_psd),
        "tblr_split": bool(config.tblr_split),
        "vae_ckpt": getattr(config, "vae_ckpt", None),
        "unet_ckpt": getattr(config, "unet_ckpt", None),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return digest[:16]


def backup_input_dataset_to_lance(
    *,
    input_dir: Path,
    output_dir: Path,
    source_paths: list[Path],
    console_obj: Console | None = None,
    log_ready: bool = True,
) -> dict[str, Any]:
    import lance

    from config.config import DATASET_SCHEMA
    from module.lanceImport import FileProcessor, VideoImportMode

    resolved_console = console_obj or console
    dataset_path = Path(input_dir) / f"{BACKUP_DATASET_NAME}.lance"
    backup_tag = build_version_tag("raw", "see_through", "backup")

    schema = pa.schema(
        [
            pa.field(
                name,
                pa_type,
                metadata={b"lance-encoding:blob": b"true"} if name == "blob" else None,
            )
            for name, pa_type in DATASET_SCHEMA
        ]
    )

    processor = FileProcessor()

    resolved_console.print(f"[cyan]Creating Lance backup:[/cyan] {input_dir}")
    resolved_console.print(f"[cyan]Backup dataset path:[/cyan] {dataset_path}")
    resolved_console.print(f"[cyan]Backup tag:[/cyan] {backup_tag}")

    def _iter_batches():
        with create_caption_progress(resolved_console, transient=False, expand=True) as progress:
            task_id = progress.add_task("[bold cyan]Backing up dataset to Lance...", total=len(source_paths))

            for source_path in source_paths:
                progress.console.print(f"[blue]Backing up image:[/blue] {source_path}")
                metadata = processor.load_metadata(
                    str(source_path),
                    save_binary=True,
                    import_mode=VideoImportMode.ALL,
                )
                if metadata is None:
                    raise RuntimeError(f"Failed to read source for Lance backup: {source_path}")

                arrays = []
                for field_name, field_type in DATASET_SCHEMA:
                    if field_name == "filepath":
                        value = str(Path(source_path).absolute())
                    elif field_name == "captions":
                        value = []
                    elif field_name == "chunk_offsets":
                        value = []
                    else:
                        value = getattr(metadata, field_name)
                        if value is None:
                            if pa.types.is_integer(field_type):
                                value = 0
                            elif pa.types.is_floating(field_type):
                                value = 0.0
                            elif pa.types.is_boolean(field_type):
                                value = False
                            elif pa.types.is_string(field_type):
                                value = ""
                    arrays.append(pa.array([value], type=field_type))

                yield pa.RecordBatch.from_arrays(
                    arrays,
                    names=[field_name for field_name, _ in DATASET_SCHEMA],
                )
                progress.console.print(f"[green]Backed up image:[/green] {source_path}")
                progress.update(task_id, advance=1)

    reader = pa.RecordBatchReader.from_batches(schema, _iter_batches())
    dataset = lance.write_dataset(reader, str(dataset_path), schema, mode="overwrite")
    version = update_or_create_tag(dataset, backup_tag)
    if log_ready:
        resolved_console.print(
            f"[green]Lance backup ready:[/green] {dataset_path} "
            f"[cyan]tag:[/cyan] {backup_tag} [cyan]version:[/cyan] {version}"
        )
    return {
        "dataset_path": str(dataset_path),
        "tag": backup_tag,
        "version": version,
    }


def prepare_output_dir(output_dir: Path, input_dir: Path, config_fingerprint: str) -> dict[str, Any]:
    requested_output_dir = Path(output_dir)
    input_dir = Path(input_dir).resolve()

    exact_probe = _probe_output_dir(requested_output_dir, input_dir, config_fingerprint)
    if exact_probe["status"] == "reusable":
        return {
            "output_dir": requested_output_dir,
            "meta_path": requested_output_dir / RUN_META_NAME,
            "run_meta": exact_probe["run_meta"],
        }
    if exact_probe["status"] == "empty":
        return _initialize_output_dir(requested_output_dir, input_dir, config_fingerprint)
    if exact_probe["status"] == "mismatch":
        raise ValueError(
            "output_dir already belongs to a different see-through run. "
            "Choose a new output_dir for different input/config combinations."
        )

    for candidate in _derived_output_candidates(requested_output_dir):
        candidate_probe = _probe_output_dir(candidate, input_dir, config_fingerprint)
        if candidate_probe["status"] == "reusable":
            return {
                "output_dir": candidate,
                "meta_path": candidate / RUN_META_NAME,
                "run_meta": candidate_probe["run_meta"],
            }
        if candidate_probe["status"] == "empty":
            return _initialize_output_dir(candidate, input_dir, config_fingerprint)

    raise ValueError("unable to allocate a derived output directory under the requested output_dir")


def _probe_output_dir(output_dir: Path, input_dir: Path, config_fingerprint: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / RUN_META_NAME

    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("input_dir") == str(input_dir) and existing.get("config_fingerprint") == config_fingerprint:
            return {"status": "reusable", "run_meta": existing}
        return {"status": "mismatch", "run_meta": existing}

    if any(output_dir.iterdir()):
        return {"status": "occupied", "run_meta": None}

    return {"status": "empty", "run_meta": None}


def _initialize_output_dir(output_dir: Path, input_dir: Path, config_fingerprint: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / RUN_META_NAME
    run_meta = {
        "config_fingerprint": config_fingerprint,
        "input_dir": str(input_dir),
        "created_at": output_dir.stat().st_mtime,
    }
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {"output_dir": output_dir, "meta_path": meta_path, "run_meta": run_meta}


def _update_run_meta(meta_path: Path, updates: dict[str, Any]) -> None:
    existing = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    existing.update(updates)
    Path(meta_path).write_text(json.dumps(existing, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _derived_output_candidates(output_dir: Path):
    base_dir = Path(output_dir)
    for index in range(1, 1000):
        suffix = "" if index == 1 else f"-{index}"
        yield base_dir / f"outputs{suffix}"


def make_relative_output_key(input_dir: Path, source_path: Path) -> Path:
    input_dir = Path(input_dir).resolve()
    source_path = Path(source_path).resolve()
    try:
        relative = source_path.relative_to(input_dir)
    except ValueError as exc:
        raise ValueError(f"source_path is outside input_dir: {source_path}") from exc
    return Path(*relative.parts)


def make_item_dir(output_dir: Path, relative_key: Path) -> Path:
    return Path(output_dir) / Path(relative_key)


def layerdiff_outputs_complete(item_dir: Path) -> bool:
    return (item_dir / "src_img.png").exists() and (item_dir / LAYERDIFF_MANIFEST).exists()


def marigold_outputs_complete(item_dir: Path) -> bool:
    return layerdiff_outputs_complete(item_dir) and (item_dir / DEPTH_FILE).exists()


def postprocess_outputs_complete(item_dir: Path, *, save_to_psd: bool = True) -> bool:
    if not marigold_outputs_complete(item_dir):
        return False
    if not (item_dir / OPTIMIZED_MANIFEST).exists():
        return False
    if save_to_psd and not (item_dir / PSD_FILE).exists():
        return False
    return True


def detect_resume_stage(item_dir: Path, *, save_to_psd: bool = True) -> str:
    action = detect_resume_action(item_dir, save_to_psd=save_to_psd)
    return action.stage


def detect_resume_action(item_dir: Path, *, save_to_psd: bool = True) -> ResumeAction:
    if postprocess_outputs_complete(item_dir, save_to_psd=save_to_psd):
        return ResumeAction.skip_item()
    if marigold_outputs_complete(item_dir):
        return ResumeAction.from_stage("postprocess")
    if layerdiff_outputs_complete(item_dir):
        return ResumeAction.from_stage("marigold")
    return ResumeAction.from_stage("layerdiff")


def collect_input_images(input_dir: Path, limit_images: int) -> list[Path]:
    input_dir = Path(input_dir)
    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if int(limit_images) > 0:
        return files[: int(limit_images)]
    return files


def build_execution_plan(config: "SeeThroughRunConfig", output_dir: Path, discovered_items: list[Path]) -> list[ExecutionItem]:
    plan: list[ExecutionItem] = []
    for source_path in discovered_items:
        relative_key = make_relative_output_key(config.input_dir, source_path)
        item_dir = make_item_dir(output_dir, relative_key)
        resume_stage = "layerdiff"
        if config.skip_completed:
            resume_stage = detect_resume_stage(item_dir, save_to_psd=config.save_to_psd)
        plan.append(
            ExecutionItem(
                source_path=source_path,
                relative_key=relative_key,
                item_dir=item_dir,
                resume_stage=resume_stage,
            )
        )
    return plan


def _format_vram_log(vram_snapshot: dict[str, float | str]) -> str:
    if not str(vram_snapshot.get("device") or "").startswith("cuda"):
        return f"[dim]VRAM[/dim] {vram_snapshot.get('stage')}: cpu"
    return (
        f"[dim]VRAM[/dim] {vram_snapshot.get('stage')}: "
        f"allocated={vram_snapshot.get('allocated_mb')}MB "
        f"reserved={vram_snapshot.get('reserved_mb')}MB "
        f"max={vram_snapshot.get('max_allocated_mb')}MB"
    )


def _log_plan_summary(console_obj: Console, plan: list[ExecutionItem]) -> None:
    counts = Counter(item.resume_stage for item in plan)
    console_obj.print(
        "[cyan]Execution plan:[/cyan] "
        f"layerdiff={counts.get('layerdiff', 0)} "
        f"marigold={counts.get('marigold', 0)} "
        f"postprocess={counts.get('postprocess', 0)} "
        f"completed={counts.get('completed', 0)}"
    )
    for item in plan:
        if item.resume_stage == "completed":
            console_obj.print(f"[yellow]Skipping completed item:[/yellow] {item.relative_key.as_posix()}")


def run_see_through_batch(config: "SeeThroughRunConfig", *, console_obj: Console | None = None) -> int:
    resolved_console = console_obj or console
    try:
        config_fingerprint = build_config_fingerprint(config)
        prepared = prepare_output_dir(config.output_dir, config.input_dir, config_fingerprint)
    except Exception as exc:
        print_exception(resolved_console, exc, prefix="See-through output preparation failed")
        return 1
    resolved_output_dir = prepared["output_dir"]
    if Path(config.output_dir).resolve() != Path(resolved_output_dir).resolve():
        resolved_console.print(f"[yellow]See-through resolved output_dir:[/yellow] {resolved_output_dir}")

    discovered_items = collect_input_images(config.input_dir, config.limit_images)
    if not discovered_items:
        resolved_console.print(f"[yellow]No supported images found under:[/yellow] {config.input_dir}")
        return 1

    try:
        backup_meta = backup_input_dataset_to_lance(
            input_dir=config.input_dir,
            output_dir=resolved_output_dir,
            source_paths=discovered_items,
            console_obj=resolved_console,
            log_ready=False,
        )
        _update_run_meta(
            prepared["meta_path"],
            {
                "lance_backup_path": backup_meta["dataset_path"],
                "lance_backup_tag": backup_meta["tag"],
                "lance_backup_version": backup_meta["version"],
            },
        )
        resolved_console.print(
            f"[green]Lance backup ready:[/green] {backup_meta['dataset_path']} "
            f"[cyan]tag:[/cyan] {backup_meta['tag']} [cyan]version:[/cyan] {backup_meta['version']}"
        )
    except Exception as exc:
        print_exception(resolved_console, exc, prefix="See-through Lance backup failed")
        return 1

    runtime_context = resolve_attention_backend(
        force_eager_attention=config.force_eager_attention,
        dtype_name=config.dtype,
    )
    runtime_device = getattr(runtime_context, "device", "unknown")
    runtime_dtype = getattr(runtime_context, "dtype", getattr(config, "dtype", "unknown"))
    runtime_reason = getattr(runtime_context, "reason", "unknown")
    resolved_console.print(
        "[cyan]Runtime context:[/cyan] "
        f"device={runtime_device} dtype={runtime_dtype} "
        f"[cyan]Attention backend:[/cyan] {runtime_context.attention_backend}"
    )
    resolved_console.print(f"[dim]Attention backend reason:[/dim] {runtime_reason}")

    model_manager = SeeThroughModelManager(
        offload_policy=config.offload_policy,
        runtime_context=runtime_context,
    )

    try:
        plan = build_execution_plan(config, resolved_output_dir, discovered_items)
        _log_plan_summary(resolved_console, plan)

        completed = sum(1 for item in plan if item.resume_stage == "completed")
        failures = 0

        layerdiff_items = [item for item in plan if item.resume_stage == "layerdiff"]
        marigold_seed = [item for item in plan if item.resume_stage == "marigold"]
        postprocess_seed = [item for item in plan if item.resume_stage == "postprocess"]

        resolved_console.print(_format_vram_log(model_manager.log_vram("before_layerdiff")))
        layerdiff_phase = LayerDiffPhase(model_manager, config, runtime_context, console_obj=resolved_console)
        try:
            layerdiff_success, layerdiff_failures = _process_phase_items(
                phase_name="layerdiff",
                items=layerdiff_items,
                handler=lambda item: layerdiff_phase.run_item(item.source_path, item.item_dir),
                continue_on_error=config.continue_on_error,
                console_obj=resolved_console,
            )
        finally:
            resolved_console.print(_format_vram_log(model_manager.log_vram("after_layerdiff")))
            model_manager.release_layerdiff()
            resolved_console.print(_format_vram_log(model_manager.log_vram("after_layerdiff_release")))
        failures += layerdiff_failures

        marigold_items = marigold_seed + layerdiff_success
        resolved_console.print(_format_vram_log(model_manager.log_vram("before_marigold")))
        marigold_phase = MarigoldPhase(model_manager, config, runtime_context, console_obj=resolved_console)
        try:
            marigold_success, marigold_failures = _process_phase_items(
                phase_name="marigold",
                items=marigold_items,
                handler=lambda item: marigold_phase.run_item(item.source_path, item.item_dir),
                continue_on_error=config.continue_on_error,
                console_obj=resolved_console,
            )
        finally:
            resolved_console.print(_format_vram_log(model_manager.log_vram("after_marigold")))
            model_manager.release_marigold()
            resolved_console.print(_format_vram_log(model_manager.log_vram("after_marigold_release")))
        failures += marigold_failures

        postprocess_items = postprocess_seed + marigold_success
        postprocess_success, postprocess_failures = _process_phase_items(
            phase_name="postprocess",
            items=postprocess_items,
            handler=lambda item: run_postprocess(
                source_path=item.source_path,
                output_dir=item.item_dir,
                save_to_psd=config.save_to_psd,
                tblr_split=config.tblr_split,
            ),
            continue_on_error=config.continue_on_error,
            console_obj=resolved_console,
        )
        failures += postprocess_failures

        completed += len(postprocess_success)
        partial = max(len(discovered_items) - completed - failures, 0)
        resolved_console.print(
            f"[bold]See-through finished.[/bold] total={len(discovered_items)} "
            f"completed={completed} partial={partial} failed={failures} "
            f"backend={runtime_context.attention_backend}"
        )
        return 1 if failures else 0
    except Exception as exc:
        print_exception(resolved_console, exc, prefix="See-through batch failed")
        return 1
    finally:
        model_manager.release_all()


def _process_phase_items(
    *,
    phase_name: str,
    items: list[ExecutionItem],
    handler: Any,
    continue_on_error: bool,
    console_obj: Console | None = None,
) -> tuple[list[ExecutionItem], int]:
    if not items:
        return [], 0

    successes: list[ExecutionItem] = []
    failures = 0
    resolved_console = console_obj or console
    phase_title = PHASE_DISPLAY_NAMES.get(phase_name, phase_name.replace("_", " ").title())

    with create_caption_progress(resolved_console, transient=False, expand=True) as progress:
        task_id = progress.add_task(f"[bold cyan]{phase_title} phase...", total=len(items))

        for item in items:
            item.item_dir.mkdir(parents=True, exist_ok=True)
            progress.console.print(f"[blue]{phase_title} processing:[/blue] {item.relative_key.as_posix()}")
            try:
                handler(item)
                _clear_error_record(item.item_dir)
                successes.append(item)
                progress.console.print(f"[green]{phase_title} succeeded:[/green] {item.relative_key.as_posix()}")
            except Exception as exc:
                failures += 1
                _write_error_record(item.item_dir, item.source_path, phase_name, exc)
                print_exception(
                    progress.console,
                    exc,
                    prefix=f"{phase_title} failed for {item.relative_key.as_posix()}",
                )
                if not continue_on_error:
                    raise
            finally:
                progress.update(task_id, advance=1)

    return successes, failures


def _write_error_record(item_dir: Path, source_path: Path, phase_name: str, exc: Exception) -> None:
    payload = {
        "source_path": str(source_path),
        "phase": phase_name,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    (item_dir / ERROR_RECORD_NAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _clear_error_record(item_dir: Path) -> None:
    error_path = item_dir / ERROR_RECORD_NAME
    if error_path.exists():
        error_path.unlink()
