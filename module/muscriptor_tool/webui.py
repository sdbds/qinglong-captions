from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .batch_profiles import recommend_batch_size
from .options import ModelVariant, TranscriptionOptions
from .runtime import load_model


def resolve_server_batch_size(value: int) -> int | None:
    value = int(value)
    if value < 0:
        raise ValueError("batch size must be 0 (auto) or a positive integer")
    return None if value == 0 else value


def resolve_profile_auto_batch_size(
    model: ModelVariant | str,
    resolved_device: str,
    *,
    torch_module: Any | None = None,
) -> int:
    if not str(resolved_device).startswith("cuda:"):
        return 1
    if torch_module is None:
        import torch as torch_module

    properties = torch_module.cuda.get_device_properties(resolved_device)
    return recommend_batch_size(model, int(properties.total_memory))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official MuScriptor WebUI with the project runtime.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8222)
    parser.add_argument(
        "--model",
        choices=[item.value for item in ModelVariant],
        default=ModelVariant.LARGE.value,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="5-second chunks per inference batch; 0 uses the recorded model VRAM profile.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    try:
        batch_size = resolve_server_batch_size(args.batch_size)
        options = TranscriptionOptions(
            model=args.model,
            device=args.device,
            batch_size=batch_size,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    import muscriptor
    import uvicorn
    from muscriptor.server import create_app

    from utils.rich_progress import resolve_rich_console

    console = resolve_rich_console()
    loaded = load_model(options, console=console)
    profile_batch_size = (
        resolve_profile_auto_batch_size(options.model, loaded.resolved_device)
        if batch_size is None
        else None
    )
    batch_label = (
        str(batch_size)
        if batch_size is not None
        else f"{profile_batch_size} (recorded VRAM profile with OOM fallback)"
    )
    console.print(f"[cyan]MuScriptor WebUI batch size:[/cyan] {batch_label}")

    web_dir = Path(muscriptor.__file__).resolve().parent / "web_dist"
    app = create_app(loaded.model, web_dir=web_dir if web_dir.is_dir() else None)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
