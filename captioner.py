import lance
from rich_pixels import Pixels
from rich.progress import Progress
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
import argparse
from lanceImport import transform2lance
from lanceexport import extract_from_lance
import pandas as pd
import pyarrow as pa
import re

# from mistralai import Mistral
import toml
from PIL import Image
import pysrt
from api_handler import api_process_batch
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None  # Disable image size limit check

console = Console()


def process_batch(args, config):
    # Load the dataset
    if not isinstance(args.dataset_dir, lance.LanceDataset):
        if args.api_key == "":
            dataset = transform2lance(dataset_dir=args.dataset_dir)
        else:
            dataset = transform2lance(dataset_dir=args.dataset_dir, save_binary=False)

    scanner = dataset.scanner(
        columns=["uris", "blob", "mime", "captions"],
        scan_in_order=True,
        late_materialization=["blob"],
        batch_size=1,
    )
    total_rows = dataset.count_rows()

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing videos...", total=total_rows)

        results = []
        processed_filepaths = []
        for batch in scanner.to_batches():
            filepaths = batch["uris"].to_pylist()
            mime = batch["mime"].to_pylist()

            for filepath, mime in zip(filepaths, mime):
                output = api_process_batch(
                    uri=filepath,
                    mime=mime,
                    config=config,
                    api_key=args.api_key,
                    wait_time=1,
                    max_retries=100,
                    model_path=args.model_path,
                )

                if not output:
                    console.print(
                        f"[red]No caption content generated for {filepath}[/red]"
                    )
                    continue

                # 预处理字幕内容
                if isinstance(output, list):
                    output = "\n".join(output)

                # 确保字幕内容格式正确
                output = output.strip()
                if not output.strip():
                    console.print(f"[red]Empty caption content for {filepath}[/red]")
                    continue

                # 格式化时间戳 - 只处理7位的时间戳 (MM:SS,ZZZ)
                output = re.sub(
                    r"(?<!:)(\d{2}):(\d{2}),(\d{3})",
                    r"00:\1:\2,\3",
                    output,
                    flags=re.MULTILINE,
                )

                results.append(output)
                processed_filepaths.append(filepath)

                filepath_path = Path(filepath)
                caption_path = filepath_path.with_suffix(".srt")
                console.print(f"[blue]Processing caption for:[/blue] {filepath_path}")
                console.print(f"[blue]Caption content length:[/blue] {len(output)}")

                try:
                    subs = pysrt.from_string(output)
                    subs.save(str(caption_path), encoding="utf-8")
                    console.print(f"[green]Saved captions to {caption_path}[/green]")
                except Exception as e:
                    console.print(
                        f"[yellow]pysrt validation failed: {e}, falling back to direct file write[/yellow]"
                    )
                    try:
                        caption_path.write_text(output, encoding="utf-8")
                        console.print(
                            f"[green]Saved captions to {caption_path}[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]Error saving SRT file: {e}[/red]")

            progress.update(task, advance=len(batch))

    # Update dataset with new captions
    if results:
        # 确保每个caption都是单个字符串
        processed_captions = []
        for caption in results:
            if isinstance(caption, list):
                # 如果是列表，合并为单个字符串
                processed_captions.append("\n".join(caption))
            else:
                processed_captions.append(caption)

        table = pa.table(
            {
                "uris": pa.array(processed_filepaths, type=pa.string()),
                "captions": pa.array(
                    [[caption] for caption in processed_captions],
                    type=pa.list_(pa.string()),
                ),
            }
        )

        dataset.merge_insert(on="uris").when_matched_update_all().execute(table)

        try:
            dataset.tags.create("gemini", 1)
        except:
            dataset.tags.update("gemini", 1)

        console.print("[green]Successfully updated dataset with new captions[/green]")

    extract_from_lance(dataset, args.dataset_dir, clip_with_caption=not args.not_clip_with_caption)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", type=str, help="directory for dataset")

    parser.add_argument(
        "--systemprompt",
        type=str,
        help="directory for train images",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for pixtral API",
    )

    parser.add_argument(
        "--dir_name",
        action="store_true",
        help="Use the directory name as the dataset name",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="gemini-exp-1206",
        help="Model path for gemini",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Mode for processing the dataset",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config file",
    )

    parser.add_argument(
        "--not_clip_with_caption",
        action="store_true",
        help="Not clip with caption",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    config = toml.load(args.config)

    process_batch(args, config)
