# /// script
# dependencies = [
#   "setuptools",
#   "pillow>=11.3",
#   "pylance>=2.0.1",
#   "rich>=13.5.0",
#   "imageio>=2.31.1",
#   "imageio-ffmpeg>=0.4.8",
#   "mutagen",
#   "toml",
#   "huggingface_hub[hf_xet]>=0.35.2",
#   "torch==2.8.0",
#   "transformers>4.50",
#   "torchvision",
#   "scipy",
#   "huggingface_hub[hf_xet]>=0.35.2",
#   "onnxruntime-gpu==1.20.2; sys_platform == 'win32'",
#   "onnxruntime-gpu>=1.20.2; sys_platform == 'linux'",
# ]
# [tool.uv.extra-build-dependencies]
#       tensorrt-cu13 = ["setuptools"]
#       tensorrt-cu13-libs = ["wheel_stub"]
# ///
import argparse
import concurrent.futures
import json
import shutil
import time
from pathlib import Path

import torch
import lance
import numpy as np
from PIL import Image
from rich.console import Console
from rich.pretty import Pretty
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
from transformers import AutoImageProcessor

from config.loader import load_config
from module.lanceImport import transform2lance
from module.onnx_runtime import OnnxModelSpec, load_single_model_bundle, resolve_tool_runtime_config
from utils.console_util import print_exception

console = Console(color_system="truecolor", force_terminal=True)
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def preprocess_image(image):
    """使用processor预处理图像"""
    try:
        # 转换为RGB模式
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL image, numpy array, or file path")

        # 使用processor预处理图像
        inputs = processor(images=image, return_tensors="pt")

        # 转换为numpy数组返回
        return inputs["pixel_values"][0].numpy()
    except Exception as e:
        print_exception(console, e, prefix="preprocess_image error")
        return None


def load_and_preprocess_batch(uris):
    """并行加载和预处理一批图像"""

    def load_single_image(uri):
        try:
            # 直接传入路径，在preprocess_image中处理转换
            return preprocess_image(uri)
        except Exception as e:
            print_exception(console, e, prefix=f"Error processing {uri}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        batch_images = list(executor.map(load_single_image, uris))

    # 过滤掉加载失败的图像
    valid_images = [(i, img) for i, img in enumerate(batch_images) if img is not None]
    images = [img for _, img in valid_images]

    return images


def process_batch(images, session, input_name):
    """处理图像批次"""
    try:
        # 图像通过processor处理后的numpy数组，直接堆叠
        batch_data = np.ascontiguousarray(np.stack(images))
        # 执行推理
        outputs = session.run(None, {input_name: batch_data})
        return outputs[0]
    except Exception as e:
        print_exception(console, e, prefix="Batch processing error")
        return None


def load_model(args):
    """加载模型和标签"""
    global processor
    processor = AutoImageProcessor.from_pretrained(args.repo_id, use_fast=True)
    runtime_config = resolve_tool_runtime_config(
        load_config(str(CONFIG_DIR)),
        tool_name="waterdetect",
        cli_override={"force_download": args.force_download},
    )
    spec = OnnxModelSpec(
        repo_id=args.repo_id,
        onnx_filename="model.onnx",
        local_dir=Path(args.model_dir) / args.repo_id.replace("/", "_"),
        bundle_key=f"waterdetect:{args.repo_id}",
    )

    start_time = time.time()
    bundle = load_single_model_bundle(spec=spec, runtime_config=runtime_config, logger=console.print)
    input_name = bundle.input_metas[0].name if bundle.input_metas else bundle.session.get_inputs()[0].name
    console.print("[cyan]Providers:[/cyan]")
    console.print(Pretty(bundle.providers, indent_guides=True, expand_all=True))
    console.print(f"[green]Model loaded in {time.time() - start_time:.2f} seconds[/green]")
    return bundle.session, input_name


def main(args):
    global console

    watermark_dir = Path(args.train_data_dir) / "watermarked"
    no_watermark_dir = Path(args.train_data_dir) / "no_watermark"
    # 确保目标文件夹存在
    if watermark_dir.exists():
        for symlink in watermark_dir.rglob("*"):
            if symlink.is_symlink():
                symlink.unlink()
    if no_watermark_dir.exists():
        for symlink in no_watermark_dir.rglob("*"):
            if symlink.is_symlink():
                symlink.unlink()

    # 初始化 Lance 数据集
    if not isinstance(args.train_data_dir, lance.LanceDataset):
        if args.train_data_dir.endswith(".lance"):
            dataset = lance.dataset(args.train_data_dir)
        elif any(file.suffix == ".lance" for file in Path(args.train_data_dir).glob("*")):
            lance_file = next(file for file in Path(args.train_data_dir).glob("*") if file.suffix == ".lance")
            dataset = lance.dataset(str(lance_file))
        else:
            console.print("[yellow]Converting dataset to Lance format...[/yellow]")
            dataset = transform2lance(
                args.train_data_dir,
                output_name="dataset",
                save_binary=False,
                not_save_disk=False,
                tag="WatermarkDetection",
            )
            console.print("[green]Dataset converted to Lance format[/green]")

    else:
        dataset = args.train_data_dir
        console.print("[green]Using existing Lance dataset[/green]")

    ort_sess, input_name = load_model(args)

    # 先计算图片总数
    total_images = len(
        dataset.to_table(
            columns=["mime"],
            filter=("mime LIKE 'image/%'"),
        )
    )

    # 然后创建带columns的scanner处理数据
    scanner = dataset.scanner(
        columns=["uris", "mime", "captions"],
        filter=("mime LIKE 'image/%'"),
        scan_in_order=True,
        batch_size=args.batch_size,
        batch_readahead=16,
        fragment_readahead=4,
        io_buffer_size=32 * 1024 * 1024,  # 32MB buffer
        late_materialization=True,
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
        task = progress.add_task("[bold cyan]Processing images...", total=total_images)

        console = progress.console

        # 用于收集结果的列表
        detection_results = []

        for batch in scanner.to_batches():
            uris = batch["uris"].to_pylist()  # 获取文件路径

            # 使用并行处理加载和预处理图像
            batch_images = load_and_preprocess_batch(uris)

            if not batch_images:
                progress.update(task, advance=len(uris))
                continue

            # 处理批次
            probs = process_batch(batch_images, ort_sess, input_name)
            # 创建对应的目标文件夹（如果不存在）
            if probs is not None:
                for path, prob in zip(uris, probs):
                    # 获取水印检测结果
                    watermark_prob = prob[1]  # 索引1对应"Watermark"标签

                    # 根据概率确定是否有水印（阈值可以调整）
                    has_watermark = watermark_prob > args.thresh

                    # 添加到结果列表
                    detection_results.append((path, float(watermark_prob)))

                    # 创建软链接
                    source_path = Path(path).absolute()
                    relative_path = source_path.relative_to(Path(args.train_data_dir).absolute())
                    target_dir = watermark_dir if has_watermark else no_watermark_dir
                    target_path = target_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # 创建软链接
                    try:
                        target_path.symlink_to(source_path)
                    except (FileExistsError, PermissionError) as e:
                        print_exception(console, e, prefix=f"Unable to create symlink for {path}")
                        # 如果无法创建软链接，尝试复制文件代替
                        try:
                            shutil.copy2(source_path, target_path)
                            console.print(f"[yellow]Created copy instead of symlink for {path}[/yellow]")
                        except Exception as copy_err:
                            print_exception(console, copy_err, prefix="Failed to copy file")

            progress.update(task, advance=len(batch["uris"].to_pylist()))

    # 统计水印图片数量
    watermark_count = sum(1 for _, prob in detection_results if prob > args.thresh)
    total_count = len(detection_results)

    # 按路径层次结构组织结果
    path_tree = {}
    for path, prob in detection_results:
        parts = Path(path).relative_to(Path(args.train_data_dir).absolute()).parts
        current = path_tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # 最后一层是文件名，存储概率
                current[part] = f"{prob:.4f}" + ("🔴(Watermarked)🔖" if prob > args.thresh else "🟢(No Watermark)📄")
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

    # 使用Pretty打印结果树
    console.print("\n[bold green]Results：[/bold green]")
    console.print(Pretty(path_tree, indent_guides=True, expand_all=True))
    # 保存检测结果树到JSON文件
    result_json_path = Path(args.train_data_dir) / "watermark_detection_results.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(path_tree, f, ensure_ascii=False, indent=2)
    console.print(f"[bold green]Results saved to:[/bold green] {result_json_path}")

    # 打印检测结果统计
    if total_count > 0:
        console.print(f"🔴Watermarked🔖: {watermark_count} ({watermark_count / total_count * 100:.2f}%)")
        console.print(f"🟢No Watermark📄: {total_count - watermark_count} ({(total_count - watermark_count) / total_count * 100:.2f}%)")
    else:
        console.print("[yellow]No images were processed[/yellow]")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir",
        type=str,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="bdsqlsz/joycaption-watermark-detection-onnx",
        help="Repository ID for Watermark Detection model on Hugging Face",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="watermark_detection",
        help="Directory to store Watermark Detection model",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force downloading Watermark Detection model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=1.0,
        help="Default threshold for tag confidence",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    main(args)
