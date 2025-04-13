import argparse
import numpy as np
import cv2
import time
from PIL import Image
from pathlib import Path
import csv
import lance
import pyarrow as pa
from rich.console import Console
from rich.pretty import Pretty
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
    MofNCompleteColumn,
)
import torch
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from module.lanceImport import transform2lance
import concurrent.futures

console = Console()

# from wd14 tagger
IMAGE_SIZE = 448

DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
FILES = ["model.onnx", "selected_tags.csv"]
CSV_FILE = "selected_tags.csv"
PARENTS_CSV = "tag_implications.csv"


def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # 使用更高效的方式计算填充
    h, w = image.shape[:2]
    size = max(h, w)

    # 使用单个 pad 操作替代多次计算
    pad_y, pad_x = size - h, size - w
    pad_t, pad_l = pad_y // 2, pad_x // 2
    pad_b, pad_r = pad_y - pad_t, pad_x - pad_l

    # 使用更高效的填充
    image = cv2.copyMakeBorder(
        image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    if size > IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_AREA)
    else:
        image = Image.fromarray(image)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        image = np.array(image)

    image = image.astype(np.float32)
    return image


def load_and_preprocess_batch(uris):
    """并行加载和预处理一批图像"""

    def load_single_image(uri):
        try:
            return preprocess_image(Image.open(uri).convert("RGB"))
        except Exception as e:
            console.print(f"[red]Error processing {uri}: {str(e)}[/red]")
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
        # 图像已经是 numpy 数组，直接堆叠并确保连续
        batch_data = np.ascontiguousarray(np.stack(images))
        # 执行推理
        outputs = session.run(None, {input_name: batch_data})
        return outputs[0]
    except Exception as e:
        console.print(f"[red]Batch processing error: {str(e)}[/red]")
        return None


def load_model_and_tags(args):
    """加载模型和标签"""
    model_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / "model.onnx"

    # 下载模型和标签文件
    if not model_path.exists() or args.force_download:
        for file in FILES:
            file_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / file
            if not file_path.exists() or args.force_download:
                file_path = Path(
                    hf_hub_download(
                        repo_id=args.repo_id,
                        filename=file,
                        local_dir=file_path.parent,
                        force_download=args.force_download,
                        force_filename=file,
                    )
                )
                console.print(f"[blue]Downloaded {file} to {file_path}[/blue]")
            else:
                console.print(f"[green]Using existing {file}[/green]")

    # 加载标签
    csv_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / CSV_FILE
    if not csv_path.exists():
        raise Exception(f"Tags file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]  # tag_id,name,category,count
        rows = line[1:]

    assert (
        header[0] == "tag_id" and header[1] == "name" and header[2] == "category"
    ), f"unexpected csv format: {header}"

    # 根据category分类标签
    rating_tags = [row[1] for row in rows if row[2] == "9"]  # rating tags
    general_tags = [row[1] for row in rows if row[2] == "0"]  # general tags
    character_tags = [row[1] for row in rows if row[2] == "4"]  # character tags

    console.print(
        f"[blue]Tags loaded: {len(rows)} total, {len(rating_tags)} rating, {len(character_tags)} character, {len(general_tags)} general[/blue]"
    )

    # 设置推理提供者
    providers = []
    if "TensorrtExecutionProvider" in ort.get_available_providers():
        providers.append("TensorrtExecutionProvider")
        console.print("[green]Using TensorRT for inference[/green]")
        console.print("[yellow]compile may take a long time, please wait...[/yellow]")
    elif "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
        console.print("[green]Using CUDA for inference[/green]")
    elif "ROCMExecutionProvider" in ort.get_available_providers():
        providers.append("ROCMExecutionProvider")
        console.print("[green]Using ROCm for inference[/green]")
    elif "OpenVINOExecutionProvider" in ort.get_available_providers():
        providers = [("OpenVINOExecutionProvider", {"device_type": "GPU_FP32"})]
        console.print("[green]Using OpenVINO for inference[/green]")
    else:
        providers.append("CPUExecutionProvider")
        console.print("[yellow]Using CPU for inference[/yellow]")

    # 创建推理会话
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )  # 启用所有优化

    if "CPUExecutionProvider" in providers:
        # CPU时启用多线程推理
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 启用并行执行
        sess_options.inter_op_num_threads = 8  # 设置线程数
        sess_options.intra_op_num_threads = 8  # 设置算子内部并行数

    # TensorRT 优化
    if "TensorrtExecutionProvider" in providers:
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        providers_with_options = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,  # Enable FP16 precision for faster inference
                    "trt_builder_optimization_level": 3,
                    "trt_max_partition_iterations": 1000,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "wd14_tagger_model/trt_engines",
                    "trt_engine_hw_compatible": True,
                    "trt_force_sequential_engine_build": False,
                    "trt_context_memory_sharing_enable": True,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": "wd14_tagger_model",
                    "trt_sparsity_enable": True,
                    "trt_min_subgraph_size": 7,
                    # "trt_detailed_build_log": True,
                },
            ),
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": "1",  # 使用最大工作空间
                    "tunable_op_enable": True,  # 启用可调优操作
                    "tunable_op_tuning_enable": True,  # 启用调优
                },
            ),
        ]

    elif "CUDAExecutionProvider" in providers:
        # CUDA GPU 优化
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        providers_with_options = [
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": "1",
                    "tunable_op_enable": True,
                    "tunable_op_tuning_enable": True,
                },
            ),
        ]
    else:
        providers_with_options = providers

    console.print(f"[cyan]Providers with options:[/cyan]")
    console.print(Pretty(providers_with_options, indent_guides=True, expand_all=True))
    start_time = time.time()
    ort_sess = ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers_with_options
    )
    input_name = ort_sess.get_inputs()[0].name
    console.print(f"[green]Model loaded in {time.time() - start_time:.2f} seconds[/green]")
    return ort_sess, input_name, rating_tags, character_tags, general_tags


def main(args):
    global console

    # 初始化 Lance 数据集
    if not isinstance(args.train_data_dir, lance.LanceDataset):
        if args.train_data_dir.endswith(".lance"):
            dataset = lance.dataset(args.train_data_dir)
        elif any(
            file.suffix == ".lance" for file in Path(args.train_data_dir).glob("*")
        ):
            lance_file = next(
                file
                for file in Path(args.train_data_dir).glob("*")
                if file.suffix == ".lance"
            )
            dataset = lance.dataset(str(lance_file))
        else:
            console.print("[yellow]Converting dataset to Lance format...[/yellow]")
            dataset = transform2lance(
                args.train_data_dir,
                output_name="dataset",
                save_binary=False,
                not_save_disk=False,
                tag="WDtagger",
            )
            console.print("[green]Dataset converted to Lance format[/green]")

    else:
        dataset = args.train_data_dir
        console.print("[green]Using existing Lance dataset[/green]")

    ort_sess, input_name, rating_tags, character_tags, general_tags = (
        load_model_and_tags(args)
    )

    # 处理标签
    if args.character_tag_expand:
        for i, tag in enumerate(character_tags):
            if tag.endswith(")"):
                tags = tag.split("(")
                character_tag = "(".join(tags[:-1])
                if character_tag.endswith("_"):
                    character_tag = character_tag[:-1]
                series_tag = tags[-1].replace(")", "")
                character_tags[i] = character_tag + args.caption_separator + series_tag

    if args.remove_underscore:
        rating_tags = [
            tag.replace("_", " ") if len(tag) > 3 else tag for tag in rating_tags
        ]
        general_tags = [
            tag.replace("_", " ") if len(tag) > 3 else tag for tag in general_tags
        ]
        character_tags = [
            tag.replace("_", " ") if len(tag) > 3 else tag for tag in character_tags
        ]

    # 处理标签替换
    if args.tag_replacement is not None:
        escaped_tag_replacements = args.tag_replacement.replace("\\,", "@@@@").replace(
            "\\;", "####"
        )
        tag_replacements = escaped_tag_replacements.split(";")
        for tag_replacement in tag_replacements:
            tags = tag_replacement.split(",")
            assert (
                len(tags) == 2
            ), f"tag replacement must be in the format of `source,target`: {args.tag_replacement}"
            source, target = [
                tag.replace("@@@@", ",").replace("####", ";") for tag in tags
            ]
            console.print(f"[blue]replacing tag: {source} -> {target}[/blue]")

            if source in general_tags:
                general_tags[general_tags.index(source)] = target
            elif source in character_tags:
                character_tags[character_tags.index(source)] = target
            elif source in rating_tags:
                rating_tags[rating_tags.index(source)] = target

    # 处理父子标签
    if args.remove_parents_tag:
        csv_file_path = Path(args.model_dir) / PARENTS_CSV
        if not csv_file_path.exists() or args.force_download:
            csv_file_path = Path(
                hf_hub_download(
                    repo_id="deepghs/danbooru_wikis_full",
                    filename=PARENTS_CSV,
                    local_dir=args.model_dir,
                    force_download=True,
                    force_filename=PARENTS_CSV,
                    repo_type="dataset",
                )
            )
            console.print(f"[blue]Downloaded {PARENTS_CSV} to {csv_file_path}[/blue]")
        else:
            console.print(f"[green]Using existing {PARENTS_CSV}[/green]")

        parent_to_child_map = {}
        with csv_file_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            line = [row for row in reader]
            header = line[0]
            rows = line[1:]
        assert (
            header[3] == "antecedent_name" and header[4] == "consequent_name"
        ), f"unexpected csv format: {header}"
        for row in rows[0:]:
            child_tag, parent_tag = row[3], row[4]
            if parent_tag not in parent_to_child_map:
                parent_to_child_map[parent_tag] = []
            parent_to_child_map[parent_tag].append(child_tag)

    # 使用 Lance 扫描器处理图像
    tag_freq = {}

    # 先计算图片总数
    total_images = len(
        dataset.to_table(
            columns=["mime", "captions"],
            filter=(
                "mime LIKE 'image/%'"
                if args.overwrite
                else "mime LIKE 'image/%' and (captions IS NULL OR array_length(captions) = 0)"
            ),
        )
    )

    # 然后创建带columns的scanner处理数据
    scanner = dataset.scanner(
        columns=["uris", "mime", "captions"],
        filter=(
            "mime LIKE 'image/%'"
            if args.overwrite
            else "mime LIKE 'image/%' and (captions IS NULL OR array_length(captions) = 0)"
        ),
        scan_in_order=True,
        batch_size=args.batch_size,
        batch_readahead=16,
        fragment_readahead=4,
        io_buffer_size=32 * 1024 * 1024,  # 32MB buffer
        late_materialization=True,
    )

    results = []

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

        for batch in scanner.to_batches():
            uris = batch["uris"].to_pylist()  # 获取文件路径

            # 使用并行处理加载和预处理图像
            batch_images = load_and_preprocess_batch(uris)

            if not batch_images:
                progress.update(task, advance=len(uris))
                continue

            # 处理批次
            probs = process_batch(batch_images, ort_sess, input_name)
            if probs is not None:
                for path, prob in zip(uris, probs):
                    # 获取高置信度的标签
                    found_tags = []
                    general_confidence = args.general_threshold or args.thresh
                    character_confidence = args.character_threshold or args.thresh

                    if args.use_rating_tags and not args.use_rating_tags_as_last_tag:
                        # 处理rating tags (前4个)
                        rating_pred = [
                            rating_tags[i]
                            for i, p in enumerate(prob[:4])
                            if p > args.thresh
                        ]
                        if rating_pred:
                            found_tags.extend(rating_pred)

                    # 处理general tags和character tags (第4个之后)
                    for i, p in enumerate(prob[4:]):
                        if i < len(general_tags) and p >= general_confidence:
                            found_tags.append(general_tags[i])
                        elif i >= len(general_tags):
                            char_idx = i - len(general_tags)
                            if (
                                char_idx < len(character_tags)
                                and p >= character_confidence
                            ):
                                if args.character_tags_first:
                                    found_tags.insert(0, character_tags[char_idx])
                                else:
                                    found_tags.append(character_tags[char_idx])

                    if args.use_rating_tags and args.use_rating_tags_as_last_tag:
                        # 处理rating tags (前4个)
                        rating_pred = [
                            rating_tags[i]
                            for i, p in enumerate(prob[:4])
                            if p > args.thresh
                        ]
                        if rating_pred:
                            found_tags.extend(rating_pred)

                    # 处理标签频率统计
                    if args.frequency_tags:
                        for tag in found_tags:
                            tag_freq[tag] = tag_freq.get(tag, 0) + 1

                    # 保存结果
                    output_path = Path(path).with_suffix(args.caption_extension)
                    if args.append_tags and output_path.exists():
                        with output_path.open("r", encoding="utf-8") as f:
                            existing_tags = f.read().strip()
                            found_tags = (
                                existing_tags.split(args.caption_separator) + found_tags
                            )
                    results.append((path, found_tags))

                    with output_path.open("w", encoding="utf-8") as f:
                        f.write(args.caption_separator.join(found_tags))

            progress.update(task, advance=len(batch["uris"].to_pylist()))

    if args.frequency_tags:
        console.print("\n[yellow]Tag frequencies:[/yellow]")
        for tag, freq in sorted(tag_freq.items(), key=lambda x: x[1], reverse=True):
            console.print(f"{tag}: {freq}")

    table = pa.table(
        {
            "uris": pa.array([str(path) for path, _ in results], type=pa.string()),
            "captions": pa.array(
                [caption for _, caption in results],
                type=pa.list_(pa.string()),
            ),
        }
    )

    dataset.merge_insert(on="uris").when_matched_update_all().execute(table)

    try:
        dataset.tags.create("WDtagger", 1)
    except:
        dataset.tags.update("WDtagger", 1)

    console.print("[green]Successfully updated dataset with new captions[/green]")


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
        default=DEFAULT_WD14_TAGGER_REPO,
        help="Repository ID for WD14 tagger model on Hugging Face",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="wd14_tagger_model",
        help="Directory to store WD14 tagger model",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force downloading WD14 tagger model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".txt",
        help="Extension for caption files",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.35,
        help="Default threshold for tag confidence",
    )
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=None,
        help="Threshold for general category tags (defaults to --thresh)",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=None,
        help="Threshold for character category tags (defaults to --thresh)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Skip processing images in subfolders",
    )
    parser.add_argument(
        "--remove_underscore",
        action="store_true",
        help="Replace underscores with spaces in output tags",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--undesired_tags",
        type=str,
        default="",
        help="Comma-separated list of tags to exclude from output",
    )
    parser.add_argument(
        "--frequency_tags",
        action="store_true",
        help="Show frequency of tags across all processed images",
    )
    parser.add_argument(
        "--append_tags",
        action="store_true",
        help="Append new tags to existing caption files instead of overwriting",
    )
    parser.add_argument(
        "--use_rating_tags",
        action="store_true",
        help="Add rating tags as the first tag",
    )
    parser.add_argument(
        "--use_rating_tags_as_last_tag",
        action="store_true",
        help="Add rating tags as the last tag",
    )
    parser.add_argument(
        "--character_tags_first",
        action="store_true",
        help="Insert character tags before general tags",
    )
    parser.add_argument(
        "--always_first_tags",
        type=str,
        default=None,
        help="Comma-separated list of tags to always put at the beginning (e.g. '1girl,1boy')",
    )
    parser.add_argument(
        "--caption_separator",
        type=str,
        default=", ",
        help="Separator for caption tags (include spaces if needed)",
    )
    parser.add_argument(
        "--tag_replacement",
        type=str,
        default=None,
        help="Tag replacements in format 'source1,target1;source2,target2'. Escape ',' and ';' with '\\'",
    )
    parser.add_argument(
        "--character_tag_expand",
        action="store_true",
        help="Expand character tags with parentheses (e.g. 'name_(series)' becomes 'name, series')",
    )
    parser.add_argument(
        "--remove_parents_tag",
        action="store_true",
        help="Remove parent tags when child tags are present (e.g. remove 'red flowers' if 'red rose' exists)",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh

    main(args)
