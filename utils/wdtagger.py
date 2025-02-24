import argparse
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import csv
import lance
import pyarrow as pa
import pyarrow.compute as pc
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
import torch
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from module.lanceImport import transform2lance

console = Console()

# from wd14 tagger
IMAGE_SIZE = 448

DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
FILES = ["model.onnx", "selected_tags.csv"]
CSV_FILE = "selected_tags.csv"
PARENTS_CSV = "tag_implications.csv"


def preprocess_image(image):
    """预处理图像"""
    # Convert BGR to RGB if needed
    if image.mode == "RGB":
        image = ImageOps.invert(
            ImageOps.invert(image)
        )  # No-op to ensure we have a copy
    else:
        image = image.convert("RGB")

    # Resize and pad the image while maintaining aspect ratio
    image = ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.LANCZOS)

    return image


def process_batch(images, session, input_name):
    """处理图像批次"""
    try:
        # 准备批处理数据并转换为float32
        batch_data = np.stack([np.array(img, dtype=np.float32) for img in images])
        # 执行推理
        outputs = session.run(None, {input_name: batch_data})
        return outputs[0]
    except Exception as e:
        console.print(f"[red]Batch processing error: {str(e)}[/red]")
        return None


def load_model_and_tags(args, force_download=False):
    """加载模型和标签"""
    model_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / "model.onnx"

    # 下载模型和标签文件
    if not model_path.exists() or force_download:
        for file in FILES:
            file_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / file
            if not file_path.exists() or force_download:
                file_path = Path(
                    hf_hub_download(
                        repo_id=args.repo_id,
                        filename=file,
                        local_dir=file_path.parent,
                        force_download=force_download,
                        force_filename=file,
                    )
                )
                print(f"Downloaded {file} to {file_path}")
            else:
                print(f"Using existing {file}")

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

    print(
        f"Tags loaded: {len(rows)} total, {len(rating_tags)} rating, {len(character_tags)} character, {len(general_tags)} general"
    )

    # 设置推理提供者
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
        print("Using CUDA for inference")
    elif "ROCMExecutionProvider" in ort.get_available_providers():
        providers.append("ROCMExecutionProvider")
        print("Using ROCm for inference")
    elif "OpenVINOExecutionProvider" in ort.get_available_providers():
        providers = [("OpenVINOExecutionProvider", {"device_type": "GPU_FP32"})]
        print("Using OpenVINO for inference")
    else:
        providers.append("CPUExecutionProvider")
        print("Using CPU for inference")

    # 加载模型
    print("Loading model...")
    if not model_path.exists():
        raise Exception(
            f"ONNX model not found: {model_path}. Please redownload with --force_download"
        )

    try:
        ort_sess = ort.InferenceSession(str(model_path), providers=providers)
        input_name = ort_sess.get_inputs()[0].name
    except Exception as e:
        raise Exception(f"Failed to create inference session: {str(e)}")
    print("Model loaded")

    return ort_sess, input_name, rating_tags, character_tags, general_tags


def main(args):
    # 初始化 Lance 数据集
    if not isinstance(args.train_data_dir, lance.LanceDataset):
        if args.train_data_dir.endswith(".lance"):
            dataset = lance.dataset(args.train_data_dir)
        else:
            console.print("[yellow]Converting dataset to Lance format...[/yellow]")
            dataset = transform2lance(
                args.train_data_dir,
                output_name="dataset",
                save_binary=True,
                not_save_disk=False,
            )
            console.print("[green]Dataset converted to Lance format[/green]")

    else:
        dataset = args.train_data_dir
        console.print("[green]Using existing Lance dataset[/green]")

    # 检查模型位置
    model_location = Path(args.model_dir)
    if not model_location.exists() or args.force_download:
        model_location.parent.mkdir(parents=True, exist_ok=True)
        console.print(
            f"[yellow]Downloading WD14 tagger model from HuggingFace. ID: {args.repo_id}[/yellow]"
        )
        ort_sess, input_name, rating_tags, character_tags, general_tags = (
            load_model_and_tags(args, args.force_download)
        )
    else:
        console.print("[green]Using existing WD14 tagger model[/green]")
        ort_sess, input_name, rating_tags, character_tags, general_tags = (
            load_model_and_tags(args, False)
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
            console.print(f"replacing tag: {source} -> {target}")

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
            print(f"Downloaded {PARENTS_CSV} to {csv_file_path}")
        else:
            print(f"Using existing {PARENTS_CSV}")

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
    # 然后创建带columns的scanner处理数据
    scanner = dataset.scanner(
        columns=["uris", "mime"], 
        scan_in_order=True, 
        batch_size=args.batch_size,
        filter="mime LIKE 'image/%'"  # 只处理图像类型的文件
    )

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing images...", total=scanner.to_table().num_rows)

        for batch in scanner.to_batches():
            batch_paths = [Path(uri) for uri in batch["uris"].to_pylist()]
            batch_images = [preprocess_image(Image.open(path)) for path in batch_paths]

            # 处理批次
            probs = process_batch(batch_images, ort_sess, input_name)
            if probs is not None:
                for path, prob in zip(batch_paths, probs):
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
                            if char_idx < len(character_tags) and p >= character_confidence:
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
                    output_path = path.with_suffix(args.caption_extension)
                    if args.append_tags and output_path.exists():
                        with output_path.open("r", encoding="utf-8") as f:
                            existing_tags = f.read().strip()
                            found_tags = (
                                existing_tags.split(args.caption_separator) + found_tags
                            )
                    results.append((path, found_tags))

                    with output_path.open("w", encoding="utf-8") as f:
                        f.write(args.caption_separator.join(found_tags))

            progress.update(task, advance=len(batch_paths))

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
        dataset.tags.create("wdtagger", 1)
    except:
        dataset.tags.update("wdtagger", 1)

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
        default=1,
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
        "--recursive",
        action="store_true",
        help="Search for images in subfolders recursively",
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
