import argparse
import numpy as np
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
import cv2
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from module.lanceImport import transform2lance
import concurrent.futures
import re
import json
import toml
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

console = Console()

# --- Config Loading ---
# Load configuration from config.toml
# This allows for dynamic configuration of the series exclusion list.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.toml"
SERIES_EXCLUDE_LIST = set()
try:
    if CONFIG_PATH.exists():
        config = toml.load(CONFIG_PATH)
        SERIES_EXCLUDE_LIST = set(
            config.get("wdtagger", {}).get("series_exclude_list", [])
        )
    else:
        console.print(
            f"[yellow]Config file not found at {CONFIG_PATH}, using default empty exclude list.[/yellow]"
        )
except Exception as e:
    console.print(
        f"[red]Error loading config file: {e}, using default empty exclude list.[/red]"
    )
# --- End Config Loading ---


@dataclass
class LabelData:
    """A data structure for holding tag information, designed for compatibility and flexibility."""

    names: List[str]
    # A dictionary where keys are category names (e.g., 'rating', 'general')
    # and values are numpy arrays of tag indices for that category.
    category_indices: Dict[str, np.ndarray]
    # A dictionary mapping tag index to its category name (lowercase)
    tag_index_to_category: Dict[int, str]

    def __post_init__(self):
        """Create direct attribute access for backward compatibility with official code."""
        # Provide direct attribute access for official code compatibility
        for category, indices in self.category_indices.items():
            setattr(self, category, indices)

    def get_tags_by_category(self, category: str) -> List[str]:
        """Get a list of tags for a given category."""
        indices = self.category_indices.get(category.lower(), np.array([], dtype=np.int64))
        return [self.names[i] for i in indices if i < len(self.names) and self.names[i]]


# from wd14 tagger
IMAGE_SIZE = 448

DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
FILES = ["model.onnx", "selected_tags.csv"]
CL_FILES = ["cl_tagger_1_02/model.onnx", "cl_tagger_1_02/tag_mapping.json"]
CSV_FILE = "selected_tags.csv"
JSON_FILE = "cl_tagger_1_02/tag_mapping.json"
PARENTS_CSV = "tag_implications.csv"


def preprocess_image(image, is_cl_tagger=False):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # 使用更高效的方式计算填充
    h, w = image.shape[:2]
    size = max(h, w)

    # 使用单个 pad 操作替代多次计算
    pad_y, pad_x = size - h, size - w
    pad_t, pad_l = pad_y // 2, pad_x // 2
    pad_b, pad_r = pad_y - pad_t, pad_x - pad_l

    # 检查是否可以使用CUDA
    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

    if use_gpu and size > 1024:
        # 转移到GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        # 使用GPU进行填充操作
        gpu_image = cv2.cuda.copyMakeBorder(
            gpu_image,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        if size > IMAGE_SIZE:
            # GPU调整大小
            gpu_image = cv2.cuda.resize(
                gpu_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
            )
        else:
            image = Image.fromarray(image)
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            image = np.array(image)

        # 下载回CPU
        image = gpu_image.download()
    else:
        # 使用更高效的填充
        image = cv2.copyMakeBorder(
            image,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        if size > IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_AREA)
        else:
            image = Image.fromarray(image)
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            image = np.array(image)

    if is_cl_tagger:
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = image.astype(np.float32) / 255.0
        # Apply normalization with mean=0.5, std=0.5
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std
    else:
        image = image.astype(np.float32)

    return image


def load_and_preprocess_batch(uris, is_cl_tagger=False):
    """并行加载和预处理一批图像"""

    def load_single_image(uri):
        try:
            return preprocess_image(Image.open(uri).convert("RGB"), is_cl_tagger)
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
        # Apply sigmoid (outputs are likely logits)
        # Use a stable sigmoid implementation
        def stable_sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -30, 30)))  # Clip to avoid overflow
        
        probs = stable_sigmoid(outputs[0])
        return probs
    except Exception as e:
        console.print(f"[red]Batch processing error: {str(e)}[/red]")
        return None


def get_tags_official(
    probs,
    labels: LabelData,
    gen_threshold,
    char_threshold,
    use_rating_tags,
    use_quality_tags,
    use_model_tags,
    processed_names=None,
):
    """官方兼容的标签处理函数"""
    tag_names = processed_names if processed_names is not None else labels.names
    result = {
        "rating": [],
        "general": [],
        "character": [],
        "copyright": [],
        "meta": [],
        "quality": [],
        "model": [],
    }

    # --- Pick-highest categories (rating, quality) ---
    pick_highest_categories = []
    if use_rating_tags:
        pick_highest_categories.append("rating")
    if use_quality_tags:
        pick_highest_categories.append("quality")
    if use_model_tags:
        pick_highest_categories.append("model")

    for category_name in pick_highest_categories:
        category_indices = labels.category_indices.get(category_name)
        if category_indices is not None and len(category_indices) > 0:
            valid_indices = category_indices[category_indices < len(probs)]
            if len(valid_indices) > 0:
                category_probs = probs[valid_indices]
                best_local_idx = np.argmax(category_probs)
                confidence = category_probs[best_local_idx]
                global_idx = valid_indices[best_local_idx]
                tag_name = tag_names[global_idx]
                result[category_name].append((tag_name, float(confidence)))

    # --- Above-threshold categories (general, character, etc.) ---
    # Dynamically create the map from the available categories in LabelData
    category_map = {}
    for category_name, category_indices in labels.category_indices.items():
        if category_name in pick_highest_categories:
            continue  # Skip categories that are already processed

        # Determine the threshold for the current category
        if category_name in ["character", "copyright", "artist"]:
            threshold = char_threshold
        else:
            threshold = gen_threshold
        category_map[category_name] = (category_indices, threshold)

    for category_name, (category_indices, threshold) in category_map.items():
        if len(category_indices) > 0:
            valid_indices = category_indices[category_indices < len(probs)]
            if len(valid_indices) == 0:
                continue

            # Directly filter probabilities and get the indices that pass the threshold
            mask = probs[valid_indices] >= threshold
            passed_indices = valid_indices[mask]

            for idx in passed_indices:
                # Check if global index is valid for names list
                if idx < len(tag_names) and tag_names[idx] is not None:
                    tag_name = tag_names[idx]
                    confidence = probs[idx]
                    result[category_name].append((tag_name, float(confidence)))

    # Sort all results by confidence
    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)

    return result


def load_model_and_tags(args):
    """加载模型和标签"""
    model_path = (
        Path(args.model_dir) / args.repo_id.replace("/", "_") / CL_FILES[0]
        if args.repo_id.startswith("cella110n/cl_tagger")
        else Path(args.model_dir) / args.repo_id.replace("/", "_") / FILES[0]
    )

    # 下载模型和标签文件
    if not model_path.exists() or args.force_download:
        # 选择正确的文件列表
        files_to_download = (
            CL_FILES if args.repo_id.startswith("cella110n/cl_tagger") else FILES
        )
        for file in files_to_download:
            file_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / file
            if not file_path.exists() or args.force_download:
                file_path = Path(
                    hf_hub_download(
                        repo_id=args.repo_id,
                        filename=file,
                        local_dir=Path(args.model_dir) / args.repo_id.replace("/", "_"),
                        force_download=args.force_download,
                    )
                )
                console.print(f"[blue]Downloaded {file} to {file_path}[/blue]")
            else:
                console.print(f"[green]Using existing {file}[/green]")

    # 加载标签
    if args.repo_id.startswith("cella110n/cl_tagger"):
        # 处理JSON格式的标签文件 - 使用官方兼容方式
        json_path = (
            Path(args.model_dir)
            / args.repo_id.replace("/", "_")
            / JSON_FILE
        )
        if not json_path.exists():
            raise Exception(f"Tags file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            tag_data = json.load(f)

        # Correctly handle potentially sparse tag indices from JSON.
        # The following logic creates a sparse list (`names`) that correctly maps
        # a model's output index to its tag name, even if indices are not contiguous.
        tag_data_int_keys = {int(k): v for k, v in tag_data.items()}
        idx_to_tag = {idx: data['tag'] for idx, data in tag_data_int_keys.items()}
        tag_to_category = {data['tag']: data['category'] for data in tag_data_int_keys.values()}

        # Create a sparse list for `names` to ensure correct index mapping.
        max_idx = max(idx_to_tag.keys())
        names = [None] * (max_idx + 1)
        for idx, tag in idx_to_tag.items():
            names[idx] = tag

        # Invert tag_to_category for faster lookups and prepare for categorization
        category_to_tags = {}
        for tag, category in tag_to_category.items():
            if category not in category_to_tags:
                category_to_tags[category] = []
            category_to_tags[category].append(tag)

        # Create a reverse map from tag to index for efficient lookup
        tag_to_idx = {tag: i for i, tag in idx_to_tag.items()}

        # Dynamically create category_indices from the data
        category_indices = {}
        for category, tags_in_category in category_to_tags.items():
            # Use the lowercase version of the category name as the key for consistency
            category_key = category.lower()
            indices = [tag_to_idx[tag] for tag in tags_in_category if tag in tag_to_idx]
            category_indices[category_key] = np.array(indices, dtype=np.int64)

        # Create a mapping from tag index to its category name (lowercase)
        tag_index_to_category = {
            idx: category.lower()
            for category, tags in category_to_tags.items()
            for tag in tags
            if (idx := tag_to_idx.get(tag)) is not None
        }

        # Create LabelData object
        label_data = LabelData(
            names=names,
            category_indices=category_indices,
            tag_index_to_category=tag_index_to_category,
        )

        total_tags = len(tag_data)
    else:
        # 处理CSV格式的标签文件 - 保持原有实现
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

        # 为CSV格式创建LabelData结构
        names = []
        rating_indices, general_indices, character_indices = [], [], []

        for i, row in enumerate(rows):
            tag_name = row[1]
            category = row[2]
            names.append(tag_name)

            if category == "9":  # rating
                rating_indices.append(i)
            elif category == "0":  # general
                general_indices.append(i)
            elif category == "4":  # character
                character_indices.append(i)

        # 创建LabelData对象
        # Create LabelData object for CSV format
        category_indices = {
            "rating": np.array(rating_indices, dtype=np.int64),
            "general": np.array(general_indices, dtype=np.int64),
            "character": np.array(character_indices, dtype=np.int64),
            # CSV format doesn't have these, so initialize as empty
            "copyright": np.array([], dtype=np.int64),
            "artist": np.array([], dtype=np.int64),
            "meta": np.array([], dtype=np.int64),
            "quality": np.array([], dtype=np.int64),
            "model": np.array([], dtype=np.int64),
        }
        tag_index_to_category = {}
        for cat, indices in category_indices.items():
            for idx in indices:
                tag_index_to_category[idx] = cat

        label_data = LabelData(
            names=names,
            category_indices=category_indices,
            tag_index_to_category=tag_index_to_category,
        )

        total_tags = len(rows)

    # 显示标签加载信息
    console.print(f"[blue]Tags loaded: {total_tags} total[/blue]")
    for category, indices in sorted(label_data.category_indices.items()):
        if len(indices) > 0:
            console.print(f"[blue]  - {category.capitalize()}: {len(indices)} tags[/blue]")

    console.print(f"[blue]Providers: {ort.get_available_providers()}[/blue]")

    # 设置推理提供者
    providers = []
    if "NvTensorRtRtxExecutionProvider" in ort.get_available_providers():
        providers.append("NvTensorRtRtxExecutionProvider")
        console.print("[green]Using TensorRT for inference[/green]")
        console.print("[yellow]compile may take a long time, please wait...[/yellow]")
    elif "TensorrtExecutionProvider" in ort.get_available_providers():
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

    if "NvTensorRtRtxExecutionProvider" in providers:
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        providers_with_options = [
            (
                "NvTensorRtRtxExecutionProvider",
                {
                    "nv_dump_subgraphs": False,
                    "nv_detailed_build_log": True,
                    "nv_cuda_graph_enable": True,
                },
            ),
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,  # Enable FP16 precision for faster inference
                    "trt_builder_optimization_level": 3,
                    "trt_max_partition_iterations": 1000,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": f"{Path(args.model_dir) / args.repo_id.replace('/', '_')}/trt_engines",
                    "trt_engine_hw_compatible": True,
                    "trt_force_sequential_engine_build": False,
                    "trt_context_memory_sharing_enable": True,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": f"{Path(args.model_dir) / args.repo_id.replace('/', '_')}",
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

    # TensorRT 优化
    elif "TensorrtExecutionProvider" in providers:
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
                    "trt_engine_cache_path": f"{Path(args.model_dir) / args.repo_id.replace('/', '_')}/trt_engines",
                    "trt_engine_hw_compatible": True,
                    "trt_force_sequential_engine_build": False,
                    "trt_context_memory_sharing_enable": True,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": f"{Path(args.model_dir) / args.repo_id.replace('/', '_')}",
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
    elif "CPUExecutionProvider" in providers:
        # CPU时启用多线程推理
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 启用并行执行
        sess_options.inter_op_num_threads = 8  # 设置线程数
        sess_options.intra_op_num_threads = 8  # 设置算子内部并行数
    else:
        providers_with_options = providers

    console.print(f"[cyan]Providers with options:[/cyan]")
    console.print(Pretty(providers_with_options, indent_guides=True, expand_all=True))
    start_time = time.time()
    ort_sess = ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers_with_options
    )
    input_name = ort_sess.get_inputs()[0].name

    parent_to_child_map = {}
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
        console.print(f"[green]Loaded {len(parent_to_child_map)} parent tags.[/green]")

    console.print(
        f"[green]Model loaded in {time.time() - start_time:.2f} seconds[/green]"
    )
    return ort_sess, input_name, label_data, parent_to_child_map


def process_tags(label_data: LabelData, args: argparse.Namespace) -> List[str]:
    """Process the master tag list based on user arguments before starting the main loop."""
    processed_names = label_data.names.copy()

    # 1. Handle undesired tags by nullifying them to preserve indices (must be first)
    if args.undesired_tags:
        undesired_set = {t.strip() for t in args.undesired_tags.split(",")}
        console.print(f"[blue]Undesired tags: {undesired_set}[/blue]")
        console.print(f"[blue]Excluding {len(undesired_set)} undesired tags...[/blue]")
        for i, name in enumerate(processed_names):
            if name in undesired_set:
                processed_names[i] = ""  # Set to empty string

    # 2. Handle tag replacements
    if args.tag_replacement:
        replacement_map = {}
        escaped_replacements = args.tag_replacement.replace("\\,", "<COMMA>").replace(
            "\\;", "<SEMICOLON>"
        )
        for pair in escaped_replacements.split(";"):
            parts = pair.split(",")
            if len(parts) == 2:
                old_tag = (
                    parts[0].strip().replace("<COMMA>", ",").replace("<SEMICOLON>", ";")
                )
                new_tag = (
                    parts[1].strip().replace("<COMMA>", ",").replace("<SEMICOLON>", ";")
                )
                if old_tag:
                    replacement_map[old_tag] = new_tag

        if replacement_map:
            console.print(f"[blue]Replacement map: {replacement_map}[/blue]")
            console.print(
                f"[blue]Applying {len(replacement_map)} tag replacements...[/blue]"
            )
            processed_names = [
                replacement_map.get(name, name) for name in processed_names
            ]

    # 2. Handle underscore replacement
    if args.remove_underscore:
        console.print("[blue]Removing underscores from tags...[/blue]")
        processed_names = [
            name.replace("_", " ") if len(name) > 3 else name
            for name in processed_names
        ]

    # 3. Handle character tag expansion
    if args.character_tag_expand:
        character_indices = label_data.category_indices.get("character", np.array([]))
        for i in character_indices:
            if i < len(processed_names):
                tag = processed_names[i]
                if tag and tag.endswith(")"):
                    parts = tag.split("(")
                    if len(parts) > 1:
                        character_name = "(".join(parts[:-1]).strip()
                        processed_names[i] = character_name

    return processed_names


def assemble_final_tags(
    tags_result: Dict[str, List[Tuple[str, float]]],
    args: argparse.Namespace,
    parent_to_child_map: Dict[str, List[str]],
    tag_freq: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Assemble and sort the final list of tags for a single image."""
    found_tags = []

    # Process all tags from the result, formatting them based on the threshold flag.
    all_tags_by_category = {}
    for category, tags_with_conf in tags_result.items():
        if not tags_with_conf:
            continue
        # Respect switches: skip disabled categories so they won't leak in via remaining categories
        if category == "quality" and not args.use_quality_tags:
            continue
        if category == "rating" and not args.use_rating_tags:
            continue
        if category == "model" and not args.use_model_tags:
            continue

        # Collapse to single best tag for quality (and rating if present), before formatting
        if category in ("quality", "rating", "model"):
            # tags_with_conf: List[Tuple[tag, confidence]]
            best_list = [max(tags_with_conf, key=lambda x: x[1])]
        else:
            best_list = tags_with_conf

        if args.add_tags_threshold:
            all_tags_by_category[category] = [f"{tag}:{conf:.2f}" for tag, conf in best_list if tag]
        else:
            all_tags_by_category[category] = [tag for tag, conf in best_list if tag]

    # Define tag groups
    rating_related_tags = ["rating", "quality", "meta", "model"]
    character_related_tags = ["character", "copyright", "artist"]

    # Assemble tags based on defined order and arguments
    if args.use_rating_tags and not args.use_rating_tags_as_last_tag:
        for category in rating_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))

    if args.character_tags_first:
        for category in character_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))
        found_tags.extend(all_tags_by_category.get("general", []))
    else:
        found_tags.extend(all_tags_by_category.get("general", []))
        for category in character_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))

    if args.use_rating_tags and args.use_rating_tags_as_last_tag:
        for category in rating_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))

    # Add any remaining categories that were not in the ordered list to ensure no data is lost.
    processed_categories = set(rating_related_tags) | set(character_related_tags) | {"general"}
    remaining_categories = set(all_tags_by_category.keys()) - processed_categories
    for category in sorted(list(remaining_categories)):
        found_tags.extend(all_tags_by_category[category])

    # Sorting: always_first_tags
    # Sorting by confidence if requested via frequency_tags flag
    if args.frequency_tags:
        # Create a dictionary of tags to their confidence scores for quick lookup
        confidence_map = {
            tag: conf for cat in tags_result.values() for tag, conf in cat
        }
        # Sort the found_tags list based on the confidence scores
        found_tags.sort(key=lambda tag: confidence_map.get(tag, 0.0), reverse=True)

    # Sorting: always_first_tags (should run after confidence sort to ensure they are first)
    if args.always_first_tags:
        always_first = [tag.strip() for tag in args.always_first_tags.split(",")]
        existing_first_tags = [tag for tag in always_first if tag in found_tags]
        other_tags = [tag for tag in found_tags if tag not in existing_first_tags]
        found_tags = existing_first_tags + other_tags

    # Filtering: remove_parents_tag
    if args.remove_parents_tag and parent_to_child_map:
        found_tags_set = set(found_tags)
        tags_to_remove = set()
        for parent_tag, child_tags in parent_to_child_map.items():
            if parent_tag in found_tags_set:
                if any(child_tag in found_tags_set for child_tag in child_tags):
                    tags_to_remove.add(parent_tag)
        if tags_to_remove:
            found_tags = [tag for tag in found_tags if tag not in tags_to_remove]

    # Additional filtering when remove_parents_tag is enabled:
    # Remove single-word tags whose noun root (last word) appears more than once
    # across the current tag set. This helps drop overly-generic single-word tags
    # when more specific multi-word tags with the same root exist.
    if args.remove_parents_tag:
        noun_root_counts = {}
        for tag in found_tags:
            parts = tag.split()
            if not parts:
                continue
            noun_root = parts[-1]
            noun_root_counts[noun_root] = noun_root_counts.get(noun_root, 0) + 1

        found_tags = [
            tag
            for tag in found_tags
            if not (
                len(tag.split()) == 1
                and noun_root_counts.get(tag.split()[-1], 0) > 1
            )
        ]

    # Update tag frequency (always track for stats)
    for tag in found_tags:
        # Use clean tag name for frequency counting (remove threshold suffix if present)
        clean_tag = tag.split(':')[0] if ':' in tag else tag
        tag_freq[clean_tag] = tag_freq.get(clean_tag, 0) + 1

    return found_tags


def assemble_tags_json(
    tags_result: Dict[str, List[Tuple[str, float]]],
    *,
    add_tags_threshold: bool,
    remove_parents_tag: bool,
    parent_to_child_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """Assemble tags per category for JSON output.

    Differences to `assemble_final_tags`:
    - Never consult any use_*_tags switches; include all categories present in `tags_result` as-is.
    - Preserve only two features:
      1) remove_parents_tag: remove parent tag if any of its children exist in the same image tag set
      2) add_tags_threshold: output string as "tag:0.97" when True, otherwise just "tag"
    - Keep all tags of each category; do not collapse to single best for rating/quality/model.

    Returns a dict mapping category -> list[str].
    """
    # 1) Build raw category mapping with explicit category coverage and optional collapsing
    category_to_tags: Dict[str, List[str]] = {}
    flat_tags_set: set = set()  # for parent removal

    all_categories = [
        "rating",
        "general",
        "character",
        "copyright",
        "artist",
        "meta",
        "quality",
        "model",
    ]

    for category in all_categories:
        tags_with_conf = tags_result.get(category, [])
        if not tags_with_conf:
            category_to_tags[category] = []
            continue

        # Collapse to best for rating/quality/model
        if category in ("quality", "rating", "model"):
            best_list = [max(tags_with_conf, key=lambda x: x[1])]
        else:
            best_list = tags_with_conf

        if add_tags_threshold:
            tags = [f"{t}:{c:.2f}" for t, c in best_list if t]
        else:
            tags = [t for t, _ in best_list if t]

        category_to_tags[category] = tags
        for t in tags:
            clean = t.split(":")[0] if ":" in t else t
            flat_tags_set.add(clean)

    # 2) Remove parent tags if any child present
    if remove_parents_tag and parent_to_child_map:
        tags_to_remove = set()
        for parent_tag, child_tags in parent_to_child_map.items():
            if parent_tag in flat_tags_set and any(child in flat_tags_set for child in child_tags):
                tags_to_remove.add(parent_tag)

        if tags_to_remove:
            # purge from each category list (consider threshold suffix preservation)
            for category, tags in list(category_to_tags.items()):
                kept = []
                for t in tags:
                    clean = t.split(":")[0] if ":" in t else t
                    if clean not in tags_to_remove:
                        kept.append(t)
                category_to_tags[category] = kept

    return category_to_tags


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

    # Load model, tags, and parent-child map
    ort_sess, input_name, label_data, parent_to_child_map = load_model_and_tags(args)

    # Process master tag list once before the loop
    processed_names = process_tags(label_data, args)

    # 使用 Lance 扫描器处理图像
    tag_freq = {}
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
    # Aggregate JSON for all images: { image_path: {category: [tags]} }
    all_json_tags: Dict[str, Dict[str, List[str]]] = {}

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
            uris = batch["uris"].to_pylist()

            # 使用并行处理加载和预处理图像
            is_cl_tagger = args.repo_id.startswith("cella110n/cl_tagger")
            batch_images = load_and_preprocess_batch(uris, is_cl_tagger)

            if not batch_images:
                progress.update(task, advance=len(uris))
                continue

            # 处理批次
            probs = process_batch(batch_images, ort_sess, input_name)
            general_confidence = args.general_threshold or args.thresh
            character_confidence = args.character_threshold or args.thresh
            if probs is not None:
                for path, prob in zip(uris, probs):

                    tags_result = get_tags_official(
                        prob,
                        label_data,
                        general_confidence,
                        character_confidence,
                        args.use_rating_tags,
                        args.use_quality_tags,
                        args.use_model_tags,
                        processed_names,  # Use the pre-processed names
                    )

                    found_tags = assemble_final_tags(
                        tags_result, args, parent_to_child_map, tag_freq
                    )

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

                    # Build JSON-ready categorized tags for this image
                    categorized = assemble_tags_json(
                        tags_result,
                        add_tags_threshold=args.add_tags_threshold,
                        remove_parents_tag=args.remove_parents_tag,
                        parent_to_child_map=parent_to_child_map,
                    )
                    all_json_tags[str(Path(path))] = categorized

            progress.update(task, advance=len(batch["uris"].to_pylist()))

    console.print("\n[yellow]Tag frequencies:[/yellow]")
    tag_classifier = TagClassifier()
    # Sort tags by frequency for printing
    sorted_tags = sorted(tag_freq.items(), key=lambda item: item[1], reverse=True)

    for tag, freq in sorted_tags:
        # Classify the tag to get its colored version
        classified_result = tag_classifier.classify([tag])
        # The result is a dict like {'category_id': ['[color]tag[/color]']}
        # Extract the colored tag safely
        colored_tag = tag  # Default fallback
        for tag_list in classified_result.values():
            if tag_list:  # Ensure the list is not empty
                colored_tag = tag_list[0]
                break
        console.print(f"{colored_tag}: {freq}")

    try:
        json_output_path = Path(args.train_data_dir) / "tags.json"
        with json_output_path.open("w", encoding="utf-8") as jf:
            json.dump(all_json_tags, jf, ensure_ascii=False, indent=2)
        console.print(f"[bold green]JSON saved to:[/bold green] {json_output_path}")
    except Exception as e:
        console.print(f"[red]Failed to save JSON: {e}[/red]")

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


def split_name_series(names: str) -> str:
    """Split and format character names and series information.

    Args:
        names: String containing character names and series info

    Returns:
        Formatted string with character names and series
    """
    name_list = []

    items = [item.strip().replace("_", ":") for item in names.split(",")]

    for item in items:
        if item.endswith(" (cosplay)"):
            item = item.replace(" (cosplay)", "")
        if ("c.c_") in item:
            item = item.replace("c.c_", "c.c.")
        if ("k:da") in item:
            item = item.replace("k:da", "k/da")
        if ("ranma 1:2") in item:
            item = item.replace("ranma 1:2", "ranma 1/2")
        # 匹配最后一对括号作为系列名
        match = re.match(r"(.*)\((.*?)\)$", item)
        if match and match.group(2).strip() not in SERIES_EXCLUDE_LIST:
            # 获取除最后一个括号外的所有内容作为名字
            full_name = match.group(1).strip()
            series = match.group(2).strip()

            # 保留名字中的其他括号
            name_list.append(f"<{full_name}> from ({series})")
        else:
            name_list.append(f"<{item}>")

    return ", ".join(name_list)


def format_description(text: str, tag_description: str = "") -> str:
    """Format description text with highlighting.

    Args:
        text: Input text to format
        tag_description: Tags to highlight in blue

    Returns:
        Formatted text with rich markup
    """
    # 高亮<>内的内容
    text = re.sub(r"<([^>]+)>", r"[magenta]\1[/magenta]", text)
    # 高亮()内的内容
    text = re.sub(r"\(([^)]+)\)", r"[dark_magenta]\1[/dark_magenta]", text)

    words = text.split()

    tagClassifier = TagClassifier()

    blue_words = set()

    # 高亮与tag_description匹配的单词
    if tag_description:
        # 将tag_description分割成单词列表
        tag_words = set(
            word.strip().lower()
            for word in re.sub(r"\d+", "", tag_description)
            .replace(",", " ")
            .replace(".", " ")
            .split()
            if word.strip()
        )
        for i, word in enumerate(words):
            highlight_word = re.sub(r"[^\w\s]", "", word.replace("'s", "").lower())
            if highlight_word in tag_words:
                blue_words.add(highlight_word)
                words[i] = tagClassifier.get_colored_tag(word)
        text = " ".join(words)

    # 统计高亮的次数
    highlight_count = 0
    has_green = False
    has_purple = False
    for word in words:
        if word.startswith("[magenta]") and word.endswith("[/magenta]"):
            has_green = True
        if word.startswith("[dark_magenta]") and word.endswith("[/dark_magenta]"):
            has_purple = True

    highlight_count = len(blue_words) + int(has_green) + int(has_purple)

    # 打印高亮率
    colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]
    rate = highlight_count / len(tag_description.replace(",", " ").split()) * 100
    # 将100%平均分配给7种颜色，每个颜色约14.3%
    color_index = min(int(rate / (100 / len(colors))), len(colors) - 1)
    color = colors[color_index]
    # 根据rate值决定是否使用粗体
    style = f"{color} bold" if rate > 50 else color
    highlight_rate = f"[{style}]{rate:.2f}%[/{style}]"

    return text, highlight_rate


class TagClassifier:
    """
    标签分类器，用于根据标签类别对标签进行分类，使用config.toml中定义的数字ID
    """

    def __init__(self):
        """
        初始化标签分类器
        """
        # 加载标签类型ID映射
        self.tag_type = self._load_tag_type()

        # 加载标签到类别的映射
        csv_path = Path("wd14_tagger_model") / "selected_tags_classified.csv"
        if not csv_path.exists():
            hf_hub_download(
                repo_id="deepghs/sankaku_tags_categorize_for_WD14Tagger",
                filename="selected_tags_classified.csv",
                local_dir="wd14_tagger_model",
                force_download=True,
                force_filename="selected_tags_classified.csv",
                repo_type="dataset",
            )

        self.tag_categories = self._load_tag_categories(csv_path)

    def _load_tag_type(self):
        """
        从config.toml加载标签类型ID映射

        Returns:
            dict: 类别名称到数字ID的映射
        """
        import toml
        from pathlib import Path

        # 使用硬编码的配置文件路径
        config_path = Path("config/config.toml")
        config = toml.load(config_path)
        # 新格式中，tag_type包含一个fields数组
        tag_type_fields = config["tag_type"]["fields"]
        # 创建一个字典，将ID映射到字段信息
        tag_type_dict = {}
        for field in tag_type_fields:
            tag_type_dict[str(field["id"])] = field
        return tag_type_dict

    def _load_tag_categories(self, csv_path):
        """
        加载标签到类别的映射

        Returns:
            dict: 标签到类别的映射字典
        """
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            tag_categories = {
                row["name"].replace("_", " "): row["category"] for row in reader
            }
        return tag_categories

    def classify(self, tags):
        """
        对标签列表进行分类，使用config.toml中定义的数字ID作为类别键

        Args:
            tags (list): 需要分类的标签列表

        Returns:
            dict: 按类别ID分组的标签字典，键为类别ID，值为带颜色格式的标签列表
        """
        colored_tags = {}
        for tag in tags:
            # 获取标签的类别名称，默认为'general'
            category_id = self.tag_categories.get(tag.lower(), "0")
            # 获取该类别的完整信息（包括颜色）
            category_info = self.tag_type.get(category_id)

            # 如果类别不存在，使用general（通用）类别
            if not category_info:
                color = "orange3"
            else:
                color = category_info["color"]

            # 使用rich格式添加颜色
            colored_tag = f"[{color}]{tag}[/{color}]"

            if category_id not in colored_tags:
                colored_tags[category_id] = []
            colored_tags[category_id].append(colored_tag)

        return colored_tags

    def get_colored_tag(self, tag):
        """
        对单个标签进行分类，返回带颜色格式的标签

        Args:
            tag (str): 需要分类的标签

        Returns:
            str: 带颜色格式的标签
        """
        # 获取标签的类别名称，默认为'0'(general)
        addtion = ""
        if tag.endswith(","):
            tag = tag[:-1]
            addtion = ","
        elif tag.endswith("."):
            tag = tag[:-1]
            addtion = "."

        category_id = self.tag_categories.get(tag.lower(), "0")
        # 获取该类别的完整信息（包括颜色）
        category_info = self.tag_type.get(category_id)

        # 如果类别不存在，使用orange3颜色
        if not category_info:
            color = "orange3"
        else:
            color = category_info["color"]

        # 使用rich格式添加颜色
        colored_tag = f"[{color}]{tag}[/{color}]{addtion}"

        return colored_tag


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
        "--undesired_tags",
        type=str,
        default="",
        help="Comma-separated list of tags to exclude from output",
    )
    parser.add_argument(
        "--frequency_tags",
        action="store_true",
        help="Sort final tags by confidence score instead of default order.",
    )
    parser.add_argument(
        "--add_tags_threshold",
        action="store_true",
        help="Add confidence threshold after each tag in output",
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
        "--use_quality_tags",
        action="store_true",
        help="Add quality tags to the output.",
    )
    parser.add_argument(
        "--use_model_tags",
        action="store_true",
        help="Add model tags to the output.",
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
        help="Remove parent tags if a child tag is present (e.g., remove 'uniform' if 'school_uniform' is present).",
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
