import argparse
import numpy as np
import time
from PIL import Image
from pathlib import Path
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
from rich.tree import Tree
import torch
from huggingface_hub import hf_hub_download
from module.lanceImport import transform2lance
import concurrent.futures
import shutil
import json
import toml

from imscore.aesthetic.model import (
    ShadowAesthetic,
    LAIONAestheticScorer,
    SiglipAestheticScorer,
    CLIPAestheticScorer,
    Dinov2AestheticScorer,
)
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.preference.model import (
    SiglipPreferenceScorer,
    CLIPPreferenceScorer,
    CLIPScore,
)
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
from imscore.vqascore.model import VQAScore
from imscore.cyclereward.model import CycleReward
from imscore.evalmuse.model import EvalMuse
from imscore.hpsv3.model import HPSv3

console = Console(color_system="truecolor", force_terminal=True)


def preprocess_image(image):
    """将输入统一转换为 RGB 的 np.ndarray（uint8, H×W×C）。其余预处理交由各模型内部完成。

    返回：np.ndarray 或 None
    """
    try:
        # 统一为 PIL Image RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Input must be a PIL image, numpy array, or file path")
        return np.array(image)
    except Exception as e:
        console.print(f"[red]preprocess_image error: {str(e)}[/red]")
        return None


def load_and_preprocess_batch(uris):
    """并行加载和预处理一批图像，返回 List[np.ndarray] 和有效索引。

    每张图像：RGB、dtype=uint8、H×W×C。
    """

    def load_single_image(uri):
        try:
            # 直接传入路径，在preprocess_image中处理转换
            return preprocess_image(uri)
        except Exception as e:
            console.print(f"[red]Error processing {uri}: {str(e)}[/red]")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        batch_images = list(executor.map(load_single_image, uris))

    # 过滤掉加载失败的图像
    valid_images = [(i, img) for i, img in enumerate(batch_images) if img is not None]
    images = [img for _, img in valid_images]
    indices = [i for i, _ in valid_images]

    return images, indices


@torch.inference_mode()
def process_batch(pixel_tensors, model, prompts):
    """使用奖励模型计算分数（逐张处理，避免尺寸不一致拼接）。

    pixel_tensors: List[np.ndarray]（RGB、uint8、H×W×C）
    prompts: List[str]，与 pixel_tensors 对齐
    返回：list 或 np.ndarray，分数（logits）
    """
    try:
        if not pixel_tensors:
            return []
        out = []
        # 推断目标设备
        try:
            model_device = next(model.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for px, pr in zip(pixel_tensors, prompts):
            # 统一为 torch.FloatTensor [C,H,W] on model_device, 0..1
            try:
                if isinstance(px, torch.Tensor):
                    t = px
                    if t.dim() == 3 and t.shape[-1] == 3:
                        # HWC -> CHW
                        t = t.permute(2, 0, 1)
                    elif t.dim() == 4:
                        # [N,C,H,W] -> take first
                        t = t[0]
                    t = t.to(dtype=torch.float32)
                    if t.max() > 1.0:
                        t = t / 255.0
                elif isinstance(px, np.ndarray):
                    if px.ndim != 3 or px.shape[2] != 3:
                        raise ValueError(f"Expected ndarray HxWx3, got shape={px.shape}")
                    t = torch.from_numpy(px).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
                elif isinstance(px, Image.Image):
                    arr = np.array(px.convert("RGB"), dtype=np.float32)
                    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous() / 255.0
                else:
                    raise TypeError(f"Unsupported pixel type: {type(px)}")

                t = t.to(model_device, non_blocking=True)
                # 增加 batch 维度 -> [1,C,H,W]
                if t.dim() == 3:
                    t = t.unsqueeze(0)

                # 仅走单样本张量路径，满足 PickScorer/CLIP 家族对 4D 的要求
                s = model.score(t, pr)
            except Exception:
                console.print(
                    f"[red]score(tensor single) failed[/red] model={type(model).__name__}, px_type={type(px).__name__}, pr_type={type(pr).__name__}, device={model_device}"
                )
                if isinstance(px, np.ndarray):
                    console.print(f"[yellow]ndarray shape={px.shape}, dtype={px.dtype}[/yellow]")
                elif isinstance(px, torch.Tensor):
                    console.print(f"[yellow]tensor shape={tuple(px.shape)}, dtype={px.dtype}, device={px.device}[/yellow]")
                # 最后兜底：再尝试原始对象（可能某些实现内部做转换）
                try:
                    s = model.score(px, pr)
                except Exception:
                    raise Exception
            console.print(f"[bold green]Score: {s}[/bold green]")
            if isinstance(s, torch.Tensor):
                s = (
                    float(s.detach().to("cpu").squeeze().item())
                    if s.numel() == 1
                    else float(s.detach().to("cpu").flatten()[0].item())
                )
            elif isinstance(s, (list, tuple)):
                s = float(s[0])
            else:
                s = float(s)
            out.append(s)

            del px, pr

            if torch.cuda.is_available() and (len(out) % 32 == 0):
                torch.cuda.empty_cache()

        return np.array(out, dtype=np.float32)
    except Exception as e:
        console.print(f"[red]Batch processing error: {str(e)}[/red]")
        return None


def load_model(args):
    """加载模型和标签"""
    registry = {
        "RE-N-Y/aesthetic-shadow-v2": ShadowAesthetic,
        "RE-N-Y/clipscore-vit-large-patch14": CLIPScore,
        "RE-N-Y/pickscore": PickScorer,
        "yuvalkirstain/PickScore_v1": PickScorer,
        "RE-N-Y/mpsv1": MPS,
        "RE-N-Y/hpsv21": HPSv2,
        "RE-N-Y/ImageReward": ImageReward,
        "RE-N-Y/laion-aesthetic": LAIONAestheticScorer,
        "NagaSaiAbhinay/CycleReward-Combo": CycleReward,
        "NagaSaiAbhinay/CycleReward-T2I": CycleReward,
        "NagaSaiAbhinay/CycleReward-I2T": CycleReward,
        "RE-N-Y/clip-t5-xxl": VQAScore,
        "RE-N-Y/evalmuse": EvalMuse,
        "RE-N-Y/hpsv3": HPSv3,
        "RE-N-Y/pickscore-siglip": SiglipPreferenceScorer,
        "RE-N-Y/pickscore-clip": CLIPPreferenceScorer,
        # imreward fidelity rating (pixel only)
        "RE-N-Y/imreward-fidelity_rating-siglip": SiglipAestheticScorer,
        "RE-N-Y/imreward-fidelity_rating-clip": CLIPAestheticScorer,
        "RE-N-Y/imreward-fidelity_rating-dinov2": Dinov2AestheticScorer,
        # imreward overall rating (pixel only)
        "RE-N-Y/imreward-overall_rating-siglip": SiglipAestheticScorer,
        "RE-N-Y/imreward-overall_rating-clip": CLIPAestheticScorer,
        "RE-N-Y/imreward-overall_rating-dinov2": Dinov2AestheticScorer,
        # AVA dataset (pixel only)
        "RE-N-Y/ava-rating-clip-sampled-True": CLIPAestheticScorer,
        "RE-N-Y/ava-rating-clip-sampled-False": CLIPAestheticScorer,
        "RE-N-Y/ava-rating-siglip-sampled-True": SiglipAestheticScorer,
        "RE-N-Y/ava-rating-siglip-sampled-False": SiglipAestheticScorer,
        "RE-N-Y/ava-rating-dinov2-sampled-True": Dinov2AestheticScorer,
        "RE-N-Y/ava-rating-dinov2-sampled-False": Dinov2AestheticScorer,
    }

    cls = registry.get(args.repo_id)
    if cls is None:
        console.print(f"[red]Invalid model repo ID: {args.repo_id}[/red]")
        return None

    return cls.from_pretrained(args.repo_id)


def main(args):
    global console

    # 从 config/config.toml 读取质量阈值并创建对应文件夹
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "config.toml"
    try:
        cfg = toml.load(config_path)
        rm_cfg = cfg.get("reward_model", {}) if isinstance(cfg, dict) else {}

        # 新格式：reward_model.quality = [{ name, score, color }]
        quality_cfg = rm_cfg.get("quality")
        if isinstance(quality_cfg, list) and quality_cfg:
            # 保留原始顺序或按得分排序（与旧逻辑一致，按 score 升序）
            quality_cfg = sorted(
                [q for q in quality_cfg if isinstance(q, dict) and "name" in q and "score" in q],
                key=lambda x: float(x["score"]),
            )
            thresholds_cfg = [
                {"name": str(q["name"]), "score": float(q["score"])} for q in quality_cfg
            ]
            # 颜色序列与阈值一一对应；缺失则给默认
            default_palette = ["bold red", "bold yellow", "bold blue", "bold green"]
            colors_by_rank = [
                str(q.get("color", default_palette[min(i, len(default_palette) - 1)]))
                for i, q in enumerate(quality_cfg)
            ]
        else:
            # 旧格式回退：quality_threshold + colors_by_rank
            thresholds_cfg = rm_cfg.get("quality_threshold", [])
            colors_by_rank = rm_cfg.get(
                "colors_by_rank", ["bold red", "bold yellow", "bold blue", "bold green"]
            )
            if not isinstance(colors_by_rank, list) or not all(
                isinstance(c, str) and c for c in colors_by_rank
            ):
                colors_by_rank = ["bold red", "bold yellow", "bold blue", "bold green"]
    except Exception as e:
        console.print(f"[red]Failed to read config.toml: {e}[/red]")
        thresholds_cfg = []
        colors_by_rank = ["bold red", "bold yellow", "bold blue", "bold green"]

    # 组装阈值与目录：按 score 升序排序；目录名用 name 将下划线替换为空格
    thresholds_cfg = sorted(
        [
            t
            for t in thresholds_cfg
            if isinstance(t, dict) and "name" in t and "score" in t
        ],
        key=lambda x: float(x["score"]),
    )
    quality_dirs = []  # List[Tuple[name, score, Path]]
    for t in thresholds_cfg:
        name = str(t["name"])  # e.g. worst_quality
        score = float(t["score"])  # e.g. 3.0
        folder = Path(args.train_data_dir) / name.replace("_", " ")
        quality_dirs.append((name, score, folder))

    # 若配置缺失，回退到默认四档
    if not quality_dirs:
        defaults = [
            ("worst_quality", 3.0),
            ("bad_quality", 5.0),
            ("normal_quality", 7.5),
            ("best_quality", 9.0),
        ]
        quality_dirs = [
            (n, s, Path(args.train_data_dir) / n.replace("_", " ")) for n, s in defaults
        ]

    # 清理已有软链接并确保目录存在
    for _, _, root in quality_dirs:
        root.mkdir(parents=True, exist_ok=True)
        for symlink in root.rglob("*"):
            if symlink.is_symlink():
                symlink.unlink()

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
                tag="RewardModel",
            )
            console.print("[green]Dataset converted to Lance format[/green]")

    else:
        dataset = args.train_data_dir
        console.print("[green]Using existing Lance dataset[/green]")

    # 加载奖励模型
    model = load_model(args)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        inferred = None
        # 1) 优先 transformers 的配置
        torch_dtype_cfg = getattr(getattr(model, "config", None), "torch_dtype", None)
        if torch_dtype_cfg is not None:
            inferred = torch_dtype_cfg
        else:
            # 2) 回退到首个参数 dtype
            p0 = next(model.parameters(), None)
            if p0 is not None:
                inferred = p0.dtype

        # 3) 环境约束与兜底
        if inferred is None:
            inferred = torch.float16 if torch.cuda.is_available() else torch.float32
        if args.device == "cpu" and inferred in (torch.float16, torch.bfloat16):
            inferred = torch.float32
        if args.repo_id.endswith("hpsv3"):
            inferred = torch.bfloat16
    elif args.dtype == "float16" or args.dtype == "fp16":
        inferred = torch.float16
    elif args.dtype == "bfloat16" or args.dtype == "bf16":
        inferred = torch.bfloat16
    elif args.dtype == "float32" or args.dtype == "fp32":
        inferred = torch.float32
    else:
        inferred = torch.float32
    args.dtype = inferred
    console.print(f"[green]Using device: {args.device}[/green]")
    console.print(f"[green]Using dtype: {args.dtype}[/green]")
    model.to(device=args.device, dtype=args.dtype)
    model.eval()
    if model is None:
        return

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

            # prompts：优先使用 --prompt，否则使用 captions 字段（为空则置空字符串）
            if hasattr(args, "prompt") and args.prompt:
                prompts = [args.prompt] * len(uris)
            else:
                if "captions" in batch.schema.names:
                    raw_caps = batch["captions"].to_pylist()
                    captions_list = raw_caps[0] if raw_caps and len(raw_caps) > 0 else []
                    # 确保 captions 长度与 uris 匹配
                    if len(captions_list) >= len(uris):
                        prompts = [
                            c if isinstance(c, str) and c.strip() else ""
                            for c in captions_list[:len(uris)]
                        ]
                    else:
                        # captions 长度不足，用空字符串填充
                        prompts = (
                            [c if isinstance(c, str) and c.strip() else "" for c in captions_list] +
                            [""] * (len(uris) - len(captions_list))
                        )
                else:
                    prompts = [""] * len(uris)

            # 使用并行处理加载和预处理图像（像素张量）
            batch_pixels, valid_idx = load_and_preprocess_batch(uris)

            if not batch_pixels:
                progress.update(task, advance=len(uris))
                continue

            # 处理批次：计算奖励分数
            eff_prompts = [prompts[i] for i in valid_idx]
            eff_uris = [uris[i] for i in valid_idx]
            scores = process_batch(batch_pixels, model, eff_prompts)
            if scores is not None:
                for path, prompt, score in zip(eff_uris, eff_prompts, scores):
                    detection_results.append((path, prompt, float(score)))

            progress.update(task, advance=len(batch["uris"].to_pylist()))

    # 统计数量
    total_count = len(detection_results)

    # 按配置阈值分配质量并创建软链接/拷贝
    if total_count:
        for path, _, score in detection_results:
            source_path = Path(path).absolute()
            relative_path = source_path.relative_to(
                Path(args.train_data_dir).absolute()
            )

            # 找到第一个 score <= 阈值 的档位；若都不满足，归到最后一个（最高档）
            target_root = None
            for _, thr, root in quality_dirs:
                if score <= thr:
                    target_root = root
                    break
            if target_root is None:
                target_root = quality_dirs[-1][2]

            target_path = target_root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                target_path.symlink_to(source_path)
            except (FileExistsError, PermissionError) as e:
                console.print(f"[red]Unable to create symlink for {path}: {e}[/red]")
                try:
                    shutil.copy2(source_path, target_path)
                    console.print(
                        f"[yellow]Created copy instead of symlink for {path}[/yellow]"
                    )
                except Exception as copy_err:
                    console.print(f"[red]Failed to copy file: {copy_err}[/red]")

    # 按路径层次结构组织结果
    path_tree = {}
    for path, prompt, score in detection_results:
        parts = Path(path).relative_to(Path(args.train_data_dir).absolute()).parts
        current = path_tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # 最后一层是文件名，存储分数
                current[part] = f"{score:.4f}|{prompt}"
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

    # 使用 Tree 打印结果树（带颜色）
    console.print("\n[bold green]Reward Scores：[/bold green]")

    # 基于质量阈值选择颜色：按阈值档位由差到优映射为配置中的颜色序列

    def color_for_score(s: float) -> str:
        # 找到属于的档位索引
        idx = None
        for i, (_, thr, _) in enumerate(quality_dirs):
            if s <= thr:
                idx = i
                break
        if idx is None:
            idx = len(quality_dirs) - 1
        # 映射到颜色表
        return colors_by_rank[min(idx, len(colors_by_rank) - 1)]

    # 构建目录树
    root = Tree("[bold]Reward Scores[/]")
    nodes = {(): root}  # key: 累积路径元组 -> Tree 节点

    for path, prompt, score in detection_results:
        rel_parts = Path(path).relative_to(Path(args.train_data_dir).absolute()).parts
        acc = ()
        parent = root
        for i, part in enumerate(rel_parts):
            acc = acc + (part,)
            if acc not in nodes:
                if i < len(rel_parts) - 1:
                    nodes[acc] = parent.add(part)
                else:
                    style = color_for_score(score)
                    # 叶子节点：文件名 + 颜色分数 + prompt
                    nodes[acc] = parent.add(
                        f"{part}  [bold {style}]{score:.4f}[/] | {prompt}"
                    )
            parent = nodes[acc]

    console.print(root)
    # 保存检测结果树到JSON文件
    result_json_path = Path(args.train_data_dir) / "reward_scores.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(path_tree, f, ensure_ascii=False, indent=2)
    console.print(f"[bold green]Results saved to:[/bold green] {result_json_path}")

    # 打印简单统计
    if total_count:
        console.print(f"[bold]Total scored images:[/bold] {total_count}")


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
        default="RE-N-Y/pickscore",
        help="Repository ID for Reward Model on Hugging Face",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt used by reward models (if not provided, captions column will be used)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16", "fp16", "fp32", "bf16"],
        help="Data type for inference",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    main(args)
