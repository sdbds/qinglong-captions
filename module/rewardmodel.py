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
from imscore.preference.model import SiglipPreferenceScorer, CLIPPreferenceScorer, CLIPScore
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
from imscore.vqascore.model import VQAScore
from imscore.cyclereward.model import CycleReward
from imscore.evalmuse.model import EvalMuse
from imscore.hpsv3.model import HPSv3

console = Console()

def preprocess_image(image):
    """将图片转为奖励模型需要的像素张量: [1, C, H, W], float32, [0,1]"""
    try:
        # 统一为 PIL Image RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL image, numpy array, or file path")

        arr = np.array(image)  # H W C, uint8
        # 转 torch，并归一化到 [0,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # 1 C H W
        return tensor
    except Exception as e:
        console.print(f"[red]preprocess_image error: {str(e)}[/red]")
        return None


def load_and_preprocess_batch(uris):
    """并行加载和预处理一批图像"""

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


def process_batch(pixel_tensors, model, prompts):
    """使用奖励模型计算分数（逐张处理，避免尺寸不一致拼接）。

    pixel_tensors: List[torch.Tensor]，每个形状为 [1, C, H, W]
    prompts: List[str]，与 pixel_tensors 对齐
    返回：list 或 np.ndarray，分数（logits）
    """
    try:
        if not pixel_tensors:
            return []
        out = []
        # 推断模型所在设备
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for px, pr in zip(pixel_tensors, prompts):
            # HPSv3 期望 images: list[Tensor]，内部会 ToPILImage 并在 prepare() 中把 batch 移到 self.device
            if isinstance(model, HPSv3):
                if isinstance(px, torch.Tensor):
                    px_list = [px.squeeze(0).to("cpu")]  # ToPILImage 需要 CPU tensor，且形状为 C H W
                else:
                    px_list = [px]
                s = model.score(px_list, [pr])
            else:
                # 其它模型通常接受 [B,C,H,W] 的张量；逐张处理时 B=1，需与模型设备一致
                if isinstance(px, torch.Tensor):
                    px = px.to(model_device)
                s = model.score(px, [pr])
            console.print(f"[bold green]Score: {s}[/bold green]")
            if isinstance(s, torch.Tensor):
                # 统一为标量或单元素数组
                s = s.detach().cpu().numpy()
                # 支持形如 [1] 或 [[1]] 的返回
                try:
                    s = float(s.squeeze())
                except Exception:
                    s = float(s.ravel()[0])
            elif isinstance(s, (list, tuple)):
                s = float(s[0])
            else:
                s = float(s)
            out.append(s)
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
        thresholds_cfg = cfg.get("reward_model", {}).get("quality_threshold", [])
    except Exception as e:
        console.print(f"[red]Failed to read config.toml: {e}[/red]")
        thresholds_cfg = []

    # 组装阈值与目录：按 score 升序排序；目录名用 name 将下划线替换为空格
    thresholds_cfg = sorted(
        [t for t in thresholds_cfg if isinstance(t, dict) and "name" in t and "score" in t],
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
        quality_dirs = [(n, s, Path(args.train_data_dir) / n.replace("_", " ")) for n, s in defaults]

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
    model.to(device=args.device)
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
                    prompts = [c if isinstance(c, str) and c.strip() else "" for c in raw_caps]
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
                for path, score in zip(eff_uris, scores):
                    detection_results.append((path, float(score)))

            progress.update(task, advance=len(batch["uris"].to_pylist()))

    # 统计数量
    total_count = len(detection_results)

    # 按配置阈值分配质量并创建软链接/拷贝
    if total_count:
        for path, score in detection_results:
            source_path = Path(path).absolute()
            relative_path = source_path.relative_to(Path(args.train_data_dir).absolute())

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
                    console.print(f"[yellow]Created copy instead of symlink for {path}[/yellow]")
                except Exception as copy_err:
                    console.print(f"[red]Failed to copy file: {copy_err}[/red]")

    # 按路径层次结构组织结果
    path_tree = {}
    for path, prob in detection_results:
        parts = Path(path).relative_to(Path(args.train_data_dir).absolute()).parts
        current = path_tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # 最后一层是文件名，存储分数
                current[part] = f"{prob:.4f} (score)"
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

    # 使用Pretty打印结果树
    console.print("\n[bold green]Reward Scores：[/bold green]")
    console.print(Pretty(path_tree, indent_guides=True, expand_all=True))
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

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    main(args)
