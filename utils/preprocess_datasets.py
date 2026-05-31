# /// script
# dependencies = [
#   "setuptools",
#   "pillow>=11.3",
#   "pyarrow>=14.0.1",
#   "rich>=13.5.0",
#   "imageio>=2.31.1",
#   "imageio-ffmpeg>=0.4.8",
#   "opencv-contrib-python; sys_platform == 'win32'",
#   "opencv-contrib-python; sys_platform == 'linux'",
#   "torch==2.11.0",
#   "toml",
# ]
# ///
import argparse
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import torch
import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config.config import get_supported_extensions
from utils.console_util import print_exception
from utils.stream_util import calculate_dimensions

# Global console for general script-level logging if needed outside the class
# However, the class will have its own console instance for its operations.
global_console = Console(color_system="truecolor", force_terminal=True)


_MATCHER_BACKEND_ALIASES = {
    "affine_steerers": "affine-steerers",
    "affine-steerers": "affine-steerers",
    "xfeat": "xfeat",
    "orb": "orb",
}


_GRAYSCALE_FRIENDLY_FORMATS = {"jpeg", "jpg", "png", "avif"}


@dataclass(frozen=True)
class ResizePlan:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    original_mode: str
    original_format: str
    requires_resize: bool
    requires_mode_conversion: bool
    requires_transparent_crop: bool

    @property
    def can_skip_pixel_decode(self) -> bool:
        return (
            not self.requires_resize
            and not self.requires_mode_conversion
            and not self.requires_transparent_crop
        )


@dataclass(frozen=True)
class ProcessImageResult:
    ok: bool
    skipped: bool = False
    resized: bool = False
    converted: bool = False
    cropped: bool = False
    error: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class ResizeBatchSummary:
    skipped: int = 0
    resized: int = 0
    converted: int = 0
    cropped: int = 0
    failed: int = 0

    def add(self, result: ProcessImageResult | bool) -> None:
        if isinstance(result, bool):
            if not result:
                self.failed += 1
            return

        if not result.ok:
            self.failed += 1
        if result.skipped:
            self.skipped += 1
        if result.resized:
            self.resized += 1
        if result.converted:
            self.converted += 1
        if result.cropped:
            self.cropped += 1


class StageTimer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._seconds: dict[str, float] = {}

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._seconds.clear()

    def add(self, stage: str, seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._seconds[stage] = self._seconds.get(stage, 0.0) + seconds

    def snapshot(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        with self._lock:
            return dict(self._seconds)


def _requires_mode_conversion(mode: str, original_format: str) -> bool:
    if mode not in ["RGB", "L"]:
        return True
    return mode == "L" and original_format not in _GRAYSCALE_FRIENDLY_FORMATS


def _build_resize_plan(
    pil_image: Image.Image,
    image_path: str,
    *,
    max_short_edge: int | None,
    max_long_edge: int | None,
    max_pixels: int | None,
    crop_transparent: bool,
) -> ResizePlan:
    original_width, original_height = pil_image.size
    target_width, target_height = calculate_dimensions(
        original_width,
        original_height,
        max_long_edge=max_long_edge,
        max_short_edge=max_short_edge,
        max_pixels=max_pixels,
    )
    original_format = Path(image_path).suffix.lower().lstrip(".")

    return ResizePlan(
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
        original_mode=pil_image.mode,
        original_format=original_format,
        requires_resize=(original_width, original_height) != (target_width, target_height),
        requires_mode_conversion=_requires_mode_conversion(pil_image.mode, original_format),
        requires_transparent_crop=crop_transparent and pil_image.mode == "RGBA",
    )


def _has_cuda() -> bool:
    cuda_module = getattr(cv2, "cuda", None)
    if cuda_module is None or not hasattr(cuda_module, "getCudaEnabledDeviceCount"):
        return False
    return cuda_module.getCudaEnabledDeviceCount() > 0


def _choose_matcher_backend(preferred: str) -> str:
    normalized = (preferred or "auto").strip().lower()
    if normalized == "auto":
        return "affine-steerers" if _has_cuda() else "xfeat"
    return _MATCHER_BACKEND_ALIASES.get(normalized, normalized)


def _load_vismatch_matcher(name: str, device: str):
    from vismatch import get_matcher

    return get_matcher(name, device=device)


def _build_matcher(preferred: str) -> dict[str, Any]:
    name = _choose_matcher_backend(preferred)
    if name == "orb":
        return {"kind": "orb", "name": "orb", "matcher": None}

    try:
        device = "cuda" if _has_cuda() else "cpu"
        return {
            "kind": "vismatch",
            "name": name,
            "matcher": _load_vismatch_matcher(name, device=device),
        }
    except Exception:
        return {"kind": "orb", "name": "orb", "matcher": None}


def _extract_matched_points(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    src_pts = np.float32(result["matched_kpts0"]).reshape(-1, 1, 2)
    dst_pts = np.float32(result["matched_kpts1"]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def _to_vismatch_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if not isinstance(image, np.ndarray):
        return image

    array = image
    if array.ndim != 3:
        return image

    if array.shape[-1] in (3, 4):
        rgb_array = array[..., :3]
    elif array.shape[0] in (3, 4):
        rgb_array = np.moveaxis(array[:3], 0, -1)
    else:
        return image

    if rgb_array.dtype != np.uint8:
        max_value = float(np.max(rgb_array)) if rgb_array.size else 0.0
        if max_value <= 1.0:
            rgb_array = np.clip(rgb_array, 0.0, 1.0) * 255.0
        else:
            rgb_array = np.clip(rgb_array, 0.0, 255.0)
        rgb_array = rgb_array.astype(np.uint8)

    return Image.fromarray(np.ascontiguousarray(rgb_array), mode="RGB")


def _match_points_with_backend(
    source_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    preferred: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    matcher_info = _build_matcher(preferred)
    if matcher_info["kind"] != "vismatch":
        return None, None, matcher_info["name"]

    result = matcher_info["matcher"](
        _to_vismatch_image(source_rgb),
        _to_vismatch_image(reference_rgb),
    )
    src_pts, dst_pts = _extract_matched_points(result)
    return src_pts, dst_pts, matcher_info["name"]


def _transform_candidates(transform_type: str) -> tuple[str, ...]:
    normalized = (transform_type or "none").strip().lower()
    if normalized == "none":
        return ()
    if normalized == "affine":
        return ("affine_partial", "affine")
    if normalized == "homography":
        return ("homography",)
    return ("affine_partial", "affine", "homography")


def _estimate_transform_from_points(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    transform_type: str,
) -> tuple[Any, str]:
    num_points = 0 if src_pts is None else len(src_pts)
    ransac_method = getattr(cv2, "RANSAC", 0)

    for candidate in _transform_candidates(transform_type):
        if candidate == "affine_partial":
            if num_points < 3:
                continue
            M, _ = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=ransac_method,
                ransacReprojThreshold=10.0,
            )
        elif candidate == "affine":
            if num_points < 3:
                continue
            M, _ = cv2.estimateAffine2D(
                src_pts,
                dst_pts,
                method=ransac_method,
                ransacReprojThreshold=10.0,
            )
        elif candidate == "homography":
            if num_points < 4:
                continue
            M, _ = cv2.findHomography(src_pts, dst_pts, ransac_method, 10.0)
        else:
            continue

        if M is not None:
            return M, candidate

    return None, ""


class ImageProcessor:
    """
    A class to handle batch image processing tasks such as resizing.
    """

    def __init__(
        self,
        recursive: bool = False,
        max_workers: int = None,
        console: Console = None,
        transform_type: str = "auto",
        matcher_backend: str = "auto",
        bg_color: Tuple[int, int, int] = (255, 255, 255),  # Default to white
        crop_transparent: bool = False,
        profile: bool = False,
    ):
        """
        Initializes the ImageProcessor.

        Args:
            recursive: Whether to process images in subdirectories.
            max_workers: The maximum number of worker threads for parallel processing.
            console: An optional Rich Console instance for output.
            transform_type: The type of transformation for alignment.
            matcher_backend: The matcher backend preference for alignment.
            bg_color: RGB background color for padding.
            crop_transparent: Whether to crop transparent borders from RGBA images.
            profile: Whether to collect and print resize-only stage timings.
        """
        self.recursive = recursive
        self.max_workers = max_workers if max_workers is not None else 16  # Default to 16 if None
        self.console = console if console else Console(color_system="truecolor", force_terminal=True)
        self.image_extensions = get_supported_extensions("image")
        self.transform_type = transform_type
        self.matcher_backend = matcher_backend
        self.bg_color = bg_color
        self.crop_transparent = crop_transparent
        self.profile = profile
        self.stage_timer = StageTimer(enabled=profile)

    def _add_stage_time(self, stage: str, start_time: float) -> None:
        self.stage_timer.add(stage, time.perf_counter() - start_time)

    def _print_resize_summary(self, summary: ResizeBatchSummary) -> None:
        self.console.print("[blue]Resize summary:[/blue]")
        self.console.print(f"  skipped without pixel decode: {summary.skipped}")
        self.console.print(f"  resized: {summary.resized}")
        self.console.print(f"  mode converted: {summary.converted}")
        self.console.print(f"  transparent cropped: {summary.cropped}")
        self.console.print(f"  failed: {summary.failed}")

    def _print_profile_summary(self, total_images: int) -> None:
        if not self.profile:
            return

        snapshot = self.stage_timer.snapshot()
        if not snapshot:
            return

        self.console.print("[blue]Resize profile:[/blue]")
        for stage in ("scan", "open_and_plan", "decode_convert_crop", "resize", "save"):
            seconds = snapshot.get(stage, 0.0)
            average = seconds / total_images if total_images else 0.0
            self.console.print(f"  {stage}: {seconds:.3f}s total, {average:.4f}s/image")

    def _save_pil_image(
        self,
        pil_image: Image.Image,
        save_path: str,
    ):
        """
        Helper method to save a PIL image, attempting to preserve original format and quality.

        Args:
            pil_image: The PIL Image to save.
            save_path: The path to save the image to.
        """
        original_format = Path(save_path).suffix.lower().lstrip(".")
        original_quality = None
        if hasattr(pil_image, "info") and "quality" in pil_image.info:
            original_quality = pil_image.info.get("quality")

        try:
            save_quality = original_quality if original_quality is not None else 95
            # Ensure image is in RGB before saving, especially for formats like JPEG
            save_image = pil_image
            if original_format in ["jpg", "jpeg"] and pil_image.mode not in [
                "RGB",
                "L",
            ]:
                # Allow grayscale 'L' for jpg, otherwise convert to RGB
                if pil_image.mode == "RGBA":
                    self.console.print(f"[yellow]Converting RGBA to RGB for saving {Path(save_path).name} as JPEG.[/yellow]")
                    save_image = pil_image.convert("RGB")
                elif pil_image.mode != "L":  # if not L and not RGB (e.g. P, LA)
                    self.console.print(
                        f"[yellow]Converting mode {pil_image.mode} to RGB for saving {Path(save_path).name} as JPEG.[/yellow]"
                    )
                    save_image = pil_image.convert("RGB")

            if original_format in ["jpg", "jpeg"]:
                save_image.save(
                    str(save_path), quality=save_quality, subsampling=0
                )  # Added subsampling=0 for higher quality Chroma
            elif original_format in ["webp"]:
                save_image.save(str(save_path), quality=save_quality)
            elif original_format in ["avif"]:
                # Pillow's default AVIF quality might be different, 'quality' param might behave differently.
                # Using a high default if original_quality is None for AVIF.
                save_image.save(
                    str(save_path),
                    quality=save_quality if original_quality is not None else 90,
                )
            else:
                save_image.save(str(save_path))
            # self.console.print(f"[green]Successfully saved {Path(save_path).name}[/green]") # Optional: for verbose saving log
        except Exception as e:
            print_exception(
                self.console,
                e,
                prefix=(
                    f"Failed to save {Path(save_path).name} with original settings "
                    f"(format: {original_format}, quality: {original_quality}), attempting default save"
                ),
                summary_style="yellow",
            )
            try:
                # Fallback: convert to RGB and save as PNG if specific format save fails badly
                if pil_image.mode != "RGB":
                    pil_image.convert("RGB").save(str(save_path), format="PNG")
                else:
                    pil_image.save(str(save_path), format="PNG")  # Default save, often PNG
                self.console.print(f"[green]Successfully saved {Path(save_path).name} using fallback (PNG).[/green]")
            except Exception as fallback_e:
                print_exception(self.console, fallback_e, prefix=f"Fallback save also failed for {Path(save_path).name}")

    def _resize_pil_to_target_with_padding(
        self,
        pil_image: Image.Image,
        target_width: int,
        target_height: int,
        resample_method: Image.Resampling = Image.LANCZOS,
        keep_aspect_ratio_and_pad: bool = False,
        pad_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """
        Resizes a PIL Image to target dimensions.

        Args:
            pil_image: The PIL Image to resize.
            target_width: The target width.
            target_height: The target height.
            resample_method: The resampling filter to use.
            keep_aspect_ratio_and_pad: If True, maintains aspect ratio and pads to fit target dimensions.
                                       If False, resizes directly to target dimensions, potentially changing aspect ratio.
            pad_color: RGB tuple for padding color if keep_aspect_ratio_and_pad is True.

        Returns:
            The resized (and optionally padded) PIL Image.
        """
        # Crop transparent borders if requested and image has alpha channel
        if self.crop_transparent and pil_image.mode == "RGBA":
            bbox = pil_image.getbbox()
            if bbox:
                pil_image = pil_image.crop(bbox)
                self.console.print("[blue]Cropped transparent borders during padding operation[/blue]")

        original_width, original_height = pil_image.size

        if not keep_aspect_ratio_and_pad:
            return pil_image.resize((target_width, target_height), resample=resample_method)

        # Calculate aspect ratios
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height

        if original_aspect > target_aspect:
            # Original image is wider relative to target: fit to target_width
            new_width = target_width
            new_height = round(target_width / original_aspect)
        else:
            # Original image is taller relative to target (or same aspect): fit to target_height
            new_height = target_height
            new_width = round(target_height * original_aspect)

        # Resize with aspect ratio preservation
        resized_image = pil_image.resize((new_width, new_height), resample=resample_method)

        # Create a new image with padding_color and paste the resized image onto it
        # Ensure pad_color is RGB for Image.new()
        final_pad_color = pad_color[:3]  # Take only R, G, B components

        # Handle images with alpha channel by creating an RGBA background if the input has alpha
        if pil_image.mode == "RGBA" or "A" in pil_image.mode:
            # If pad_color was (R,G,B), make it (R,G,B,A) for RGBA background
            # Assuming full opacity for padding unless alpha is provided in pad_color (which it isn't currently)
            final_pad_color_with_alpha = final_pad_color + (255,)
            padded_image = Image.new("RGBA", (target_width, target_height), final_pad_color_with_alpha)
            # When pasting an RGBA image, its alpha channel is used as a mask
            paste_position = (
                (target_width - new_width) // 2,
                (target_height - new_height) // 2,
            )
            padded_image.paste(
                resized_image,
                paste_position,
                resized_image if resized_image.mode == "RGBA" else None,
            )
        else:
            # For RGB images or images without alpha, create an RGB background
            padded_image = Image.new("RGB", (target_width, target_height), final_pad_color)
            paste_position = (
                (target_width - new_width) // 2,
                (target_height - new_height) // 2,
            )
            padded_image.paste(resized_image, paste_position)

        return padded_image

    def resize_image(
        self,
        image_path: str,
        max_short_edge: int = None,
        max_long_edge: int = None,
        max_pixels: int = None,
    ) -> ProcessImageResult:
        """
        Resizes an image, ensuring edges do not exceed max_short_edge/max_long_edge,
        while maintaining the aspect ratio. Overwrites the source file.

        Args:
            image_path: Path to the image file.
            max_short_edge: The maximum length of the shorter edge of the image.
            max_long_edge: The maximum length of the longer edge of the image.
            max_pixels: The maximum number of pixels in the image.

        Returns:
            ProcessImageResult: Processing status. It is bool-compatible for older callers.
        """
        resized_pil = None
        resized = False
        converted = False
        cropped = False

        try:
            open_start = time.perf_counter()
            with Image.open(image_path) as pil_image:
                plan = _build_resize_plan(
                    pil_image,
                    image_path,
                    max_short_edge=max_short_edge,
                    max_long_edge=max_long_edge,
                    max_pixels=max_pixels,
                    crop_transparent=self.crop_transparent,
                )
                self._add_stage_time("open_and_plan", open_start)

                if plan.can_skip_pixel_decode:
                    return ProcessImageResult(ok=True, skipped=True)

                decode_start = time.perf_counter()
                working_image = pil_image

                if plan.requires_transparent_crop:
                    bbox = working_image.getbbox()
                    if bbox:
                        working_image = working_image.crop(bbox)
                        cropped = True

                img_for_processing_pil = working_image
                if working_image.mode not in ["RGB", "L"]:
                    img_for_processing_pil = working_image.convert("RGB")
                    converted = True
                elif working_image.mode == "L" and plan.original_format not in _GRAYSCALE_FRIENDLY_FORMATS:
                    img_for_processing_pil = working_image.convert("RGB")
                    converted = True

                self._add_stage_time("decode_convert_crop", decode_start)

                w, h = img_for_processing_pil.size
                new_w, new_h = calculate_dimensions(
                    w,
                    h,
                    max_long_edge=max_long_edge,
                    max_short_edge=max_short_edge,
                    max_pixels=max_pixels,
                )
                resized = (w, h) != (new_w, new_h)

                if resized:
                    resize_start = time.perf_counter()
                    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0 and min(h, w) > 1024  # Condition for GPU resize
                    if use_gpu:
                        image_cv = np.array(img_for_processing_pil.convert("RGB"))  # Preserve existing GPU path behavior
                        if img_for_processing_pil.mode == "RGB":  # if original was RGB, convert to BGR for OpenCV
                            image_cv = image_cv[:, :, ::-1].copy()

                        gpu_image = cv2.cuda.GpuMat()
                        gpu_image.upload(image_cv)
                        gpu_resized_image = cv2.cuda.resize(gpu_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        resized_image_cv = gpu_resized_image.download()

                        if resized_image_cv.ndim == 3 and resized_image_cv.shape[2] == 3:
                            resized_pil = Image.fromarray(cv2.cvtColor(resized_image_cv, cv2.COLOR_BGR2RGB))
                        elif resized_image_cv.ndim == 2 and img_for_processing_pil.mode == "L":
                            resized_pil = Image.fromarray(resized_image_cv, mode="L")
                        else:
                            self.console.print(f"[yellow]Unexpected image format after resize for {image_path}. Converting to RGB.[/yellow]")
                            if resized_image_cv.ndim == 2:
                                resized_image_cv = cv2.cvtColor(resized_image_cv, cv2.COLOR_GRAY2BGR)
                            resized_pil = Image.fromarray(cv2.cvtColor(resized_image_cv, cv2.COLOR_BGR2RGB))
                    else:
                        image_rgb = np.array(img_for_processing_pil.convert("RGB"))
                        resized_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        resized_pil = Image.fromarray(resized_rgb)
                    self._add_stage_time("resize", resize_start)
                else:
                    resized_pil = img_for_processing_pil.copy()

        except Exception as e:
            print_exception(self.console, e, prefix=f"Cannot read image: {image_path}")
            return ProcessImageResult(ok=False, error=str(e))

        save_start = time.perf_counter()
        self._save_pil_image(resized_pil, image_path)
        self._add_stage_time("save", save_start)
        return ProcessImageResult(
            ok=True,
            resized=resized,
            converted=converted,
            cropped=cropped,
        )

    def process_directory(
        self,
        input_dir: str,
        align_dir: str = None,
        max_short_edge: int = None,
        max_long_edge: int = None,
        max_pixels: int = None,
    ) -> Tuple[int, int]:
        """
        Processes all image files in a directory, with optional alignment.

        Args:
            input_dir: The primary input directory.
            align_dir: Optional. Directory with reference images for alignment.
            max_short_edge: Max short edge for resizing. Applied to input_dir images if not aligning,
                            or to align_dir images to determine target alignment size.
            max_long_edge: Max long edge for resizing. Similar application as max_short_edge.

        Returns:
            Tuple[int, int]: (Number of successfully processed images, Total number of images)
        """
        self.stage_timer.clear()
        scan_start = time.perf_counter()
        input_path = Path(input_dir)
        image_files_set = set()
        if self.recursive:
            for ext in self.image_extensions:
                for file_path in input_path.rglob(f"*{ext}"):
                    image_files_set.add(file_path.absolute())
        else:
            for ext in self.image_extensions:
                for file_path in input_path.glob(f"*{ext}"):
                    image_files_set.add(file_path.absolute())

        source_image_paths = list(image_files_set)
        self._add_stage_time("scan", scan_start)
        self.console.print(f"[blue]Supported image extensions: {', '.join(self.image_extensions)}[/blue]")

        total_images = len(source_image_paths)
        if total_images == 0:
            self.console.print(f"[yellow]No image files found in {input_dir}[/yellow]")
            return 0, 0
        self.console.print(f"[green]Found {total_images} image files in source directory {input_dir}[/green]")

        successful = 0
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "|",
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            "|",
            TextColumn("[green]Completed: {task.completed}/{task.total}"),
            console=self.console,
        ) as progress:
            task_description = "[Processing images]..."
            if align_dir:
                task_description = "[Aligning images]..."
                self.console.print(f"[blue]Aligning images from {input_dir} to references in {align_dir}[/blue]")
            else:
                self.console.print(f"[blue]Resizing images in {input_dir}[/blue]")

            task_id = progress.add_task(task_description, total=total_images)

            if align_dir:
                align_path = Path(align_dir)
                if not align_path.is_dir():
                    self.console.print(f"[red]Alignment directory {align_dir} does not exist. Aborting.[/red]")
                    return 0, total_images

                for src_img_path_obj in source_image_paths:
                    ref_img_path = align_path / src_img_path_obj.name
                    try:
                        if ref_img_path.is_file():
                            img1_pil = Image.open(src_img_path_obj)

                            orig_w_ref, orig_h_ref = img1_pil.size
                            target_w_for_ref, target_h_for_ref = calculate_dimensions(
                                orig_w_ref,
                                orig_h_ref,
                                max_long_edge=max_long_edge,
                                max_short_edge=max_short_edge,
                                max_pixels=max_pixels,
                            )

                            img2_pil = Image.open(ref_img_path)

                            if target_w_for_ref == orig_w_ref and target_h_for_ref == orig_h_ref:
                                self.console.print(
                                    f"[cyan]Source image {src_img_path_obj.name} (from input_dir) "
                                    f"is already at its calculated target dimensions. "
                                    f"These dimensions will be used as the target for resizing it within align_images.[/cyan]"
                                )

                            ref_img, src_img = self.align_images(img1_pil, img2_pil, target_w_for_ref, target_h_for_ref)

                            self._save_pil_image(ref_img, str(ref_img_path))
                            self._save_pil_image(src_img, str(src_img_path_obj))
                            successful += 1
                        else:
                            self.console.print(
                                f"[yellow]Reference {ref_img_path.name} not found. Skipping alignment for {src_img_path_obj.name}.[/yellow]"
                            )
                    except Exception as e:
                        print_exception(self.console, e, prefix=f"Error during alignment for {src_img_path_obj.name}")
                    finally:
                        progress.update(task_id, advance=1)
            else:  # Resize-only mode
                use_gpu_for_batch_resize = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if use_gpu_for_batch_resize:
                    self.console.print("[green]GPU detected, will use GPU for resizing where applicable.[/green]")
                else:
                    self.console.print("[yellow]No GPU detected, will use CPU for resizing.[/yellow]")

                resize_summary = ResizeBatchSummary()
                batch_size = 256  # Process images in batches to manage memory
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for i in range(0, len(source_image_paths), batch_size):
                        batch_paths = source_image_paths[i : i + batch_size]
                        self.console.print(
                            f"Processing resize batch {i // batch_size + 1}/{(len(source_image_paths) + batch_size - 1) // batch_size}..."
                        )

                        submitted_tasks = [
                            executor.submit(
                                self.resize_image,
                                str(img_path_obj),
                                max_short_edge,
                                max_long_edge,
                                max_pixels,
                            )
                            for img_path_obj in batch_paths
                        ]

                        for future in concurrent.futures.as_completed(submitted_tasks):
                            result = future.result()
                            resize_summary.add(result)
                            if result:
                                successful += 1
                            progress.update(task_id, advance=1)
                self._print_resize_summary(resize_summary)
                self._print_profile_summary(total_images)
        return successful, total_images

    def align_images(
        self,
        reference_image_pil: Image.Image,
        image_to_warp_pil: Image.Image,
        target_width: int,
        target_height: int,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Aligns image_to_warp_pil to reference_image_pil using feature matching and perspective transformation.
        reference_image_pil is resized to target_width and target_height.
        image_to_warp_pil is resized to the original size of reference_image_pil before alignment for feature detection context.

        Args:
            reference_image_pil: The reference PIL Image.
            image_to_warp_pil: The PIL Image to be warped.
            target_width: The target width for reference_image_pil.
            target_height: The target height for reference_image_pil.

        Returns:
            Tuple[Image.Image, Image.Image]: (aligned_warped_image_pil, resized_reference_image_pil)
        """
        use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

        # Resize images
        # image_to_warp is resized to original size of reference_image for initial feature matching context
        reference_image_pil_resized_to_target = reference_image_pil.resize((target_width, target_height), resample=Image.LANCZOS)

        # Prepare the version of image_to_warp_pil that will be returned if no alignment is performed.
        # This version is resized to target dimensions with aspect ratio preservation and padding.
        image_to_warp_pil_padded_to_target_dims = self._resize_pil_to_target_with_padding(
            image_to_warp_pil,
            target_width,
            target_height,
            resample_method=Image.LANCZOS,  # Or Image.Resampling.LANCZOS for Pillow >= 9.1.0
            keep_aspect_ratio_and_pad=True,
            pad_color=self.bg_color,  # self.bg_color is now guaranteed to be (R, G, B)
        )

        # If no transformation is requested, skip feature detection/matching entirely
        if self.transform_type == "none":
            self.console.print("[cyan]transform_type is 'none'; skipping alignment and returning resized images.[/cyan]")
            return (
                image_to_warp_pil_padded_to_target_dims,
                reference_image_pil_resized_to_target,
            )

        image_to_warp_pil_resized_to_ref_orig_size = image_to_warp_pil.resize((target_width, target_height), Image.LANCZOS)

        color_image_to_warp_cv = np.array(image_to_warp_pil_resized_to_ref_orig_size.convert("RGB"))
        reference_image_cv = np.array(reference_image_pil_resized_to_target.convert("RGB"))

        # Convert PIL images to OpenCV format (numpy arrays)
        # For image_to_warp, use the version resized to reference_image's original size for feature detection
        warped_cv_gray = cv2.cvtColor(
            color_image_to_warp_cv,
            cv2.COLOR_RGB2GRAY,
        )
        reference_cv_gray = cv2.cvtColor(
            reference_image_cv,
            cv2.COLOR_RGB2GRAY,
        )

        vismatch_src_pts = None
        vismatch_dst_pts = None
        vismatch_backend_name = "orb"
        if self.transform_type != "none":
            try:
                vismatch_src_pts, vismatch_dst_pts, vismatch_backend_name = _match_points_with_backend(
                    color_image_to_warp_cv,
                    reference_image_cv,
                    self.matcher_backend,
                )
            except Exception as e:
                print_exception(
                    self.console,
                    e,
                    prefix="Matcher backend failed before ORB fallback",
                    summary_style="yellow",
                )
                vismatch_src_pts = None
                vismatch_dst_pts = None
                vismatch_backend_name = "orb"

        if vismatch_src_pts is not None and vismatch_dst_pts is not None:
            M, transform_name = _estimate_transform_from_points(
                vismatch_src_pts,
                vismatch_dst_pts,
                self.transform_type,
            )
            if M is not None:
                self.console.print(
                    f"[green]Using {vismatch_backend_name} backend with {transform_name} transform.[/green]"
                )
                warp_function = cv2.warpPerspective if transform_name == "homography" else cv2.warpAffine
                h_target, w_target = reference_cv_gray.shape
                aligned_image_cv = warp_function(
                    color_image_to_warp_cv,
                    M,
                    (w_target, h_target),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
                aligned_warped_image_pil = Image.fromarray(aligned_image_cv)
                return aligned_warped_image_pil, reference_image_pil_resized_to_target

        if use_gpu:
            try:
                self.console.print("[blue]Using GPU for image alignment.[/blue]")
                gpu_warped_cv_gray = cv2.cuda.GpuMat()
                gpu_warped_cv_gray.upload(warped_cv_gray)
                gpu_reference_cv_gray = cv2.cuda.GpuMat()
                gpu_reference_cv_gray.upload(reference_cv_gray)

                # ORB Feature Detector on GPU
                orb_gpu = cv2.cuda.ORB_create(
                    nfeatures=800, scoreType=cv2.ORB_FAST_SCORE
                )  # Increased nfeatures for potentially better matching
                keypoints_warped_gpu, descriptors_warped_gpu = orb_gpu.detectAndComputeAsync(gpu_warped_cv_gray, None)
                keypoints_ref_gpu, descriptors_ref_gpu = orb_gpu.detectAndComputeAsync(gpu_reference_cv_gray, None)

                # Download keypoints and descriptors
                keypoints_warped = orb_gpu.convert(keypoints_warped_gpu)
                descriptors_warped = descriptors_warped_gpu.download()
                keypoints_ref = orb_gpu.convert(keypoints_ref_gpu)
                descriptors_ref = descriptors_ref_gpu.download()

                if (
                    descriptors_warped is None
                    or descriptors_ref is None
                    or len(descriptors_warped) == 0
                    or len(descriptors_ref) == 0
                ):
                    self.console.print(
                        "[yellow]Not enough descriptors found for GPU alignment. Falling back to CPU or returning original.[/yellow]"
                    )
                    # Fallback or return original if not enough features
                    return (
                        image_to_warp_pil_padded_to_target_dims,
                        reference_image_pil_resized_to_target,
                    )

                # Feature Matching on GPU
                matcher_gpu = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
                matches_gpu = matcher_gpu.knnMatch(descriptors_warped_gpu, descriptors_ref_gpu, k=2)

                # Apply ratio test as per Lowe's paper for good matches (robust to tuple/list structures)
                good_matches = []
                for item in matches_gpu:
                    # KNN case: item is a list/tuple of DMatch
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        m, n = item[0], item[1]
                        if m.distance < 0.85 * n.distance:
                            good_matches.append(m)
                    # Non-KNN (rare here): single DMatch
                    elif hasattr(item, "distance"):
                        good_matches.append(item)

                # We need at least 3 points for affine, 4 for homography.
                if len(good_matches) < 5:
                    self.console.print(
                        f"[yellow]Not enough good matches found ({len(good_matches)}). Returning unaligned.[/yellow]"
                    )
                    return (
                        image_to_warp_pil_padded_to_target_dims,
                        reference_image_pil_resized_to_target,
                    )

                # Extract location of good matches
                src_pts = np.float32([keypoints_warped[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, transform_name = _estimate_transform_from_points(src_pts, dst_pts, self.transform_type)
                if M is None:
                    self.console.print("[yellow]Could not compute any transformation matrix. Returning unaligned.[/yellow]")
                    return (
                        image_to_warp_pil_padded_to_target_dims,
                        reference_image_pil_resized_to_target,
                    )

                self.console.print(f"[green]Successfully computed {transform_name} matrix.[/green]")

                # Prepare color image for warping on GPU
                gpu_color_image_to_warp = cv2.cuda.GpuMat()
                gpu_color_image_to_warp.upload(color_image_to_warp_cv)

                h_target, w_target = reference_cv_gray.shape

                # Perform warping with the selected function
                gpu_warped_with_borders = (cv2.cuda.warpPerspective if transform_name == "homography" else cv2.cuda.warpAffine)(
                    gpu_color_image_to_warp,
                    M,
                    (w_target, h_target),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),  # White for RGB
                )
                aligned_image_cv = gpu_warped_with_borders.download()
                aligned_warped_image_pil = Image.fromarray(aligned_image_cv)

                return aligned_warped_image_pil, reference_image_pil_resized_to_target

            except cv2.error as e:
                print_exception(self.console, e, prefix="OpenCV CUDA error during alignment. Falling back to CPU")
                use_gpu = False  # Fallback to CPU
            except Exception as e:
                print_exception(self.console, e, prefix="Generic error during GPU alignment. Falling back to CPU")
                use_gpu = False  # Fallback to CPU

        # CPU Alignment (or fallback)
        if not use_gpu:
            self.console.print("[blue]Using CPU for image alignment.[/blue]")
            orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
            keypoints_warped, descriptors_warped = orb.detectAndCompute(warped_cv_gray, None)
            keypoints_ref, descriptors_ref = orb.detectAndCompute(reference_cv_gray, None)

            if descriptors_warped is None or descriptors_ref is None or len(descriptors_warped) < 2 or len(descriptors_ref) < 2:
                self.console.print("[yellow]Not enough descriptors for CPU alignment. Returning unaligned.[/yellow]")
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # crossCheck=False for knnMatch
            matches = bf.knnMatch(descriptors_warped, descriptors_ref, k=2)

            good_matches = []
            # Apply ratio test as per Lowe's paper (robust to tuple/list/single DMatch)
            if matches and len(matches) > 0:
                for item in matches:
                    # KNN case: item is (m, n) or [m, n]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        m, n = item[0], item[1]
                        if m.distance < 0.85 * n.distance:
                            good_matches.append(m)
                    # Non-KNN: item is a single DMatch
                    elif hasattr(item, "distance"):
                        good_matches.append(item)
            else:
                self.console.print("[yellow]Not enough matches or match pairs for CPU ratio test. Returning unaligned.[/yellow]")
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            # We need at least 3 points for affine, 4 for homography.
            if len(good_matches) < 3:
                self.console.print(f"[yellow]Not enough good matches found ({len(good_matches)}). Returning unaligned.[/yellow]")
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            src_pts = np.float32([keypoints_warped[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, transform_name = _estimate_transform_from_points(src_pts, dst_pts, self.transform_type)
            if M is None:
                self.console.print("[yellow]Could not compute any transformation matrix (CPU path). Returning unaligned.[/yellow]")
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            self.console.print(f"[green]Successfully computed {transform_name} matrix on CPU.[/green]")

            h_target, w_target = reference_cv_gray.shape

            # Perform warp with the selected function
            aligned_image_cv = (cv2.warpPerspective if transform_name == "homography" else cv2.warpAffine)(
                color_image_to_warp_cv,
                M,
                (w_target, h_target),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),  # White for RGB
            )
            aligned_warped_image_pil = Image.fromarray(aligned_image_cv)

            return aligned_warped_image_pil, reference_image_pil_resized_to_target


def main():
    parser = argparse.ArgumentParser(
        description="Batch resize images, ensuring the shortest edge does not exceed a specified value, with optional alignment."
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory path (source images)")
    parser.add_argument(
        "-a",
        "--align-input",
        type=str,
        default=None,
        help="Optional: Second input directory with reference images for alignment.",
    )
    parser.add_argument(
        "-ms",
        "--max-short-edge",
        type=int,
        default=None,
        help="Maximum value for the shortest edge (default: None)",
    )
    parser.add_argument(
        "-ml",
        "--max-long-edge",
        type=int,
        default=None,
        help="Maximum value for the longest edge (default: None)",
    )
    parser.add_argument(
        "-mp",
        "--max-pixels",
        type=int,
        default=None,
        help="Maximum value for the number of pixels (default: None)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Maximum number of worker threads (default: 16)",
    )
    parser.add_argument(
        "-t",
        "--transform-type",
        type=str,
        default="none",
        choices=["auto", "affine", "homography", "none"],
        help="Transformation type for alignment: auto, affine, homography, or none (default: none)",
    )
    parser.add_argument(
        "--matcher-backend",
        type=str,
        default="auto",
        choices=["auto", "xfeat", "affine_steerers", "orb"],
        help="Matcher backend: auto prefers affine_steerers on CUDA, xfeat otherwise, and falls back to ORB on failure.",
    )
    parser.add_argument(
        "-bc",
        "--bg-color",
        type=int,
        nargs=3,
        default=[255, 255, 255],
        metavar=("R", "G", "B"),
        help="RGB background color for padding (e.g., 0 0 0 for black, 255 255 255 for white). Default: 255 255 255",
    )
    parser.add_argument(
        "-ct",
        "--crop-transparent",
        action="store_true",
        help="Crop transparent borders from RGBA images before processing",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print resize-only stage timing summary.",
    )

    args = parser.parse_args()

    # Use the global console for script-level messages before processor instantiation
    global_console.print("[bold blue]Starting image processing...[/bold blue]")
    global_console.print(f"Input directory: {args.input}")
    global_console.print(f"Max short edge: {args.max_short_edge}")
    global_console.print(f"Max long edge: {args.max_long_edge}")
    global_console.print(f"Max pixels: {args.max_pixels}")
    global_console.print(f"Recursive processing: {'Yes' if args.recursive else 'No'}")
    global_console.print(f"Number of workers: {args.workers if args.workers is not None else 16}")
    if args.align_input:
        global_console.print(f"Alignment reference directory: {args.align_input}")
    global_console.print(f"Matcher backend: {args.matcher_backend}")
    global_console.print(f"Crop transparent borders: {'Yes' if args.crop_transparent else 'No'}")
    global_console.print(f"Resize profiling: {'Yes' if args.profile else 'No'}")
    global_console.print("[red bold]Warning: Original image files in the primary input directory will be overwritten![/red bold]")

    # Instantiate the processor, it will create its own console or use one if passed
    processor = ImageProcessor(
        recursive=args.recursive,
        max_workers=args.workers,
        console=global_console,  # Pass the global console to the processor
        transform_type=args.transform_type,
        matcher_backend=args.matcher_backend,
        bg_color=tuple(args.bg_color),
        crop_transparent=args.crop_transparent,
        profile=args.profile,
    )

    start_time = time.time()

    successful, total = processor.process_directory(
        args.input,
        align_dir=args.align_input,
        max_short_edge=args.max_short_edge,
        max_long_edge=args.max_long_edge,
        max_pixels=args.max_pixels,
    )

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    processor.console.print("[bold green]Processing complete![/bold green]")
    processor.console.print(f"Successfully processed: {successful}/{total} images")
    processor.console.print(f"Total time taken: {minutes}m {seconds}s")

    if total > 0:
        processor.console.print(f"Average processing time per image: {elapsed_time / total:.2f}s")

    if successful < total:
        processor.console.print(f"[yellow]{total - successful} images failed to process[/yellow]")


if __name__ == "__main__":
    main()
