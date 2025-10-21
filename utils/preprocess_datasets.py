# /// script
# dependencies = [
#   "setuptools",
#   "pillow>=11.3",
#   "rich>=13.5.0",
#   "imageio>=2.31.1",
#   "imageio-ffmpeg>=0.4.8",
#   "opencv-contrib-python-rolling @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.11.0.20250124/opencv_contrib_python_rolling-4.12.0.86-cp37-abi3-win_amd64.whl; sys_platform == 'win32'",
#   "opencv-contrib-python-rolling @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.11.0.20250210/opencv_contrib_python_rolling-4.12.0.20250210-cp37-abi3-linux_x86_64.whl; sys_platform == 'linux'",
#   "torch>=2.8.0",
#   "toml",
# ]
# ///
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import concurrent.futures
from PIL import Image
from utils.stream_util import calculate_dimensions
from config.config import get_supported_extensions
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)
import time

# Global console for general script-level logging if needed outside the class
# However, the class will have its own console instance for its operations.
global_console = Console()


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
        bg_color: Tuple[int, int, int] = (255, 255, 255), # Default to white
        crop_transparent: bool = False,
    ):
        """
        Initializes the ImageProcessor.

        Args:
            recursive: Whether to process images in subdirectories.
            max_workers: The maximum number of worker threads for parallel processing.
            console: An optional Rich Console instance for output.
            transform_type: The type of transformation for alignment.
            bg_color: RGB background color for padding.
            crop_transparent: Whether to crop transparent borders from RGBA images.
        """
        self.recursive = recursive
        self.max_workers = (
            max_workers if max_workers is not None else 16
        )  # Default to 16 if None
        self.console = (
            console if console else Console()
        )  # Use provided console or create a new one
        self.image_extensions = get_supported_extensions("image")
        self.transform_type = transform_type
        self.bg_color = bg_color
        self.crop_transparent = crop_transparent

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
                    self.console.print(
                        f"[yellow]Converting RGBA to RGB for saving {Path(save_path).name} as JPEG.[/yellow]"
                    )
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
            self.console.print(
                f"[yellow]Failed to save {Path(save_path).name} with original settings (format: {original_format}, quality: {original_quality}), attempting default save: {str(e)}[/yellow]"
            )
            try:
                # Fallback: convert to RGB and save as PNG if specific format save fails badly
                if pil_image.mode != "RGB":
                    pil_image.convert("RGB").save(str(save_path), format="PNG")
                else:
                    pil_image.save(
                        str(save_path), format="PNG"
                    )  # Default save, often PNG
                self.console.print(
                    f"[green]Successfully saved {Path(save_path).name} using fallback (PNG).[/green]"
                )
            except Exception as fallback_e:
                self.console.print(
                    f"[red]Fallback save also failed for {Path(save_path).name}: {fallback_e}[/red]"
                )

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
                self.console.print(
                    "[blue]Cropped transparent borders during padding operation[/blue]"
                )

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
        final_pad_color = pad_color[:3] # Take only R, G, B components
        
        # Handle images with alpha channel by creating an RGBA background if the input has alpha
        if pil_image.mode == 'RGBA' or 'A' in pil_image.mode:
            # If pad_color was (R,G,B), make it (R,G,B,A) for RGBA background
            # Assuming full opacity for padding unless alpha is provided in pad_color (which it isn't currently)
            final_pad_color_with_alpha = final_pad_color + (255,)
            padded_image = Image.new("RGBA", (target_width, target_height), final_pad_color_with_alpha)
            # When pasting an RGBA image, its alpha channel is used as a mask
            paste_position = ((target_width - new_width) // 2, (target_height - new_height) // 2)
            padded_image.paste(resized_image, paste_position, resized_image if resized_image.mode == 'RGBA' else None)
        else:
            # For RGB images or images without alpha, create an RGB background
            padded_image = Image.new("RGB", (target_width, target_height), final_pad_color)
            paste_position = ((target_width - new_width) // 2, (target_height - new_height) // 2)
            padded_image.paste(resized_image, paste_position)
            
        return padded_image

    def resize_image(
        self, image_path: str, max_short_edge: int = None, max_long_edge: int = None, max_pixels: int = None
    ) -> bool:
        """
        Resizes an image, ensuring edges do not exceed max_short_edge/max_long_edge,
        while maintaining the aspect ratio. Overwrites the source file.

        Args:
            image_path: Path to the image file.
            max_short_edge: The maximum length of the shorter edge of the image.
            max_long_edge: The maximum length of the longer edge of the image.
            max_pixels: The maximum number of pixels in the image.

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        try:
            pil_image = Image.open(image_path)
            original_format = Path(image_path).suffix.lower().lstrip(".")
            original_quality = None
            if hasattr(pil_image, "info") and "quality" in pil_image.info:
                original_quality = pil_image.info.get("quality")

            # Crop transparent borders if requested and image has alpha channel
            if self.crop_transparent and pil_image.mode == "RGBA":
                bbox = pil_image.getbbox()
                if bbox:
                    pil_image = pil_image.crop(bbox)
                    self.console.print(
                        f"[blue]Cropped transparent borders from {Path(image_path).name}[/blue]"
                    )

            # Preserve original mode if it's L (grayscale) for certain formats, otherwise convert to RGB for processing
            # This is a delicate balance: color conversion for consistency vs. preserving original color space.
            # For most resizing/alignment, RGB is safer. Grayscale might be kept if no color manipulation is done.
            # Let's assume internal processing prefers RGB, but _save_pil_image can handle 'L' for JPEG.
            img_for_processing_pil = pil_image
            if pil_image.mode not in ["RGB", "L"]:
                img_for_processing_pil = pil_image.convert("RGB")
            elif pil_image.mode == "L" and original_format not in [
                "jpeg",
                "jpg",
                "png",
                "avif",
            ]:
                # If grayscale but not a typically grayscale-friendly save format, convert for broader compatibility
                img_for_processing_pil = pil_image.convert("RGB")

            image_cv = np.array(
                img_for_processing_pil.convert("RGB")
            )  # Ensure RGB for OpenCV processing
            if (
                img_for_processing_pil.mode == "RGB"
            ):  # if original was RGB, convert to BGR for OpenCV
                image_cv = image_cv[:, :, ::-1].copy()
            # if original was 'L', image_cv is already (h,w) and fine for grayscale cv processing if needed
            # but resize usually expects 3 channels if color. Forcing RGB for np.array ensures 3 channels.

        except Exception as e:
            self.console.print(f"[red]Cannot read image: {image_path} - {str(e)}[/red]")
            return False

        h, w = image_cv.shape[:2]
        # If both are None, calculate_dimensions should return original w,h
        new_w, new_h = calculate_dimensions(
            w, h, max_long_edge=max_long_edge, max_short_edge=max_short_edge, max_pixels=max_pixels
        )

        if (w, h) == (new_w, new_h) and pil_image.mode == img_for_processing_pil.mode:
            self.console.print(
                f"[yellow]Skipping {image_path}: Already meets size and mode requirements[/yellow]"
            )
            # Still, we might want to re-save it to normalize it or apply quality settings
            # For now, if no dimension change, assume no action needed.
            return True

        # Perform resize
        use_gpu = (
            cv2.cuda.getCudaEnabledDeviceCount() > 0 and min(h, w) > 1024
        )  # Condition for GPU resize
        if use_gpu:
            gpu_image = cv2.cuda.GpuMat()
            gpu_image.upload(image_cv)
            gpu_resized_image = cv2.cuda.resize(
                gpu_image, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            resized_image_cv = gpu_resized_image.download()
        else:
            resized_image_cv = cv2.resize(
                image_cv, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

        # Convert back to PIL Image
        # If original was RGB or converted to RGB for processing, convert BGR (OpenCV) back to RGB for PIL
        if resized_image_cv.ndim == 3 and resized_image_cv.shape[2] == 3:
            resized_pil = Image.fromarray(
                cv2.cvtColor(resized_image_cv, cv2.COLOR_BGR2RGB)
            )
        elif (
            resized_image_cv.ndim == 2 and img_for_processing_pil.mode == "L"
        ):  # Original was Grayscale and processed as such
            resized_pil = Image.fromarray(resized_image_cv, mode="L")
        else:  # Fallback, should ideally not happen if input was RGB or L
            self.console.print(
                f"[yellow]Unexpected image format after resize for {image_path}. Converting to RGB.[/yellow]"
            )
            # Ensure it's 3 channels BGR before converting to RGB for PIL
            if resized_image_cv.ndim == 2:
                resized_image_cv = cv2.cvtColor(resized_image_cv, cv2.COLOR_GRAY2BGR)
            resized_pil = Image.fromarray(
                cv2.cvtColor(resized_image_cv, cv2.COLOR_BGR2RGB)
            )

        self._save_pil_image(resized_pil, image_path)
        return True

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
        self.console.print(
            f"[blue]Supported image extensions: {', '.join(self.image_extensions)}[/blue]"
        )

        total_images = len(source_image_paths)
        if total_images == 0:
            self.console.print(f"[yellow]No image files found in {input_dir}[/yellow]")
            return 0, 0
        self.console.print(
            f"[green]Found {total_images} image files in source directory {input_dir}[/green]"
        )

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
                self.console.print(
                    f"[blue]Aligning images from {input_dir} to references in {align_dir}[/blue]"
                )
            else:
                self.console.print(f"[blue]Resizing images in {input_dir}[/blue]")

            task_id = progress.add_task(task_description, total=total_images)

            if align_dir:
                align_path = Path(align_dir)
                if not align_path.is_dir():
                    self.console.print(
                        f"[red]Alignment directory {align_dir} does not exist. Aborting.[/red]"
                    )
                    return 0, total_images

                for src_img_path_obj in source_image_paths:
                    ref_img_path = align_path / src_img_path_obj.name
                    try:
                        if ref_img_path.is_file():
                            img1_pil = Image.open(src_img_path_obj)

                            orig_w_ref, orig_h_ref = img1_pil.size
                            target_w_for_ref, target_h_for_ref = calculate_dimensions(
                                orig_w_ref, orig_h_ref, max_long_edge=max_long_edge, max_short_edge=max_short_edge, max_pixels=max_pixels
                            )

                            img2_pil = Image.open(ref_img_path)

                            if (
                                target_w_for_ref == orig_w_ref
                                and target_h_for_ref == orig_h_ref
                            ):
                                self.console.print(
                                    f"[cyan]Source image {src_img_path_obj.name} (from input_dir) "
                                    f"is already at its calculated target dimensions. "
                                    f"These dimensions will be used as the target for resizing it within align_images.[/cyan]"
                                )

                            ref_img, src_img = self.align_images(
                                img1_pil, img2_pil, target_w_for_ref, target_h_for_ref
                            )

                            self._save_pil_image(ref_img, str(ref_img_path))
                            self._save_pil_image(src_img, str(src_img_path_obj))
                            successful += 1
                        else:
                            self.console.print(
                                f"[yellow]Reference {ref_img_path.name} not found. Skipping alignment for {src_img_path_obj.name}.[/yellow]"
                            )
                    except Exception as e:
                        self.console.print(
                            f"[red]Error during alignment for {src_img_path_obj.name}: {e}[/red]"
                        )
                    finally:
                        progress.update(task_id, advance=1)
            else:  # Resize-only mode
                use_gpu_for_batch_resize = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if use_gpu_for_batch_resize:
                    self.console.print(
                        f"[green]GPU detected, will use GPU for resizing where applicable.[/green]"
                    )
                else:
                    self.console.print(
                        f"[yellow]No GPU detected, will use CPU for resizing.[/yellow]"
                    )

                batch_size = 256  # Process images in batches to manage memory
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    for i in range(0, len(source_image_paths), batch_size):
                        batch_paths = source_image_paths[i : i + batch_size]
                        self.console.print(
                            f"Processing resize batch {i//batch_size + 1}/{(len(source_image_paths) + batch_size - 1)//batch_size}..."
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
                            if future.result():
                                successful += 1
                            progress.update(task_id, advance=1)
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
        reference_image_pil_resized_to_target = reference_image_pil.resize(
            (target_width, target_height), resample=Image.LANCZOS
        )

        # Prepare the version of image_to_warp_pil that will be returned if no alignment is performed.
        # This version is resized to target dimensions with aspect ratio preservation and padding.
        image_to_warp_pil_padded_to_target_dims = self._resize_pil_to_target_with_padding(
            image_to_warp_pil,
            target_width,
            target_height,
            resample_method=Image.LANCZOS, # Or Image.Resampling.LANCZOS for Pillow >= 9.1.0
            keep_aspect_ratio_and_pad=True,
            pad_color=self.bg_color # self.bg_color is now guaranteed to be (R, G, B)
        )

        # If no transformation is requested, skip feature detection/matching entirely
        if self.transform_type == "none":
            self.console.print(
                "[cyan]transform_type is 'none'; skipping alignment and returning resized images.[/cyan]"
            )
            return (
                image_to_warp_pil_padded_to_target_dims,
                reference_image_pil_resized_to_target,
            )

        image_to_warp_pil_resized_to_ref_orig_size = image_to_warp_pil.resize(
            (target_width, target_height), Image.LANCZOS
        )

        # Convert PIL images to OpenCV format (numpy arrays)
        # For image_to_warp, use the version resized to reference_image's original size for feature detection
        warped_cv_gray = cv2.cvtColor(
            np.array(image_to_warp_pil_resized_to_ref_orig_size.convert("RGB")),
            cv2.COLOR_RGB2GRAY,
        )
        reference_cv_gray = cv2.cvtColor(
            np.array(reference_image_pil_resized_to_target.convert("RGB")),
            cv2.COLOR_RGB2GRAY,
        )

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
                keypoints_warped_gpu, descriptors_warped_gpu = (
                    orb_gpu.detectAndComputeAsync(gpu_warped_cv_gray, None)
                )
                keypoints_ref_gpu, descriptors_ref_gpu = orb_gpu.detectAndComputeAsync(
                    gpu_reference_cv_gray, None
                )

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
                matcher_gpu = cv2.cuda.DescriptorMatcher_createBFMatcher(
                    cv2.NORM_HAMMING
                )
                matches_gpu = matcher_gpu.knnMatch(
                    descriptors_warped_gpu, descriptors_ref_gpu, k=2
                )

                # Apply ratio test as per Lowe's paper for good matches (robust to tuple/list structures)
                good_matches = []
                for item in matches_gpu:
                    # KNN case: item is a list/tuple of DMatch
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        m, n = item[0], item[1]
                        if m.distance < 0.85 * n.distance:
                            good_matches.append(m)
                    # Non-KNN (rare here): single DMatch
                    elif hasattr(item, 'distance'):
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
                src_pts = np.float32(
                    [keypoints_warped[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [keypoints_ref[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                # --- NEW LOGIC based on similarity ---
                similarity_threshold = 10  # Min good matches to attempt homography
                M = None
                warp_function = None

                # Decide on transformation based on number of matches
                if (
                    len(good_matches) >= similarity_threshold
                    and self.transform_type != "affine"
                ):
                    self.console.print(
                        f"[green]High similarity ({len(good_matches)} matches), attempting homography.[/green]"
                    )
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
                    if M is not None:
                        warp_function = cv2.cuda.warpPerspective
                        transform_type = "Homography"
                    else:
                        self.console.print(
                            "[yellow]Homography failed, will fall back to affine.[/yellow]"
                        )

                # If homography was not attempted or failed, try affine
                if M is None or self.transform_type == "affine":
                    self.console.print(
                        f"[yellow]Low similarity or homography failed ({len(good_matches)} matches), attempting affine transformation.[/yellow]"
                    )
                    M, _ = cv2.estimateAffine2D(
                        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0
                    )
                    if M is not None:
                        warp_function = cv2.cuda.warpAffine
                        transform_type = "Affine"

                if M is None:
                    self.console.print(
                        "[yellow]Could not compute any transformation matrix. Returning unaligned.[/yellow]"
                    )
                    return (
                        image_to_warp_pil_padded_to_target_dims,
                        reference_image_pil_resized_to_target,
                    )

                self.console.print(
                    f"[green]Successfully computed {transform_type} matrix.[/green]"
                )

                # Prepare color image for warping on GPU
                color_image_to_warp_cv_for_gpu_upload = np.array(
                    image_to_warp_pil_resized_to_ref_orig_size.convert("RGB")
                )
                gpu_color_image_to_warp = cv2.cuda.GpuMat()
                gpu_color_image_to_warp.upload(color_image_to_warp_cv_for_gpu_upload)

                h_target, w_target = reference_cv_gray.shape

                # Perform warping with the selected function
                gpu_warped_with_borders = warp_function(
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
                self.console.print(
                    f"[red]OpenCV CUDA error during alignment: {e}. Falling back to CPU.[/red]"
                )
                use_gpu = False  # Fallback to CPU
            except Exception as e:
                self.console.print(
                    f"[red]Generic error during GPU alignment: {e}. Falling back to CPU.[/red]"
                )
                use_gpu = False  # Fallback to CPU

        # CPU Alignment (or fallback)
        if not use_gpu:
            self.console.print("[blue]Using CPU for image alignment.[/blue]")
            orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
            keypoints_warped, descriptors_warped = orb.detectAndCompute(
                warped_cv_gray, None
            )
            keypoints_ref, descriptors_ref = orb.detectAndCompute(
                reference_cv_gray, None
            )

            if (
                descriptors_warped is None
                or descriptors_ref is None
                or len(descriptors_warped) < 2
                or len(descriptors_ref) < 2
            ):
                self.console.print(
                    "[yellow]Not enough descriptors for CPU alignment. Returning unaligned.[/yellow]"
                )
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            bf = cv2.BFMatcher(
                cv2.NORM_HAMMING, crossCheck=False
            )  # crossCheck=False for knnMatch
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
                    elif hasattr(item, 'distance'):
                        good_matches.append(item)
            else:
                self.console.print(
                    "[yellow]Not enough matches or match pairs for CPU ratio test. Returning unaligned.[/yellow]"
                )
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            # We need at least 3 points for affine, 4 for homography.
            if len(good_matches) < 3:
                self.console.print(
                    f"[yellow]Not enough good matches found ({len(good_matches)}). Returning unaligned.[/yellow]"
                )
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            src_pts = np.float32(
                [keypoints_warped[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints_ref[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # --- NEW LOGIC based on similarity ---
            similarity_threshold = 10  # Min good matches to attempt homography
            M = None
            warp_function = None
            transform_type = ""

            # Decide on transformation based on number of matches
            if len(good_matches) >= similarity_threshold:
                self.console.print(
                    f"[green]High similarity ({len(good_matches)} matches), attempting homography.[/green]"
                )
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
                if M is not None:
                    warp_function = cv2.warpPerspective
                    transform_type = "Homography"
                else:
                    self.console.print(
                        "[yellow]Homography failed, will fall back to affine.[/yellow]"
                    )

            # If homography was not attempted or failed, try affine
            if M is None:
                self.console.print(
                    f"[yellow]Low similarity or homography failed ({len(good_matches)} matches), attempting affine transformation.[/yellow]"
                )
                M, _ = cv2.estimateAffine2D(
                    src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0
                )
                if M is not None:
                    warp_function = cv2.warpAffine
                    transform_type = "Affine"

            if M is None:
                self.console.print(
                    "[yellow]Could not compute any transformation matrix (CPU path). Returning unaligned.[/yellow]"
                )
                return (
                    image_to_warp_pil_resized_to_ref_orig_size,
                    reference_image_pil_resized_to_target,
                )

            self.console.print(
                f"[green]Successfully computed {transform_type} matrix on CPU.[/green]"
            )

            # Prepare color image for warping on CPU
            color_image_to_warp_cv = np.array(
                image_to_warp_pil_resized_to_ref_orig_size.convert("RGB")
            )

            h_target, w_target = reference_cv_gray.shape

            # Perform warp with the selected function
            aligned_image_cv = warp_function(
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
    parser.add_argument(
        "-i", "--input", required=True, help="Input directory path (source images)"
    )
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
        help="Transformation type for alignment: auto (default), affine, homography, or none",
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

    args = parser.parse_args()

    # Use the global console for script-level messages before processor instantiation
    global_console.print(f"[bold blue]Starting image processing...[/bold blue]")
    global_console.print(f"Input directory: {args.input}")
    global_console.print(f"Max short edge: {args.max_short_edge}")
    global_console.print(f"Max long edge: {args.max_long_edge}")
    global_console.print(f"Max pixels: {args.max_pixels}")
    global_console.print(f"Recursive processing: {'Yes' if args.recursive else 'No'}")
    global_console.print(
        f"Number of workers: {args.workers if args.workers is not None else 16}"
    )
    if args.align_input:
        global_console.print(f"Alignment reference directory: {args.align_input}")
    global_console.print(f"Crop transparent borders: {'Yes' if args.crop_transparent else 'No'}")
    global_console.print(
        f"[red bold]Warning: Original image files in the primary input directory will be overwritten![/red bold]"
    )

    # Instantiate the processor, it will create its own console or use one if passed
    processor = ImageProcessor(
        recursive=args.recursive,
        max_workers=args.workers,
        console=global_console,  # Pass the global console to the processor
        transform_type=args.transform_type,
        bg_color=tuple(args.bg_color),
        crop_transparent=args.crop_transparent,
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

    processor.console.print(f"[bold green]Processing complete![/bold green]")
    processor.console.print(f"Successfully processed: {successful}/{total} images")
    processor.console.print(f"Total time taken: {minutes}m {seconds}s")

    if total > 0:
        processor.console.print(
            f"Average processing time per image: {elapsed_time/total:.2f}s"
        )

    if successful < total:
        processor.console.print(
            f"[yellow]{total - successful} images failed to process[/yellow]"
        )


if __name__ == "__main__":
    main()
