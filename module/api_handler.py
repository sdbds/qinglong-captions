import base64
import functools
import io
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from mistralai import Mistral
from openai import OpenAI
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from module.providers.ark_provider import attempt_ark as ark_attempt
from module.providers.gemini_provider import attempt_gemini as gemini_attempt
from module.providers.gemini_utils import upload_or_get
from module.providers.glm_provider import attempt_glm as glm_attempt
from module.providers.pixtral_provider import attempt_pixtral as pixtral_attempt
from module.providers.qwenvl_provider import attempt_qwenvl as qwenvl_attempt
from module.providers.stepfun_provider import attempt_stepfun as stepfun_attempt
from utils.stream_util import sanitize_filename, split_name_series

console = Console(color_system="truecolor", force_terminal=True)


def api_process_batch(
    uri: str,
    mime: str,
    config,
    args,
    sha256hash: str,
    progress: Progress = None,
    task_id=None,
) -> str:
    # Use global console if no progress is provided
    global console
    if progress is not None:
        console = progress.console

    def get_provider(args, mime):
        """Decide which provider branch to use based on args and mime.
        Order matches existing branch order to preserve behavior.
        """
        try:
            if getattr(args, "step_api_key", "") != "":
                return "stepfun"
            if getattr(args, "ark_api_key", "") != "" and mime.startswith("video"):
                return "ark"
            if getattr(args, "qwenVL_api_key", "") != "" and mime.startswith("video"):
                return "qwenvl"
            if getattr(args, "glm_api_key", "") != "" and mime.startswith("video"):
                return "glm"
            # OCR model selection with document_image logic
            ocr_model = getattr(args, "ocr_model", "")
            if ocr_model != "":
                # For PDF and application files, always process with OCR
                if mime.startswith("application"):
                    return ocr_model
                # For images, only process if document_image is enabled
                elif mime.startswith("image") and getattr(args, "document_image", False):
                    return ocr_model
            # VLM model selection for image tasks
            vlm_model = getattr(args, "vlm_image_model", "")
            if vlm_model != "" and mime.startswith("image") and getattr(args, "pair_dir", "") == "":
                return vlm_model
            if getattr(args, "pixtral_api_key", "") != "" and (mime.startswith("image") or mime.startswith("application")):
                return "pixtral"
            if getattr(args, "gemini_api_key", "") != "":
                return "gemini"
        except Exception:
            pass
        return "none"

    def get_prompts(config, mime, args, provider, console):
        """Return (system_prompt, prompt) per provider/mime and args.
        Mirrors existing per-branch selection. Gemini task prompt is synthesized here.
        """
        prompts = config["prompts"]

        # Base defaults by mime
        system_prompt = prompts.get("system_prompt", "")
        prompt = prompts.get("prompt", "")
        if mime.startswith("video"):
            system_prompt = prompts.get("video_system_prompt", system_prompt)
            prompt = prompts.get("video_prompt", prompt)
        elif mime.startswith("audio"):
            system_prompt = prompts.get("audio_system_prompt", system_prompt)
            prompt = prompts.get("audio_prompt", prompt)
        elif mime.startswith("image"):
            system_prompt = prompts.get("image_system_prompt", system_prompt)
            prompt = prompts.get("image_prompt", prompt)

        # Provider-specific overrides
        if provider == "stepfun":
            if mime.startswith("video"):
                system_prompt = prompts.get("step_video_system_prompt", system_prompt)
                prompt = prompts.get("step_video_prompt", prompt)
            elif mime.startswith("image"):
                if getattr(args, "pair_dir", "") != "":
                    system_prompt = prompts.get("image_pair_system_prompt", system_prompt)
                    prompt = prompts.get("image_pair_prompt", prompt)
                else:
                    system_prompt = prompts.get("image_system_prompt", system_prompt)
                    prompt = prompts.get("image_prompt", prompt)

        elif provider == "qwenvl":
            if mime.startswith("video"):
                system_prompt = prompts.get("qwenvl_video_system_prompt", system_prompt)
                prompt = prompts.get("qwenvl_video_prompt", prompt)

        elif provider == "ark":
            if mime.startswith("video"):
                system_prompt = prompts.get("ark_video_system_prompt", system_prompt)
                prompt = prompts.get("ark_video_prompt", prompt)

        elif provider == "glm":
            if mime.startswith("video"):
                system_prompt = prompts.get("glm_video_system_prompt", system_prompt)
                prompt = prompts.get("glm_video_prompt", prompt)

        elif provider == "pixtral":
            if mime.startswith("image"):
                if getattr(args, "pair_dir", "") != "":
                    system_prompt = prompts.get("pair_image_system_prompt", system_prompt)
                    prompt = prompts.get("pair_image_prompt", prompt)
                else:
                    system_prompt = prompts.get("pixtral_image_system_prompt", system_prompt)
                    prompt = prompts.get("pixtral_image_prompt", prompt)

        elif provider == "moondream":
            if mime.startswith("image"):
                if getattr(args, "pair_dir", "") != "":
                    system_prompt = prompts.get("pair_image_system_prompt", system_prompt)
                    prompt = prompts.get("pair_image_prompt", prompt)
                else:
                    system_prompt = prompts.get("pixtral_image_system_prompt", system_prompt)
                    prompt = prompts.get("pixtral_image_prompt", prompt)

        elif provider == "qwen_vl_local":
            if mime.startswith("image"):
                if getattr(args, "pair_dir", "") != "":
                    system_prompt = prompts.get("pair_image_system_prompt", system_prompt)
                    prompt = prompts.get("pair_image_prompt", prompt)
                else:
                    system_prompt = prompts.get("pixtral_image_system_prompt", system_prompt)
                    prompt = prompts.get("pixtral_image_prompt", prompt)

        elif provider == "step_vl_local":
            if mime.startswith("image"):
                if getattr(args, "pair_dir", "") != "":
                    system_prompt = prompts.get("pair_image_system_prompt", system_prompt)
                    prompt = prompts.get("pair_image_prompt", prompt)
                else:
                    system_prompt = prompts.get("pixtral_image_system_prompt", system_prompt)
                    prompt = prompts.get("pixtral_image_prompt", prompt)

        elif provider == "gemini":
            # Gemini image task mode builds prompt from task templates
            if getattr(args, "gemini_task", "") and mime.startswith("image"):
                system_prompt = prompts.get("task_system_prompt", system_prompt)
                task_prompts = prompts.get("task", {})
                raw_task = str(getattr(args, "gemini_task"))

                def apply_template(template_key: str, a_val: str, b_val: str) -> Optional[str]:
                    template = task_prompts.get(template_key)
                    if not template:
                        return None
                    p = template
                    if "{a}" in p or "{b}" in p or "<a>" in p or "<b>" in p:
                        p = p.replace("{a}", a_val).replace("{b}", b_val)
                        p = p.replace("<a>", a_val).replace("<b>", b_val)
                    else:
                        p = re.sub(r"\ba\b", a_val, p)
                        p = re.sub(r"\bb\b", b_val, p)
                    return p

                built = None
                m = re.match(
                    r"^\s*change\s+(.+?)\s+to\s+(.+?)\s*$",
                    raw_task,
                    flags=re.IGNORECASE,
                )
                if m and built is None:
                    built = apply_template("change_a_to_b", m.group(1).strip(), m.group(2).strip())

                if built is None:
                    m = re.match(
                        r"^\s*(transform|convert)\s+style\s+(.+?)\s+to\s+(.+?)\s*$",
                        raw_task,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        built = apply_template(
                            "transform_style_a_to_b",
                            m.group(2).strip(),
                            m.group(3).strip(),
                        )

                if built is None:
                    m = re.match(
                        r"^\s*combine\s+(.+?)\s+(and|with)\s+(.+?)\s*$",
                        raw_task,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        built = apply_template("combine_a_and_b", m.group(1).strip(), m.group(3).strip())

                if built is None:
                    m = re.match(
                        r"^\s*add\s+(.+?)\s+to\s+(.+?)\s*$",
                        raw_task,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        built = apply_template("add_a_to_b", m.group(1).strip(), m.group(2).strip())

                if built is None:
                    prompt = task_prompts.get(raw_task) or raw_task
                else:
                    prompt = built
                try:
                    console.print(f"[blue]prompt: {prompt}[/blue]")
                except Exception:
                    pass
            elif getattr(args, "pair_dir", "") and mime.startswith("image"):
                system_prompt = prompts.get("pair_image_system_prompt", system_prompt)
                prompt = prompts.get("pair_image_prompt", prompt)

        return system_prompt, prompt

    def prepare_media(uri, mime, args, console, scan_pair_extras: bool = False, to_rgb: bool = False):
        """Prepare media for requests.
        Returns a dict with optional keys: 'image': {blob, pixels, pair?, pair_extras?}, 'audio': {bytes}
        No exceptions are raised here; callers decide how to handle missing parts.
        """
        result: Dict[str, Any] = {}

        if mime.startswith("image"):
            base64_image, pixels = encode_image(uri, to_rgb=to_rgb)
            image_obj: Dict[str, Any] = {
                "blob": base64_image,
                "pixels": pixels,
            }

            pair_dir = getattr(args, "pair_dir", "")
            if pair_dir:
                pair_uri = (Path(pair_dir) / Path(uri).name).resolve()
                if not pair_uri.exists():
                    console.print(f"[red]Pair image {pair_uri} not found[/red]")
                else:
                    console.print(f"[yellow]Pair image {pair_uri} found[/yellow]")
                    pair_blob, pair_pixels = encode_image(str(pair_uri), to_rgb=to_rgb)
                    image_obj["pair"] = {"blob": pair_blob, "pixels": pair_pixels}

                if scan_pair_extras:
                    try:
                        base_dir = Path(pair_dir).resolve()
                        stem = Path(uri).stem
                        primary_ext = Path(uri).suffix.lower()
                        extras: List[Tuple[int, Path]] = []
                        for pth in base_dir.iterdir():
                            if (
                                pth.is_file()
                                and pth.name.startswith(f"{stem}_")
                                and pth.suffix.lower() == primary_ext
                                and pth.resolve() != pair_uri
                            ):
                                name_stem = pth.stem
                                if len(name_stem) > len(stem) + 1 and name_stem[len(stem)] == "_":
                                    num_part = name_stem[len(stem) + 1 :]
                                    if num_part.isdigit():
                                        extras.append((int(num_part), pth))
                        extras.sort(key=lambda t: t[0])
                        pair_extras: List[str] = []
                        for _, pth in extras:
                            try:
                                extra_blob, _ = encode_image(str(pth), to_rgb=to_rgb)
                                if extra_blob:
                                    pair_extras.append(extra_blob)
                                    console.print(f"[blue]Paired extra: {pth.name}[/blue]")
                            except Exception as ee:
                                console.print(f"[red]Failed to encode paired extra {pth}: {ee}[/red]")
                        if pair_extras:
                            image_obj["pair_extras"] = pair_extras
                    except Exception as scan_err:
                        console.print(f"[yellow]Scan pair_dir extras failed: {scan_err}[/yellow]")

            result["image"] = image_obj

        if mime.startswith("audio"):
            try:
                audio_blob = Path(uri).read_bytes()
            except Exception:
                audio_blob = None
            result["audio"] = {"bytes": audio_blob}

        return result

    provider = get_provider(args, mime)
    system_prompt, prompt = get_prompts(config, mime, args, provider, console)

    if provider == "stepfun":
        client = OpenAI(api_key=args.step_api_key, base_url="https://api.stepfun.com/v1")

        # Predefine pair placeholders for both image/video paths
        pair_blob = None
        pair_pixels = None
        pair_uri_path = None

        if mime.startswith("video"):
            file = client.files.create(file=open(uri, "rb"), purpose="storage")
            console.print(f"[blue]Uploaded video file:[/blue] {file}")
        elif mime.startswith("image"):
            media = prepare_media(uri, mime, args, console, to_rgb=True)
            image_media = media.get("image", {})
            blob = image_media.get("blob")
            pixels = image_media.get("pixels")
            if args.pair_dir != "":
                if not image_media.get("pair"):
                    return ""
                pair_blob = image_media["pair"]["blob"]
                pair_pixels = image_media["pair"]["pixels"]
                pair_uri_path = str((Path(args.pair_dir) / Path(uri).name).resolve())

        def _attempt_stepfun() -> str:
            # Use outer-scope pair_pixels directly; it's predefined above
            has_pair = bool(args.pair_dir and pair_pixels)
            return stepfun_attempt(
                client=client,
                model_path=args.step_model_path,
                mime=mime,
                system_prompt=system_prompt,
                prompt=prompt,
                console=console,
                progress=progress,
                task_id=task_id,
                uri=uri,
                image_blob=(blob if mime.startswith("image") else None),
                image_pixels=(pixels if mime.startswith("image") else None),
                has_pair=has_pair,
                pair_blob=(pair_blob if has_pair else None),
                pair_pixels=(pair_pixels if has_pair else None),
                pair_uri=(pair_uri_path if has_pair else None),
                video_file_id=(file.id if mime.startswith("video") else None),
            )

        result = with_retry(
            _attempt_stepfun,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e)) else None)
            ),
            on_exhausted=lambda e: (console.print(Text(f"StepFun retries exhausted: {e}", style="yellow")) or ""),
        )
        return result

    elif provider == "qwenvl":
        file = f"file://{Path(uri).resolve().as_posix()}"

        if mime.startswith("video"):
            console.print(f"[blue]Uploading video file:[/blue] {file}")
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "video": file,
                            # "video": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241115/cqqkru/1.mp4",
                        },
                        {"text": prompt},
                    ],
                },
            ]
        elif mime.startswith("image"):
            console.print(f"[blue]Preparing image file:[/blue] {file}")
            content_items = []
            pair_dir = getattr(args, "pair_dir", "")
            if pair_dir:
                pair_path = (Path(pair_dir) / Path(uri).name).resolve()
                if pair_path.exists():
                    pair_file = f"file://{pair_path.as_posix()}"
                    console.print(f"[yellow]Pair image found:[/yellow] {pair_file}")
                    content_items.extend([
                        {"image": file},
                        {"image": pair_file},
                    ])
                else:
                    console.print(f"[red]Pair image not found:[/red] {pair_path}")
                    return ""
            else:
                content_items.append({"image": file})

            content_items.append({"text": prompt})

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": content_items,
                },
            ]
        else:
            console.print(f"[yellow]Unsupported mime for qwenvl branch:[/yellow] {mime}")
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                    ],
                },
            ]

        def _attempt_qwenvl() -> str:
            return qwenvl_attempt(
                model_path=args.qwenVL_model_path,
                api_key=args.qwenVL_api_key,
                messages=messages,
                console=console,
                progress=progress,
                task_id=task_id,
            )

        content = with_retry(
            _attempt_qwenvl,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e)) else None)
            ),
            on_exhausted=lambda e: (console.print(Text(f"QwenVL retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "glm":
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=args.glm_api_key)

        with open(uri, "rb") as video_file:
            video_base = base64.b64encode(video_file.read()).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_base}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        def _attempt_glm() -> str:
            return glm_attempt(
                client=client,
                model_path=args.glm_model_path,
                messages=messages,
                console=console,
                progress=progress,
                task_id=task_id,
            )

        content = with_retry(
            _attempt_glm,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e)) else None)
            ),
            on_exhausted=lambda e: (console.print(Text(f"GLM retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "ark":
        try:
            from volcenginesdkarkruntime import Ark  # local import to avoid hard dep
        except Exception as e:
            console.print(Text(f"Ark SDK not installed: {e}", style="red"))
            return ""

        if not getattr(args, "ark_model_path", ""):
            console.print(Text("Ark model path is empty. Please set --ark_model_path.", style="red"))
            return ""

        client = Ark(api_key=args.ark_api_key)
        console.print(f"[blue]Ark model:[/blue] {getattr(args, 'ark_model_path', '')}")
        ark_section = {}
        try:
            if isinstance(config, dict):
                ark_section = config.get("ark", {}) or {}
        except Exception:
            ark_section = {}
        cfg_fps = ark_section.get("fps")
        ark_fps = float(cfg_fps) if cfg_fps is not None else getattr(args, "ark_fps", 2)
        console.print(f"[blue]Ark fps:[/blue] {ark_fps}")

        with open(uri, "rb") as video_file:
            video_base = base64.b64encode(video_file.read()).decode("utf-8")
        try:
            file_size = Path(uri).stat().st_size
        except Exception:
            file_size = -1
        console.print(f"[blue]Ark input size:[/blue] {file_size} bytes; base64 length: {len(video_base)}")

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{mime};base64,{video_base}",
                            "fps": ark_fps,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        def _attempt_ark() -> str:
            try:
                return ark_attempt(
                    client=client,
                    model_path=args.ark_model_path,
                    messages=messages,
                    console=console,
                    progress=progress,
                    task_id=task_id,
                )
            except Exception as e:
                # Extra diagnostics for Ark attempt
                try:
                    console.print(Text(f"Ark attempt raised: {type(e).__name__}: {e}", style="red"))
                    console.print(Text(traceback.format_exc(), style="red"))
                except Exception:
                    pass
                raise

        content = with_retry(
            _attempt_ark,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e)) else None)
            ),
            on_exhausted=lambda e: (console.print(Text(f"Ark retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "deepseek_ocr":
        # Prepare media preview only for images
        pixels = None
        if mime.startswith("image"):
            media = prepare_media(uri, mime, args, console, to_rgb=True)
            image_media = media.get("image", {})
            pixels = image_media.get("pixels")
        # Build DeepSeek-OCR prompt: use prompts.deepseek_ocr_prompt from config, then hardcoded default.
        prompts_section = config.get("prompts", {})
        deepseek_prompt = prompts_section.get(
            "deepseek_ocr_prompt",
            "<image>\n<|grounding|>Convert the document to markdown. ",
        )

        # Output directory near input path (same behavior as other providers storing alongside inputs)
        output_dir = str(Path(uri).with_suffix(""))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        deepseek_section = {}
        try:
            if isinstance(config, dict):
                deepseek_section = config.get("deepseek_ocr", {}) or {}
        except Exception:
            deepseek_section = {}
        cfg_model_id = deepseek_section.get("model_id", "deepseek-ai/DeepSeek-OCR-2")
        cfg_base_size = deepseek_section.get("base_size")
        cfg_image_size = deepseek_section.get("image_size", 768)
        cfg_crop_mode = deepseek_section.get("crop_mode")

        def _attempt_deepseek() -> str:
            try:
                from module.providers.deepseek_ocr_provider import (
                    attempt_deepseek_ocr as deepseek_attempt,
                )
            except Exception as e:
                console.print(Text(f"DeepSeek-OCR provider not available: {e}", style="red"))
                raise

            return deepseek_attempt(
                uri=uri,
                console=console,
                progress=progress,
                task_id=task_id,
                model_id=cfg_model_id,
                prompt_text=deepseek_prompt,
                pixels=pixels,
                output_dir=output_dir,
                base_size=(int(cfg_base_size) if cfg_base_size is not None else getattr(args, "deepseek_base_size", 1024)),
                image_size=(int(cfg_image_size) if cfg_image_size is not None else getattr(args, "deepseek_image_size", 768)),
                crop_mode=(bool(cfg_crop_mode) if cfg_crop_mode is not None else getattr(args, "deepseek_crop_mode", True)),
            )

        content = with_retry(
            _attempt_deepseek,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: args.wait_time,
            on_exhausted=lambda e: (console.print(Text(f"DeepSeek-OCR retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "hunyuan_ocr":
        # Prepare media preview only for images
        pixels = None
        if mime.startswith("image"):
            media = prepare_media(uri, mime, args, console, to_rgb=True)
            image_media = media.get("image", {})
            pixels = image_media.get("pixels")

        # Build HunyuanOCR prompt from config; fallback to default Chinese document parsing prompt
        prompts_section = config.get("prompts", {})
        hunyuan_prompt = prompts_section.get("hunyuan_ocr_prompt", "")

        # Output directory near input path
        output_dir = str(Path(uri).with_suffix(""))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        hunyuan_section = {}
        try:
            if isinstance(config, dict):
                hunyuan_section = config.get("hunyuan_ocr", {}) or {}
        except Exception:
            hunyuan_section = {}
        cfg_model_id = hunyuan_section.get("model_id", "tencent/HunyuanOCR")
        cfg_max_new_tokens = hunyuan_section.get("max_new_tokens")

        def _attempt_hunyuan() -> str:
            try:
                from module.providers.hunyuan_ocr_provider import (
                    attempt_hunyuan_ocr as hunyuan_attempt,
                )
            except Exception as e:
                console.print(Text(f"HunyuanOCR provider not available: {e}", style="red"))
                raise

            return hunyuan_attempt(
                uri=uri,
                console=console,
                progress=progress,
                task_id=task_id,
                model_id=cfg_model_id,
                prompt_text=hunyuan_prompt if hunyuan_prompt else None,
                pixels=pixels,
                output_dir=output_dir,
                max_new_tokens=(int(cfg_max_new_tokens) if cfg_max_new_tokens is not None else 16384),
            )

        content = with_retry(
            _attempt_hunyuan,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: args.wait_time,
            on_exhausted=lambda e: (console.print(Text(f"HunyuanOCR retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "olmocr":
        # Prepare media preview only for images
        pixels = None
        if mime.startswith("image"):
            media = prepare_media(uri, mime, args, console, to_rgb=True)
            image_media = media.get("image", {})
            blob = image_media.get("blob")
            pixels = image_media.get("pixels")

        # Build OLMOCR prompt from config; fallback to empty to let provider decide
        prompts_section = config.get("prompts", {})
        olmocr_prompt = prompts_section.get(
            "olmocr_prompt",
            "",
        )

        # Output directory near input path
        output_dir = str(Path(uri).with_suffix(""))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        olmocr_section = {}
        try:
            if isinstance(config, dict):
                olmocr_section = config.get("olmocr", {}) or {}
        except Exception:
            olmocr_section = {}
        cfg_temperature = olmocr_section.get("temperature")
        cfg_max_new_tokens = olmocr_section.get("max_new_tokens")

        def _attempt_olmocr() -> str:
            try:
                from module.providers.olmocr_provider import (
                    attempt_olmocr as olmocr_attempt,
                )
            except Exception as e:
                console.print(Text(f"OLMOCR provider not available: {e}", style="red"))
                raise

            return olmocr_attempt(
                uri=uri,
                mime=mime,
                console=console,
                progress=progress,
                task_id=task_id,
                prompt_text=olmocr_prompt,
                pixels=pixels,
                output_dir=output_dir,
                base64_image=blob,
                model_id=olmocr_section.get("model_id", "allenai/olmOCR-2-7B-1025"),
                temperature=(float(cfg_temperature) if cfg_temperature is not None else 0.1),
                max_new_tokens=(int(cfg_max_new_tokens) if cfg_max_new_tokens is not None else 512),
            )

        content = with_retry(
            _attempt_olmocr,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: args.wait_time,
            on_exhausted=lambda e: (console.print(Text(f"OLMOCR retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "paddle_ocr":
        media = prepare_media(uri, mime, args, console, to_rgb=True)
        image_media = media.get("image", {})
        pixels = image_media.get("pixels")

        output_dir = str(Path(uri).with_suffix(""))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def _attempt_paddle_ocr() -> str:
            try:
                from module.providers.paddle_ocr_provider import (
                    attempt_paddle_ocr as paddle_attempt,
                )
            except Exception as e:
                console.print(Text(f"PaddleOCR provider not available: {e}", style="red"))
                raise

            return paddle_attempt(
                uri=uri,
                console=console,
                progress=progress,
                task_id=task_id,
                pixels=pixels,
                output_dir=output_dir,
            )

        content = with_retry(
            _attempt_paddle_ocr,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: args.wait_time,
            on_exhausted=lambda e: (console.print(Text(f"PaddleOCR retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "nanonets_ocr":
        # Prepare media preview only for images
        pixels = None
        if mime.startswith("image"):
            media = prepare_media(uri, mime, args, console, to_rgb=True)
            image_media = media.get("image", {})
            pixels = image_media.get("pixels")

        # Build Nanonets OCR prompt from config
        prompts_section = config.get("prompts", {})
        nanonets_prompt = prompts_section.get("nanonets_ocr_prompt", "")

        # Output directory near input path
        output_dir = str(Path(uri).with_suffix(""))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        nanonets_section = {}
        try:
            if isinstance(config, dict):
                nanonets_section = config.get("nanonets_ocr", {}) or {}
        except Exception:
            nanonets_section = {}
        cfg_model_id = nanonets_section.get("model_id", "nanonets/Nanonets-OCR2-3B")
        cfg_max_new_tokens = nanonets_section.get("max_new_tokens")

        def _attempt_nanonets() -> str:
            try:
                from module.providers.nanonets_ocr_provider import (
                    attempt_nanonets_ocr as nanonets_attempt,
                )
            except Exception as e:
                console.print(Text(f"Nanonets OCR provider not available: {e}", style="red"))
                raise

            return nanonets_attempt(
                uri=uri,
                console=console,
                progress=progress,
                task_id=task_id,
                model_id=cfg_model_id,
                prompt_text=nanonets_prompt if nanonets_prompt else None,
                pixels=pixels,
                output_dir=output_dir,
                max_new_tokens=(int(cfg_max_new_tokens) if cfg_max_new_tokens is not None else 15000),
            )

        content = with_retry(
            _attempt_nanonets,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: args.wait_time,
            on_exhausted=lambda e: (console.print(Text(f"Nanonets OCR retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider in ["moondream", "qwen_vl_local", "step_vl_local"] and mime.startswith("image"):
        media = prepare_media(uri, mime, args, console)
        image_media = media.get("image", {})
        pixels = image_media.get("pixels")

        captions: List[str] = []
        captions_path = Path(uri).with_suffix(".txt")
        if captions_path.exists():
            with open(captions_path, "r", encoding="utf-8") as f:
                captions = [line.strip() for line in f.readlines()]

        # Read VLM config from corresponding section
        vlm_section = {}
        try:
            if isinstance(config, dict):
                vlm_section = config.get(provider, {}) or {}
        except Exception:
            vlm_section = {}
        cfg_reasoning = vlm_section.get("reasoning")

        # Build provider prompt from config
        try:
            _, provider_prompt = get_prompts(config, mime, args, provider, console)
        except Exception:
            provider_prompt = ""

        # If no prompt from config, use VLM section prompt
        if not provider_prompt:
            provider_prompt = vlm_section.get("prompt", "")

        def _attempt_vlm() -> str:
            if provider == "moondream":
                try:
                    from module.providers.moondream_provider import (
                        attempt_moondream as vlm_attempt,
                    )
                except Exception as e:
                    console.print(Text(f"Moondream provider not available: {e}", style="red"))
                    raise

                return vlm_attempt(
                    model_id=vlm_section.get("model_id", "moondream/moondream3-preview"),
                    mime=mime,
                    console=console,
                    progress=progress,
                    task_id=task_id,
                    uri=uri,
                    pixels=pixels,
                    image=image_media,
                    captions=captions,
                    tags_highlightrate=getattr(args, "tags_highlightrate", 0.0),
                    prompt_text=provider_prompt,
                    reasoning=(bool(cfg_reasoning) if cfg_reasoning is not None else False),
                    ocr=(getattr(args, "ocr_model", "none") == "moondream"),
                    task=vlm_section.get("tasks", "caption"),
                )
            elif provider == "qwen_vl_local":
                # Build messages for Qwen-VL local model
                file = f"file://{Path(uri).resolve().as_posix()}"
                content_items = []

                pair_dir = getattr(args, "pair_dir", "")
                if pair_dir:
                    pair_path = (Path(pair_dir) / Path(uri).name).resolve()
                    if pair_path.exists():
                        pair_file = f"file://{pair_path.as_posix()}"
                        console.print(f"[yellow]Pair image found:[/yellow] {pair_file}")
                        content_items.extend([{"image": file}, {"image": pair_file}])
                    else:
                        console.print(f"[red]Pair image not found:[/red] {pair_path}")
                        return ""
                else:
                    content_items.append({"image": file})

                content_items.append({"text": provider_prompt})

                messages = [
                    {"role": "system", "content": [{"text": system_prompt}]},
                    {"role": "user", "content": content_items},
                ]

                return qwenvl_attempt(
                    model_path=vlm_section.get("model_id", "Qwen/Qwen2-VL-7B-Instruct"),
                    api_key="",
                    messages=messages,
                    console=console,
                    progress=progress,
                    task_id=task_id,
                )
            elif provider == "step_vl_local":
                try:
                    from module.providers.stepfun_provider import attempt_stepfun
                except Exception as e:
                    console.print(Text(f"Step3-VL Local provider not available: {e}", style="red"))
                    raise

                # Prepare media for Step3-VL local model
                media = prepare_media(uri, mime, args, console, to_rgb=True)
                image_media_full = media.get("image", {})
                blob = image_media_full.get("blob")
                pixels_full = image_media_full.get("pixels")
                pair_blob = None
                pair_pixels_full = None

                if args.pair_dir != "":
                    if not image_media_full.get("pair"):
                        return ""
                    pair_blob = image_media_full["pair"]["blob"]
                    pair_pixels_full = image_media_full["pair"]["pixels"]

                has_pair = bool(args.pair_dir and pair_pixels_full)
                return attempt_stepfun(
                    client=None,
                    model_path="",
                    mime=mime,
                    system_prompt=system_prompt,
                    prompt=provider_prompt,
                    console=console,
                    progress=progress,
                    task_id=task_id,
                    uri=uri,
                    image_blob=blob,
                    image_pixels=pixels_full,
                    has_pair=has_pair,
                    pair_blob=pair_blob,
                    pair_pixels=pair_pixels_full,
                    video_file_id=None,
                )
            else:
                raise ValueError(f"Unsupported VLM provider: {provider}")

        content = with_retry(
            _attempt_vlm,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_MOONDREAM_" in str(e)) else None)
            ),
            on_exhausted=lambda e: (console.print(Text(f"Moondream retries exhausted: {e}", style="yellow")) or ""),
        )
        return content

    elif provider == "pixtral" or provider == "pixtral_ocr":
        client = Mistral(api_key=args.pixtral_api_key)
        captions = []
        character_name = ""

        if mime.startswith("image") and args.pair_dir == "":
            system_prompt, _ = get_prompts(config, mime, args, provider, console)
            character_prompt = ""
            if args.dir_name:
                dir_prompt = Path(uri).parent.name or ""
                character_name = split_name_series(dir_prompt)
                character_prompt = (
                    f"If there is a person/character or more in the image you must refer to them as {character_name}.\n"
                )
                character_name = f"{character_name}, " if character_name else ""
            config_prompt = config["prompts"]["pixtral_image_prompt"]

            # Read captions from file if it exists
            captions_path = Path(uri).with_suffix(".txt")
            if captions_path.exists():
                with open(captions_path, "r", encoding="utf-8") as f:
                    captions = [line.strip() for line in f.readlines()]

            prompt = Text(
                f"<s>[INST]{character_prompt}{character_name}{captions[0] if len(captions) > 0 else config_prompt}\n[IMG][/INST]"
            ).plain

            media = prepare_media(uri, mime, args, console)
            image_media = media.get("image", {})
            base64_image = image_media.get("blob")
            pixels = image_media.get("pixels")
            if base64_image is None or pixels is None:
                return ""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                },
            ]

        elif mime.startswith("image") and args.pair_dir != "":
            system_prompt, prompt = get_prompts(config, mime, args, provider, console)

            media = prepare_media(uri, mime, args, console)
            image_media = media.get("image", {})
            base64_image = image_media.get("blob")
            pixels = image_media.get("pixels")
            if base64_image is None or pixels is None:
                return ""

            pair = image_media.get("pair")
            if not pair:
                return ""
            base64_image2 = pair.get("blob")
            pixels2 = pair.get("pixels")
            if base64_image2 is None or pixels2 is None:
                return ""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image2}",
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

        elif mime.startswith("application"):
            for upload_attempt in range(args.max_retries):
                try:
                    uploaded_pdf = client.files.upload(
                        file={
                            "file_name": f"{sanitize_filename(uri)}.pdf",
                            "content": open(uri, "rb"),
                        },
                        purpose="ocr",
                    )
                    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
                    break
                except Exception as e:
                    error_msg = Text(str(e), style="red")
                    console.print(f"[red]Error uploading PDF: {error_msg}[/red]")
                    if upload_attempt < args.max_retries - 1:
                        console.print(f"[yellow]Retrying in {args.wait_time} seconds...[/yellow]")
                        time.sleep(args.wait_time)
                    else:
                        console.print(f"[red]Failed to upload PDF after {args.max_retries} attempts. Skipping.[/red]")
                        return ""

        def _attempt_pixtral() -> str:
            # Route to provider attempt by scenario
            if mime.startswith("application"):
                return pixtral_attempt(
                    client=client,
                    model_path=args.pixtral_model_path,
                    mime=mime,
                    console=console,
                    progress=progress,
                    task_id=task_id,
                    uri=uri,
                    document_image=args.document_image,
                    signed_url_url=signed_url.url,
                )
            elif getattr(args, "ocr_model", "none") == "pixtral":
                return pixtral_attempt(
                    client=client,
                    model_path=args.pixtral_model_path,
                    mime=mime,
                    console=console,
                    progress=progress,
                    task_id=task_id,
                    uri=uri,
                    ocr=True,
                    base64_image=base64_image,
                    pixels=pixels,
                )
            else:
                # image chat
                return pixtral_attempt(
                    client=client,
                    model_path=args.pixtral_model_path,
                    mime=mime,
                    console=console,
                    progress=progress,
                    task_id=task_id,
                    uri=uri,
                    messages=messages,
                    pixels=pixels,
                    captions=captions,
                    prompt_text=prompt,
                    character_name=character_name,
                    tags_highlightrate=getattr(args, "tags_highlightrate", 0.0),
                )

        result = with_retry(
            _attempt_pixtral,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_PIXTRAL_" in str(e)) else None)
            ),
            on_exhausted=lambda e: (console.print(Text(f"Pixtral retries exhausted: {e}", style="yellow")) or ""),
        )
        return result

    elif provider == "gemini":
        generation_config = (
            config["generation_config"][args.gemini_model_path.replace(".", "_")]
            if config["generation_config"][args.gemini_model_path.replace(".", "_")]
            else config["generation_config"]["default"]
        )

        if args.gemini_task and mime.startswith("image"):
            args.gemini_model_path = "gemini-3-pro-image-preview"
            # get_prompts already constructed the prompt for the task
            pass
        elif args.pair_dir and mime.startswith("image"):
            # get_prompts already selected pair image prompts
            pass

        if args.pair_dir != "":
            image_response_schema = genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={
                    "prompt": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                },
            )
        else:
            image_response_schema = genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["scores", "average_score", "description"],
                properties={
                    "scores": genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=[
                            "Costume & Makeup & Prop Presentation/Accuracy",
                            "Character Portrayal & Posing",
                            "Setting & Environment Integration",
                            "Lighting & Mood",
                            "Composition & Framing",
                            "Storytelling & Concept",
                            "Level of S*e*x*y",
                            "Figure",
                            "Overall Impact & Uniqueness",
                        ],
                        properties={
                            "Costume & Makeup & Prop Presentation/Accuracy": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Character Portrayal & Posing": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Setting & Environment Integration": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Lighting & Mood": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Composition & Framing": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Storytelling & Concept": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Level of S*e*x*y": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Figure": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "Overall Impact & Uniqueness": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                        },
                    ),
                    "total_score": genai.types.Schema(
                        type=genai.types.Type.INTEGER,
                    ),
                    "average_score": genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                    ),
                    "description": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                    "character_name": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                    "series": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                },
            )

        genai_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            top_k=generation_config["top_k"],
            candidate_count=config["generation_config"]["candidate_count"],
            max_output_tokens=generation_config["max_output_tokens"],
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                # types.SafetySetting(
                #     category=types.HarmCategory.HARM_CATEGORY_IMAGE_HATE,
                #     threshold=types.HarmBlockThreshold.OFF,
                # ),
                # types.SafetySetting(
                #     category=types.HarmCategory.HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT,
                #     threshold=types.HarmBlockThreshold.OFF,
                # ),
                # types.SafetySetting(
                #     category=types.HarmCategory.HARM_CATEGORY_IMAGE_HARASSMENT,
                #     threshold=types.HarmBlockThreshold.OFF,
                # ),
                # types.SafetySetting(
                #     category=types.HarmCategory.HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT,
                #     threshold=types.HarmBlockThreshold.OFF,
                # ),
            ],
            response_mime_type=(
                "application/json"
                if mime.startswith("image") and args.gemini_task == ""
                else generation_config["response_mime_type"]
            ),
            response_modalities=generation_config["response_modalities"],
            response_schema=(image_response_schema if mime.startswith("image") and args.gemini_task == "" else None),
            thinking_config=(
                (
                    types.ThinkingConfig(
                        thinking_budget=(generation_config["thinking_budget"] if "thinking_budget" in generation_config else -1),
                    )
                )
                if args.gemini_task == ""
                else None
            ),
        )

        console.print(f"generation_config: {generation_config}")

        client = genai.Client(api_key=args.gemini_api_key)

        pair_blob_list = []

        if mime.startswith("video") or mime.startswith("audio") and Path(uri).stat().st_size >= 20 * 1024 * 1024:
            upload_success, files = with_retry(
                lambda: upload_or_get(
                    client=client,
                    uri=uri,
                    mime=mime,
                    sha256hash=sha256hash,
                    max_retries=args.max_retries,
                    wait_time=args.wait_time,
                    output_console=console,
                ),
                max_retries=args.max_retries,
                base_wait=args.wait_time,
                console=console,
            )
            if not upload_success:
                return ""
        elif mime.startswith("image"):
            media = prepare_media(uri, mime, args, console, scan_pair_extras=True)
            image_media = media.get("image", {})
            blob = image_media.get("blob")
            pixels = image_media.get("pixels")
            if args.pair_dir != "":
                pair = image_media.get("pair")
                if not pair:
                    console.print(f"[red]Pair image not prepared for {Path(uri).name}[/red]")
                    return ""
                pair_blob = pair.get("blob")
                pair_pixels = pair.get("pixels")
                # Additionally load extras collected by prepare_media
                pair_blob_list = image_media.get("pair_extras", [])

        # Some files have a processing delay. Wrap generation and processing in with_retry
        def _attempt_gemini() -> str:
            audio_bytes = None
            if mime.startswith("audio") and not (Path(uri).stat().st_size >= 20 * 1024 * 1024):
                media_local = prepare_media(uri, mime, args, console)
                audio_bytes = media_local.get("audio", {}).get("bytes") or None

            return gemini_attempt(
                client=client,
                model_path=args.gemini_model_path,
                mime=mime,
                prompt=prompt,
                console=console,
                progress=progress,
                task_id=task_id,
                uri=uri,
                genai_config=genai_config,
                files=(
                    files
                    if (mime.startswith("video") or (mime.startswith("audio") and Path(uri).stat().st_size >= 20 * 1024 * 1024))
                    else None
                ),
                audio_bytes=audio_bytes,
                image_blob=(blob if mime.startswith("image") else None),
                pixels=(pixels if mime.startswith("image") else None),
                pair_blob=(pair_blob if (mime.startswith("image") and args.pair_dir != "") else None),
                pair_pixels=(pair_pixels if (mime.startswith("image") and args.pair_dir != "") else None),
                pair_blob_list=(pair_blob_list if mime.startswith("image") else None),
                gemini_task=getattr(args, "gemini_task", ""),
            )

        result = with_retry(
            _attempt_gemini,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0
                if "429" in str(e)
                else (
                    args.wait_time
                    if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e) or "RETRY_UNSUPPORTED_MIME" in str(e))
                    else None
                )
            ),
            on_exhausted=lambda e: (console.print(Text(f"Gemini retries exhausted: {e}", style="yellow")) or ""),
        )
        return result


@functools.lru_cache(maxsize=128)
def encode_image(image_path: str, to_rgb: bool = False) -> Optional[Tuple[str, Pixels]]:
    """Encode the image to base64 format with size optimization.

    Args:
        image_path: Path to the image file
        to_rgb: If True, convert the image to RGB before further processing (for local models)

    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        with Image.open(image_path) as image:
            image.load()
            if "xmp" in image.info:
                del image.info["xmp"]

            # Optional early RGB conversion for local providers
            if to_rgb:
                try:
                    image = image.convert("RGB")
                except Exception:
                    pass

            # Calculate dimensions that are multiples of 16
            max_size = 1024
            width, height = image.size

            def calculate_dimensions(width, height, max_size: int) -> Tuple[int, int]:
                aspect_ratio = width / height
                if width > height:
                    new_width = min(max_size, (width // 16) * 16)
                    new_height = ((int(new_width / aspect_ratio)) // 16) * 16
                else:
                    new_height = min(max_size, (height // 16) * 16)
                    new_width = ((int(new_height * aspect_ratio)) // 16) * 16

                # Ensure dimensions don't exceed max_size
                if new_width > max_size:
                    new_width = max_size
                    new_height = ((int(new_width / aspect_ratio)) // 16) * 16
                if new_height > max_size:
                    new_height = max_size
                    new_width = ((int(new_height * aspect_ratio)) // 16) * 16

                return new_width, new_height

            new_width, new_height = calculate_dimensions(width, height, max_size)
            image = image.resize((new_width, new_height), Image.LANCZOS).convert("RGB")

            pixels = Pixels.from_image(
                image,
                resize=(image.width // 18, image.height // 18),
            )

            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8"), pixels

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found - {image_path}")
    except Image.UnidentifiedImageError:
        console.print(f"[red]Error:[/red] Cannot identify image file - {image_path}")
    except PermissionError:
        console.print(f"[red]Error:[/red] Permission denied accessing file - {image_path}")
    except OSError as e:
        # Specifically handle XMP and metadata-related errors
        if "XMP data is too long" in str(e):
            console.print(f"[yellow]Warning:[/yellow] Skipping image with XMP data error - {image_path}")
        else:
            console.print(f"[red]Error:[/red] OS error processing file {image_path}: {str(e)}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid value while processing {image_path}: {str(e)}")
    except Exception as e:
        console.print(f"[red]Error:[/red] Unexpected error processing {image_path}: {str(e)}")
    return None, None


def with_retry(
    fn: Callable[[], Any],
    max_retries: int,
    base_wait: float,
    console: Optional[Console] = None,
    classify_err: Optional[Callable[[Exception], Optional[float]]] = None,
    on_exhausted: Optional[Callable[[Exception], Any]] = None,
) -> Any:
    """Run callable with retries and jitter backoff.

    - If classify_err is not provided, default classifier applies:
      - '429' -> wait 59 seconds
      - '502' -> wait base_wait
    - Adds ±20% jitter to avoid thundering herd
    """

    def default_classifier(e: Exception) -> Optional[float]:
        s = str(e)
        if "429" in s:
            return 59.0
        if "502" in s:
            return base_wait
        return None

    classifier = classify_err or default_classifier

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries - 1:
                if console:
                    console.print(Text(str(e), style="red"))
                if on_exhausted is not None:
                    try:
                        return on_exhausted(e)
                    except Exception:
                        pass
                raise
            # Log error location and type for diagnostics on every retry attempt
            if console:
                try:
                    tb = traceback.extract_tb(e.__traceback__)
                    if tb:
                        last = tb[-1]
                        console.print(
                            Text(
                                f"[with_retry] {attempt + 1}/{max_retries} failed at "
                                f"{Path(last.filename).name}:{last.lineno} in {last.name} -> "
                                f"{type(e).__name__}: {e}",
                                style="yellow",
                            )
                        )
                    else:
                        console.print(
                            Text(
                                f"[with_retry] {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}",
                                style="yellow",
                            )
                        )
                except Exception:
                    try:
                        console.print(
                            Text(
                                f"[with_retry] {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}",
                                style="yellow",
                            )
                        )
                    except Exception:
                        pass
            wait = classifier(e) or base_wait
            jitter = wait * 0.2
            sleep_for = max(0.0, wait + random.uniform(-jitter, jitter))
            elapsed = time.time() - start_time
            remaining = max(0.0, sleep_for - elapsed)
            if console and remaining > 0:
                console.print(Text(f"{attempt + 1}/{max_retries}", style="yellow"))
                console.print(f"[yellow]Retrying in {remaining:.0f} seconds...[/yellow]")
            if remaining > 0:
                time.sleep(remaining)
