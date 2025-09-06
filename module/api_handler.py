import time
import io
import re
import base64
from typing import List, Optional, Dict, Any, Union, Tuple, Callable, Iterable
import json
from google import genai
from google.genai import types
from mistralai import Mistral
from openai import OpenAI
from rich_pixels import Pixels
from rich.progress import Progress
from rich.console import Console
from rich.text import Text
from PIL import Image
from pathlib import Path
import functools
import random
from utils.console_util import (
    CaptionLayout,
    CaptionPairImageLayout,
    CaptionAndRateLayout,
    MarkdownLayout,
)
from utils.wdtagger import format_description, split_name_series
from module.providers.gemini_utils import upload_or_get

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
            if getattr(args, "qwenVL_api_key", "") != "" and mime.startswith("video"):
                return "qwenvl"
            if getattr(args, "glm_api_key", "") != "" and mime.startswith("video"):
                return "glm"
            if getattr(args, "pixtral_api_key", "") != "" and (
                mime.startswith("image") or mime.startswith("application")
            ):
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
                m = re.match(r"^\s*change\s+(.+?)\s+to\s+(.+?)\s*$", raw_task, flags=re.IGNORECASE)
                if m and built is None:
                    built = apply_template("change_a_to_b", m.group(1).strip(), m.group(2).strip())

                if built is None:
                    m = re.match(r"^\s*(transform|convert)\s+style\s+(.+?)\s+to\s+(.+?)\s*$", raw_task, flags=re.IGNORECASE)
                    if m:
                        built = apply_template("transform_style_a_to_b", m.group(2).strip(), m.group(3).strip())

                if built is None:
                    m = re.match(r"^\s*combine\s+(.+?)\s+(and|with)\s+(.+?)\s*$", raw_task, flags=re.IGNORECASE)
                    if m:
                        built = apply_template("combine_a_and_b", m.group(1).strip(), m.group(3).strip())

                if built is None:
                    m = re.match(r"^\s*add\s+(.+?)\s+to\s+(.+?)\s*$", raw_task, flags=re.IGNORECASE)
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

    def prepare_media(uri, mime, args, console, scan_pair_extras: bool = False):
        """Prepare media for requests.
        Returns a dict with optional keys: 'image': {blob, pixels, pair?, pair_extras?}, 'audio': {bytes}
        No exceptions are raised here; callers decide how to handle missing parts.
        """
        result: Dict[str, Any] = {}

        if mime.startswith("image"):
            base64_image, pixels = encode_image(uri)
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
                    pair_blob, pair_pixels = encode_image(str(pair_uri))
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
                                extra_blob, _ = encode_image(str(pth))
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
        client = OpenAI(
            api_key=args.step_api_key, base_url="https://api.stepfun.com/v1"
        )

        if mime.startswith("video"):
            file = client.files.create(file=open(uri, "rb"), purpose="storage")
            console.print(f"[blue]Uploaded video file:[/blue] {file}")
        elif mime.startswith("image"):
            media = prepare_media(uri, mime, args, console)
            image_media = media.get("image", {})
            blob = image_media.get("blob")
            pixels = image_media.get("pixels")
            if args.pair_dir != "":
                if not image_media.get("pair"):
                    return ""
                pair_blob = image_media["pair"]["blob"]
                pair_pixels = image_media["pair"]["pixels"]

        def _attempt_stepfun() -> str:
            start_time = time.time()
            if mime.startswith("video"):
                completion = client.chat.completions.create(
                    model=args.step_model_path,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {"url": "stepfile://" + file.id},
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        },
                    ],
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=8192,
                    stream=True,
                )
            elif mime.startswith("image"):
                if args.pair_dir != "":
                    completion = client.chat.completions.create(
                        model=args.step_model_path,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": blob,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": pair_blob,
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            },
                        ],
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=8192,
                        stream=True,
                    )
                else:
                    completion = client.chat.completions.create(
                        model=args.step_model_path,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": blob,
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            },
                        ],
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=8192,
                        stream=True,
                    )

            if progress and task_id is not None:
                progress.update(task_id, description="Generating captions")
            response_text = collect_stream_stepfun(completion, console)

            elapsed_time = time.time() - start_time
            console.print(
                f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
            )

            try:
                console.print(response_text)
            except Exception:
                console.print(Text(response_text))

            response_text = response_text.replace(
                "[green]", "<font color='green'>"
            ).replace("[/green]", "</font>")

            if mime.startswith("video"):
                content = extract_code_block_content(response_text, "srt", console)
                if not content:
                    raise Exception("RETRY_EMPTY_CONTENT")
                if progress and task_id is not None:
                    progress.update(task_id, description="Processing media...")
                return content
            elif mime.startswith("image"):
                if args.pair_dir and pair_pixels:
                    caption_and_rate_layout = CaptionPairImageLayout(
                        description=response_text,
                        pixels=pixels,
                        pair_pixels=pair_pixels,
                        panel_height=32,
                        console=console,
                    )
                    caption_and_rate_layout.print(title=Path(uri).name)
                    return response_text
                else:
                    caption_and_rate_layout = CaptionAndRateLayout(
                        tag_description="",
                        rating=[],
                        average_score=0,
                        long_description=response_text,
                        pixels=pixels,
                        panel_height=32,
                        console=console,
                    )
                    caption_and_rate_layout.print(title=Path(uri).name)
                    return response_text

        result = with_retry(
            _attempt_stepfun,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e)) else None)),
        )
        return result

    elif provider == "qwenvl":
        import dashscope

        file = f"file://{Path(uri).resolve().as_posix()}"

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

        def _attempt_qwenvl() -> str:
            start_time = time.time()
            responses = dashscope.MultiModalConversation.call(
                model=args.qwenVL_model_path,
                messages=messages,
                api_key=args.qwenVL_api_key,
                stream=True,
                incremental_output=True,
            )

            if progress and task_id is not None:
                progress.update(task_id, description="Generating captions")
            response_text = collect_stream_qwen(responses, console)

            elapsed_time = time.time() - start_time
            console.print(
                f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
            )

            try:
                console.print(response_text)
            except Exception:
                console.print(Text(response_text))

            response_text = response_text.replace(
                "[green]", "<font color='green'>"
            ).replace("[/green]", "</font>")

            content = extract_code_block_content(response_text, "srt", console)
            if not content:
                # Trigger retry when content is empty
                raise Exception("RETRY_EMPTY_CONTENT")

            if progress and task_id is not None:
                progress.update(task_id, description="Processing media...")
            return content

        content = with_retry(
            _attempt_qwenvl,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e)) else None)),
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
            start_time = time.time()
            responses = client.chat.completions.create(
                model=args.glm_model_path,
                messages=messages,
                stream=True,
            )

            if progress and task_id is not None:
                progress.update(task_id, description="Generating captions")
            response_text = collect_stream_glm(responses, console)

            elapsed_time = time.time() - start_time
            console.print(
                f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
            )

            try:
                console.print(response_text)
            except Exception:
                console.print(Text(response_text))

            response_text = response_text.replace(
                "[green]", "<font color='green'>"
            ).replace("[/green]", "</font>")

            content = extract_code_block_content(response_text, "srt", console)
            if not content:
                raise Exception("RETRY_EMPTY_CONTENT")

            if progress and task_id is not None:
                progress.update(task_id, description="Processing media...")
            return content

        content = with_retry(
            _attempt_glm,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e)) else None)),
        )
        return content

    elif provider == "pixtral" and (
        mime.startswith("image") or mime.startswith("application")
    ):

        client = Mistral(api_key=args.pixtral_api_key)
        captions = []

        if mime.startswith("image") and args.pair_dir == "":
            system_prompt, _ = get_prompts(config, mime, args, provider, console)
            character_prompt = ""
            character_name = ""
            if args.dir_name:
                dir_prompt = Path(uri).parent.name or ""
                character_name = split_name_series(dir_prompt)
                character_prompt = f"If there is a person/character or more in the image you must refer to them as {character_name}.\n"
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
                        console.print(
                            f"[yellow]Retrying in {args.wait_time} seconds...[/yellow]"
                        )
                        elapsed_time = time.time() - start_time
                        if elapsed_time < args.wait_time:
                            time.sleep(args.wait_time - elapsed_time)
                    else:
                        console.print(
                            f"[red]Failed to upload PDF after {args.max_retries} attempts. Skipping.[/red]"
                        )
                        return ""

        attempt_counter = {"i": 0}

        def _attempt_pixtral() -> str:
            attempt_counter["i"] += 1
            attempt = attempt_counter["i"]
            start_time = time.time()
            if mime.startswith("application"):
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "document_url",
                        "document_url": signed_url.url,
                    },
                    include_image_base64=args.document_image,
                )
                content = ocr_response.pages
                console.print(f"[bold cyan]PDF共有 {len(content)} 页[/bold cyan]")

                for page in content:
                    # Extract the first image from the page if available
                    if page.images and len(page.images) > 0:
                        first_image = page.images[0]
                        if (
                            hasattr(first_image, "image_base64")
                            and first_image.image_base64
                        ):
                            try:
                                base64_str = first_image.image_base64
                                # 处理data URL格式
                                if base64_str.startswith("data:"):
                                    # 提取实际的base64内容
                                    base64_content = base64_str.split(",", 1)[1]
                                    image_data = base64.b64decode(base64_content)
                                else:
                                    image_data = base64.b64decode(base64_str)

                                ocr_image = Image.open(io.BytesIO(image_data))
                                ocr_pixels = Pixels.from_image(
                                    ocr_image,
                                    resize=(
                                        ocr_image.width // 18,
                                        ocr_image.height // 18,
                                    ),
                                )
                            except Exception as e:
                                console.print(f"[yellow]Error loading image: {e}[/yellow]")
                                ocr_pixels = None
                        else:
                            console.print(
                                "[yellow]Image found but no base64 data available[/yellow]"
                            )
                            ocr_pixels = None
                    else:
                        ocr_pixels = None

                    markdown_layout = MarkdownLayout(
                        pixels=ocr_pixels,
                        markdown_content=page.markdown,
                        panel_height=32,
                        console=console,
                    )
                    markdown_layout.print(
                        title=f"{Path(uri).name} -  Page {page.index+1}"
                    )

            elif args.ocr:
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                )
                content = ocr_response.pages[0].markdown

                # 使用MarkdownLayout显示Markdown内容
                markdown_layout = MarkdownLayout(
                    pixels=pixels,
                    markdown_content=content,
                    panel_height=32,
                    console=console,
                )
                markdown_layout.print(title=Path(uri).name)

            else:
                chat_response = client.chat.complete(
                    model=args.pixtral_model_path, messages=messages
                )
                content = chat_response.choices[0].message.content

                short_description, long_description = process_llm_response(content)

                if len(captions) > 0:
                    tag_description = (
                        (
                            prompt.rsplit("<s>[INST]", 1)[-1]
                            .rsplit(">.", 1)[-1]
                            .rsplit(").", 1)[-1]
                            .replace(" from", ",")
                        )
                        .rsplit("[IMG][/INST]", 1)[0]
                        .strip()
                    )
                    short_description, short_highlight_rate = format_description(
                        short_description, tag_description
                    )
                    long_description, long_highlight_rate = format_description(
                        long_description, tag_description
                    )
                else:
                    tag_description = ""
                    short_highlight_rate = 0
                    long_highlight_rate = 0

                # 使用CaptionLayout显示图片和字幕
                caption_layout = CaptionLayout(
                    tag_description=tag_description,
                    short_description=short_description,
                    long_description=long_description,
                    pixels=pixels,
                    short_highlight_rate=short_highlight_rate,
                    long_highlight_rate=long_highlight_rate,
                    panel_height=32,
                    console=console,
                )

                caption_layout.print(title=Path(uri).name)

            # 计算已经消耗的时间，动态调整等待时间
            elapsed_time = time.time() - start_time
            if elapsed_time < args.wait_time:
                time.sleep(args.wait_time - elapsed_time)
            console.print(
                f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
            )

            # 校验，仅针对非 OCR 情况
            if not args.ocr:
                try:
                    cn = character_name
                except Exception:
                    cn = ""
                if cn:
                    clean_char_name = cn.split(",")[0].split(" from ")[0].strip("<>")
                    if clean_char_name not in content:
                        console.print()
                        console.print(Text(content))
                        console.print(
                            f"Attempt {attempt}/{args.max_retries}: Character name [green]{clean_char_name}[/green] not found"
                        )
                        raise Exception("RETRY_PIXTRAL_CHAR")

                if "###" not in content:
                    console.print(Text(content))
                    console.print(Text("No ###, retrying...", style="yellow"))
                    raise Exception("RETRY_PIXTRAL_NO_MARK")

                if (
                    any(f"{i}women" in tag_description for i in range(2, 5))
                    or ("1man" in tag_description and "1woman" in tag_description)
                    or "multiple girls" in tag_description
                    or "multiple boys" in tag_description
                ):
                    tags_highlightrate = args.tags_highlightrate * 100 / 2
                else:
                    tags_highlightrate = args.tags_highlightrate * 100
                if (
                    int(re.search(r"\d+", str(long_highlight_rate)).group())
                    < tags_highlightrate
                ) and len(captions) > 0:
                    console.print(
                        f"[red]long_description highlight rate is too low: {long_highlight_rate}%, retrying...[/red]"
                    )
                    raise Exception("RETRY_PIXTRAL_RATE")

            if isinstance(content, str) and "502" in content:
                console.print(
                    f"[yellow]Attempt {attempt}/{args.max_retries}: Received 502 error[/yellow]"
                )
                raise Exception("RETRY_PIXTRAL_502")

            return content

        result = with_retry(
            _attempt_pixtral,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (
                59.0
                if "429" in str(e)
                else (
                    args.wait_time
                    if (
                        "502" in str(e)
                        or "RETRY_PIXTRAL_" in str(e)
                    )
                    else None
                )
            ),
        )
        return result

    elif provider == "gemini":
        generation_config = (
            config["generation_config"][args.gemini_model_path.replace(".", "_")]
            if config["generation_config"][args.gemini_model_path.replace(".", "_")]
            else config["generation_config"]["default"]
        )

        if args.gemini_task and mime.startswith("image"):
            args.gemini_model_path = "gemini-2.5-flash-image-preview"
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
            response_schema=(
                image_response_schema
                if mime.startswith("image") and args.gemini_task == ""
                else None
            ),
            thinking_config=(
                (
                    types.ThinkingConfig(
                        thinking_budget=(
                            generation_config["thinking_budget"]
                            if "thinking_budget" in generation_config
                            else -1
                        ),
                    )
                )
                if args.gemini_task == ""
                else None
            ),
        )

        console.print(f"generation_config: {generation_config}")

        client = genai.Client(api_key=args.gemini_api_key)

        pair_blob_list = []

        if (
            mime.startswith("video")
            or mime.startswith("audio")
            and Path(uri).stat().st_size >= 20 * 1024 * 1024
        ):
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
            console.print(f"[blue]Generating captions...[/blue]")
            start_time = time.time()

            if mime.startswith("video") or (
                mime.startswith("audio") and Path(uri).stat().st_size >= 20 * 1024 * 1024
            ):
                response = client.models.generate_content_stream(
                    model=args.gemini_model_path,
                    contents=[
                        types.Part.from_uri(file_uri=files[0].uri, mime_type=mime),
                        types.Part.from_text(text=prompt),
                    ],
                    config=genai_config,
                )
            elif mime.startswith("audio"):
                media_local = prepare_media(uri, mime, args, console)
                audio_blob = media_local.get("audio", {}).get("bytes") or Path(uri).read_bytes()
                response = client.models.generate_content_stream(
                    model=args.gemini_model_path,
                    contents=[
                        types.Part.from_bytes(data=audio_blob, mime_type=mime),
                        types.Part.from_text(text=prompt),
                    ],
                    config=genai_config,
                )
            elif mime.startswith("image"):
                if args.pair_dir != "":
                    image_parts = [
                        types.Part.from_bytes(data=pair_blob, mime_type="image/jpeg")
                    ]
                    if pair_blob_list:
                        image_parts.extend(
                            [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in pair_blob_list]
                        )
                    image_parts.append(types.Part.from_bytes(data=blob, mime_type="image/jpeg"))
                    image_parts.append(types.Part.from_text(text=prompt))
                    response = client.models.generate_content_stream(
                        model=args.gemini_model_path,
                        contents=image_parts,
                        config=genai_config,
                    )
                else:
                    response = client.models.generate_content_stream(
                        model=args.gemini_model_path,
                        contents=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=blob, mime_type="image/jpeg"),
                        ],
                        config=genai_config,
                    )
            else:
                # Fallback: shouldn't happen, treat as retryable
                raise Exception("RETRY_UNSUPPORTED_MIME")

            if progress and task_id is not None:
                progress.update(task_id, description="Generating captions")
            response_text = collect_stream_gemini(response, uri, console)
            if mime.startswith("image"):
                response_text = response_text.replace("*", "").strip()

            elapsed_time = time.time() - start_time
            console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

            try:
                console.print(response_text)
            except Exception:
                console.print(Text(response_text))

            if mime.startswith("image"):
                if isinstance(response_text, str) and args.gemini_task == "":
                    try:
                        captions = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        console.print(f"[red]Error decoding JSON: {e}[/red]")
                        if "Expecting value: line 1 column 1 (char 0)" in str(e):
                            console.print("[red]Image was filtered, skipping[/red]")
                            return ""
                        else:
                            raise e
                else:
                    captions = response_text
                if args.gemini_task != "":
                    caption_and_rate_layout = CaptionAndRateLayout(
                        tag_description="",
                        rating=[],
                        average_score=0.0,
                        long_description=response_text,
                        pixels=pixels,
                        panel_height=32,
                        console=console,
                    )
                    caption_and_rate_layout.print(title=Path(uri).name)
                    return response_text
                elif args.pair_dir and pair_pixels:
                    description = captions.get("prompt", "")
                    caption_and_rate_layout = CaptionPairImageLayout(
                        description=description,
                        pixels=pixels,
                        pair_pixels=pair_pixels,
                        panel_height=32,
                        console=console,
                    )
                    caption_and_rate_layout.print(title=Path(uri).name)
                    return captions.get("prompt", "")
                else:
                    description = captions.get("description", "")
                    scores = captions.get("scores", [])
                    average_score = captions.get("average_score", 0.0)
                    caption_and_rate_layout = CaptionAndRateLayout(
                        tag_description="",
                        rating=scores,
                        average_score=average_score,
                        long_description=description,
                        pixels=pixels,
                        panel_height=32,
                        console=console,
                    )
                    caption_and_rate_layout.print(title=Path(uri).name)
                    return response_text

            response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")
            content = extract_code_block_content(response_text, "srt", console)
            if not content:
                raise Exception("RETRY_EMPTY_CONTENT")
            if progress and task_id is not None:
                progress.update(task_id, description="Processing media...")
            return content

        result = with_retry(
            _attempt_gemini,
            max_retries=args.max_retries,
            base_wait=args.wait_time,
            console=console,
            classify_err=lambda e: (59.0 if "429" in str(e) else (args.wait_time if ("502" in str(e) or "RETRY_EMPTY_CONTENT" in str(e) or "RETRY_UNSUPPORTED_MIME" in str(e)) else None)),
        )
        return result


@functools.lru_cache(maxsize=128)
def encode_image(image_path: str) -> Optional[Tuple[str, Pixels]]:
    """Encode the image to base64 format with size optimization.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        with Image.open(image_path) as image:
            image.load()
            if "xmp" in image.info:
                del image.info["xmp"]

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
        console.print(
            f"[red]Error:[/red] Permission denied accessing file - {image_path}"
        )
    except OSError as e:
        # Specifically handle XMP and metadata-related errors
        if "XMP data is too long" in str(e):
            console.print(
                f"[yellow]Warning:[/yellow] Skipping image with XMP data error - {image_path}"
            )
        else:
            console.print(
                f"[red]Error:[/red] OS error processing file {image_path}: {str(e)}"
            )
    except ValueError as e:
        console.print(
            f"[red]Error:[/red] Invalid value while processing {image_path}: {str(e)}"
        )
    except Exception as e:
        console.print(
            f"[red]Error:[/red] Unexpected error processing {image_path}: {str(e)}"
        )
    return None, None


def extract_code_block_content(response_text, code_type=None, console=None):
    """从响应文本中提取被```包围的代码块内容

    Args:
        response_text (str): API返回的完整响应文本
        code_type (str, optional): 代码块类型标识符(如'srt')，如果提供则会从内容开头移除
        console (Console, optional): Rich Console对象，用于输出日志信息

    Returns:
        str: 提取出的代码块内容，如果没有找到则返回空字符串
    """
    if not response_text:
        return ""

    # 查找所有的 ``` 标记位置
    markers = []
    start = 0
    while True:
        pos = response_text.find("```", start)
        if pos == -1:
            break
        markers.append(pos)
        start = pos + 3

    # 确保找到至少一对标记
    if len(markers) >= 2:
        # 获取最后一对标记之间的内容
        first_marker = markers[-2]
        second_marker = markers[-1]
        content = response_text[first_marker + 3 : second_marker].strip()

        # 如果指定了代码类型，移除开头的代码类型标识
        if code_type and content.startswith(code_type):
            content = content[len(code_type) :].strip()

        if console:
            console.print(f"[blue]Extracted content length:[/blue] {len(content)}")
            console.print(f"[blue]Found {len(markers)} ``` markers[/blue]")
        return content
    else:
        if console:
            console.print(f"[red]Not enough ``` markers: found {len(markers)}[/red]")
        return ""


def with_retry(
    fn: Callable[[], Any],
    max_retries: int,
    base_wait: float,
    console: Optional[Console] = None,
    classify_err: Optional[Callable[[Exception], Optional[float]]] = None,
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
                raise
            wait = classifier(e) or base_wait
            jitter = wait * 0.2
            sleep_for = max(0.0, wait + random.uniform(-jitter, jitter))
            elapsed = time.time() - start_time
            remaining = max(0.0, sleep_for - elapsed)
            if console and remaining > 0:
                console.print(
                    f"[yellow]Retrying in {remaining:.0f} seconds...[/yellow]"
                )
            if remaining > 0:
                time.sleep(remaining)


def collect_stream_stepfun(completion: Iterable[Any], console: Console) -> str:
    """Collect streamed text from StepFun(OpenAI-compatible) responses."""
    chunks: List[str] = []
    for chunk in completion:
        if (
            hasattr(chunk.choices[0].delta, "content")
            and chunk.choices[0].delta.content is not None
        ):
            chunks.append(chunk.choices[0].delta.content)
            console.print(".", end="", style="blue")
    console.print("\n")
    return "".join(chunks)


def collect_stream_qwen(responses: Iterable[Any], console: Console) -> str:
    """Collect streamed text from QwenVL responses.

    Preserve original behavior: print raw chunk, print the whole aggregated text each step.
    """
    chunks = ""
    for chunk in responses:
        print(chunk)
        try:
            # Original code assumes first element exists
            chunks += chunk.output.choices[0].message.content[0]["text"]
        except Exception:
            # Fallback: try generic text fields if shape differs
            try:
                chunks += getattr(chunk, "text", "") or ""
            except Exception:
                pass
        try:
            console.print(chunks, end="", overflow="ellipsis")
        except Exception:
            console.print(Text(chunks), end="", overflow="ellipsis")
        finally:
            console.file.flush()
    console.print("\n")
    return chunks


def collect_stream_glm(responses: Iterable[Any], console: Console) -> str:
    """Collect streamed text from GLM responses.

    Preserve original behavior: print raw chunk, print the whole aggregated text each step.
    """
    chunks = ""
    for chunk in responses:
        print(chunk)
        if (
            hasattr(chunk.choices[0].delta, "content")
            and chunk.choices[0].delta.content is not None
        ):
            chunks += chunk.choices[0].delta.content
        try:
            console.print(chunks, end="", overflow="ellipsis")
        except Exception:
            console.print(Text(chunks), end="", overflow="ellipsis")
        finally:
            console.file.flush()
    console.print("\n")
    return chunks


def collect_stream_gemini(response: Iterable[Any], uri: str, console: Console) -> str:
    """Collect streamed text and inline_data from Gemini responses.

    - Accumulate chunk.text into final response_text (same as original)
    - For inline_data, save paired text buffer and image/file to disk
    - Preserve printing/flush behaviors
    """
    chunks: List[str] = []
    part_index = 0
    text_buffer: List[str] = []
    for chunk in response:
        if (
            not getattr(chunk, "candidates", None)
            or not chunk.candidates[0].content
            or not chunk.candidates[0].content.parts
        ):
            continue
        if getattr(chunk, "text", None):
            chunks.append(chunk.text)
            console.print("")
            try:
                console.print(chunk.text, end="", overflow="ellipsis")
            except Exception:
                console.print(Text(chunk.text), end="", overflow="ellipsis")
            finally:
                console.file.flush()
        for part in chunk.candidates[0].content.parts:
            if getattr(part, "text", None):
                text_content = str(part.text)
                if text_content:
                    console.print(text_content)
                    text_buffer.append(text_content)
            if getattr(part, "inline_data", None):
                part_index += 1
                clean_text = "".join(text_buffer).strip()
                if clean_text:
                    text_path = Path(uri).with_name(f"{Path(uri).stem}_{part_index}.txt")
                    save_binary_file(text_path, clean_text.encode("utf-8"))
                    console.print(
                        f"[blue]Text part saved to: {text_path.name}[/blue]"
                    )
                image_path = Path(uri).with_stem(f"{Path(uri).stem}_{part_index}")
                save_binary_file(image_path, part.inline_data.data)
                console.print(
                    f"[blue]File of mime type {part.inline_data.mime_type} saved to: {image_path.name}[/blue]"
                )
                text_buffer.clear()
    console.print("\n")
    return "".join(chunks)


def process_llm_response(result: str) -> tuple[str, str]:
    """处理LLM返回的结果, 提取短描述和长描述。

    Args:
        result: LLM返回的原始结果文本

    Returns:
        tuple[str, str]: 返回 (short_description, long_description) 元组
    """
    if result and "###" in result:
        short_description, long_description = result.split("###")[-2:]

        # 更彻底地清理描述
        short_description = " ".join(short_description.split(":", 1)[-1].split())
        long_description = " ".join(long_description.split(":", 1)[-1].split())
    else:
        short_description = ""
        long_description = ""

    return short_description, long_description


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
