import time
import io
import re
import base64
from typing import List, Optional, Dict, Any, Union, Tuple
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
from utils.console_util import (
    CaptionLayout,
    CaptionPairImageLayout,
    CaptionAndRateLayout,
    MarkdownLayout,
)
from utils.stream_util import sanitize_filename
from utils.wdtagger import format_description, split_name_series

console = Console()


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

    system_prompt = config["prompts"]["system_prompt"]
    prompt = config["prompts"]["prompt"]

    if mime.startswith("video"):
        system_prompt = config["prompts"]["video_system_prompt"]
        prompt = config["prompts"]["video_prompt"]
    elif mime.startswith("audio"):
        system_prompt = config["prompts"]["audio_system_prompt"]
        prompt = config["prompts"]["audio_prompt"]
    elif mime.startswith("image"):
        system_prompt = config["prompts"]["image_system_prompt"]
        prompt = config["prompts"]["image_prompt"]

    if args.step_api_key != "":

        if mime.startswith("video"):
            system_prompt = config["prompts"]["step_video_system_prompt"]
            prompt = config["prompts"]["step_video_prompt"]
        elif mime.startswith("image"):
            if args.pair_dir != "":
                system_prompt = config["prompts"]["image_pair_system_prompt"]
                prompt = config["prompts"]["image_pair_prompt"]
            else:
                system_prompt = config["prompts"]["image_system_prompt"]
                prompt = config["prompts"]["image_prompt"]

        client = OpenAI(
            api_key=args.step_api_key, base_url="https://api.stepfun.com/v1"
        )

        if mime.startswith("video"):
            file = client.files.create(file=open(uri, "rb"), purpose="storage")
            console.print(f"[blue]Uploaded video file:[/blue] {file}")
        elif mime.startswith("image"):
            blob, pixels = encode_image(uri)
            if args.pair_dir != "":
                pair_uri = (Path(args.pair_dir) / Path(uri).name).resolve()
                if not pair_uri.exists():
                    console.print(f"[red]Pair image {pair_uri} not found[/red]")
                    return ""
                else:
                    console.print(f"[yellow]Pair image {pair_uri} found[/yellow]")
                pair_blob, pair_pixels = encode_image(pair_uri)

        for attempt in range(args.max_retries):
            try:
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
                chunks = []
                for chunk in completion:
                    if (
                        hasattr(chunk.choices[0].delta, "content")
                        and chunk.choices[0].delta.content is not None
                    ):
                        chunks.append(chunk.choices[0].delta.content)
                        console.print(".", end="", style="blue")

                console.print("\n")
                response_text = "".join(chunks)

                elapsed_time = time.time() - start_time
                console.print(
                    f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
                )

                try:
                    console.print(response_text)
                except Exception as e:
                    console.print(Text(response_text))

                response_text = response_text.replace(
                    "[green]", "<font color='green'>"
                ).replace("[/green]", "</font>")

                if mime.startswith("video"):
                    content = extract_code_block_content(response_text, "srt", console)
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
                if not content:
                    continue

                if progress and task_id is not None:
                    progress.update(task_id, description="Processing media...")
                return content

            except Exception as e:
                error_msg = Text(str(e), style="red")
                console.print(f"[red]Error processing: {error_msg}[/red]")
                if attempt < args.max_retries - 1:
                    console.print(
                        f"[yellow]Retrying in {args.wait_time} seconds...[/yellow]"
                    )
                    elapsed_time = time.time() - start_time
                    if elapsed_time < args.wait_time:
                        time.sleep(args.wait_time - elapsed_time)
                else:
                    console.print(
                        f"[red]Failed to process after {args.max_retries} attempts. Skipping.[/red]"
                    )
                continue
        return ""

    elif args.qwenVL_api_key != "" and mime.startswith("video"):
        import dashscope

        system_prompt = config["prompts"]["qwenvl_video_system_prompt"]
        prompt = config["prompts"]["qwenvl_video_prompt"]

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

        for attempt in range(args.max_retries):
            try:
                start_time = time.time()
                responses = dashscope.MultiModalConversation.call(
                    model=args.qwenVL_model_path,
                    messages=messages,
                    api_key=args.qwenVL_api_key,
                    stream=True,
                    incremental_output=True,
                )

                chunks = ""
                if progress and task_id is not None:
                    progress.update(task_id, description="Generating captions")
                for chunk in responses:
                    print(chunk)
                    chunks += chunk.output.choices[0].message.content[0]["text"]
                    try:
                        console.print(chunks, end="", overflow="ellipsis")
                    except Exception as e:
                        console.print(Text(chunks), end="", overflow="ellipsis")
                    finally:
                        console.file.flush()
                console.print("\n")
                response_text = chunks

                elapsed_time = time.time() - start_time
                console.print(
                    f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
                )

                try:
                    console.print(response_text)
                except Exception as e:
                    console.print(Text(response_text))

                response_text = response_text.replace(
                    "[green]", "<font color='green'>"
                ).replace("[/green]", "</font>")

                content = extract_code_block_content(response_text, "srt", console)
                if not content:
                    continue

                if progress and task_id is not None:
                    progress.update(task_id, description="Processing media...")
                return content
            except Exception as e:
                error_msg = Text(str(e), style="red")
                console.print(f"[red]Error processing: {error_msg}[/red]")
                if attempt < args.max_retries - 1:
                    console.print(
                        f"[yellow]Retrying in {args.wait_time} seconds...[/yellow]"
                    )
                    elapsed_time = time.time() - start_time
                    if elapsed_time < args.wait_time:
                        time.sleep(args.wait_time - elapsed_time)
                else:
                    console.print(
                        f"[red]Failed to process after {args.max_retries} attempts. Skipping.[/red]"
                    )
                continue
        return ""

    elif args.glm_api_key != "" and mime.startswith("video"):
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=args.glm_api_key)

        system_prompt = config["prompts"]["glm_video_system_prompt"]
        prompt = config["prompts"]["glm_video_prompt"]

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

        for attempt in range(args.max_retries):
            try:
                start_time = time.time()
                responses = client.chat.completions.create(
                    model=args.glm_model_path,
                    messages=messages,
                    stream=True,
                )

                chunks = ""
                if progress and task_id is not None:
                    progress.update(task_id, description="Generating captions")
                for chunk in responses:
                    print(chunk)
                    if (
                        hasattr(chunk.choices[0].delta, "content")
                        and chunk.choices[0].delta.content is not None
                    ):
                        chunks += chunk.choices[0].delta.content
                    try:
                        console.print(chunks, end="", overflow="ellipsis")
                    except Exception as e:
                        console.print(Text(chunks), end="", overflow="ellipsis")
                    finally:
                        console.file.flush()
                console.print("\n")
                response_text = chunks

                elapsed_time = time.time() - start_time
                console.print(
                    f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
                )

                try:
                    console.print(response_text)
                except Exception as e:
                    console.print(Text(response_text))

                response_text = response_text.replace(
                    "[green]", "<font color='green'>"
                ).replace("[/green]", "</font>")

                content = extract_code_block_content(response_text, "srt", console)
                if not content:
                    continue

                if progress and task_id is not None:
                    progress.update(task_id, description="Processing media...")
                return content
            except Exception as e:
                error_msg = Text(str(e), style="red")
                console.print(f"[red]Error processing: {error_msg}[/red]")
                if attempt < args.max_retries - 1:
                    console.print(
                        f"[yellow]Retrying in {args.wait_time} seconds...[/yellow]"
                    )
                    elapsed_time = time.time() - start_time
                    if elapsed_time < args.wait_time:
                        time.sleep(args.wait_time - elapsed_time)
                else:
                    console.print(
                        f"[red]Failed to process after {args.max_retries} attempts. Skipping.[/red]"
                    )
                continue
        return ""

    elif args.pixtral_api_key != "" and (
        mime.startswith("image") or mime.startswith("application")
    ):

        client = Mistral(api_key=args.pixtral_api_key)
        captions = []

        if mime.startswith("image") and args.pair_dir == "":
            system_prompt = config["prompts"]["pixtral_image_system_prompt"]
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

            base64_image, pixels = encode_image(uri)
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
            system_prompt = config["prompts"]["pair_image_system_prompt"]
            prompt = config["prompts"]["pair_image_prompt"]

            base64_image, pixels = encode_image(uri)
            if base64_image is None or pixels is None:
                return ""

            uri2_path = Path(args.pair_dir) / Path(uri).name
            if not uri2_path.exists():
                return ""
            uri2 = str(uri2_path)
            base64_image2, pixels2 = encode_image(uri2)
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

        for attempt in range(args.max_retries):
            start_time = time.time()
            try:
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
                                    console.print(
                                        f"[yellow]Error loading image: {e}[/yellow]"
                                    )
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

                if not args.ocr:
                    if character_name:
                        clean_char_name = (
                            character_name.split(",")[0].split(" from ")[0].strip("<>")
                        )
                        if clean_char_name not in content:
                            console.print()
                            console.print(Text(content))
                            console.print(
                                f"Attempt {attempt + 1}/{args.max_retries}: Character name [green]{clean_char_name}[/green] not found"
                            )
                            continue

                    if "###" not in content:
                        console.print(Text(content))
                        console.print(Text("No ###, retrying...", style="yellow"))
                        continue

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
                        continue

                if "502" in content:
                    console.print(
                        f"[yellow]Attempt {attempt + 1}/{args.max_retries}: Received 502 error[/yellow]"
                    )
                    continue

                return content

            except Exception as e:
                error_msg = Text(str(e), style="red")
                console.print(
                    f"[red]Attempt {attempt + 1}/{args.max_retries}: Error - ", end=""
                )
                console.print(error_msg)
                if attempt < args.max_retries - 1:
                    wait_time = args.wait_time
                    if "429" in str(e):
                        wait_time = 59
                        console.print(
                            f"[yellow]429 error, waiting {wait_time} seconds and retrying...[/yellow]"
                        )

                        elapsed_time = time.time() - start_time
                        if elapsed_time < wait_time:
                            time.sleep(wait_time - elapsed_time)
                        console.print("[green]Retrying...[/green]")
                    time.sleep(wait_time)
                    continue
        return ""

    elif args.gemini_api_key != "":
        generation_config = (
            config["generation_config"][args.gemini_model_path.replace(".", "_")]
            if config["generation_config"][args.gemini_model_path.replace(".", "_")]
            else config["generation_config"]["default"]
        )

        if args.gemini_task and mime.startswith("image"):
            args.gemini_model_path = "gemini-2.0-flash-exp"
            system_prompt = None
            prompt = config["prompts"]["task"][args.gemini_task]

        if args.pair_dir and mime.startswith("image"):
            system_prompt = config["prompts"]["pair_image_system_prompt"]
            prompt = config["prompts"]["pair_image_prompt"]

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
            presence_penalty=0.0,
            frequency_penalty=0.0,
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
            ],
            response_mime_type=(
                "application/json"
                if mime.startswith("image")
                else generation_config["response_mime_type"]
            ),
            response_modalities=generation_config["response_modalities"],
            response_schema=image_response_schema if mime.startswith("image") else None,
            thinking_config=(
                types.ThinkingConfig(
                    thinking_budget=(
                        generation_config["thinking_budget"]
                        if "thinking_budget" in generation_config
                        else -1
                    ),
                )
            ),
        )

        console.print(f"generation_config: {generation_config}")

        client = genai.Client(api_key=args.gemini_api_key)

        if (
            mime.startswith("video")
            or mime.startswith("audio")
            and Path(uri).stat().st_size >= 20 * 1024 * 1024
        ):
            upload_success = False
            files = []
            file = types.File()

            for upload_attempt in range(args.max_retries):
                try:
                    console.print()
                    console.print(f"[blue]checking files for:[/blue] {uri}")
                    try:
                        file = client.files.get(name=sanitize_filename(Path(uri).name))

                        console.print(file)
                        if (
                            base64.b64decode(file.sha256_hash).decode("utf-8")
                            == sha256hash
                            or file.size_bytes == Path(uri).stat().st_size
                        ):
                            console.print()
                            console.print(
                                f"[cyan]File {file.name} is already at {file.uri}[/cyan]"
                            )
                            files = [file]
                            wait_for_files_active(client, files, console)
                            console.print()
                            console.print(
                                f"[green]File {file.name} is already active at {file.uri}[/green]"
                            )
                            upload_success = True
                            break
                        else:
                            console.print(
                                f"[yellow]File {file.name} is already exist but {base64.b64decode(file.sha256_hash).decode('utf-8')} not have same sha256hash {sha256hash}[/yellow]"
                            )
                            client.files.delete(name=sanitize_filename(Path(uri).name))
                            raise Exception("Delete same name file and retry")

                    except Exception as e:
                        console.print()
                        console.print(
                            f"[yellow]File {Path(uri).name} is not exist[/yellow]"
                        )
                        console.print(f"[blue]uploading files for:[/blue] {uri}")
                        try:
                            files = [upload_to_gemini(client, uri, mime_type=mime)]
                        except Exception as uploade:
                            files = [
                                upload_to_gemini(
                                    client,
                                    uri,
                                    mime_type=mime,
                                    name=f"{Path(uri).name}_{int(time.time())}",
                                )
                            ]
                        wait_for_files_active(client, files, console)
                        upload_success = True
                        break

                except Exception as e:
                    console.print(
                        f"[yellow]Upload attempt {upload_attempt + 1}/{args.max_retries} failed: {e}[/yellow]"
                    )
                    if upload_attempt < args.max_retries - 1:
                        time.sleep(
                            args.wait_time * 2
                        )  # Increase wait time between retries
                    else:
                        console.print("[red]All upload attempts failed[/red]")
                        return ""

            if not upload_success:
                console.print("[red]Failed to upload file[/red]")
                return ""
        elif mime.startswith("image"):
            blob, pixels = encode_image(uri)
            if args.pair_dir != "":
                pair_uri = (Path(args.pair_dir) / Path(uri).name).resolve()
                if not pair_uri.exists():
                    console.print(f"[red]Pair image {pair_uri} not found[/red]")
                    return ""
                else:
                    console.print(f"[yellow]Pair image {pair_uri} found[/yellow]")
                pair_blob, pair_pixels = encode_image(pair_uri)

        # Some files have a processing delay. Wait for them to be ready.
        # wait_for_files_active(files)
        for attempt in range(args.max_retries):
            try:
                console.print(f"[blue]Generating captions...[/blue]")
                start_time = time.time()

                if mime.startswith("video") or (
                    mime.startswith("audio")
                    and Path(uri).stat().st_size >= 20 * 1024 * 1024
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
                    audio_blob = Path(uri).read_bytes()
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
                        response = client.models.generate_content_stream(
                            model=args.gemini_model_path,
                            contents=[
                                types.Part.from_bytes(
                                    data=pair_blob, mime_type="image/jpeg"
                                ),
                                types.Part.from_bytes(
                                    data=blob, mime_type="image/jpeg"
                                ),
                                types.Part.from_text(text=prompt),
                            ],
                            config=genai_config,
                        )
                    else:
                        response = client.models.generate_content_stream(
                            model=args.gemini_model_path,
                            contents=[
                                types.Part.from_text(text=prompt),
                                types.Part.from_bytes(
                                    data=blob, mime_type="image/jpeg"
                                ),
                            ],
                            config=genai_config,
                        )

                if progress and task_id is not None:
                    progress.update(task_id, description="Generating captions")
                # 收集流式响应
                chunks = []
                for chunk in response:
                    if (
                        not chunk.candidates
                        or not chunk.candidates[0].content
                        or not chunk.candidates[0].content.parts
                    ):
                        continue
                    if chunk.text:
                        chunks.append(chunk.text)
                        console.print("")
                        try:
                            console.print(chunk.text, end="", overflow="ellipsis")
                        except Exception as e:
                            console.print(Text(chunk.text), end="", overflow="ellipsis")
                        finally:
                            console.file.flush()
                    if chunk.candidates[0].content.parts[0].inline_data:
                        save_binary_file(
                            Path(uri).with_stem(Path(uri).stem + "_s"),
                            chunk.candidates[0].content.parts[0].inline_data.data,
                        )
                        console.print(
                            f"[blue]File of mime type"
                            f" {chunk.candidates[0].content.parts[0].inline_data.mime_type} saved"
                            f"to: {Path(uri).with_stem(Path(uri).stem + '_s').name}[/blue]"
                        )

                console.print("\n")
                response_text = "".join(chunks)
                if mime.startswith("image"):
                    response_text = response_text.replace("*", "").strip()

                elapsed_time = time.time() - start_time
                console.print(
                    f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
                )

                try:
                    console.print(response_text)
                except Exception as e:
                    console.print(Text(response_text))

                if mime.startswith("image"):
                    if isinstance(response_text, str):
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
                        # If it's already a dict/list, use it directly
                        captions = response_text
                    if args.pair_dir and pair_pixels:
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

                response_text = response_text.replace(
                    "[green]", "<font color='green'>"
                ).replace("[/green]", "</font>")

                # Extract SRT content between first and second ```
                content = extract_code_block_content(response_text, "srt", console)
                if not content:
                    continue

                if progress and task_id is not None:
                    progress.update(task_id, description="Processing media...")
                return content
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                console.print(f"[red]Error processing:[/red]")
                console.print(Text(f"{str(e)}", style="red"))
                console.print(Text(f"Error trace: {error_trace}", style="red"))
                if attempt < args.max_retries - 1:
                    console.print(
                        f"[yellow]Retrying in {args.wait_time} seconds...[/yellow]"
                    )
                    elapsed_time = time.time() - start_time
                    if elapsed_time < args.wait_time:
                        time.sleep(args.wait_time - elapsed_time)
                else:
                    console.print(
                        f"[red]Failed to process after {args.max_retries} attempts. Skipping.[/red]"
                    )
                continue
        return ""


def upload_to_gemini(client, path, mime_type=None, name=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    original_name = Path(path).name
    safe_name = sanitize_filename(original_name if name is None else name)

    file = client.files.upload(
        file=path,
        config=types.UploadFileConfig(
            name=safe_name,
            mime_type=mime_type,
            display_name=original_name,
        ),
    )
    console.print()
    console.print(f"[blue]Uploaded file[/blue] '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(client, files, output_console=None):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    Args:
        client: The Gemini client
        files: List of files to wait for
        output_console: Optional console to use for output. If None, uses global console.
    """
    # Use provided console or fall back to global
    output_console = output_console or console

    output_console.print("[yellow]Waiting for file processing...[/yellow]")
    for name in (file.name for file in files):
        file = client.files.get(name=name)
        while file.state.name != "ACTIVE":
            if file.state.name == "FAILED":
                raise Exception(f"File {file.name} failed to process")
            output_console.print(".", end="", style="yellow")
            time.sleep(10)
            file = client.files.get(name=name)
        output_console.print()  # New line after dots
    output_console.print("[green]...all files ready[/green]")
    output_console.print()


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
