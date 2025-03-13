import time
import io
import base64
from typing import List, Optional, Dict, Any, Union, Tuple
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
from utils.console_util import CaptionLayout, MarkdownLayout
from utils.stream_util import sanitize_filename

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

    if args.step_api_key != "" and mime.startswith("video"):

        system_prompt = config["prompts"]["step_video_system_prompt"]
        prompt = config["prompts"]["step_video_prompt"]

        client = OpenAI(
            api_key=args.step_api_key, base_url="https://api.stepfun.com/v1"
        )

        file = client.files.create(file=open(uri, "rb"), purpose="storage")

        console.print(f"[blue]Uploaded video file:[/blue] {file}")

        for attempt in range(args.max_retries):
            try:
                start_time = time.time()
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

                # Convert HTML font tags to rich format
                display_text = response_text.replace(
                    '<font color="green">', "[green]"
                ).replace("</font>", "[/green]")
                try:
                    console.print(display_text)
                except Exception as e:
                    console.print(Text(display_text))

                # Extract SRT content between first and second ```
                text = response_text
                if text:
                    # 查找所有的 ``` 标记位置
                    markers = []
                    start = 0
                    while True:
                        pos = text.find("```", start)
                        if pos == -1:
                            break
                        markers.append(pos)
                        start = pos + 3

                    # 确保找到至少一对标记
                    if len(markers) >= 2:
                        # 获取最后一对标记之间的内容
                        first_marker = markers[-2]
                        second_marker = markers[-1]
                        content = text[first_marker + 3 : second_marker]

                        # Remove "srt" if present at the start
                        if content.startswith("srt"):
                            content = content[3:]

                        console.print(
                            f"[blue]Extracted SRT content length:[/blue] {len(content)}"
                        )
                        console.print(f"[blue]Found {len(markers)} ``` markers[/blue]")
                    else:
                        console.print(
                            f"[red]Not enough ``` markers: found {len(markers)}[/red]"
                        )
                        content = ""
                        continue
                else:
                    content = ""
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

                # Convert HTML font tags to rich format
                display_text = response_text.replace(
                    '<font color="green">', "[green]"
                ).replace("</font>", "[/green]")
                try:
                    console.print(display_text)
                except Exception as e:
                    console.print(Text(display_text))

                # Extract SRT content between first and second ```
                text = response_text
                if text:
                    # 查找所有的 ``` 标记位置
                    markers = []
                    start = 0
                    while True:
                        pos = text.find("```", start)
                        if pos == -1:
                            break
                        markers.append(pos)
                        start = pos + 3

                    # 确保找到至少一对标记
                    if len(markers) >= 2:
                        # 获取最后一对标记之间的内容
                        first_marker = markers[-2]
                        second_marker = markers[-1]
                        content = text[first_marker + 3 : second_marker]

                        # Remove "srt" if present at the start
                        if content.startswith("srt"):
                            content = content[3:]

                        console.print(
                            f"[blue]Extracted SRT content length:[/blue] {len(content)}"
                        )
                        console.print(f"[blue]Found {len(markers)} ``` markers[/blue]")
                    else:
                        console.print(
                            f"[red]Not enough ``` markers: found {len(markers)}[/red]"
                        )
                        content = ""
                        continue
                else:
                    content = ""
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
        start_time = time.time()

        if mime.startswith("image"):
            system_prompt = config["prompts"]["pixtral_image_system_prompt"]
            prompt = config["prompts"]["pixtral_image_prompt"]

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
                        if ocr_pixels:
                            del ocr_pixels

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
                    del pixels

                else:
                    chat_response = client.chat.complete(
                        model=args.pixtral_model_path, messages=messages
                    )
                    content = chat_response.choices[0].message.content

                    # if character_name:
                    #     clean_char_name = (
                    #         character_name.split(",")[0].split(" from ")[0].strip("<>")
                    #     )
                    #     if clean_char_name not in content:
                    #         console.print()
                    #         console.print(Text(content))
                    #         console.print(
                    #             f"Attempt {attempt + 1}/{args.max_retries}: Character name [green]{clean_char_name}[/green] not found"
                    #         )
                    #         continue

                    if "###" not in content:
                        console.print(Text(content))
                        console.print(Text("No ###, retrying...", style="yellow"))
                        continue

                    short_description, long_description = process_llm_response(content)
                    tag_description = ""
                    # tag_description = (
                    #     prompt.rsplit("<s>[INST]", 1)[-1]
                    #     .rsplit(">.", 1)[-1]
                    #     .rsplit(").", 1)[-1]
                    #     .replace(" from", ",")
                    # )
                    # tag_description = tag_description.rsplit("[IMG][/INST]", 1)[0].strip()
                    short_highlight_rate = 0
                    long_highlight_rate = 0
                    # short_description, short_highlight_rate = format_description(
                    #     short_description, tag_description
                    # )
                    # long_description, long_highlight_rate = format_description(
                    #     long_description, tag_description
                    # )

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
                    del pixels

                if "502" in content:
                    console.print(
                        f"[yellow]Attempt {attempt + 1}/{args.max_retries}: Received 502 error[/yellow]"
                    )
                    continue

                # 计算已经消耗的时间，动态调整等待时间
                elapsed_time = time.time() - start_time
                if elapsed_time < args.wait_time:
                    time.sleep(args.wait_time - elapsed_time)

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
                        with Progress() as progress2:
                            task = progress2.add_task(
                                "[magenta]Waiting...", total=wait_time
                            )
                            for _ in range(wait_time):
                                time.sleep(1)
                                progress2.update(task, advance=1)
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

        genai_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            top_k=generation_config["top_k"],
            candidate_count=config["generation_config"]["candidate_count"],
            max_output_tokens=generation_config["max_output_tokens"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            # tools=(
            #     [types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]
            #     if mime.startswith("image") or mime.startswith("audio")
            #     else None
            # ),
            # tool_config=(
            #     types.ToolConfig(
            #         function_calling_config=types.FunctionCallingConfig(
            #             mode="AUTO", allowed_function_names=["google_search_retrieval"]
            #         )
            #     )
            #     if mime.startswith("image") or mime.startswith("audio")
            #     else None
            # ),
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
            response_mime_type=generation_config["response_mime_type"],
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
                else:
                    blob, pixels = encode_image(uri)
                    response = client.models.generate_content_stream(
                        model=args.gemini_model_path,
                        contents=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=blob, mime_type="image/jpeg"),
                        ],
                        config=genai_config,
                    )

                if progress and task_id is not None:
                    progress.update(task_id, description="Generating captions")
                # 收集流式响应
                chunks = []
                for chunk in response:
                    if chunk.text:
                        chunks.append(chunk.text)
                        console.print("")
                        try:
                            console.print(chunk.text, end="", overflow="ellipsis")
                        except Exception as e:
                            console.print(Text(chunk.text), end="", overflow="ellipsis")
                        finally:
                            console.file.flush()

                console.print("\n")
                response_text = "".join(chunks)

                elapsed_time = time.time() - start_time
                console.print(
                    f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
                )

                # Convert HTML font tags to rich format
                display_text = response_text.replace(
                    '<font color="green">', "[green]"
                ).replace("</font>", "[/green]")
                try:
                    console.print(display_text)
                except Exception as e:
                    console.print(Text(display_text))

                # Extract SRT content between first and second ```
                text = response_text
                if text:
                    # 查找所有的 ``` 标记位置
                    markers = []
                    start = 0
                    while True:
                        pos = text.find("```", start)
                        if pos == -1:
                            break
                        markers.append(pos)
                        start = pos + 3

                    # 确保找到至少一对标记
                    if len(markers) >= 2:
                        # 获取最后一对标记之间的内容
                        first_marker = markers[-2]
                        second_marker = markers[-1]
                        content = text[first_marker + 3 : second_marker]

                        # Remove "srt" if present at the start
                        if content.startswith("srt"):
                            content = content[3:]

                        console.print(
                            f"[blue]Extracted SRT content length:[/blue] {len(content)}"
                        )
                        console.print(f"[blue]Found {len(markers)} ``` markers[/blue]")
                    else:
                        console.print(
                            f"[red]Not enough ``` markers: found {len(markers)}[/red]"
                        )
                        content = ""
                        continue
                else:
                    content = ""

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
            aspect_ratio = width / height

            def calculate_dimensions(max_size: int) -> Tuple[int, int]:
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

            new_width, new_height = calculate_dimensions(max_size)
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
