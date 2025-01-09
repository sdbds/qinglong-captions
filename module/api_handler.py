import time
import io
import base64
from typing import List, Optional, Dict, Any, Union, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from mistralai import Mistral
from rich_pixels import Pixels
from rich.progress import Progress
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from PIL import Image
from pathlib import Path

console = Console()


def api_process_batch(
    uri: str,
    mime: str,
    config,
    args,
) -> str:

    system_prompt = config["prompts"]["system_prompt"]
    prompt = config["prompts"]["prompt"]

    if (
        mime.startswith("video")
        or mime.startswith("audio")
        or args.pixtral_api_key == ""
    ):

        generation_config = config["geimini_generation_config"]

        genai.configure(api_key=args.gemini_api_key)

        if mime.startswith("video"):
            system_prompt = config["prompts"]["video_system_prompt"]
            prompt = config["prompts"]["video_prompt"]
        elif mime.startswith("audio"):
            system_prompt = config["prompts"]["audio_system_prompt"]
            prompt = config["prompts"]["audio_prompt"]
        elif mime.startswith("image"):
            system_prompt = config["prompts"]["image_system_prompt"]
            prompt = config["prompts"]["image_prompt"]

        model = genai.GenerativeModel(
            model_name=args.gemini_model_path,
            generation_config=generation_config,
            system_instruction=system_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
            },
        )

        # # Check existing files
        # target_name = Path(uri).name
        # console.print(f"[blue]Checking files for:[/blue] {target_name}")

        # existing_files = list(genai.list_files())

        # def get_url_and_filename(file_id: str) -> tuple:
        #     """获取完整 URL 和原始文件名

        #     Args:
        #         file_id: 文件ID (files/xxx 格式)

        #     Returns:
        #         (完整URL, 原始文件名)
        #     """
        #     base_url = "https://generativelanguage.googleapis.com/v1beta"
        #     full_url = f"{base_url}/{file_id}"
        #     # 从 URL 中提取文件名
        #     try:
        #         response = requests.head(full_url)
        #         filename = response.headers.get('content-disposition', '').split('filename=')[-1]
        #         if not filename:
        #             # 如果无法从 header 获取，使用 URL 最后一部分
        #             filename = Path(full_url).name
        #     except:
        #         filename = Path(full_url).name
        #     return full_url, filename

        # # 显示所有文件的信息
        # if existing_files:
        #     console.print("[blue]Files in storage:[/blue]")
        #     for f in existing_files:
        #         full_url, orig_name = get_url_and_filename(f.name)
        #         console.print(f"  API ID: {f.name}")
        #         console.print(f"  Original Name: {orig_name}")
        #         console.print(f"  URL: {full_url}")
        #         console.print("---")

        # # 查找匹配的文件
        # existing_file = next(
        #     (f for f in existing_files
        #     if target_name == get_url_and_filename(f.name)[1]),
        #     None
        # )

        # if existing_file:
        #     full_url, orig_name = get_url_and_filename(existing_file.name)
        #     console.print(f"[green]Found existing file:[/green] {orig_name}")
        #     console.print(f"[green]API ID:[/green] {existing_file.name}")
        #     console.print(f"[green]Full URL:[/green] {full_url}")
        if (
            mime.startswith("video")
            or mime.startswith("audio")
            and Path(uri).stat().st_size >= 20 * 1024 * 1024
        ):
            upload_success = False
            files = []

            for upload_attempt in range(args.max_retries):
                try:
                    # if existing_file:
                    #     files = [existing_file]
                    #     upload_success = True
                    #     break
                    # else:
                    console.print()
                    console.print(f"[blue]uploading files for:[/blue] {uri}")
                    files = [
                        upload_to_gemini(path=uri, mime_type=mime),
                    ]
                    wait_for_files_active(files)
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

                # 使用 chat 模式
                chat = model.start_chat(
                    history=[],
                )

                if mime.startswith("video") or (
                    mime.startswith("audio")
                    and Path(uri).stat().st_size >= 20 * 1024 * 1024
                ):
                    response = chat.send_message([files[0], prompt], stream=True)
                elif mime.startswith("audio"):
                    audio_blob = Path(uri).read_bytes()
                    response = chat.send_message(
                        [
                            prompt,
                            {
                                "mime_type": mime,
                                "data": audio_blob,
                            },
                        ],
                        stream=True,
                    )
                else:
                    blob, pixels = encode_image(uri)
                    response = chat.send_message(
                        [
                            {
                                "mime_type": "image/jpeg",
                                "data": blob,
                            },
                            prompt,
                        ],
                        stream=True,
                    )

                # 收集流式响应
                chunks = []
                for chunk in response:
                    if chunk.text:
                        chunks.append(chunk.text)
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
                console.print(display_text)

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

                return content
            except Exception as e:
                console.print(f"[red]Error processing: {e}[/red]")
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

    elif mime.startswith("image") and args.pixtral_api_key != "":

        system_prompt = config["prompts"]["pixtral_image_system_prompt"]
        prompt = config["prompts"]["pixtral_image_prompt"]

        base64_image, pixels = encode_image(uri)
        if base64_image is None or pixels is None:
            return ""

        client = Mistral(api_key=args.pixtral_api_key)
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

        for attempt in range(args.max_retries):
            try:
                start_time = time.time()
                chat_response = client.chat.complete(
                    model=args.pixtral_model_path, messages=messages
                )
                content = chat_response.choices[0].message.content

                if "502" in content:
                    console.print(
                        f"[yellow]Attempt {attempt + 1}/{args.max_retries}: Received 502 error[/yellow]"
                    )
                    continue

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

                console.print()
                console.print()
                # 获取图片实际高度
                panel_height = 32  # 加上面板的边框高度

                # 创建布局
                layout = Layout()

                # 创建右侧的垂直布局
                right_layout = Layout()

                # 创建上半部分的水平布局（tag和short并排）
                top_layout = Layout()
                top_layout.split_row(
                    Layout(
                        Panel(
                            Text(tag_description, style="magenta"),
                            title="tags",
                            height=panel_height // 2,
                            padding=0,
                            expand=True,
                        ),
                        ratio=1,
                    ),
                    Layout(
                        Panel(
                            short_description,
                            title=f"short_description - [yellow]highlight rate:[/yellow] {short_highlight_rate}",
                            height=panel_height // 2,
                            padding=0,
                            expand=True,
                        ),
                        ratio=1,
                    ),
                )

                # 将右侧布局分为上下两部分
                right_layout.split_column(
                    Layout(top_layout, ratio=1),
                    Layout(
                        Panel(
                            long_description,
                            title=f"long_description - [yellow]highlight rate:[/yellow] {long_highlight_rate}",
                            height=panel_height // 2,
                            padding=0,
                            expand=True,
                        )
                    ),
                )

                # 主布局分为左右两部分
                layout.split_row(
                    Layout(
                        Panel(pixels, height=panel_height, padding=0, expand=True),
                        name="image",
                        ratio=1,
                    ),
                    Layout(right_layout, name="caption", ratio=2),
                )

                # 将整个布局放在一个高度受控的面板中
                console.print(
                    Panel(
                        layout,
                        title=Path(uri).name,
                        height=panel_height + 2,
                        padding=0,
                    )
                )
                del pixels

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
                        with Progress() as progress:
                            task = progress.add_task(
                                "[magenta]Waiting...", total=wait_time
                            )
                            for _ in range(wait_time):
                                time.sleep(1)
                                progress.update(task, advance=1)
                        console.print("[green]Retrying...[/green]")
                    time.sleep(wait_time)
                    continue
        return ""


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    console.print()
    console.print(f"[blue]Uploaded file[/blue] '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(files):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    console.print("[yellow]Waiting for file processing...[/yellow]")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        with console.status("[yellow]Processing...[/yellow]", spinner="dots") as status:
            while file.state.name == "PROCESSING":
                time.sleep(10)
                file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
    console.print("[green]...all files ready[/green]")
    console.print()


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
        short_description = ' '.join(short_description.split(":", 1)[-1].split())
        long_description = ' '.join(long_description.split(":", 1)[-1].split())
    else:
        short_description = ""
        long_description = ""
    
    return short_description, long_description