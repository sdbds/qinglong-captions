import time
from typing import List, Optional, Dict, Any, Union, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from rich_pixels import Pixels
from rich.console import Console
from rich.text import Text
from pathlib import Path

console = Console()


def api_process_batch(
    uri: str,
    mime: str,
    config,
    api_key: str,
    wait_time: int = 1,
    max_retries: int = 10,
    model_path: Optional[str] = "gemini-exp-1206",
) -> str:
    if model_path.startswith("gemini"):
        genai.configure(api_key=api_key)

        generation_config = config["generation_config"]

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

        model = genai.GenerativeModel(
            model_name=model_path,
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

            for upload_attempt in range(max_retries):
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
                        f"[yellow]Upload attempt {upload_attempt + 1}/{max_retries} failed: {e}[/yellow]"
                    )
                    if upload_attempt < max_retries - 1:
                        time.sleep(wait_time * 2)  # Increase wait time between retries
                    else:
                        console.print("[red]All upload attempts failed[/red]")
                        return ""

            if not upload_success:
                console.print("[red]Failed to upload file[/red]")
                return ""

        # Some files have a processing delay. Wait for them to be ready.
        # wait_for_files_active(files)
        for attempt in range(max_retries):
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
                else:
                    content = ""

                return content
            except Exception as e:
                console.print(f"[red]Error processing: {e}[/red]")
                if attempt < max_retries - 1:
                    console.print(
                        f"[yellow]Retrying in {wait_time} seconds...[/yellow]"
                    )
                    elapsed_time = time.time() - start_time
                    if elapsed_time < wait_time:
                        time.sleep(wait_time - elapsed_time)
                else:
                    console.print(
                        f"[red]Failed to process after {max_retries} attempts. Skipping.[/red]"
                    )
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
