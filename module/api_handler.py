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
from utils.console_util import CaptionLayout, MarkdownLayout, CaptionAndRateLayout
from utils.stream_util import sanitize_filename
from utils.wdtagger import format_description, split_name_series

console = Console()


def _handle_api_call_with_retries(
    api_call_func,
    max_retries: int,
    wait_time_seconds: int,
    console: Console,
    api_name: str,
    progress: Optional[Progress] = None,
    task_id: Optional[Any] = None,
    on_success_description: Optional[str] = None,
) -> Any:
    """Handles API calls with retry logic."""
    for attempt in range(max_retries):
        start_time = time.time()
        try:
            result = api_call_func()
            if progress and task_id is not None and on_success_description:
                progress.update(task_id, description=on_success_description)
            return result
        except Exception as e:
            error_msg = Text(str(e), style="red")
            console.print(f"[red]Error processing {api_name}: {error_msg}[/red]")
            if attempt < max_retries - 1:
                elapsed_time = time.time() - start_time
                actual_wait_time = wait_time_seconds
                if elapsed_time < wait_time_seconds:
                    actual_wait_time = wait_time_seconds - elapsed_time

                if "429" in str(e):
                    actual_wait_time = 60  # Longer wait for 429 errors
                    console.print(
                        f"[yellow]Rate limit error for {api_name}. Waiting {actual_wait_time} seconds before retrying...[/yellow]"
                    )
                else:
                    console.print(
                        f"[yellow]Retrying {api_name} in {actual_wait_time:.2f} seconds...[/yellow]"
                    )
                
                time.sleep(actual_wait_time)
                continue
            else:
                console.print(
                    f"[red]Failed to process {api_name} after {max_retries} attempts. Skipping.[/red]"
                )
    return "" # Return empty string on failure, consistent with existing API blocks


def _collect_streaming_response(
    stream: Any,  # Iterable streaming response
    extract_content_callback: callable,
    console: Console,
    progress: Optional[Progress] = None,
    task_id: Optional[Any] = None,
    on_chunk_arrival_description: Optional[str] = "Receiving stream...",
    accumulate_as_string: bool = False,
) -> str:
    """Collects and processes a streaming response from an API."""
    if progress and task_id and on_chunk_arrival_description:
        progress.update(task_id, description=on_chunk_arrival_description)

    collected_chunks: Union[List[str], str] = "" if accumulate_as_string else []

    for chunk in stream:
        content_piece = extract_content_callback(chunk)
        if content_piece:
            if accumulate_as_string:
                collected_chunks += content_piece
            else:
                collected_chunks.append(content_piece)
            console.print(".", end="", style="blue")

    console.print("\n")  # Move past the progress dots

    if not accumulate_as_string:
        response_text = "".join(collected_chunks)
    else:
        response_text = collected_chunks
    
    return response_text


def _post_process_srt_response(
    response_text: str,
    console_instance: Console,
    progress_instance: Optional[Progress] = None,
    task_id: Optional[Any] = None,
    final_progress_description: str = "Processing media...",
) -> str:
    """Handles common post-processing for SRT responses."""
    if not response_text:
        return ""

    processed_text = response_text.replace("[green]", "<font color='green'>").replace(
        "[/green]", "</font>"
    )
    content = extract_code_block_content(processed_text, "srt", console_instance)

    if not content:
        console_instance.print("[yellow]Failed to extract SRT content from response.[/yellow]")
        return ""

    if progress_instance and task_id is not None:
        progress_instance.update(task_id, description=final_progress_description)
    
    return content


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

        # File upload remains before the retry handler
        try:
            file = client.files.create(file=open(uri, "rb"), purpose="storage")
            console.print(f"[blue]Uploaded video file for StepConnect:[/blue] {file}")
        except Exception as e:
            console.print(f"[red]StepConnect file upload failed: {e}[/red]")
            return ""

        # Prompts specific to StepConnect
        system_prompt_step = config["prompts"]["step_video_system_prompt"]
        prompt_step = config["prompts"]["step_video_prompt"]

        messages_step = [
            {
                "role": "system",
                "content": system_prompt_step,
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
                        "text": prompt_step,
                    },
                ],
            },
        ]

        def _make_step_api_call():
            completion = client.chat.completions.create(
                model=args.step_model_path,
                messages=messages_step,
                temperature=0.7,
                top_p=0.95,
                max_tokens=8192,
                stream=True,
            )
            
            extract_step_content_callback = lambda chunk: chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None else None
            
            response_text = _collect_streaming_response(
                completion, 
                extract_step_content_callback, 
                console, 
                progress, 
                task_id, 
                "Collecting StepConnect stream...",
                accumulate_as_string=True 
            )
            # Log generation time immediately after successful collection
            # Note: This time is for the successful API call and stream collection, not the entire block with retries.
            # To get a more precise time for just the API call itself, it would need to be timed inside _collect_streaming_response
            # or this _make_step_api_call function. For now, this is a reasonable placement.
            # We need a start_time for this specific call if we want to log it.
            # Let's add a start_time capture at the beginning of _make_step_api_call.
            # However, the original elapsed_time was for the whole block including retries.
            # The new _handle_api_call_with_retries already logs time for retries.
            # Let's omit the specific "Caption generation took" log here to avoid confusion,
            # as the retry handler provides its own timing details for attempts.
            return response_text

        response_text = _handle_api_call_with_retries(
            api_call_func=_make_step_api_call,
            max_retries=args.max_retries,
            wait_time_seconds=args.wait_time,
            console=console,
            api_name="StepConnect",
            progress=progress,
            task_id=task_id,
            on_success_description="StepConnect response received.",
        )

        if not response_text:
            return ""

        try:
            console.print(response_text)
        except Exception as e:
            console.print(Text(response_text)) # Fallback for potential rich formatting issues

        content = _post_process_srt_response(
            response_text,
            console, # console instance from api_process_batch
            progress,
            task_id,
            "Processing StepConnect media..." 
        )
        if not content:
            return ""
        return content

    elif args.qwenVL_api_key != "" and mime.startswith("video"):
        import dashscope # Ensured import

        system_prompt_qwen = config["prompts"]["qwenvl_video_system_prompt"]
        prompt_qwen = config["prompts"]["qwenvl_video_prompt"]

        file_qwen = f"file://{Path(uri).resolve().as_posix()}"
        console.print(f"[blue]Preparing video file for QwenVL:[/blue] {file_qwen}")

        messages_qwen = [
            {
                "role": "system",
                "content": [
                    {"text": system_prompt_qwen},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "video": file_qwen,
                    },
                    {"text": prompt_qwen},
                ],
            },
        ]

        def _make_qwen_api_call():
            responses_stream = dashscope.MultiModalConversation.call(
                model=args.qwenVL_model_path,
                messages=messages_qwen,
                api_key=args.qwenVL_api_key,
                stream=True,
                incremental_output=True, # Qwen specific: important for the callback
            )
            
            extract_qwen_content_callback = lambda chunk: chunk.output.choices[0].message.content[0]["text"] if chunk.output and chunk.output.choices and chunk.output.choices[0].message and chunk.output.choices[0].message.content and isinstance(chunk.output.choices[0].message.content, list) and len(chunk.output.choices[0].message.content) > 0 and chunk.output.choices[0].message.content[0].get("text") else None
            
            response_text = _collect_streaming_response(
                responses_stream,
                extract_qwen_content_callback,
                console,
                progress,
                task_id,
                "Collecting QwenVL stream...",
                accumulate_as_string=True # Qwen returns parts of the full string
            )
            return response_text

        response_text = _handle_api_call_with_retries(
            api_call_func=_make_qwen_api_call,
            max_retries=args.max_retries,
            wait_time_seconds=args.wait_time,
            console=console,
            api_name="QwenVL",
            progress=progress,
            task_id=task_id,
            on_success_description="QwenVL response received.",
        )

        if not response_text:
            return ""

        try:
            console.print(response_text)
        except Exception as e:
            console.print(Text(response_text))

        content = _post_process_srt_response(
            response_text,
            console, # console instance from api_process_batch
            progress,
            task_id,
            "Processing QwenVL media..."
        )
        if not content:
            return ""
        return content

    elif args.glm_api_key != "" and mime.startswith("video"):
        from zhipuai import ZhipuAI # Import ensured

        client = ZhipuAI(api_key=args.glm_api_key)

        system_prompt_glm = config["prompts"]["glm_video_system_prompt"]
        prompt_glm = config["prompts"]["glm_video_prompt"]

        try:
            with open(uri, "rb") as video_file:
                video_base64_glm = base64.b64encode(video_file.read()).decode("utf-8")
        except Exception as e:
            console.print(f"[red]GLM: Failed to read and encode video file {uri}: {e}[/red]")
            return ""

        messages_glm = [
            {
                "role": "system",
                "content": system_prompt_glm,
            },
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_base64_glm}},
                    {"type": "text", "text": prompt_glm},
                ],
            },
        ]

        def _make_glm_api_call():
            responses_stream = client.chat.completions.create(
                model=args.glm_model_path,
                messages=messages_glm,
                stream=True,
            )
            
            extract_glm_content_callback = lambda chunk: chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None else None
            
            response_text = _collect_streaming_response(
                responses_stream,
                extract_glm_content_callback,
                console,
                progress,
                task_id,
                "Collecting GLM stream...",
                accumulate_as_string=True 
            )
            return response_text

        response_text = _handle_api_call_with_retries(
            api_call_func=_make_glm_api_call,
            max_retries=args.max_retries,
            wait_time_seconds=args.wait_time,
            console=console,
            api_name="GLM",
            progress=progress,
            task_id=task_id,
            on_success_description="GLM response received.",
        )

        if not response_text:
            return ""

        try:
            console.print(response_text)
        except Exception as e:
            console.print(Text(response_text))

        content = _post_process_srt_response(
            response_text,
            console, # console instance from api_process_batch
            progress,
            task_id,
            "Processing GLM media..."
        )
        if not content:
            return ""
        return content

    elif args.pixtral_api_key != "" and (
        mime.startswith("image") or mime.startswith("application")
    ):
        client = Mistral(api_key=args.pixtral_api_key)
        
        # Variables that might be needed across different paths
        base64_image = None
        pixels = None
        signed_url_actual = None
        messages_pixtral = []
        captions_from_file = [] # Renamed from captions to avoid conflict
        character_name_pixtral = ""
        prompt_pixtral_image_final = ""


        if mime.startswith("application"):
            def _upload_pixtral_pdf_core(client_instance, pdf_uri):
                uploaded_pdf = client_instance.files.upload(
                    file={
                        "file_name": f"{sanitize_filename(pdf_uri)}.pdf",
                        "content": open(pdf_uri, "rb"),
                    },
                    purpose="ocr",
                )
                return client_instance.files.get_signed_url(file_id=uploaded_pdf.id)

            signed_url_obj = _handle_api_call_with_retries(
                api_call_func=lambda: _upload_pixtral_pdf_core(client, uri),
                max_retries=args.max_retries,
                wait_time_seconds=args.wait_time,
                console=console,
                api_name="PixtralPDFUpload",
                progress=progress,
                task_id=task_id,
                on_success_description="Pixtral PDF uploaded."
            )
            if not signed_url_obj:
                console.print("[red]Pixtral: PDF upload failed after retries. Skipping.[/red]")
                return ""
            signed_url_actual = signed_url_obj.url
            console.print(f"[blue]Pixtral: PDF uploaded, signed URL obtained: {signed_url_actual[:50]}...[/blue]")

        elif mime.startswith("image"):
            system_prompt_pixtral = config["prompts"]["pixtral_image_system_prompt"]
            character_prompt_pixtral = ""
            if args.dir_name:
                dir_prompt_pixtral = Path(uri).parent.name or ""
                character_name_pixtral = split_name_series(dir_prompt_pixtral)
                character_prompt_pixtral = f"If there is a person/character or more in the image you must refer to them as {character_name_pixtral}.\n"
                character_name_pixtral = f"{character_name_pixtral}, " if character_name_pixtral else ""
            
            config_prompt_pixtral_image = config["prompts"]["pixtral_image_prompt"]
            captions_path = Path(uri).with_suffix(".txt")
            if captions_path.exists():
                with open(captions_path, "r", encoding="utf-8") as f:
                    captions_from_file = [line.strip() for line in f.readlines()]

            prompt_pixtral_image_final = Text(
                f"<s>[INST]{character_prompt_pixtral}{character_name_pixtral}{captions_from_file[0] if len(captions_from_file) > 0 else config_prompt_pixtral_image}\n[IMG][/INST]"
            ).plain

            base64_image, pixels = encode_image(uri)
            if base64_image is None or pixels is None:
                console.print(f"[red]Pixtral: Failed to encode image {uri}. Skipping.[/red]")
                return ""

            messages_pixtral = [
                {"role": "system", "content": system_prompt_pixtral},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_pixtral_image_final},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                },
            ]
        elif args.ocr: # This implies it's an image if not application, but we need base64_image
             base64_image, pixels = encode_image(uri)
             if base64_image is None or pixels is None:
                console.print(f"[red]Pixtral OCR: Failed to encode image {uri}. Skipping.[/red]")
                return ""


        def _make_pixtral_api_call():
            if mime.startswith("application"):
                # PDF OCR
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "document_url",
                        "document_url": signed_url_actual, # Use the obtained signed URL
                    },
                    include_image_base64=args.document_image,
                )
                return ocr_response.pages # Returns list of page objects
            elif args.ocr: # Image OCR
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                )
                return ocr_response.pages[0].markdown # Returns markdown string
            elif mime.startswith("image"): # Image Captioning
                chat_response = client.chat.complete(
                    model=args.pixtral_model_path, messages=messages_pixtral
                )
                return chat_response.choices[0].message.content # Returns string
            return None # Should not happen if logic is correct

        content_or_pages = _handle_api_call_with_retries(
            api_call_func=_make_pixtral_api_call,
            max_retries=args.max_retries,
            wait_time_seconds=args.wait_time,
            console=console,
            api_name="Pixtral",
            progress=progress,
            task_id=task_id,
            on_success_description="Pixtral response received."
        )

        if not content_or_pages:
            console.print("[red]Pixtral: API call failed after retries or no content returned. Skipping.[/red]")
            return ""

        # Post-processing based on the type of content
        if mime.startswith("application"):
            content_pages = content_or_pages
            console.print(f"[bold cyan]Pixtral PDF OCR: Received {len(content_pages)} pages.[/bold cyan]")
            # The original loop for displaying MarkdownLayout for each page
            for page_idx, page_content in enumerate(content_pages):
                page_ocr_pixels = None
                if page_content.images and len(page_content.images) > 0:
                    first_image = page_content.images[0]
                    if hasattr(first_image, "image_base64") and first_image.image_base64:
                        try:
                            base64_str = first_image.image_base64
                            base64_content = base64_str.split(",",1)[1] if base64_str.startswith("data:") else base64_str
                            image_data = base64.b64decode(base64_content)
                            ocr_image_pil = Image.open(io.BytesIO(image_data))
                            page_ocr_pixels = Pixels.from_image(
                                ocr_image_pil,
                                resize=(ocr_image_pil.width // 18, ocr_image_pil.height // 18)
                            )
                        except Exception as e:
                            console.print(f"[yellow]Pixtral PDF: Error loading image from page {page_idx}: {e}[/yellow]")
                
                markdown_layout = MarkdownLayout(
                    pixels=page_ocr_pixels,
                    markdown_content=page_content.markdown,
                    panel_height=32,
                    console=console,
                )
                markdown_layout.print(title=f"{Path(uri).name} - Page {page_content.index+1}")
                if page_ocr_pixels: del page_ocr_pixels
            return content_pages # Return the list of page objects for captioner.py

        elif args.ocr: # Image OCR
            content_markdown = content_or_pages
            markdown_layout = MarkdownLayout(
                pixels=pixels, # pixels from the initial encode_image
                markdown_content=content_markdown,
                panel_height=32,
                console=console,
            )
            markdown_layout.print(title=Path(uri).name)
            if pixels: del pixels
            return content_markdown

        elif mime.startswith("image"): # Image Captioning
            content_string = content_or_pages
            short_description, long_description = process_llm_response(content_string)
            tag_description_pixtral = ""
            short_highlight_rate_pixtral = 0
            long_highlight_rate_pixtral = 0

            if len(captions_from_file) > 0:
                tag_description_pixtral = (
                    (
                        prompt_pixtral_image_final.rsplit("<s>[INST]", 1)[-1]
                        .rsplit(">.", 1)[-1]
                        .rsplit(").", 1)[-1]
                        .replace(" from", ",")
                    )
                    .rsplit("[IMG][/INST]", 1)[0]
                    .strip()
                )
                short_description, short_highlight_rate_pixtral = format_description(
                    short_description, tag_description_pixtral
                )
                long_description, long_highlight_rate_pixtral = format_description(
                    long_description, tag_description_pixtral
                )

            caption_layout = CaptionLayout(
                tag_description=tag_description_pixtral,
                short_description=short_description,
                long_description=long_description,
                pixels=pixels, # pixels from initial encode_image
                short_highlight_rate=short_highlight_rate_pixtral,
                long_highlight_rate=long_highlight_rate_pixtral,
                panel_height=32,
                console=console,
            )
            caption_layout.print(title=Path(uri).name)
            if pixels: del pixels

            # Content validation checks (outside retry loop now)
            if character_name_pixtral: # Check using the prepared character_name_pixtral
                clean_char_name = character_name_pixtral.split(",")[0].split(" from ")[0].strip("<>")
                if clean_char_name not in content_string:
                    console.print(Text(content_string))
                    console.print(f"[yellow]Pixtral: Character name [green]{clean_char_name}[/green] not found in response. Skipping.[/yellow]")
                    return ""
            
            if "###" not in content_string:
                console.print(Text(content_string))
                console.print(Text("[yellow]Pixtral: No '###' in response. Skipping.[/yellow]", style="yellow"))
                return ""

            if len(captions_from_file) > 0: # Only apply rate check if original captions were present
                current_tags_highlightrate = args.tags_highlightrate * 100
                if (
                    any(f"{i}women" in tag_description_pixtral for i in range(2, 5))
                    or ("1man" in tag_description_pixtral and "1woman" in tag_description_pixtral)
                    or "multiple girls" in tag_description_pixtral
                    or "multiple boys" in tag_description_pixtral
                ):
                    current_tags_highlightrate = args.tags_highlightrate * 100 / 2
                
                # Ensure long_highlight_rate_pixtral is a number for comparison
                long_hl_rate_numeric = 0
                if isinstance(long_highlight_rate_pixtral, (int, float)):
                    long_hl_rate_numeric = long_highlight_rate_pixtral
                elif isinstance(long_highlight_rate_pixtral, str) and long_highlight_rate_pixtral.endswith('%'):
                    try:
                        long_hl_rate_numeric = float(long_highlight_rate_pixtral.rstrip('%'))
                    except ValueError:
                        pass # Keep as 0 if conversion fails
                
                if long_hl_rate_numeric < current_tags_highlightrate:
                    console.print(f"[red]Pixtral: long_description highlight rate is too low: {long_highlight_rate_pixtral}%. Skipping.[/red]")
                    return ""

            if "502" in content_string: # This check might be less relevant if _handle_api_call_with_retries handles HTTP errors
                console.print(f"[yellow]Pixtral: Received 502 error in content. Skipping.[/yellow]")
                return ""
            
            return content_string
        
        return "" # Should not be reached if logic is correct, but as a fallback.

    elif args.gemini_api_key != "":
        # --- Initial Gemini Setup ---
        current_system_prompt = system_prompt # Use the general system_prompt first
        current_prompt = prompt # Use the general prompt first

        generation_config_gemini = (
            config["generation_config"][args.gemini_model_path.replace(".", "_")]
            if args.gemini_model_path.replace(".", "_") in config["generation_config"] else config["generation_config"]["default"]
        )

        if args.gemini_task and mime.startswith("image"):
            args.gemini_model_path = "gemini-2.0-flash-exp" # Model override
            current_system_prompt = None # Task-specific prompts might not use a system prompt
            current_prompt = config["prompts"]["task"][args.gemini_task]

        image_response_schema_gemini = None
        if mime.startswith("image"):
            image_response_schema_gemini = genai.types.Schema(
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
                            "Level of S*e*x*y", # Keeping original schema key
                            "Figure",
                            "Overall Impact & Uniqueness",
                        ],
                        properties={
                            "Costume & Makeup & Prop Presentation/Accuracy": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Character Portrayal & Posing": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Setting & Environment Integration": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Lighting & Mood": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Composition & Framing": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Storytelling & Concept": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Level of S*e*x*y": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Figure": genai.types.Schema(type=genai.types.Type.INTEGER),
                            "Overall Impact & Uniqueness": genai.types.Schema(type=genai.types.Type.INTEGER),
                        },
                    ),
                    "total_score": genai.types.Schema(type=genai.types.Type.INTEGER),
                    "average_score": genai.types.Schema(type=genai.types.Type.NUMBER),
                    "description": genai.types.Schema(type=genai.types.Type.STRING),
                    "character_name": genai.types.Schema(type=genai.types.Type.STRING),
                    "series": genai.types.Schema(type=genai.types.Type.STRING),
                },
            )

        genai_content_config = types.GenerateContentConfig(
            system_instruction=current_system_prompt,
            temperature=generation_config_gemini["temperature"],
            top_p=generation_config_gemini["top_p"],
            top_k=generation_config_gemini["top_k"],
            candidate_count=config["generation_config"]["candidate_count"],
            max_output_tokens=generation_config_gemini["max_output_tokens"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            safety_settings=[
                types.SafetySetting(category=cat, threshold=types.HarmBlockThreshold.OFF)
                for cat in [
                    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, # Added for completeness if needed
                    types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                ]
            ],
            response_mime_type=("application/json" if mime.startswith("image") else generation_config_gemini["response_mime_type"]),
            response_modalities=generation_config_gemini.get("response_modalities"), # Use .get for safety
            response_schema=image_response_schema_gemini if mime.startswith("image") else None,
            thinking_config=(
                types.ThinkingConfig(thinking_budget=generation_config_gemini["thinking_budget"])
                if "thinking_budget" in generation_config_gemini
                else None
            ),
        )
        console.print(f"Gemini generation_config: {genai_content_config}")
        gemini_client = genai.Client(api_key=args.gemini_api_key)
        gemini_uploaded_files = None # Will store the list of file(s) if upload happens

        # --- Refactor File Upload Logic ---
        if mime.startswith("video") or (mime.startswith("audio") and Path(uri).stat().st_size >= 20 * 1024 * 1024):
            def _upload_and_wait_gemini_file_core(client_instance, file_uri, file_mime, file_sha256hash, C):
                # This function contains the original loop for get/upload/wait_active
                # It should return the list of active files or raise an exception if its internal retries fail.
                # Original code's try-except for client.files.get and subsequent upload logic:
                C.print()
                C.print(f"[blue]Gemini: Checking files for:[/blue] {file_uri}")
                try:
                    existing_file = client_instance.files.get(name=sanitize_filename(Path(file_uri).name))
                    C.print(existing_file)
                    if (base64.b64decode(existing_file.sha256_hash).decode("utf-8") == file_sha256hash or
                            existing_file.size_bytes == Path(file_uri).stat().st_size):
                        C.print()
                        C.print(f"[cyan]Gemini: File {existing_file.name} is already at {existing_file.uri}[/cyan]")
                        active_files = [existing_file]
                        wait_for_files_active(client_instance, active_files, C) # Ensure it's active
                        C.print()
                        C.print(f"[green]Gemini: File {existing_file.name} is already active at {existing_file.uri}[/green]")
                        return active_files # Success
                    else:
                        C.print(f"[yellow]Gemini: File {existing_file.name} exists but hash mismatch. Deleting and re-uploading.[/yellow]")
                        client_instance.files.delete(name=sanitize_filename(Path(file_uri).name))
                        # Fall through to upload
                except Exception as e: # Not found or other error
                    C.print()
                    C.print(f"[yellow]Gemini: File {Path(file_uri).name} not found or error: {e}. Proceeding to upload.[/yellow]")
                
                # Upload logic (if not found or deleted)
                C.print(f"[blue]Gemini: Uploading file:[/blue] {file_uri}")
                try:
                    newly_uploaded_files = [upload_to_gemini(client_instance, file_uri, mime_type=file_mime)]
                except Exception: # Potentially name collision if sanitize_filename isn't unique enough
                    newly_uploaded_files = [upload_to_gemini(client_instance, file_uri, mime_type=file_mime, name=f"{Path(file_uri).name}_{int(time.time())}")]
                
                wait_for_files_active(client_instance, newly_uploaded_files, C)
                return newly_uploaded_files # Success

            uploaded_file_list_result = _handle_api_call_with_retries(
                api_call_func=lambda: _upload_and_wait_gemini_file_core(gemini_client, uri, mime, sha256hash, console),
                max_retries=args.max_retries, # The inner loop has its own retries, this is an outer layer.
                wait_time_seconds=args.wait_time * 2, # Longer wait for uploads
                console=console,
                api_name="GeminiFileUpload",
                progress=progress,
                task_id=task_id,
                on_success_description="Gemini file uploaded and active."
            )

            if not uploaded_file_list_result or not isinstance(uploaded_file_list_result, list):
                console.print("[red]Gemini: File upload failed after all retries. Skipping.[/red]")
                return ""
            gemini_uploaded_files = uploaded_file_list_result
        
        # --- Define Core API Call Function for Gemini Content Generation ---
        gemini_pixels_for_layout = None # For image processing

        def _make_gemini_api_call():
            nonlocal gemini_pixels_for_layout # To assign pixels from image encoding
            
            contents_for_gemini = []
            if gemini_uploaded_files: # Large video/audio
                contents_for_gemini.extend([
                    types.Part.from_uri(file_uri=gemini_uploaded_files[0].uri, mime_type=mime),
                    types.Part.from_text(text=current_prompt)
                ])
            elif mime.startswith("audio"): # Small audio
                audio_blob_data = Path(uri).read_bytes()
                contents_for_gemini.extend([
                    types.Part.from_bytes(data=audio_blob_data, mime_type=mime),
                    types.Part.from_text(text=current_prompt)
                ])
            elif mime.startswith("image"):
                img_blob, pixels_from_encode = encode_image(uri)
                if img_blob is None:
                    raise Exception(f"Gemini: Failed to encode image {uri}")
                gemini_pixels_for_layout = pixels_from_encode # Store for later layout
                contents_for_gemini.extend([
                    types.Part.from_text(text=current_prompt),
                    types.Part.from_bytes(data=img_blob, mime_type="image/jpeg") # Assuming JPEG
                ])
            else:
                raise ValueError(f"Gemini: Unsupported mime type for direct content generation: {mime}")

            stream_response = gemini_client.models.generate_content_stream(
                model=args.gemini_model_path,
                contents=contents_for_gemini,
                config=genai_content_config,
            )

            # Handle binary data saving side-effect if present in stream
            # This needs to be done while iterating, _collect_streaming_response is for text.
            collected_text_parts = []
            for chunk in stream_response:
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                
                part = chunk.candidates[0].content.parts[0]
                if part.text:
                    collected_text_parts.append(part.text)
                    console.print(".", end="", style="blue") # Mimic _collect_streaming_response visual
                
                if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                    saved_file_path_stem = Path(uri).with_stem(Path(uri).stem + "_s")
                    save_binary_file(saved_file_path_stem, part.inline_data.data)
                    console.print(f"\n[blue]Gemini: Inline binary data saved to {saved_file_path_stem} (mime: {part.inline_data.mime_type})[/blue]")
            
            console.print("\n") # After dots
            final_text = "".join(collected_text_parts)
            return final_text, gemini_pixels_for_layout # Return text and any pixels

        # --- Invoke Retry Handler for Content Generation ---
        api_result_tuple = _handle_api_call_with_retries(
            api_call_func=_make_gemini_api_call,
            max_retries=args.max_retries,
            wait_time_seconds=args.wait_time,
            console=console,
            api_name="GeminiContentGeneration",
            progress=progress,
            task_id=task_id,
            on_success_description="Gemini response received."
        )

        if not api_result_tuple or not isinstance(api_result_tuple, tuple):
            console.print("[red]Gemini: API call for content generation failed or returned unexpected result. Skipping.[/red]")
            return ""
        
        response_text, returned_pixels_for_layout = api_result_tuple

        if not response_text and not mime.startswith("image"): # Image might return empty text but valid JSON
             console.print("[red]Gemini: Empty response text received. Skipping.[/red]")
             return ""

        try:
            console.print(response_text)
        except Exception: # Fallback for rich text issues
            console.print(Text(response_text))

        # --- Process the Response ---
        if mime.startswith("image"):
            response_text_cleaned = response_text.replace("*", "").strip()
            try:
                captions_data = json.loads(response_text_cleaned)
            except json.JSONDecodeError as e:
                console.print(f"[red]Gemini: Error decoding JSON from image response: {e}. Response was: '{response_text_cleaned}'[/red]")
                return ""
            
            description = captions_data.get("description", "")
            scores = captions_data.get("scores", {}) # Default to empty dict
            average_score = captions_data.get("average_score", 0.0)

            caption_and_rate_layout = CaptionAndRateLayout(
                tag_description="", # Gemini image task doesn't use tag_description here
                rating=scores,
                average_score=average_score,
                long_description=description,
                pixels=returned_pixels_for_layout, # Use pixels returned from _make_gemini_api_call
                long_highlight_rate=0, # Not applicable here
                panel_height=32,
                console=console,
            )
            caption_and_rate_layout.print(title=Path(uri).name)
            if returned_pixels_for_layout: del returned_pixels_for_layout
            return response_text_cleaned # Return the JSON string

        else: # Video/Audio
            content = _post_process_srt_response(
                response_text,
                console, # console instance from api_process_batch
                progress,
                task_id,
                "Processing Gemini media..."
            )
            if not content:
                return ""
            return content
        
        return "" # Should be unreachable if logic is correct

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
