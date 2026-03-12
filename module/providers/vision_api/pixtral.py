"""Mistral OCR Provider."""

from __future__ import annotations

import base64
import io
import re
import time
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from providers.catalog import get_first_attr, route_matches_provider
from providers.base import CaptionResult, MediaContext, PromptContext
from providers.registry import register_provider
from providers.utils import build_vision_messages
from providers.vision_api_base import VisionAPIProvider
from utils.parse_display import (
    display_caption_layout,
    display_markdown,
    process_llm_response,
)
from utils.stream_util import format_description


# ---------------------------------------------------------------------------
# attempt_pixtral  –  moved from module.providers.pixtral_provider
# ---------------------------------------------------------------------------

def attempt_pixtral(
    *,
    client: Any,
    model_path: str,
    mime: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    # Common
    uri: str,
    # Image chat path
    messages: Optional[List[dict]] = None,
    pixels: Optional[Pixels] = None,
    captions: Optional[List[str]] = None,
    prompt_text: Optional[str] = None,
    character_name: str = "",
    tags_highlightrate: float = 0.0,
    # Image OCR path
    ocr: bool = False,
    base64_image: Optional[str] = None,
    # Document (application) OCR path
    document_image: bool = False,
    signed_url_url: Optional[str] = None,
) -> Any:
    """Single-attempt Mistral OCR request.

    Returns:
      - For image chat: str content (model response)
      - For image OCR: str markdown
      - For application OCR: list of pages (each with markdown/optional image), and side-effect displays
    May raise exceptions like RETRY_PIXTRAL_* to trigger with_retry.
    """
    start_time = time.time()

    if mime.startswith("application"):
        # PDF OCR path
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url_url,
            },
            include_image_base64=document_image,
        )
        pages = ocr_response.pages
        console.print(f"[bold cyan]PDF共有 {len(pages)} 页[/bold cyan]")

        for page in pages:
            # Try preview first image on the page
            ocr_pixels = None
            if page.images and len(page.images) > 0:
                first_image = page.images[0]
                if hasattr(first_image, "image_base64") and first_image.image_base64:
                    try:
                        base64_str = first_image.image_base64
                        if base64_str.startswith("data:"):
                            base64_content = base64_str.split(",", 1)[1]
                            image_data = base64.b64decode(base64_content)
                        else:
                            image_data = base64.b64decode(base64_str)
                        ocr_image = Image.open(io.BytesIO(image_data))
                        ocr_pixels = Pixels.from_image(
                            ocr_image,
                            resize=(ocr_image.width // 18, ocr_image.height // 18),
                        )
                    except Exception as e:
                        console.print(f"[yellow]Error loading image: {e}[/yellow]")
                        ocr_pixels = None
                else:
                    console.print("[yellow]Image found but no base64 data available[/yellow]")
            display_markdown(
                title=f"{Path(uri).name} - Page {page.index + 1}",
                markdown_content=page.markdown,
                pixels=ocr_pixels,
                panel_height=32,
                console=console,
            )

        elapsed_time = time.time() - start_time
        if progress and task_id is not None:
            progress.update(task_id, description="Generating captions")
        if elapsed_time < 0:
            elapsed_time = 0
        console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")
        # Preserve original behavior: side-effect display, return empty string
        return ""

    if ocr:
        # Image OCR
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            },
        )
        content = ocr_response.pages[0].markdown
        display_markdown(
            title=Path(uri).name,
            markdown_content=content,
            pixels=pixels,
            panel_height=32,
            console=console,
        )
    else:
        # Image chat
        chat_response = client.chat.complete(model=model_path, messages=messages)
        content = chat_response.choices[0].message.content

        short_description, long_description = process_llm_response(content)
        captions = captions or []
        if len(captions) > 0:
            tag_description = (
                (
                    prompt_text.rsplit("<s>[INST]", 1)[-1]
                    .rsplit("<|end_of_focus|>", 1)[-1]  # fallback safe split no-op
                    .rsplit(">.", 1)[-1]
                    .rsplit(").", 1)[-1]
                    .replace(" from", ",")
                )
                .rsplit("[IMG][/INST]", 1)[0]
                .strip()
            )
            short_description, short_highlight_rate = format_description(short_description, tag_description)
            long_description, long_highlight_rate = format_description(long_description, tag_description)
        else:
            tag_description = ""
            short_highlight_rate = 0
            long_highlight_rate = 0

        # Display
        display_caption_layout(
            title=Path(uri).name,
            tag_description=tag_description,
            short_description=short_description,
            long_description=long_description,
            pixels=pixels,
            short_highlight_rate=short_highlight_rate,
            long_highlight_rate=long_highlight_rate,
            panel_height=32,
            console=console,
        )

        # Validations (non OCR only)
        if character_name:
            clean_char_name = character_name.split(",")[0].split(" from ")[0].strip("<>")
            if clean_char_name not in content:
                # Check case-insensitive match and fix if found
                pattern = re.compile(re.escape(clean_char_name), re.IGNORECASE)
                if pattern.search(content):
                    content = pattern.sub(clean_char_name, content)
                    console.print(f"[yellow]Fixed character name case: [green]{clean_char_name}[/green][/yellow]")
                else:
                    console.print()
                    console.print(Text(content))
                    console.print(f"Character name [green]{clean_char_name}[/green] not found")
                    raise Exception("RETRY_PIXTRAL_CHAR")

        if "###" not in content:
            console.print(Text(content))
            console.print(Text("No ###, retrying...", style="yellow"))
            raise Exception("RETRY_PIXTRAL_NO_MARK")

        # Tags highlight-rate threshold
        if (
            any(f"{i}women" in tag_description for i in range(2, 5))
            or ("1man" in tag_description and "1woman" in tag_description)
            or "multiple girls" in tag_description
            or "multiple boys" in tag_description
        ):
            threshold = tags_highlightrate * 100 / 2
        else:
            threshold = tags_highlightrate * 100
        if int(re.search(r"\d+", str(long_highlight_rate)).group()) < threshold and len(captions) > 0:
            console.print(f"[red]long_description highlight rate is too low: {long_highlight_rate}%, retrying...[/red]")
            raise Exception("RETRY_PIXTRAL_RATE")

        if isinstance(content, str) and "502" in content:
            console.print("[yellow]Received 502 error[/yellow]")
            raise Exception("RETRY_PIXTRAL_502")

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")
    return content


# ---------------------------------------------------------------------------
# MistralOCRProvider  –  V2 provider class
# ---------------------------------------------------------------------------

@register_provider("mistral_ocr")
class MistralOCRProvider(VisionAPIProvider):
    """Mistral OCR Provider."""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        # OCR 模式：用户选择 ocr_model 时走 Mistral OCR
        if route_matches_provider("ocr_model", getattr(args, "ocr_model", ""), cls.name):
            if mime.startswith("application"):
                return True
            if mime.startswith("image") and getattr(args, "document_image", False):
                return True
        return get_first_attr(args, "mistral_api_key", "pixtral_api_key", default="") != "" and (
            mime.startswith("image") or mime.startswith("application")
        )

    def prepare_media(self, uri: str, mime: str, args) -> MediaContext:
        """Mistral OCR 支持 PDF，需要特殊处理"""
        media = super().prepare_media(uri, mime, args)

        # PDF 处理
        if mime.startswith("application"):
            from mistralai import Mistral

            client = Mistral(api_key=get_first_attr(self.ctx.args, "mistral_api_key", "pixtral_api_key", default=""))

            # 上传 PDF
            for upload_attempt in range(getattr(args, "max_retries", 10)):
                try:
                    from utils.stream_util import sanitize_filename

                    with open(uri, "rb") as pdf_f:
                        uploaded_pdf = client.files.upload(
                            file={
                                "file_name": f"{sanitize_filename(uri)}.pdf",
                                "content": pdf_f,
                            },
                            purpose="ocr",
                        )
                    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
                    media.extras["signed_url"] = signed_url.url
                    break
                except Exception as e:
                    self.log(f"Error uploading PDF: {e}", "red")
                    if upload_attempt < args.max_retries - 1:
                        self.log(f"Retrying in {args.wait_time}s...", "yellow")
                        import time

                        time.sleep(args.wait_time)
                    else:
                        self.log(f"Failed to upload PDF after {args.max_retries} attempts", "red")

        return media

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from mistralai import Mistral

        client = Mistral(api_key=get_first_attr(self.ctx.args, "mistral_api_key", "pixtral_api_key", default=""))

        # 检查是否是 OCR 模式
        ocr_mode = route_matches_provider("ocr_model", getattr(self.ctx.args, "ocr_model", ""), self.name)

        # 准备 captions（用于 tag highlight）
        captions = []
        if media.mime.startswith("image"):
            captions_path = Path(media.uri).with_suffix(".txt")
            if captions_path.exists():
                with open(captions_path, "r", encoding="utf-8") as f:
                    captions = [line.strip() for line in f.readlines()]

        # 构建 messages
        if media.mime.startswith("application"):
            # PDF 模式
            signed_url = media.extras.get("signed_url")
            if not signed_url:
                return CaptionResult(raw="", metadata={"error": "PDF upload failed"})

            result = attempt_pixtral(
                client=client,
                model_path=get_first_attr(self.ctx.args, "mistral_model_path", "pixtral_model_path", default=""),
                mime=media.mime,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                uri=media.uri,
                document_image=self.ctx.args.document_image,
                signed_url_url=signed_url,
            )
        elif ocr_mode:
            # OCR 模式
            result = attempt_pixtral(
                client=client,
                model_path=get_first_attr(self.ctx.args, "mistral_model_path", "pixtral_model_path", default=""),
                mime=media.mime,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                uri=media.uri,
                ocr=True,
                base64_image=media.blob,
                pixels=media.pixels,
            )
        else:
            # 标准图像模式
            if media.pair_blob:
                character_name = ""
                prompt_text = prompts.user
                messages = build_vision_messages(
                    prompts.system,
                    prompt_text,
                    media.blob,
                    pair_blob=media.pair_blob,
                    text_first=False,
                )
            else:
                character_name = ""
                if prompts.character_name:
                    character_name = f"{prompts.character_name}, "

                # 非 pair 图像模式保留旧的 Pixtral 指令包装
                prompts_config = self.ctx.config.get("prompts", {})
                config_prompt = prompts_config.get("mistral_ocr_image_prompt", prompts_config.get("pixtral_image_prompt", ""))
                prompt_text = Text(
                    f"<s>[INST]{prompts.character_prompt}{character_name}{captions[0] if captions else config_prompt}\n[IMG][/INST]"
                ).plain
                messages = build_vision_messages(prompts.system, prompt_text, media.blob, text_first=True)

            result = attempt_pixtral(
                client=client,
                model_path=get_first_attr(self.ctx.args, "mistral_model_path", "pixtral_model_path", default=""),
                mime=media.mime,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                uri=media.uri,
                messages=messages,
                pixels=media.pixels,
                captions=captions,
                prompt_text=prompt_text,
                character_name=character_name,
                tags_highlightrate=getattr(self.ctx.args, "tags_highlightrate", 0.0),
            )

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def post_validate(self, result: CaptionResult, media: MediaContext, args) -> CaptionResult:
        """Mistral OCR 特殊的后验证：角色名校验"""
        description = result.description

        # 角色名校验（如果设置了）
        if result.parsed and "character_name" in result.parsed:
            expected_char = result.parsed["character_name"]
            if expected_char and expected_char not in description:
                # 需要重试 - 抛出特定异常让 retry 机制处理
                raise Exception(f"RETRY_CHARACTER_NOT_FOUND: Character '{expected_char}' not found")

        return result

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "429" in msg:
                return 59.0
            if "502" in msg or "RETRY_PIXTRAL_" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
