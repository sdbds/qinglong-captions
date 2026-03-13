"""FireRed OCR Provider"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider
from utils.parse_display import display_markdown
from utils.output_writer import write_markdown_output
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_TRANS_LOADER: Optional[transformerLoader] = None

# Default OCR prompt for document parsing (from FireRed-OCR official)
DEFAULT_OCR_PROMPT = '''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with,(,). For example: This is an inline formula,( E = mc^2,)
- Enclose block formulas with,\\[,\\]. For example:,\\[,frac{-b,pm,sqrt{b^2 - 4ac}}{2a},\\]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with and .

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
'''


def _build_messages(image_path: str, prompt_text: str) -> list[dict]:
    """Build chat messages for FireRed-OCR inference.

    Delegates to OCRProvider.build_ocr_messages for the shared logic.
    """
    from module.providers.ocr_base import OCRProvider
    return OCRProvider.build_ocr_messages(image_path, prompt_text)


@torch.inference_mode()
def attempt_firered_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "FireRedTeam/FireRed-OCR",
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 8192,
) -> str:
    """Run local FireRed-OCR on a single image and return markdown text.

    Args:
      uri: path to the image file or PDF
      console: Rich console for output
      progress: Optional progress bar
      task_id: Optional task ID for progress updates
      model_id: HuggingFace model ID (default: FireRedTeam/FireRed-OCR)
      prompt_text: OCR instruction; default extracts document to markdown
      pixels: Optional pixel art display
      output_dir: Output directory for results
      max_new_tokens: Maximum tokens to generate (default: 8192)

    Returns:
      Extracted text content in markdown format
    """
    start_time = time.time()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not prompt_text:
        prompt_text = DEFAULT_OCR_PROMPT

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="_attn_implementation", device_map="auto")

    # Load processor and model
    processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        Qwen3VLForConditionalGeneration,
        dtype=dtype,
        attn_impl=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_safetensors=True,
        console=console,
    )

    def _infer_single_image(image_path: str, pil_image: Image.Image) -> str:
        """Perform OCR inference on a single image."""
        messages = _build_messages(image_path, prompt_text)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_texts[0] if output_texts else ""

    if p.suffix.lower() == ".pdf":
        images = pdf_to_images_high_quality(str(p))
        all_contents = []

        for idx, pil_img in enumerate(images):
            page_dir = Path(output_dir) / f"page_{idx + 1:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = page_dir / f"page_{idx + 1:04d}.png"

            try:
                pil_img.save(page_img_path)
            except Exception:
                try:
                    pil_img.convert("RGB").save(page_img_path)
                except Exception:
                    continue

            page_content = _infer_single_image(str(page_img_path), pil_img)

            # Save individual page result
            try:
                write_markdown_output(page_dir, page_content)
            except Exception:
                pass

            all_contents.append(page_content.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)

        # Save combined result
        try:
            write_markdown_output(Path(output_dir), content)
        except Exception:
            pass

        try:
            display_markdown(
                title=p.name,
                markdown_content=content,
                pixels=pixels,
                panel_height=32,
                console=console,
            )
        except Exception:
            pass
    else:
        # Single image processing
        pil_img = Image.open(str(p))
        content = _infer_single_image(str(p), pil_img)

        # Save result
        try:
            write_markdown_output(Path(output_dir), content)
        except Exception:
            pass

        try:
            display_markdown(
                title=p.name,
                markdown_content=content,
                pixels=pixels,
                panel_height=32,
                console=console,
            )
        except Exception:
            pass

    elapsed = time.time() - start_time
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")

    return content


@register_provider("firered_ocr")
class FireRedOCRProvider(OCRProvider):
    """FireRed OCR Provider"""

    default_model_id = "FireRedTeam/FireRed-OCR"
    default_prompt = ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        output_dir = media.extras.get("output_dir")

        result = attempt_firered_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", 8192),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
