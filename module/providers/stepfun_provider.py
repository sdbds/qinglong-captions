# -*- coding: utf-8 -*-
"""
StepFun provider attempt logic extracted from api_handler for Phase 5.
Keeps behavior and logging identical. English logs/comments by convention.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from utils.parse_display import (
    display_caption_and_rate,
    display_pair_image_description,
    extract_code_block_content,
)


def _collect_stream_stepfun(completion: Any, console: Console) -> str:
    """Collect streamed text from StepFun(OpenAI-compatible) responses."""
    chunks: list[str] = []
    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
            chunks.append(chunk.choices[0].delta.content)
            console.print(".", end="", style="blue")
    console.print("\n")
    return "".join(chunks)


def attempt_stepfun(
    *,
    client: Optional[OpenAI],
    model_path: str,
    mime: str,
    system_prompt: str,
    prompt: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    uri: str,
    image_blob: Optional[str] = None,
    image_pixels: Optional[Pixels] = None,
    has_pair: bool = False,
    pair_blob: Optional[str] = None,
    pair_pixels: Optional[Pixels] = None,
    pair_uri: Optional[str] = None,
    video_file_id: Optional[str] = None,
) -> str:
    """Single-attempt StepFun request.

    Returns the SRT content for video, otherwise the response text for image.
    May raise exceptions (e.g., RETRY_EMPTY_CONTENT) to trigger with_retry.
    """
    # Local model path when client is None
    if client is None:
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor

        from utils.transformer_loader import resolve_device_dtype, transformerLoader

        start_time = time.time()
        root_dir = Path(__file__).resolve().parent.parent.parent
        cfg_model_id = "stepfun-ai/Step3-VL-10B"
        gen_defaults = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "eos_token_id": [151643, 151645, 151679]}
        try:
            import toml  # type: ignore

            cfg = toml.load(root_dir / "config" / "config.toml")
            section = cfg.get("stepfun_local", {}) or {}
            cfg_model_id = section.get("model_id", cfg_model_id)
            for k in gen_defaults.keys():
                if k in section:
                    gen_defaults[k] = section[k]
        except Exception:
            pass

        device, dtype, attn_impl = resolve_device_dtype()
        # Step3-VL does not support Flash Attention 2.0, force eager mode
        attn_impl = "eager"
        loader = getattr(attempt_stepfun, "_TRANS_LOADER", None)
        if loader is None:
            loader = transformerLoader(attn_kw="attn_implementation", device_map="auto")
            setattr(attempt_stepfun, "_TRANS_LOADER", loader)

        if console:
            console.print(
                f"[blue]Loading local Step3-VL model:[/blue] {cfg_model_id} (device={device}, dtype={dtype}, attn={attn_impl})"
            )

        # Step3-VL requires special key_mapping parameter
        key_mapping = {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        }

        processor = loader.get_or_load_processor(cfg_model_id, AutoProcessor, console=console, trust_remote_code=True)
        model = loader.get_or_load_model(
            cfg_model_id,
            AutoModelForCausalLM,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            device_map="auto",
            console=console,
            extra_kwargs={"key_mapping": key_mapping},
        )

        # Build messages for Step3-VL format
        messages: list[dict[str, Any]] = []

        # Add system message if present (content must be a list for Step3-VL)
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        # Build user message content using PIL Images
        user_content: list[dict[str, Any]] = []

        # Load and add primary image
        try:
            img = Image.open(uri).convert("RGB")
            user_content.append({"type": "image", "image": img})
        except Exception as e:
            console.print(f"[red]Failed to load image {uri}: {e}[/red]")
            raise

        # Load and add pair image if present
        if has_pair and pair_uri:
            try:
                pair_img = Image.open(pair_uri).convert("RGB")
                user_content.append({"type": "image", "image": pair_img})
            except Exception as e:
                console.print(f"[red]Failed to load pair image {pair_uri}: {e}[/red]")
                raise

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        # Prepare inputs using processor
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        except Exception as e:
            console.print(f"[red]Error in apply_chat_template:[/red] {e}")
            console.print("[red]Full traceback:[/red]")
            import traceback

            console.print(traceback.format_exc())
            raise

        # Generation parameters
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(gen_defaults.get("max_new_tokens", 1024)),
            "do_sample": False,
        }

        if "repetition_penalty" in gen_defaults:
            generation_kwargs["repetition_penalty"] = float(gen_defaults.get("repetition_penalty", 1.0))

        if progress and task_id is not None:
            progress.update(task_id, description="Generating captions")

        console.print("[blue]Starting inference...[/blue]")
        console.print("[dim]Generating tokens:[/dim] ", end="")

        # Use TextStreamer for real-time output
        streamer = loader.get_text_streamer(processor, skip_prompt=True, skip_special_tokens=True, buffered=True, min_chars=0)

        # Generate response with streaming
        generated_ids = model.generate(**inputs, streamer=streamer, **generation_kwargs)

        console.print()  # New line after streaming

        # Decode only the newly generated tokens
        decoded = processor.decode(
            generated_ids[0, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )

        response_text = decoded

        # Remove think section if present (Step3-VL outputs thinking process)
        if "</think>" in response_text:
            response_text = response_text.split("</think>", 1)[1].strip()

        elapsed_time = time.time() - start_time
        console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

        try:
            console.print(response_text)
        except Exception:
            console.print(Text(response_text))

        response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

        # Image branch
        if has_pair and pair_pixels is not None and image_pixels is not None:
            display_pair_image_description(
                title=Path(uri).name,
                description=response_text,
                pixels=image_pixels,
                pair_pixels=pair_pixels,
                panel_height=32,
                console=console,
            )
            return response_text
        else:
            display_caption_and_rate(
                title=Path(uri).name,
                tag_description="",
                long_description=response_text,
                pixels=image_pixels,
                rating=[],
                average_score=0,
                panel_height=32,
                console=console,
            )
            return response_text

    # API client path
    start_time = time.time()

    if mime.startswith("video"):
        if not video_file_id:
            raise RuntimeError("Missing video_file_id for StepFun video request")
        completion = client.chat.completions.create(
            model=model_path,
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
                            "video_url": {"url": "stepfile://" + video_file_id},
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
        if has_pair:
            completion = client.chat.completions.create(
                model=model_path,
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
                                "image_url": image_blob,
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
                model=model_path,
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
                                "image_url": image_blob,
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
        raise RuntimeError(f"Unsupported mime for StepFun: {mime}")

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_stepfun(completion, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    if mime.startswith("video"):
        content = extract_code_block_content(response_text, "srt", console)
        if not content:
            raise Exception("RETRY_EMPTY_CONTENT")
        if progress and task_id is not None:
            progress.update(task_id, description="Processing media...")
        return content

    # image branch
    if has_pair and pair_pixels is not None and image_pixels is not None:
        display_pair_image_description(
            title=Path(uri).name,
            description=response_text,
            pixels=image_pixels,
            pair_pixels=pair_pixels,
            panel_height=32,
            console=console,
        )
        return response_text
    else:
        display_caption_and_rate(
            title=Path(uri).name,
            tag_description="",
            long_description=response_text,
            pixels=image_pixels,
            rating=[],
            average_score=0,
            panel_height=32,
            console=console,
        )
        return response_text
