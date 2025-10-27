# -*- coding: utf-8 -*-
"""
Qwen-VL provider attempt logic extracted for Phase 5.
Keeps behavior and logging identical.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.text import Text
from rich.progress import Progress

from utils.parse_display import extract_code_block_content
from pathlib import Path


def _collect_stream_qwen(responses: Iterable[Any], console: Console) -> str:
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


def attempt_qwenvl(
    *,
    model_path: str,
    api_key: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
) -> str:
    """Single-attempt Qwen-VL request.

    Returns SRT content, raises on retryable conditions.
    """
    if not api_key or not str(api_key).strip():
        from utils.transformer_loader import transformerLoader, resolve_device_dtype
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        start_time = time.time()
        root_dir = Path(__file__).resolve().parent.parent.parent
        cfg_model_id = "Qwen/Qwen3-VL-8B-Instruct"
        gen_defaults = {
            "greedy": False,
            "top_p": 0.8,
            "top_k": 20,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "out_seq_length": 16384,
        }
        try:
            import toml  # type: ignore
            cfg = toml.load(root_dir / "config" / "config.toml")
            section = cfg.get("qwen_vl_local", {}) or {}
            cfg_model_id = section.get("model_id", cfg_model_id)
            for k in gen_defaults.keys():
                if k in section:
                    gen_defaults[k] = section[k]
        except Exception:
            pass

        device, dtype, attn_impl = resolve_device_dtype()
        loader = getattr(attempt_qwenvl, "_TRANS_LOADER", None)
        if loader is None:
            loader = transformerLoader(attn_kw="attn_implementation", device_map="auto")
            setattr(attempt_qwenvl, "_TRANS_LOADER", loader)

        if console:
            console.print(f"[blue]Loading local Qwen-VL model:[/blue] {cfg_model_id} (device={device}, dtype={dtype})")

        processor = loader.get_or_load_processor(cfg_model_id, AutoProcessor, console=console)
        model = loader.get_or_load_model(
            cfg_model_id,
            Qwen3VLForConditionalGeneration,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            device_map="auto",
            console=console,
        )

        norm_messages: list[dict[str, Any]] = []
        for m in messages:
            content = []
            for item in m.get("content", []):
                if isinstance(item, dict):
                    if "type" in item:
                        content.append(item)
                    elif "video" in item:
                        content.append({"type": "video", "video": item.get("video")})
                    elif "image" in item:
                        content.append({"type": "image", "image": item.get("image")})
                    elif "text" in item:
                        content.append({"type": "text", "text": item.get("text", "")})
            norm_messages.append({"role": m.get("role", "user"), "content": content})

        inputs = loader.prepare_image_inputs(
            processor,
            norm_messages,
            device=model.device,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        )

        do_sample = not bool(gen_defaults.get("greedy", False))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(gen_defaults.get("out_seq_length", 1024)),
            "do_sample": do_sample,
            "temperature": float(gen_defaults.get("temperature", 0.7)),
            "top_p": float(gen_defaults.get("top_p", 0.8)),
            "top_k": int(gen_defaults.get("top_k", 20)),
            "repetition_penalty": float(gen_defaults.get("repetition_penalty", 1.0)),
        }
        try:
            pres = float(gen_defaults.get("presence_penalty", 0.0))
            if hasattr(model, "generation_config") and hasattr(model.generation_config, "presence_penalty"):
                model.generation_config.presence_penalty = pres  # type: ignore[attr-defined]
        except Exception:
            pass

        generated_ids = model.generate(**inputs, **generation_kwargs)
        try:
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
        except Exception:
            generated_ids_trimmed = [generated_ids[0]] if hasattr(generated_ids, "__getitem__") else []

        output_text_list = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response_text = output_text_list[0] if output_text_list else ""

        elapsed_time = time.time() - start_time
        console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

        try:
            console.print(response_text)
        except Exception:
            console.print(Text(response_text))

        response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")
        if progress and task_id is not None:
            progress.update(task_id, description="Processing media...")
        return response_text

    # Import dashscope lazily to avoid hard dependency at module import
    import dashscope  # type: ignore

    start_time = time.time()

    responses = dashscope.MultiModalConversation.call(
        model=model_path,
        messages=messages,
        api_key=api_key,
        stream=True,
        incremental_output=True,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_qwen(responses, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    content = extract_code_block_content(response_text, "srt", console)
    if not content:
        # Trigger retry when content is empty
        raise Exception("RETRY_EMPTY_CONTENT")

    if progress and task_id is not None:
        progress.update(task_id, description="Processing media...")
    return content
