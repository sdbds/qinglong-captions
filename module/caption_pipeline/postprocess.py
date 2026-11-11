from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path

import pysrt

from config.config import (
    APPLICATION_EXTENSIONS_SET,
    AUDIO_EXTENSIONS_SET,
    IMAGE_EXTENSIONS_SET,
    VIDEO_EXTENSIONS_SET,
)
from module.providers.base import CaptionResult, CaptionStatus
from module.providers.image_template import (
    active_image_template,
    postprocess_image_template_text,
)
from utils.caption_modes import normalize_sectioned_caption
from utils.parse_display import extract_code_block_content, process_llm_response
from utils.path_safety import safe_child_path, safe_leaf_name


_FULL_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", re.IGNORECASE | re.DOTALL)


def _deduplicate_filename(filename: str, used_names: set[str]) -> str:
    candidate = filename
    stem = Path(filename).stem or "image"
    suffix = Path(filename).suffix
    counter = 2

    while candidate.lower() in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate.lower())
    return candidate


def _assign_ocr_image_names(images) -> tuple[list[str], dict[str, str], dict[str, str]]:
    assigned_names: list[str] = []
    raw_name_map: dict[str, str] = {}
    leaf_name_map: dict[str, str] = {}
    used_names: set[str] = set()

    for index, image in enumerate(images, start=1):
        raw_name = str(getattr(image, "id", "") or "")
        default_name = f"image_{index}.png"
        safe_name = _deduplicate_filename(
            safe_leaf_name(raw_name, default_name=default_name),
            used_names,
        )
        assigned_names.append(safe_name)

        if raw_name and raw_name not in raw_name_map:
            raw_name_map[raw_name] = safe_name

        if raw_name:
            leaf_name = Path(raw_name.replace("\\", "/")).name
            if leaf_name and leaf_name not in leaf_name_map:
                leaf_name_map[leaf_name] = safe_name

    return assigned_names, raw_name_map, leaf_name_map


def _rewrite_ocr_image_paths(markdown: str, parent_dir: str, raw_name_map: dict[str, str], leaf_name_map: dict[str, str]) -> str:
    if not raw_name_map and not leaf_name_map:
        return markdown

    pattern = re.compile(r"!\[(.*?)\]\(([^)]+)\)")

    def replace(match: re.Match[str]) -> str:
        target = match.group(2).strip()
        normalized_target = target.replace("\\", "/")
        safe_name = raw_name_map.get(target) or raw_name_map.get(normalized_target)
        if not safe_name:
            safe_name = leaf_name_map.get(Path(normalized_target).name)
        if not safe_name:
            return match.group(0)
        return f"![{match.group(1)}]({parent_dir}/{safe_name})"

    return pattern.sub(replace, markdown)


def _decode_base64_image(image_base64: str) -> bytes:
    if image_base64.startswith("data:"):
        return base64.b64decode(image_base64.split(",", 1)[1])
    return base64.b64decode(image_base64)


def _normalize_subtitle_timestamps(output: str) -> str:
    timestamp_pattern = re.compile(
        r"(?<!:)(\d):(\d{2})[,:.](\d{3})|"
        r"(?<!:)(\d{2}):(\d{2})[,:.](\d{3})|"
        r"(?<!:)(\d{2}):(\d{2}):(\d{2})[,:.](\d{3})|"
        r"(?<![0-9:])([1-9][0-9][0-9]+):(\d{2}):(\d{2})[,:.](\d{3})|"
        r"(?<!:)(\d{2}):(\d{2})[,:.](\d{2})[,:.](\d{3})",
        re.MULTILINE,
    )

    def normalize_timestamp(match):
        groups = match.groups()
        if groups[0] is not None:
            return f"00:0{groups[0]}:{groups[1]},{groups[2]}"
        if groups[3] is not None:
            return f"00:{groups[3]}:{groups[4]},{groups[5]}"
        if groups[6] is not None:
            return f"{groups[6]}:{groups[7]}:{groups[8]},{groups[9]}"
        if groups[10] is not None:
            return f"{groups[10]}:{groups[11]}:{groups[12]},{groups[13]}"
        if groups[14] is not None:
            return f"00:{groups[15]}:{groups[16]},{groups[17]}"
        return match.group(0)

    return timestamp_pattern.sub(normalize_timestamp, output)


def strip_reasoning_sections(output: str) -> str:
    if not output:
        return ""
    return re.sub(r"<think\b[^>]*>.*?</think>", "", output, flags=re.IGNORECASE | re.DOTALL).strip()


def extract_subtitle_text(output: str, console=None) -> str:
    cleaned = strip_reasoning_sections(output)
    if not cleaned:
        return ""

    extracted = extract_code_block_content(cleaned, "srt", console)
    return (extracted or cleaned).strip()


def normalize_and_validate_subtitle_text(output: str, console=None) -> str:
    subtitle_text = extract_subtitle_text(output, console)
    normalized = _normalize_subtitle_timestamps(subtitle_text).strip()
    if not normalized:
        raise ValueError("EMPTY_SUBTITLE_OUTPUT")

    subtitles = pysrt.from_string(normalized)
    if len(subtitles) == 0:
        raise ValueError("EMPTY_SUBTITLE_ITEMS")

    for subtitle in subtitles:
        if subtitle.end.ordinal <= subtitle.start.ordinal:
            raise ValueError("INVALID_SUBTITLE_RANGE")
        if not str(subtitle.text or "").strip():
            raise ValueError("EMPTY_SUBTITLE_TEXT")

    return normalized


def _parse_json_object(value: str) -> dict | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _caption_result_from_processed(output, metadata: dict | None = None, *, mode: str | None = None, raw=None) -> CaptionResult:
    metadata = dict(metadata or {})
    if isinstance(output, CaptionResult):
        return output
    if isinstance(output, dict):
        normalized = normalize_sectioned_caption(output, mode) if mode is not None else None
        if normalized is not None:
            output = normalized
            raw = None
        if raw is None:
            raw = json.dumps(output, ensure_ascii=False)
        return CaptionResult(raw=raw, parsed=output, metadata=metadata)
    if isinstance(output, list):
        return CaptionResult(raw="\n".join(str(line) for line in output), metadata=metadata)
    return CaptionResult(raw=str(output), metadata=metadata)


def postprocess_caption_content(output, filepath, args, console):
    if isinstance(output, CaptionResult) and output.status is not CaptionStatus.SUCCESS:
        return output

    metadata = dict(output.metadata) if isinstance(output, CaptionResult) else {}
    if isinstance(output, CaptionResult):
        output = output.payload

    if not output:
        console.print(f"[red]No caption content generated for {filepath}[/red]")
        return CaptionResult(raw="", metadata=metadata)

    suffix = Path(filepath).suffix.lower()
    mime = mimetypes.guess_type(str(filepath))[0] or ""
    is_image = suffix in IMAGE_EXTENSIONS_SET or mime.startswith("image/")
    image_mode = getattr(args, "mode", "all") if is_image else None

    if isinstance(output, list):
        if output and hasattr(output[0], "markdown") and hasattr(output[0], "index"):
            combined_output = []
            for page in output:
                page_index = page.index if hasattr(page, "index") else "unknown"
                combined_output.append(
                    '<header style="background-color: #f5f5f5; padding: 8px; margin-bottom: 20px; text-align: center; border-bottom: 1px solid #ddd;">\n'
                    f"<strong> Page {page_index + 1} </strong>\n"
                    "</header>\n\n"
                )

                page_markdown = page.markdown if hasattr(page, "markdown") else str(page)
                page_images = list(getattr(page, "images", []) or [])
                image_names, raw_name_map, leaf_name_map = _assign_ocr_image_names(page_images)

                if page_images and args.document_image:
                    parent_dir = Path(filepath).stem
                    page_markdown = _rewrite_ocr_image_paths(page_markdown, parent_dir, raw_name_map, leaf_name_map)

                    for image_index, image in enumerate(page_images):
                        if not getattr(image, "image_base64", ""):
                            continue
                        try:
                            image_data = _decode_base64_image(image.image_base64)
                            image_dir = Path(filepath).with_suffix("")
                            image_dir.mkdir(parents=True, exist_ok=True)
                            image_path = safe_child_path(
                                image_dir,
                                image_names[image_index],
                                default_name=f"image_{image_index + 1}.png",
                            )
                            image_path.write_bytes(image_data)
                        except Exception as exc:
                            console.print(f"[yellow]Error saving OCR image: {exc}[/yellow]")

                combined_output.append(f"{page_markdown}\n\n")
                combined_output.append(
                    '<footer style="background-color: #f5f5f5; padding: 8px; margin-top: 20px; text-align: center; border-top: 1px solid #ddd;">\n'
                    f"<strong> Page {page_index + 1} </strong>\n"
                    "</footer>\n\n"
                )
                combined_output.append('<div style="page-break-after: always;"></div>\n\n')
            output = "".join(combined_output)
        else:
            output = "\n".join(output)

    if isinstance(output, dict):
        return _caption_result_from_processed(output, metadata, mode=image_mode)

    output = str(output).strip()
    if not output:
        console.print(f"[red]Empty caption content for {filepath}[/red]")
        return CaptionResult(raw="", metadata=metadata)

    template_id = "" if getattr(args, "pair_dir", "") else active_image_template(args)
    output = postprocess_image_template_text(output, template_id)

    if suffix in VIDEO_EXTENSIONS_SET or suffix in AUDIO_EXTENSIONS_SET:
        return CaptionResult(raw=normalize_and_validate_subtitle_text(output, console), metadata=metadata)

    if suffix in APPLICATION_EXTENSIONS_SET and getattr(args, "ocr_model", "") != "":
        return CaptionResult(raw=output, metadata=metadata)

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return _caption_result_from_processed(parsed, metadata, mode=image_mode, raw=output)
    except json.JSONDecodeError:
        has_template = bool(active_image_template(args))
        # Backticks inside a JSON string do not delimit the outer fence.
        match = _FULL_JSON_FENCE.fullmatch(output) if is_image else None
        fenced = match[1].strip() if match else ""
        fenced_parsed = _parse_json_object(fenced)
        if fenced_parsed is None and has_template:
            fenced = extract_code_block_content(output, "json", console)
            fenced_parsed = _parse_json_object(fenced)
        if fenced_parsed is not None:
            result = _caption_result_from_processed(fenced_parsed, metadata, mode=image_mode, raw=fenced)
            if has_template or result.parsed != fenced_parsed:
                return result
            # Custom JSON is not a plaintext caption, even when nested strings contain headings.
            return CaptionResult(raw=output, metadata=metadata)

    if "###" in output:
        shortdescription, long_description = process_llm_response(output)
        if args.mode == "all":
            return CaptionResult(raw="\n".join([shortdescription, long_description]), metadata=metadata)
        if args.mode == "long":
            return CaptionResult(raw=long_description, metadata=metadata)
        if args.mode == "short":
            return CaptionResult(raw=shortdescription, metadata=metadata)

    return CaptionResult(raw=output, metadata=metadata)
