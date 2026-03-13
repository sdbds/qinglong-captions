from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

import lance
import pyarrow as pa
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from config.config import get_supported_extensions
from module.lanceImport import transform2lance
from module.lanceexport import save_caption
from module.providers.local_llm.hy_mt import HYMTProvider
from utils.lance_blob import take_blob_files
from utils.doc_normalize import NormalizationError, normalize_asset
from utils.lance_utils import build_version_tag, get_latest_version_number, sanitize_tag_component, update_or_create_tag
from utils.text_chunker import compute_chunk_offsets, slice_by_offsets

console = Console(color_system="truecolor", force_terminal=True)
TEXT_EXTENSIONS = frozenset(get_supported_extensions("text"))
DOCUMENT_EXTENSIONS = frozenset(ext for ext in get_supported_extensions("application") if ext != ".psd")
_PROTECTED_RE = re.compile(
    r"(```.*?```|~~~.*?~~~|`[^`\n]+`|https?://[^\s)>\]]+|"
    r"(?:[A-Za-z]:\\|\\\\|(?:\./|\.\./|/))[^\s)>\]]+)",
    re.DOTALL,
)


class TranslationRuntimeError(RuntimeError):
    """Raised when a translation stage cannot continue."""


class Translator(Protocol):
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        context: str = "",
        glossary: str = "",
    ) -> str: ...


def classify_translation_asset(uri: Path) -> Optional[str]:
    suffix = uri.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in DOCUMENT_EXTENSIONS:
        return "application"
    return None


def protect_markdown(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        key = f"__QLP_{len(replacements)}__"
        replacements[key] = match.group(0)
        return key

    return _PROTECTED_RE.sub(repl, text), replacements


def restore_placeholders(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for key, value in replacements.items():
        restored = restored.replace(key, value)
    return restored


def preserve_chunk_whitespace(source: str, translated: str) -> str:
    leading_len = len(source) - len(source.lstrip())
    trailing_len = len(source) - len(source.rstrip())
    leading = source[:leading_len]
    trailing = source[len(source) - trailing_len :] if trailing_len else ''
    core = translated.strip()
    if not core:
        return source
    return f'{leading}{core}{trailing}'


def captions_to_text(captions: Optional[List[str]]) -> str:
    if not captions:
        return ""
    if len(captions) == 1:
        return captions[0] or ''
    return "\n".join(line for line in captions if line)


def ensure_chunk_offsets_schema(source_schema: pa.Schema) -> pa.Schema:
    if "chunk_offsets" in source_schema.names:
        return source_schema
    return source_schema.append(pa.field("chunk_offsets", pa.list_(pa.int32())))


def build_record_batch(
    batch: pa.RecordBatch,
    schema: pa.Schema,
    captions: List[List[str]],
    chunk_offsets: List[List[int]],
    blob_values: Optional[List[Optional[bytes]]] = None,
) -> pa.RecordBatch:
    batch_field_names = set(batch.schema.names)
    arrays = []
    for field in schema:
        if field.name == "captions":
            arrays.append(pa.array(captions, type=field.type))
        elif field.name == "chunk_offsets":
            arrays.append(pa.array(chunk_offsets, type=field.type))
        elif field.name == "blob":
            values = blob_values if blob_values is not None else [None] * len(batch)
            arrays.append(pa.array(values, type=field.type))
        elif field.name in batch_field_names:
            arrays.append(batch.column(field.name))
        else:
            arrays.append(pa.array([None] * len(batch), type=field.type))
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def load_batch_blobs(dataset: lance.LanceDataset, batch: pa.RecordBatch, row_offset: int) -> List[Optional[bytes]]:
    if "blob" not in dataset.schema.names or len(batch) == 0:
        return [None] * len(batch)
    indices = list(range(row_offset, row_offset + len(batch)))
    blobs: List[Optional[bytes]] = []
    for blob_file in take_blob_files(dataset, indices, "blob"):
        blobs.append(blob_file.readall() if blob_file is not None else None)
    return blobs


def sanitize_model_tag(model_id: str) -> str:
    model_name = Path(model_id).name or model_id
    return sanitize_tag_component(model_name).replace("-", "_").replace(".", "_").lower()


def try_open_dataset(dataset_path: Path, version: str | int) -> Optional[lance.LanceDataset]:
    try:
        return lance.dataset(str(dataset_path), version=version)
    except Exception:
        return None


def resolve_export_base_path(uri: Path, export_root: Optional[Path]) -> Path:
    if uri.exists():
        return uri
    if export_root is None:
        return uri
    if uri.is_absolute():
        relative_parts = [part for part in uri.parts if part not in {uri.anchor, uri.drive, "\\"}]
        relative_uri = Path(*relative_parts) if relative_parts else Path(uri.name)
    else:
        relative_uri = uri
    return export_root / relative_uri


def resolve_translated_markdown_path(uri: Path, export_root: Optional[Path], target_lang: str) -> Path:
    base_path = resolve_export_base_path(uri, export_root)
    suffix = sanitize_lang_suffix(target_lang)
    return base_path.with_name(f"{base_path.stem}_{suffix}.md")


def load_saved_translation(uri: Path, export_root: Optional[Path], target_lang: str) -> str:
    translated_path = resolve_translated_markdown_path(uri, export_root, target_lang)
    if not translated_path.exists():
        return ""
    try:
        return translated_path.read_text(encoding="utf-8")
    except Exception as exc:
        console.print(f"[yellow]Failed to read existing translation {translated_path}: {exc}[/yellow]")
        return ""


def save_translated_markdown(uri: Path, translated_markdown: str, export_root: Optional[Path], target_lang: str) -> bool:
    if not translated_markdown.strip():
        return False
    base_path = resolve_export_base_path(uri, export_root)
    return save_caption(
        str(base_path),
        [translated_markdown],
        "application",
        caption_suffix=f"_{sanitize_lang_suffix(target_lang)}",
        caption_extension=".md",
    )


def merge_translations(
    dataset_path: Path,
    base_version: str,
    translation_tag: str,
    merge_candidates: list[str],
    current_run_translations: dict[str, str],
    export_root: Optional[Path],
    target_lang: str,
    max_chars: int,
    merge_batch_size: int,
) -> int:
    target_ds = lance.dataset(str(dataset_path), version=base_version)
    target_schema = target_ds.schema
    include_chunk_offsets = "chunk_offsets" in target_schema.names

    merge_batch_size = max(1, merge_batch_size)
    console.print(
        f"[cyan]Merging translated documents into Lance...[/cyan] candidates={len(merge_candidates)} batch_size={merge_batch_size}"
    )

    def flush_batch(batch_rows: list[tuple[str, str, list[int]]]) -> None:
        data = {
            "uris": pa.array([row[0] for row in batch_rows], type=pa.string()),
            "captions": pa.array([[row[1]] for row in batch_rows], type=pa.list_(pa.string())),
        }
        if include_chunk_offsets:
            data["chunk_offsets"] = pa.array([row[2] for row in batch_rows], type=pa.list_(pa.int32()))
        table = pa.table(data)
        target_ds.merge_insert(on="uris").when_matched_update_all().execute(table)

    merged_count = 0
    batch_rows: list[tuple[str, str, list[int]]] = []
    batch_index = 0

    for uri_value in merge_candidates:
        translated_markdown = current_run_translations.get(uri_value, "")
        if not translated_markdown and export_root is not None:
            translated_markdown = load_saved_translation(Path(uri_value), export_root, target_lang)
        if not translated_markdown.strip():
            continue

        translated_offsets = compute_chunk_offsets(translated_markdown, max_chars=max_chars) if include_chunk_offsets else []
        batch_rows.append((uri_value, translated_markdown, translated_offsets))

        if len(batch_rows) < merge_batch_size:
            continue

        flush_batch(batch_rows)
        merged_count += len(batch_rows)
        batch_index += 1
        if len(merge_candidates) > merge_batch_size:
            console.print(f"[cyan]Merged batch {batch_index}[/cyan]")
        batch_rows = []

    if batch_rows:
        flush_batch(batch_rows)
        merged_count += len(batch_rows)
        batch_index += 1
        if len(merge_candidates) > merge_batch_size:
            console.print(f"[cyan]Merged batch {batch_index}[/cyan]")

    if merged_count == 0:
        console.print("[yellow]No translated documents available for final merge.[/yellow]")
        return 0

    latest_ds = lance.dataset(str(dataset_path))
    update_or_create_tag(latest_ds, translation_tag)
    console.print(f"[green]Successfully updated dataset with translated captions.[/green] tag={translation_tag}")
    return merged_count


def load_or_create_dataset(
    input_path: str,
    output_name: str,
    raw_tag: str,
    force_reimport: bool = False,
) -> tuple[Path, bool]:
    source_path = Path(input_path)
    if source_path.suffix.lower() == ".lance":
        return source_path, False
    if not source_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not force_reimport:
        existing = sorted(source_path.glob('*.lance'))
        if existing:
            preferred = source_path / f"{output_name}.lance"
            if preferred in existing:
                return preferred, False
            return existing[0], False

    console.print("[yellow]Importing source assets into Lance...[/yellow]")
    dataset = transform2lance(
        dataset_dir=str(source_path),
        output_name=output_name,
        save_binary=True,
        not_save_disk=False,
        tag=raw_tag,
        include_text_assets=True,
    )
    if dataset is None:
        raise TranslationRuntimeError("Failed to create Lance dataset for translation")
    return source_path / f"{output_name}.lance", True


def normalize_dataset(
    dataset_path: Path,
    source_version: str,
    norm_tag: str,
    max_chars: int,
) -> None:
    source_ds = lance.dataset(str(dataset_path), version=source_version)
    target_schema = ensure_chunk_offsets_schema(source_ds.schema)

    def batches() -> Iterable[pa.RecordBatch]:
        with Progress(
            "[progress.description]{task.description}",
            SpinnerColumn(spinner_name="dots"),
            TextColumn('{task.completed}/{task.total}'),
            BarColumn(bar_width=32, complete_style='green', finished_style='bold green'),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task('[green]Normalizing documents...', total=source_ds.count_rows())
            row_offset = 0
            for batch in source_ds.to_batches():
                batch_field_names = set(batch.schema.names)
                uri_values = batch.column("uris").to_pylist()
                captions = batch.column("captions").to_pylist() if "captions" in batch_field_names else [[] for _ in range(len(batch))]
                chunk_values = batch.column("chunk_offsets").to_pylist() if "chunk_offsets" in batch_field_names else [[] for _ in range(len(batch))]
                blob_values = load_batch_blobs(source_ds, batch, row_offset)

                for index, uri_value in enumerate(uri_values):
                    uri = Path(uri_value)
                    asset_type = classify_translation_asset(uri)
                    if asset_type is None:
                        progress.advance(task)
                        continue

                    blob = None if uri.exists() else blob_values[index]
                    try:
                        markdown = normalize_asset(uri, blob)
                    except NormalizationError as exc:
                        console.print(f"[red]Normalize failed for {uri}: {exc}[/red]")
                        progress.advance(task)
                        continue
                    except Exception as exc:
                        console.print(f"[red]Unexpected normalize failure for {uri}: {exc}[/red]")
                        progress.advance(task)
                        continue

                    captions[index] = [markdown] if markdown else []
                    chunk_values[index] = compute_chunk_offsets(markdown, max_chars=max_chars) if markdown else []
                    progress.advance(task)

                yield build_record_batch(batch, target_schema, captions, chunk_values, blob_values)
                row_offset += len(batch)

    reader = pa.RecordBatchReader.from_batches(target_schema, batches())
    dataset = lance.write_dataset(reader, str(dataset_path), target_schema, mode="overwrite")
    update_or_create_tag(dataset, norm_tag)


def translate_dataset(
    dataset_path: Path,
    source_version: str,
    translation_tag: str,
    translator: Translator,
    source_lang: str,
    target_lang: str,
    max_chars: int,
    context_chars: int,
    glossary: str,
    export_root: Optional[Path] = None,
    merge_batch_size: int = 100,
) -> None:
    source_ds = lance.dataset(str(dataset_path), version=source_version)
    translated_count = 0
    skipped_count = 0
    merge_candidates: list[str] = []
    current_run_translations: dict[str, str] = {}

    if export_root is None:
        console.print("[yellow]File export disabled; incremental resume is unavailable in this mode.[/yellow]")
    else:
        console.print(
            f"[yellow]File-based incremental mode enabled.[/yellow] existing '*_{sanitize_lang_suffix(target_lang)}.md' files will be skipped"
        )

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        TextColumn('{task.completed}/{task.total}'),
        BarColumn(bar_width=32, complete_style='green', finished_style='bold green'),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task('[green]Translating documents...', total=source_ds.count_rows())

        for source_batch in source_ds.to_batches():
            source_fields = set(source_batch.schema.names)
            uri_values = source_batch.column("uris").to_pylist()
            source_captions = source_batch.column("captions").to_pylist() if "captions" in source_fields else [[] for _ in range(len(source_batch))]
            source_chunk_values = source_batch.column("chunk_offsets").to_pylist() if "chunk_offsets" in source_fields else [[] for _ in range(len(source_batch))]

            for index, uri_value in enumerate(uri_values):
                uri = Path(uri_value)
                asset_type = classify_translation_asset(uri)
                if asset_type is None:
                    progress.advance(task)
                    continue

                source_markdown = captions_to_text(source_captions[index])
                if not source_markdown.strip():
                    progress.advance(task)
                    continue

                merge_candidates.append(uri_value)
                existing_translation = load_saved_translation(uri, export_root, target_lang) if export_root is not None else ""
                if existing_translation.strip():
                    skipped_count += 1
                    translated_path = resolve_translated_markdown_path(uri, export_root, target_lang)
                    console.print(f"[yellow]Skipping existing translation:[/yellow] {translated_path}")
                    progress.advance(task)
                    continue

                offsets = source_chunk_values[index] or compute_chunk_offsets(source_markdown, max_chars=max_chars)
                chunks = slice_by_offsets(source_markdown, offsets)
                translated_chunks: list[str] = []
                console.print(f"[cyan]Translating {uri} ({len(chunks)} chunks)...[/cyan]")
                for chunk_index, chunk in enumerate(chunks):
                    masked_chunk, replacements = protect_markdown(chunk)
                    context = ''
                    if context_chars > 0 and chunk_index > 0:
                        context = chunks[chunk_index - 1][-context_chars:]
                    translated = translator.translate(
                        masked_chunk,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        context=context,
                        glossary=glossary,
                    )
                    translated = restore_placeholders(translated, replacements)
                    if not translated.strip():
                        translated = chunk
                    translated = preserve_chunk_whitespace(chunk, translated)
                    translated_chunks.append(translated)
                    console.print(f"[blue]Chunk {chunk_index + 1}/{len(chunks)} result:[/blue]")
                    console.print(Text(translated))

                translated_markdown = ''.join(translated_chunks)
                if translated_markdown.strip() and not translated_markdown.endswith("\n"):
                    translated_markdown += "\n"

                console.print(f"[green]Translation complete for {uri}[/green]")
                console.print(f"[blue]Translated content length:[/blue] {len(translated_markdown)}")
                console.print(Text(translated_markdown))

                translated_count += 1
                saved_to_disk = False

                if export_root is not None:
                    saved_to_disk = save_translated_markdown(uri, translated_markdown, export_root, target_lang)
                if export_root is None or not saved_to_disk:
                    current_run_translations[uri_value] = translated_markdown

                progress.advance(task)

    merged_count = merge_translations(
        dataset_path=dataset_path,
        base_version=source_version,
        translation_tag=translation_tag,
        merge_candidates=merge_candidates,
        current_run_translations=current_run_translations,
        export_root=export_root,
        target_lang=target_lang,
        max_chars=max_chars,
        merge_batch_size=merge_batch_size,
    )
    console.print(
        f"[green]Translation update complete.[/green] translated={translated_count} skipped={skipped_count} merged={merged_count} tag={translation_tag}"
    )


def load_glossary(glossary_path: Optional[str]) -> str:
    if not glossary_path:
        return ""
    return Path(glossary_path).read_text(encoding='utf-8').strip()


def sanitize_lang_suffix(value: str) -> str:
    return sanitize_tag_component(value).replace('-', '_')


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and translate text/document assets stored in Lance.")
    parser.add_argument("input_path", help="Dataset directory or .lance path")
    parser.add_argument("--output_name", default="dataset", help="Lance dataset name when importing a directory")
    parser.add_argument("--model_id", default="tencent/HY-MT1.5-7B", help="Translation model id")
    parser.add_argument(
        "--runtime_backend",
        choices=["direct", "openai"],
        default="direct",
        help="Translation execution backend: direct transformers or OpenAI-compatible local server",
    )
    parser.add_argument("--openai_base_url", default="", help="Base URL for OpenAI-compatible translation backend")
    parser.add_argument("--openai_api_key", default="", help="API key for OpenAI-compatible translation backend")
    parser.add_argument(
        "--openai_model_name",
        default="",
        help="Optional OpenAI-compatible model name override for translation backend",
    )
    parser.add_argument("--source_lang", default="auto", help="Source language code")
    parser.add_argument("--target_lang", default="zh_cn", help="Target language code used for tags and file suffixes")
    parser.add_argument("--max_chars", type=int, default=2200, help="Maximum characters per translation chunk")
    parser.add_argument("--context_chars", type=int, default=300, help="Characters of previous chunk passed as translation context")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens generated per chunk")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the translation model")
    parser.add_argument("--glossary_file", default="", help="Optional glossary text file passed to the model")
    parser.add_argument("--source_version", default="", help="Source Lance tag/version to read from")
    parser.add_argument("--raw_tag", default="", help="Optional raw import tag override")
    parser.add_argument("--norm_tag", default="", help="Optional normalized markdown tag override")
    parser.add_argument("--translation_tag", default="", help="Optional translated result tag override")
    parser.add_argument("--skip_normalize", action="store_true", help="Treat source_version as an existing normalized markdown version")
    parser.add_argument("--normalize_only", action="store_true", help="Stop after writing the normalized markdown version")
    parser.add_argument("--no_export", action="store_true", help="Skip exporting translated markdown files to disk")
    parser.add_argument("--force_reimport", action="store_true", help="Rebuild the Lance dataset from the input directory even if one already exists")
    parser.add_argument(
        "--merge_batch_size",
        type=int,
        default=100,
        help="Batch size for merge_insert to avoid memory overflow on large datasets (default: 100)",
    )
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    raw_tag = sanitize_tag_component(args.raw_tag) if args.raw_tag else build_version_tag("raw", "import")
    norm_tag = sanitize_tag_component(args.norm_tag) if args.norm_tag else build_version_tag("norm", "docling")
    default_translation_tag = build_version_tag('tr', sanitize_model_tag(args.model_id), sanitize_lang_suffix(args.target_lang))
    translation_tag = sanitize_tag_component(args.translation_tag) if args.translation_tag else default_translation_tag

    dataset_path, dataset_created = load_or_create_dataset(
        args.input_path,
        output_name=args.output_name,
        raw_tag=raw_tag,
        force_reimport=args.force_reimport,
    )

    if args.source_version:
        source_version = args.source_version
    elif dataset_created:
        source_version = raw_tag
    else:
        source_version = get_latest_version_number(lance.dataset(str(dataset_path)))

    if not args.skip_normalize:
        existing_norm = try_open_dataset(dataset_path, norm_tag)
        if existing_norm is not None:
            console.print(f"[yellow]Using existing normalized version:[/yellow] {norm_tag}")
            source_version = norm_tag
        else:
            normalize_dataset(dataset_path, source_version=source_version, norm_tag=norm_tag, max_chars=args.max_chars)
            source_version = norm_tag

    if args.normalize_only:
        console.print(f"[green]Normalization complete. Version tag:[/green] {norm_tag}")
        return

    glossary = load_glossary(args.glossary_file)
    translator = HYMTProvider(
        model_id=args.model_id,
        console=console,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        backend=args.runtime_backend,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        openai_model_name=args.openai_model_name,
    )
    export_root = None if args.no_export else (
        Path(args.input_path).parent if Path(args.input_path).suffix.lower() == '.lance' else Path(args.input_path)
    )
    translate_dataset(
        dataset_path=dataset_path,
        source_version=source_version,
        translation_tag=translation_tag,
        translator=translator,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_chars=args.max_chars,
        context_chars=args.context_chars,
        glossary=glossary,
        export_root=export_root,
        merge_batch_size=args.merge_batch_size,
    )

    console.print(f"[green]Translation pipeline completed.[/green] norm={norm_tag} tr={translation_tag}")


if __name__ == "__main__":
    main()
