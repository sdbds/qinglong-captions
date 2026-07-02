from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
from rich.console import Console
from rich.progress import Progress

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config
from module.onnx_runtime import (
    OnnxModelSpec,
    OnnxRuntimeConfig,
    build_local_model_dir,
    load_single_model_bundle,
    resolve_tool_runtime_config,
)
from utils.console_util import print_exception

DEFAULT_MUSVIT_ONNX_REPO_ID = "bdsqlsz/musvit-onnx"
DEFAULT_MUSVIT_MODEL_DIR = "huggingface"
MODEL_FILENAME = "model.onnx"
PREPROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_PDF_DPI = 144
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
SUPPORTED_PDF_EXTENSIONS = (".pdf",)
SUPPORTED_INPUT_EXTENSIONS = (*SUPPORTED_IMAGE_EXTENSIONS, *SUPPORTED_PDF_EXTENSIONS)
SUPPORTED_PREPROCESS_MODES = ("page_resize", "pad_square")

console = Console(color_system="truecolor", force_terminal=True)


@dataclass(frozen=True)
class MuSViTPreprocessorConfig:
    image_size: tuple[int, int]
    do_resize: bool
    do_rescale: bool
    rescale_factor: float
    do_normalize: bool
    image_mean: tuple[float, float, float] | None = None
    image_std: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class MuSViTEmbeddingResult:
    source_path: Path
    output_dir: Path
    embedding_path: Path
    metadata_path: Path
    skipped: bool = False


@dataclass(frozen=True)
class MuSViTInputPage:
    source_path: Path
    source_type: str = "image"
    page_index: int | None = None
    page_count: int | None = None
    image: Image.Image | None = None
    rendered_page_size: tuple[int, int] | None = None

    @property
    def page_number(self) -> int | None:
        if self.page_index is None:
            return None
        return self.page_index + 1


def _as_size_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        width = int(value.get("width", value.get("shortest_edge", 1024)))
        height = int(value.get("height", value.get("shortest_edge", width)))
        return width, height
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if len(items) >= 2:
            return int(items[0]), int(items[1])
        if len(items) == 1:
            return int(items[0]), int(items[0])
    if value:
        parsed = int(value)
        return parsed, parsed
    return 1024, 1024


def _float_triplet(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [float(item) for item in value]
        if len(items) == 3:
            return items[0], items[1], items[2]
    return None


def load_musvit_preprocessor_config(path: str | Path) -> MuSViTPreprocessorConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    width, height = _as_size_tuple(payload.get("size"))
    return MuSViTPreprocessorConfig(
        image_size=(width, height),
        do_resize=bool(payload.get("do_resize", True)),
        do_rescale=bool(payload.get("do_rescale", True)),
        rescale_factor=float(payload.get("rescale_factor", 1 / 255)),
        do_normalize=bool(payload.get("do_normalize", False)),
        image_mean=_float_triplet(payload.get("image_mean")),
        image_std=_float_triplet(payload.get("image_std")),
    )


def preprocess_pil_image(
    image: Image.Image,
    config: MuSViTPreprocessorConfig,
    *,
    preprocess_mode: str = "page_resize",
) -> np.ndarray:
    if preprocess_mode not in SUPPORTED_PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode!r}")

    image = image.convert("RGB")
    if preprocess_mode == "pad_square":
        image = ImageOps.pad(image, config.image_size, color=(255, 255, 255), method=Image.Resampling.BICUBIC)
    elif config.do_resize:
        image = image.resize(config.image_size, Image.Resampling.BICUBIC)

    array = np.asarray(image, dtype=np.float32)

    if config.do_rescale:
        array *= float(config.rescale_factor)

    if config.do_normalize and config.image_mean is not None and config.image_std is not None:
        mean = np.asarray(config.image_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(config.image_std, dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean) / std

    return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def preprocess_image(
    image_path: str | Path,
    config: MuSViTPreprocessorConfig,
    *,
    preprocess_mode: str = "page_resize",
) -> np.ndarray:
    with Image.open(image_path) as image:
        return preprocess_pil_image(image, config, preprocess_mode=preprocess_mode)


def collect_image_inputs(input_path: Path, *, recursive: bool = True) -> list[Path]:
    input_path = Path(input_path).expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)


def collect_musvit_source_inputs(input_path: Path, *, recursive: bool = True) -> list[Path]:
    input_path = Path(input_path).expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            raise ValueError(f"Unsupported input file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS)


def _default_pdf_renderer(pdf_path: str | Path, *, dpi: int, image_format: str) -> Iterable[Any]:
    try:
        from utils.stream_util import iter_pdf_pages_high_quality
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF input requires PyMuPDF. Install the musvit-onnx extra before scanning PDFs.") from exc

    try:
        yield from iter_pdf_pages_high_quality(pdf_path, dpi=dpi, image_format=image_format)
    except ModuleNotFoundError as exc:
        if exc.name == "fitz":
            raise RuntimeError("PDF input requires PyMuPDF. Install the musvit-onnx extra before scanning PDFs.") from exc
        raise


def _iter_pdf_pages(
    pdf_path: str | Path,
    *,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    pdf_renderer: Callable[..., Iterable[Any]] | None = None,
) -> Iterable[MuSViTInputPage]:
    pdf_path = Path(pdf_path).expanduser()
    renderer = pdf_renderer or _default_pdf_renderer
    rendered_count = 0
    for fallback_index, rendered_page in enumerate(renderer(pdf_path, dpi=int(pdf_dpi), image_format="PNG")):
        rendered_count += 1
        image = getattr(rendered_page, "image", rendered_page)
        page_index = int(getattr(rendered_page, "page_index", fallback_index))
        page_count = getattr(rendered_page, "page_count", None)
        rendered_size = getattr(rendered_page, "size", image.size)
        yield MuSViTInputPage(
            source_path=Path(getattr(rendered_page, "pdf_path", pdf_path)).expanduser(),
            source_type="pdf_page",
            page_index=page_index,
            page_count=int(page_count) if page_count is not None else None,
            image=image,
            rendered_page_size=rendered_size,
        )
    if rendered_count == 0:
        raise ValueError(f"PDF contains no renderable pages: {pdf_path}")


def iter_musvit_input_pages(
    input_path: Path,
    *,
    recursive: bool = True,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    pdf_renderer: Callable[..., Iterable[Any]] | None = None,
) -> Iterable[MuSViTInputPage]:
    for source_path in collect_musvit_source_inputs(input_path, recursive=recursive):
        if source_path.suffix.lower() in SUPPORTED_PDF_EXTENSIONS:
            yield from _iter_pdf_pages(source_path, pdf_dpi=pdf_dpi, pdf_renderer=pdf_renderer)
        else:
            yield MuSViTInputPage(source_path=source_path, source_type="image")


def _batched_pages(pages: Iterable[MuSViTInputPage], batch_size: int) -> Iterable[list[MuSViTInputPage]]:
    batch: list[MuSViTInputPage] = []
    for page in pages:
        batch.append(page)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _close_pdf_page_images(pages: Sequence[MuSViTInputPage]) -> None:
    for page in pages:
        if page.source_type == "pdf_page" and page.image is not None:
            try:
                page.image.close()
            except Exception:
                pass


def resolve_musvit_model_dir(model_dir: str | Path, repo_id: str) -> Path:
    candidate = Path(model_dir).expanduser()
    if (candidate / MODEL_FILENAME).exists():
        return candidate
    return build_local_model_dir(candidate, repo_id)


def resolve_musvit_runtime_config(
    *,
    force_download: bool = False,
    config_dir: str | Path = CONFIG_DIR,
) -> OnnxRuntimeConfig:
    config = load_config(str(config_dir))
    return resolve_tool_runtime_config(
        config,
        tool_name="musvit",
        cli_override={"force_download": force_download},
    )


def _relative_output_dir(output_dir: Path, source_path: Path, source_root: Path | None) -> Path:
    output_dir = Path(output_dir).resolve()
    source_path = Path(source_path).resolve()
    if source_root is None:
        target = output_dir / source_path.name
    else:
        target = output_dir / source_path.relative_to(Path(source_root).resolve())
    target = target.resolve()
    try:
        target.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside output directory: {source_path}") from exc
    return target


def _read_image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


class MuSViTOnnxEmbedder:
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_MUSVIT_ONNX_REPO_ID,
        model_dir: str | Path = DEFAULT_MUSVIT_MODEL_DIR,
        force_download: bool = False,
        runtime_config: OnnxRuntimeConfig | None = None,
        config_dir: str | Path = CONFIG_DIR,
        bundle_loader: Callable[..., Any] = load_single_model_bundle,
        artifact_loader: Callable[..., Path] | None = None,
        support_file_loader: Callable[..., dict[str, Path]] | None = None,
        session_bundle_loader: Callable[..., Any] | None = None,
        logger: Callable[..., Any] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.model_dir = resolve_musvit_model_dir(model_dir, repo_id)
        self.runtime_config = runtime_config or resolve_musvit_runtime_config(
            force_download=force_download,
            config_dir=config_dir,
        )
        self.logger = logger
        self.spec = OnnxModelSpec(
            repo_id=repo_id,
            onnx_filename=MODEL_FILENAME,
            local_dir=self.model_dir,
            bundle_key=f"musvit:{repo_id}",
            support_files={"preprocessor_config": PREPROCESSOR_CONFIG_FILENAME},
        )
        self.bundle = bundle_loader(
            spec=self.spec,
            runtime_config=self.runtime_config,
            artifact_loader=artifact_loader,
            support_file_loader=support_file_loader,
            session_bundle_loader=session_bundle_loader,
            logger=logger,
        )
        self.model_path = Path(self.bundle.model_path)
        self.preprocessor_config_path = Path(self.bundle.support_paths["preprocessor_config"])
        self.preprocessor_config = load_musvit_preprocessor_config(self.preprocessor_config_path)
        self.session = self.bundle.session
        self.providers = tuple(self.bundle.providers)
        self.input_name = self.bundle.input_metas[0].name if self.bundle.input_metas else "pixel_values"
        outputs = tuple(self.session.get_outputs()) if hasattr(self.session, "get_outputs") else ()
        self.output_name = outputs[0].name if outputs else "last_hidden_state"

    def _output_paths(
        self,
        source_path: str | Path,
        *,
        output_dir: str | Path,
        source_root: str | Path | None = None,
        page_number: int | None = None,
    ) -> tuple[Path, Path, Path]:
        source_path = Path(source_path).expanduser()
        root = Path(source_root).expanduser() if source_root is not None else None
        item_dir = _relative_output_dir(Path(output_dir).expanduser(), source_path, root)
        if page_number is not None:
            item_dir = item_dir / f"page_{page_number:04d}"
        return item_dir, item_dir / "embedding.npz", item_dir / "metadata.json"

    @staticmethod
    def _coerce_input_page(value: str | Path | MuSViTInputPage) -> MuSViTInputPage:
        if isinstance(value, MuSViTInputPage):
            return value
        return MuSViTInputPage(source_path=Path(value).expanduser(), source_type="image")

    def _preprocess_input_page(self, page: MuSViTInputPage, *, preprocess_mode: str) -> np.ndarray:
        if page.image is not None:
            return preprocess_pil_image(page.image, self.preprocessor_config, preprocess_mode=preprocess_mode)
        return preprocess_image(page.source_path, self.preprocessor_config, preprocess_mode=preprocess_mode)

    @staticmethod
    def _source_image_size(page: MuSViTInputPage) -> tuple[int, int]:
        if page.rendered_page_size is not None:
            return page.rendered_page_size
        return _read_image_size(page.source_path)

    def _write_embedding_result(
        self,
        *,
        page: MuSViTInputPage,
        item_dir: Path,
        embedding_path: Path,
        metadata_path: Path,
        hidden: np.ndarray,
        preprocess_mode: str,
        elapsed: float,
    ) -> MuSViTEmbeddingResult:
        item_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            embedding_path,
            last_hidden_state=hidden,
            cls_embedding=hidden[0],
            patch_embeddings=hidden[1:],
        )
        metadata = {
            "source_path": str(page.source_path),
            "source_type": page.source_type,
            "source_image_size": list(self._source_image_size(page)),
            "output_dir": str(item_dir),
            "embedding_path": str(embedding_path),
            "model_path": str(self.model_path),
            "preprocessor_config_path": str(self.preprocessor_config_path),
            "repo_id": self.repo_id,
            "providers": list(self.providers),
            "input_name": self.input_name,
            "output_name": self.output_name,
            "last_hidden_state_shape": list(hidden.shape),
            "preprocess_mode": preprocess_mode,
            "elapsed_seconds": elapsed,
        }
        if page.source_type == "pdf_page":
            metadata.update(
                {
                    "pdf_page_index": page.page_index,
                    "pdf_page_number": page.page_number,
                    "pdf_page_count": page.page_count,
                    "rendered_page_size": list(page.rendered_page_size or self._source_image_size(page)),
                }
            )
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return MuSViTEmbeddingResult(page.source_path, item_dir, embedding_path, metadata_path)

    def embed_file(
        self,
        source_path: str | Path,
        *,
        output_dir: str | Path,
        source_root: str | Path | None = None,
        preprocess_mode: str = "page_resize",
        overwrite: bool = False,
        skip_completed: bool = True,
    ) -> MuSViTEmbeddingResult:
        source_path = Path(source_path).expanduser()
        page = MuSViTInputPage(source_path=source_path, source_type="image")
        item_dir, embedding_path, metadata_path = self._output_paths(
            source_path,
            output_dir=output_dir,
            source_root=source_root,
        )

        if skip_completed and not overwrite and embedding_path.exists() and metadata_path.exists():
            return MuSViTEmbeddingResult(source_path, item_dir, embedding_path, metadata_path, skipped=True)

        start_time = time.perf_counter()
        pixels = self._preprocess_input_page(page, preprocess_mode=preprocess_mode)
        feed = {self.input_name: np.expand_dims(pixels, axis=0)}
        output = self.session.run([self.output_name], feed)[0]
        hidden = np.asarray(output[0], dtype=np.float32)
        elapsed = time.perf_counter() - start_time
        return self._write_embedding_result(
            page=page,
            item_dir=item_dir,
            embedding_path=embedding_path,
            metadata_path=metadata_path,
            hidden=hidden,
            preprocess_mode=preprocess_mode,
            elapsed=elapsed,
        )

    def embed_batch(
        self,
        image_paths: Sequence[str | Path | MuSViTInputPage],
        *,
        output_dir: str | Path,
        source_root: str | Path | None = None,
        preprocess_mode: str = "page_resize",
        overwrite: bool = False,
        skip_completed: bool = True,
    ) -> list[MuSViTEmbeddingResult]:
        pending: list[tuple[MuSViTInputPage, Path, Path, Path]] = []
        results: list[MuSViTEmbeddingResult] = []
        for image_path in image_paths:
            page = self._coerce_input_page(image_path)
            item_dir, embedding_path, metadata_path = self._output_paths(
                page.source_path,
                output_dir=output_dir,
                source_root=source_root,
                page_number=page.page_number,
            )
            if skip_completed and not overwrite and embedding_path.exists() and metadata_path.exists():
                results.append(MuSViTEmbeddingResult(page.source_path, item_dir, embedding_path, metadata_path, skipped=True))
                continue
            pending.append((page, item_dir, embedding_path, metadata_path))

        if not pending:
            return results

        start_time = time.perf_counter()
        batch = np.stack(
            [
                self._preprocess_input_page(page, preprocess_mode=preprocess_mode)
                for page, _item_dir, _embedding_path, _metadata_path in pending
            ],
            axis=0,
        )
        outputs = np.asarray(self.session.run([self.output_name], {self.input_name: batch})[0], dtype=np.float32)
        elapsed = time.perf_counter() - start_time
        if outputs.shape[0] != len(pending):
            raise ValueError(f"MuSViT output batch mismatch: expected {len(pending)}, got {outputs.shape[0]}")

        per_item_elapsed = elapsed / max(len(pending), 1)
        for index, (page, item_dir, embedding_path, metadata_path) in enumerate(pending):
            results.append(
                self._write_embedding_result(
                    page=page,
                    item_dir=item_dir,
                    embedding_path=embedding_path,
                    metadata_path=metadata_path,
                    hidden=outputs[index],
                    preprocess_mode=preprocess_mode,
                    elapsed=per_item_elapsed,
                )
            )
        return results

    def embed_inputs(
        self,
        input_path: str | Path,
        *,
        output_dir: str | Path,
        batch_size: int = 1,
        recursive: bool = True,
        pdf_dpi: int = DEFAULT_PDF_DPI,
        pdf_renderer: Callable[..., Iterable[Any]] | None = None,
        preprocess_mode: str = "page_resize",
        overwrite: bool = False,
        skip_completed: bool = True,
    ) -> list[MuSViTEmbeddingResult]:
        input_path = Path(input_path).expanduser()
        input_pages = iter_musvit_input_pages(
            input_path,
            recursive=recursive,
            pdf_dpi=pdf_dpi,
            pdf_renderer=pdf_renderer,
        )
        source_root = input_path if input_path.is_dir() else None
        batch_size = max(1, int(batch_size))
        results: list[MuSViTEmbeddingResult] = []
        for batch_pages in _batched_pages(input_pages, batch_size):
            try:
                results.extend(
                    self.embed_batch(
                        batch_pages,
                        output_dir=output_dir,
                        source_root=source_root,
                        preprocess_mode=preprocess_mode,
                        overwrite=overwrite,
                        skip_completed=skip_completed,
                    )
                )
            finally:
                _close_pdf_page_images(batch_pages)
        return results

    def embed_pdf(
        self,
        pdf_path: str | Path,
        *,
        output_dir: str | Path,
        batch_size: int = 1,
        pdf_dpi: int = DEFAULT_PDF_DPI,
        pdf_renderer: Callable[..., Iterable[Any]] | None = None,
        preprocess_mode: str = "page_resize",
        overwrite: bool = False,
        skip_completed: bool = True,
    ) -> list[MuSViTEmbeddingResult]:
        pdf_path = Path(pdf_path).expanduser()
        batch_size = max(1, int(batch_size))
        results: list[MuSViTEmbeddingResult] = []
        for batch_pages in _batched_pages(_iter_pdf_pages(pdf_path, pdf_dpi=pdf_dpi, pdf_renderer=pdf_renderer), batch_size):
            try:
                results.extend(
                    self.embed_batch(
                        batch_pages,
                        output_dir=output_dir,
                        preprocess_mode=preprocess_mode,
                        overwrite=overwrite,
                        skip_completed=skip_completed,
                    )
                )
            finally:
                _close_pdf_page_images(batch_pages)
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MuSViT ONNX sheet-music page embeddings.")
    parser.add_argument("input_path", help="Input image file, PDF file, or directory")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--repo_id", default=DEFAULT_MUSVIT_ONNX_REPO_ID)
    parser.add_argument("--model_dir", default=DEFAULT_MUSVIT_MODEL_DIR)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pdf_dpi", type=int, default=DEFAULT_PDF_DPI)
    parser.add_argument("--preprocess_mode", choices=SUPPORTED_PREPROCESS_MODES, default="page_resize")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    return parser


def run_sheet_music_musvit(args: argparse.Namespace) -> int:
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        console.print(f"[red]Input path does not exist:[/red] {input_path}")
        return 1

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        input_path.parent / f"{input_path.stem}_musvit" if input_path.is_file() else input_path / "musvit_embeddings"
    )
    embedder = MuSViTOnnxEmbedder(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        force_download=bool(args.force_download),
        logger=console.print,
    )
    console.print("[cyan]MuSViT ONNX providers:[/cyan] " + ", ".join(str(provider) for provider in embedder.providers))
    input_pages = iter_musvit_input_pages(input_path, recursive=bool(args.recursive), pdf_dpi=int(args.pdf_dpi))

    processed = 0
    skipped = 0
    failed = 0
    seen_pages = False
    source_root = input_path if input_path.is_dir() else None
    batch_size = max(1, int(args.batch_size))
    with Progress(console=console, transient=False) as progress:
        task = progress.add_task("[bold cyan]Extracting MuSViT embeddings...", total=None)
        for batch_pages in _batched_pages(input_pages, batch_size):
            seen_pages = True
            try:
                results = embedder.embed_batch(
                    batch_pages,
                    output_dir=output_dir,
                    source_root=source_root,
                    preprocess_mode=args.preprocess_mode,
                    overwrite=bool(args.overwrite),
                    skip_completed=bool(args.skip_completed),
                )
                for result in results:
                    if result.skipped:
                        skipped += 1
                    else:
                        processed += 1
            except Exception as exc:
                failed += len(batch_pages)
                print_exception(
                    progress.console,
                    exc,
                    prefix=f"Failed MuSViT embedding batch starting at {batch_pages[0].source_path}",
                )
            finally:
                _close_pdf_page_images(batch_pages)
                progress.update(task, advance=len(batch_pages))

    if not seen_pages:
        console.print(f"[yellow]No supported image or PDF files found under:[/yellow] {input_path}")
        return 1

    manifest_path = Path(output_dir) / "manifest.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "pdf_dpi": int(args.pdf_dpi),
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"[bold]Finished.[/bold] processed={processed} skipped={skipped} failed={failed}")
    console.print(f"[green]Manifest:[/green] {manifest_path}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_sheet_music_musvit(args)


if __name__ == "__main__":
    sys.exit(main())
