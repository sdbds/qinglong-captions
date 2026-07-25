"""Page-level MuSViT OMR orchestration, exports, resume, and manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from module.music_export import ExportJob, atomic_output_path, run_export_jobs
from module.music_export.music21_writers import (
    write_midi_score,
    write_musicxml_score,
)
from module.music_export.validation import (
    validate_midi_file,
    validate_musicxml_file,
)

from .aggregate import (
    KernPageScore,
    combine_page_scores,
    normalize_page_score,
)
from .decode import (
    KernStructureError,
    diagnostic_kern_candidate,
    parse_kern_score,
    reconstruct_kern,
)
from .inputs import (
    DEFAULT_PDF_DPI,
    InputPage,
    collect_source_inputs,
    iter_source_pages,
)

SCHEMA_VERSION = 2
EXPORT_SCHEMA_VERSION = 6
AGGREGATE_SCHEMA_VERSION = 6
_FORMAT_FILES = {
    "musicxml": "score.musicxml",
    "midi": "score.mid",
}
_SYMBOLIC_OUTPUT_FILES = tuple(_FORMAT_FILES.values())
_PAGE_ARTIFACT_FILES = (
    "tokens.json",
    "score.krn",
    *_SYMBOLIC_OUTPUT_FILES,
)


@dataclass(frozen=True)
class PageResult:
    source_path: Path
    output_dir: Path
    ok: bool
    skipped: bool
    failures: dict[str, str]


@dataclass(frozen=True)
class DocumentResult:
    source_path: Path
    output_dir: Path
    ok: bool
    skipped: bool
    outputs: dict[str, str]
    failures: dict[str, str]
    warnings: dict[str, str]


@dataclass(frozen=True)
class PipelineRunResult:
    input_path: Path
    output_dir: Path
    manifest_path: Path
    pages: tuple[PageResult, ...]
    documents: tuple[DocumentResult, ...]
    source_failures: dict[str, str]
    processed: int
    skipped: int
    failed: int
    ok: bool


def parse_output_format(value: str) -> tuple[str, ...]:
    normalized = str(value).strip().lower()
    if normalized == "both":
        return ("musicxml", "midi")
    if normalized in _FORMAT_FILES:
        return (normalized,)
    raise ValueError(f"Unsupported output format {value!r}; expected musicxml, midi, or both")


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    with atomic_output_path(path) as temporary:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _write_text_atomic(path: Path, text: str) -> None:
    with atomic_output_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _score_has_notes(score: Any) -> bool:
    return any(True for _ in score.recurse().notes)


def _remove_artifacts(root: Path, filenames: Iterable[str]) -> None:
    for filename in filenames:
        (root / filename).unlink(missing_ok=True)


class MuSViTOmrPipeline:
    def __init__(
        self,
        *,
        recognizer: Any,
        output_dir: str | Path,
        output_format: str = "musicxml",
        pdf_dpi: int = DEFAULT_PDF_DPI,
        recursive: bool = True,
        skip_completed: bool = True,
        overwrite: bool = False,
        max_tokens: int | None = None,
        pdf_renderer: Callable[..., Iterable[Any]] | None = None,
        progress_callback: Callable[[InputPage, int], None] | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.output_dir = Path(output_dir).expanduser()
        self.requested_formats = parse_output_format(output_format)
        self.pdf_dpi = int(pdf_dpi)
        if self.pdf_dpi <= 0:
            raise ValueError("pdf_dpi must be positive")
        self.recursive = bool(recursive)
        self.skip_completed = bool(skip_completed)
        self.overwrite = bool(overwrite)
        model_limit = int(self.recognizer.model_config.max_length)
        self.max_tokens = model_limit if max_tokens is None else int(max_tokens)
        if self.max_tokens < 2 or self.max_tokens > model_limit:
            raise ValueError(f"max_tokens must be in 2..{model_limit}")
        self.pdf_renderer = pdf_renderer
        self.progress_callback = progress_callback

    def _source_output_dir(
        self,
        source_path: Path,
        *,
        source_root: Path | None,
    ) -> Path:
        source = source_path.resolve()
        if source_root is None:
            relative_source = Path(source.name)
        else:
            relative_source = source.relative_to(source_root.resolve())
        target = self.output_dir.resolve() / relative_source
        target = target.resolve()
        target.relative_to(self.output_dir.resolve())
        return target

    def _page_output_dir(
        self,
        page: InputPage,
        *,
        source_root: Path | None,
    ) -> Path:
        target = self._source_output_dir(
            page.source_path,
            source_root=source_root,
        )
        if page.page_number is not None:
            target = target / f"page_{page.page_number:04d}"
        target = target.resolve()
        target.relative_to(self.output_dir.resolve())
        return target

    def _signature_payload(self, page: InputPage) -> dict[str, Any]:
        stat = page.source_path.stat()
        preprocessor = self.recognizer.preprocessor_config
        bundle_hashes = dict(getattr(self.recognizer, "bundle_hashes", {}))
        return {
            "contract": "musvit-onnx-omr",
            "schema_version": SCHEMA_VERSION,
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "decode_policy": "kern-syntax-constrained-greedy-v1",
            "source": {
                "path": str(page.source_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "source_type": page.source_type,
                "page_index": page.page_index,
                "page_count": page.page_count,
            },
            "model": {
                "repo_id": self.recognizer.repo_id,
                "revision": self.recognizer.revision,
                "bundle_hashes": bundle_hashes,
            },
            "preprocessor": {
                "image_size": list(preprocessor.image_size),
                "interpolation": preprocessor.interpolation,
                "rescale_factor": preprocessor.rescale_factor,
            },
            "max_tokens": self.max_tokens,
            "pdf_dpi": self.pdf_dpi,
            "requested_formats": list(self.requested_formats),
        }

    def _can_resume(
        self,
        page_dir: Path,
        signature: str,
    ) -> bool:
        metadata_path = page_dir / "metadata.json"
        if not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("resume_signature") != signature:
                return False
            if metadata.get("kern_status") != "ok" or metadata.get("failures"):
                return False
            if not (page_dir / "tokens.json").is_file():
                return False
            if not (page_dir / "score.krn").is_file():
                return False
            score_has_notes = bool(metadata.get("score_has_notes"))
            for format_name in self.requested_formats:
                target = page_dir / _FORMAT_FILES[format_name]
                if format_name == "musicxml":
                    validate_musicxml_file(target)
                else:
                    validate_midi_file(
                        target,
                        require_note_events=score_has_notes,
                    )
        except Exception:
            return False
        return True

    def _base_metadata(
        self,
        page: InputPage,
        *,
        signature_payload: Mapping[str, Any],
        signature: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source_path": str(page.source_path),
            "source_type": page.source_type,
            "page_number": page.page_number,
            "page_count": page.page_count,
            "model_repo_id": self.recognizer.repo_id,
            "model_revision": self.recognizer.revision,
            "providers": list(self.recognizer.providers),
            "resolution": list(self.recognizer.preprocessor_config.image_size),
            "effective_max_tokens": self.max_tokens,
            "pdf_dpi": self.pdf_dpi,
            "requested_formats": list(self.requested_formats),
            "resume_signature": signature,
            "resume_signature_payload": dict(signature_payload),
            "outputs": {},
            "failures": {},
            "warnings": {},
        }
        if page.page_index is not None:
            metadata["page_index"] = page.page_index
        if page.rendered_page_size is not None:
            metadata["rendered_page_size"] = list(page.rendered_page_size)
        return metadata

    def _export_score(
        self,
        score: Any,
        page_dir: Path,
        *,
        musicxml_make_notation: bool | None = None,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        has_notes = _score_has_notes(score)
        jobs: list[ExportJob] = []
        for format_name in self.requested_formats:
            target = page_dir / _FORMAT_FILES[format_name]
            if format_name == "musicxml":
                jobs.append(
                    ExportJob(
                        format="musicxml",
                        target=target,
                        writer=partial(
                            write_musicxml_score,
                            score,
                            make_notation=musicxml_make_notation,
                        ),
                        validator=validate_musicxml_file,
                    )
                )
            else:
                jobs.append(
                    ExportJob(
                        format="midi",
                        target=target,
                        writer=partial(write_midi_score, score),
                        validator=partial(
                            validate_midi_file,
                            require_note_events=has_notes,
                        ),
                    )
                )

        outputs: dict[str, str] = {}
        failures: dict[str, str] = {}
        warnings: dict[str, str] = {}
        for status in run_export_jobs(jobs):
            if status.ok:
                outputs[status.format] = status.target.name
                if status.warning:
                    warnings[status.format] = status.warning
            else:
                failures[status.format] = status.error or "unknown export failure"
        return outputs, failures, warnings

    def _document_signature_payload(
        self,
        source_path: Path,
        page_metadata: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        stat = source_path.stat()
        return {
            "contract": "musvit-onnx-omr-pdf-aggregate",
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "source": {
                "path": str(source_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            "page_signatures": [str(metadata.get("resume_signature", "")) for metadata in page_metadata],
            "requested_formats": list(self.requested_formats),
        }

    def _can_resume_document(
        self,
        document_dir: Path,
        signature: str,
    ) -> bool:
        metadata_path = document_dir / "metadata.json"
        if not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("resume_signature") != signature:
                return False
            if metadata.get("status") != "ok" or metadata.get("failures"):
                return False
            score_has_notes = bool(metadata.get("score_has_notes"))
            for format_name in self.requested_formats:
                target = document_dir / _FORMAT_FILES[format_name]
                if format_name == "musicxml":
                    validate_musicxml_file(target)
                else:
                    validate_midi_file(
                        target,
                        require_note_events=score_has_notes,
                    )
        except Exception:
            return False
        return True

    def _aggregate_pdf(
        self,
        source_path: Path,
        *,
        source_root: Path | None,
        page_results: Sequence[PageResult],
        page_metadata: Sequence[Mapping[str, Any]],
        source_error: str | None = None,
    ) -> DocumentResult:
        document_dir = self._source_output_dir(
            source_path,
            source_root=source_root,
        )
        document_dir.mkdir(parents=True, exist_ok=True)
        signature_payload = self._document_signature_payload(
            source_path,
            page_metadata,
        )
        signature = _canonical_digest(signature_payload)
        metadata: dict[str, Any] = {
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "source_path": str(source_path),
            "source_type": "pdf",
            "page_count": len(page_results),
            "requested_formats": list(self.requested_formats),
            "resume_signature": signature,
            "resume_signature_payload": signature_payload,
            "outputs": {},
            "failures": {},
            "warnings": {},
        }
        failures: dict[str, str] = metadata["failures"]
        warnings: dict[str, str] = metadata["warnings"]

        failed_pages = sum(not result.ok for result in page_results)
        if source_error is not None:
            failures["source"] = source_error
        if failed_pages:
            failures["pages"] = f"{failed_pages} of {len(page_results)} PDF pages failed"
        if not page_results and source_error is None:
            failures["pages"] = "PDF produced no pages"
        if failures:
            _remove_artifacts(document_dir, _SYMBOLIC_OUTPUT_FILES)
            metadata["status"] = "blocked"
            _write_json_atomic(document_dir / "metadata.json", metadata)
            return DocumentResult(
                source_path=source_path,
                output_dir=document_dir,
                ok=False,
                skipped=False,
                outputs={},
                failures=dict(failures),
                warnings={},
            )

        if self.skip_completed and not self.overwrite and self._can_resume_document(document_dir, signature):
            resumed = json.loads((document_dir / "metadata.json").read_text(encoding="utf-8"))
            return DocumentResult(
                source_path=source_path,
                output_dir=document_dir,
                ok=True,
                skipped=True,
                outputs=dict(resumed.get("outputs", {})),
                failures={},
                warnings=dict(resumed.get("warnings", {})),
            )

        _remove_artifacts(document_dir, _SYMBOLIC_OUTPUT_FILES)
        try:
            pages = []
            for result in page_results:
                kern_text = (result.output_dir / "score.krn").read_text(encoding="utf-8")
                pages.append(
                    KernPageScore(
                        score=parse_kern_score(kern_text),
                        kern_text=kern_text,
                    )
                )
            aggregate = combine_page_scores(tuple(pages))
        except Exception as exc:
            failures["aggregate"] = _error_text(exc)
            metadata["status"] = "failed"
        else:
            metadata["padding"] = {
                "count": aggregate.padding_count,
                "quarter_length": float(aggregate.padding_quarter_length),
            }
            if aggregate.padding_count:
                warnings["alignment"] = (
                    "inserted hidden alignment padding at "
                    f"{aggregate.padding_count} measure positions totaling "
                    f"{float(aggregate.padding_quarter_length):g} "
                    "quarter lengths"
                )
            metadata["score_has_notes"] = _score_has_notes(aggregate.score)
            outputs, export_failures, export_warnings = self._export_score(
                aggregate.score,
                document_dir,
                musicxml_make_notation=False,
            )
            failures.update(export_failures)
            warnings.update(export_warnings)
            if failures:
                _remove_artifacts(document_dir, _SYMBOLIC_OUTPUT_FILES)
                metadata["status"] = "failed"
            else:
                metadata["outputs"] = outputs
                metadata["status"] = "ok"

        _write_json_atomic(document_dir / "metadata.json", metadata)
        ok = not failures and all(format_name in metadata["outputs"] for format_name in self.requested_formats)
        return DocumentResult(
            source_path=source_path,
            output_dir=document_dir,
            ok=ok,
            skipped=False,
            outputs=dict(metadata["outputs"]),
            failures=dict(failures),
            warnings=dict(warnings),
        )

    def _process_page(
        self,
        page: InputPage,
        *,
        source_root: Path | None,
    ) -> tuple[PageResult, dict[str, Any]]:
        page_dir = self._page_output_dir(page, source_root=source_root)
        signature_payload = self._signature_payload(page)
        signature = _canonical_digest(signature_payload)
        if self.skip_completed and not self.overwrite and self._can_resume(page_dir, signature):
            metadata = json.loads((page_dir / "metadata.json").read_text(encoding="utf-8"))
            return (
                PageResult(
                    source_path=page.source_path,
                    output_dir=page_dir,
                    ok=True,
                    skipped=True,
                    failures={},
                ),
                metadata,
            )

        page_dir.mkdir(parents=True, exist_ok=True)
        _remove_artifacts(page_dir, _PAGE_ARTIFACT_FILES)
        metadata = self._base_metadata(
            page,
            signature_payload=signature_payload,
            signature=signature,
        )
        failures: dict[str, str] = metadata["failures"]
        image_input: Any = page.image if page.image is not None else page.source_path
        progress_callback = self.progress_callback
        callback = None
        if progress_callback is not None:

            def callback(count: int) -> None:
                progress_callback(page, count)

        try:
            generation = self.recognizer.generate(
                image_input,
                max_tokens=self.max_tokens,
                progress_callback=callback,
            )
        except Exception as exc:
            failures["generation"] = _error_text(exc)
            metadata["kern_status"] = "not_generated"
            _write_json_atomic(page_dir / "metadata.json", metadata)
            return (
                PageResult(
                    source_path=page.source_path,
                    output_dir=page_dir,
                    ok=False,
                    skipped=False,
                    failures=dict(failures),
                ),
                metadata,
            )

        tokens_payload = {
            "token_ids": list(generation.token_ids),
            "tokens": list(generation.tokens),
            "terminated_by_eos": bool(generation.terminated_by_eos),
            "truncated": bool(generation.truncated),
        }
        _write_json_atomic(page_dir / "tokens.json", tokens_payload)
        metadata.update(
            {
                "generated_token_count": len(generation.token_ids),
                "terminated_by_eos": bool(generation.terminated_by_eos),
                "truncated": bool(generation.truncated),
                "elapsed_seconds": float(generation.elapsed_seconds),
            }
        )

        candidate = diagnostic_kern_candidate(generation.tokens)
        if generation.truncated:
            _write_text_atomic(page_dir / "score.krn", candidate)
            metadata["kern_status"] = "truncated"
            failures["generation"] = f"generation reached max_tokens={self.max_tokens} without EOS"
        else:
            try:
                kern_text = reconstruct_kern(generation.tokens)
            except KernStructureError as exc:
                diagnostic = exc.candidate or candidate
                _write_text_atomic(page_dir / "score.krn", diagnostic)
                metadata["kern_status"] = "invalid"
                failures["kern"] = _error_text(exc)
            else:
                _write_text_atomic(page_dir / "score.krn", kern_text)
                metadata["kern_status"] = "ok"
                try:
                    parsed_score = parse_kern_score(kern_text)
                except Exception as exc:
                    metadata["kern_status"] = "parse_failed"
                    failures["kern"] = _error_text(exc)
                else:
                    try:
                        normalized = normalize_page_score(
                            KernPageScore(
                                score=parsed_score,
                                kern_text=kern_text,
                            )
                        )
                    except Exception as exc:
                        failures["normalization"] = _error_text(exc)
                    else:
                        metadata["padding"] = {
                            "count": normalized.padding_count,
                            "quarter_length": float(normalized.padding_quarter_length),
                        }
                        page_warnings: dict[str, str] = {}
                        if normalized.padding_count:
                            page_warnings["alignment"] = (
                                "inserted hidden alignment padding at "
                                f"{normalized.padding_count} measure "
                                "positions totaling "
                                f"{float(normalized.padding_quarter_length):g} "
                                "quarter lengths"
                            )
                        metadata["score_has_notes"] = _score_has_notes(normalized.score)
                        (
                            outputs,
                            export_failures,
                            export_warnings,
                        ) = self._export_score(
                            normalized.score,
                            page_dir,
                            musicxml_make_notation=False,
                        )
                        page_warnings.update(export_warnings)
                        metadata["outputs"] = outputs
                        metadata["warnings"] = page_warnings
                        failures.update(export_failures)

        _write_json_atomic(page_dir / "metadata.json", metadata)
        ok = not failures and all(format_name in metadata["outputs"] for format_name in self.requested_formats)
        return (
            PageResult(
                source_path=page.source_path,
                output_dir=page_dir,
                ok=ok,
                skipped=False,
                failures=dict(failures),
            ),
            metadata,
        )

    def run(self, input_path: str | Path) -> PipelineRunResult:
        source_input = Path(input_path).expanduser()
        sources = collect_source_inputs(
            source_input,
            recursive=self.recursive,
        )
        source_root = source_input if source_input.is_dir() else None
        pages: list[PageResult] = []
        documents: list[DocumentResult] = []
        manifest_pages: list[dict[str, Any]] = []
        source_failures: dict[str, str] = {}

        for source_path in sources:
            source_page_results: list[PageResult] = []
            source_page_metadata: list[Mapping[str, Any]] = []
            source_error: str | None = None
            try:
                for page in iter_source_pages(
                    source_path,
                    pdf_dpi=self.pdf_dpi,
                    pdf_renderer=self.pdf_renderer,
                ):
                    page_result, metadata = self._process_page(
                        page,
                        source_root=source_root,
                    )
                    pages.append(page_result)
                    source_page_results.append(page_result)
                    source_page_metadata.append(metadata)
                    manifest_pages.append(
                        {
                            "source_path": str(page_result.source_path),
                            "output_dir": str(page_result.output_dir),
                            "ok": page_result.ok,
                            "skipped": page_result.skipped,
                            "failures": page_result.failures,
                            "page_number": metadata.get("page_number"),
                        }
                    )
            except Exception as exc:
                source_error = _error_text(exc)
                source_failures[str(source_path)] = source_error

            if source_path.suffix.lower() == ".pdf":
                documents.append(
                    self._aggregate_pdf(
                        source_path,
                        source_root=source_root,
                        page_results=source_page_results,
                        page_metadata=source_page_metadata,
                        source_error=source_error,
                    )
                )

        if not sources:
            source_failures[str(source_input)] = "No supported image or PDF files found"

        processed = sum(not page.skipped for page in pages)
        skipped = sum(page.skipped for page in pages)
        failed = sum(not page.ok for page in pages) + len(source_failures)
        for document in documents:
            if document.ok:
                continue
            has_failed_page = any(page.source_path.resolve() == document.source_path.resolve() and not page.ok for page in pages)
            has_source_failure = str(document.source_path) in source_failures
            if not has_failed_page and not has_source_failure:
                failed += 1
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "input_path": str(source_input),
            "output_dir": str(self.output_dir),
            "model_repo_id": self.recognizer.repo_id,
            "model_revision": self.recognizer.revision,
            "requested_formats": list(self.requested_formats),
            "pages": manifest_pages,
            "documents": [
                {
                    "source_path": str(document.source_path),
                    "output_dir": str(document.output_dir),
                    "ok": document.ok,
                    "skipped": document.skipped,
                    "outputs": document.outputs,
                    "failures": document.failures,
                    "warnings": document.warnings,
                }
                for document in documents
            ],
            "source_failures": source_failures,
            "summary": {
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        ok = bool(pages) and failed == 0
        return PipelineRunResult(
            input_path=source_input,
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            pages=tuple(pages),
            documents=tuple(documents),
            source_failures=source_failures,
            processed=processed,
            skipped=skipped,
            failed=failed,
            ok=ok,
        )
