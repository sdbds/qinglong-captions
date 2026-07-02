# Streaming PDF Rendering Design

## Context

The project currently has one shared PDF rasterization helper:

```text
utils.stream_util.pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG")
```

It opens a PDF with PyMuPDF, renders every page, appends each PIL image to a list, closes the document, and returns the full list. That is simple, but it makes peak memory proportional to the rendered size of the whole PDF. The new MuSViT sheet-music scanner inherited this behavior, so a long score PDF can render all pages before any ONNX inference starts.

The target change is to replace the shared eager PDF rasterization helper with a streaming PDF page interface and migrate project-owned callers that use the shared helper. Do not keep a compatibility wrapper. Do not change OCR providers or integrations that use their own PDF pipeline, SDK upload flow, or upstream page loader.

## Review Findings

### Shared Helper

- `utils/stream_util.py`
  - `pdf_to_images_high_quality` eagerly materializes all pages and should be removed.
  - It has no page metadata object, so callers that need page count call `len(images)`, forcing eager rendering.
  - It does not provide a way for callers to close page images after each page is processed.

### Project-Owned Callers Using Shared Helper

These should be migrated to the streaming interface:

- `module/sheet_music_musvit.py`
  - `_default_pdf_renderer` wraps the shared helper in `list(...)`.
  - `_expand_pdf_pages` creates a full list of `MuSViTInputPage` objects with PIL images before batching starts.
  - `collect_musvit_input_pages` returns a list, so directory input with PDFs still front-loads all rendered pages.

- `module/providers/ocr_base.py`
  - `attempt_via_openai_backend` renders all PDF pages before calling the OpenAI-compatible backend page by page.

- Local OCR providers that render pages through the shared helper:
  - `module/providers/ocr/deepseek.py`
  - `module/providers/ocr/chandra.py`
  - `module/providers/ocr/firered.py`
  - `module/providers/ocr/hunyuan.py`
  - `module/providers/ocr/glm.py`
  - `module/providers/ocr/infinity_parser2.py`
  - `module/providers/ocr/lighton.py`
  - `module/providers/ocr/logics.py`
  - `module/providers/ocr/nanonets.py`
  - `module/providers/ocr/olmocr.py`
  - `module/providers/ocr/qianfan.py`
  - `module/providers/ocr/unlimited.py`

### Explicitly Out Of Scope

Do not migrate these in this spec:

- `module/providers/ocr/dots.py`
  - It uses the upstream dots OCR PDF loader and preprocessing path.

- `module/providers/ocr/paddle.py`
  - It delegates PDF handling to PaddleOCR's pipeline and restructures provider-native page results.

- `module/providers/vision_api/pixtral.py`
  - It uploads PDFs to the Mistral OCR API and consumes API page objects.

These paths may be reviewed later, but they are not part of the shared `utils.stream_util` helper problem.

## Goals

1. Replace the eager shared PDF render API with a first-class streaming API.
2. Remove the existing eager helper completely.
3. Migrate project-owned shared-helper callers to page-by-page consumption.
4. Preserve current output layout and markdown/page split semantics.
5. Keep memory bounded to the active page or active chunk, not the full PDF.
6. Make page metadata available without forcing eager rendering.

## Non-Goals

- No OCR algorithm changes.
- No changes to provider-native PDF pipelines.
- No new rendered-page cache by default.
- No cross-provider output format redesign.
- No change to default DPI unless a provider already uses a different configured DPI.
- No compatibility wrapper for `pdf_to_images_high_quality`.
- No mixed migration where some project-owned callers still import the eager helper.

## Proposed API

Add a page metadata object and generator to `utils.stream_util.py`. Delete `pdf_to_images_high_quality` in the same change.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass(frozen=True)
class PdfRenderPage:
    pdf_path: Path
    page_index: int
    page_count: int
    image: Image.Image
    dpi: int
    image_format: str

    @property
    def page_number(self) -> int:
        return self.page_index + 1

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


def iter_pdf_pages_high_quality(
    pdf_path: str | Path,
    dpi: int = 144,
    image_format: str = "PNG",
) -> Iterator[PdfRenderPage]:
    ...
```

The generator must:

- open the PDF lazily when iteration starts;
- read `page_count` once from the PyMuPDF document;
- render exactly one page per iteration;
- return a loaded PIL image detached from PyMuPDF buffers;
- close the PyMuPDF document in `finally`;
- include `pdf_path`, `page_index`, `page_count`, `dpi`, `image_format`, and image size through the page object;
- raise page-specific errors with the PDF path and page number when rendering fails.

All project-owned callers must import and use `iter_pdf_pages_high_quality`. After migration, `rg "pdf_to_images_high_quality" module utils tests` should return no active code references. References in historical design docs are acceptable.

## Memory Contract

The streaming API owns PDF document lifetime. The caller owns page image lifetime.

Callers should process each `PdfRenderPage.image` immediately and close it in a `finally` block when they no longer need it:

```python
for page in iter_pdf_pages_high_quality(path, dpi=144):
    try:
        process(page.image)
    finally:
        page.image.close()
```

If a caller needs to keep page data after the loop, it must explicitly copy or serialize the image. That makes memory retention visible in the caller instead of hidden in the shared helper.

## MuSViT Migration Design

MuSViT must stop expanding PDFs into a list of rendered PIL pages before inference.

Replace the current `collect_musvit_input_pages(...) -> list[MuSViTInputPage]` behavior with an iterator-based path:

```python
def iter_musvit_input_pages(...):
    for source_path in collect_musvit_source_inputs(...):
        if source_path.suffix.lower() == ".pdf":
            for pdf_page in iter_pdf_pages_high_quality(source_path, dpi=pdf_dpi, image_format="PNG"):
                yield MuSViTInputPage(
                    source_path=pdf_page.pdf_path,
                    source_type="pdf_page",
                    page_index=pdf_page.page_index,
                    page_count=pdf_page.page_count,
                    image=pdf_page.image,
                    rendered_page_size=pdf_page.size,
                )
        else:
            yield MuSViTInputPage(source_path=source_path, source_type="image")
```

Batching should consume this iterator with a small chunk helper:

```python
def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
```

After each batch, close any `MuSViTInputPage.image` that came from a PDF page. This keeps peak memory to roughly `batch_size` pages plus inference tensors.

Keep output semantics:

```text
<output_dir>/<relative_pdf_path>/page_0001/embedding.npz
<output_dir>/<relative_pdf_path>/page_0001/metadata.json
```

Metadata must continue to include:

- `source_type = "pdf_page"`
- `pdf_page_index`
- `pdf_page_number`
- `pdf_page_count`
- `rendered_page_size`

## OCR Provider Migration Design

For page-by-page OCR providers, replace:

```python
images = pdf_to_images_high_quality(path)
for idx, pil_img in enumerate(images):
    ...
```

with:

```python
for page in iter_pdf_pages_high_quality(path):
    page_index = page.page_index
    page_number = page.page_number
    pil_img = page.image
    try:
        ...
    finally:
        pil_img.close()
```

Providers that currently log `len(images)` should use `page.page_count`.

Providers that save rendered page previews should keep the same output path:

```text
<output_dir>/page_0001/page_0001.png
```

Providers that join page markdown with:

```text
<--- Page Split --->
```

must preserve that behavior.

## Unlimited-OCR Special Case

`module/providers/ocr/unlimited.py` uses `model.infer_multi(...)`, so it cannot be converted to one-page-at-a-time inference without changing model behavior.

The correct migration is:

1. Stream-render one PDF page at a time.
2. Save each page image to its existing `page_000N/page_000N.png` path.
3. Append only the saved image path string to the current chunk.
4. When `page_budget > 0` and the chunk reaches `page_budget`, run `_run_infer_multi(...)` for that chunk and clear the chunk.
5. If `page_budget <= 0`, collect page image paths only, not PIL images, then run the single existing `infer_multi` call.
6. Merge chunk outputs exactly as today.

This keeps rendered image memory bounded while preserving Unlimited-OCR's multi-page inference semantics.

## Error Handling

The streaming API should provide clearer failures than the current eager helper:

- missing PyMuPDF: raise an import error that names `PyMuPDF` / `fitz`;
- corrupt or encrypted PDF: raise with PDF path before yielding pages;
- zero-page PDF: raise `ValueError` with PDF path;
- page render failure: raise with PDF path and 1-based page number;
- caller page-processing failure: keep existing provider behavior where possible.

Directory-style callers that already continue after per-file failures should continue to do so. Single-file callers may fail the run.

## Testing Strategy

### Shared Helper Tests

Add tests for `utils.stream_util`:

- `iter_pdf_pages_high_quality` does not render pages until iterated.
- It yields page objects with `page_index`, `page_number`, `page_count`, `size`, and `image`.
- It closes the PyMuPDF document when iteration completes.
- It closes the PyMuPDF document when iteration exits due to an exception.
- `utils.stream_util` no longer exposes `pdf_to_images_high_quality`.
- Zero-page PDFs fail clearly.

Use monkeypatched fake `fitz` objects for laziness and cleanup tests. A small real PyMuPDF smoke test can be kept if the test suite already depends on `PyMuPDF` in that profile.

Add a deletion/usage guard:

- no function named `pdf_to_images_high_quality` remains in `utils.stream_util`;
- no project-owned module imports `pdf_to_images_high_quality`;
- provider tests patch `iter_pdf_pages_high_quality`, not the deleted helper.

### MuSViT Tests

Update `tests/test_sheet_music_musvit.py`:

- PDF pages are consumed through an iterator, not pre-collected into a list.
- Batch size controls the number of live PDF page images held by MuSViT.
- Page images are closed after each batch.
- Metadata remains unchanged for PDF pages.
- A two-page fake PDF still writes `page_0001` and `page_0002` outputs.

### OCR Provider Tests

For providers with existing PDF tests, update monkeypatches from `pdf_to_images_high_quality` to `iter_pdf_pages_high_quality`:

- `tests/test_infinity_parser2_ocr_provider.py`
- `tests/test_qianfan_ocr_provider.py`
- `tests/test_unlimited_ocr_provider.py`
- any provider-specific tests added during migration

Add at least one generic test that the iterator page object's `page_count` replaces `len(images)` in logging/metadata-sensitive paths.

### Regression Tests

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sheet_music_musvit.py tests\test_infinity_parser2_ocr_provider.py tests\test_qianfan_ocr_provider.py tests\test_unlimited_ocr_provider.py -q
```

Run dependency/config smoke where relevant:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_audio_separator_dependency_profiles.py tests\test_see_through_config.py -q
```

Run a real PDF smoke for MuSViT:

```powershell
.\.venv\Scripts\python.exe module\sheet_music_musvit.py path\to\score.pdf --output_dir workspace\musvit_pdf_stream_smoke --batch_size=2 --pdf_dpi=72 --overwrite
```

Acceptance for the smoke:

- one output directory per PDF page;
- each page has `embedding.npz`;
- `last_hidden_state.shape == (4097, 768)`;
- metadata records the page number and page count;
- no permanent intermediate rendered page images are written by MuSViT.

## Implementation Order

1. Add tests for the streaming helper in `tests/test_stream_util_pdf.py`.
2. Implement `PdfRenderPage` and `iter_pdf_pages_high_quality` in `utils/stream_util.py`.
3. Delete `pdf_to_images_high_quality`.
4. Migrate MuSViT to iterator-based PDF page collection and batch consumption.
5. Add/adjust MuSViT tests for streaming and image closing.
6. Migrate page-by-page OCR providers that use the shared helper.
7. Migrate Unlimited-OCR with path-only chunk accumulation.
8. Update provider tests to monkeypatch the streaming interface.
9. Add a grep-style guard or focused test proving no active code imports the deleted helper.
10. Run focused regression tests.
11. Run one real PDF smoke for MuSViT and one lightweight OCR provider if dependencies are available.

## Acceptance Criteria

- The shared streaming API exists and is documented by tests.
- The legacy `pdf_to_images_high_quality` API no longer exists in active code.
- MuSViT no longer renders all PDF pages before inference.
- Page-by-page OCR providers no longer render all PDF pages before processing page 1.
- Unlimited-OCR no longer holds all PIL page images in memory; it may still hold page image path strings for one-shot mode.
- Output directory layout and markdown split markers remain unchanged.
- Provider-native PDF pipelines are untouched.
- `rg "pdf_to_images_high_quality" module utils tests` returns no active code references.
- Focused tests pass under local `.venv`.

## Risks

### Iterator Lifetime Bugs

If callers store `PdfRenderPage.image` beyond the current loop without copying it, later cleanup may close the image too early. The migration should keep image use local and close images only after inference/save operations complete.

### Subtle Test Monkeypatch Breakage

Existing tests monkeypatch `pdf_to_images_high_quality`. Migrated providers must update tests to patch `iter_pdf_pages_high_quality` instead. There is no compatibility wrapper, so stale tests should fail loudly.

### Unlimited-OCR Semantics

Unlimited-OCR has real multi-page behavior through `infer_multi`. Do not degrade it to page-by-page OCR. Stream rendered images to disk, then batch by path.

### Provider-Specific Page Counts

Several providers use `len(images)` only for logs. After migration, use `page.page_count`. Do not compute `list(iterator)` just to get the count.

## Lower-Cost Alternative

Only migrate MuSViT and leave OCR providers on the eager helper. This would solve the immediate sheet-music issue, but it would leave the same memory problem in the rest of the project-owned PDF OCR paths. Given the shared helper is the root cause, the recommended implementation is the broader but still bounded migration described above.
