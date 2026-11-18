"""Unlimited-OCR-2 structured block output and normalized layout sidecars."""

from pathlib import Path

from PIL import Image

from .document import page_metadata, write_layout_json
from .vendor import unlimited_v2 as upstream


def process_output(raw: str, pages: list[Path], output_dir: Path, *, model_id: str) -> str:
    page_count = len(list(upstream.PAGE_SPLIT_RE.finditer(raw)))
    if page_count != len(pages):
        raise RuntimeError(f"Unlimited-OCR-2 returned {page_count} pages for {len(pages)} inputs; output is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = upstream._parse_blocks(raw)
    layout = []
    for block in blocks:
        page_index = block["page"] - 1
        if not 0 <= page_index < len(pages):
            raise ValueError("Unlimited-OCR-2 emitted an invalid page index")
        bbox = [upstream._scale(value) for value in block["bbox"]] if block["bbox"] else None
        layout.append(
            {
                "page_index": page_index,
                "label": block["label"],
                "bbox": bbox,
                "level": block["level"],
                "text": block["content"],
                "merge_previous": block["merge_prev"],
            }
        )
        if bbox and upstream._base_type(block["label"]) in upstream.IMAGE_LABELS:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Unlimited-OCR-2 emitted an invalid image bbox")
            with Image.open(pages[page_index]) as image:
                width, height = image.size
                box = (
                    int(x1 * width / 1000),
                    int(y1 * height / 1000),
                    max(1, int(x2 * width / 1000)),
                    max(1, int(y2 * height / 1000)),
                )
                with image.crop(box).convert("RGB") as crop:
                    crop.save(output_dir / f"page{page_index + 1}_{x1}_{y1}_{x2}_{y2}.jpg")
    markdown = upstream.convert(raw)
    (output_dir / "result_raw.txt").write_text(raw, encoding="utf-8")
    write_layout_json(
        output_dir,
        {
            "schema_version": 1,
            "model_id": model_id,
            "coordinate_system": "normalized_0_1000",
            "page_count": len(pages),
            "pages": page_metadata(pages, output_dir),
            "blocks": layout,
            "raw_output": raw,
            "markdown": markdown,
            "complete_generation": True,
        },
    )
    return markdown
