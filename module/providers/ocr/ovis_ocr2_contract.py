"""Document Markdown prompts shared by the OvisOCR runtimes and GUI."""

from pathlib import Path, PureWindowsPath

OVIS_OCR3_MODEL_ID = "ATH-MaaS/OvisOCR3"

# ATH-MaaS/OvisOCR3: prompts/document_markdown.txt
OVIS_OCR3_DEFAULT_PROMPT = (
    "Extract all readable content from the image in natural human reading order and output the result as a single "
    "Markdown document. For charts or images, represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."
)
OVIS_OCR2_DEFAULT_PROMPT = "\n" + OVIS_OCR3_DEFAULT_PROMPT
_DOCUMENT_PROMPTS = {
    OVIS_OCR3_MODEL_ID.casefold(): OVIS_OCR3_DEFAULT_PROMPT,
    "ath-maas/ovisocr2": OVIS_OCR2_DEFAULT_PROMPT,
}


def _checkpoint_prompt(model_id: str) -> str | None:
    key = (model_id or "").strip().replace("\\", "/").rstrip("/").casefold()
    if key in _DOCUMENT_PROMPTS:
        return _DOCUMENT_PROMPTS[key]
    path = PureWindowsPath(key)  # Accept paths from either the client or the serving host.
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return _DOCUMENT_PROMPTS.get(path.parent.parent.name.removeprefix("models--").replace("--", "/"))
    named = _DOCUMENT_PROMPTS.get(f"ath-maas/{path.name}")
    if named and (path.anchor or key.startswith(("./", "../", "~/")) or Path(model_id).is_dir()):
        return named
    return None


def ovis_document_prompt(runtime_model_id: str, model_id: str) -> str:
    """For a custom serving name, take the prompt from the configured checkpoint."""
    return _checkpoint_prompt(runtime_model_id) or _checkpoint_prompt(model_id) or OVIS_OCR3_DEFAULT_PROMPT
