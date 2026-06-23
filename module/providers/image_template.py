"""Shared helpers for the image VLM prompt template feature.

The image prompt template lets users pick a caption "style" (a system/user
prompt pair plus an output contract) independent of the chosen provider. This
module is the single source of truth for:

- detecting whether a non-default template is active (`active_image_template`)
- resolving a template's output contract (`image_template_output`)
- building a freeform caption prompt for structured-output providers that must
  yield their forced schema (`build_freeform_caption_prompt`)
- parsing freeform caption output back into ``(raw, parsed)``
  (`parse_freeform_caption_output`)

Keeping this logic in one place avoids re-deriving the ``tid and tid != "custom"``
special case across the resolver, Gemini, Codex, and Grok Build providers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from module.providers.codex_schema import strip_markdown_json_fence

DEFAULT_TEMPLATE_OUTPUT = "text"


def active_image_template(args: Any) -> str:
    """Return the active non-default image template id, or '' when following the model.

    Empty string and the sentinel ``"custom"`` both mean "follow the model"
    (current behavior), so callers only need a truthiness check.
    """
    template_id = getattr(args, "image_prompt_template", "") or ""
    if template_id == "custom":
        return ""
    return template_id


def image_template_output(prompts: Dict[str, Any], template_id: str) -> str:
    """Return the output contract ('text' | 'json') for a template id.

    Unknown templates or a missing ``output`` field fall back to ``'text'``.
    """
    if not template_id:
        return ""
    templates = (prompts or {}).get("image_templates", {})
    template = templates.get(template_id) or {}
    output = str(template.get("output", "") or "").strip().lower()
    return output or DEFAULT_TEMPLATE_OUTPUT


def build_freeform_caption_prompt(*, system_prompt: str, user_prompt: str, output: str) -> str:
    """Build a freeform caption prompt that follows the template without rating wrap.

    Used by structured-output CLI providers (Codex / Grok Build) when an image
    prompt template is active, so the model is not coerced into the rating schema.
    """
    sections = []
    system_text = (system_prompt or "").strip()
    user_text = (user_prompt or "").strip()
    if system_text:
        sections.append(system_text)
    if user_text:
        sections.append(user_text)
    if output == "json":
        sections.append("Output valid JSON only. Do not wrap it in Markdown code fences.")
    return "\n\n".join(sections)


def parse_freeform_caption_output(text: str, output: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Parse freeform model output into ``(raw, parsed)``.

    For ``output == "json"``: strip an optional ```json fence then ``json.loads``;
    a dict result is returned as ``parsed`` (raw re-serialized for stable storage).
    On any failure or non-dict JSON, fall back to the cleaned text with
    ``parsed=None``. For text output: return cleaned text with ``parsed=None``.
    """
    cleaned = (text or "").strip()
    if output == "json":
        candidate = strip_markdown_json_fence(cleaned)
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return cleaned, None
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False), parsed
        return cleaned, None
    return cleaned, None
