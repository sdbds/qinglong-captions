"""Normalize explicit short/long caption fields without rewriting custom JSON."""

import re


_SECTION_HEADING = re.compile(
    r"^[ \t]*###[ \t]*(short|long)(?:[ \t_]+description)?[ \t]*(?:[:\uff1a][ \t]*|(?=\r?$))",
    re.IGNORECASE | re.MULTILINE,
)
_CAPTION_FIELDS = ("short_description", "long_description", "short", "long", "description")


def _split_caption_sections(value: object) -> dict[str, str]:
    if not isinstance(value, str):
        return {}
    headings = list(_SECTION_HEADING.finditer(value))
    # Require an unambiguous pair, so prose, partial output, and repeated sections stay intact.
    if len(headings) != 2 or {match.group(1).lower() for match in headings} != {"short", "long"}:
        return {}
    if value[:headings[0].start()].strip():
        return {}
    sections = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(value)
        text = value[heading.end():end].strip()
        if not text:
            return {}
        sections[f"{heading.group(1).lower()}_description"] = text
    return sections


def normalize_sectioned_caption(payload: dict, mode: str) -> dict | None:
    """Filter an explicit short/long pair, or return None for ordinary structured data."""
    description = payload.get("description")
    if description is not None and not isinstance(description, str):
        return None

    result = dict(payload)
    embedded = {}
    for field in _CAPTION_FIELDS:
        sections = _split_caption_sections(result.get(field))
        if not sections:
            continue
        if field == "description":
            del result[field]
        else:
            # A matching section keeps the priority of the field it came from.
            key = f"{field.removesuffix('_description')}_description"
            result[field] = sections.pop(key)
        for key, value in sections.items():
            embedded.setdefault(key, value)

    if not embedded:
        return None

    # A legacy description string must not override the selected short caption.
    if isinstance(result.get("description"), str):
        del result["description"]

    for name in ("short", "long"):
        key = f"{name}_description"
        alias = result.get(name)
        if isinstance(alias, str):
            del result[name]
        if mode in ("short", "long") and mode != name:
            result.pop(key, None)
            continue
        for text in (result.get(key), alias, embedded.get(key)):
            if isinstance(text, str) and text.strip():
                result[key] = text
                break
    return result
