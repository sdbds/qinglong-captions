from __future__ import annotations

import re

from ..local_llm_base import LocalLLMProvider

_LANGUAGE_NAMES = {
    "zh_cn": "Chinese (Simplified)",
    "zh-hans": "Chinese (Simplified)",
    "zh_tw": "Chinese (Traditional)",
    "zh-hant": "Chinese (Traditional)",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "es": "Spanish",
}


def clean_translation_output(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def build_translation_prompt(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    context: str = "",
    glossary: str = "",
) -> str:
    source_name = _LANGUAGE_NAMES.get(source_lang.lower(), source_lang)
    target_name = _LANGUAGE_NAMES.get(target_lang.lower(), target_lang)
    sections = [
        f"Translate the CURRENT markdown fragment from {source_name} to {target_name}.",
        "Requirements:",
        "- Output only the translated markdown fragment.",
        "- Preserve markdown structure, headings, lists, tables, and blockquotes.",
        "- Keep placeholders such as __QLP_0__ unchanged.",
        "- Keep code fences, inline code, URLs, file paths, and numbers unchanged when possible.",
        "- Do not explain your translation and do not wrap the answer in code fences.",
    ]
    if glossary.strip():
        sections.append("Terminology glossary (follow strictly):\n<glossary>\n" + glossary.strip() + "\n</glossary>")
    if context.strip():
        sections.append("Context for disambiguation. Do not translate this context directly:\n<context>\n" + context.strip() + "\n</context>")
    sections.append("<source>\n" + text + "\n</source>")
    sections.append("Return only the translated fragment.")
    return "\n\n".join(sections)


class HYMTProvider(LocalLLMProvider):
    default_model_id = "tencent/HY-MT1.5-7B"

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        context: str = "",
        glossary: str = "",
    ) -> str:
        prompt = build_translation_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            glossary=glossary,
        )
        return clean_translation_output(self.generate_text(prompt))
