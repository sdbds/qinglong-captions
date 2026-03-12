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

    def _generate_hy_mt(self, prompt: str) -> str:
        tokenizer, model = self._get_or_load_components()
        model_device = next(model.parameters()).device
        messages = [{"role": "user", "content": prompt}]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_k"] = 20
            generation_kwargs["top_p"] = 0.6

        if isinstance(tokenized, dict):
            inputs = {key: value.to(model_device) for key, value in tokenized.items()}
            output_ids = model.generate(**inputs, **generation_kwargs)
            prompt_length = inputs["input_ids"].shape[1]
        else:
            input_ids = tokenized.to(model_device)
            output_ids = model.generate(input_ids, **generation_kwargs)
            prompt_length = input_ids.shape[-1]

        generated_ids = output_ids[0][prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

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
        return clean_translation_output(self._generate_hy_mt(prompt))
