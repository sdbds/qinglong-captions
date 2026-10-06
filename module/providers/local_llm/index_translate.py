from __future__ import annotations

from typing import Any

from ..backends import OpenAIChatRuntime
from ..local_llm_base import LocalLLMProvider
from .hy_mt import clean_translation_output as clean_markdown_output

_LANGUAGE_NAMES = {
    "zh": "中文",
    "zh-cn": "简体中文",
    "zh-hans": "简体中文",
    "zh-tw": "繁体中文",
    "zh-hant": "繁体中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "ru": "俄语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "ar": "阿拉伯语",
    "it": "意大利语",
    "nl": "荷兰语",
    "pl": "波兰语",
    "ro": "罗马尼亚语",
    "sv": "瑞典语",
    "tr": "土耳其语",
    "hi": "印地语",
    "vi": "越南语",
    "th": "泰语",
    "id": "印尼语",
    "ms": "马来语",
    "fil": "菲律宾语",
}


def build_translation_prompt(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    context: str = "",
    glossary: str = "",
) -> str:
    source_key = source_lang.strip().lower().replace("_", "-")
    target_key = target_lang.strip().lower().replace("_", "-")
    source_name = "" if source_key in {"", "auto"} else _LANGUAGE_NAMES.get(source_key, source_lang.strip())
    target_name = _LANGUAGE_NAMES.get(target_key, target_lang.strip())
    # Index's training-aligned instTrans format carries document constraints explicitly.
    constraints = [
        "〖硬性要求〗保留 Markdown 结构，包括标题、列表、表格和引用。",
        "〖硬性要求〗原样保留 __QLP_0__ 等所有占位符，不得翻译、删除或改写。",
        "〖硬性要求〗保留代码块、行内代码、URL、文件路径和数字。",
        "〖硬性要求〗仅翻译源文，不要添加解释，也不要用额外的代码块包裹译文。",
    ]
    if glossary.strip():
        constraints.append("〖硬性要求〗严格遵循以下术语表：\n<glossary>\n" + glossary.strip() + "\n</glossary>")
    if context.strip():
        constraints.append("〖注意〗以下上文仅用于消歧，不要翻译或输出上文：\n<context>\n" + context.strip() + "\n</context>")
    requirements = "\n".join(f"{index}. {constraint}" for index, constraint in enumerate(constraints, start=1))
    return (
        f"请将以下{source_name}文本翻译成{target_name}，并且严格遵循所有约束要求。\n\n"
        f"〖源文〗\n{text}\n\n〖约束要求〗\n{requirements}\n\n"
        "只输出译文，不要有任何额外说明。"
    )


def clean_translation_output(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    elif cleaned.startswith("<think>"):
        raise RuntimeError("Index-Translate returned unfinished thinking instead of a translation; increase max_new_tokens.")
    cleaned = clean_markdown_output(cleaned)
    if not cleaned:
        raise RuntimeError("Index-Translate returned no translation; check the output token budget and server configuration.")
    return cleaned


class IndexTranslateProvider(LocalLLMProvider):
    model_ids = ("IndexTeam/Index-Translate-2B", "IndexTeam/Index-Translate-9B")
    default_model_id = "IndexTeam/Index-Translate-9B"

    def __init__(self, *args, max_new_tokens: int = 4096, **kwargs) -> None:
        super().__init__(*args, max_new_tokens=max_new_tokens, **kwargs)

    def _load_components(self) -> tuple[Any, Any]:
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        from utils.transformer_loader import load_pretrained_component

        device, dtype, attn_impl = self._resolve_device_dtype()
        if self.console:
            self.console.print(f"[green]Loading text model:[/green] {self.model_id} ({device}, {dtype})")
        tokenizer = load_pretrained_component(
            AutoTokenizer,
            self.model_id,
            console=self.console,
            component_name="tokenizer",
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "dtype": dtype,
        }
        if attn_impl != "eager":
            model_kwargs["attn_implementation"] = attn_impl
        model = load_pretrained_component(
            Qwen3_5ForConditionalGeneration,
            self.model_id,
            console=self.console,
            component_name="model",
            **model_kwargs,
        ).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer, model

    def generate_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self.runtime_backend.is_openai:
            return OpenAIChatRuntime(self.runtime_backend).complete(
                messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        tokenizer, model = self._get_or_load_components()
        model_device = next(model.parameters()).device
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(model_device) for key, value in tokenized.items()}
        inputs = self._filter_generation_inputs(model, inputs)
        # Preserve the checkpoint's EOS list (endoftext and im_end), not just the tokenizer's EOS.
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
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
        return clean_translation_output(self.generate_text(prompt))
