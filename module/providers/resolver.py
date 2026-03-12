"""
PromptResolver - 集中化的 Prompt 选择逻辑

解决原代码中 15+ 个 Provider 各自重复实现 prompt fallback 链的问题
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .catalog import canonicalize_provider_name, provider_prompt_prefixes
from .base import PromptContext


@dataclass
class PromptTemplate:
    """Prompt 模板定义"""

    name: str
    description: str
    system: str
    user: str
    supported_mimes: List[str]
    supported_providers: List[str]


class PromptResolver:
    """
    集中化的 Prompt 选择逻辑

    优先级（从高到低）：
    1. Gemini Task 动态模板（如果是 Gemini + image）
    2. Pair 图像模式（如果 pair_dir 存在）
    3. Provider 特定覆盖
    4. Mime 类型基础选择
    5. Character prompt 注入
    """

    def __init__(self, config: Dict[str, Any], provider_name: str):
        self.config = config
        self.provider_name = canonicalize_provider_name(provider_name)
        self.prompts = config.get("prompts", {})

    def resolve(
        self,
        mime: str,
        args: Any,
        character_prompt: str = "",
        character_name: str = "",
    ) -> PromptContext:
        """解析最终使用的 prompt"""
        # 基础选择
        system, user = self._base_prompts(mime)

        # Provider 特定覆盖
        system, user = self._provider_override(system, user, mime)

        # Pair 模式覆盖
        if getattr(args, "pair_dir", ""):
            system, user = self._pair_override(system, user, mime)

        # Gemini Task 模板系统
        if self.provider_name == "gemini" and mime.startswith("image"):
            gemini_task = getattr(args, "gemini_task", "")
            if gemini_task:
                user = self._apply_gemini_task_template(gemini_task, user)

        # 注入 character prompt
        if character_prompt:
            user = character_prompt + user

        return PromptContext(
            system=system,
            user=user,
            character_name=character_name,
            character_prompt=character_prompt,
        )

    def _base_prompts(self, mime: str) -> Tuple[str, str]:
        """基于 mime 的基础选择"""
        system = self.prompts.get("system_prompt", "")
        user = self.prompts.get("prompt", "")

        if mime.startswith("video"):
            system = self.prompts.get("video_system_prompt", system)
            user = self.prompts.get("video_prompt", user)
        elif mime.startswith("audio"):
            system = self.prompts.get("audio_system_prompt", system)
            user = self.prompts.get("audio_prompt", user)
        elif mime.startswith("image"):
            system = self.prompts.get("image_system_prompt", system)
            user = self.prompts.get("image_prompt", user)

        return system, user

    def _provider_override(self, system: str, user: str, mime: str) -> Tuple[str, str]:
        """Provider 特定覆盖"""
        provider_names = provider_prompt_prefixes(self.provider_name)
        provider_prefixes = _unique_preserve_order(name.replace("_", "") for name in provider_names)

        if mime.startswith("video"):
            system_keys = [f"{name}_video_system_prompt" for name in provider_names]
            system_keys.extend(f"{name}_video_system_prompt" for name in provider_prefixes)
            system = self._get_with_fallback(*system_keys, "step_video_system_prompt", system)

            user_keys = [f"{name}_video_prompt" for name in provider_names]
            user_keys.extend(f"{name}_video_prompt" for name in provider_prefixes)
            user = self._get_with_fallback(*user_keys, "step_video_prompt", user)
        elif mime.startswith("image"):
            # 修复：kimi_code/kimi_vl/minimax 兼容性，添加特定 fallback
            kimi_fallback = ""
            if self.provider_name in ("kimi_code", "kimi_vl"):
                kimi_fallback = "kimi_image_system_prompt"

            system_keys = [f"{name}_image_system_prompt" for name in provider_names]
            system_keys.extend(f"{name}_image_system_prompt" for name in provider_prefixes)
            system = self._get_with_fallback(
                *system_keys,
                kimi_fallback,
                "mistral_ocr_image_system_prompt",
                "pixtral_image_system_prompt",  # 通用兼容性 fallback
                system,
            )

            # 修复：kimi_code/kimi_vl/minimax 兼容性，添加特定 fallback
            kimi_prompt_fallback = ""
            if self.provider_name in ("kimi_code", "kimi_vl"):
                kimi_prompt_fallback = "kimi_image_prompt"
            # minimax_code 可以回退到 minimax_api 的 prompt
            minimax_fallback = ""
            if self.provider_name == "minimax_code":
                minimax_fallback = "minimax_api_image_prompt"

            user_keys = [f"{name}_image_prompt" for name in provider_names]
            user_keys.extend(f"{name}_image_prompt" for name in provider_prefixes)
            user = self._get_with_fallback(
                *user_keys,
                kimi_prompt_fallback,
                minimax_fallback,
                "mistral_ocr_image_prompt",
                "pixtral_image_prompt",
                user,
            )
        elif mime.startswith("audio"):
            system_keys = [f"{name}_audio_system_prompt" for name in provider_names]
            user_keys = [f"{name}_audio_prompt" for name in provider_names]
            system = self._get_with_fallback(*system_keys, system)
            user = self._get_with_fallback(*user_keys, user)

        return system, user

    def _pair_override(self, system: str, user: str, mime: str) -> Tuple[str, str]:
        """Pair 模式覆盖（处理命名不一致）"""
        system = self._get_first_existing(["image_pair_system_prompt", "pair_image_system_prompt"], system)

        user = self._get_first_existing(["image_pair_prompt", "pair_image_prompt"], user)

        return system, user

    def _get_with_fallback(self, *keys: str) -> str:
        """
        获取第一个存在的 prompt key
        最后一个参数是默认值
        """
        *search_keys, default = keys
        for key in search_keys:
            if key in self.prompts:
                return self.prompts[key]
        return default

    def _get_first_existing(self, keys: List[str], default: str) -> str:
        """获取第一个存在的 key"""
        for key in keys:
            if key in self.prompts:
                return self.prompts[key]
        return default

    def _apply_gemini_task_template(self, task: str, base_prompt: str) -> str:
        """
        Gemini Task 模板系统

        从原代码 api_handler.py:188-230 提取的正则匹配逻辑
        """
        task_prompts = self.prompts.get("task", {})

        # 规则匹配
        patterns = [
            # change X to Y
            (r"^\s*change\s+(.+?)\s+to\s+(.+?)\s*$", "change_a_to_b"),
            # transform style X to Y
            (r"^\s*(transform|convert)\s+style\s+(.+?)\s+to\s+(.+?)\s*$", "transform_style_a_to_b"),
            # combine X and Y
            (r"^\s*combine\s+(.+?)\s+(and|with)\s+(.+?)\s*$", "combine_a_and_b"),
            # add X to Y
            (r"^\s*add\s+(.+?)\s+to\s+(.+?)\s*$", "add_a_to_b"),
        ]

        for pattern, template_key in patterns:
            match = re.match(pattern, task, re.IGNORECASE)
            if match:
                template = task_prompts.get(template_key)
                if template:
                    # 替换占位符
                    groups = match.groups()
                    result = template

                    # 处理 {a}, {b} 和 <a>, <b> 格式
                    for i, val in enumerate(groups[-2:] if len(groups) > 2 else groups, 1):
                        placeholder = "{a}" if i == 1 else "{b}"
                        alt_placeholder = "<a>" if i == 1 else "<b>"
                        result = result.replace(placeholder, val.strip())
                        result = result.replace(alt_placeholder, val.strip())

                    # 处理 \b 单词边界格式（如 "change a to b" -> "change X to Y"）
                    if "{a}" not in template and "<a>" not in template:
                        result = re.sub(r"\ba\b", groups[0].strip(), result)
                        if len(groups) > 1:
                            result = re.sub(r"\bb\b", groups[1].strip(), result)

                    return result

        # 直接匹配模板名或返回原 task
        return task_prompts.get(task, task)


def _unique_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
