"""
PromptResolver - 集中化的 Prompt 选择逻辑

解决原代码中 15+ 个 Provider 各自重复实现 prompt fallback 链的问题
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from .base import PromptContext, Provider
from .catalog import canonicalize_provider_name, provider_prompt_fallback_keys, provider_prompt_prefixes
from .image_template import active_image_template


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
    2. Pair 图像模式（如果实际存在配对媒体）
    3. Provider 特定覆盖
    4. Mime 类型基础选择
    5. Character prompt 注入
    """

    def __init__(self, config: Dict[str, Any], provider_name: str, provider_class: Type[Provider] | None = None):
        self.config = config
        self.provider_name = canonicalize_provider_name(provider_name)
        self.provider_class = provider_class
        self.prompts = config.get("prompts", {})

    def resolve(
        self,
        mime: str,
        args: Any,
        character_prompt: str = "",
        character_name: str = "",
        media: Any | None = None,
    ) -> PromptContext:
        """解析最终使用的 prompt"""
        self._args = args

        # 基础选择
        system, user = self._base_prompts(mime)

        # Provider 特定覆盖
        system, user = self._provider_override(system, user, mime)

        # Image template override (higher than provider-specific keys)
        if mime.startswith("image"):
            system, user = self._image_template_override(system, user)

        # Pair 模式覆盖
        if mime.startswith("image") and self._is_pair_mode(args, media):
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

        if mime.startswith("video"):
            system_keys = [f"{name}_video_system_prompt" for name in provider_names]
            system = self._get_with_fallback(*system_keys, *self._provider_fallback_keys(mime, "system"), system)

            user_keys = [f"{name}_video_prompt" for name in provider_names]
            user = self._get_with_fallback(*user_keys, *self._provider_fallback_keys(mime, "user"), user)
        elif mime.startswith("image"):
            system_keys = [f"{name}_image_system_prompt" for name in provider_names]
            system = self._get_with_fallback(*system_keys, *self._provider_fallback_keys(mime, "system"), system)

            user_keys = [f"{name}_image_prompt" for name in provider_names]
            user = self._get_with_fallback(*user_keys, *self._provider_fallback_keys(mime, "user"), user)
        elif mime.startswith("audio"):
            system_keys = [f"{name}_audio_system_prompt" for name in provider_names]
            user_keys = [f"{name}_audio_prompt" for name in provider_names]
            system = self._get_with_fallback(*system_keys, *self._provider_fallback_keys(mime, "system"), system)
            user = self._get_with_fallback(*user_keys, *self._provider_fallback_keys(mime, "user"), user)

        return system, user

    def _provider_fallback_keys(self, mime: str, field: str) -> Tuple[str, ...]:
        if self.provider_class is not None:
            return tuple(self.provider_class.prompt_fallback_keys(mime, field))
        return provider_prompt_fallback_keys(self.provider_name, mime, field)

    def _image_template_override(self, system: str, user: str) -> Tuple[str, str]:
        """Image VLM prompt template override layer.

        When args.image_prompt_template is set to a non-empty, non-'custom' value,
        look up the template in prompts['image_templates'] and override system/user
        with the referenced prompt keys. Unknown ids fall back silently.
        """
        template_id = active_image_template(self._args)
        if not template_id:
            return system, user
        templates = self.prompts.get("image_templates", {})
        tpl = templates.get(template_id)
        if not tpl:
            return system, user
        sys_key = tpl.get("system_key", "")
        usr_key = tpl.get("user_key", "")
        new_system = self.prompts.get(sys_key, system) if sys_key else system
        new_user = self.prompts.get(usr_key, "") if usr_key else ""
        return new_system, new_user

    def _pair_override(self, system: str, user: str, mime: str) -> Tuple[str, str]:
        """Pair 模式覆盖（处理命名不一致）"""
        system = self._get_first_existing(["image_pair_system_prompt", "pair_image_system_prompt"], system)

        user = self._get_first_existing(["image_pair_prompt", "pair_image_prompt"], user)

        return system, user

    @staticmethod
    def _is_pair_mode(args: Any, media: Any | None) -> bool:
        """Prefer prepared media state over raw CLI args when deciding pair prompt mode."""
        if media is not None:
            extras = getattr(media, "extras", {}) or {}
            pair_blob = getattr(media, "pair_blob", None)
            pair_pixels = getattr(media, "pair_pixels", None)
            pair_extras = getattr(media, "pair_extras", None)
            if pair_blob is not None or pair_pixels is not None:
                return True
            if pair_extras:
                return True
            return bool(extras.get("pair_uri"))

        return bool(getattr(args, "pair_dir", ""))

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
