"""OpenAI Compatible Provider

通用 OpenAI API 兼容 Provider，支持对接：
- vLLM (python -m vllm.entrypoints.openai.api_server)
- Ollama (ollama serve)
- LM Studio (本地服务器)
- SGLang (python -m sglang.launch_server)
- 任何其他 OpenAI 兼容服务

配置参数：
- openai_api_key: API 密钥（可选，本地服务可填任意值）
- openai_base_url: API 基础地址（如 http://localhost:8000/v1）
- openai_model_name: 模型名称（如 Qwen2-VL-7B-Instruct）
"""

import json
from typing import Optional

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.cloud_vlm_base import CloudVLMProvider
from module.providers.registry import register_provider
from utils.caption_modes import normalize_sectioned_caption
from utils.console_util import print_exception


def _ensure_json_instruction(messages: list[dict]) -> list[dict]:
    """Meet JSON mode's text requirement without changing reusable messages."""
    if not messages:
        return messages
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts = [content]
        elif isinstance(content, list):
            texts = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        else:
            texts = []
        if any(isinstance(text, str) and "json" in text.lower() for text in texts):
            return messages

    instruction = (
        "Return only a valid JSON object, without Markdown code fences. "
        "Follow the requested JSON schema if one is specified; otherwise put the caption "
        'in a string field named "description".'
    )
    result = list(messages)
    for index, message in enumerate(messages):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = [*content, {"type": "text", "text": instruction}]
        else:
            content = f"{content}\n\n{instruction}" if content else instruction
        result[index] = {**message, "content": content}
        break
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result


def _is_json_mode_unsupported(error: Exception) -> bool:
    """Only explicit request-format incompatibilities justify a plaintext fallback."""
    if getattr(error, "status_code", None) not in (400, 422):
        return False
    details = getattr(error, "body", None)
    if isinstance(details, dict):
        details = details.get("error", details)
    if isinstance(details, dict):
        message = " ".join(str(details.get(key) or "") for key in ("message", "param", "code")).lower()
    else:
        message = str(error).lower()
    return any(name in message for name in ("response_format", "json_object")) and any(
        marker in message for marker in (
            "unsupported", "not supported", "does not support", "not support",
            "unknown parameter", "unrecognized", "not allowed",
        )
    )


@register_provider("openai_compatible")
class OpenAICompatibleProvider(CloudVLMProvider):
    """通用 OpenAI 兼容 Provider
    
    支持任何 OpenAI API 格式的本地或远程服务。
    可对接 vLLM、Ollama、LM Studio、SGLang 等。
    """

    name = "openai_compatible"

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        """
        当配置了 openai_base_url 时启用此 provider
        支持 image 和 video 类型
        """
        has_base_url = getattr(args, "openai_base_url", "") != ""
        supports_mime = mime.startswith(("image", "video"))
        has_explicit_local_route = bool(
            getattr(args, "vlm_image_model", "") or getattr(args, "ocr_model", "")
        )
        if mime.startswith("image") and has_explicit_local_route:
            return False
        return has_base_url and supports_mime

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """执行 API 调用"""
        from openai import OpenAI

        # 获取配置
        api_key = getattr(self.ctx.args, "openai_api_key", "") or "sk-no-key-required"
        base_url = getattr(self.ctx.args, "openai_base_url", "")
        model_name = getattr(self.ctx.args, "openai_model_name", "default")

        if not base_url:
            self.log("openai_base_url is not configured", "red")
            return CaptionResult(raw="")

        # 构建消息
        messages = self._build_messages(media, prompts)

        # 获取生成参数
        temperature = getattr(self.ctx.args, "openai_temperature", 0.7)
        max_tokens = getattr(self.ctx.args, "openai_max_tokens", 2048)
        
        # 检查是否支持 JSON 模式
        use_json_mode = bool(getattr(self.ctx.args, "openai_json_mode", True)) and media.mime.startswith("image")

        request_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_mode:
            request_params["response_format"] = {"type": "json_object"}
            request_params["messages"] = _ensure_json_instruction(messages)

        # Provider.execute owns transport retries; the SDK must not multiply them.
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        try:
            try:
                completion = client.chat.completions.create(**request_params)
            except Exception as e:
                print_exception(self.ctx.console, e, prefix="API call failed")
                if not use_json_mode or not _is_json_mode_unsupported(e):
                    raise
                self.log("Retrying without JSON mode...", "yellow")
                del request_params["response_format"]
                request_params["messages"] = messages
                completion = client.chat.completions.create(**request_params)
            result = completion.choices[0].message.content or ""
        finally:
            client.close()

        # 解析 JSON 结果
        parsed = self._parse_result(result)
        
        if parsed:
            # 根据 mode 过滤字段
            mode = getattr(self.ctx.args, "mode", "all")
            filtered = self._filter_by_mode(parsed, mode, split_sections=media.mime.startswith("image"))
            return CaptionResult(
                raw=json.dumps(filtered, ensure_ascii=False),
                parsed=filtered,
                metadata={"provider": self.name, "model": model_name}
            )
        
        return CaptionResult(
            raw=result,
            metadata={"provider": self.name, "model": model_name}
        )

    def _build_messages(self, media: MediaContext, prompts: PromptContext) -> list[dict]:
        """构建 OpenAI 格式的消息（委托给基类通用实现）"""
        return self.build_cloud_vlm_messages(media, prompts)

    def _parse_result(self, result: str) -> Optional[dict]:
        """尝试解析结果为 JSON"""
        if not result:
            return None
            
        try:
            # 清理 markdown 代码块
            cleaned = result.strip()
            if cleaned.startswith("```"):
                # 移除开头的 ```json 或 ```
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                # 移除结尾的 ```
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _filter_by_mode(self, parsed: dict, mode: str, *, split_sections: bool = True) -> dict:
        """根据 mode 过滤字段"""
        if split_sections:
            normalized = normalize_sectioned_caption(parsed, mode)
            if normalized is not None:
                return normalized

        result = dict(parsed)
        for name in ("short", "long"):
            key = f"{name}_description"
            if mode in ("short", "long") and mode != name:
                result.pop(name, None)
                result.pop(key, None)
            elif name in result and key not in result:
                result[key] = result.pop(name)
        if mode not in ("short", "long") and "description" in result and "long_description" not in result:
            result["long_description"] = result.pop("description")
        return result

    def get_retry_config(self):
        """配置重试策略"""
        from module.providers.utils import classify_remote_api_error

        cfg = super().get_retry_config()

        def classify(e):
            return classify_remote_api_error(
                e,
                base_wait=cfg.base_wait,
                rate_limit_wait=10.0,
                retry_http_statuses=(500, 502, 503),
                transport_wait=2.0,
            )

        cfg.classify_error = classify
        cfg.on_exhausted = lambda e: (
            print_exception(self.ctx.console, e, prefix="OpenAI compatible provider exhausted", summary_style="yellow")
            or CaptionResult.failed(str(e), metadata={"provider": self.name, "retry_exhausted": True})
        )
        return cfg
