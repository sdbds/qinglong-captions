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
from pathlib import Path
from typing import Any, Optional

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider


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
        if has_explicit_local_route:
            return False
        return has_base_url and supports_mime

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """执行 API 调用"""
        from openai import OpenAI

        # 获取配置
        api_key = getattr(self.ctx.args, "openai_api_key", "sk-no-key-required")
        base_url = getattr(self.ctx.args, "openai_base_url", "")
        model_name = getattr(self.ctx.args, "openai_model_name", "default")

        if not base_url:
            self.log("openai_base_url is not configured", "red")
            return CaptionResult(raw="")

        # 创建客户端
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 构建消息
        messages = self._build_messages(media, prompts)

        # 获取生成参数
        temperature = getattr(self.ctx.args, "openai_temperature", 0.7)
        max_tokens = getattr(self.ctx.args, "openai_max_tokens", 2048)
        
        # 检查是否支持 JSON 模式
        use_json_mode = getattr(self.ctx.args, "openai_json_mode", True)

        try:
            # 构建请求参数
            request_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # 添加 JSON 响应格式（如果支持）
            if use_json_mode:
                request_params["response_format"] = {"type": "json_object"}
            
            # 发送请求
            completion = client.chat.completions.create(**request_params)
            
            # 提取结果
            result = completion.choices[0].message.content or ""
            
        except Exception as e:
            self.log(f"API call failed: {e}", "red")
            # 如果 JSON 模式失败，尝试不用 JSON 模式重试
            if use_json_mode and "response_format" in request_params:
                self.log("Retrying without JSON mode...", "yellow")
                try:
                    del request_params["response_format"]
                    completion = client.chat.completions.create(**request_params)
                    result = completion.choices[0].message.content or ""
                except Exception as e2:
                    self.log(f"Retry failed: {e2}", "red")
                    return CaptionResult(raw="")
            else:
                return CaptionResult(raw="")

        # 解析 JSON 结果
        parsed = self._parse_result(result)
        
        if parsed:
            # 根据 mode 过滤字段
            mode = getattr(self.ctx.args, "mode", "all")
            filtered = self._filter_by_mode(parsed, mode)
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

    def _filter_by_mode(self, parsed: dict, mode: str) -> dict:
        """根据 mode 过滤字段"""
        result = dict(parsed)  # 复制
        
        if mode == "short":
            # short 模式：移除长描述相关字段
            result.pop("long", None)
            result.pop("long_description", None)
            # 保留 short 或 short_description
            if "short" in result and "short_description" not in result:
                result["short_description"] = result.pop("short")
                
        elif mode == "long":
            # long 模式：移除短描述相关字段
            result.pop("short", None)
            result.pop("short_description", None)
            # 保留 long 或 long_description
            if "long" in result and "long_description" not in result:
                result["long_description"] = result.pop("long")
        
        else:  # mode == "all"
            # 规范化字段名
            if "short" in result and "short_description" not in result:
                result["short_description"] = result.pop("short")
            if "long" in result and "long_description" not in result:
                result["long_description"] = result.pop("long")
            if "description" in result and "long_description" not in result:
                result["long_description"] = result.pop("description")
        
        return result

    def get_retry_config(self):
        """配置重试策略"""
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e).lower()
            # 429 限流：等待较长时间
            if "429" in msg or "rate limit" in msg:
                return 10.0
            # 连接错误：短等待
            if "connection" in msg or "connect" in msg:
                return 2.0
            # 服务器错误：中等等待
            if "500" in msg or "502" in msg or "503" in msg:
                return 5.0
            return cfg.base_wait

        cfg.classify_error = classify
        cfg.on_exhausted = lambda e: self.log(
            f"OpenAI compatible provider exhausted: {e}", "yellow"
        ) or ""
        return cfg
