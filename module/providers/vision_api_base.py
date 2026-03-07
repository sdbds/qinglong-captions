"""
VisionAPIProvider - 视觉 API Provider 基类

适用于：Gemini, Pixtral

特性：
- 结构化输出配置（JSON Schema）
- 多模态支持（图像/PDF/视频/音频）
- inline_data 文件保存副作用
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import CaptionResult, MediaContext, Provider, ProviderType
from .capabilities import ProviderCapabilities


@dataclass
class StructuredOutputConfig:
    """结构化输出配置"""

    enabled: bool = False
    mime_type: str = "application/json"
    schema: Optional[Dict] = None
    response_modalities: Optional[List[str]] = None


class VisionAPIProvider(Provider):
    """
    视觉 API Provider 基类

    支持结构化输出 Schema 定义
    """

    provider_type = ProviderType.VISION_API
    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_structured_output=True,
        supports_video=True,
        supports_audio=True,
        supports_images=True,
        supports_documents=True,
    )

    def get_structured_output_config(self, media: MediaContext, args: Any) -> StructuredOutputConfig:
        """
        获取结构化输出配置

        子类可以覆盖此方法
        默认返回 disabled
        """
        return StructuredOutputConfig(enabled=False)

    def _build_caption_result(
        self, raw_response: str, structured_data: Optional[Dict] = None, metadata: Optional[Dict] = None
    ) -> CaptionResult:
        """构建统一的结果对象"""
        meta = metadata or {}
        meta["provider"] = self.name

        return CaptionResult(raw=raw_response, parsed=structured_data, metadata=meta)

    def _build_rating_schema(self) -> Dict:
        """
        构建标准评分 schema

        用于 Gemini 等支持结构化输出的 provider
        """
        return {
            "type": "object",
            "required": ["scores", "average_score", "description"],
            "properties": {
                "scores": {
                    "type": "object",
                    "required": [
                        "Costume & Makeup & Prop Presentation/Accuracy",
                        "Character Portrayal & Posing",
                        "Setting & Environment Integration",
                        "Lighting & Mood",
                        "Composition & Framing",
                        "Storytelling & Concept",
                        "Level of S*e*x*y",
                        "Figure",
                        "Overall Impact & Uniqueness",
                    ],
                    "properties": {
                        "Costume & Makeup & Prop Presentation/Accuracy": {"type": "integer"},
                        "Character Portrayal & Posing": {"type": "integer"},
                        "Setting & Environment Integration": {"type": "integer"},
                        "Lighting & Mood": {"type": "integer"},
                        "Composition & Framing": {"type": "integer"},
                        "Storytelling & Concept": {"type": "integer"},
                        "Level of S*e*x*y": {"type": "integer"},
                        "Figure": {"type": "integer"},
                        "Overall Impact & Uniqueness": {"type": "integer"},
                    },
                },
                "total_score": {"type": "integer"},
                "average_score": {"type": "number"},
                "description": {"type": "string"},
                "character_name": {"type": "string"},
                "series": {"type": "string"},
            },
        }

    def _build_pair_image_schema(self) -> Dict:
        """构建 Pair 图像的简化 schema"""
        return {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
        }
