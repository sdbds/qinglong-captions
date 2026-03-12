"""
VisionAPIProvider - 视觉 API Provider 基类

适用于：Gemini, Pixtral

特性：
- 结构化输出配置（JSON Schema）
- 多模态支持（图像/PDF/视频/音频）
- inline_data 文件保存副作用
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import CaptionResult, MediaContext, MediaModality, Provider, ProviderType
from .capabilities import ProviderCapabilities
from .utils import encode_image_to_blob


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

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        """Vision API provider 通用媒体准备。"""
        file_path = Path(uri)
        file_size = file_path.stat().st_size if file_path.exists() else 0

        if mime.startswith("video"):
            modality = MediaModality.VIDEO
        elif mime.startswith("audio"):
            modality = MediaModality.AUDIO
        elif mime.startswith("image"):
            modality = MediaModality.IMAGE
        elif mime.startswith("application"):
            modality = MediaModality.DOCUMENT
        else:
            modality = MediaModality.UNKNOWN

        blob = None
        pixels = None
        pair_blob = None
        pair_pixels = None
        pair_extras: List[str] = []
        audio_blob = None
        extras: Dict[str, Any] = {}

        if mime.startswith("image"):
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)
            pair_dir = getattr(args, "pair_dir", "")
            if pair_dir and blob:
                pair_uri = (Path(pair_dir) / file_path.name).resolve()
                if pair_uri.exists():
                    pair_blob, pair_pixels = encode_image_to_blob(str(pair_uri), to_rgb=True)
                    extras["pair_uri"] = str(pair_uri)
                    pair_extras = self._scan_pair_extras(uri, pair_dir)
        elif mime.startswith("audio") and file_size < 20 * 1024 * 1024:
            try:
                audio_blob = file_path.read_bytes()
            except OSError as exc:
                self.log(f"Failed to read audio: {exc}", "yellow")
        elif mime.startswith("application"):
            output_dir = file_path.with_suffix("")
            output_dir.mkdir(parents=True, exist_ok=True)
            extras["output_dir"] = output_dir

        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=modality,
            file_size=file_size,
            blob=blob,
            pixels=pixels,
            pair_blob=pair_blob,
            pair_pixels=pair_pixels,
            pair_extras=pair_extras,
            audio_blob=audio_blob,
            extras=extras,
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

    def _scan_pair_extras(self, uri: str, pair_dir: str) -> List[str]:
        """扫描额外配对图 (foo_1.jpg, foo_2.jpg ...)。"""
        base_dir = Path(pair_dir).resolve()
        stem = Path(uri).stem
        primary_ext = Path(uri).suffix.lower()
        extras_paths: List[tuple[int, Path]] = []

        try:
            if not base_dir.exists():
                return []

            for path in base_dir.iterdir():
                if not (path.is_file() and path.name.startswith(f"{stem}_") and path.suffix.lower() == primary_ext):
                    continue
                name_stem = path.stem
                if len(name_stem) <= len(stem) + 1 or name_stem[len(stem)] != "_":
                    continue
                num_part = name_stem[len(stem) + 1 :]
                if num_part.isdigit():
                    extras_paths.append((int(num_part), path))
        except OSError as exc:
            self.log(f"Failed to scan pair extras: {exc}", "yellow")
            return []

        extras_paths.sort(key=lambda item: item[0])

        encoded: List[str] = []
        for _, path in extras_paths:
            try:
                extra_blob, _ = encode_image_to_blob(str(path), to_rgb=True)
                if extra_blob:
                    encoded.append(extra_blob)
                    self.log(f"Paired extra: {path.name}", "blue")
            except Exception as exc:
                self.log(f"Failed to encode pair extra {path}: {exc}", "yellow")

        return encoded
