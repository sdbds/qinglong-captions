"""
CloudVLMProvider - 云端 VLM Provider 基类

适用于：StepFun, QwenVL, GLM, Ark, KimiCode, KimiVL

特性：
- 支持 Pair 图像
- 支持多图输入（pair_extras）
- 支持视频上传
- 支持音频
"""

from pathlib import Path
from typing import Any, List

from .base import MediaContext, MediaModality, Provider, ProviderContext, ProviderType
from .capabilities import ProviderCapabilities
from .utils import encode_image_to_blob


class CloudVLMProvider(Provider):
    """
    云端 VLM Provider 基类

    提供通用的媒体准备逻辑，支持 Pair 图像和多图输入
    """

    provider_type = ProviderType.CLOUD_VLM
    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_video=True,
        supports_images=True,
        supports_audio=True,
    )

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        """
        Cloud VLM 通用的媒体准备

        - 图像：编码 base64 + 配对图扫描
        - 音频：小文件读 bytes，大文件留待上传
        - 视频：留待子类处理上传
        """
        file_path = Path(uri)
        file_size = file_path.stat().st_size if file_path.exists() else 0

        # 确定模态
        if mime.startswith("video"):
            modality = MediaModality.VIDEO
        elif mime.startswith("audio"):
            modality = MediaModality.AUDIO
        elif mime.startswith("image"):
            modality = MediaModality.IMAGE
        else:
            modality = MediaModality.UNKNOWN

        # 初始化字段
        blob = None
        pixels = None
        pair_blob = None
        pair_pixels = None
        pair_extras: List[str] = []
        audio_blob = None

        # 图像处理
        if mime.startswith("image"):
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)

            # Pair 图像处理
            pair_dir = getattr(args, "pair_dir", "")
            if pair_dir and blob:
                pair_uri = Path(pair_dir) / file_path.name
                if pair_uri.exists():
                    pair_blob, pair_pixels = encode_image_to_blob(str(pair_uri), to_rgb=True)
                    # 扫描额外配对图
                    pair_extras = self._scan_pair_extras(uri, pair_dir)

        # 音频处理（小文件直接读）
        elif mime.startswith("audio") and file_size < 20 * 1024 * 1024:
            try:
                with open(uri, "rb") as f:
                    audio_blob = f.read()
            except Exception as e:
                self.log(f"Failed to read audio: {e}", "yellow")

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
        )

    def _scan_pair_extras(self, uri: str, pair_dir: str) -> List[str]:
        """
        扫描额外的配对图 (pair_1.jpg, pair_2.jpg 等)

        返回 base64 编码的图像列表
        """
        from pathlib import Path

        base_dir = Path(pair_dir).resolve()
        stem = Path(uri).stem
        primary_ext = Path(uri).suffix.lower()
        extras_paths = []

        try:
            if not base_dir.exists():
                return []

            for pth in base_dir.iterdir():
                if pth.is_file() and pth.name.startswith(f"{stem}_") and pth.suffix.lower() == primary_ext:
                    name_stem = pth.stem
                    if len(name_stem) > len(stem) + 1 and name_stem[len(stem)] == "_":
                        num_part = name_stem[len(stem) + 1 :]
                        if num_part.isdigit():
                            extras_paths.append((int(num_part), pth))
        except Exception as e:
            self.log(f"Failed to scan pair extras: {e}", "yellow")
            return []

        # 按数字排序
        extras_paths.sort(key=lambda t: t[0])

        # 编码所有额外图片
        result = []
        for _, pth in extras_paths:
            try:
                blob, _ = encode_image_to_blob(str(pth), to_rgb=True)
                if blob:
                    result.append(blob)
                    self.log(f"Paired extra: {pth.name}", "blue")
            except Exception as e:
                self.log(f"Failed to encode pair extra {pth}: {e}", "yellow")

        return result
