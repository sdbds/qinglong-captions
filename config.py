"""Configuration constants for the dataset processing."""

from typing import Tuple, List, Dict, Any
import os
import toml
import pyarrow as pa

# Base image extensions
BASE_IMAGE_EXTENSIONS: List[str] = [
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".tif",
    ".tiff",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".GIF",
    ".WEBP",
    ".BMP",
    ".ICO",
    ".TIF",
    ".TIFF",
]

BASE_ANIMATION_EXTENSIONS: List[str] = [
    ".gif",
    ".webp",
    ".GIF",
    ".WEBP",
]

BASE_VIDEO_EXTENSIONS: List[str] = [
    ".mp4",
    ".webm",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
]

BASE_AUDIO_EXTENSIONS: List[str] = [
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aifc",
    ".aif",
    ".au",
    ".snd",
    ".mid",
    ".midi",
    ".mka",
]


def get_supported_extensions(media_type: str = "image") -> Tuple[str, ...]:
    """Get all supported media extensions including optional formats."""
    if media_type == "image" or media_type == "animation":
        extensions = (
            BASE_IMAGE_EXTENSIONS.copy()
            if media_type == "image"
            else BASE_ANIMATION_EXTENSIONS.copy()
        )

        # Try to add AVIF support
        try:
            import pillow_avif

            extensions.extend([".avif", ".AVIF"])
        except ImportError:
            pass

        # Try to add JPEG-XL support
        try:
            import pillow_jxl

            extensions.extend([".jxl", ".JXL"])
        except ImportError:
            pass

        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
            extensions.extend([".heic", ".heif", ".HEIC", ".HEIF"])
        except ImportError:
            pass

        if media_type == "animation":
            try:
                from apng import APNG

                extensions.extend([".apng", ".APNG"])
            except ImportError:
                pass

    elif media_type == "video":
        extensions = BASE_VIDEO_EXTENSIONS.copy()
    elif media_type == "audio":
        extensions = BASE_AUDIO_EXTENSIONS.copy()

    return tuple(extensions)


# Default schema definition
DEFAULT_DATASET_SCHEMA = [
    ("uris", pa.string()),
    ("mime", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("channels", pa.int32()),
    ("depth", pa.int32()),
    ("hash", pa.string()),
    ("size", pa.int64()),
    ("has_audio", pa.bool_()),
    ("duration", pa.int32()),
    ("num_frames", pa.int32()),
    ("frame_rate", pa.float32()),
    ("blob", pa.binary()),
    ("captions", pa.list_(pa.string())),
]


def load_schema_from_toml(schema_path: str) -> List[Tuple[str, str]]:
    """Load a custom dataset schema from a TOML file.

    Args:
        schema_path: Path to the TOML file containing schema definition

    Returns:
        List of tuples containing (field_name, field_type)

    Example TOML format:
    [schema]
    fields = [
        { name = "filepath", type = "string" },
        { name = "extension", type = "string" },
        # ... other fields
    ]
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        config = toml.load(schema_path)
        schema_def = config.get("schema", {}).get("fields", [])

        if not schema_def:
            raise ValueError("No schema definition found in TOML file")

        return [(field["name"], field["type"]) for field in schema_def]
    except Exception as e:
        raise ValueError(f"Failed to parse schema file: {str(e)}")


# Current active schema - defaults to DEFAULT_DATASET_SCHEMA
DATASET_SCHEMA = DEFAULT_DATASET_SCHEMA.copy()


def set_custom_schema(schema_path: str) -> None:
    """Set a custom schema from a TOML file.

    Args:
        schema_path: Path to the TOML file containing schema definition
    """
    global DATASET_SCHEMA
    DATASET_SCHEMA = load_schema_from_toml(schema_path)
