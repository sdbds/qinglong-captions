"""Configuration constants for the dataset processing."""

import os
from typing import Any, Dict, FrozenSet, List, Tuple

import pyarrow as pa
import toml

# Base extensions (lowercase-only canonical forms).
# All comparison sites must normalize with .lower() before checking membership.
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
    ".avif",
]

BASE_ANIMATION_EXTENSIONS: List[str] = [
    ".gif",
    ".webp",
    ".avif",
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

BASE_TEXT_EXTENSIONS: List[str] = [
    ".txt",
    ".md",
]

BASE_APPLICATION_EXTENSIONS: List[str] = [
    ".pdf",
    ".psd",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".rtf",
    ".epub",
]

# Frozen sets for O(1) membership testing (already lowercase).
IMAGE_EXTENSIONS_SET: FrozenSet[str] = frozenset(BASE_IMAGE_EXTENSIONS)
ANIMATION_EXTENSIONS_SET: FrozenSet[str] = frozenset(BASE_ANIMATION_EXTENSIONS)
VIDEO_EXTENSIONS_SET: FrozenSet[str] = frozenset(BASE_VIDEO_EXTENSIONS)
AUDIO_EXTENSIONS_SET: FrozenSet[str] = frozenset(BASE_AUDIO_EXTENSIONS)
TEXT_EXTENSIONS_SET: FrozenSet[str] = frozenset(BASE_TEXT_EXTENSIONS)
APPLICATION_EXTENSIONS_SET: FrozenSet[str] = frozenset(BASE_APPLICATION_EXTENSIONS)


def get_supported_extensions(media_type: str = "image") -> Tuple[str, ...]:
    """Get all supported media extensions including optional formats.

    Returns lowercase-only extensions. Callers must normalize suffixes
    with .lower() before membership testing.
    """
    if media_type == "image" or media_type == "animation":
        extensions = BASE_IMAGE_EXTENSIONS.copy() if media_type == "image" else BASE_ANIMATION_EXTENSIONS.copy()

        # Try to add JPEG-XL support
        try:
            import pillow_jxl

            extensions.append(".jxl")
        except ImportError:
            pass

        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
            extensions.extend([".heic", ".heif"])
        except ImportError:
            pass

        if media_type == "animation":
            try:
                from apng import APNG

                extensions.append(".apng")
            except ImportError:
                pass

    elif media_type == "video":
        extensions = BASE_VIDEO_EXTENSIONS.copy()
    elif media_type == "audio":
        extensions = BASE_AUDIO_EXTENSIONS.copy()
    elif media_type == "text":
        extensions = BASE_TEXT_EXTENSIONS.copy()
    elif media_type == "application":
        extensions = BASE_APPLICATION_EXTENSIONS.copy()

    return tuple(extensions)


def load_toml_config(config_path: str, section: str) -> Dict[str, Any]:
    """Load a configuration section from TOML config files.

    Supports both the new split config directory layout (prompts.toml,
    model.toml, general.toml) and the legacy single config.toml file.

    Args:
        config_path: Path to a TOML file or a config directory
        section: Name of the section to load

    Returns:
        Dictionary containing the configuration data
    """
    from config.loader import load_config as _load_merged

    path = os.path.normpath(config_path)

    # If config_path is a directory, load merged config from it
    if os.path.isdir(path):
        config = _load_merged(path)
    else:
        # If it's a file, try treating its parent as a config dir first
        parent = os.path.dirname(path)
        config = _load_merged(parent)
        # Fall back to loading the single file directly
        if not config and os.path.exists(path):
            config = toml.load(path)

    if not config:
        raise FileNotFoundError(f"Config not found at: {config_path}")

    section_data = config.get(section, {})
    if not section_data:
        raise ValueError(f"No {section} configuration found in TOML file")

    return section_data


def load_schema_from_toml(config_path: str) -> List[Tuple[str, str]]:
    """Load dataset schema from a TOML file.

    Args:
        config_path: Path to the TOML file containing schema definition

    Returns:
        List of tuples containing (field_name, field_type)
    """
    schema_data = load_toml_config(config_path, "schema")
    fields = schema_data.get("fields", [])
    return [(field["name"], field["type"]) for field in fields]


def load_colors_from_toml(config_path: str) -> Dict[str, str]:
    """Load console colors from a TOML file.

    Args:
        config_path: Path to the TOML file containing colors definition

    Returns:
        Dictionary mapping media types to color names
    """
    return load_toml_config(config_path, "colors")


def load_prompts_from_toml(config_path: str) -> Dict[str, str]:
    """Load prompts from a TOML file.

    Args:
        config_path: Path to the TOML file containing prompts definition

    Returns:
        Dictionary containing prompt configurations
    """
    return load_toml_config(config_path, "prompts")


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
    ("blob", pa.large_binary()),
    ("captions", pa.list_(pa.string())),
    ("chunk_offsets", pa.list_(pa.int32())),
]

# Default console colors
DEFAULT_CONSOLE_COLORS = {
    "image": "green",
    "animation": "bold green",
    "video": "magenta",
    "audio": "orange1",
    "application": "bright_red",
    "text": "yellow",
    "caption": "yellow",
    "unknown": "cyan",
}

# Current active configurations - defaults to built-in values
DATASET_SCHEMA = DEFAULT_DATASET_SCHEMA.copy()
CONSOLE_COLORS = DEFAULT_CONSOLE_COLORS.copy()
SYSTEM_PROMPT = ""  # Will be loaded from config


def load_config(config_path: str) -> None:
    """Load all configurations from a TOML file.

    Args:
        config_path: Path to the TOML file containing configurations
    """
    global DATASET_SCHEMA, CONSOLE_COLORS, SYSTEM_PROMPT

    try:
        DATASET_SCHEMA = load_schema_from_toml(config_path)
    except Exception as e:
        print(f"Warning: Failed to load schema configuration: {e}")

    try:
        colors = load_colors_from_toml(config_path)
        if colors:
            CONSOLE_COLORS.update(colors)
    except Exception as e:
        print(f"Warning: Failed to load colors configuration: {e}")

    try:
        prompts = load_prompts_from_toml(config_path)
        if prompts and "system_prompt" in prompts:
            SYSTEM_PROMPT = prompts["system_prompt"]
    except Exception as e:
        print(f"Warning: Failed to load prompts configuration: {e}")
