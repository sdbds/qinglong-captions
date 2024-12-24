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
    ".webp",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".WEBP",
    ".BMP",
]


def get_supported_extensions() -> Tuple[str, ...]:
    """Get all supported image extensions including optional formats."""
    extensions = BASE_IMAGE_EXTENSIONS.copy()

    # Try to add AVIF support
    try:
        import pillow_avif

        extensions.extend([".avif", ".AVIF"])
    except ImportError:
        pass

    # Try to add JPEG-XL support
    try:
        from jxlpy import JXLImagePlugin

        extensions.extend([".jxl", ".JXL"])
    except ImportError:
        try:
            import pillow_jxl

            extensions.extend([".jxl", ".JXL"])
        except ImportError:
            pass

    return tuple(extensions)


# Default schema definition
DEFAULT_DATASET_SCHEMA = [
    ("filepath", pa.string()),
    ("extension", pa.string()),
    ("hash", pa.string()),
    ("size", pa.int64()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("data", pa.binary()),
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
