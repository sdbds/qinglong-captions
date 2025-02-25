import base64
import io
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich_pixels import Pixels

import functools


@functools.lru_cache(maxsize=128)
def encode_image(image_path: str) -> Optional[Tuple[str, Pixels]]:
    """Encode the image to base64 format with size optimization.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        with Image.open(image_path) as image:
            image.load()
            if "xmp" in image.info:
                del image.info["xmp"]

            # Calculate dimensions that are multiples of 16
            max_size = 1024
            width, height = image.size
            aspect_ratio = width / height

            def calculate_dimensions(max_size: int) -> Tuple[int, int]:
                if width > height:
                    new_width = min(max_size, (width // 16) * 16)
                    new_height = ((int(new_width / aspect_ratio)) // 16) * 16
                else:
                    new_height = min(max_size, (height // 16) * 16)
                    new_width = ((int(new_height * aspect_ratio)) // 16) * 16

                # Ensure dimensions don't exceed max_size
                if new_width > max_size:
                    new_width = max_size
                    new_height = ((int(new_width / aspect_ratio)) // 16) * 16
                if new_height > max_size:
                    new_height = max_size
                    new_width = ((int(new_height * aspect_ratio)) // 16) * 16

                return new_width, new_height

            new_width, new_height = calculate_dimensions(max_size)
            image = image.resize((new_width, new_height), Image.LANCZOS).convert("RGB")

            pixels = Pixels.from_image(
                image,
                resize=(image.width // 18, image.height // 18),
            )

            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8"), pixels

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found - {image_path}")
    except Image.UnidentifiedImageError:
        console.print(f"[red]Error:[/red] Cannot identify image file - {image_path}")
    except PermissionError:
        console.print(
            f"[red]Error:[/red] Permission denied accessing file - {image_path}"
        )
    except OSError as e:
        # Specifically handle XMP and metadata-related errors
        if "XMP data is too long" in str(e):
            console.print(
                f"[yellow]Warning:[/yellow] Skipping image with XMP data error - {image_path}"
            )
        else:
            console.print(
                f"[red]Error:[/red] OS error processing file {image_path}: {str(e)}"
            )
    except ValueError as e:
        console.print(
            f"[red]Error:[/red] Invalid value while processing {image_path}: {str(e)}"
        )
    except Exception as e:
        console.print(
            f"[red]Error:[/red] Unexpected error processing {image_path}: {str(e)}"
        )
    return None, None


def display_image_with_captions(
    uri: str,
    pixels: Pixels,
    tag_description: str,
    short_description: str,
    long_description: str,
    short_highlight_rate: float = 0,
    long_highlight_rate: float = 0,
    console: Console = None,
):
    """Display an image with its captions in a rich layout.

    Args:
        uri: Path to the image file
        pixels: Rich pixels object representing the image
        tag_description: Tags description text
        short_description: Short description text
        long_description: Long description text
        short_highlight_rate: Highlight rate for short description
        long_highlight_rate: Highlight rate for long description
        console: Rich console object for output, uses default if None
    """
    if console is None:
        from rich.console import Console

        console = Console()

    # 获取图片实际高度
    panel_height = 32  # 加上面板的边框高度

    # 创建布局
    layout = Layout()

    # 创建右侧的垂直布局
    right_layout = Layout()

    # 创建上半部分的水平布局（tag和short并排）
    top_layout = Layout()
    top_layout.split_row(
        Layout(
            Panel(
                Text(tag_description, style="magenta"),
                title="tags",
                height=panel_height // 2,
                padding=0,
                expand=True,
            ),
            ratio=1,
        ),
        Layout(
            Panel(
                short_description,
                title=f"short_description - [yellow]highlight rate:[/yellow] {short_highlight_rate}",
                height=panel_height // 2,
                padding=0,
                expand=True,
            ),
            ratio=1,
        ),
    )

    # 将右侧布局分为上下两部分
    right_layout.split_column(
        Layout(top_layout, ratio=1),
        Layout(
            Panel(
                long_description,
                title=f"long_description - [yellow]highlight rate:[/yellow] {long_highlight_rate}",
                height=panel_height // 2,
                padding=0,
                expand=True,
            )
        ),
    )

    # 主布局分为左右两部分
    layout.split_row(
        Layout(
            Panel(pixels, height=panel_height, padding=0, expand=True),
            name="image",
            ratio=1,
        ),
        Layout(right_layout, name="caption", ratio=2),
    )

    # 将整个布局放在一个高度受控的面板中
    console.print(
        Panel(
            layout,
            title=Path(uri).name,
            height=panel_height + 2,
            padding=0,
        )
    )

    del pixels
