import argparse
import lance
import os
from PIL import Image
import io
from typing import Optional, Union, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

def save_image(image_path: str, image_data: bytes, quality: int = 100) -> bool:
    """Save image data to disk.
    
    Args:
        image_path: Path to save the image
        image_data: Binary image data
        quality: Image quality for JPEG compression
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with Image.open(io.BytesIO(image_data)) as img:
            img.save(image_path, quality=quality)
            console.print(f"[green]image: {image_path} saved successfully.[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error processing image '{image_path}': {e}[/red]")
        return False

def save_caption(caption_path: str, caption_lines: List[str]) -> bool:
    """Save caption data to disk.
    
    Args:
        caption_path: Path to save the caption
        caption_lines: List of caption lines
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(caption_path), exist_ok=True)
        with open(caption_path, "w", encoding="utf-8") as f:
            for line in caption_lines:
                if line and line.strip():
                    f.write(line.strip() + "\n")
            console.print(f"[yellow]Saving caption to {caption_path}[/yellow]")
        return True
    except Exception as e:
        console.print(f"[red]Error saving caption '{caption_path}': {e}[/red]")
        return False

def extract_from_lance(
    lance_file: Union[str, lance.LanceDataset],
    output_dir: str,
    version: Optional[str] = None
) -> None:
    """Extract images and captions from a Lance dataset.
    
    Args:
        lance_file: Path to lance file or LanceDataset object
        output_dir: Directory to save extracted data
        version: Optional dataset version
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open dataset
    dataset = lance.dataset(lance_file, version=version) if isinstance(lance_file, str) else lance_file
    
    total_records = len(dataset)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Extracting data...", total=total_records)
        
        for batch in dataset.to_batches():
            filepaths = batch.column("filepath").to_pylist()
            extensions = batch.column("extension").to_pylist()
            images = batch.column("data").to_pylist()
            captions = batch.column("captions").to_pylist()

            for filepath, extension, image, caption in zip(
                filepaths, extensions, images, captions
            ):
                if not os.path.exists(filepath):
                    if not save_image(filepath, image):
                        progress.advance(task)
                        continue

                caption_path = os.path.splitext(filepath)[0] + ".txt"
                if caption:
                    save_caption(caption_path, caption)
                
                progress.advance(task)

def main():

    parser = argparse.ArgumentParser(
        description="Extract images and captions from a Lance dataset"
    )
    parser.add_argument("lance_file", help="Path to the .lance file")
    parser.add_argument(
        "--output_dir",
        default="./dataset",
        help="Directory to save extracted data",
    )
    parser.add_argument(
        "--version",
        help="Dataset version",
    )

    args = parser.parse_args()
    extract_from_lance(args.lance_file, args.output_dir, args.version)

if __name__ == "__main__":
    main()
