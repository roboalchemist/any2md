#!/usr/bin/env python3
"""
img2md.py - Image to Markdown Conversion Tool

Converts images to markdown using a Vision Language Model (VLM) via mlx-vlm.
Supports single images and batch directory processing.

Usage:
    python img2md.py [options] <image_or_directory>

Examples:
    python img2md.py photo.jpg
    python img2md.py screenshot.png -o ~/notes/
    python img2md.py /path/to/images/ -o ~/output/
    python img2md.py diagram.png --model qwen2.5-vl-7b
    python img2md.py scan.tiff -f txt
"""

import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from typing_extensions import Annotated

from md_common import build_frontmatter, setup_logging, write_output

# mlx-vlm imports — loaded at module level so tests can patch img2md.generate etc.
# Falls back gracefully if mlx-vlm is not installed; ImportError is raised at call time.
try:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    _MLX_VLM_AVAILABLE = True
except Exception:
    generate = None  # type: ignore[assignment]
    load = None  # type: ignore[assignment]
    apply_chat_template = None  # type: ignore[assignment]
    load_config = None  # type: ignore[assignment]
    _MLX_VLM_AVAILABLE = False

# Configure logging (will be reconfigured by setup_logging at CLI entry point)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image file extensions
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Short alias -> full HuggingFace model ID
MODEL_ALIASES = {
    "qwen2.5-vl-7b": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "qwen2.5-vl-3b": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "qwen2.5-vl-2b": "mlx-community/Qwen2.5-VL-2B-Instruct-4bit",
    "qwen2.5-vl-72b": "mlx-community/Qwen2.5-VL-72B-Instruct-4bit",
    "smoldocling": "mlx-community/SmolDocling-256M-4bit",
}

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

EXTRACTION_PROMPT = """Extract all text from this image as clean markdown.
- Preserve tables as markdown tables
- Convert equations to LaTeX ($...$ inline, $$...$$ block)
- Use proper heading levels (#, ##, ###)
- Preserve code blocks with language tags
- Do NOT add text not present in the image
Output ONLY the markdown."""


def resolve_model(model_alias: str) -> str:
    """
    Resolve a model alias or short name to its full HuggingFace model ID.

    Args:
        model_alias: Short alias or full HuggingFace model ID

    Returns:
        Full HuggingFace model ID
    """
    return MODEL_ALIASES.get(model_alias, model_alias)


def get_image_metadata(image_path: Path, model_name: str) -> dict:
    """
    Extract image metadata: dimensions, format, file size.

    Uses Pillow for dimensions if available, falls back to file stat only.

    Args:
        image_path: Path to the image file
        model_name: The VLM model used for inference (recorded in metadata)

    Returns:
        Metadata dict suitable for YAML frontmatter
    """
    image_path = Path(image_path)
    stat = image_path.stat()
    suffix = image_path.suffix.lstrip(".").lower()

    metadata: dict = {
        "source": str(image_path),
        "format": suffix,
        "file_size_bytes": stat.st_size,
        "model_used": model_name,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Try to get image dimensions via Pillow
    try:
        from PIL import Image as PilImage
        with PilImage.open(image_path) as img:
            width, height = img.size
            metadata["width"] = width
            metadata["height"] = height
    except Exception as exc:
        logger.debug("Could not read image dimensions via Pillow: %s", exc)

    return metadata


def load_vlm_model(model_name: str) -> Tuple:
    """
    Load VLM model and processor via mlx-vlm.

    Args:
        model_name: HuggingFace model ID or local path

    Returns:
        Tuple of (model, processor, config)

    Raises:
        ImportError: If mlx-vlm is not installed
    """
    if load is None or load_config is None:
        raise ImportError("mlx-vlm is required. Install it with: pip install mlx-vlm")

    logger.info("Loading VLM model: %s", model_name)
    model, processor = load(model_name)
    config = load_config(model_name)
    return model, processor, config


def image_to_markdown_text(
    image_path: Path,
    model,
    processor,
    config: dict,
    model_name: str,
    prompt: str = EXTRACTION_PROMPT,
    max_tokens: int = 2048,
) -> str:
    """
    Run VLM inference on an image, return raw markdown text.

    Args:
        image_path: Path to the image file
        model: Loaded mlx-vlm model
        processor: Loaded mlx-vlm processor
        config: Model config dict from load_config()
        model_name: Model name (for logging)
        prompt: Instruction prompt for the VLM
        max_tokens: Maximum tokens to generate

    Returns:
        Raw markdown text output from the VLM
    """
    if generate is None or apply_chat_template is None:
        raise ImportError("mlx-vlm is required. Install it with: pip install mlx-vlm")

    logger.info("Running VLM inference on: %s", image_path.name)

    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=1
    )

    output = generate(
        model,
        processor,
        formatted_prompt,
        image=str(image_path),
        max_tokens=max_tokens,
        verbose=False,
    )

    return output


def image_to_markdown(content: str, metadata: dict) -> str:
    """
    Wrap VLM content with YAML frontmatter.

    Args:
        content: Raw markdown text from VLM
        metadata: Metadata dict for YAML frontmatter

    Returns:
        Full markdown string with frontmatter
    """
    fm = build_frontmatter(metadata)
    return f"{fm}\n\n{content}\n"


def image_to_text(content: str) -> str:
    """
    Return raw text content without frontmatter.

    Args:
        content: Raw text from VLM

    Returns:
        Plain text string
    """
    return content.strip() + "\n"


def find_images_in_directory(directory: Path) -> List[Path]:
    """
    Find all supported image files in a directory (non-recursive).

    Args:
        directory: Directory to search

    Returns:
        Sorted list of image file paths
    """
    images = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
            images.append(path)
    return images


def process_single_image(
    image_path: Path,
    model,
    processor,
    config: dict,
    model_name: str,
    output_dir: Path,
    fmt: str,
    prompt: str = EXTRACTION_PROMPT,
    max_tokens: int = 2048,
) -> Path:
    """
    Process one image: run VLM, format output, write file.

    Args:
        image_path: Path to the input image
        model: Loaded mlx-vlm model
        processor: Loaded mlx-vlm processor
        config: Model config dict
        model_name: Full model name (for metadata)
        output_dir: Directory to write output file
        fmt: Output format — "md" or "txt"
        prompt: VLM instruction prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Path to the written output file
    """
    metadata = get_image_metadata(image_path, model_name)

    raw_text = image_to_markdown_text(
        image_path, model, processor, config, model_name,
        prompt=prompt, max_tokens=max_tokens
    )

    base_name = image_path.stem
    output_path = output_dir / f"{base_name}.{fmt}"

    if fmt == "md":
        content = image_to_markdown(raw_text, metadata)
    else:
        content = image_to_text(raw_text)

    write_output(content, output_path)
    logger.info("Output saved to: %s", output_path)
    return output_path


# --- CLI ---

class OutputFormat(str, Enum):
    md = "md"
    txt = "txt"


# Build model help text
_model_help_lines = "\n".join(
    f"  {alias} -> {full}" for alias, full in MODEL_ALIASES.items()
)
_model_help = (
    "VLM model alias or HuggingFace ID.\n\n"
    "Aliases:\n" + _model_help_lines
)

app = typer.Typer(
    help="Convert images to markdown using a local VLM (mlx-vlm).",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="Image file or directory of images to convert.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path.cwd(),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help=_model_help,
    )] = DEFAULT_MODEL,
    prompt: Annotated[str, typer.Option(
        "--prompt",
        help="Custom instruction prompt for the VLM.",
    )] = EXTRACTION_PROMPT,
    max_tokens: Annotated[int, typer.Option(
        "--max-tokens",
        help="Maximum tokens to generate per image.",
    )] = 2048,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
):
    """
    Convert images to markdown (default) or plain text using a local VLM.

    Accepts a single image file or a directory of images. Supports JPEG, PNG,
    GIF, BMP, WebP, and TIFF formats. Loads the model once and processes all
    images in a batch, writing one output file per image.
    """
    setup_logging(verbose)

    if not input_path.exists():
        logger.error("Path not found: %s", input_path)
        raise typer.Exit(code=1)

    # Collect images to process
    if input_path.is_dir():
        images = find_images_in_directory(input_path)
        if not images:
            logger.error(
                "No supported image files found in directory: %s", input_path
            )
            raise typer.Exit(code=1)
        logger.info("Found %d image(s) in: %s", len(images), input_path)
    elif input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_FORMATS:
            logger.error(
                "Unsupported file format '%s'. Supported: %s",
                input_path.suffix,
                ", ".join(sorted(SUPPORTED_FORMATS)),
            )
            raise typer.Exit(code=1)
        images = [input_path]
    else:
        logger.error("Input is not a file or directory: %s", input_path)
        raise typer.Exit(code=1)

    resolved_model = resolve_model(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once, process all images
    try:
        vlm_model, processor, config = load_vlm_model(resolved_model)
    except ImportError:
        raise typer.Exit(code=1)
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", resolved_model, exc)
        raise typer.Exit(code=1)

    output_paths = []
    failed = []

    for image_path in images:
        try:
            out = process_single_image(
                image_path,
                vlm_model,
                processor,
                config,
                resolved_model,
                output_dir,
                format.value,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            output_paths.append(out)
        except Exception as exc:
            logger.error("Failed to process %s: %s", image_path.name, exc)
            failed.append(image_path)

    logger.info(
        "Done. %d succeeded, %d failed.", len(output_paths), len(failed)
    )
    if failed:
        logger.warning("Failed images: %s", [str(p) for p in failed])


if __name__ == "__main__":
    app()
