#!/usr/bin/env python3
"""
html2md.py — Convert local HTML files to markdown via ReaderLM-v2 (mlx-lm)

Reads one or more local HTML files, extracts metadata from <meta> tags and
<title>, and converts the HTML to clean markdown using ReaderLM-v2 running
locally on Apple Silicon via mlx-lm.

Usage:
    python html2md.py <file.html> [OPTIONS]
    python html2md.py <directory/> [OPTIONS]    # batch: all .html/.htm files

Examples:
    python html2md.py article.html
    python html2md.py article.html -o ~/notes/
    python html2md.py saved_pages/ -o ~/notes/ -f txt
    python html2md.py page.html --model mlx-community/jinaai-ReaderLM-v2 -v
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output

# Import shared ReaderLM functions from any2md.web to avoid code duplication
from any2md.web import load_reader_model, html_to_markdown, DEFAULT_MODEL, page_to_markdown, page_to_text

# Configure logging (will be overridden by setup_logging in main)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata extraction from HTML meta tags
# ---------------------------------------------------------------------------

def extract_meta_tags(html: str) -> dict:
    """
    Extract metadata from HTML <meta> tags and <title> tag.

    Parses title, description, and author from the HTML without any external
    library dependencies — pure regex-based extraction.

    Args:
        html: Raw HTML string to parse.

    Returns:
        Dict with any of: title, description, author (only keys that were found).
    """
    metadata: dict = {}

    # <title>...</title>
    title_match = re.search(
        r'<title[^>]*>(.*?)</title>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        title = title_match.group(1).strip()
        # Collapse internal whitespace (multi-line titles)
        title = re.sub(r'\s+', ' ', title)
        if title:
            metadata['title'] = title

    # <meta name="description" content="..."> (both attribute orders)
    desc = _extract_meta(html, 'description')
    if desc:
        metadata['description'] = desc

    # <meta name="author" content="..."> (both attribute orders)
    author = _extract_meta(html, 'author')
    if author:
        metadata['author'] = author

    return metadata


def _extract_meta(html: str, name: str) -> Optional[str]:
    """
    Extract a <meta name="X" content="Y"> value by attribute name.

    Handles both attribute orderings: name-first and content-first.

    Args:
        html: HTML string to search.
        name: Meta tag name attribute (e.g. 'description', 'author').

    Returns:
        Content value string, or None if not found.
    """
    patterns = [
        # name before content
        rf'<meta\s+name=["\'](?i:{re.escape(name)})["\'][^>]*content=["\']([^"\']*)["\']',
        # content before name
        rf'<meta\s+content=["\']([^"\']*)["\'][^>]*name=["\'](?i:{re.escape(name)})["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


# ---------------------------------------------------------------------------
# Output filename
# ---------------------------------------------------------------------------

def html_path_to_stem(html_path: Path) -> str:
    """
    Derive a safe output filename stem from an HTML file path.

    Uses the file's stem (name without extension). No URL-style mangling
    needed since it's already a file path.

    Args:
        html_path: Path to the HTML file.

    Returns:
        Filename-safe string (no extension, max 80 chars).
    """
    stem = html_path.stem
    # Replace non-alphanumeric (except hyphen/underscore) with underscore
    stem = re.sub(r'[^\w\-]', '_', stem)
    # Collapse multiple underscores
    stem = re.sub(r'_+', '_', stem)
    stem = stem.strip('_')
    return stem[:80] if stem else 'output'


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_html_file(
    html_path: Path,
    model,
    tokenizer,
    output_dir: Path,
    fmt: OutputFormat,
) -> Path:
    """
    Convert a single HTML file to markdown (or txt) and write to output_dir.

    Reads the file, extracts metadata from <meta> tags, converts HTML to
    markdown via ReaderLM-v2, assembles the output with frontmatter (if md),
    and writes the result.

    Args:
        html_path: Path to the input HTML file.
        model: Loaded mlx-lm model.
        tokenizer: Loaded tokenizer.
        output_dir: Directory to write the output file.
        fmt: OutputFormat.md or OutputFormat.txt.

    Returns:
        Path to the written output file.
    """
    logger.info("Processing: %s", html_path)

    html = html_path.read_text(encoding='utf-8', errors='replace')
    logger.debug("Read %d chars from %s", len(html), html_path)

    # Extract metadata from the HTML itself
    metadata = extract_meta_tags(html)
    metadata['source'] = str(html_path.resolve())
    metadata['fetched_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Convert HTML → markdown via ReaderLM-v2
    markdown_content = html_to_markdown(html, model=model, tokenizer=tokenizer)
    logger.info("Generated %d chars of markdown from %s", len(markdown_content), html_path.name)

    # Determine output path
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = html_path_to_stem(html_path)
    output_file = output_dir / f"{stem}.{fmt.value}"

    # Format and write
    if fmt == OutputFormat.md:
        content = page_to_markdown(markdown_content, metadata)
    else:
        content = page_to_text(markdown_content)

    write_output(content, output_file)
    logger.info("Written: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert local HTML files to markdown using ReaderLM-v2 (local MLX inference).",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="HTML file or directory of .html/.htm files to convert.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path("."),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="ReaderLM model (HuggingFace ID or local path).",
    )] = DEFAULT_MODEL,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert local HTML files to markdown using ReaderLM-v2 (local MLX inference).

    Reads HTML from a single file or every .html/.htm file in a directory,
    extracts metadata (title, description, author) from <meta> tags, and
    converts the HTML to clean markdown using ReaderLM-v2 running locally
    on Apple Silicon via mlx-lm.
    """
    setup_logging(verbose)

    # Collect HTML files to process
    input_path = Path(input_path)
    if input_path.is_dir():
        html_files: List[Path] = sorted(
            list(input_path.glob("*.html")) + list(input_path.glob("*.htm"))
        )
        if not html_files:
            typer.echo(
                f"No .html or .htm files found in {input_path}", err=True
            )
            raise typer.Exit(1)
        logger.info("Batch mode: found %d HTML files in %s", len(html_files), input_path)
    elif input_path.is_file():
        if input_path.suffix.lower() not in ('.html', '.htm'):
            logger.warning(
                "Input file %s has unexpected extension — proceeding anyway",
                input_path.name,
            )
        html_files = [input_path]
    else:
        typer.echo(f"Input path does not exist: {input_path}", err=True)
        raise typer.Exit(1)

    output_dir_path = Path(output_dir)

    # Load model once for all files
    reader_model, tokenizer = load_reader_model(model)

    # Process each file
    for html_file in html_files:
        out = process_html_file(
            html_file,
            model=reader_model,
            tokenizer=tokenizer,
            output_dir=output_dir_path,
            fmt=format,
        )
        typer.echo(str(out))


if __name__ == "__main__":
    app()
