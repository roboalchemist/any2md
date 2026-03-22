#!/usr/bin/env python3
"""
nb.py - Jupyter Notebook to Markdown Converter

Converts .ipynb (Jupyter Notebook) files to markdown (default) or plain text
using only the stdlib json module — no nbformat or nbconvert dependency.

Usage:
    python nb.py [options] <notebook.ipynb>
    python nb.py [options] <directory/>

Examples:
    python nb.py analysis.ipynb
    python nb.py notebooks/ -o ~/notes/
    python nb.py experiment.ipynb -f txt --no-outputs
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source field normalisation
# ---------------------------------------------------------------------------

def _source_text(source) -> str:
    """
    Normalise a notebook cell's ``source`` field to a plain string.

    The nbformat spec allows ``source`` to be either a single string *or* a
    list of strings (lines, each ending in ``\\n`` except the last).  Both
    forms are valid and both appear in real notebooks.

    Args:
        source: String or list-of-strings from a cell's ``source`` field.

    Returns:
        A single string with lines joined (no trailing newline added).
    """
    if isinstance(source, list):
        return "".join(source)
    return source or ""


# ---------------------------------------------------------------------------
# Output-cell rendering
# ---------------------------------------------------------------------------

def _render_outputs(outputs: List[Dict]) -> str:
    """
    Render a list of cell output objects to markdown-friendly text.

    Handles:
    - ``stream``  → fenced ``output`` block
    - ``execute_result`` / ``display_data`` → fenced ``result`` block for
      text/plain; ``[image output omitted]`` for image/* MIME types
    - ``error``   → fenced ``output`` block with traceback

    Base64 image data (``image/*`` MIME types) is never decoded; a short
    placeholder note is emitted instead.

    Args:
        outputs: List of output dicts from the cell's ``outputs`` field.

    Returns:
        Rendered string (may be empty if there are no renderable outputs).
    """
    parts: List[str] = []

    for out in outputs:
        output_type = out.get("output_type", "")

        if output_type == "stream":
            text = _source_text(out.get("text", ""))
            if text.strip():
                parts.append(f"```output\n{text.rstrip()}\n```")

        elif output_type in ("execute_result", "display_data"):
            data = out.get("data", {})

            # Check for image outputs first (any image/* MIME key)
            has_image = any(k.startswith("image/") for k in data)
            if has_image:
                parts.append("[image output omitted]")
                continue

            # Render text/plain
            plain = data.get("text/plain", "")
            plain_text = _source_text(plain)
            if plain_text.strip():
                parts.append(f"```result\n{plain_text.rstrip()}\n```")

        elif output_type == "error":
            traceback = out.get("traceback", [])
            # ANSI escape sequences from tracebacks are stripped
            tb_text = "\n".join(
                re.sub(r'\x1b\[[0-9;]*m', '', line) for line in traceback
            )
            if tb_text.strip():
                parts.append(f"```output\n{tb_text.rstrip()}\n```")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Notebook → markdown conversion
# ---------------------------------------------------------------------------

def notebook_to_markdown(nb: Dict, kernel_language: str = "python",
                         include_outputs: bool = True) -> str:
    """
    Convert a parsed notebook dict to a markdown string.

    Markdown cells are passed through verbatim.  Code cells become fenced code
    blocks tagged with *kernel_language*.  Output cells are rendered according
    to their MIME type (text/plain, stream, error) or replaced with a
    placeholder for images.

    Args:
        nb: Parsed notebook JSON dict (top-level ``nbformat`` object).
        kernel_language: Programming language tag for code fences (e.g. ``python``).
        include_outputs: If False, skip all output cells.

    Returns:
        Full notebook content as a markdown string.
    """
    cells = nb.get("cells", [])
    parts: List[str] = []

    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = _source_text(cell.get("source", ""))

        if cell_type == "markdown":
            if source.strip():
                parts.append(source.rstrip())

        elif cell_type == "code":
            if source.strip():
                parts.append(f"```{kernel_language}\n{source.rstrip()}\n```")

            if include_outputs:
                outputs = cell.get("outputs", [])
                rendered = _render_outputs(outputs)
                if rendered:
                    parts.append(rendered)

        elif cell_type == "raw":
            # Raw cells: pass through as plain text (no fence)
            if source.strip():
                parts.append(source.rstrip())

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_nb_metadata(nb: Dict, source_path: Path) -> Dict:
    """
    Extract frontmatter metadata from a parsed notebook dict.

    Fields produced:
    - ``title``                 — first ``# Heading`` in markdown cells, or filename stem
    - ``kernel_language``       — from ``kernelspec.language`` (metadata block)
    - ``cell_count``            — total number of cells
    - ``code_cell_count``       — number of code cells
    - ``markdown_cell_count``   — number of markdown cells
    - ``source``                — absolute path to the source file
    - ``fetched_at``            — ISO-8601 UTC timestamp

    Args:
        nb: Parsed notebook JSON dict.
        source_path: Path to the source ``.ipynb`` file.

    Returns:
        Metadata dict suitable for ``build_frontmatter()``.
    """
    cells = nb.get("cells", [])
    nb_metadata = nb.get("metadata", {})

    # Kernel language
    kernelspec = nb_metadata.get("kernelspec", {})
    language_info = nb_metadata.get("language_info", {})
    kernel_language = (
        kernelspec.get("language")
        or language_info.get("name")
        or "python"
    )

    # Cell counts
    cell_count = len(cells)
    code_cell_count = sum(1 for c in cells if c.get("cell_type") == "code")
    markdown_cell_count = sum(1 for c in cells if c.get("cell_type") == "markdown")

    # Title: first H1 heading found in markdown cells, else filename stem
    title: Optional[str] = None
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            source = _source_text(cell.get("source", ""))
            match = re.search(r'^#\s+(.+)$', source, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                break
    if title is None:
        title = source_path.stem

    metadata: Dict = {
        'title': title,
        'kernel_language': kernel_language,
        'cell_count': cell_count,
        'code_cell_count': code_cell_count,
        'markdown_cell_count': markdown_cell_count,
        'source': str(source_path.resolve()),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    return {k: v for k, v in metadata.items() if v is not None and v != ''}


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def nb_to_full_markdown(md_content: str, metadata: Optional[Dict] = None) -> str:
    """
    Assemble final markdown output with optional YAML frontmatter.

    Args:
        md_content: Converted notebook content from ``notebook_to_markdown()``.
        metadata: Optional metadata dict for YAML frontmatter.

    Returns:
        Full markdown string ready to write to disk.
    """
    lines: List[str] = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append("")

    lines.append(md_content)
    return "\n".join(lines)


def nb_to_plain_text(md_content: str) -> str:
    """
    Strip markdown syntax from notebook content to produce plain text.

    Removes code-fence markers, ATX heading markers, and bold/italic
    decoration.  Output/result block labels are also stripped.

    Args:
        md_content: Markdown string from ``notebook_to_markdown()``.

    Returns:
        Plain text string.
    """
    text = md_content
    # Remove ATX heading markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove code-fence opening lines (``` or ```python etc.)
    text = re.sub(r'^```[^\n]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Single-file processor
# ---------------------------------------------------------------------------

def process_nb_file(nb_path: Path, output_dir: Path, fmt: str,
                    include_outputs: bool = True) -> Path:
    """
    Convert one ``.ipynb`` notebook file and write to disk.

    Args:
        nb_path: Path to the source ``.ipynb`` file.
        output_dir: Directory in which to write the output file.
        fmt: Output format string (``'md'`` or ``'txt'``).
        include_outputs: If False, skip all output cells.

    Returns:
        Path to the written output file.
    """
    logger.info("Processing: %s", nb_path)

    try:
        nb_text = nb_path.read_text(encoding='utf-8', errors='replace')
        nb = json.loads(nb_text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in notebook %s: %s", nb_path, exc)
        raise typer.Exit(code=1)

    metadata = extract_nb_metadata(nb, nb_path)
    kernel_language = metadata.get("kernel_language", "python")

    md_content = notebook_to_markdown(nb, kernel_language=kernel_language,
                                      include_outputs=include_outputs)

    if fmt == 'md':
        output = nb_to_full_markdown(md_content, metadata=metadata)
    else:
        output = nb_to_plain_text(md_content)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (nb_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert Jupyter notebooks (.ipynb) to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="Notebook file (.ipynb) or directory containing .ipynb files.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path("."),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    no_outputs: Annotated[bool, typer.Option(
        "--no-outputs",
        help="Skip all cell output blocks (stream, execute_result, errors).",
    )] = False,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert Jupyter notebooks to markdown (default) or plain text.

    Accepts a single .ipynb file or a directory of .ipynb files.
    Produces YAML frontmatter with notebook metadata (kernel language,
    cell counts, title).  Use --no-outputs to omit all cell outputs.
    """
    setup_logging(verbose)

    if input_path.is_dir():
        nb_files: List[Path] = sorted(input_path.glob("*.ipynb"))
        if not nb_files:
            typer.echo(f"No .ipynb files found in {input_path}", err=True)
            raise typer.Exit(1)
    else:
        if not input_path.exists():
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        nb_files = [input_path]

    fmt = format.value
    include_outputs = not no_outputs
    for nb_file in nb_files:
        out = process_nb_file(nb_file, output_dir, fmt,
                              include_outputs=include_outputs)
        typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
