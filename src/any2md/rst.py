#!/usr/bin/env python3
"""
rst2md.py - reStructuredText to Markdown Converter

Converts .rst (reStructuredText) files to markdown (default) or plain text
using pypandoc (pandoc) as the primary conversion engine, with fallback to
docutils (RST→HTML→strip-tags) if pandoc is unavailable.

Usage:
    python rst2md.py [options] <input.rst>
    python rst2md.py [options] <directory/>

Examples:
    python rst2md.py README.rst
    python rst2md.py docs/ -o ~/notes/
    python rst2md.py CHANGES.rst -f txt
"""

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
# RST → Markdown conversion
# ---------------------------------------------------------------------------

def rst_to_markdown_text(rst_content: str) -> str:
    """
    Convert an RST string to markdown using the best available library.

    Tries in order:
    1. pypandoc (pandoc binary) — most accurate, handles all RST features
    2. docutils → HTML → tag stripping — fallback, handles basics

    Args:
        rst_content: Raw reStructuredText string

    Returns:
        Markdown-formatted string

    Raises:
        RuntimeError: If no conversion library is available
    """
    # Option 1: pypandoc (preferred)
    try:
        import pypandoc
        logger.debug("Using pypandoc for RST→markdown conversion")
        return pypandoc.convert_text(rst_content, 'markdown', format='rst')
    except ImportError:
        logger.debug("pypandoc not available, trying docutils")
    except Exception as exc:
        logger.debug("pypandoc conversion failed (%s), trying docutils", exc)

    # Option 2: docutils → HTML → strip tags
    try:
        from docutils.core import publish_string
        logger.debug("Using docutils for RST→HTML→markdown fallback")
        html_bytes = publish_string(rst_content, writer_name='html')
        html = html_bytes.decode('utf-8') if isinstance(html_bytes, bytes) else html_bytes
        return _html_to_markdown(html)
    except ImportError:
        logger.debug("docutils not available")
    except Exception as exc:
        logger.debug("docutils conversion failed: %s", exc)

    raise RuntimeError(
        "No RST conversion library found. "
        "Install pypandoc and pandoc: pip install pypandoc && python -c "
        "\"import pypandoc; pypandoc.download_pandoc()\""
    )


def _html_to_markdown(html: str) -> str:
    """
    Very lightweight HTML→markdown converter for docutils fallback output.

    Handles the subset of HTML that docutils produces:
    headings, paragraphs, bold, italic, code, lists, hrules.

    Args:
        html: HTML string from docutils

    Returns:
        Approximate markdown string
    """
    # Extract just the <body> content
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
    if body_match:
        html = body_match.group(1)

    # Remove docutils-specific wrapper divs/sections by collapsing them
    html = re.sub(r'<div[^>]*>', '', html)
    html = re.sub(r'</div>', '', html)
    html = re.sub(r'<section[^>]*>', '', html)
    html = re.sub(r'</section>', '', html)

    # Headings h1..h6
    for level in range(6, 0, -1):
        prefix = '#' * level
        html = re.sub(
            rf'<h{level}[^>]*>(.*?)</h{level}>',
            lambda m, p=prefix: f'\n{p} {_strip_tags(m.group(1)).strip()}\n',
            html, flags=re.DOTALL | re.IGNORECASE
        )

    # Bold and italic
    html = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', html, flags=re.DOTALL | re.IGNORECASE)

    # Inline code
    html = re.sub(r'<tt[^>]*>(.*?)</tt>', r'`\1`', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html, flags=re.DOTALL | re.IGNORECASE)

    # Code blocks (pre)
    html = re.sub(
        r'<pre[^>]*>(.*?)</pre>',
        lambda m: '\n```\n' + _strip_tags(m.group(1)) + '\n```\n',
        html, flags=re.DOTALL | re.IGNORECASE
    )

    # Links
    html = re.sub(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        r'[\2](\1)',
        html, flags=re.DOTALL | re.IGNORECASE
    )

    # List items
    html = re.sub(r'<li[^>]*>(.*?)</li>', lambda m: '- ' + _strip_tags(m.group(1)).strip() + '\n', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[uo]l[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</[uo]l>', '\n', html, flags=re.IGNORECASE)

    # Paragraphs
    html = re.sub(r'<p[^>]*>(.*?)</p>', lambda m: '\n' + _strip_tags(m.group(1)).strip() + '\n', html, flags=re.DOTALL | re.IGNORECASE)

    # Horizontal rules
    html = re.sub(r'<hr[^>]*/?>',  '\n---\n', html, flags=re.IGNORECASE)

    # Strip any remaining tags
    html = _strip_tags(html)

    # Decode HTML entities
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')

    # Collapse excessive blank lines
    html = re.sub(r'\n{3,}', '\n\n', html)

    return html.strip()


def _strip_tags(html: str) -> str:
    """Remove all HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', html)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_rst_metadata(rst_content: str, source_path: Path) -> Dict:
    """
    Extract metadata from RST docinfo fields and the document title.

    RST docinfo fields look like::

        :Author: Jane Doe
        :Date: 2026-01-15
        :Version: 1.0

    The first heading line (underlined with ===, ---, ~~~, etc.) becomes
    the document title.

    Args:
        rst_content: Raw RST file content
        source_path: Path to the source .rst file (for the 'source' field)

    Returns:
        Metadata dict suitable for YAML frontmatter via build_frontmatter()
    """
    metadata: Dict = {
        'source': str(source_path.resolve()),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    lines = rst_content.strip().split('\n')

    # Extract document title: first non-blank line followed by an underline
    # of =, -, ~, ^, ", `, _, * or + (RST section adornment chars)
    # The underline must be at least as long as the title text.
    title_chars = r'[=\-~^"`_*+#<>]'
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if (
                next_line
                and re.match(rf'^{title_chars}+$', next_line)
                and len(next_line) >= len(stripped)
            ):
                metadata['title'] = stripped
                break

    # Extract docinfo fields (:Field: value) — case-insensitive field names
    known_fields = {
        'author', 'date', 'version', 'copyright', 'contact',
        'organization', 'status', 'revision', 'dedication', 'abstract',
    }
    for match in re.finditer(r'^:(\w[\w -]*):\s*(.+)$', rst_content, re.MULTILINE):
        field = match.group(1).lower().replace(' ', '_').replace('-', '_')
        value = match.group(2).strip()
        if field in known_fields:
            metadata[field] = value

    # Remove None/empty values
    return {k: v for k, v in metadata.items() if v is not None and v != ''}


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def rst_to_full_markdown(md_content: str, metadata: Optional[Dict] = None,
                         title: Optional[str] = None) -> str:
    """
    Assemble final markdown output with optional YAML frontmatter and title.

    Args:
        md_content: Converted markdown from rst_to_markdown_text()
        metadata: Optional metadata dict for YAML frontmatter
        title: Optional document title as H1 heading (usually embedded in md_content
               already when using pypandoc, so pass None to avoid duplicating)

    Returns:
        Full markdown string ready to write to disk
    """
    lines = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append('')

    if title:
        lines.append(f'# {title}')
        lines.append('')

    lines.append(md_content)
    return '\n'.join(lines)


def rst_to_plain_text(md_content: str) -> str:
    """
    Strip markdown syntax from converted content to produce plain text.

    A lightweight pass that removes the most common markdown decorations
    produced by pandoc: headings (#), bold/italic markers, code fences.

    Args:
        md_content: Markdown string from rst_to_markdown_text()

    Returns:
        Plain text string
    """
    text = md_content
    # Remove ATX headings markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove code fences
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

def process_rst_file(rst_path: Path, output_dir: Path, fmt: str) -> Path:
    """
    Convert one RST file to the requested output format and write to disk.

    Args:
        rst_path: Path to the source .rst file
        output_dir: Directory in which to write the output file
        fmt: Output format string ('md' or 'txt')

    Returns:
        Path to the written output file
    """
    logger.info("Processing: %s", rst_path)

    rst_content = rst_path.read_text(encoding='utf-8', errors='replace')
    metadata = extract_rst_metadata(rst_content, rst_path)

    try:
        md_content = rst_to_markdown_text(rst_content)
    except RuntimeError as exc:
        logger.error("Conversion failed for %s: %s", rst_path, exc)
        raise typer.Exit(code=1)

    if fmt == 'md':
        # pypandoc already embeds the title as # heading in the markdown;
        # don't pass title= to avoid duplication.
        output = rst_to_full_markdown(md_content, metadata=metadata)
    else:
        output = rst_to_plain_text(md_content)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (rst_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert reStructuredText (.rst) files to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="RST file or directory containing .rst/.rest files.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path("."),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert reStructuredText files to markdown (default) or plain text.

    Accepts a single .rst file or a directory of .rst/.rest files.
    Produces YAML frontmatter from RST docinfo fields (Author, Date, Version, etc.)
    and the document title.
    """
    setup_logging(verbose)

    if input_path.is_dir():
        rst_files: List[Path] = (
            list(input_path.glob("*.rst")) + list(input_path.glob("*.rest"))
        )
        if not rst_files:
            typer.echo(f"No .rst or .rest files found in {input_path}", err=True)
            raise typer.Exit(1)
        rst_files = sorted(rst_files)
    else:
        if not input_path.exists():
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        rst_files = [input_path]

    fmt = format.value
    for rst_file in rst_files:
        out = process_rst_file(rst_file, output_dir, fmt)
        typer.echo(f"Written: {out}")


if __name__ == "__main__":
    app()
