#!/usr/bin/env python3
"""
doc2md.py - Office Document to Markdown Converter

Converts DOCX, PPTX, XLSX, and EPUB files to markdown (default) or plain text
using Microsoft's markitdown library. Produces output with YAML frontmatter from
document properties.

Usage:
    python doc2md.py [options] <input.docx>

Examples:
    python doc2md.py report.docx
    python doc2md.py slides.pptx -o ~/notes/
    python doc2md.py data.xlsx -f txt
    python doc2md.py book.epub -o ~/books/
"""

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output, is_json_mode, write_json_error, write_json_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".docx", ".pptx", ".xlsx", ".epub", ".odt", ".rtf"}


def detect_format(path: Path) -> str:
    """
    Return the normalized file format from the path extension.

    Args:
        path: Path to the document file

    Returns:
        Lowercase extension without the dot (e.g. "docx", "pptx")
    """
    return path.suffix.lower().lstrip(".")


def _extract_docx_metadata(path: Path) -> Dict:
    """Extract metadata from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        logger.debug("python-docx not available; skipping DOCX metadata")
        return {}

    doc = Document(str(path))
    props = doc.core_properties

    def _fmt_dt(dt) -> Optional[str]:
        if dt is None:
            return None
        try:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return str(dt)

    result = {
        'title': props.title or None,
        'author': props.author or None,
        'subject': props.subject or None,
        'keywords': props.keywords or None,
        'created': _fmt_dt(props.created),
        'modified': _fmt_dt(props.modified),
    }
    return {k: v for k, v in result.items() if v}


def _extract_pptx_metadata(path: Path) -> Dict:
    """Extract metadata from a PPTX file using python-pptx."""
    try:
        from pptx import Presentation
    except ImportError:
        logger.debug("python-pptx not available; skipping PPTX metadata")
        return {}

    prs = Presentation(str(path))
    props = prs.core_properties
    slides = len(prs.slides)

    def _fmt_dt(dt) -> Optional[str]:
        if dt is None:
            return None
        try:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return str(dt)

    result = {
        'title': props.title or None,
        'author': props.author or None,
        'subject': props.subject or None,
        'slides': slides,
        'created': _fmt_dt(props.created),
        'modified': _fmt_dt(props.modified),
    }
    return {k: v for k, v in result.items() if v is not None and v != 0}


def _extract_xlsx_metadata(path: Path) -> Dict:
    """Extract metadata from an XLSX file using openpyxl."""
    try:
        import openpyxl
    except ImportError:
        logger.debug("openpyxl not available; skipping XLSX metadata")
        return {}

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    props = wb.properties
    sheet_names = wb.sheetnames
    wb.close()

    def _fmt_dt(dt) -> Optional[str]:
        if dt is None:
            return None
        try:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return str(dt)

    result = {
        'title': props.title or None,
        'author': props.creator or None,
        'subject': props.subject or None,
        'keywords': props.keywords or None,
        'sheets': len(sheet_names),
        'sheet_names': sheet_names if sheet_names else None,
        'created': _fmt_dt(props.created),
        'modified': _fmt_dt(props.modified),
    }
    return {k: v for k, v in result.items() if v is not None and v != 0 and v != []}


def extract_doc_metadata(path: Path, fmt: str) -> Dict:
    """
    Extract metadata from a document file based on its format.

    Uses format-specific libraries (python-docx, python-pptx, openpyxl) where
    available, with graceful fallback to filename-only metadata.

    Args:
        path: Path to the document file
        fmt: File format string (e.g. "docx", "pptx", "xlsx")

    Returns:
        Metadata dict suitable for YAML frontmatter
    """
    metadata: Dict = {}

    if fmt == "docx":
        metadata = _extract_docx_metadata(path)
    elif fmt == "pptx":
        metadata = _extract_pptx_metadata(path)
    elif fmt == "xlsx":
        metadata = _extract_xlsx_metadata(path)
    else:
        # epub, odt, rtf — no specialized metadata extraction
        logger.debug("No specialized metadata extractor for format: %s", fmt)

    # Always include format and source
    metadata['format'] = fmt
    metadata['source'] = str(path.resolve())
    metadata['fetched_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    return metadata


def convert_document(path: Path) -> str:
    """
    Convert a document to markdown text using markitdown.

    Args:
        path: Path to the document file

    Returns:
        Markdown text content of the document

    Raises:
        ImportError: If markitdown is not installed
    """
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise ImportError(
            "markitdown is required. Install it with: uv pip install markitdown"
        )

    logger.info("Converting document: %s", path)
    md = MarkItDown()
    result = md.convert(str(path))
    return result.text_content


def doc_to_markdown(content: str, metadata: Optional[Dict] = None,
                    title: Optional[str] = None) -> str:
    """
    Wrap converted document content in markdown with YAML frontmatter.

    Args:
        content: Markdown text from markitdown
        metadata: Optional metadata dict for YAML frontmatter
        title: Optional document title for the H1 heading

    Returns:
        Full markdown string with frontmatter, heading, and content
    """
    lines = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append("")

    if title:
        lines.append(f"# {title}")
        lines.append("")

    lines.append(content)

    return "\n".join(lines)


def doc_to_text(content: str) -> str:
    """
    Return plain text content without frontmatter or headings.

    Args:
        content: Markdown text from markitdown

    Returns:
        Plain text string
    """
    return content


# --- CLI ---

app = typer.Typer(
    help="Convert office documents (DOCX, PPTX, XLSX, EPUB) to markdown using markitdown.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_file: Annotated[str, typer.Argument(
        help="Path to a document file (DOCX, PPTX, XLSX, EPUB, ODT, RTF).",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path.cwd(),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
    json_output: Annotated[bool, typer.Option(
        "--json", "-j",
        help="Output as JSON to stdout instead of writing a file.",
    )] = False,
    fields: Annotated[Optional[str], typer.Option(
        "--fields",
        help="Comma-separated dot-notation fields to include in JSON output (e.g. 'frontmatter,content').",
    )] = None,
) -> None:
    """
    Convert an office document to markdown (default) or plain text.

    Produces output with YAML frontmatter from document properties.
    Supports DOCX, PPTX, XLSX, EPUB, ODT, and RTF formats.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    doc_path = Path(os.path.abspath(input_file))

    if not doc_path.exists():
        if is_json_mode():
            write_json_error("FILE_NOT_FOUND", f"File not found: {doc_path}")
        else:
            logger.error("File not found: %s", doc_path)
        raise typer.Exit(code=1)

    fmt = detect_format(doc_path)
    if f".{fmt}" not in SUPPORTED_FORMATS:
        if is_json_mode():
            write_json_error(
                "INVALID_INPUT",
                f"Unsupported file format: .{fmt} (supported: {', '.join(sorted(SUPPORTED_FORMATS))})",
            )
        else:
            logger.error(
                "Unsupported file format: .%s (supported: %s)",
                fmt,
                ", ".join(sorted(SUPPORTED_FORMATS)),
            )
        raise typer.Exit(code=1)

    # Extract metadata
    metadata = extract_doc_metadata(doc_path, fmt)
    logger.info("Extracted metadata for %s document", fmt.upper())

    # Convert document
    try:
        content = convert_document(doc_path)
    except ImportError as exc:
        if is_json_mode():
            write_json_error("MISSING_DEPENDENCY", str(exc))
        else:
            logger.error("%s", exc)
        raise typer.Exit(code=1)

    logger.info("Conversion complete (%d chars)", len(content))

    if json_output or is_json_mode():
        write_json_output(metadata, content, doc_path, "doc", fields)
        return

    # Determine output filename
    os.makedirs(str(output_dir), exist_ok=True)
    base_name = doc_path.stem
    ext = format.value
    output_file = Path(output_dir) / f"{base_name}.{ext}"

    # Format and write
    title = metadata.get('title') or base_name
    if format == OutputFormat.md:
        output_content = doc_to_markdown(content, metadata=metadata, title=title)
    else:
        output_content = doc_to_text(content)

    write_output(output_content, output_file)
    logger.info("Output saved to: %s", output_file)


if __name__ == "__main__":
    app()
