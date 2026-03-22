#!/usr/bin/env python3
"""
csv.py - CSV/TSV to Markdown Converter

Converts .csv and .tsv files to markdown pipe tables (default) or plain
aligned-column text using only Python stdlib (csv module).

Auto-detects delimiter via csv.Sniffer(). Handles pipes in cell data,
multiline cells, empty cells, and large files via row truncation.

Usage:
    python csv.py [options] <input.csv>
    python csv.py [options] <directory/>

Examples:
    python csv.py data.csv
    python csv.py report.tsv -o ~/notes/
    python csv.py data.csv -f txt --max-rows 100 --max-col-width 40
"""

import csv as _csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_ROWS = 500
DEFAULT_MAX_COL_WIDTH = 80

# Delimiter display names for frontmatter
_DELIMITER_NAMES = {
    ',': 'comma',
    '\t': 'tab',
    ';': 'semicolon',
    '|': 'pipe',
    ':': 'colon',
}


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------

def detect_delimiter(content: str) -> str:
    """
    Auto-detect CSV delimiter using csv.Sniffer.

    Falls back to comma if sniffing fails (e.g. single-column file).

    Args:
        content: Raw file content string (first ~4KB is sufficient)

    Returns:
        Single-character delimiter string
    """
    sample = content[:4096]
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=',\t;|:')
        return dialect.delimiter
    except _csv.Error:
        logger.debug("Sniffer failed to detect delimiter, defaulting to comma")
        return ','


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_csv(content: str, delimiter: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse CSV content into header row and data rows.

    Args:
        content: Raw CSV/TSV file content
        delimiter: Field delimiter character

    Returns:
        Tuple of (headers, rows) where headers is a list of column names
        and rows is a list of lists of cell strings. Both may be empty.
    """
    reader = _csv.reader(content.splitlines(), delimiter=delimiter)
    rows_raw = list(reader)

    if not rows_raw:
        return [], []

    headers = rows_raw[0]
    data_rows = rows_raw[1:]
    return headers, data_rows


def sanitize_cell(value: str, max_col_width: int) -> str:
    """
    Sanitize a single cell value for table output.

    - Replaces newlines with a single space
    - Escapes pipe characters as \\|
    - Truncates to max_col_width (appending … if truncated)

    Args:
        value: Raw cell string
        max_col_width: Maximum cell width in characters

    Returns:
        Sanitized cell string safe for use in a markdown pipe table
    """
    # Replace all newline variants with space
    value = value.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    # Escape pipe characters
    value = value.replace('|', r'\|')
    # Truncate
    if len(value) > max_col_width:
        value = value[:max_col_width - 1] + '\u2026'
    return value


def prepare_table(
    headers: List[str],
    rows: List[List[str]],
    max_rows: int,
    max_col_width: int,
) -> Tuple[List[str], List[List[str]], bool]:
    """
    Sanitize headers and rows, apply row truncation.

    Args:
        headers: Raw header strings
        rows: Raw data rows (list of lists)
        max_rows: Maximum number of data rows to include
        max_col_width: Maximum cell width before truncation

    Returns:
        Tuple of (sanitized_headers, sanitized_rows, truncated) where
        truncated=True if rows were cut short.
    """
    san_headers = [sanitize_cell(h, max_col_width) for h in headers]

    truncated = len(rows) > max_rows
    display_rows = rows[:max_rows]

    # Normalise row length to match header count
    col_count = len(headers)
    san_rows: List[List[str]] = []
    for row in display_rows:
        # Pad short rows, trim long rows
        padded = list(row) + [''] * max(0, col_count - len(row))
        padded = padded[:col_count]
        san_rows.append([sanitize_cell(cell, max_col_width) for cell in padded])

    return san_headers, san_rows, truncated


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_csv_metadata(
    csv_path: Path,
    headers: List[str],
    rows: List[List[str]],
    delimiter: str,
) -> Dict:
    """
    Build metadata dict for YAML frontmatter.

    Args:
        csv_path: Path to the source CSV/TSV file
        headers: Parsed column names
        rows: All data rows (before truncation)
        delimiter: Detected delimiter character

    Returns:
        Metadata dict suitable for build_frontmatter()
    """
    return {
        'rows': len(rows),
        'columns': len(headers),
        'column_names': headers if headers else [],
        'delimiter': _DELIMITER_NAMES.get(delimiter, repr(delimiter)),
        'file_size': csv_path.stat().st_size,
        'source': str(csv_path.resolve()),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def table_to_markdown(
    headers: List[str],
    rows: List[List[str]],
    truncated: bool = False,
    total_rows: int = 0,
) -> str:
    """
    Render sanitized headers and rows as a GitHub-flavoured markdown pipe table.

    Args:
        headers: Sanitized header strings
        rows: Sanitized data rows
        truncated: Whether rows were cut short by max_rows
        total_rows: Original row count (shown in truncation notice)

    Returns:
        Markdown table string (no frontmatter)
    """
    if not headers:
        return ''

    # Compute column widths (at least 3 for the separator dashes)
    col_widths = [max(3, len(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    def _row_line(cells: List[str]) -> str:
        padded = [cells[i].ljust(col_widths[i]) if i < len(cells) else ' ' * col_widths[i]
                  for i in range(len(headers))]
        return '| ' + ' | '.join(padded) + ' |'

    lines = []
    lines.append(_row_line(headers))
    lines.append('| ' + ' | '.join('-' * w for w in col_widths) + ' |')
    for row in rows:
        lines.append(_row_line(row))

    if truncated:
        lines.append('')
        lines.append(
            f'> **Note:** Table truncated to {len(rows)} of {total_rows} rows.'
        )

    return '\n'.join(lines)


def table_to_plain_text(
    headers: List[str],
    rows: List[List[str]],
    truncated: bool = False,
    total_rows: int = 0,
) -> str:
    """
    Render sanitized headers and rows as a plain aligned-column text table.

    Uses spaces for alignment, no pipe characters.

    Args:
        headers: Sanitized header strings
        rows: Sanitized data rows
        truncated: Whether rows were cut short by max_rows
        total_rows: Original row count (shown in truncation notice)

    Returns:
        Plain text table string
    """
    if not headers:
        return ''

    col_widths = [max(3, len(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    def _row_line(cells: List[str]) -> str:
        parts = []
        for i in range(len(headers)):
            cell = cells[i] if i < len(cells) else ''
            parts.append(cell.ljust(col_widths[i]))
        return '  '.join(parts).rstrip()

    lines = []
    lines.append(_row_line(headers))
    lines.append('  '.join('-' * w for w in col_widths))
    for row in rows:
        lines.append(_row_line(row))

    if truncated:
        lines.append('')
        lines.append(f'(Table truncated to {len(rows)} of {total_rows} rows.)')

    return '\n'.join(lines)


def rows_to_full_markdown(
    table_md: str,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Assemble final markdown output with optional YAML frontmatter.

    Args:
        table_md: Rendered markdown table string
        metadata: Optional metadata dict for YAML frontmatter

    Returns:
        Full markdown string ready to write to disk
    """
    lines = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append('')

    lines.append(table_md)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Single-file processor
# ---------------------------------------------------------------------------

def process_csv_file(
    csv_path: Path,
    output_dir: Path,
    fmt: str,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_col_width: int = DEFAULT_MAX_COL_WIDTH,
) -> Path:
    """
    Convert one CSV/TSV file to the requested output format and write to disk.

    Args:
        csv_path: Path to the source .csv/.tsv file
        output_dir: Directory in which to write the output file
        fmt: Output format string ('md' or 'txt')
        max_rows: Maximum number of data rows to include
        max_col_width: Maximum column width in characters

    Returns:
        Path to the written output file
    """
    logger.info("Processing: %s", csv_path)

    content = csv_path.read_text(encoding='utf-8', errors='replace')

    if not content.strip():
        logger.warning("Empty file: %s", csv_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (csv_path.stem + '.' + fmt)
        out_path.write_text('', encoding='utf-8')
        return out_path

    delimiter = detect_delimiter(content)
    logger.debug("Detected delimiter: %r", delimiter)

    headers, all_rows = parse_csv(content, delimiter)
    total_rows = len(all_rows)

    san_headers, san_rows, truncated = prepare_table(
        headers, all_rows, max_rows, max_col_width
    )

    if fmt == 'md':
        table_md = table_to_markdown(san_headers, san_rows, truncated, total_rows)
        metadata = extract_csv_metadata(csv_path, headers, all_rows, delimiter)
        output = rows_to_full_markdown(table_md, metadata=metadata)
    else:
        table_txt = table_to_plain_text(san_headers, san_rows, truncated, total_rows)
        output = table_txt

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (csv_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert CSV/TSV files to markdown pipe tables or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="CSV/TSV file or directory containing .csv/.tsv files.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path("."),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown table with frontmatter), [bold]txt[/bold] (plain aligned text).",
    )] = OutputFormat.md,
    max_rows: Annotated[int, typer.Option(
        "--max-rows",
        help="Maximum number of data rows to include (0 = unlimited).",
    )] = DEFAULT_MAX_ROWS,
    max_col_width: Annotated[int, typer.Option(
        "--max-col-width",
        help="Maximum column width in characters before truncation.",
    )] = DEFAULT_MAX_COL_WIDTH,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert CSV/TSV files to markdown pipe tables (default) or plain aligned text.

    Auto-detects delimiter (comma, tab, semicolon, pipe, colon) via csv.Sniffer.
    Produces YAML frontmatter with row count, column names, delimiter, and file size.
    Pipes in cell data are escaped as \\|; multiline cells have newlines collapsed.
    """
    setup_logging(verbose)

    effective_max_rows = max_rows if max_rows > 0 else sys.maxsize

    if input_path.is_dir():
        csv_files: List[Path] = (
            list(input_path.glob("*.csv")) + list(input_path.glob("*.tsv"))
        )
        if not csv_files:
            typer.echo(f"No .csv or .tsv files found in {input_path}", err=True)
            raise typer.Exit(1)
        csv_files = sorted(csv_files)
    else:
        if not input_path.exists():
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        csv_files = [input_path]

    fmt = format.value
    for csv_file in csv_files:
        out = process_csv_file(csv_file, output_dir, fmt, effective_max_rows, max_col_width)
        typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
