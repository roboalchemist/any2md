#!/usr/bin/env python3
"""
db.py - SQLite to Markdown Converter

Converts a SQLite database to markdown by discovering all tables (and optionally
views), rendering each table's schema as a fenced SQL block, and showing sample
rows as a markdown table.

Usage:
    python db.py [options] <database.db>

Examples:
    python db.py myapp.db
    python db.py myapp.db -o ~/notes/
    python db.py myapp.db --max-rows 5 --max-tables 20
    python db.py myapp.db --skip-views -f txt
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Maximum display length for text values before truncation
_MAX_CELL_LEN = 80


# ---------------------------------------------------------------------------
# Cell value rendering
# ---------------------------------------------------------------------------

def render_cell(value: Any, max_len: int = _MAX_CELL_LEN) -> str:
    """
    Render a single cell value as a safe markdown table cell string.

    Rules:
    - None → empty string
    - bytes (BLOB) → "[BLOB: N bytes]"
    - str longer than max_len → truncated with "…"
    - Pipe characters escaped to avoid breaking the markdown table

    Args:
        value: Raw value from sqlite3 cursor
        max_len: Maximum character length before truncation

    Returns:
        String safe for embedding in a markdown table cell
    """
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        if isinstance(value, memoryview):
            value = bytes(value)
        return f"[BLOB: {len(value)} bytes]"
    text = str(value)
    if len(text) > max_len:
        text = text[:max_len] + "…"
    # Escape pipe to prevent breaking table columns
    text = text.replace("|", "\\|")
    # Replace newlines with space
    text = text.replace("\n", " ").replace("\r", " ")
    return text


# ---------------------------------------------------------------------------
# Schema & table introspection
# ---------------------------------------------------------------------------

def discover_tables(
    conn: sqlite3.Connection,
    include_views: bool = True,
) -> List[Tuple[str, str]]:
    """
    Discover all user-created tables (and optionally views) in the database.

    Uses sqlite_master to find objects; excludes internal sqlite_ tables.

    Args:
        conn: Open SQLite connection
        include_views: If True, include VIEWs alongside TABLEs

    Returns:
        List of (object_type, name) tuples in sqlite_master order
    """
    type_filter = "('table', 'view')" if include_views else "('table')"
    cursor = conn.execute(
        f"SELECT type, name FROM sqlite_master "
        f"WHERE type IN {type_filter} AND name NOT LIKE 'sqlite_%' "
        f"ORDER BY type, name"
    )
    return [(row[0], row[1]) for row in cursor.fetchall()]


def get_table_schema(conn: sqlite3.Connection, name: str) -> str:
    """
    Return the CREATE TABLE/VIEW statement for a named object.

    Args:
        conn: Open SQLite connection
        name: Table or view name

    Returns:
        CREATE statement string, or empty string if not found
    """
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else ""


def get_column_names(conn: sqlite3.Connection, name: str) -> List[str]:
    """
    Return column names for a table or view via PRAGMA table_info().

    Args:
        conn: Open SQLite connection
        name: Table or view name

    Returns:
        Ordered list of column name strings
    """
    cursor = conn.execute(f"PRAGMA table_info(\"{name}\")")
    return [row[1] for row in cursor.fetchall()]


def get_row_count(conn: sqlite3.Connection, name: str) -> int:
    """
    Return the total number of rows in a table or view.

    Args:
        conn: Open SQLite connection
        name: Table or view name

    Returns:
        Row count as integer
    """
    cursor = conn.execute(f'SELECT COUNT(*) FROM "{name}"')
    row = cursor.fetchone()
    return row[0] if row else 0


def get_sample_rows(
    conn: sqlite3.Connection,
    name: str,
    limit: int,
) -> List[Tuple]:
    """
    Fetch up to `limit` sample rows from a table or view.

    Args:
        conn: Open SQLite connection
        name: Table or view name
        limit: Maximum number of rows to return

    Returns:
        List of row tuples (raw sqlite3 values)
    """
    cursor = conn.execute(f'SELECT * FROM "{name}" LIMIT {limit}')
    return cursor.fetchall()


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------

def format_schema_block(schema_sql: str) -> str:
    """
    Render a CREATE statement as a fenced SQL code block.

    Args:
        schema_sql: Raw SQL string from sqlite_master

    Returns:
        Fenced markdown code block string
    """
    if not schema_sql:
        return "_No schema available._\n"
    return f"```sql\n{schema_sql.strip()}\n```\n"


def format_sample_table(
    columns: List[str],
    rows: List[Tuple],
    total_rows: int,
    shown: int,
) -> str:
    """
    Render sample rows as a GitHub-flavored markdown table.

    Appends a note if only a subset of rows is shown.

    Args:
        columns: Column name list
        rows: Row tuples from get_sample_rows()
        total_rows: Total row count in the table
        shown: Maximum rows shown (max_rows CLI option)

    Returns:
        Markdown table string, or a "no rows" message if the table is empty
    """
    if not rows:
        return "_No rows._\n"

    # Header
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines = [header, separator]
    for row in rows:
        cells = [render_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")

    result = "\n".join(lines) + "\n"

    if total_rows > shown:
        result += f"\n_Showing {shown} of {total_rows} rows._\n"
    else:
        result += f"\n_{total_rows} row{'s' if total_rows != 1 else ''}._\n"

    return result


def format_table_section(
    obj_type: str,
    name: str,
    schema_sql: str,
    columns: List[str],
    rows: List[Tuple],
    total_rows: int,
    max_rows: int,
) -> str:
    """
    Render a complete markdown section for one table or view.

    Structure:
        ## table_name  (type badge)

        ### Schema
        ```sql
        CREATE TABLE ...
        ```

        ### Data (N rows)
        | col1 | col2 | ...

    Args:
        obj_type: "table" or "view"
        name: Object name
        schema_sql: CREATE statement
        columns: Column names
        rows: Sample rows
        total_rows: Total row count
        max_rows: Row limit for the sample

    Returns:
        Markdown string for the section
    """
    type_badge = f" _(view)_" if obj_type == "view" else ""
    lines = [
        f"## `{name}`{type_badge}",
        "",
        "### Schema",
        "",
        format_schema_block(schema_sql),
        f"### Data ({total_rows} row{'s' if total_rows != 1 else ''})",
        "",
        format_sample_table(columns, rows, total_rows, max_rows),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Database-level extraction
# ---------------------------------------------------------------------------

def extract_db_info(
    db_path: Path,
    max_rows: int = 10,
    max_tables: int = 50,
    include_views: bool = True,
) -> Tuple[Dict, str]:
    """
    Open the database and extract metadata + rendered markdown body.

    Args:
        db_path: Path to the SQLite file
        max_rows: Maximum sample rows per table
        max_tables: Maximum number of tables to process
        include_views: Whether to include VIEW objects

    Returns:
        Tuple of (metadata_dict, markdown_body_string)
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None  # keep raw tuples

    try:
        objects = discover_tables(conn, include_views=include_views)
        if len(objects) > max_tables:
            logger.warning(
                "Database has %d objects; limiting to first %d (--max-tables).",
                len(objects), max_tables,
            )
            objects = objects[:max_tables]

        table_count = len(objects)
        total_rows_all = 0
        sections: List[str] = []

        for obj_type, name in objects:
            logger.debug("Processing %s: %s", obj_type, name)
            schema_sql = get_table_schema(conn, name)
            columns = get_column_names(conn, name)

            try:
                total_rows = get_row_count(conn, name)
            except sqlite3.OperationalError as exc:
                logger.warning("Could not count rows in %s: %s", name, exc)
                total_rows = 0

            try:
                rows = get_sample_rows(conn, name, max_rows)
            except sqlite3.OperationalError as exc:
                logger.warning("Could not read rows from %s: %s", name, exc)
                rows = []

            total_rows_all += total_rows

            section = format_table_section(
                obj_type=obj_type,
                name=name,
                schema_sql=schema_sql,
                columns=columns,
                rows=rows,
                total_rows=total_rows,
                max_rows=max_rows,
            )
            sections.append(section)

    finally:
        conn.close()

    metadata: Dict = {
        "source": str(db_path.resolve()),
        "file_size": db_path.stat().st_size,
        "table_count": table_count,
        "total_rows": total_rows_all,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    body = "\n\n---\n\n".join(sections)
    return metadata, body


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def db_to_markdown(metadata: Dict, body: str) -> str:
    """
    Assemble YAML frontmatter + markdown body for a database dump.

    Args:
        metadata: Frontmatter fields (source, file_size, table_count, etc.)
        body: Rendered table sections

    Returns:
        Full markdown string
    """
    fm = build_frontmatter(metadata)
    db_name = Path(metadata["source"]).name
    lines = [
        fm,
        "",
        f"# {db_name}",
        "",
        body,
    ]
    return "\n".join(lines)


def db_to_plain_text(body: str) -> str:
    """
    Strip markdown decorations for plain text output.

    Removes heading markers, fences, and table pipes.

    Args:
        body: Markdown body from db_to_markdown()

    Returns:
        Plain-text string
    """
    import re

    text = body
    # Remove ATX headings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove code fences
    text = re.sub(r'^```[^\n]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    # Remove markdown table delimiters (leading/trailing pipes)
    text = re.sub(r'^\|(.+)\|$', lambda m: m.group(1), text, flags=re.MULTILINE)
    # Remove separator rows (--- | --- | ---)
    text = re.sub(r'^[\s\-|]+$', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    # Remove inline code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Single-file processor
# ---------------------------------------------------------------------------

def process_db_file(
    db_path: Path,
    output_dir: Path,
    fmt: str,
    max_rows: int,
    max_tables: int,
    include_views: bool,
) -> Path:
    """
    Convert one SQLite database to the requested output format and write to disk.

    Args:
        db_path: Path to the .db file
        output_dir: Directory in which to write the output file
        fmt: Output format string ('md' or 'txt')
        max_rows: Maximum sample rows per table
        max_tables: Maximum tables to process
        include_views: Whether to include VIEWs

    Returns:
        Path to the written output file
    """
    logger.info("Processing: %s", db_path)

    metadata, body = extract_db_info(
        db_path,
        max_rows=max_rows,
        max_tables=max_tables,
        include_views=include_views,
    )

    if fmt == "md":
        output = db_to_markdown(metadata, body)
    else:
        output = db_to_plain_text(body)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (db_path.stem + "." + fmt)
    write_output(output, out_path)
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert a SQLite database to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="Path to the SQLite database file (.db, .sqlite, etc.).",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path("."),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    max_rows: Annotated[int, typer.Option(
        "--max-rows",
        help="Maximum sample rows to show per table.",
    )] = 10,
    max_tables: Annotated[int, typer.Option(
        "--max-tables",
        help="Maximum number of tables/views to process.",
    )] = 50,
    skip_views: Annotated[bool, typer.Option(
        "--skip-views",
        help="Exclude VIEWs from the output.",
    )] = False,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert a SQLite database to markdown (default) or plain text.

    Discovers all tables and views, renders each schema as a fenced SQL block,
    and shows sample rows as a markdown table. Produces YAML frontmatter with
    source path, file size, table count, total rows, and fetch timestamp.
    """
    setup_logging(verbose)

    if not input_path.exists():
        typer.echo(f"File not found: {input_path}", err=True)
        raise typer.Exit(1)

    out = process_db_file(
        db_path=input_path,
        output_dir=output_dir,
        fmt=format.value,
        max_rows=max_rows,
        max_tables=max_tables,
        include_views=not skip_views,
    )
    typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
