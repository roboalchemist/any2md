#!/usr/bin/env python3
"""
data.py - JSON / YAML / JSONL to Markdown Converter

Converts structured data files (.json, .yaml, .yml, .jsonl) to markdown
(default) or plain text. Applies smart strategy selection:

  - Array of consistent objects → markdown table
  - Small flat dict → key-value bullet list
  - Everything else or large/nested data → fenced code block

Usage:
    python -m any2md.data [options] <input.json>
    python -m any2md.data [options] <input.yaml>
    python -m any2md.data [options] <input.jsonl>

Examples:
    python -m any2md.data data.json
    python -m any2md.data records.jsonl --max-items 50 -o ~/notes/
    python -m any2md.data config.yaml -f txt
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A flat dict is "small" if it has ≤ this many keys (key-value list strategy)
_SMALL_DICT_THRESHOLD = 20

# Maximum nesting depth we inspect when computing nesting_depth metadata
_MAX_DEPTH_SCAN = 64


# ---------------------------------------------------------------------------
# Optional YAML import
# ---------------------------------------------------------------------------

def _load_yaml_module():
    """Return the yaml module if PyYAML is available, else None."""
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def detect_format(path: Path) -> str:
    """
    Detect the data format from the file extension.

    Returns one of: 'json', 'jsonl', 'yaml'.

    Args:
        path: Path to the data file.

    Returns:
        Format string ('json', 'jsonl', or 'yaml').

    Raises:
        ValueError: If the extension is not recognised.
    """
    ext = path.suffix.lower()
    if ext == '.jsonl':
        return 'jsonl'
    if ext in ('.yaml', '.yml'):
        return 'yaml'
    if ext == '.json':
        return 'json'
    raise ValueError(
        f"Unrecognised extension '{ext}'. Supported: .json, .jsonl, .yaml, .yml"
    )


def parse_json(text: str) -> Any:
    """Parse a JSON string and return the Python object."""
    return json.loads(text)


def parse_jsonl(text: str) -> List[Any]:
    """
    Parse a JSON Lines file (one JSON value per non-blank line).

    Args:
        text: Raw file content.

    Returns:
        List of parsed JSON objects, one per non-blank line.
    """
    results = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL parse error on line {i}: {exc}") from exc
    return results


def parse_yaml(text: str) -> Any:
    """
    Parse a YAML string.

    Args:
        text: Raw YAML file content.

    Returns:
        Parsed Python object.

    Raises:
        ImportError: If PyYAML is not installed.
        ValueError: If the YAML is invalid.
    """
    yaml = _load_yaml_module()
    if yaml is None:
        raise ImportError(
            "PyYAML is required to parse YAML files. "
            "Install it with: pip install pyyaml"
        )
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc


def load_data(path: Path, fmt: str) -> Tuple[Any, str]:
    """
    Load and parse a data file.

    Args:
        path: Path to the file.
        fmt: One of 'json', 'jsonl', 'yaml'.

    Returns:
        Tuple of (parsed_data, raw_text).
    """
    text = path.read_text(encoding='utf-8', errors='replace')
    if fmt == 'jsonl':
        return parse_jsonl(text), text
    if fmt == 'yaml':
        return parse_yaml(text), text
    return parse_json(text), text


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _nesting_depth(obj: Any, _depth: int = 0) -> int:
    """Recursively compute the maximum nesting depth of a Python object."""
    if _depth > _MAX_DEPTH_SCAN:
        return _depth
    if isinstance(obj, dict):
        if not obj:
            return _depth
        return max(_nesting_depth(v, _depth + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return _depth
        return max(_nesting_depth(item, _depth + 1) for item in obj)
    return _depth


def _is_array_of_consistent_objects(data: Any, max_items: int) -> bool:
    """
    Return True if *data* is a non-empty list of dicts that all share the
    same set of keys (suitable for a markdown table).

    Only the first *max_items* rows are inspected.

    Args:
        data: Parsed data to test.
        max_items: Maximum number of items to consider.

    Returns:
        True if data qualifies for table rendering.
    """
    if not isinstance(data, list) or not data:
        return False
    rows = data[:max_items]
    if not all(isinstance(row, dict) for row in rows):
        return False
    first_keys = set(rows[0].keys())
    return all(set(row.keys()) == first_keys for row in rows)


def _is_small_flat_dict(data: Any) -> bool:
    """
    Return True if *data* is a dict with only scalar values and ≤
    _SMALL_DICT_THRESHOLD keys.

    Args:
        data: Parsed data to test.

    Returns:
        True if data qualifies for key-value list rendering.
    """
    if not isinstance(data, dict):
        return False
    if len(data) > _SMALL_DICT_THRESHOLD:
        return False
    return all(not isinstance(v, (dict, list)) for v in data.values())


# ---------------------------------------------------------------------------
# Rendering strategies
# ---------------------------------------------------------------------------

def _escape_md_cell(value: Any) -> str:
    """Escape a value for use inside a markdown table cell."""
    s = str(value) if not isinstance(value, str) else value
    # Replace pipe characters to avoid breaking table structure
    return s.replace('|', '\\|').replace('\n', ' ')


def render_table(data: List[Dict], max_items: int) -> str:
    """
    Render a list of dicts as a GFM markdown table.

    Args:
        data: List of dicts (all sharing the same keys).
        max_items: Maximum number of rows to include.

    Returns:
        Markdown table string.
    """
    rows = data[:max_items]
    truncated = len(data) > max_items

    headers = list(rows[0].keys())
    header_row = '| ' + ' | '.join(headers) + ' |'
    sep_row = '| ' + ' | '.join('---' for _ in headers) + ' |'

    body_rows = []
    for row in rows:
        cells = [_escape_md_cell(row.get(h, '')) for h in headers]
        body_rows.append('| ' + ' | '.join(cells) + ' |')

    lines = [header_row, sep_row] + body_rows
    if truncated:
        lines.append('')
        lines.append(
            f'*Showing {max_items} of {len(data)} items. '
            f'Use --max-items to adjust.*'
        )
    return '\n'.join(lines)


def render_key_value(data: Dict) -> str:
    """
    Render a flat dict as a markdown bullet list of key: value pairs.

    Args:
        data: Dict with scalar values.

    Returns:
        Markdown bullet list string.
    """
    lines = []
    for key, value in data.items():
        lines.append(f'- **{key}**: {value}')
    return '\n'.join(lines)


def render_code_block(data: Any, fmt: str) -> str:
    """
    Render arbitrary data as a fenced markdown code block.

    Uses JSON pretty-printing regardless of source format (simpler and
    universally readable).

    Args:
        data: Any parsed Python object.
        fmt: Original format hint (for the fence language tag).

    Returns:
        Fenced code block string.
    """
    lang = 'json' if fmt != 'yaml' else 'yaml'
    if fmt == 'yaml':
        yaml = _load_yaml_module()
        if yaml is not None:
            try:
                body = yaml.dump(data, default_flow_style=False, allow_unicode=True)
                return f'```{lang}\n{body.rstrip()}\n```'
            except Exception:
                pass  # fall through to JSON rendering
    body = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    return f'```json\n{body}\n```'


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def choose_strategy(data: Any, max_items: int) -> str:
    """
    Select the best rendering strategy for *data*.

    Returns one of: 'table', 'key_value', 'code_block'.

    Args:
        data: Parsed data.
        max_items: Item limit (used to evaluate table eligibility).

    Returns:
        Strategy name string.
    """
    if _is_array_of_consistent_objects(data, max_items):
        return 'table'
    if _is_small_flat_dict(data):
        return 'key_value'
    return 'code_block'


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_data_metadata(path: Path, data: Any, raw_text: str, fmt: str) -> Dict:
    """
    Build frontmatter metadata for a data file.

    Args:
        path: Source file path.
        data: Parsed Python object.
        raw_text: Raw file text (for file_size).
        fmt: Detected format string ('json', 'jsonl', 'yaml').

    Returns:
        Metadata dict suitable for build_frontmatter().
    """
    metadata: Dict = {
        'source': str(path.resolve()),
        'format': fmt,
        'file_size': len(raw_text.encode('utf-8')),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    if isinstance(data, dict):
        metadata['top_level_type'] = 'object'
        metadata['key_count'] = len(data)
    elif isinstance(data, list):
        metadata['top_level_type'] = 'array'
        metadata['item_count'] = len(data)
    else:
        metadata['top_level_type'] = type(data).__name__

    metadata['nesting_depth'] = _nesting_depth(data)

    return metadata


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def data_to_markdown(
    data: Any,
    metadata: Dict,
    fmt: str,
    max_items: int,
    title: Optional[str] = None,
) -> str:
    """
    Render parsed data as full markdown with YAML frontmatter.

    Args:
        data: Parsed Python object.
        metadata: Frontmatter metadata dict.
        fmt: Original format string (for code-block language hint).
        max_items: Maximum array items to render.
        title: Optional H1 title to include after frontmatter.

    Returns:
        Full markdown string.
    """
    strategy = choose_strategy(data, max_items)
    logger.debug("Rendering strategy: %s", strategy)

    if strategy == 'table':
        body = render_table(data, max_items)
    elif strategy == 'key_value':
        body = render_key_value(data)
    else:
        body = render_code_block(data, fmt)

    lines = [build_frontmatter(metadata), '']
    if title:
        lines += [f'# {title}', '']
    lines.append(body)
    return '\n'.join(lines)


def data_to_plain_text(data: Any, fmt: str, max_items: int) -> str:
    """
    Render parsed data as plain text (no frontmatter or markdown syntax).

    Args:
        data: Parsed Python object.
        fmt: Original format string.
        max_items: Maximum array items to render.

    Returns:
        Plain text string.
    """
    strategy = choose_strategy(data, max_items)

    if strategy == 'table':
        # Plain text: strip the | table syntax to tab-separated values
        rows = data[:max_items]
        headers = list(rows[0].keys())
        lines = ['\t'.join(str(h) for h in headers)]
        for row in rows:
            lines.append('\t'.join(str(row.get(h, '')) for h in headers))
        if len(data) > max_items:
            lines.append(f'(Showing {max_items} of {len(data)} items)')
        return '\n'.join(lines)
    elif strategy == 'key_value':
        return '\n'.join(f'{k}: {v}' for k, v in data.items())
    else:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Single-file processor
# ---------------------------------------------------------------------------

def process_data_file(
    data_path: Path,
    output_dir: Path,
    output_fmt: str,
    max_items: int,
) -> Path:
    """
    Convert one data file to the requested output format and write to disk.

    Args:
        data_path: Path to the source .json/.jsonl/.yaml/.yml file.
        output_dir: Directory in which to write the output file.
        output_fmt: Output format string ('md' or 'txt').
        max_items: Maximum number of array items to render.

    Returns:
        Path to the written output file.
    """
    logger.info("Processing: %s", data_path)

    data_fmt = detect_format(data_path)
    logger.debug("Detected format: %s", data_fmt)

    try:
        data, raw_text = load_data(data_path, data_fmt)
    except (ValueError, ImportError) as exc:
        logger.error("Failed to parse %s: %s", data_path, exc)
        raise typer.Exit(code=1)

    metadata = extract_data_metadata(data_path, data, raw_text, data_fmt)

    if output_fmt == 'md':
        output = data_to_markdown(data, metadata, data_fmt, max_items)
    else:
        output = data_to_plain_text(data, data_fmt, max_items)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (data_path.stem + '.' + output_fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert JSON / YAML / JSONL files to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="JSON, JSONL, YAML, or YML file to convert.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path("."),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    max_items: Annotated[int, typer.Option(
        "--max-items",
        help="Maximum number of array items to render (for tables and lists).",
    )] = 100,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert a JSON, YAML, or JSONL file to markdown (default) or plain text.

    Automatically selects the best rendering strategy:

    \b
    - Array of consistent objects → markdown table
    - Small flat dict (≤20 keys, scalar values) → key-value bullet list
    - Everything else → fenced code block

    Produces YAML frontmatter with file_size, format, top_level_type,
    key_count or item_count, nesting_depth, source, and fetched_at.
    """
    setup_logging(verbose)

    if not input_path.exists():
        typer.echo(f"File not found: {input_path}", err=True)
        raise typer.Exit(1)

    if input_path.is_dir():
        typer.echo("Directory input is not supported. Provide a single file.", err=True)
        raise typer.Exit(1)

    try:
        detect_format(input_path)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    out = process_data_file(input_path, output_dir, format.value, max_items)
    typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
