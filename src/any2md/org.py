#!/usr/bin/env python3
"""
org.py - Org-mode to Markdown Converter

Converts .org (Emacs Org-mode) files to markdown (default) or plain text
using a pure-regex pipeline — no external dependencies (no orgparse, no pandoc).

Handles:
- Headings (* / ** / *** etc.)
- TODO/DONE keywords → GitHub-flavoured checkboxes
- Code blocks (#+BEGIN_SRC lang … #+END_SRC)
- Quote blocks (#+BEGIN_QUOTE … #+END_QUOTE)
- Links ([[url][text]] and [[url]])
- Emphasis (*bold*, /italic/, _underline_, +strikethrough+, ~code~, =verbatim=)
- Org tables (pass-through, adding separator row when missing)
- Export keywords (#+TITLE:, #+AUTHOR:, #+DATE:, #+LANGUAGE:) → frontmatter
- Property drawers (:PROPERTIES: … :END:) → stripped
- Tags (:tag1:tag2:) at end of headings → stripped + collected into frontmatter
- Comment lines (# …) → stripped

Usage:
    python org.py [options] <input.org>
    python org.py [options] <directory/>

Examples:
    python org.py notes.org
    python org.py ~/org/ -o ~/notes/
    python org.py project.org -f txt
"""

import logging
import re
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


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

# Export keywords: #+TITLE: value
_RE_KEYWORD = re.compile(
    r'^#\+(\w+):\s*(.*)',
    re.IGNORECASE,
)

# Heading: * Heading  or  ** TODO Heading  :tag1:tag2:
_RE_HEADING = re.compile(
    r'^(\*+)\s+(.+)$',
)

# TODO/DONE keyword at start of heading text
_RE_TODO = re.compile(
    r'^(TODO|DONE)\s+(.*)$',
    re.IGNORECASE,
)

# Tags at end of heading text (e.g. " :tag1:tag2:")
_RE_TAGS = re.compile(
    r'\s+:([\w@#%:]+):\s*$',
)

# Drawer markers
_RE_DRAWER_START = re.compile(r'^\s*:PROPERTIES:\s*$', re.IGNORECASE)
_RE_DRAWER_END = re.compile(r'^\s*:END:\s*$', re.IGNORECASE)

# Block markers
_RE_BEGIN_SRC = re.compile(r'^#\+BEGIN_SRC(?:\s+(\S+))?.*$', re.IGNORECASE)
_RE_END_SRC = re.compile(r'^#\+END_SRC\s*$', re.IGNORECASE)
_RE_BEGIN_QUOTE = re.compile(r'^#\+BEGIN_QUOTE\s*$', re.IGNORECASE)
_RE_END_QUOTE = re.compile(r'^#\+END_QUOTE\s*$', re.IGNORECASE)
_RE_BEGIN_EXAMPLE = re.compile(r'^#\+BEGIN_EXAMPLE\s*$', re.IGNORECASE)
_RE_END_EXAMPLE = re.compile(r'^#\+END_EXAMPLE\s*$', re.IGNORECASE)
_RE_BEGIN_VERSE = re.compile(r'^#\+BEGIN_VERSE\s*$', re.IGNORECASE)
_RE_END_VERSE = re.compile(r'^#\+END_VERSE\s*$', re.IGNORECASE)
_RE_BEGIN_CENTER = re.compile(r'^#\+BEGIN_CENTER\s*$', re.IGNORECASE)
_RE_END_CENTER = re.compile(r'^#\+END_CENTER\s*$', re.IGNORECASE)

# Comment lines (# … but NOT #+ keywords)
_RE_COMMENT = re.compile(r'^#(?!\+)\s')

# Table row
_RE_TABLE_ROW = re.compile(r'^\s*\|')
_RE_TABLE_SEPARATOR = re.compile(r'^\s*\|[-+|]+\s*$')


# ---------------------------------------------------------------------------
# Inline markup
# ---------------------------------------------------------------------------

def _convert_links(text: str) -> str:
    """
    Convert Org-mode links to Markdown.

    ``[[url][description]]`` → ``[description](url)``
    ``[[url]]``              → ``<url>``

    Args:
        text: Line or inline text containing Org links.

    Returns:
        Text with links replaced by Markdown equivalents.
    """
    # [[url][text]] — named link
    text = re.sub(
        r'\[\[([^\]]+)\]\[([^\]]+)\]\]',
        lambda m: f'[{m.group(2)}]({m.group(1)})',
        text,
    )
    # [[url]] — bare link
    text = re.sub(
        r'\[\[([^\]]+)\]\]',
        lambda m: f'<{m.group(1)}>',
        text,
    )
    return text


def _convert_emphasis(text: str) -> str:
    """
    Convert Org-mode inline emphasis markers to Markdown equivalents.

    Org markup rules require a non-space character immediately inside the
    markers. We use a non-greedy pattern that:
    - starts after a word-boundary or start-of-string / punctuation
    - does not allow the content to start or end with whitespace

    Conversions:
    - ``*bold*``           → ``**bold**``
    - ``/italic/``         → ``*italic*``
    - ``_underline_``      → ``*underline*``
    - ``+strikethrough+``  → ``~~strikethrough~~``
    - ``~code~``           → `` `code` ``
    - ``=verbatim=``       → `` `verbatim` ``

    Args:
        text: Inline text (single line) to transform.

    Returns:
        Text with Org emphasis replaced by Markdown.
    """
    # Inner content: at least one char, no leading/trailing space.
    _inner = r'(\S(?:.*?\S)?|\S)'

    # Order matters: process longer/more-specific patterns first.
    # ~code~ and =verbatim= first to avoid nesting confusion.
    text = re.sub(r'~' + _inner + r'~', r'`\1`', text)
    text = re.sub(r'=' + _inner + r'=', r'`\1`', text)

    # +strikethrough+ — avoid matching list markers (- item)
    text = re.sub(r'(?<!\w)\+(' + _inner[1:-1] + r')\+(?!\w)', r'~~\1~~', text)

    # _underline_ — avoid matching snake_case identifiers
    text = re.sub(r'(?<![_\w])_(' + _inner[1:-1] + r')_(?![_\w])', r'*\1*', text)

    # /italic/
    text = re.sub(r'(?<!\w)/(' + _inner[1:-1] + r')/(?!\w)', r'*\1*', text)

    # *bold* — must come after strikethrough/underline/italic so we don't
    # double-escape already-converted markers
    text = re.sub(r'(?<![*\w])\*(' + _inner[1:-1] + r')\*(?![*\w])', r'**\1**', text)

    return text


# ---------------------------------------------------------------------------
# Table processing
# ---------------------------------------------------------------------------

def _process_table(table_lines: List[str]) -> List[str]:
    """
    Pass an Org table through to Markdown, inserting a separator row after
    the header row if one is not already present.

    Org tables look like::

        | Name  | Age |
        |-------+-----|
        | Alice |  30 |

    The separator ``|---+---|`` is already compatible with GitHub Markdown
    if we replace ``+`` with ``|``::

        | Name  | Age |
        | ----- | --- |
        | Alice |  30 |

    Args:
        table_lines: Lines that belong to one contiguous Org table block.

    Returns:
        Transformed table lines ready for Markdown output.
    """
    if not table_lines:
        return table_lines

    out: List[str] = []
    for i, line in enumerate(table_lines):
        stripped = line.strip()
        # Replace Org separator (|---+---| or |---|----|) with proper MD separator
        if _RE_TABLE_SEPARATOR.match(stripped):
            # Normalise: replace + with | and reformat
            sep_line = re.sub(r'[+]', '|', stripped)
            # Replace inner cell content with ---
            sep_line = re.sub(r'\|[-]+', lambda m: '|' + '-' * (len(m.group(0)) - 1), sep_line)
            out.append(sep_line)
        else:
            out.append(stripped)
            # Insert a separator after the first data row if the next row is
            # not already a separator
            if i == 0 and len(table_lines) > 1:
                next_stripped = table_lines[1].strip()
                if not _RE_TABLE_SEPARATOR.match(next_stripped):
                    # Build an auto-separator from the column count
                    col_count = stripped.count('|') - 1
                    sep = '|' + '|'.join([' --- '] * col_count) + '|'
                    out.append(sep)

    return out


# ---------------------------------------------------------------------------
# Main conversion engine
# ---------------------------------------------------------------------------

def org_to_markdown_lines(org_content: str) -> Tuple[List[str], Dict]:
    """
    Convert Org-mode content to Markdown lines and extract metadata.

    This is the core conversion function.  It processes the input line-by-line
    using a small state machine with the following states:

    - ``normal`` — default processing
    - ``src`` — inside a #+BEGIN_SRC … #+END_SRC block
    - ``quote`` — inside a #+BEGIN_QUOTE … #+END_QUOTE block
    - ``example`` — inside a #+BEGIN_EXAMPLE … #+END_EXAMPLE block
    - ``drawer`` — inside a :PROPERTIES: … :END: drawer (stripped)

    Args:
        org_content: Raw Org-mode file content as a string.

    Returns:
        A tuple of:
        - List[str]: Markdown lines (no trailing newline on each)
        - Dict: Extracted metadata (title, author, date, language, tags)
    """
    lines = org_content.splitlines()
    metadata: Dict = {}
    all_tags: List[str] = []

    out: List[str] = []
    state = 'normal'   # normal | src | quote | example | drawer

    # Accumulate consecutive table lines so we can post-process them
    table_buffer: List[str] = []

    def flush_table() -> None:
        """Flush the accumulated table buffer into the output."""
        if table_buffer:
            out.extend(_process_table(table_buffer))
            out.append('')
            table_buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        # ------------------------------------------------------------------ #
        # State: property drawer — strip everything until :END:
        # ------------------------------------------------------------------ #
        if state == 'drawer':
            if _RE_DRAWER_END.match(line):
                state = 'normal'
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # State: source code block
        # ------------------------------------------------------------------ #
        if state == 'src':
            if _RE_END_SRC.match(line):
                out.append('```')
                state = 'normal'
            else:
                out.append(line)
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # State: quote block — prefix every line with "> "
        # ------------------------------------------------------------------ #
        if state == 'quote':
            if _RE_END_QUOTE.match(line):
                state = 'normal'
                out.append('')
            else:
                out.append('> ' + line.strip())
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # State: example / verse / center block — treat as pre-formatted text
        # ------------------------------------------------------------------ #
        if state in ('example', 'verse', 'center'):
            end_re = {
                'example': _RE_END_EXAMPLE,
                'verse': _RE_END_VERSE,
                'center': _RE_END_CENTER,
            }[state]
            if end_re.match(line):
                out.append('```')
                state = 'normal'
            else:
                out.append(line)
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Normal state — line-by-line dispatch
        # ------------------------------------------------------------------ #

        # --- Property drawer start ---
        if _RE_DRAWER_START.match(line):
            flush_table()
            state = 'drawer'
            i += 1
            continue

        # --- Code block start ---
        m = _RE_BEGIN_SRC.match(line)
        if m:
            flush_table()
            lang = m.group(1) or ''
            out.append(f'```{lang}')
            state = 'src'
            i += 1
            continue

        # --- Quote block start ---
        if _RE_BEGIN_QUOTE.match(line):
            flush_table()
            state = 'quote'
            i += 1
            continue

        # --- Example block start ---
        if _RE_BEGIN_EXAMPLE.match(line):
            flush_table()
            out.append('```')
            state = 'example'
            i += 1
            continue

        # --- Verse block start ---
        if _RE_BEGIN_VERSE.match(line):
            flush_table()
            out.append('```')
            state = 'verse'
            i += 1
            continue

        # --- Center block start ---
        if _RE_BEGIN_CENTER.match(line):
            flush_table()
            out.append('```')
            state = 'center'
            i += 1
            continue

        # --- Export keywords (#+TITLE: etc.) ---
        km = _RE_KEYWORD.match(line)
        if km:
            flush_table()
            keyword = km.group(1).upper()
            value = km.group(2).strip()
            _KNOWN_KEYWORDS = {'TITLE', 'AUTHOR', 'DATE', 'LANGUAGE', 'EMAIL', 'DESCRIPTION'}
            if keyword in _KNOWN_KEYWORDS and value:
                metadata[keyword.lower()] = value
            # Always strip export keywords from body output
            i += 1
            continue

        # --- Comment lines (# … but NOT #+ keywords) ---
        if _RE_COMMENT.match(line) or line.strip() == '#':
            flush_table()
            i += 1
            continue

        # --- Headings ---
        hm = _RE_HEADING.match(line)
        if hm:
            flush_table()
            level = len(hm.group(1))
            heading_text = hm.group(2)

            # Extract tags from end of heading
            tm = _RE_TAGS.search(heading_text)
            if tm:
                raw_tags = tm.group(1)
                # Tags are colon-separated inside the outer colons
                tags = [t for t in raw_tags.split(':') if t]
                all_tags.extend(tags)
                heading_text = heading_text[:tm.start()]

            # Handle TODO/DONE keyword
            todo_m = _RE_TODO.match(heading_text.strip())
            if todo_m:
                keyword = todo_m.group(1).upper()
                rest = todo_m.group(2).strip()
                checkbox = '- [ ]' if keyword == 'TODO' else '- [x]'
                out.append(f'{checkbox} {rest}')
            else:
                prefix = '#' * min(level, 6)
                out.append(f'{prefix} {heading_text.strip()}')

            i += 1
            continue

        # --- Table rows ---
        if _RE_TABLE_ROW.match(line):
            table_buffer.append(line)
            i += 1
            continue
        else:
            # If we were accumulating a table, flush it now
            flush_table()

        # --- Regular lines (apply inline conversion) ---
        converted = _convert_links(line)
        converted = _convert_emphasis(converted)
        out.append(converted)
        i += 1

    # Flush any remaining table
    flush_table()

    # Collect all_tags into metadata
    if all_tags:
        # Deduplicate while preserving order
        seen = set()
        unique_tags = []
        for t in all_tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)
        metadata['tags'] = unique_tags

    return out, metadata


def org_to_markdown_text(org_content: str) -> str:
    """
    Convert Org-mode content to a Markdown body string (no frontmatter).

    Calls ``org_to_markdown_lines`` and joins the result, collapsing
    excessive blank lines.

    Args:
        org_content: Raw Org-mode file content.

    Returns:
        Markdown body string.
    """
    lines, _ = org_to_markdown_lines(org_content)
    text = '\n'.join(lines)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_org_metadata(org_content: str, source_path: Path) -> Dict:
    """
    Extract frontmatter metadata from Org-mode export keywords and tags.

    Recognised export keywords:
    - ``#+TITLE:``    → ``title``
    - ``#+AUTHOR:``   → ``author``
    - ``#+DATE:``     → ``date``
    - ``#+LANGUAGE:`` → ``language``
    - ``#+EMAIL:``    → ``email``
    - ``#+DESCRIPTION:`` → ``description``

    Tags collected from all headings (``* Heading :tag1:tag2:``) are
    deduplicated and stored as a list under ``tags``.

    Args:
        org_content: Raw Org-mode file content.
        source_path: Filesystem path of the source ``.org`` file.

    Returns:
        Metadata dict suitable for ``build_frontmatter()``.
    """
    _, inline_meta = org_to_markdown_lines(org_content)

    metadata: Dict = {
        'source': str(source_path.resolve()),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    metadata.update(inline_meta)

    # Remove empty values
    return {k: v for k, v in metadata.items() if v is not None and v != '' and v != []}


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def org_to_full_markdown(md_body: str, metadata: Optional[Dict] = None) -> str:
    """
    Assemble the final Markdown output with optional YAML frontmatter.

    Args:
        md_body: Converted Markdown body from ``org_to_markdown_text()``.
        metadata: Optional metadata dict for YAML frontmatter.

    Returns:
        Full Markdown string with frontmatter followed by body.
    """
    parts: List[str] = []
    if metadata:
        parts.append(build_frontmatter(metadata))
        parts.append('')
    parts.append(md_body)
    return '\n'.join(parts)


def org_to_plain_text(md_body: str) -> str:
    """
    Strip Markdown syntax from the converted body to produce plain text.

    Removes ATX heading markers, code fences, bold/italic markers, and
    collapses excessive blank lines.

    Args:
        md_body: Markdown body string.

    Returns:
        Plain text string.
    """
    text = md_body
    # Remove heading markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove code fences
    text = re.sub(r'^```[^\n]*\n?', '', text, flags=re.MULTILINE)
    # Remove bold markers (** … **)
    text = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', text)
    # Remove italic/underline markers (* … *)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Remove strikethrough (~~…~~)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # Remove inline code (`…`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Single-file processor
# ---------------------------------------------------------------------------

def process_org_file(org_path: Path, output_dir: Path, fmt: str) -> Path:
    """
    Convert one Org-mode file to the requested output format and write to disk.

    Args:
        org_path: Path to the source ``.org`` file.
        output_dir: Directory in which to write the output file.
        fmt: Output format string (``'md'`` or ``'txt'``).

    Returns:
        Path to the written output file.
    """
    logger.info("Processing: %s", org_path)

    org_content = org_path.read_text(encoding='utf-8', errors='replace')
    metadata = extract_org_metadata(org_content, org_path)
    md_body = org_to_markdown_text(org_content)

    if fmt == 'md':
        output = org_to_full_markdown(md_body, metadata=metadata)
    else:
        output = org_to_plain_text(md_body)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (org_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert Org-mode (.org) files to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="Org-mode file or directory containing .org files.",
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
    Convert Org-mode files to markdown (default) or plain text.

    Accepts a single .org file or a directory of .org files.
    Produces YAML frontmatter from Org export keywords (#+TITLE:, #+AUTHOR:,
    #+DATE:, #+LANGUAGE:) and heading tags (:tag1:tag2:).
    """
    setup_logging(verbose)

    if input_path.is_dir():
        org_files: List[Path] = sorted(input_path.glob("*.org"))
        if not org_files:
            typer.echo(f"No .org files found in {input_path}", err=True)
            raise typer.Exit(1)
    else:
        if not input_path.exists():
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        org_files = [input_path]

    fmt = format.value
    for org_file in org_files:
        out = process_org_file(org_file, output_dir, fmt)
        typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
