#!/usr/bin/env python3
"""
man.py - Unix Man Page to Markdown Converter

Converts Unix man pages (.1, .2, ..., .9) to markdown (default) or plain text.

Primary conversion uses `mandoc -T markdown` (macOS built-in, best quality).
If `mandoc -T markdown` is unsupported, falls back to `mandoc -T html` followed
by lightweight HTML-to-markdown stripping.
If `mandoc` is not installed, falls back to pure regex parsing of troff macros.

Usage:
    python man.py [options] <file.1>
    python man.py [options] <directory/>

Examples:
    python man.py /usr/share/man/man1/ls.1
    python man.py /usr/share/man/man1/ -o ~/notes/
    python man.py /usr/share/man/man8/mount.8 -f txt
"""

import logging
import re
import shutil
import subprocess
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

# Man page section file extensions
MAN_EXTENSIONS = {".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"}


# ---------------------------------------------------------------------------
# mandoc-based conversion (primary)
# ---------------------------------------------------------------------------

def _mandoc_available() -> bool:
    """Return True if the mandoc binary is on PATH."""
    return shutil.which("mandoc") is not None


def _mandoc_supports_markdown() -> bool:
    """
    Return True if `mandoc -T markdown` is supported by the installed version.

    Checks by running mandoc with -T markdown on empty input and verifying
    there is no 'unsupported' or 'unknown' error message.
    """
    try:
        result = subprocess.run(
            ["mandoc", "-T", "markdown"],
            input=b"",
            capture_output=True,
            timeout=5,
        )
        # mandoc exits 0 on empty input with -T markdown if the mode is supported.
        # Some older versions print an error and exit non-zero.
        stderr = result.stderr.decode("utf-8", errors="replace").lower()
        if "unknown" in stderr or "unsupported" in stderr or "invalid" in stderr:
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def man_to_markdown_via_mandoc(man_path: Path) -> str:
    """
    Convert a man page file to markdown using `mandoc -T markdown`.

    Args:
        man_path: Path to the man page source file (.1, .2, etc.)

    Returns:
        Markdown string produced by mandoc

    Raises:
        RuntimeError: If mandoc fails or is unavailable
    """
    logger.debug("Using mandoc -T markdown for %s", man_path)
    try:
        result = subprocess.run(
            ["mandoc", "-T", "markdown", str(man_path)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"mandoc failed (exit {result.returncode}): {stderr}")
        return result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        raise RuntimeError("mandoc timed out after 30 seconds")
    except FileNotFoundError:
        raise RuntimeError("mandoc not found on PATH")


def man_to_markdown_via_mandoc_html(man_path: Path) -> str:
    """
    Convert a man page to markdown via `mandoc -T html` + HTML stripping.

    Used as a fallback when `mandoc -T markdown` is not supported.

    Args:
        man_path: Path to the man page source file

    Returns:
        Markdown string converted from mandoc HTML output

    Raises:
        RuntimeError: If mandoc fails
    """
    logger.debug("Using mandoc -T html fallback for %s", man_path)
    try:
        result = subprocess.run(
            ["mandoc", "-T", "html", str(man_path)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"mandoc -T html failed (exit {result.returncode}): {stderr}")
        html = result.stdout.decode("utf-8", errors="replace")
        return _html_to_markdown(html)
    except subprocess.TimeoutExpired:
        raise RuntimeError("mandoc -T html timed out after 30 seconds")
    except FileNotFoundError:
        raise RuntimeError("mandoc not found on PATH")


# ---------------------------------------------------------------------------
# HTML → markdown (for mandoc -T html fallback)
# ---------------------------------------------------------------------------

def _html_to_markdown(html: str) -> str:
    """
    Lightweight HTML→markdown converter for mandoc -T html output.

    Handles the subset of HTML that mandoc produces: headings, paragraphs,
    bold, italic, code, pre blocks, definition lists, tables.

    Args:
        html: HTML string from mandoc -T html

    Returns:
        Approximate markdown string
    """
    # Extract <body> content only
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
    if body_match:
        html = body_match.group(1)

    # Remove wrapper divs and sections
    html = re.sub(r'<div[^>]*>', '', html)
    html = re.sub(r'</div>', '', html)
    html = re.sub(r'<section[^>]*>', '', html)
    html = re.sub(r'</section>', '', html)
    html = re.sub(r'<table[^>]*>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'</table>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'<tr[^>]*>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'</tr>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'<t[dh][^>]*>(.*?)</t[dh]>', lambda m: _strip_tags(m.group(1)).strip() + '  ', html, flags=re.DOTALL | re.IGNORECASE)

    # Headings h1..h6
    for level in range(6, 0, -1):
        prefix = '#' * level
        html = re.sub(
            rf'<h{level}[^>]*>(.*?)</h{level}>',
            lambda m, p=prefix: f'\n{p} {_strip_tags(m.group(1)).strip()}\n',
            html, flags=re.DOTALL | re.IGNORECASE,
        )

    # Bold and italic
    html = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', html, flags=re.DOTALL | re.IGNORECASE)

    # Inline code
    html = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<tt[^>]*>(.*?)</tt>', r'`\1`', html, flags=re.DOTALL | re.IGNORECASE)

    # Pre blocks (code fences)
    html = re.sub(
        r'<pre[^>]*>(.*?)</pre>',
        lambda m: '\n```\n' + _strip_tags(m.group(1)).strip() + '\n```\n',
        html, flags=re.DOTALL | re.IGNORECASE,
    )

    # Links
    html = re.sub(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        r'[\2](\1)',
        html, flags=re.DOTALL | re.IGNORECASE,
    )

    # Definition lists (mandoc uses dl/dt/dd)
    html = re.sub(r'<dl[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</dl>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'<dt[^>]*>(.*?)</dt>', lambda m: '\n**' + _strip_tags(m.group(1)).strip() + '**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<dd[^>]*>(.*?)</dd>', lambda m: '\n:   ' + _strip_tags(m.group(1)).strip() + '\n', html, flags=re.DOTALL | re.IGNORECASE)

    # List items
    html = re.sub(r'<li[^>]*>(.*?)</li>', lambda m: '\n- ' + _strip_tags(m.group(1)).strip(), html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[uo]l[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</[uo]l>', '\n', html, flags=re.IGNORECASE)

    # Paragraphs
    html = re.sub(r'<p[^>]*>(.*?)</p>', lambda m: '\n' + _strip_tags(m.group(1)).strip() + '\n', html, flags=re.DOTALL | re.IGNORECASE)

    # Horizontal rules
    html = re.sub(r'<hr[^>]*/?>',  '\n---\n', html, flags=re.IGNORECASE)

    # Strip remaining tags
    html = _strip_tags(html)

    # Decode HTML entities
    html = (html
            .replace('&amp;', '&')
            .replace('&lt;', '<')
            .replace('&gt;', '>')
            .replace('&quot;', '"')
            .replace('&#39;', "'")
            .replace('&nbsp;', ' ')
            .replace('&mdash;', '—')
            .replace('&ndash;', '–'))

    # Collapse excessive blank lines
    html = re.sub(r'\n{3,}', '\n\n', html)

    return html.strip()


def _strip_tags(html: str) -> str:
    """Remove all HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', html)


# ---------------------------------------------------------------------------
# Regex-based troff/man macro fallback
# ---------------------------------------------------------------------------

def man_to_markdown_regex(content: str) -> Tuple[str, Dict]:
    """
    Parse man page troff macros using regex and produce markdown + metadata.

    Handles the most common man(7) macros:
    - .TH  → frontmatter (name, section, date, source, manual)
    - .SH  → ## SECTION
    - .SS  → ### subsection
    - .B   → **bold**
    - .I   → *italic*
    - .BR  → bold (man cross-reference)
    - .TP  → definition list item
    - .PP / .P → paragraph break
    - .nf / .fi → preformatted code block
    - \\fBtext\\fR → **text**
    - \\fItext\\fR → *text*

    Args:
        content: Raw man page source text (troff/mdoc)

    Returns:
        Tuple of (markdown_string, metadata_dict)
    """
    lines = content.split('\n')
    output: List[str] = []
    metadata: Dict = {}
    in_preformatted = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # --- .TH name section date source manual ---
        th_match = re.match(
            r'^\.TH\s+"?([^"\s]+)"?\s+"?(\d+)"?\s*(?:"?([^"]*)"?)?\s*(?:"?([^"]*)"?)?\s*(?:"?([^"]*)"?)?',
            line,
        )
        if th_match:
            metadata['name'] = th_match.group(1).strip().strip('"') or ''
            metadata['section'] = th_match.group(2).strip().strip('"') or ''
            metadata['date'] = (th_match.group(3) or '').strip().strip('"')
            metadata['source'] = (th_match.group(4) or '').strip().strip('"')
            metadata['manual'] = (th_match.group(5) or '').strip().strip('"')
            # Clean up empties
            metadata = {k: v for k, v in metadata.items() if v}
            i += 1
            continue

        # Skip comment lines
        if line.startswith('.\"') or line.startswith(".'"):
            i += 1
            continue

        # --- .nf (no-fill = preformatted block) ---
        if re.match(r'^\.nf\b', line):
            in_preformatted = True
            output.append('\n```')
            i += 1
            continue

        # --- .fi (fill = end preformatted) ---
        if re.match(r'^\.fi\b', line):
            in_preformatted = False
            output.append('```\n')
            i += 1
            continue

        # --- Inside preformatted block ---
        if in_preformatted:
            # Still apply font escapes for clarity, then pass through
            output.append(_expand_font_escapes(line))
            i += 1
            continue

        # --- .SH SECTION NAME ---
        sh_match = re.match(r'^\.SH\s*(.*)', line)
        if sh_match:
            section = sh_match.group(1).strip().strip('"')
            output.append(f'\n## {section}\n')
            i += 1
            continue

        # --- .SS subsection ---
        ss_match = re.match(r'^\.SS\s*(.*)', line)
        if ss_match:
            subsection = ss_match.group(1).strip().strip('"')
            output.append(f'\n### {subsection}\n')
            i += 1
            continue

        # --- .B text (bold) ---
        b_match = re.match(r'^\.B\s+(.*)', line)
        if b_match:
            text = _expand_font_escapes(b_match.group(1).strip())
            output.append(f'**{text}**')
            i += 1
            continue

        # --- .I text (italic) ---
        italic_match = re.match(r'^\.I\s+(.*)', line)
        if italic_match:
            text = _expand_font_escapes(italic_match.group(1).strip())
            output.append(f'*{text}*')
            i += 1
            continue

        # --- .BR text (bold, used for man cross-references) ---
        br_match = re.match(r'^\.BR\s+(.*)', line)
        if br_match:
            text = _expand_font_escapes(br_match.group(1).strip())
            output.append(f'**{text}**')
            i += 1
            continue

        # --- .IR text (italic + roman alternating) ---
        ir_match = re.match(r'^\.IR\s+(.*)', line)
        if ir_match:
            text = _expand_font_escapes(ir_match.group(1).strip())
            output.append(f'*{text}*')
            i += 1
            continue

        # --- .TP (tagged paragraph / definition list entry) ---
        if re.match(r'^\.TP\b', line):
            # Next non-comment line is the term (may itself be a .B/.I/.BR macro),
            # following text lines are the definition.
            i += 1
            term_line = ''
            while i < len(lines):
                if lines[i].startswith('.\"') or lines[i].startswith(".'"):
                    i += 1
                    continue
                term_line = lines[i]
                i += 1
                break

            # If term line is a formatting macro (.B, .I, .BR, .IR), apply it.
            term = ''
            if term_line:
                b_m = re.match(r'^\.B\s+(.*)', term_line)
                i_m = re.match(r'^\.I\s+(.*)', term_line)
                br_m = re.match(r'^\.BR\s+(.*)', term_line)
                ir_m = re.match(r'^\.IR\s+(.*)', term_line)
                if b_m:
                    term = f'**{_expand_font_escapes(b_m.group(1).strip())}**'
                elif i_m:
                    term = f'*{_expand_font_escapes(i_m.group(1).strip())}*'
                elif br_m:
                    term = f'**{_expand_font_escapes(br_m.group(1).strip())}**'
                elif ir_m:
                    term = f'*{_expand_font_escapes(ir_m.group(1).strip())}*'
                else:
                    term = _expand_font_escapes(term_line.strip())

            # Collect definition lines until next macro
            definition_lines: List[str] = []
            while i < len(lines) and not lines[i].startswith('.'):
                definition_lines.append(_expand_font_escapes(lines[i]))
                i += 1
            definition = ' '.join(ln.strip() for ln in definition_lines if ln.strip())
            if term:
                output.append(f'\n{term}')
            if definition:
                output.append(f':   {definition}\n')
            continue

        # --- .PP / .P (paragraph break) ---
        if re.match(r'^\.PP\b|^\.P\b', line):
            output.append('\n')
            i += 1
            continue

        # --- .LP (like .PP) ---
        if re.match(r'^\.LP\b', line):
            output.append('\n')
            i += 1
            continue

        # --- .RE / .RS (indent/dedent — ignore structural markers) ---
        if re.match(r'^\.R[SE]\b', line):
            i += 1
            continue

        # --- Other macros — skip the dot-line ---
        if re.match(r'^\.[A-Za-z]', line):
            i += 1
            continue

        # --- Regular text line ---
        text = _expand_font_escapes(line)
        output.append(text)
        i += 1

    # Close any unclosed preformatted block
    if in_preformatted:
        output.append('```\n')

    markdown = '\n'.join(output)

    # Clean up excessive blank lines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown.strip(), metadata


def _expand_font_escapes(text: str) -> str:
    """
    Expand troff font escape sequences to markdown equivalents.

    Handles:
    - \\fBtext\\fR  or  \\fBtext\\fP  → **text**
    - \\fItext\\fR  or  \\fItext\\fP  → *text*
    - \\fR          → (reset, remove)
    - \\fP          → (reset, remove)
    - \\-           → hyphen-minus
    - \\(co         → ©
    - \\e           → backslash
    - \\&           → empty (word joiner)

    Args:
        text: Single line or fragment of troff text

    Returns:
        Text with escape sequences expanded to markdown
    """
    # Bold: \fBtext\fR or \fBtext\fP
    text = re.sub(r'\\fB(.*?)\\f[RP]', r'**\1**', text)
    # Italic: \fItext\fR or \fItext\fP
    text = re.sub(r'\\fI(.*?)\\f[RP]', r'*\1*', text)
    # Remaining font resets
    text = re.sub(r'\\f[BIRP]', '', text)
    # Common troff escapes
    text = text.replace('\\-', '-')
    text = text.replace('\\(co', '©')
    text = text.replace('\\e', '\\')
    text = text.replace('\\&', '')
    text = text.replace('\\~', ' ')
    return text


# ---------------------------------------------------------------------------
# Unified conversion entry point
# ---------------------------------------------------------------------------

def man_to_markdown_text(man_path: Path) -> Tuple[str, Dict]:
    """
    Convert a man page file to markdown using the best available method.

    Tries in order:
    1. mandoc -T markdown (preferred, best quality)
    2. mandoc -T html → HTML-to-markdown stripping (mandoc available, old version)
    3. Regex-based troff macro parser (no mandoc installed)

    Args:
        man_path: Path to the man page source file

    Returns:
        Tuple of (markdown_string, metadata_dict_from_TH_macro)
        Note: when using mandoc, metadata is extracted from file content separately.
    """
    content = man_path.read_text(encoding='utf-8', errors='replace')

    if _mandoc_available():
        if _mandoc_supports_markdown():
            logger.debug("mandoc -T markdown available")
            try:
                md = man_to_markdown_via_mandoc(man_path)
                # Still extract metadata from raw content for frontmatter
                _, metadata = man_to_markdown_regex(content)
                return md, metadata
            except RuntimeError as exc:
                logger.warning("mandoc -T markdown failed (%s), trying HTML fallback", exc)

        # mandoc available but -T markdown not supported
        try:
            md = man_to_markdown_via_mandoc_html(man_path)
            _, metadata = man_to_markdown_regex(content)
            return md, metadata
        except RuntimeError as exc:
            logger.warning("mandoc -T html failed (%s), falling back to regex parser", exc)

    # Pure regex fallback
    logger.debug("Using regex troff parser for %s", man_path)
    return man_to_markdown_regex(content)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_man_metadata(man_path: Path) -> Dict:
    """
    Extract metadata from a man page for YAML frontmatter.

    Parses the .TH macro and supplements with file information.

    Frontmatter fields:
    - name: command/topic name (from .TH)
    - section: manual section number (from .TH)
    - date: last revision date (from .TH)
    - source: software source e.g. "GNU coreutils" (from .TH)
    - manual: manual title e.g. "User Commands" (from .TH)
    - source_file: absolute path to the input man page
    - fetched_at: ISO-8601 timestamp of conversion

    Args:
        man_path: Path to the man page file

    Returns:
        Metadata dict suitable for build_frontmatter()
    """
    content = man_path.read_text(encoding='utf-8', errors='replace')
    _, th_meta = man_to_markdown_regex(content)

    metadata: Dict = {
        'source_file': str(man_path.resolve()),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    # Add .TH fields if present
    for field in ('name', 'section', 'date', 'source', 'manual'):
        if field in th_meta and th_meta[field]:
            metadata[field] = th_meta[field]

    return {k: v for k, v in metadata.items() if v is not None and v != ''}


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def man_to_full_markdown(md_content: str, metadata: Optional[Dict] = None) -> str:
    """
    Assemble final markdown output with optional YAML frontmatter.

    Args:
        md_content: Converted markdown from man_to_markdown_text()
        metadata: Optional metadata dict for YAML frontmatter

    Returns:
        Full markdown string ready to write to disk
    """
    lines = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append('')

    lines.append(md_content)
    return '\n'.join(lines)


def man_to_plain_text(md_content: str) -> str:
    """
    Strip markdown syntax from converted content to produce plain text.

    Args:
        md_content: Markdown string from man_to_markdown_text()

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

def process_man_file(man_path: Path, output_dir: Path, fmt: str) -> Path:
    """
    Convert one man page to the requested output format and write to disk.

    Args:
        man_path: Path to the source man page (.1 through .9)
        output_dir: Directory in which to write the output file
        fmt: Output format string ('md' or 'txt')

    Returns:
        Path to the written output file
    """
    logger.info("Processing: %s", man_path)

    metadata = extract_man_metadata(man_path)

    try:
        md_content, _ = man_to_markdown_text(man_path)
    except Exception as exc:
        logger.error("Conversion failed for %s: %s", man_path, exc)
        raise typer.Exit(code=1)

    if fmt == 'md':
        output = man_to_full_markdown(md_content, metadata=metadata)
    else:
        output = man_to_plain_text(md_content)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (man_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert Unix man pages to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="Man page file (.1-.9) or directory containing man page files.",
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
    Convert Unix man pages to markdown (default) or plain text.

    Accepts a single man page file or a directory of man page files
    (.1 through .9 extensions). Produces YAML frontmatter from the
    .TH macro (name, section, date, source, manual).

    Uses mandoc -T markdown when available (macOS built-in, best quality),
    falls back to mandoc -T html → markdown strip, then pure regex parser.
    """
    setup_logging(verbose)

    if input_path.is_dir():
        man_files: List[Path] = []
        for ext in MAN_EXTENSIONS:
            man_files.extend(input_path.glob(f"*{ext}"))
        if not man_files:
            typer.echo(f"No man page files (.1-.9) found in {input_path}", err=True)
            raise typer.Exit(1)
        man_files = sorted(man_files)
    else:
        if not input_path.exists():
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        man_files = [input_path]

    fmt = format.value
    for man_file in man_files:
        out = process_man_file(man_file, output_dir, fmt)
        typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
