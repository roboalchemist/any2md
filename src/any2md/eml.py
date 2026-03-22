#!/usr/bin/env python3
"""
eml.py - Email (.eml / .mbox) to Markdown Converter

Converts RFC 822 email files (.eml) and Unix mbox archives (.mbox) to
markdown with YAML frontmatter.

- .eml  → single markdown file
- .mbox → one markdown file per message (numbered)

Body preference: HTML (converted to markdown via lightweight regex) → plain text.
Attachments are listed in frontmatter; contents are not extracted.

Usage:
    python eml.py [options] <input.eml>
    python eml.py [options] <input.mbox>

Examples:
    python eml.py email.eml
    python eml.py archive.mbox -o ~/notes/
    python eml.py email.eml -f txt
"""

import email
import email.policy
import logging
import mailbox
import re
import sys
from datetime import datetime, timezone
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Header decoding helpers
# ---------------------------------------------------------------------------

def _decode_header_value(raw: Optional[str]) -> str:
    """
    Decode an RFC 2047-encoded email header value to a plain string.

    Args:
        raw: Raw header value string, possibly RFC 2047 encoded.

    Returns:
        Decoded unicode string, or empty string if raw is None.
    """
    if not raw:
        return ''
    parts = decode_header(raw)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            try:
                decoded.append(part.decode(charset or 'utf-8', errors='replace'))
            except (LookupError, UnicodeDecodeError):
                decoded.append(part.decode('latin-1', errors='replace'))
        else:
            decoded.append(part)
    return ' '.join(decoded).strip()


def _parse_address(raw: Optional[str]) -> str:
    """
    Decode and normalise an email address header (From / To / Cc).

    Args:
        raw: Raw address header value (possibly RFC 2047 encoded).

    Returns:
        Human-readable "Name <addr>" string.
    """
    if not raw:
        return ''
    decoded = _decode_header_value(raw)
    name, addr = parseaddr(decoded)
    if name and addr:
        return f'{name} <{addr}>'
    return addr or name or decoded


def _parse_date(raw: Optional[str]) -> str:
    """
    Parse an email Date header into an ISO-8601 UTC string.

    Args:
        raw: Raw Date header value.

    Returns:
        ISO-8601 string (e.g. "2024-01-15T10:30:00Z"), or the raw string
        unchanged if parsing fails.
    """
    if not raw:
        return ''
    try:
        dt = parsedate_to_datetime(raw)
        # Normalise to UTC
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        return raw.strip()


# ---------------------------------------------------------------------------
# HTML → Markdown conversion (no external deps)
# ---------------------------------------------------------------------------

def _html_to_markdown(html: str) -> str:
    """
    Convert an HTML string to approximate Markdown using regex transformations.

    Handles: <p>, <br>, <a href>, <b>/<strong>, <i>/<em>, <ul>/<ol>/<li>,
    <h1>-<h6>, <code>, <pre>, HTML entities. Strips all other tags.

    Args:
        html: HTML input string.

    Returns:
        Markdown-formatted string.
    """
    # Strip <head> and surrounding chrome, keep <body> content
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
    if body_match:
        html = body_match.group(1)

    # Headings h1-h6
    for level in range(6, 0, -1):
        prefix = '#' * level
        html = re.sub(
            rf'<h{level}[^>]*>(.*?)</h{level}>',
            lambda m, p=prefix: f'\n{p} {_strip_tags(m.group(1)).strip()}\n',
            html, flags=re.DOTALL | re.IGNORECASE,
        )

    # Bold and italic (before stripping tags so inner text is preserved)
    html = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', html, flags=re.DOTALL | re.IGNORECASE)

    # Inline code
    html = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html, flags=re.DOTALL | re.IGNORECASE)

    # Code blocks
    html = re.sub(
        r'<pre[^>]*>(.*?)</pre>',
        lambda m: '\n```\n' + _strip_tags(m.group(1)) + '\n```\n',
        html, flags=re.DOTALL | re.IGNORECASE,
    )

    # Links — must come before generic tag stripping
    html = re.sub(
        r'<a[^>]+href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        r'[\2](\1)',
        html, flags=re.DOTALL | re.IGNORECASE,
    )

    # List items
    html = re.sub(
        r'<li[^>]*>(.*?)</li>',
        lambda m: '- ' + _strip_tags(m.group(1)).strip() + '\n',
        html, flags=re.DOTALL | re.IGNORECASE,
    )
    html = re.sub(r'<[uo]l[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</[uo]l>', '\n', html, flags=re.IGNORECASE)

    # Paragraphs → separated blocks
    html = re.sub(
        r'<p[^>]*>(.*?)</p>',
        lambda m: '\n' + _strip_tags(m.group(1)).strip() + '\n',
        html, flags=re.DOTALL | re.IGNORECASE,
    )

    # Line breaks
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)

    # Horizontal rules
    html = re.sub(r'<hr[^>]*/?>',  '\n---\n', html, flags=re.IGNORECASE)

    # Strip remaining tags
    html = _strip_tags(html)

    # Decode common HTML entities
    html = (
        html
        .replace('&amp;', '&')
        .replace('&lt;', '<')
        .replace('&gt;', '>')
        .replace('&quot;', '"')
        .replace('&#39;', "'")
        .replace('&apos;', "'")
        .replace('&nbsp;', ' ')
    )

    # Collapse excessive blank lines
    html = re.sub(r'\n{3,}', '\n\n', html)

    return html.strip()


def _strip_tags(html: str) -> str:
    """Remove all HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', html)


# ---------------------------------------------------------------------------
# Email message parsing
# ---------------------------------------------------------------------------

def _extract_body_and_attachments(
    msg: email.message.Message,
) -> Tuple[str, List[str]]:
    """
    Walk a parsed email message to extract the best body and list attachments.

    Preference order: text/html → text/plain. Parts with a filename or a
    Content-Disposition of "attachment" are listed as attachments.

    Args:
        msg: Parsed email.message.Message object.

    Returns:
        (body_markdown, attachment_names) where body_markdown is already
        converted from HTML if applicable.
    """
    plain: Optional[str] = None
    html: Optional[str] = None
    attachments: List[str] = []

    def _decode_payload(part: email.message.Message) -> str:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ''
        charset = part.get_content_charset() or 'utf-8'
        try:
            return payload.decode(charset, errors='replace')
        except (LookupError, UnicodeDecodeError):
            return payload.decode('latin-1', errors='replace')

    if msg.is_multipart():
        for part in msg.walk():
            disposition = part.get('Content-Disposition', '')
            filename = part.get_filename()

            if filename:
                attachments.append(_decode_header_value(filename))
                continue

            if 'attachment' in disposition.lower():
                # Unnamed attachment
                ct = part.get_content_type()
                attachments.append(f'[unnamed {ct}]')
                continue

            ct = part.get_content_type()
            if ct == 'text/plain' and plain is None:
                plain = _decode_payload(part)
            elif ct == 'text/html' and html is None:
                html = _decode_payload(part)
    else:
        ct = msg.get_content_type()
        if ct == 'text/html':
            html = _decode_payload(msg)
        else:
            plain = _decode_payload(msg)

    if html is not None:
        body = _html_to_markdown(html)
    elif plain is not None:
        body = plain.strip()
    else:
        body = ''

    return body, attachments


def extract_email_metadata(msg: email.message.Message, source_path: Path) -> Dict:
    """
    Build a metadata dict from an email message's headers.

    Args:
        msg: Parsed email.message.Message object.
        source_path: Path to the source file (used for the 'source' field).

    Returns:
        Metadata dict suitable for build_frontmatter().
    """
    subject = _decode_header_value(msg.get('Subject'))
    from_addr = _parse_address(msg.get('From'))
    to_addr = _parse_address(msg.get('To'))
    cc_addr = _parse_address(msg.get('Cc'))
    date_str = _parse_date(msg.get('Date'))
    message_id = (msg.get('Message-ID') or '').strip()
    in_reply_to = (msg.get('In-Reply-To') or '').strip()
    content_type = msg.get_content_type()

    metadata: Dict = {
        'subject': subject or None,
        'from': from_addr or None,
        'to': to_addr or None,
        'cc': cc_addr or None,
        'date': date_str or None,
        'message_id': message_id or None,
        'in_reply_to': in_reply_to or None,
        'content_type': content_type or None,
        'source': str(source_path.resolve()),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    # Strip None/empty values
    return {k: v for k, v in metadata.items() if v is not None and v != ''}


def email_to_markdown(
    msg: email.message.Message,
    source_path: Path,
    fmt: str = 'md',
) -> str:
    """
    Convert a single email.message.Message to a markdown (or plain text) string.

    Args:
        msg: Parsed email message.
        source_path: Path to the source file (for frontmatter).
        fmt: 'md' for markdown with frontmatter, 'txt' for body only.

    Returns:
        Formatted output string.
    """
    metadata = extract_email_metadata(msg, source_path)
    body, attachments = _extract_body_and_attachments(msg)

    if attachments:
        metadata['attachments'] = attachments

    subject = metadata.get('subject', 'Email')

    if fmt == 'md':
        parts = [build_frontmatter(metadata), '']
        parts.append(f'# {subject}')
        parts.append('')
        if body:
            parts.append(body)
        return '\n'.join(parts)
    else:
        lines = []
        if subject:
            lines.append(subject)
            lines.append('=' * len(subject))
            lines.append('')
        if body:
            lines.append(body)
        return '\n'.join(lines).strip()


# ---------------------------------------------------------------------------
# File processors
# ---------------------------------------------------------------------------

def process_eml_file(eml_path: Path, output_dir: Path, fmt: str) -> Path:
    """
    Convert a single .eml file to the requested output format and write to disk.

    Args:
        eml_path: Path to the source .eml file.
        output_dir: Directory in which to write the output file.
        fmt: Output format ('md' or 'txt').

    Returns:
        Path to the written output file.
    """
    logger.info("Processing: %s", eml_path)

    raw = eml_path.read_bytes()
    # Use the default policy for maximum compatibility
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)

    output = email_to_markdown(msg, eml_path, fmt)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (eml_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


def process_mbox_file(
    mbox_path: Path, output_dir: Path, fmt: str
) -> List[Path]:
    """
    Convert all messages in an .mbox file to individual output files.

    Each message is written as <stem>_NNN.<fmt> where NNN is a zero-padded
    sequential number starting at 001.

    Args:
        mbox_path: Path to the source .mbox file.
        output_dir: Directory in which to write the output files.
        fmt: Output format ('md' or 'txt').

    Returns:
        List of paths to written output files.
    """
    logger.info("Processing mbox: %s", mbox_path)

    mbox = mailbox.mbox(str(mbox_path))
    messages = list(mbox)

    if not messages:
        logger.warning("No messages found in %s", mbox_path)
        return []

    logger.info("Found %d message(s)", len(messages))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    width = len(str(len(messages)))

    for idx, msg in enumerate(messages, start=1):
        num = str(idx).zfill(max(width, 3))
        out_name = f'{mbox_path.stem}_{num}.{fmt}'
        out_path = output_dir / out_name
        output = email_to_markdown(msg, mbox_path, fmt)
        out_path.write_text(output, encoding='utf-8')
        logger.info("Written: %s", out_path)
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert email files (.eml / .mbox) to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="Path to a .eml file or .mbox archive.",
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
    Convert email files to markdown (default) or plain text.

    Supports single RFC 822 messages (.eml) and Unix mbox archives (.mbox).
    For .mbox files each message is written to a separate numbered file.
    Attachments are listed in the YAML frontmatter but not extracted.
    """
    setup_logging(verbose)

    if not input_path.exists():
        typer.echo(f"File not found: {input_path}", err=True)
        raise typer.Exit(1)

    fmt = format.value
    suffix = input_path.suffix.lower()

    if suffix == '.mbox':
        written = process_mbox_file(input_path, output_dir, fmt)
        for p in written:
            typer.echo(f"Written: {p}", err=True)
    elif suffix in ('.eml', '.email', ''):
        out = process_eml_file(input_path, output_dir, fmt)
        typer.echo(f"Written: {out}", err=True)
    else:
        # Try to parse as .eml anyway
        logger.warning("Unknown extension %s, treating as .eml", suffix)
        out = process_eml_file(input_path, output_dir, fmt)
        typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
