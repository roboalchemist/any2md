#!/usr/bin/env python3
"""
sub.py - Subtitle (SRT/VTT/ASS) to Markdown Converter

Converts subtitle files to timestamped markdown (default) or plain text using
pysubs2 for parsing. Supports SRT, VTT, and ASS/SSA formats.

For ASS files with speaker names in the Name field, speaker attribution is
preserved with consecutive same-speaker lines merged.

Usage:
    python sub.py [options] <input.srt>
    python sub.py [options] <input.ass>
    python sub.py [options] <directory/>

Examples:
    python sub.py interview.srt
    python sub.py panel.ass -o ~/notes/
    python sub.py subtitles/ -f txt
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
from any2md.yt import format_timestamp_md

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Subtitle file extensions supported by pysubs2
SUBTITLE_EXTENSIONS = {'.srt', '.vtt', '.ass', '.ssa'}

# Regex patterns for HTML/ASS tag stripping
_HTML_BOLD_RE = re.compile(r'<b>(.*?)</b>', re.IGNORECASE | re.DOTALL)
_HTML_ITALIC_RE = re.compile(r'<i>(.*?)</i>', re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r'<[^>]+>')

# ASS override code patterns (pysubs2 converts HTML to these internally)
# e.g. {\i1}text{\i0} and {\b1}text{\b0}
_ASS_ITALIC_RE = re.compile(r'\{\\i1\}(.*?)\{\\i0\}', re.DOTALL)
_ASS_BOLD_RE = re.compile(r'\{\\b1\}(.*?)\{\\b0\}', re.DOTALL)
_ASS_OVERRIDE_RE = re.compile(r'\{[^}]*\}')


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def strip_html_tags(text: str, convert_formatting: bool = True) -> str:
    """
    Remove HTML tags from subtitle text, optionally converting formatting tags
    to their markdown equivalents.

    Handles both raw HTML (<b>, <i>) and ASS override codes ({\\b1}...{\\b0},
    {\\i1}...{\\i0}) that pysubs2 uses internally when loading SRT/VTT files.

    <b>bold</b>          -> **bold**   (or {\\b1}bold{\\b0})
    <i>italic</i>        -> *italic*   (or {\\i1}italic{\\i0})
    All other tags/codes -> stripped

    Args:
        text: Subtitle text, possibly containing HTML tags or ASS override codes
        convert_formatting: If True, convert formatting marks to markdown

    Returns:
        Clean text string
    """
    if convert_formatting:
        # ASS override codes (pysubs2 internal representation)
        text = _ASS_BOLD_RE.sub(r'**\1**', text)
        text = _ASS_ITALIC_RE.sub(r'*\1*', text)
        # HTML tags (if raw HTML is present)
        text = _HTML_BOLD_RE.sub(r'**\1**', text)
        text = _HTML_ITALIC_RE.sub(r'*\1*', text)
    # Strip remaining ASS override codes and HTML tags
    text = _ASS_OVERRIDE_RE.sub('', text)
    return _HTML_TAG_RE.sub('', text)


def ms_to_seconds(ms: int) -> float:
    """Convert pysubs2 milliseconds timestamp to seconds."""
    return ms / 1000.0


# ---------------------------------------------------------------------------
# Subtitle loading and parsing
# ---------------------------------------------------------------------------

def load_subtitles(path: Path):
    """
    Load a subtitle file via pysubs2.

    Args:
        path: Path to the subtitle file (.srt, .vtt, .ass, .ssa)

    Returns:
        pysubs2.SSAFile instance

    Raises:
        ImportError: If pysubs2 is not installed
        Exception: If the file cannot be parsed
    """
    try:
        import pysubs2
    except ImportError:
        raise ImportError(
            "pysubs2 is required. Install it with: pip install pysubs2"
        )
    logger.debug("Loading subtitles: %s", path)
    return pysubs2.load(str(path))


def extract_subtitle_metadata(subs, source_path: Path) -> Dict:
    """
    Extract metadata from a loaded SSAFile for use in YAML frontmatter.

    Args:
        subs: pysubs2.SSAFile instance
        source_path: Path to the source subtitle file

    Returns:
        Metadata dict suitable for build_frontmatter()
    """
    events = [e for e in subs if e.is_text]

    # Detect format
    fmt = getattr(subs, 'format', None) or source_path.suffix.lstrip('.')

    # Duration: end time of last event in seconds
    duration: Optional[float] = None
    if events:
        last_end = max(e.end for e in events)
        duration = ms_to_seconds(last_end)

    # Speakers (ASS/SSA only — Name field is non-empty and meaningful)
    speakers: List[str] = []
    if fmt in ('ass', 'ssa'):
        seen = set()
        for e in events:
            name = (e.name or '').strip()
            if name and name not in seen:
                seen.add(name)
                speakers.append(name)

    metadata: Dict = {
        'source': str(source_path.resolve()),
        'subtitle_count': len(events),
        'format': fmt,
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    if duration is not None:
        metadata['duration'] = round(duration, 3)

    if speakers:
        metadata['speakers'] = speakers

    return metadata


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _merge_consecutive_speaker_lines(
    events: List,
) -> List[Tuple[Optional[str], int, str]]:
    """
    Merge consecutive subtitle events from the same speaker into one block.

    Events with the same non-empty Name are merged if they are consecutive.
    The start time of the first event in each group is used.

    Args:
        events: List of pysubs2.SSAEvent objects

    Returns:
        List of (speaker_or_None, start_ms, merged_text) tuples
    """
    merged: List[Tuple[Optional[str], int, str]] = []

    for event in events:
        speaker = (event.name or '').strip() or None
        # Use raw e.text so ASS override codes are preserved for markdown conversion
        text = strip_html_tags(event.text or '').strip()
        if not text:
            continue

        if merged and merged[-1][0] == speaker and speaker is not None:
            # Append to previous block (use space as separator)
            prev_speaker, prev_start, prev_text = merged[-1]
            merged[-1] = (prev_speaker, prev_start, prev_text + ' ' + text)  # type: ignore[index]
        else:
            merged.append((speaker, event.start, text))

    return merged


def subs_to_markdown(subs, metadata: Optional[Dict] = None) -> str:
    """
    Convert a loaded SSAFile to timestamped markdown.

    For plain SRT/VTT (no speaker names):
        **[MM:SS]** text here

    For ASS with speaker names:
        **SPEAKER** [MM:SS]

        text here

    Consecutive same-speaker lines are merged.

    Args:
        subs: pysubs2.SSAFile instance
        metadata: Optional metadata dict for YAML frontmatter

    Returns:
        Markdown string
    """
    events = [e for e in subs if e.is_text]
    fmt = getattr(subs, 'format', 'srt')
    has_speakers = fmt in ('ass', 'ssa') and any(
        (e.name or '').strip() for e in events
    )

    lines: List[str] = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append('')

    if has_speakers:
        merged = _merge_consecutive_speaker_lines(events)
        for speaker, start_ms, text in merged:
            ts = format_timestamp_md(ms_to_seconds(start_ms))
            if speaker:
                lines.append(f'**{speaker}** [{ts}]')
            else:
                lines.append(f'**[{ts}]**')
            lines.append('')
            lines.append(text)
            lines.append('')
    else:
        for event in events:
            # Use e.text (raw with ASS override codes) so formatting is preserved
            raw = event.text or ''
            text = strip_html_tags(raw).strip()
            if not text:
                continue
            ts = format_timestamp_md(ms_to_seconds(event.start))
            lines.append(f'**[{ts}]** {text}')
            lines.append('')

    return '\n'.join(lines)


def subs_to_plain_text(subs) -> str:
    """
    Convert a loaded SSAFile to plain text (no markdown, no frontmatter).

    Consecutive same-speaker lines are merged. Timestamps are omitted.

    Args:
        subs: pysubs2.SSAFile instance

    Returns:
        Plain text string
    """
    events = [e for e in subs if e.is_text]
    fmt = getattr(subs, 'format', 'srt')
    has_speakers = fmt in ('ass', 'ssa') and any(
        (e.name or '').strip() for e in events
    )

    lines: List[str] = []

    if has_speakers:
        merged = _merge_consecutive_speaker_lines(events)
        for speaker, _start_ms, text in merged:
            if speaker:
                lines.append(f'{speaker}: {text}')
            else:
                lines.append(text)
            lines.append('')
    else:
        for event in events:
            text = strip_html_tags(event.text or '', convert_formatting=False).strip()
            if not text:
                continue
            lines.append(text)
            lines.append('')

    return '\n'.join(lines).strip()


# ---------------------------------------------------------------------------
# Single-file processor
# ---------------------------------------------------------------------------

def process_sub_file(sub_path: Path, output_dir: Path, fmt: str) -> Path:
    """
    Convert one subtitle file to the requested output format and write to disk.

    Args:
        sub_path: Path to the source subtitle file
        output_dir: Directory in which to write the output file
        fmt: Output format string ('md' or 'txt')

    Returns:
        Path to the written output file
    """
    logger.info("Processing: %s", sub_path)

    subs = load_subtitles(sub_path)
    metadata = extract_subtitle_metadata(subs, sub_path)

    if fmt == 'md':
        output = subs_to_markdown(subs, metadata=metadata)
    else:
        output = subs_to_plain_text(subs)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (sub_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert subtitle files (SRT/VTT/ASS) to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="Subtitle file (.srt/.vtt/.ass/.ssa) or directory containing subtitle files.",
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
    Convert subtitle files to markdown (default) or plain text.

    Accepts a single subtitle file or a directory of subtitle files.
    Supports SRT, VTT, ASS, and SSA formats via pysubs2.

    ASS files with speaker names produce speaker-attributed output
    with consecutive same-speaker lines merged.
    """
    setup_logging(verbose)

    if input_path.is_dir():
        sub_files: List[Path] = []
        for ext in SUBTITLE_EXTENSIONS:
            sub_files.extend(input_path.glob(f'*{ext}'))
        if not sub_files:
            typer.echo(
                f"No subtitle files (.srt/.vtt/.ass/.ssa) found in {input_path}",
                err=True,
            )
            raise typer.Exit(1)
        sub_files = sorted(sub_files)
    else:
        if not input_path.exists():
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        sub_files = [input_path]

    fmt = format.value
    for sub_file in sub_files:
        out = process_sub_file(sub_file, output_dir, fmt)
        typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
