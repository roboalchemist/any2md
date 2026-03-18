#!/usr/bin/env python3
"""
cli.py — Unified 2md CLI entry point

Auto-detects input type by extension/URL and dispatches to the appropriate
converter. Also exposes each converter as an explicit subcommand.

Usage:
    2md video.mp4                    # auto-detect -> yt
    2md document.pdf                 # auto-detect -> pdf
    2md https://example.com          # auto-detect -> web

    2md yt video.mp4 --diarize       # explicit subcommand
    2md pdf document.pdf --pages 1-10
    2md img photo.jpg
"""

import re
import sys
from pathlib import Path

import typer

from tomd.yt import app as yt_app
from tomd.pdf import app as pdf_app
from tomd.img import app as img_app
from tomd.web import app as web_app
from tomd.html import app as html_app
from tomd.doc import app as doc_app
from tomd.rst import app as rst_app

# Extension -> tool mapping
_AUDIO_VIDEO_EXTS = {".mp3", ".wav", ".mp4", ".webm", ".m4a", ".flac", ".ogg", ".aac", ".mov", ".avi", ".mkv"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
_DOC_EXTS = {".docx", ".pptx", ".xlsx", ".epub", ".odt", ".rtf"}
_YOUTUBE_PATTERN = re.compile(
    r'(?:youtube\.com/(?:[^/\n\s]+/\S+/|(?:v|e(?:mbed)?)/'
    r'|\S*?[?&]v=)|youtu\.be/|^[a-zA-Z0-9_-]{11}$)'
)

_SUBCOMMANDS = {"yt", "pdf", "img", "web", "html", "doc", "rst"}

_TOOL_APPS = {
    "yt": yt_app,
    "pdf": pdf_app,
    "img": img_app,
    "web": web_app,
    "html": html_app,
    "doc": doc_app,
    "rst": rst_app,
}


def _detect_tool(input_path: str) -> str:
    """Return tool name for the given input, or empty string if unknown."""
    if input_path.startswith("http://") or input_path.startswith("https://"):
        if _YOUTUBE_PATTERN.search(input_path):
            return "yt"
        return "web"

    if re.match(r'^[a-zA-Z0-9_-]{11}$', input_path):
        return "yt"

    suffix = Path(input_path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in _IMG_EXTS:
        return "img"
    if suffix in (".html", ".htm"):
        return "html"
    if suffix in _DOC_EXTS:
        return "doc"
    if suffix in (".rst", ".rest"):
        return "rst"
    if suffix in _AUDIO_VIDEO_EXTS:
        return "yt"
    return ""


def app():
    """Main entry point — dispatches to subcommand or auto-detects."""
    args = sys.argv[1:]

    # No args -> show help
    if not args or args == ["--help"] or args == ["-h"]:
        _show_help()
        return

    first = args[0]

    # Explicit subcommand
    if first in _SUBCOMMANDS:
        _TOOL_APPS[first](standalone_mode=True)
        return

    # Auto-detect from first arg
    tool = _detect_tool(first)
    if tool:
        _TOOL_APPS[tool](standalone_mode=True)
        return

    # Unknown input
    typer.echo(
        f"Cannot auto-detect type for: {first}\n"
        "Use an explicit subcommand: 2md yt|pdf|img|web|html|doc|rst\n"
        "Run '2md --help' for usage.",
        err=True,
    )
    raise SystemExit(1)


def _show_help():
    """Print help text."""
    typer.echo("""Usage: 2md [INPUT | COMMAND] [OPTIONS]

Convert media files to markdown. Auto-detects input type by extension or URL.

Auto-detect examples:
  2md video.mp4                  Audio/video → markdown (Parakeet STT)
  2md podcast.mp3 --diarize      With speaker diarization
  2md document.pdf               PDF → markdown
  2md photo.jpg                  Image → markdown (Qwen VLM)
  2md https://example.com        Web page → markdown (ReaderLM)
  2md page.html                  Local HTML → markdown
  2md report.docx                Office doc → markdown
  2md readme.rst                 RST → markdown

Subcommands (for full options, run '2md <cmd> --help'):
  yt    Audio/video transcription (YouTube, local files)
  pdf   PDF extraction
  img   Image OCR via VLM
  web   Web URL conversion
  html  Local HTML conversion
  doc   Office document conversion (DOCX, PPTX, XLSX)
  rst   reStructuredText conversion""")


if __name__ == "__main__":
    app()
