#!/usr/bin/env python3
"""
cli.py — Unified any2md CLI entry point

Auto-detects input type by extension/URL and dispatches to the appropriate
converter. Also exposes each converter as an explicit subcommand.

Usage:
    any2md video.mp4                    # auto-detect -> yt
    any2md document.pdf                 # auto-detect -> pdf
    any2md https://example.com          # auto-detect -> web

    any2md yt video.mp4 --diarize       # explicit subcommand
    any2md pdf document.pdf --pages 1-10
"""

import re
import sys
from pathlib import Path
from typing import Optional

import typer

from any2md import __version__

# Extension -> tool mapping
_AUDIO_VIDEO_EXTS = {".mp3", ".wav", ".mp4", ".webm", ".m4a", ".flac", ".ogg", ".aac", ".mov", ".avi", ".mkv"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
_DOC_EXTS = {".docx", ".pptx", ".xlsx", ".epub", ".odt", ".rtf"}
_DATA_EXTS = {".json", ".yaml", ".yml", ".jsonl"}
_SUB_EXTS = {".srt", ".vtt", ".ass", ".ssa"}
_YOUTUBE_PATTERN = re.compile(
    r'(?:youtube\.com/(?:[^/\n\s]+/\S+/|(?:v|e(?:mbed)?)/'
    r'|\S*?[?&]v=)|youtu\.be/|^[a-zA-Z0-9_-]{11}$)'
)


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
    if suffix in (".csv", ".tsv"):
        return "csv"
    if suffix in _DATA_EXTS:
        return "data"
    if suffix in (".db", ".sqlite", ".sqlite3"):
        return "db"
    if suffix in _SUB_EXTS:
        return "sub"
    if suffix == ".ipynb":
        return "nb"
    if suffix in (".eml", ".mbox"):
        return "eml"
    if suffix == ".org":
        return "org"
    if suffix == ".tex":
        return "tex"
    if suffix in (".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"):
        return "man"
    return ""


def _get_tool_apps() -> dict:
    """Lazy-load tool apps to avoid import errors for uninstalled optional deps."""
    apps = {}

    def _try_import(name, module_path):
        try:
            mod = __import__(module_path, fromlist=["app"])
            apps[name] = mod.app
        except ImportError:
            pass

    _try_import("yt", "any2md.yt")
    _try_import("audio", "any2md.yt")
    _try_import("video", "any2md.yt")
    _try_import("pdf", "any2md.pdf")
    _try_import("img", "any2md.img")
    _try_import("web", "any2md.web")
    _try_import("html", "any2md.html")
    _try_import("doc", "any2md.doc")
    _try_import("rst", "any2md.rst")
    _try_import("csv", "any2md.csv")
    _try_import("data", "any2md.data")
    _try_import("db", "any2md.db")
    _try_import("sub", "any2md.sub")
    _try_import("nb", "any2md.nb")
    _try_import("eml", "any2md.eml")
    _try_import("org", "any2md.org")
    _try_import("tex", "any2md.tex")
    _try_import("man", "any2md.man")

    return apps


_SUBCOMMANDS = {
    "yt", "audio", "video", "pdf", "img", "web", "html", "doc", "rst",
    "csv", "data", "db", "sub", "nb", "eml", "org", "tex", "man",
}


def app():
    """Main entry point — dispatches to subcommand or auto-detects."""
    args = sys.argv[1:]

    if not args or args == ["--help"] or args == ["-h"]:
        _show_help()
        return

    if args == ["--version"] or args == ["-V"]:
        _show_version()
        return

    first = args[0]

    tool_apps = _get_tool_apps()

    # Explicit subcommand
    if first in _SUBCOMMANDS:
        if first not in tool_apps:
            typer.echo(f"Subcommand '{first}' not available. Install its dependencies first.", err=True)
            raise SystemExit(1)
        tool_apps[first](standalone_mode=True)
        return

    # Auto-detect from first arg
    tool = _detect_tool(first)
    if tool:
        if tool not in tool_apps:
            typer.echo(f"Detected '{tool}' converter but it's not installed. Install its dependencies.", err=True)
            raise SystemExit(1)
        tool_apps[tool](standalone_mode=True)
        return

    typer.echo(
        f"Cannot auto-detect type for: {first}\n"
        f"Available subcommands: {', '.join(sorted(tool_apps.keys()))}\n"
        "Run 'any2md --help' for usage.",
        err=True,
    )
    raise SystemExit(1)


def _show_version():
    """Print version info and exit 0."""
    typer.echo(
        f"any2md {__version__}\n"
        "Copyright 2024 roboalchemist\n"
        "License MIT: <https://opensource.org/licenses/MIT>"
    )


def _show_help():
    """Print help text."""
    tool_apps = _get_tool_apps()
    available = ", ".join(sorted(tool_apps.keys()))

    typer.echo(f"""Usage: any2md [INPUT | COMMAND] [OPTIONS]

Convert anything to markdown. Auto-detects input type by extension or URL.

Auto-detect examples:
  any2md video.mp4                  Audio/video → markdown (Parakeet STT)
  any2md podcast.mp3 --diarize      With speaker diarization
  any2md document.pdf               PDF → markdown
  any2md screenshot.png             Image → markdown (Qwen VLM)
  any2md https://example.com        Web page → markdown (ReaderLM)
  any2md page.html                  Local HTML → markdown
  any2md report.docx                Office doc → markdown
  any2md data.csv                   CSV/TSV → markdown table
  any2md config.json                JSON/YAML → markdown
  any2md app.db                     SQLite → markdown (schema + data)
  any2md captions.srt               Subtitles → markdown
  any2md notebook.ipynb             Jupyter → markdown
  any2md message.eml                Email → markdown
  any2md notes.org                  Org-mode → markdown
  any2md paper.tex                  LaTeX → markdown
  any2md command.1                  Man page → markdown
  any2md readme.rst                 RST → markdown

Subcommands (run 'any2md <cmd> --help' for options):
  audio Audio transcription            video Video transcription
  yt    Audio/video + YouTube          pdf   PDF extraction
  img   Image OCR via VLM              web   Web URL conversion
  html  Local HTML conversion          doc   Office docs (DOCX/PPTX/XLSX/EPUB)
  rst   reStructuredText               csv   CSV/TSV tables
  data  JSON/YAML structured data      db    SQLite databases
  sub   Subtitles (SRT/VTT/ASS)        nb    Jupyter notebooks
  eml   Email (.eml/.mbox)             org   Org-mode
  tex   LaTeX                          man   Unix man pages

Available on this system: {available}""")


if __name__ == "__main__":
    app()
