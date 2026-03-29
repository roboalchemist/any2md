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

import importlib.resources
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer

from any2md import __version__

# Respect NO_COLOR env var (https://no-color.org/)
if os.environ.get("NO_COLOR"):
    os.environ["TERM"] = "dumb"  # Disables rich colors

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
    # Detect git repository directories
    p = Path(input_path)
    if p.is_dir() and (p / ".git").exists():
        return "repo"
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
    _try_import("repo", "any2md.repo")

    # Speaker catalog management — uses speaker_app (not app) from speaker module
    try:
        from any2md import speaker as speaker_mod
        apps["speaker"] = speaker_mod.speaker_app
    except ImportError:
        pass

    return apps


_SUBCOMMANDS = {
    "yt", "audio", "video", "pdf", "img", "web", "html", "doc", "rst",
    "csv", "data", "db", "sub", "nb", "eml", "org", "tex", "man", "repo",
    "speaker", "deps", "docs", "completion",
}

# Stable-ordered list of all subcommands for completion scripts
_COMPLETION_SUBCOMMANDS = [
    "yt", "audio", "video", "pdf", "img", "web", "html", "doc", "rst",
    "csv", "data", "db", "sub", "nb", "eml", "org", "tex", "man", "repo",
    "speaker", "deps", "completion",
]

_GLOBAL_FLAGS = [
    "--json", "-j", "--fields", "--quiet", "-q", "--version", "-V",
    "--output-dir", "-o", "--format", "-f", "--verbose", "-v", "--help", "-h",
]


def _generate_completion(shell: str) -> str:
    """Generate a shell completion script for bash, zsh, or fish."""
    cmds = _COMPLETION_SUBCOMMANDS
    cmds_str = " ".join(cmds)

    if shell == "bash":
        flags_str = " ".join(_GLOBAL_FLAGS)
        return (
            "# any2md bash completion\n"
            "# Add to ~/.bashrc or source directly:\n"
            "#   eval \"$(any2md completion bash)\"\n"
            "\n"
            "_any2md_completions() {\n"
            "    local cur prev\n"
            f'    cur="${{COMP_WORDS[COMP_CWORD]}}"\n'
            f'    prev="${{COMP_WORDS[COMP_CWORD-1]}}"\n'
            "\n"
            f'    local subcommands="{cmds_str}"\n'
            f'    local flags="{flags_str}"\n'
            "\n"
            "    if [[ $COMP_CWORD -eq 1 ]]; then\n"
            '        COMPREPLY=( $(compgen -W "$subcommands $flags" -- "$cur") )\n'
            "        return 0\n"
            "    fi\n"
            "\n"
            '    if [[ "$prev" == "--output-dir" || "$prev" == "-o" ]]; then\n'
            '        COMPREPLY=( $(compgen -d -- "$cur") )\n'
            "        return 0\n"
            "    fi\n"
            "\n"
            '    if [[ "$prev" == "--format" || "$prev" == "-f" ]]; then\n'
            '        COMPREPLY=( $(compgen -W "md txt srt" -- "$cur") )\n'
            "        return 0\n"
            "    fi\n"
            "\n"
            '    COMPREPLY=( $(compgen -W "$flags" -- "$cur") )\n'
            "    return 0\n"
            "}\n"
            "\n"
            "complete -F _any2md_completions any2md\n"
        )

    elif shell == "zsh":
        cmds_with_desc = "\n".join(f"        \'{c}\'" for c in cmds)
        return (
            "#compdef any2md\n"
            "# any2md zsh completion\n"
            "# Source directly or add to fpath:\n"
            "#   eval \"$(any2md completion zsh)\"\n"
            "\n"
            "_any2md() {\n"
            "    local -a subcommands\n"
            "    subcommands=(\n"
            f"{cmds_with_desc}\n"
            "    )\n"
            "\n"
            "    local -a flags\n"
            "    flags=(\n"
            "        \'--json[Output as JSON to stdout]\'\n"
            "        \'-j[Output as JSON to stdout]\'\n"
            "        \'--fields[Dot-notation field selection for --json]:fields:\'\n"
            "        \'--quiet[Suppress INFO logs]\'\n"
            "        \'-q[Suppress INFO logs]\'\n"
            "        \'--version[Print version and exit]\'\n"
            "        \'-V[Print version and exit]\'\n"
            "        \'--output-dir[Output directory]:directory:_directories\'\n"
            "        \'-o[Output directory]:directory:_directories\'\n"
            "        \'--format[Output format]:format:(md txt srt)\'\n"
            "        \'-f[Output format]:format:(md txt srt)\'\n"
            "        \'--verbose[DEBUG logging]\'\n"
            "        \'-v[DEBUG logging]\'\n"
            "        \'--help[Show help message]\'\n"
            "        \'-h[Show help message]\'\n"
            "    )\n"
            "\n"
            "    if (( CURRENT == 2 )); then\n"
            "        _describe \'subcommand\' subcommands\n"
            "        _arguments $flags\n"
            "        return 0\n"
            "    fi\n"
            "\n"
            "    _arguments $flags \'*:file:_files\'\n"
            "}\n"
            "\n"
            "_any2md \"$@\"\n"
        )

    elif shell == "fish":
        cmds_completions = "\n".join(
            f"complete -c any2md -f -n \'__fish_use_subcommand\' -a \'{c}\'"
            for c in cmds
        )
        return (
            "# any2md fish completion\n"
            "# Install: any2md completion fish > ~/.config/fish/completions/any2md.fish\n"
            "\n"
            "function __fish_use_subcommand\n"
            "    set cmd (commandline -opc)\n"
            "    set -e cmd[1]\n"
            "    for i in $cmd\n"
            f"        if contains -- $i {cmds_str}\n"
            "            return 1\n"
            "        end\n"
            "    end\n"
            "    return 0\n"
            "end\n"
            "\n"
            f"{cmds_completions}\n"
            "\n"
            "# Global flags\n"
            "complete -c any2md -l json -s j -d \'Output as JSON to stdout\'\n"
            "complete -c any2md -l quiet -s q -d \'Suppress INFO logs\'\n"
            "complete -c any2md -l version -s V -d \'Print version and exit\'\n"
            "complete -c any2md -l verbose -s v -d \'DEBUG logging\'\n"
            "complete -c any2md -l help -s h -d \'Show help message\'\n"
            "complete -c any2md -l output-dir -s o -r -d \'Output directory\'\n"
            "complete -c any2md -l format -s f -r -a \'md txt srt\' -d \'Output format\'\n"
            "complete -c any2md -l fields -r -d \'Dot-notation field selection for --json\'\n"
        )

    else:
        raise ValueError(f"Unknown shell: {shell!r}. Choose bash, zsh, or fish.")


def _try_import_dep(module: str) -> bool:
    """Return True if the given module can be imported."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _show_deps():
    """Print dependency status for all optional packages."""
    deps = {
        "typer": True,  # always present
        "pymupdf4llm": _try_import_dep("pymupdf4llm"),
        "mlx-vlm": _try_import_dep("mlx_vlm"),
        "mlx-lm": _try_import_dep("mlx_lm"),
        "mlx-audio": _try_import_dep("mlx_audio"),
        "markitdown": _try_import_dep("markitdown"),
        "httpx": _try_import_dep("httpx"),
        "pysubs2": _try_import_dep("pysubs2"),
        "pypandoc": _try_import_dep("pypandoc"),
    }
    typer.echo("any2md dependency status:", err=True)
    for name, installed in deps.items():
        status = "installed" if installed else "MISSING"
        typer.echo(f"  {name}: {status}", err=True)

    missing = [name for name, installed in deps.items() if not installed]
    if missing:
        typer.echo("", err=True)
        typer.echo("Install missing dependencies:", err=True)
        typer.echo("  uv pip install pymupdf4llm          # pdf", err=True)
        typer.echo("  uv pip install mlx-vlm              # img, pdf --ocr", err=True)
        typer.echo("  uv pip install mlx-lm httpx         # web, html", err=True)
        typer.echo('  uv pip install "mlx-audio[stt]" yt-dlp  # yt/audio/video', err=True)
        typer.echo("  uv pip install markitdown           # doc", err=True)


_EMBEDDED_DOCS = """\
# any2md — Convert Anything to Markdown

A toolkit for converting any media, document, or data format to markdown.
AI inference runs locally on Apple Silicon via MLX — no cloud APIs.

## Subcommands

  yt / audio / video   Audio/video transcription + YouTube download (Parakeet STT)
  pdf                  PDF text extraction (+ optional VLM OCR)
  img                  Image OCR via Qwen VLM
  web                  Web URL → markdown via ReaderLM
  html                 Local HTML → markdown via ReaderLM
  doc                  Office docs (DOCX/PPTX/XLSX/EPUB/ODT/RTF)
  rst                  reStructuredText conversion
  csv                  CSV/TSV → markdown tables
  data                 JSON/YAML/JSONL → markdown
  db                   SQLite → schema + sample data
  sub                  Subtitles (SRT/VTT/ASS/SSA)
  nb                   Jupyter notebooks
  eml                  Email (.eml/.mbox)
  org                  Org-mode → markdown
  tex                  LaTeX → markdown
  man                  Unix man pages → markdown
  repo                 Git repository → markdown via repomix
  deps                 Show optional dependency status
  docs                 Print this documentation

## Quick Start

  any2md video.mp4                  # auto-detect → transcribe
  any2md document.pdf               # auto-detect → extract text
  any2md https://example.com        # auto-detect → web page
  any2md --help                     # full usage

## Installation

  uv pip install 'any2md[all]'       # all optional deps
  brew install ffmpeg                # required for audio/video
  npm install -g repomix             # required for repo subcommand

See https://github.com/roboalchemist/any2md for full documentation.
"""


def _show_docs() -> None:
    """Print README.md to stdout, with fallbacks for packaged/brew installs."""
    # 1. Try src layout: __file__ is src/any2md/cli.py, README is 3 levels up
    candidate = Path(__file__).parent.parent.parent / "README.md"
    if candidate.is_file():
        print(candidate.read_text(encoding="utf-8"), end="")
        raise SystemExit(0)

    # 2. Try importlib.resources (works for installed packages that bundle README)
    try:
        ref = importlib.resources.files("any2md").joinpath("README.md")
        text = ref.read_text(encoding="utf-8")
        print(text, end="")
        raise SystemExit(0)
    except (FileNotFoundError, TypeError, AttributeError):
        pass

    # 3. Fall back to embedded summary (e.g. brew install without source tree)
    print(_EMBEDDED_DOCS, end="")
    raise SystemExit(0)


def app():
    """Main entry point — dispatches to subcommand or auto-detects."""
    args = sys.argv[1:]

    if args == ["--help"] or args == ["-h"] or not args:
        _show_help()
        if not args:
            raise SystemExit(2)
        return

    # Standard subcommand help: any2md <cmd> --help
    if len(args) >= 2 and args[1] in ("--help", "-h") and args[0] in _SUBCOMMANDS:
        tool_apps = _get_tool_apps()
        if args[0] == "deps":
            _show_deps()
            return
        if args[0] == "docs":
            _show_docs()
            return  # pragma: no cover
        if args[0] in tool_apps:
            tool_apps[args[0]](["--help"], standalone_mode=True)
            return

    if args == ["--version"] or args == ["-V"]:
        _show_version()
        return

    # Handle --quiet / -q / --silent early: suppress INFO logs and strip flag from args
    if "--quiet" in args or "-q" in args or "--silent" in args:
        os.environ["ANY2MD_QUIET"] = "1"
        # Silence root logger immediately (catches import-time messages like NumExpr)
        logging.root.setLevel(logging.WARNING)
        # Also silence common noisy loggers that fire at import time
        logging.getLogger("numexpr").setLevel(logging.WARNING)
        args = [a for a in args if a not in ("--quiet", "-q", "--silent")]

    # Handle --json early: enable JSON mode and strip flag from args
    if "--json" in args:
        from any2md.common import set_json_mode
        set_json_mode(True)
        args = [a for a in args if a != "--json"]
        # Suppress import-time log noise so stderr stays clean for JSON
        logging.root.setLevel(logging.WARNING)
        logging.getLogger("numexpr").setLevel(logging.WARNING)

    first = args[0]

    # deps subcommand — no tool_apps needed
    if first == "deps":
        _show_deps()
        return

    # docs subcommand — no tool_apps needed
    if first == "docs":
        _show_docs()
        # _show_docs raises SystemExit, but return for clarity
        return  # pragma: no cover

    # completion subcommand — no tool_apps needed
    if first == "completion":
        rest = args[1:]
        if not rest or rest[0] in ("--help", "-h"):
            typer.echo("Usage: any2md completion [bash|zsh|fish]", err=True)
            typer.echo("Output a shell completion script to stdout.", err=True)
            return
        shell = rest[0]
        try:
            script = _generate_completion(shell)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise SystemExit(1)
        sys.stdout.write(script)
        return

    tool_apps = _get_tool_apps()

    # Explicit subcommand
    if first in _SUBCOMMANDS:
        if first not in tool_apps:
            typer.echo(f"Subcommand '{first}' not available. Install its dependencies first.", err=True)
            raise SystemExit(1)
        # Pass args after the subcommand name (skip 'yt', 'csv', etc.)
        tool_apps[first](args[1:], standalone_mode=True)
        return

    # Auto-detect from first arg
    tool = _detect_tool(first)
    if tool:
        if tool not in tool_apps:
            typer.echo(f"Detected '{tool}' converter but it's not installed. Install its dependencies.", err=True)
            raise SystemExit(1)
        # Pass all args — auto-detect, so first arg is the input file
        tool_apps[tool](args, standalone_mode=True)
        return

    typer.echo(
        f"Cannot auto-detect type for: {first}\n"
        f"Available subcommands: {', '.join(sorted(tool_apps.keys()))}\n"
        "Run 'any2md --help' for usage.",
        err=True,
    )
    raise SystemExit(2)


def _show_version():
    """Print version info and exit 0."""
    typer.echo(
        f"any2md {__version__}\n"
        "Copyright 2024 roboalchemist\n"
        "License MIT: <https://opensource.org/licenses/MIT>"
    )


def _show_help():
    """Print help text in a format compatible with standard CLI conventions."""
    tool_apps = _get_tool_apps()
    available = ", ".join(sorted(tool_apps.keys()))

    typer.echo(f"""Usage: any2md [INPUT | COMMAND] [OPTIONS]

Convert anything to markdown. Auto-detects input type by extension or URL.

Examples:
  any2md video.mp4                  Audio/video → markdown (Parakeet STT)
  any2md podcast.mp3 --diarize      With speaker diarization
  any2md document.pdf               PDF → markdown
  any2md screenshot.png             Image → markdown (Qwen VLM)
  any2md https://example.com        Web page → markdown (ReaderLM)
  any2md data.csv                   CSV/TSV → markdown table
  any2md config.json                JSON/YAML → markdown
  any2md app.db                     SQLite → markdown (schema + data)

Global Options:
  --json, -j            Output as JSON to stdout
  --fields FIELDS       Dot-notation field selection for --json
  --quiet, -q, --silent Suppress INFO logs
  --version, -V         Print version and exit
  -o, --output-dir DIR  Output directory (default: cwd)
  -f, --format FMT      Output format: md, txt (yt also: srt)
  -v, --verbose         DEBUG logging
  --help, -h            Show this help message

Commands:
  yt          Audio/video transcription + YouTube download
  audio       Alias for yt
  video       Alias for yt
  pdf         PDF text extraction (+ optional VLM OCR)
  img         Image OCR via Qwen VLM
  web         Web URL → markdown via ReaderLM
  html        Local HTML → markdown via ReaderLM
  doc         Office docs (DOCX/PPTX/XLSX/EPUB/ODT/RTF)
  rst         reStructuredText conversion
  csv         CSV/TSV → markdown tables
  data        JSON/YAML/JSONL → markdown
  db          SQLite → schema + sample data
  sub         Subtitles (SRT/VTT/ASS/SSA)
  nb          Jupyter notebooks
  eml         Email (.eml/.mbox)
  org         Org-mode → markdown
  tex         LaTeX → markdown
  man         Unix man pages → markdown
  repo        Git repository → markdown via repomix
  speaker     Manage speaker enrollment catalog
  deps        Show optional dependency status
  docs        Print README documentation to stdout

Available on this system: {available}""")
    typer.echo("\nRun 'any2md <command> --help' for command-specific options.")
    typer.echo("Report bugs: https://github.com/roboalchemist/any2md/issues")


if __name__ == "__main__":
    app()
