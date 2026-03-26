# any2md — Convert Anything to Markdown

## Overview

A toolkit for converting any media, document, or data format to markdown with YAML frontmatter. AI inference runs locally on Apple Silicon via MLX. No cloud APIs. 20 converters, 8 zero-dependency (stdlib only).

**Version**: 0.2.2 | **Package**: `any2md` | **Entry point**: `any2md.cli:app`

## Project Structure

```
any2md/
├── src/any2md/
│   ├── __init__.py      # Package init, __version__ = "0.2.2"
│   ├── cli.py           # Unified entry point — auto-detect + subcommands (298 lines)
│   ├── common.py        # Shared: build_frontmatter(), setup_logging(), JSON mode (287 lines)
│   ├── yt.py            # Audio/video/YouTube transcription + Sortformer diarization (859 lines)
│   ├── pdf.py           # PDF extraction + optional VLM OCR via mlx-vlm (607 lines)
│   ├── img.py           # Image OCR via Qwen3.5 (mlx-vlm) (474 lines)
│   ├── web.py           # Web URL → markdown via ReaderLM-v2 (mlx-lm) (465 lines)
│   ├── html.py          # Local HTML → markdown via ReaderLM-v2 (341 lines)
│   ├── doc.py           # Office docs via markitdown (DOCX/PPTX/XLSX/EPUB/ODT/RTF) (356 lines)
│   ├── rst.py           # reStructuredText via pypandoc/docutils fallback (439 lines)
│   ├── csv.py           # CSV/TSV → markdown tables, stdlib only (519 lines)
│   ├── data.py          # JSON/YAML/JSONL → smart markdown, stdlib + optional PyYAML (600 lines)
│   ├── db.py            # SQLite → schema + sample data, stdlib only (572 lines)
│   ├── sub.py           # Subtitles (SRT/VTT/ASS) via pysubs2 (442 lines)
│   ├── nb.py            # Jupyter notebooks, stdlib only (441 lines)
│   ├── eml.py           # Email (.eml/.mbox), stdlib only (538 lines)
│   ├── org.py           # Org-mode, pure regex (704 lines)
│   ├── tex.py           # LaTeX, pure regex (801 lines)
│   ├── man.py           # Unix man pages, mandoc + regex fallback (766 lines)
│   └── repo.py          # Git repositories via repomix (npm wrapper) (172 lines)
├── tests/
│   ├── conftest.py      # Pytest configuration
│   ├── create_fixtures.py
│   ├── fixtures/        # sample.{docx,html,ipynb,jpg,pdf,pptx,rst,xlsx}
│   ├── test_yt.py, test_pdf.py, test_web.py, test_html.py
│   ├── test_doc.py, test_img.py, test_rst.py, test_csv.py
│   ├── test_data.py, test_db.py, test_sub.py, test_nb.py
│   ├── test_eml.py, test_org.py, test_tex.py, test_man.py
│   ├── test_repo.py, test_cli.py, test_common.py
│   ├── test_inference.py    # 14 @pytest.mark.slow (real model inference)
│   └── test_integration.py  # 9 @pytest.mark.integration (real file fixtures)
├── scripts/
│   ├── download_models.py   # Pre-download MLX models (--stt, --vlm, --reader, --all)
│   └── benchmark_models.py  # Benchmark Parakeet STT models, compute RTF
├── pyproject.toml       # hatchling build, optional deps groups, entry points
├── pytest.ini           # pytest markers config
├── requirements.txt
└── README.md
```

**Total**: ~9,700 lines of source code across 20 modules. 808 tests (8,810 lines).

## Key Libraries & Dependencies

| Library | Purpose | Optional Group | llm-code-docs |
|---------|---------|----------------|---------------|
| `typer>=0.9.0` | CLI framework (required for all) | core | ✅ `docs/github-scraped/typer/` |
| `mlx-audio[stt]>=0.4.0` | Parakeet STT + Sortformer diarization | `[stt]` | ❌ |
| `yt-dlp>=2023.11.14` | YouTube audio download | `[stt]` | ❌ |
| `pymupdf4llm>=0.2.0` | PDF text extraction | `[pdf]` | ~`docs/readthedocs/pymupdf/` (base lib) |
| `mlx-vlm` | Image/PDF OCR via Qwen3.5 | `[img]` | ❌ |
| `mlx-lm` | HTML→markdown via ReaderLM-v2 | `[web]` | ✅ `docs/github-scraped/mlx-lm/` |
| `httpx` | HTTP fetching for web URLs | `[web]` | ✅ `docs/github-scraped/httpx/` |
| `markitdown>=0.1.0` | Office document conversion | `[doc]` | ❌ |
| `mammoth>=1.12.0` | DOCX enhancement | `[doc]` | ❌ |
| `pypandoc` | RST conversion (docutils fallback) | `[rst]` | ~`docs/github-scraped/pandoc/` (base tool) |
| `pysubs2` | Subtitle parsing | runtime | ❌ |
| `Pillow` | Image handling for VLM | runtime | ❌ |
| `ffmpeg`/`ffprobe` | Audio/video conversion | system dep | — |
| `repomix` (npm) | Repository to markdown | system dep (`repo`) | — |
| `mandoc` | Man page rendering | optional system dep | — |

**Zero-dep converters** (stdlib only): `csv`, `data`, `db`, `nb`, `eml`, `org`, `tex`, `man`

## Architecture

### Entry Point: `cli.py`

`any2md.cli:app` — registered as `any2md` console script via pyproject.toml.

**Key functions**:
- `_detect_tool(input_path: str) -> str` — Maps file extensions/URLs to converter names
  - YouTube URLs and 11-char IDs → `yt`
  - HTTP(S) URLs → `web`
  - Paths with `.git` dir → `repo`
  - Extension maps: `_AUDIO_VIDEO_EXTS`, `_IMG_EXTS`, `_DOC_EXTS`, `_DATA_EXTS`, `_SUB_EXTS`
  - Returns empty string if undetectable

- `_get_tool_apps() -> dict` — Lazy-loads all converter apps via `__import__()` wrapped in try/except to tolerate missing optional deps

- `app()` — Main entry point: parses sys.argv, handles early flags (`--help`, `--version`, `--quiet`, `--json`), routes to converter

**Subcommands**: `yt`, `audio`, `video`, `pdf`, `img`, `web`, `html`, `doc`, `rst`, `csv`, `data`, `db`, `sub`, `nb`, `eml`, `org`, `tex`, `man`, `repo`, `deps`. (`audio` and `video` are aliases for `yt`)

### Converter Pattern

Every converter module follows the same pattern:
1. Imports `build_frontmatter`, `OutputFormat`, `write_output`, `setup_logging` from `any2md.common`
2. Defines a `typer.Typer()` app with consistent CLI flags: `--output-dir/-o`, `--format/-f`, `--verbose/-v`, `--json/-j`, `--quiet/-q`
3. Has a `process_*_file()` main function (or `process_*_dir()` for batch)
4. Extracts metadata dict → builds frontmatter → converts content → writes output
5. External deps wrapped in try/except with graceful fallbacks

### Shared Module: `common.py`

- `build_frontmatter(metadata: dict) -> str` — Hand-rolled YAML formatter (no PyYAML dep). Handles scalars, lists, nested dicts (chapters), multi-line strings (description via `|` block scalar), auto-quoting special chars (`:#{}[]&*?|->!%@`).
- `OutputFormat(str, Enum)` — `md`, `txt`
- `setup_logging(verbose: bool)` — Configures root logger to stderr; respects `ANY2MD_QUIET` env var
- `write_output(content: str, output_path: Path)` — File writer with parent dir creation
- `write_json_output(metadata, content, source, converter, fields=None)` — JSON to stdout for `--json` mode
- `_filter_fields(data: dict, fields_str: str) -> dict` — Dot-notation field selection (e.g., `"frontmatter.rows,content"`)
- `write_json_error(code, message, recoverable)` — Structured JSON errors to stderr
- `set_json_mode(enabled)` / `is_json_mode()` — Global JSON mode flag
- `load_vlm(model)` — Stub for mlx-vlm loading

**Logging**: All output to stderr via explicit `StreamHandler(sys.stderr)`; stdout reserved for machine-readable data.

### Data Flow

```
cli.py: argv → _detect_tool() or explicit subcommand → lazy import converter → converter.app()
Each converter: input → extract metadata → convert content → build_frontmatter() → write_output()
yt.py:   input → [yt-dlp | local file] → ffmpeg 16kHz WAV → mlx-audio transcribe → format
pdf.py:  PDF → pymupdf4llm extract (fast) or Qwen3.5 VLM OCR (scanned pages) → format
web.py:  URL → httpx fetch → ReaderLM-v2 (mlx-lm) → markdown
img.py:  image → Qwen3.5 (mlx-vlm) → markdown
repo.py: git dir → repomix CLI → markdown with file stats frontmatter
```

### Converter-Specific Details

| Module | Key Function | External Dep | Notes |
|--------|-------------|-------------|-------|
| `yt.py` | `transcribe_audio_via_mlx()` | mlx-audio, yt-dlp, ffmpeg | Supports diarization, SRT output |
| `pdf.py` | `process_pdf_file()` | pymupdf4llm, mlx-vlm | Auto-fallback OCR when < 50 chars/page |
| `img.py` | `process_image_file()` | mlx-vlm (Qwen3.5) | Supports JPEG, PNG, GIF, BMP, WebP, TIFF |
| `web.py` | `html_to_markdown()` | mlx-lm (ReaderLM-v2), httpx | 200K char limit, 8192 max output tokens |
| `html.py` | `process_html_file()` | mlx-lm (ReaderLM-v2) | Local HTML files; batch dir support |
| `doc.py` | `process_doc_file()` | markitdown, mammoth | DOCX/PPTX/XLSX/EPUB/ODT/RTF |
| `rst.py` | `process_rst_file()` | pypandoc (docutils fallback) | Pure regex fallback if neither available |
| `csv.py` | `table_to_markdown()` | stdlib only | Auto-detects delimiter; 500 row default |
| `data.py` | `load_data()` | stdlib + optional PyYAML | Auto-selects table vs bullet vs code block |
| `db.py` | `process_db_file()` | stdlib only | Schema + sample rows per table |
| `sub.py` | `process_sub_file()` | pysubs2 | SRT, VTT, ASS, SSA |
| `nb.py` | `process_nb_file()` | stdlib only | Code + markdown + outputs cells |
| `eml.py` | `process_eml_file()` | stdlib only | Headers + MIME body extraction |
| `org.py` | `process_org_file()` | stdlib only | Pure regex conversion |
| `tex.py` | `process_tex_file()` | stdlib only | Pure regex conversion |
| `man.py` | `process_man_file()` | mandoc (optional) | Regex fallback if mandoc unavailable |
| `repo.py` | `process_repo_dir()` | repomix (npm) | Thin wrapper (172 lines), requires `repomix` installed globally |

### Model System

| Model | Library | Default HF ID | Use |
|-------|---------|--------------|-----|
| Parakeet v3 | mlx-audio | `mlx-community/parakeet-tdt-0.6b-v3` | Audio/video STT |
| Sortformer | mlx-audio | (auto-configured) | Speaker diarization |
| ReaderLM-v2 | mlx-lm | `mlx-community/jinaai-ReaderLM-v2` | HTML/web → markdown |
| Qwen3.5-9B | mlx-vlm | `mlx-community/Qwen3.5-9B-MLX-4bit` | Image OCR, PDF OCR |

**Model aliases in `yt.py`**: `parakeet-v3`, `parakeet-v2`, `parakeet-1.1b`, `parakeet-ctc`
**Model aliases in `img.py` / `pdf.py`**: `qwen3.5-4b`, `qwen3.5-9b` (default), `qwen3.5-27b`, `qwen3.5-35b`, `smoldocling`

## Installation

```bash
# Editable install with all optional deps
uv pip install -e '.[all]'

# Or only what you need
uv pip install -e '.[stt]'    # Audio/video transcription
uv pip install -e '.[pdf]'    # PDF extraction
uv pip install -e '.[img]'    # Image OCR
uv pip install -e '.[web]'    # Web pages
uv pip install -e '.[doc]'    # Office docs

# System dependencies
brew install ffmpeg            # Required for audio/video
npm install -g repomix         # Required for repo subcommand

# Pre-download AI models (Apple Silicon only)
python scripts/download_models.py --stt      # Parakeet + Sortformer
python scripts/download_models.py --vlm      # Qwen3.5 for img/pdf --ocr
python scripts/download_models.py --reader   # ReaderLM-v2 for web/html
python scripts/download_models.py --all      # Everything
```

## Build System

- **Build backend**: `hatchling`
- **Build config**: `pyproject.toml` (`[tool.hatch.build.targets.wheel]`)
- **Package source**: `src/any2md/` layout
- **Requires Python**: `>=3.11`

## Testing

- **Framework**: `unittest.TestCase` classes, collected and run via `pytest`
- **Test count**: 808 tests across 19 test files
- **Run all unit tests**: `python -m pytest tests/`
- **Run specific file**: `python -m pytest tests/test_csv.py -v`
- **Slow tests** (real inference, >10s): `python -m pytest -m slow -v -s`
- **Integration tests** (real fixtures): `python -m pytest -m integration -v`
- **Markers**: `@pytest.mark.slow` (14 tests), `@pytest.mark.integration` (9 tests)
- **Mocking**: `unittest.mock.patch()` for all external deps; `FakeAlignedSentence` objects for mlx-audio
- **Auto-skip**: Tests skip if required models are not cached locally
- **Fixtures**: `tests/fixtures/` (sample files), `tests/audio/` (audio files)
- **Config**: `pyproject.toml` `[tool.pytest.ini_options]` + `pytest.ini` (both define markers)

## CLI Reference

```bash
# Auto-detect (just pass a file, URL, or git directory)
any2md <input>

# Explicit subcommands
any2md yt <input>  [--model NAME] [--diarize] [--format md|srt|txt] [--chunk-duration FLOAT] [--keep-audio]
any2md pdf <input> [--pages "1-10,15"] [--ocr] [--model NAME]
any2md img <input> [--model NAME]
any2md web <url>
any2md html <input>
any2md doc <input>
any2md csv <input> [--max-rows N] [--max-col-width N]
any2md db  <input> [--max-rows N] [--max-tables N] [--skip-views]
any2md sub <input>
any2md nb  <input> [--no-outputs]
any2md repo <dir>  [--style markdown|json] [--compress] [--remove-comments]

# Global flags (all subcommands)
  --json, -j                # JSON output to stdout (agent-friendly)
  --fields FIELDS           # Dot-notation field selection for --json (e.g., "frontmatter.rows,content")
  --quiet, -q               # Suppress INFO logs (or set ANY2MD_QUIET=1)
  --version, -V             # Print version and exit
  -o, --output-dir PATH     # Output directory (default: cwd)
  -f, --format [md|txt]     # Output format (default: md)
  -v, --verbose             # DEBUG logging

# Utility subcommands
any2md deps                 # Show installed/missing optional dependencies
```

**JSON output format** (for `--json` mode):
```json
{
  "frontmatter": { "title": "...", "rows": 42, ... },
  "content": "## markdown content...",
  "source": "/path/to/file.csv",
  "converter": "csv"
}
```

## Project History

Originally `yt2srt` on `lightning-whisper-mlx`. Migrated to `yt2md` on `mlx-audio` with Parakeet, rewritten from argparse to typer. Then expanded from 2 tools (yt2md + pdf2md) to 20 converters as a proper Python package (`src/any2md/`) with unified CLI, optional dependency groups, and 808 tests. v0.2.0 added quality & polish pass; v0.2.2 added `repo` subcommand via repomix.
