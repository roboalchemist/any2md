# any2md — Convert Anything to Markdown

## Overview

A toolkit for converting any media, document, or data format to markdown with YAML frontmatter. AI inference runs locally on Apple Silicon via MLX. No cloud APIs. 20 converters, 8 zero-dependency (stdlib only).

**Version**: 0.4.0 | **Package**: `any2md` | **Entry point**: `any2md.cli:app`

## Project Structure

```
any2md/
├── src/any2md/
│   ├── __init__.py      # Package init, __version__ = "0.3.0"
│   ├── cli.py           # Unified entry point — auto-detect + subcommands, lazy converter loading (585 lines)
│   ├── common.py        # Shared: build_frontmatter(), setup_logging(), JSON mode, MLX memory mgmt (360 lines)
│   ├── yt.py            # Audio/video/YouTube transcription + diarization (1700 lines)
│   ├── speaker.py       # WeSpeaker speaker catalog: enrollment, gallery, KNN matching (2082 lines)
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
├── llms.txt             # LLM-friendly quick reference (links to README + CLAUDE.md)
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
│   ├── download_models.py   # Pre-download MLX models (--stt, --vlm, --reader, --diarize, --all)
│   └── memory_profile.py    # Footprint + MLX memory profiler for any command
│   └── benchmark_models.py  # Benchmark Parakeet STT models, compute RTF
├── pyproject.toml       # hatchling build, optional deps groups, entry points
├── pytest.ini           # pytest markers config
├── requirements.txt
└── README.md
```

**Total**: ~10,000 lines of source code across 20 modules plus memory-profiling and model-download scripts.

## Key Libraries & Dependencies

| Library | Purpose | Optional Group | llm-code-docs |
|---------|---------|----------------|---------------|
| `typer>=0.9.0` | CLI framework (required for all) | core | ✅ `docs/github-scraped/typer/` |
| `mlx-audio[stt]>=0.5.9` | Parakeet STT + Nemotron/Sortformer diarization | `[stt]` | ❌ |
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

**Subcommands**: `yt`, `audio`, `video`, `pdf`, `img`, `web`, `html`, `doc`, `rst`, `csv`, `data`, `db`, `sub`, `nb`, `eml`, `org`, `tex`, `man`, `repo`, `speaker`, `deps`, `docs`, `completion`. (`audio` and `video` are aliases for `yt`)

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
| `yt.py` | `transcribe_audio_via_mlx()` | mlx-audio, yt-dlp, ffmpeg | Supports diarization, SRT output, speaker identification; multilingual via Qwen3-ASR-1.7B |
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
| `speaker.py` | `match_speaker()` | wespeaker, torch, sqlite-vec | WeSpeaker ResNet34 (ONNX) embeddings, KNN gallery matching |

### Model System

| Model | Library | Default HF ID | Use |
|-------|---------|--------------|-----|
| Parakeet v3 | mlx-audio | `mlx-community/parakeet-tdt-0.6b-v3` | Audio/video STT (English) |
| Qwen3-ASR-1.7B | mlx-audio | `mlx-community/Qwen3-ASR-1.7B-bf16` | Multilingual STT (30 langs) |
| Qwen3-ForcedAligner | mlx-audio | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | Word-level alignment for diarization (one ASR segment at a time) |
| Whisper-tiny (LID) | mlx-audio | `mlx-community/whisper-tiny-asr-fp16` | Language detection on the first 30 s window |
| Nemotron-3-Diarization | mlx-audio (vad) | `mlx-community/Nemotron-3-Diarization` | Speaker diarization (8 speakers, bounded-memory streaming windows) |
| Legacy Sortformer | mlx-audio (vad) | `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` | Diarization via `--diarize-model` (driven through chunked `generate_stream`) |
| ReaderLM-v2 | mlx-lm | `mlx-community/jinaai-ReaderLM-v2` | HTML/web → markdown |
| Qwen3.5-9B | mlx-vlm | `mlx-community/Qwen3.5-9B-MLX-4bit` | Image OCR, PDF OCR |

**Model aliases in `yt.py`** — English (fast): `parakeet-v3`, `parakeet-v2`, `parakeet-1.1b`, `parakeet-ctc`; Multilingual (Qwen3-ASR): `qwen3-asr` (alias of `qwen3-asr-bf16`), `qwen3-asr-bf16`, `qwen3-asr-8bit`, `qwen3-asr-4bit`
**Model aliases in `img.py` / `pdf.py`**: `qwen3.5-4b`, `qwen3.5-9b` (default), `qwen3.5-27b`, `qwen3.5-35b`, `qwen3.6-35b` (→ Qwen3.6-35B-A3B-4bit), `qwen3.8-27b` (→ Qwen3.8-27B-4bit), `smoldocling` (→ `docling-project/SmolDocling-256M-preview-mlx-bf16`)

## Memory Management

any2md keeps memory bounded by design (v0.4.0); large runs used to look like
leaks because of three compounding issues, all fixed:

1. **MLX buffer cache** — MLX reuses freed Metal buffers from a cache that
   could grow to the whole memory limit and never shrinks.
   `common.configure_mlx_memory()` caps it at **2 GB** at CLI startup
   (override with `ANY2MD_MLX_CACHE_LIMIT_GB`, `0` disables caching), and
   `common.release_mlx_memory()` (gc + `mx.clear_cache()`) runs between
   pipeline stages.
2. **Models stay resident** — the STT model, diarization model, forced
   aligner, and speaker model are dropped as soon as each stage finishes
   instead of all living at once.
3. **Unbounded inputs** — diarization never runs full-clip attention
   (Nemotron streams internally; Sortformer is driven through chunked
   `generate_stream()` fed by `_iter_wav_chunks`), the forced aligner aligns
   one ASR segment at a time, and LID runs on just the first 30 s.

`scripts/memory_profile.py` measures peak/lifetime physical footprint plus
MLX active/cache memory for any command (add `--repeat N` to check for
cross-run growth):
`python scripts/memory_profile.py -- yt meeting.m4a --diarize`

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
python scripts/download_models.py --stt      # Parakeet + Whisper-tiny LID
python scripts/download_models.py --vlm      # Qwen3.5 for img/pdf --ocr
python scripts/download_models.py --reader   # ReaderLM-v2 for web/html
python scripts/download_models.py --diarize  # Nemotron-3 + streaming Sortformer
python scripts/download_models.py --all      # Everything
```

## Build System

- **Build backend**: `hatchling`
- **Build config**: `pyproject.toml` (`[tool.hatch.build.targets.wheel]`)
- **Package source**: `src/any2md/` layout
- **Requires Python**: `>=3.11`

## Testing

- **Framework**: `unittest.TestCase` classes, collected and run via `pytest`
- **Test count**: 1105 unit tests + 16 slow-inference tests + 11 integration tests across 21 test files
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
any2md yt <input>  [--model NAME] [--language LANG] [--diarize] [--format md|srt|txt] [--chunk-duration FLOAT] [--keep-audio]
                   [--aligner-model NAME] [--identify] [--auto-enroll] [--no-enroll]
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

# Speaker catalog management (requires --diarize + wespeaker)
any2md speaker add NAME --audio FILE   # Enroll a speaker from an audio file
any2md speaker list                    # List all enrolled speakers
any2md speaker remove NAME             # Delete a speaker and their embeddings
any2md speaker merge NAME_A NAME_B     # Merge FROM into INTO (deletes NAME_A)
any2md speaker stats NAME              # Distance stats for a speaker's gallery
any2md speaker gallery NAME            # Show enrolled embedding gallery entries

# Global flags (all subcommands)
  --json, -j                # JSON output to stdout (agent-friendly)
  --fields FIELDS           # Dot-notation field selection for --json (e.g., "frontmatter.rows,content")
  --quiet, -q, --silent     # Suppress INFO logs (or set ANY2MD_QUIET=1)
  --version, -V             # Print version and exit
  -o, --output-dir PATH     # Output directory (default: cwd)
  -f, --format [md|txt]     # Output format (default: md)
  -v, --verbose             # DEBUG logging

# Utility subcommands
any2md deps                 # Show installed/missing optional dependencies
any2md docs                 # Print README documentation to stdout
any2md completion bash      # Output bash completion script
any2md completion zsh       # Output zsh completion script
any2md completion fish      # Output fish completion script
```

**yt speaker identification flags** (require `--diarize`):
- `--identify` — Match diarized segments against speaker catalog (requires wespeaker)
- `--auto-enroll` — Auto-enroll unmatched speakers as Unknown_N without prompting (requires `--identify`)
- `--no-enroll` — Skip enrollment prompts; leave unmatched speakers as SPEAKER_N labels (requires `--identify`; mutually exclusive with `--auto-enroll`)

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

Originally `yt2srt` on `lightning-whisper-mlx`. Migrated to `yt2md` on `mlx-audio` with Parakeet, rewritten from argparse to typer. Then expanded from 2 tools (yt2md + pdf2md) to 20 converters as a proper Python package (`src/any2md/`) with unified CLI, optional dependency groups, and 808 tests. v0.2.0 added quality & polish pass; v0.2.3 added `repo` subcommand via repomix; v0.3.0 added `speaker` catalog, `docs`/`completion` subcommands, `--silent` flag, and `--identify`/`--auto-enroll`/`--no-enroll` flags on `yt`; v0.3.4 added Qwen3-ASR-1.7B multilingual STT + Whisper-tiny LID auto-routing + `--language` / `--aligner-model` flags (ANY2-28 series). v0.4.0 (ANY2-36) upgraded all dependencies to latest (mlx 0.32, mlx-audio 0.5.9, mlx-vlm 0.7.8, mlx-lm 0.32, transformers 5.20), fixed the memory issues behind intermittent leaks (MLX cache cap + stage-wise model release + bounded diarization/alignment/LID), switched diarization to Nemotron-3-Diarization, and added Qwen3.6/Qwen3.8 OCR aliases.
