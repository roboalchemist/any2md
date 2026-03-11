# 2md - Media to Markdown Toolkit

## Overview

A toolkit for converting media (YouTube videos, audio files, PDFs) to markdown with YAML frontmatter. Built on [mlx-audio](https://github.com/Blaizzy/mlx-audio) (Parakeet v3) for transcription and [pymupdf4llm](https://github.com/pymupdf/RAG) for PDF extraction. Optimized for Apple Silicon.

## Project Structure

```
2md/
├── yt2md.py               # YouTube/audio/video → markdown/SRT/text (typer CLI)
├── pdf2md.py              # PDF → markdown/text (typer CLI, imports build_frontmatter from yt2md)
├── whisper_benchmark.py   # Interactive benchmark (STALE: imports from yt2srt)
├── benchmark_models.py    # Automated benchmark (STALE: imports from yt2srt)
├── download_models.py     # Pre-download models (STALE: imports from yt2srt)
├── quant_test.py          # Quick smoke test (STALE: uses old lightning-whisper-mlx)
├── test_quant.py          # Old quantization test (STALE: uses lightning-whisper-mlx)
├── transcription_cleanup_prompt.txt  # LLM prompt template for cleaning raw transcripts
├── requirements.txt       # Python dependencies
├── test_yt2md.py          # 15 unit tests for yt2md
├── test_pdf2md.py         # 10 unit tests for pdf2md
├── test_benchmark.py      # 3 tests for whisper_benchmark (STALE: depends on yt2srt)
├── test_audio/            # Test audio files (mp3/wav, tracked via .gitignore exceptions)
├── benchmark-results/     # Generated benchmark reports (gitignored)
├── worklog.md             # Development history (mostly from lightning-whisper-mlx era)
└── .cursor/mcp.json       # Cursor MCP config
```

**Stale files**: `whisper_benchmark.py`, `benchmark_models.py`, `download_models.py`, `quant_test.py`, and `test_quant.py` all import from `yt2srt` (the old module name) or `lightning_whisper_mlx` (the old library). They will fail if run.

## Key Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `mlx-audio[stt]` | >=0.4.0 | Parakeet/Whisper STT on Apple Silicon (MLX) |
| `pymupdf4llm` | >=0.2.0 | PDF text extraction to markdown |
| `typer` | >=0.9.0 | CLI framework (with `typing_extensions`, `rich_markup_mode`) |
| `yt-dlp` | >=2023.11.14 | YouTube audio download |
| `numba` | >=0.61.0 | JIT compilation (mlx-audio dependency) |
| `numpy` | >=2.0.0 | Numerical arrays |
| `tqdm` | >=4.64.1 | Progress bars |
| `more-itertools` | >=10.1.0 | Iterator utilities |
| `ffmpeg`/`ffprobe` | system | Audio conversion and duration detection |

## Architecture

### Code Organization

- **yt2md.py** (759 lines) — Main transcription tool, also exports shared utilities
  - `build_frontmatter(metadata)` — Hand-rolled YAML frontmatter generator (no PyYAML dep)
  - `resolve_model(name)` — Alias resolution for model short names
  - `extract_video_id(url_or_id)` — YouTube URL/ID parser
  - `auto_detect_input(path)` — Determines if input is YouTube URL, YouTube ID, or local file
  - `download_youtube_audio(url_or_id)` — yt-dlp wrapper returning (audio_path, metadata)
  - `convert_audio_for_whisper(input)` — ffmpeg to 16kHz mono WAV
  - `transcribe(audio_file, ...)` — mlx-audio model.generate() wrapper
  - `segments_to_markdown/srt/text()` — Output formatters
  - `OutputFormat` — Enum: md, srt, txt
  - `app` — typer.Typer CLI app

- **pdf2md.py** (320 lines) — PDF extraction tool
  - Imports `build_frontmatter` from `yt2md` (shared code)
  - `parse_page_range(range_str, total)` — "1-10,15,20-25" → 0-based indices
  - `extract_pdf_metadata(doc)` — fitz Document metadata extraction
  - `extract_pages(pdf_path, page_indices)` — pymupdf4llm.to_markdown() with page_chunks
  - `pages_to_markdown/text()` — Output formatters
  - `OutputFormat` — Enum: md, txt (no SRT for PDFs)
  - `app` — typer.Typer CLI app

### Data Flow

```
yt2md: Input → auto_detect → [yt-dlp download | local file] → ffmpeg convert → mlx-audio transcribe → format → write
pdf2md: PDF → fitz metadata → pymupdf4llm extract → format with frontmatter → write
```

### Model System

Default model: `mlx-community/parakeet-tdt-0.6b-v3`

| Alias | Full HuggingFace ID |
|-------|---------------------|
| `parakeet-v3` | `mlx-community/parakeet-tdt-0.6b-v3` |
| `parakeet-v2` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `parakeet-1.1b` | `mlx-community/parakeet-tdt-1.1b` |
| `parakeet-ctc` | `mlx-community/parakeet-ctc-0.6b` |
| `whisper-turbo` | `mlx-community/whisper-large-v3-turbo-asr-fp16` |

### Frontmatter

Both tools generate YAML frontmatter using a shared `build_frontmatter()` that manually formats YAML (no PyYAML dependency). Handles scalars, lists, nested dicts (chapters), multi-line strings (description via `|` block scalar), and auto-quoting of special characters.

**YouTube frontmatter fields**: title, video_id, url, channel, channel_url, uploader, upload_date, duration, duration_human, language, location, availability, live_status, view_count, like_count, comment_count, channel_follower_count, thumbnail, categories, tags, subtitles, auto_captions, chapters, description, fetched_at

**PDF frontmatter fields**: title, author, subject, keywords, creator, producer, created, modified, format, pages, source, fetched_at

## Testing

- **Framework**: `unittest` (stdlib classes), run via `pytest`
- **Test count**: 25 passing (15 yt2md + 10 pdf2md)
- **Run all**: `python -m pytest test_yt2md.py test_pdf2md.py`
- **Run individually**: `python -m pytest test_yt2md.py -v` or `python -m pytest test_pdf2md.py -v`
- **Test patterns**: FakeAlignedSentence objects mock mlx-audio output; dict-based test data for backward compat
- **No integration tests**: Tests cover pure functions only (formatting, parsing, frontmatter), not actual transcription or download
- **Benchmark tests** (`test_benchmark.py`): Currently broken — imports from `yt2srt`

## CLI Reference

### yt2md.py

```bash
python yt2md.py <input> [OPTIONS]

# Input: YouTube URL, video ID (11 chars), or local file path
# Options:
#   -m, --model TEXT          Model alias or HuggingFace ID [default: mlx-community/parakeet-tdt-0.6b-v3]
#   -o, --output-dir PATH     Output directory [default: cwd]
#   -f, --format [md|srt|txt] Output format [default: md]
#   -c, --chunk-duration FLOAT Chunk length in seconds [default: 30.0]
#   -k, --keep-audio          Keep downloaded/converted audio files
#   -v, --verbose             DEBUG logging
```

### pdf2md.py

```bash
python pdf2md.py <input.pdf> [OPTIONS]

# Options:
#   -o, --output-dir PATH     Output directory [default: cwd]
#   -f, --format [md|txt]     Output format [default: md]
#   -p, --pages TEXT           Page range (e.g. "1-10,15,20-25")
#   -v, --verbose             DEBUG logging
```

## Installation

```bash
brew install ffmpeg
pip install -r requirements.txt

# Optional: pre-download STT models (requires fixing download_models.py imports first)
# python download_models.py
```

## Project History

Originally built as `yt2srt` on `lightning-whisper-mlx`. Migrated to `yt2md` on `mlx-audio` with Parakeet v3, then rewritten from argparse to typer. Several auxiliary scripts (`whisper_benchmark.py`, `benchmark_models.py`, `download_models.py`, `quant_test.py`) still reference the old `yt2srt` module name and need updating.
