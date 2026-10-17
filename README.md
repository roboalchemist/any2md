# any2md — Convert Anything to Markdown

A toolkit for converting any media, document, or data format to markdown. AI inference runs locally on Apple Silicon via MLX — no cloud APIs.

## Install

```bash
# Homebrew (macOS)
brew install roboalchemist/tap/any2md

# From source
git clone https://github.com/roboalchemist/any2md.git && cd any2md
uv pip install -e '.[all]'
```

The base install includes all 8 zero-dependency converters (csv, data, db, nb, eml, org, tex, man). For AI-powered converters, add optional dependencies:

```bash
uv pip install "mlx-audio[stt]" yt-dlp   # Audio/video transcription
uv pip install pymupdf4llm               # PDF extraction
uv pip install mlx-vlm                   # Image OCR via VLM
uv pip install mlx-lm httpx              # Web page conversion
uv pip install markitdown                # Office documents

# System dependency for audio/video
brew install ffmpeg
```

## Usage

Just pass a file — any2md auto-detects the format:

```bash
any2md lecture.mp4                  # audio/video → markdown
any2md podcast.mp3 --diarize        # with speaker diarization
any2md document.pdf                 # PDF → markdown
any2md screenshot.png               # image → markdown (VLM OCR)
any2md https://example.com          # web page → markdown
any2md page.html                    # local HTML → markdown
any2md report.docx                  # office doc → markdown
any2md data.csv                     # CSV/TSV → markdown table
any2md config.json                  # JSON/YAML → markdown
any2md app.db                       # SQLite → schema + data
any2md captions.srt                 # subtitles → markdown
any2md notebook.ipynb               # Jupyter → markdown
any2md message.eml                  # email → markdown
any2md notes.org                    # Org-mode → markdown
any2md paper.tex                    # LaTeX → markdown
any2md command.1                    # man page → markdown
any2md readme.rst                   # RST → markdown
```

Use explicit subcommands for full control:

```bash
any2md audio podcast.mp3 --diarize --model parakeet-1.1b
any2md video lecture.mp4 -f srt -o ~/subtitles
any2md yt "https://youtube.com/watch?v=..." --diarize
any2md pdf document.pdf --pages 1-10 --ocr
any2md csv data.tsv --max-rows 50 --max-col-width 40
any2md db app.sqlite --max-rows 20 --skip-views
any2md sub captions.vtt -f txt
any2md nb notebook.ipynb --no-outputs
```

## Supported Formats

| Subcommand | Input | Engine | Dependencies |
|------------|-------|--------|-------------|
| `audio` | MP3, WAV, FLAC, OGG, AAC, M4A | Parakeet STT (English) / Qwen3-ASR (multilingual) + Nemotron diarization | mlx-audio |
| `video` | MP4, MKV, AVI, MOV, WebM | Parakeet STT (English) / Qwen3-ASR (multilingual) + Nemotron diarization | mlx-audio |
| `yt` | YouTube URLs + all audio/video | Parakeet STT (English) / Qwen3-ASR (multilingual) + yt-dlp download | mlx-audio, yt-dlp |
| `pdf` | PDF files | pymupdf4llm + optional Qwen3.5 VLM OCR | pymupdf4llm |
| `img` | JPEG, PNG, GIF, BMP, WebP, TIFF | Qwen3.5 (mlx-vlm) | mlx-vlm |
| `web` | Web URLs | ReaderLM-v2 (mlx-lm) | mlx-lm |
| `html` | Local HTML files | ReaderLM-v2 (mlx-lm) | mlx-lm |
| `doc` | DOCX, PPTX, XLSX, EPUB, ODT, RTF | markitdown | markitdown |
| `rst` | reStructuredText | pypandoc / docutils | — |
| `csv` | CSV, TSV | stdlib csv | **none** |
| `data` | JSON, YAML, JSONL | stdlib json, optional PyYAML | **none** |
| `db` | SQLite databases | stdlib sqlite3 | **none** |
| `sub` | SRT, VTT, ASS/SSA subtitles | pysubs2 | pysubs2 |
| `nb` | Jupyter notebooks | stdlib json | **none** |
| `eml` | Email .eml, .mbox | stdlib email/mailbox | **none** |
| `org` | Emacs Org-mode | pure regex | **none** |
| `tex` | LaTeX | pure regex | **none** |
| `man` | Unix man pages | mandoc + regex fallback | **none** |
| `repo` | Git repositories | repomix (npm) | repomix |

**8 of 20 converters are zero-dependency** (stdlib only). `audio` and `video` are aliases for `yt` (same engine, clearer naming).

## Architecture

```
src/any2md/
├── cli.py       # Unified entry point — auto-detect + subcommands
├── common.py    # Shared: frontmatter builder, logging, output helpers
├── yt.py        # Audio/video/YouTube transcription + speaker diarization
├── pdf.py       # PDF extraction + optional VLM OCR
├── img.py       # Image OCR via vision-language model
├── web.py       # Web URL → markdown via ReaderLM
├── html.py      # Local HTML → markdown via ReaderLM
├── doc.py       # Office documents via markitdown
├── rst.py       # reStructuredText conversion
├── csv.py       # CSV/TSV → markdown tables
├── data.py      # JSON/YAML/JSONL → smart markdown
├── db.py        # SQLite → schema + sample data tables
├── sub.py       # Subtitles (SRT/VTT/ASS) → timestamped markdown
├── nb.py        # Jupyter notebooks → markdown
├── eml.py       # Email (.eml/.mbox) → markdown
├── org.py       # Org-mode → markdown
├── tex.py       # LaTeX → markdown
├── man.py       # Unix man pages → markdown
└── repo.py      # Git repositories via repomix
```

All tools produce YAML frontmatter with source metadata followed by the converted markdown body.

## Pre-download AI models

```bash
python scripts/download_models.py --stt       # Parakeet (audio/video, English)
python scripts/download_models.py --asr-zh    # Qwen3-ASR + ForcedAligner + LID (~9.4 GB, multilingual)
python scripts/download_models.py --diarize   # Nemotron-3 diarization (--diarize)
python scripts/download_models.py --vlm       # Qwen3.5 (img, pdf --ocr)
python scripts/download_models.py --reader    # ReaderLM-v2 (web, html)
python scripts/download_models.py --all       # Everything
```

## Multilingual transcription

any2md supports 30+ languages via Qwen3-ASR-1.7B. By default, language is auto-detected from audio using a Whisper-tiny language identification model — no flag needed.

```bash
# Auto-detects language — Chinese, Japanese, Korean, German, French, and 25 more
any2md interview_zh.mp3 --diarize --identify

# Override language explicitly (skips auto-detection)
any2md lecture.mp4 --language zh

# English-only fast path (Parakeet)
any2md podcast.mp3 --language en
```

Supported languages: Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, Japanese (and more via `--language` code). Pass `--language auto` (default) to let the model detect automatically.

## Requirements

- Python 3.11+
- Apple Silicon Mac (M1/M2/M3/M4) — for AI-powered converters
- ffmpeg (`brew install ffmpeg`) — for audio/video
- Non-AI converters (csv, data, db, nb, eml, org, tex, man) work on any platform

## License

[MIT](LICENSE)
