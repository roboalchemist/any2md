# 2md — Anything to Markdown

A toolkit for converting media, documents, and web content to markdown. All AI inference runs locally on Apple Silicon via MLX — no cloud APIs.

## Install

```bash
# From PyPI (coming soon)
uv pip install '2md[all]'

# From source
git clone https://github.com/roboalchemist/2md.git
cd 2md
uv pip install -e '.[all]'

# Or install only what you need
uv pip install -e '.[stt]'    # YouTube/audio/video transcription
uv pip install -e '.[pdf]'    # PDF extraction
uv pip install -e '.[img]'    # Image OCR via VLM
uv pip install -e '.[web]'    # Web page conversion

# System dependency for audio/video
brew install ffmpeg
```

## Usage

### Auto-detect (just pass a file)

```bash
2md lecture.mp4                        # audio/video → markdown
2md lecture.mp4 --diarize              # with speaker diarization
2md document.pdf                       # PDF → markdown
2md screenshot.png                     # image → markdown (VLM)
2md https://example.com/article        # web page → markdown
2md page.html                          # local HTML → markdown
2md report.docx                        # office doc → markdown
2md readme.rst                         # RST → markdown
```

### Explicit subcommands (for full options)

```bash
2md yt --help       # audio/video options
2md pdf --help      # PDF options
2md img --help      # image options
2md web --help      # web URL options
2md html --help     # local HTML options
2md doc --help      # office document options
2md rst --help      # RST options
```

### Examples

```bash
# YouTube with speaker diarization
2md yt "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --diarize -o ~/notes

# PDF with page range
2md pdf document.pdf -p 1-10,15 -o ~/notes

# PDF with VLM OCR for scanned pages
2md pdf scanned.pdf --ocr

# Image directory
2md img ~/screenshots/ --prompt "Extract all text and tables"

# Specific model
2md yt podcast.mp3 -m parakeet-1.1b -f srt
```

## Tools

| Subcommand | Input | Engine | Output |
|------------|-------|--------|--------|
| `yt` | YouTube URLs, audio, video | Parakeet STT + Sortformer diarization (mlx-audio) | md / srt / txt |
| `pdf` | PDF files | pymupdf4llm, optional Qwen VLM OCR | md / txt |
| `img` | JPEG, PNG, GIF, BMP, WebP, TIFF | Qwen2.5-VL (mlx-vlm) | md / txt |
| `web` | Web URLs | ReaderLM-v2 (mlx-lm) | md / txt |
| `html` | Local HTML files | ReaderLM-v2 (mlx-lm) | md / txt |
| `doc` | DOCX, PPTX, XLSX, EPUB, ODT, RTF | markitdown | md / txt |
| `rst` | reStructuredText | pypandoc / docutils | md / txt |

## Architecture

```
src/tomd/
├── cli.py       # Unified entry point with auto-detect + subcommands
├── common.py    # Shared: frontmatter builder, logging, output helpers
├── yt.py        # Audio/video transcription + speaker diarization
├── pdf.py       # PDF extraction + optional VLM OCR
├── img.py       # Image OCR via vision-language model
├── web.py       # Web URL → markdown via ReaderLM
├── html.py      # Local HTML → markdown via ReaderLM
├── doc.py       # Office documents via markitdown
└── rst.py       # reStructuredText conversion
```

All tools produce YAML frontmatter with source metadata (title, author, dates, etc.) followed by the converted markdown body.

AI runs locally on Apple Silicon via MLX:
- **STT**: mlx-audio Parakeet — speech recognition
- **Diarization**: mlx-audio Sortformer — speaker identification
- **VLM**: mlx-vlm Qwen2.5-VL — image OCR and understanding
- **HTML→MD**: mlx-lm ReaderLM-v2 — HTML to markdown

## Pre-download models

```bash
python scripts/download_models.py --stt       # Parakeet (yt)
python scripts/download_models.py --diarize   # Sortformer (yt --diarize)
python scripts/download_models.py --vlm       # Qwen2.5-VL (img, pdf --ocr)
python scripts/download_models.py --reader    # ReaderLM-v2 (web, html)
python scripts/download_models.py --all       # Everything
```

## Requirements

- Python 3.11+
- Apple Silicon Mac (M1/M2/M3/M4)
- ffmpeg (`brew install ffmpeg`) — for audio/video
- pandoc (`brew install pandoc`) — optional, for rst2md

## License

[MIT](LICENSE)
