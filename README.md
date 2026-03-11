# 2md — Anything to Markdown

A toolkit for converting media, documents, and web content to markdown. All AI inference runs locally on Apple Silicon via MLX — no cloud APIs.

## Tools

| Tool | Input | Method | Output |
|------|-------|--------|--------|
| `yt2md.py` | YouTube URLs, audio, video files | Parakeet STT (mlx-audio) | md / srt / txt |
| `pdf2md.py` | PDF files | pymupdf4llm, optional VLM OCR | md / txt |
| `web2md.py` | Web URLs | ReaderLM-v2 (mlx-lm) | md / txt |
| `doc2md.py` | DOCX, PPTX, XLSX, EPUB, ODT, RTF | markitdown | md / txt |
| `img2md.py` | JPEG, PNG, GIF, BMP, WebP, TIFF | Qwen2.5-VL (mlx-vlm) | md / txt |
| `html2md.py` | Local HTML files | ReaderLM-v2 (mlx-lm) | md / txt |
| `rst2md.py` | reStructuredText files | pypandoc / docutils | md / txt |

## Quick Start

```bash
brew install ffmpeg
pip install -r requirements.txt

# Optional: pre-download MLX models before first use
python download_models.py --stt       # Parakeet (yt2md)
python download_models.py --vlm       # Qwen2.5-VL (img2md, pdf2md --ocr)
python download_models.py --reader    # ReaderLM-v2 (web2md, html2md)
```

## Usage

### yt2md — YouTube / audio / video

```bash
# YouTube URL or video ID
python yt2md.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python yt2md.py dQw4w9WgXcQ

# Local audio/video file
python yt2md.py lecture.mp4 -f srt -o ~/subtitles

# Options
python yt2md.py dQw4w9WgXcQ -m parakeet-v3 -f md -o ~/notes -k -v
#   -m MODEL     Model alias or HuggingFace ID [default: mlx-community/parakeet-tdt-0.6b-v3]
#   -f FORMAT    md | srt | txt  [default: md]
#   -o DIR       Output directory
#   -k           Keep downloaded/converted audio
#   -v           Verbose logging
```

Model aliases: `parakeet-v3`, `parakeet-v2`, `parakeet-1.1b`, `parakeet-ctc`

### pdf2md — PDF files

```bash
python pdf2md.py document.pdf
python pdf2md.py document.pdf -p 1-10,15 -o ~/notes

# VLM OCR for scanned/image-only PDFs
python pdf2md.py scanned.pdf --ocr           # VLM fallback on thin-text pages
python pdf2md.py scanned.pdf --force-ocr     # VLM on every page

# Options
#   -p RANGE     Page range, e.g. "1-10,15,20-25"
#   --ocr        VLM fallback for scanned pages
#   --force-ocr  Force VLM on all pages
#   --vlm-model  VLM model [default: mlx-community/Qwen2.5-VL-7B-Instruct-4bit]
```

### web2md — Web URLs

```bash
python web2md.py https://example.com/article
python web2md.py https://example.com/article -o ~/notes -f md
```

### doc2md — Office documents

```bash
python doc2md.py report.docx
python doc2md.py slides.pptx -o ~/notes
python doc2md.py spreadsheet.xlsx -f txt
```

Supported: DOCX, PPTX, XLSX, EPUB, ODT, RTF

### img2md — Images

```bash
python img2md.py screenshot.png
python img2md.py ~/screenshots/    # Process entire directory

# Options
#   -m MODEL     VLM alias or HuggingFace ID [default: qwen2.5-vl-7b]
#   --prompt     Custom instruction for the VLM
#   --max-tokens Max tokens per image [default: 2048]
```

Model aliases: `qwen2.5-vl-7b`, `qwen2.5-vl-3b`, `qwen2.5-vl-2b`, `qwen2.5-vl-72b`, `smoldocling`

### html2md — Local HTML files

```bash
python html2md.py page.html
python html2md.py ~/exported-site/    # Process all .html/.htm files in directory
```

### rst2md — reStructuredText

```bash
python rst2md.py docs/readme.rst
python rst2md.py docs/            # Process all .rst/.rest files in directory
```

## Architecture

All tools share `md_common.py` for frontmatter generation, logging setup, and output utilities. Each tool produces YAML frontmatter with source metadata (title, author, dates, etc.) followed by the converted markdown body.

AI tools use local MLX inference only:
- **STT**: mlx-audio (Parakeet) — fast Apple Silicon speech recognition
- **VLM**: mlx-vlm (Qwen2.5-VL) — vision-language model for images and OCR
- **HTML/URL**: mlx-lm (ReaderLM-v2) — HTML-to-markdown conversion

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- ffmpeg (`brew install ffmpeg`) — required for yt2md
- pypandoc requires pandoc (`brew install pandoc`) — required for rst2md primary path
