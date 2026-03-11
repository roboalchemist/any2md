# 2md Goals

## Vision

Turn 2md into a comprehensive **anything-to-markdown** toolkit that runs entirely on Apple Silicon, leveraging MLX for all AI inference. Every converter produces clean markdown with YAML frontmatter. No cloud APIs required.

## Philosophy

- **Local-first**: All AI runs on-device via MLX (mlx-audio, mlx-lm, mlx-vlm)
- **Consistent output**: Every tool produces markdown with YAML frontmatter
- **One tool per source type**: `yt2md`, `pdf2md`, `web2md`, `doc2md`, `img2md`, etc.
- **Shared code**: Common frontmatter builder, CLI patterns (typer), and output formatting
- **Minimal dependencies**: Prefer single-purpose libraries over kitchen-sink frameworks

## MLX Model Stack

| Layer | Library | Models | Purpose |
|-------|---------|--------|---------|
| **STT** | `mlx-audio[stt]` | Parakeet v3 (0.6b, 1.1b) | Audio/video transcription |
| **HTML→MD** | `mlx-lm` | ReaderLM-v2 (`mlx-community/jinaai-ReaderLM-v2`, 4-bit, 869MB) | Clean web content extraction from raw HTML |
| **Vision** | `mlx-vlm` | Qwen3.5-27B or Qwen2.5-VL-7B (4-bit) | Image OCR, slide understanding, diagram description |
| **Text** | `mlx-lm` | (optional, for cleanup) | Transcript cleanup, summarization |

### Key Model Facts

- **ReaderLM-v2**: 1.5B params, 512K context, MIT license. Outperforms GPT-4o on HTML→markdown (ROUGE-L 0.84). Feed raw HTML, get clean markdown. MLX 4-bit at `mlx-community/jinaai-ReaderLM-v2`.
- **Qwen3.5-27B**: Multimodal (text+image+video, NOT audio). 262K context. MLX 4-bit needs ~27-36GB RAM. For smaller Macs, use Qwen3.5-9B (4-bit, ~10GB) or Qwen2.5-VL-7B.
- **Parakeet v3**: Already integrated. 0.6B default, 1.1B for higher accuracy. Handles chunked long audio internally.

## New Converters

### Priority 1: web2md.py — Web Pages to Markdown

**Pipeline**: URL → fetch HTML (requests/httpx) → ReaderLM-v2 (mlx-lm) → markdown with frontmatter

- Fetch page HTML with proper headers
- Run through ReaderLM-v2 locally to extract clean article content
- Generate frontmatter: title, author, date, sitename, url, description, fetched_at
- Fall back to `trafilatura` for metadata extraction if ReaderLM doesn't capture it
- Support: single URL, list of URLs from file, stdin pipe

**Why ReaderLM-v2 over trafilatura alone**: ReaderLM handles malformed HTML, complex layouts, nested tables, and code blocks better than rule-based extractors. It's a 1.5B model that fits in <1GB RAM at 4-bit.

### Priority 2: doc2md.py — Office Documents to Markdown

**Pipeline**: DOCX/PPTX/XLSX/EPUB → `markitdown` extraction → markdown with frontmatter

- Use Microsoft's `markitdown` library (`pip install markitdown[all]`)
- Auto-detect format from extension
- DOCX: headings, lists, tables, images preserved
- PPTX: slide-per-section layout with speaker notes
- XLSX: sheets as markdown tables
- EPUB: chapter structure preserved
- Generate frontmatter from document properties: title, author, created, modified, subject, format, pages/slides/sheets

### Priority 3: img2md.py — Images to Markdown

**Pipeline**: Image → Qwen3.5 VLM (mlx-vlm) → markdown with frontmatter

- OCR text extraction from photos of documents, whiteboards, screenshots
- Diagram/chart description and data extraction
- Slide photo → structured markdown
- Handwriting recognition
- Generate frontmatter: source, dimensions, format, model_used, fetched_at
- Support batch processing (directory of images)
- Model selection: `--model qwen3.5-27b` (high quality) or `--model qwen2.5-vl-7b` (fast, lower RAM)

### Priority 4: html2md.py — Local HTML Files to Markdown

**Pipeline**: HTML file → ReaderLM-v2 (mlx-lm) → markdown with frontmatter

- For local `.html` files (saved pages, exports, scraped archives)
- Same ReaderLM-v2 engine as web2md, but reads from file instead of fetching
- Batch mode: process a directory of HTML files
- Frontmatter from `<meta>` tags and `<title>`

### Future Ideas (Lower Priority)

| Tool | Source | Library | Notes |
|------|--------|---------|-------|
| `email2md.py` | EML/MBOX files | `email` stdlib + ReaderLM-v2 for HTML body | Underserved niche. Frontmatter: from, to, subject, date, attachments |
| `rss2md.py` | RSS/Atom feeds | `feedparser` + ReaderLM-v2 per article | Podcast feed → episode list with show notes |
| `csv2md.py` | CSV/TSV files | `pytablewriter` or pandas | Data tables with column type detection |
| `slide2md.py` | PPTX with VLM | `mlx-vlm` + Qwen3.5 | VLM-powered slide understanding (vs pure text extraction in doc2md) |
| `repo2md.py` | Git repositories | Tree walking + ast-grep | Codebase structure, README, key files as single markdown |

## Architecture

### Shared Module: `md_common.py`

Extract shared code from yt2md.py into a common module:

```
md_common.py
├── build_frontmatter(metadata: dict) -> str     # Already exists in yt2md
├── write_output(content: str, path: str) -> str  # Common file writer
├── detect_format(path: str) -> str               # File type detection
└── setup_logging(verbose: bool) -> Logger        # Common logging setup
```

### CLI Pattern

Every tool follows the same typer pattern:
```bash
python <tool>.py <input> [OPTIONS]
  -o, --output-dir PATH     # Output directory (default: cwd)
  -f, --format [md|txt]     # Output format (default: md)
  -m, --model NAME          # Model override (where applicable)
  -v, --verbose             # DEBUG logging
```

### Model Management

Models are downloaded on first use via HuggingFace Hub. Consider a shared `download_models.py` that pre-fetches all models:

```bash
python download_models.py              # Download all default models
python download_models.py --stt        # STT models only (Parakeet)
python download_models.py --vlm        # Vision models only (Qwen3.5)
python download_models.py --reader     # ReaderLM-v2 only
```

## Memory Budget (Apple Silicon)

| Model | 4-bit Size | Use Case |
|-------|-----------|----------|
| Parakeet v3 0.6B | ~0.5 GB | Audio transcription |
| ReaderLM-v2 1.5B | ~0.9 GB | HTML→markdown |
| Qwen2.5-VL-7B | ~4 GB | Image understanding (default) |
| Qwen3.5-27B | ~14 GB | Image understanding (high quality) |

- **16GB Mac**: Can run Parakeet + ReaderLM-v2 + Qwen2.5-VL-7B comfortably
- **32GB+ Mac**: Can run everything including Qwen3.5-27B
- Models are loaded on-demand, not all at once

## Non-Goals

- No cloud API dependencies (no OpenAI, no Jina API, no Google)
- No GUI — CLI only
- No real-time/streaming — batch processing only
- No translation — output matches source language
- Not a general-purpose LLM chat tool — focused on conversion to markdown
