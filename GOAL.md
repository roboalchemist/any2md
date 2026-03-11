# Goal: Comprehensive anything-to-markdown toolkit on Apple Silicon

2md converts any media source to clean markdown with YAML frontmatter. All AI inference runs locally via MLX — no cloud APIs. Currently has `yt2md.py` (audio/video via Parakeet) and `pdf2md.py` (PDF via pymupdf4llm). Goal: expand to web pages, office docs, images, and local HTML, all sharing a common code module.

**IMPORTANT — Research-first approach**: Before implementing any tool, dispatch internet-researcher agents to verify library versions, API changes, and MLX model availability.

## Phase 1 — Foundation (md_common.py + infrastructure) ⬜ NOT STARTED

Extract shared code from yt2md.py into md_common.py so all tools can import from it. Update existing tools. Rewrite download_models.py with typer.

### 1A — Create md_common.py
Extract from yt2md.py: `build_frontmatter()`, `setup_logging()`, `write_output()`, `OutputFormat` base. Also add `load_vlm(model)` stub for future VLM tools.

### 1B — Update yt2md.py and pdf2md.py
Import `build_frontmatter` and `setup_logging` from `md_common` instead of defining them locally. All existing tests must still pass.

### 1C — Rewrite download_models.py
Convert from argparse to typer. Add flags: `--stt`, `--vlm`, `--reader`, `--docling`. Support downloading all model groups or individual ones.

**Success**: All 25 existing tests pass. `from md_common import build_frontmatter` works. `python download_models.py --help` shows typer-style help.

## Phase 2 — web2md.py (ReaderLM-v2) ⬜ NOT STARTED

Build web page → markdown converter using ReaderLM-v2 locally via mlx-lm.

### 2A — Verify ReaderLM-v2 on mlx-lm (CRITICAL — DO THIS FIRST)
Install mlx-lm, load `mlx-community/jinaai-ReaderLM-v2`, verify it generates clean markdown from HTML. Check current mlx-lm API (generate function signature may differ from training data).

### 2B — Build web2md.py
URL fetch via httpx → ReaderLM-v2 for extraction → frontmatter (title, author, date, sitename, url, description, fetched_at) via trafilatura metadata fallback. Typer CLI matching existing tool patterns.

### 2C — Tests for web2md.py
Unit tests mocking the model and httpx calls. At least 8 tests covering: URL fetch, metadata extraction, markdown output, frontmatter fields, error handling.

**Success**: `python web2md.py https://example.com` produces clean markdown with frontmatter. All tests pass.

## Phase 3 — doc2md.py (markitdown) ⬜ NOT STARTED

Build office document → markdown converter using Microsoft's markitdown library.

### 3A — Verify markitdown library
Install `markitdown[all]`, verify DOCX/PPTX/XLSX/EPUB conversion works. Check current API surface.

### 3B — Build doc2md.py
Auto-detect format from extension → markitdown extraction → frontmatter from doc properties (title, author, created, modified, format, pages/slides/sheets). Typer CLI.

### 3C — Tests for doc2md.py
Unit tests with sample documents (create minimal test files). At least 8 tests.

**Success**: `python doc2md.py document.docx` produces markdown with frontmatter. All tests pass.

## Phase 4 — img2md.py (Qwen3.5 VLM) ⬜ NOT STARTED

Build image → markdown converter using Qwen3.5 via mlx-vlm.

### 4A — Verify mlx-vlm + Qwen3.5 (CRITICAL)
Install mlx-vlm, load Qwen3.5-27B (or 9B for speed), verify image inference works. Check current mlx-vlm API.

### 4B — Build img2md.py
Image loading → VLM inference with document extraction prompt → markdown + frontmatter (source, dimensions, format, model_used, fetched_at). Model selection via `--model` flag.

### 4C — Batch mode
Process a directory of images. Output one .md file per image.

### 4D — Tests for img2md.py
Unit tests mocking mlx-vlm. At least 8 tests.

**Success**: `python img2md.py photo.jpg` produces OCR markdown with frontmatter. All tests pass.

## Phase 5 — html2md.py (local HTML files) ⬜ NOT STARTED

Reuse ReaderLM-v2 engine from web2md for local HTML files.

### 5A — Build html2md.py
Read local HTML file → ReaderLM-v2 extraction (reuse from web2md) → frontmatter from `<meta>` tags. Batch mode for directories.

### 5B — Tests for html2md.py
At least 8 tests.

**Success**: `python html2md.py page.html` produces markdown with frontmatter. All tests pass.

## Phase 6 — Enhanced pdf2md (VLM fallback for scanned pages) ⬜ NOT STARTED

Depends on Phase 4 (img2md VLM patterns).

### 6A — Add --ocr flag and thin-page detection
pdf2md already has THIN_PAGE_THRESHOLD. Wire it up to trigger VLM fallback automatically.

### 6B — Integrate Qwen3.5 VLM
For pages with <50 chars (scanned), render page as image → Qwen3.5 via mlx-vlm → markdown. Reuse load_vlm() from md_common.

### 6C — Hybrid logic + tests
Text path for born-digital, VLM path for scanned. At least 5 new tests.

**Success**: `python pdf2md.py scanned.pdf --ocr` produces markdown from image-only pages.

## Phase 7 — Polish & Keep Going ⬜ NOT STARTED

- Update download_models.py for all new models
- Update requirements.txt with all new dependencies
- README.md refresh with all tools
- Performance benchmarks
- Future tools: email2md, rss2md, csv2md, repo2md

## Available Resources

- **Code**: `/Users/joseph.schlesinger/gitea/2md/` — all source files
- **Existing tools**: `yt2md.py` (759 lines), `pdf2md.py` (320 lines) — patterns to follow
- **Tests**: 25 passing tests across test_yt2md.py and test_pdf2md.py — study for patterns
- **Machine**: 36GB Apple Silicon Mac — fits Qwen3.5-27B 4-bit (~17GB) easily
- **Python env**: venv at project root (or system Python with mlx-audio installed)
- **MLX models available**: mlx-community/* on HuggingFace Hub

## Lessons Learned

### What works
- Parakeet v3 via mlx-audio for STT — fast, accurate, already integrated
- pymupdf4llm for text-layer PDF extraction — 0.1s/page, no GPU needed
- typer for CLI — much cleaner than argparse, use type hints over decorators
- Hand-rolled YAML frontmatter builder — no PyYAML dep, handles all edge cases
- unittest with mock objects for testing — FakeAlignedSentence pattern works well

### What fails
- (none yet — this section accumulates as we learn)

### Critical issues
- mlx-lm and mlx-vlm APIs change frequently — always verify current API before implementing
- ReaderLM-v2 context limit is 512K tokens — HTML pages must be truncated if larger
- Qwen3.5-27B takes ~17GB — don't load it alongside other large models

## Success Criteria

- **Phase 1**: All 25 existing tests pass after md_common.py extraction
- **Phase 2**: web2md.py converts a real URL to clean markdown with frontmatter
- **Phase 3**: doc2md.py converts a real DOCX/PPTX file to markdown with frontmatter
- **Phase 4**: img2md.py extracts text from a real image via Qwen3.5
- **Phase 5**: html2md.py converts a local HTML file to markdown with frontmatter
- **Phase 6**: pdf2md.py handles scanned PDFs with --ocr flag
- **Phase 7**: All tools documented, requirements.txt complete, README refreshed
- After all: keep going — add email2md, rss2md, etc.

## Operating Rules

- All AI inference via MLX — no cloud APIs
- Every tool follows the same typer CLI pattern as existing tools
- Every tool produces YAML frontmatter
- All tests use unittest (stdlib), run via pytest
- Import at module top — no lazy imports
- Follow existing code style in yt2md.py and pdf2md.py
- Run all tests after every phase: `python -m pytest test_*.py -q`
