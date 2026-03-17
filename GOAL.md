# Goal: Comprehensive anything-to-markdown toolkit on Apple Silicon

2md converts any media source to clean markdown with YAML frontmatter. All AI inference runs locally via MLX — no cloud APIs. Currently has `yt2md.py` (audio/video via Parakeet) and `pdf2md.py` (PDF via pymupdf4llm). Goal: expand to web pages, office docs, images, and local HTML, all sharing a common code module.

**IMPORTANT — Research-first approach**: Before implementing any tool, dispatch internet-researcher agents to verify library versions, API changes, and MLX model availability.

## Phase 1 — Foundation (md_common.py + infrastructure) ✅ DONE

Extract shared code from yt2md.py into md_common.py so all tools can import from it. Update existing tools. Rewrite download_models.py with typer.

### 1A — Create md_common.py
Extract from yt2md.py: `build_frontmatter()`, `setup_logging()`, `write_output()`, `OutputFormat` base. Also add `load_vlm(model)` stub for future VLM tools.

### 1B — Update yt2md.py and pdf2md.py
Import `build_frontmatter` and `setup_logging` from `md_common` instead of defining them locally. All existing tests must still pass.

### 1C — Rewrite download_models.py
Convert from argparse to typer. Add flags: `--stt`, `--vlm`, `--reader`, `--docling`. Support downloading all model groups or individual ones.

**Success**: All 25 existing tests pass. `from md_common import build_frontmatter` works. `python download_models.py --help` shows typer-style help.

## Phase 2 — web2md.py (ReaderLM-v2) ✅ DONE

Build web page → markdown converter using ReaderLM-v2 locally via mlx-lm.

### 2A — Verify ReaderLM-v2 on mlx-lm (CRITICAL — DO THIS FIRST)
Install mlx-lm, load `mlx-community/jinaai-ReaderLM-v2`, verify it generates clean markdown from HTML. Check current mlx-lm API (generate function signature may differ from training data).

### 2B — Build web2md.py
URL fetch via httpx → ReaderLM-v2 for extraction → frontmatter (title, author, date, sitename, url, description, fetched_at) via trafilatura metadata fallback. Typer CLI matching existing tool patterns.

### 2C — Tests for web2md.py
Unit tests mocking the model and httpx calls. At least 8 tests covering: URL fetch, metadata extraction, markdown output, frontmatter fields, error handling.

**Success**: `python web2md.py https://example.com` produces clean markdown with frontmatter. All tests pass.

## Phase 3 — doc2md.py (markitdown) ✅ DONE

Build office document → markdown converter using Microsoft's markitdown library.

### 3A — Verify markitdown library
Install `markitdown[all]`, verify DOCX/PPTX/XLSX/EPUB conversion works. Check current API surface.

### 3B — Build doc2md.py
Auto-detect format from extension → markitdown extraction → frontmatter from doc properties (title, author, created, modified, format, pages/slides/sheets). Typer CLI.

### 3C — Tests for doc2md.py
Unit tests with sample documents (create minimal test files). At least 8 tests.

**Success**: `python doc2md.py document.docx` produces markdown with frontmatter. All tests pass.

## Phase 4 — img2md.py (Qwen3.5 VLM) ✅ DONE

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

## Phase 5 — html2md.py (local HTML files) ✅ DONE

Reuse ReaderLM-v2 engine from web2md for local HTML files.

### 5A — Build html2md.py
Read local HTML file → ReaderLM-v2 extraction (reuse from web2md) → frontmatter from `<meta>` tags. Batch mode for directories.

### 5B — Tests for html2md.py
At least 8 tests.

**Success**: `python html2md.py page.html` produces markdown with frontmatter. All tests pass.

## Phase 6 — Enhanced pdf2md (VLM fallback for scanned pages) ✅ DONE

Depends on Phase 4 (img2md VLM patterns).

### 6A — Add --ocr flag and thin-page detection
pdf2md already has THIN_PAGE_THRESHOLD. Wire it up to trigger VLM fallback automatically.

### 6B — Integrate Qwen3.5 VLM
For pages with <50 chars (scanned), render page as image → Qwen3.5 via mlx-vlm → markdown. Reuse load_vlm() from md_common.

### 6C — Hybrid logic + tests
Text path for born-digital, VLM path for scanned. At least 5 new tests.

**Success**: `python pdf2md.py scanned.pdf --ocr` produces markdown from image-only pages.

## Phase 7 — rst2md.py (reStructuredText to Markdown) ✅ DONE

Convert `.rst` files (Sphinx docs, Python package READMEs, documentation) to clean markdown.

### 7A — Choose conversion library
Options (in preference order):
1. `pypandoc` — wraps pandoc, most accurate RST→MD (handles all RST features: directives, roles, footnotes)
2. `docutils` — stdlib-adjacent, convert RST→HTML then use markitdown for HTML→MD
3. `m2r2` — direct RST→MD converter (Sphinx-focused)

Check which is installed/installable cleanly. Prefer `pypandoc` if pandoc is on PATH, else `docutils`+markitdown.

### 7B — Build rst2md.py
RST file → library conversion → markdown with frontmatter. Frontmatter from RST docinfo fields (`:Author:`, `:Date:`, `:Version:`) and file metadata. Support batch mode (directory of .rst files).

```bash
python rst2md.py README.rst
python rst2md.py docs/  # batch: all .rst files in directory
```

Frontmatter fields: title, author, date, version, source, fetched_at.

### 7C — Tests for rst2md.py
At least 8 unit tests. Create minimal RST strings in-memory for testing (no real files needed). Mock the conversion library.

**Success**: `python rst2md.py README.rst` produces markdown with frontmatter. All tests pass.

## Phase 9 — Complete Live Testing (unit + integration) ✅ DONE

Current tests are unit-only with mocked dependencies. Nothing actually runs a model, fetches a URL, or reads a real file. This phase adds two layers:

### 9A — Live unit tests (real fixtures, no model inference)

Create `test/fixtures/` directory with minimal real files:
- `test/fixtures/sample.docx` — tiny DOCX with title, author, one paragraph
- `test/fixtures/sample.pptx` — 2-slide PPTX
- `test/fixtures/sample.xlsx` — 1-sheet workbook with 3 rows
- `test/fixtures/sample.epub` — minimal EPUB
- `test/fixtures/sample.rst` — RST with docinfo fields and a code block
- `test/fixtures/sample.html` — HTML with meta tags, title, body text
- `test/fixtures/sample.jpg` — small test image (100x100 pixels, text visible)
- `test/fixtures/sample.pdf` — born-digital PDF (1 page, text-layer)

These tests run without any model inference — they test the actual file parsing, metadata extraction, and output formatting with real files instead of mocks. Skipped if the fixture file is missing.

For each tool: at least 3 tests with real fixtures:
1. File reads and parses without error
2. Metadata fields are populated from actual file properties
3. Output contains valid YAML frontmatter + non-empty markdown body

### 9B — Integration tests (real model inference, skipped if model not downloaded)

Create `test_integration.py` with `@pytest.mark.integration` marker. Each test:
- Checks if the required model is cached locally (skip if not)
- Runs the actual tool end-to-end on a real fixture
- Verifies output is non-empty markdown with valid frontmatter
- Verifies output is deterministic enough to be useful (>50 chars of content)

Tools to cover:
- `doc2md.py` — convert sample.docx (no model, always runs)
- `rst2md.py` — convert sample.rst with real pypandoc (no model, always runs)
- `web2md.py` — mock HTTP fetch but use real ReaderLM-v2 model (skip if model absent)
- `html2md.py` — use sample.html with real ReaderLM-v2 model (skip if model absent)
- `img2md.py` — use sample.jpg with real Qwen3.5 VLM (skip if model absent)
- `pdf2md.py --ocr` — use sample.pdf scanned page with real VLM (skip if model absent)
- `yt2md.py` — transcribe test_audio/test_voice.mp3 with real Parakeet (skip if model absent)

Run integration tests with: `pytest test_integration.py -m integration -v`

### 9C — CI marker separation

Add `pytest.ini` or `pyproject.toml` `[tool.pytest.ini_options]` to define markers:
```ini
[tool.pytest.ini_options]
markers = [
    "integration: requires model inference or network access",
    "slow: takes more than 10 seconds",
]
```

Fast unit tests: `pytest test_*.py -m "not integration"`
All tests incl. integration: `pytest test_*.py test_integration.py`

**Success**:
- `pytest test_*.py -q` still passes (204+)
- `pytest test/` passes with real fixtures (no mocks, no models)
- `pytest test_integration.py -m integration -v` runs and either passes or skips (never errors) based on model availability

### 9D — Real inference integration tests (NO skipping) ✅ DONE

9A–9C done. Gap remaining: integration tests skip when models aren't cached. This sub-goal makes inference tests actually run end-to-end on this machine.

**Strategy**: Download the smallest viable model for each inference path, run real inference, assert meaningful output. These are slow tests — mark `@pytest.mark.slow` so normal CI can skip them, but they MUST pass (not skip) when run explicitly.

Models to use (smallest that actually work):
- **STT** (`yt2md`): `mlx-community/parakeet-tdt-0.6b-v3` (~600MB, already the default)
- **HTML→MD** (`web2md`, `html2md`): `mlx-community/ReaderLM-v2` (~869MB, 4-bit)
- **VLM/OCR** (`img2md`, `pdf2md --ocr`): `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (~2GB, smallest Qwen VL that works with mlx-vlm)

**What to build**:
1. `test/test_inference.py` — slow inference tests using `@pytest.mark.slow`. Each test:
   - Downloads the model if not cached (using `mlx_lm.load()` or `mlx_vlm.load()` — they auto-download on first use)
   - Runs real inference on the corresponding fixture file
   - Asserts output is non-empty, contains recognizable content from the fixture
   - Must PASS, not skip

2. Tests to include:
   - `test_readerlm_html_to_markdown` — run real ReaderLM-v2 on `sample.html`, assert output contains "heading" or "paragraph" content
   - `test_readerlm_web_fetch_and_convert` — fetch a real stable URL (e.g. `https://example.com`), run ReaderLM-v2, assert non-empty markdown
   - `test_vlm_image_ocr` — run Qwen2.5-VL-3B on `sample.jpg` (has "Hello World" text), assert "Hello" or "World" appears in output
   - `test_vlm_pdf_scanned_page` — render `sample.pdf` page as image, run VLM OCR, assert content recognized
   - `test_parakeet_transcription` — transcribe `test_audio/test_voice.mp3` with Parakeet, assert non-empty transcript

3. Run with: `pytest test/test_inference.py -m slow -v -s` (the `-s` shows model loading progress)

**Success**: `pytest test/test_inference.py -m slow -v` passes with 0 failures, 0 skips on this machine (36GB Apple Silicon).

## Phase 10 — Speaker Diarization for yt2md.py ⬜ NOT STARTED

Add `--diarize` flag to yt2md.py that identifies who is speaking. Uses an MLX-native pipeline (~32MB total):

### 10A — Diarization pipeline (CRITICAL — verify all models load)

Install and verify the 3 MLX diarization models:
- **Silero VAD v5** (`aufklarer/Silero-VAD-v5-MLX`, ~1.2MB) — voice activity detection
- **Pyannote Segmentation 3.0** (`mlx-community/pyannote-segmentation-3.0-mlx`, ~5.7MB) — per-frame speaker probabilities (up to 3 simultaneous)
- **WeSpeaker ResNet34-LM** (`aufklarer/WeSpeaker-ResNet34-LM-MLX`, ~25MB) — speaker embeddings

Check the actual API for each model before writing code. Reference implementation: https://github.com/ivan-digital/qwen3-asr-swift (Swift, but same pipeline).

### 10B — Build diarization module

Create `diarize.py` with these functions:
- `run_vad(audio_path) -> list[tuple[float, float]]` — speech segment boundaries via Silero
- `run_segmentation(audio_path, speech_segments) -> ndarray` — per-frame speaker activity probabilities via pyannote
- `extract_embeddings(audio_path, segments) -> ndarray` — speaker embeddings via WeSpeaker
- `cluster_speakers(embeddings) -> list[int]` — agglomerative clustering via scipy/sklearn
- `diarize(audio_path) -> list[dict]` — full pipeline: returns `[{speaker: "SPEAKER_0", start: 0.0, end: 5.2}, ...]`

### 10C — Integrate into yt2md.py

Add `--diarize` flag to the CLI:
- When enabled: run diarization pipeline AFTER transcription
- Align diarization segments to transcription segments (match by timestamp overlap)
- Output format for markdown: `**SPEAKER_0**: text here...` per speaker turn
- Output format for SRT: `[SPEAKER_0] text here...` in subtitle text
- Frontmatter: add `speakers: N` field when diarization is used
- Add diarization models to `download_models.py --diarize` flag

### 10D — Tests
- Unit tests mocking all 3 models (VAD, segmentation, embeddings)
- Integration test with real audio (test_voice.mp3 — single speaker should produce SPEAKER_0 only)
- Test alignment logic: given mock diarization + mock transcription segments, verify correct speaker assignment

**Success**: `python yt2md.py test_audio/test_voice.mp3 --diarize` produces markdown with speaker labels. Multi-speaker audio correctly identifies different speakers.

**New deps**: `scikit-learn>=1.0`, `scipy>=1.10` (for agglomerative clustering)

## Phase 8 — Polish & Keep Going ✅ DONE

- Update download_models.py for all new models
- Update requirements.txt with all new dependencies
- README.md refresh with all tools
- Performance benchmarks
- Future tools: email2md, rss2md, csv2md, repo2md
- rst2md already done in Phase 7 — consider adding org2md (Emacs Org-mode) as next

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
- All existing tests mock external deps — no live coverage of actual model inference or real file parsing
- mlx-vlm 0.1.21 has torchvision ABI mismatch bug — VLM inference untested live

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
- **Phase 7**: rst2md.py converts a real .rst file to markdown with frontmatter
- **Phase 8**: All tools documented, requirements.txt complete, README refreshed
- **Phase 9**: Real fixture tests pass without mocks; integration tests pass or skip cleanly based on model availability; `pytest test_*.py -q` still green
- After all: keep going — add email2md, rss2md, org2md, etc.

## Operating Rules

- All AI inference via MLX — no cloud APIs
- Every tool follows the same typer CLI pattern as existing tools
- Every tool produces YAML frontmatter
- All tests use unittest (stdlib), run via pytest
- Import at module top — no lazy imports
- Follow existing code style in yt2md.py and pdf2md.py
- Run all tests after every phase: `python -m pytest test_*.py -q`
