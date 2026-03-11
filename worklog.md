# 2md Worklog (cant-stop-wont-stop session)

Newest entries at top. Old lightning-whisper-mlx worklog archived to worklog-archive.md.

---

## 2026-03-10 Bootstrap: cant-stop-wont-stop session started

### What was done
- Assessed current project state: 25 tests passing (15 yt2md + 10 pdf2md), both tools working
- Created GOAL.md with 7 phases and full structure
- Created fresh WORKLOG.md (old worklog from lightning-whisper-mlx era archived)
- Set terminal title to `2md: cant-stop-wont-stop`

### Current state
- Phase 1 (md_common.py extraction) is next — critical path, everything depends on it
- 25 existing tests all pass, no regressions

### Plan
1. Phase 1 worker: extract md_common.py, update yt2md/pdf2md, rewrite download_models.py
2. After Phase 1: Phase 2 (web2md) + Phase 3 (doc2md) in parallel
3. Phase 5 (html2md) after Phase 2 (shares ReaderLM engine)
4. Phase 4 (img2md) + Phase 6 (enhanced pdf2md) after VLM verification
5. Phase 7 (polish) when all converters done

## 2026-03-10 Phase 1 DONE: md_common.py extracted

### What was done
- Created md_common.py: build_frontmatter(), setup_logging(), OutputFormat enum, write_output(), load_vlm() stub
- yt2md.py: removed 75-line local build_frontmatter, imports from md_common
- pdf2md.py: changed `from yt2md import build_frontmatter` → `from md_common import build_frontmatter`
- download_models.py: rewrote with typer, added --stt/--vlm/--reader/--docling/--all flags

### Results
- 25/25 tests pass
- `from md_common import build_frontmatter` works
- `python download_models.py --help` shows typer help

### Next
- Phase 2 (web2md.py) + Phase 3 (doc2md.py) dispatched in parallel

## 2026-03-10 Phase 3 DONE: doc2md.py with markitdown

### What was done
- doc2md.py (230 lines): DOCX/PPTX/XLSX/EPUB → markdown via markitdown 0.1.0
- Metadata extraction: python-docx (DOCX), python-pptx (PPTX), openpyxl (XLSX), fallback for epub/odt/rtf
- test_doc2md.py: 33 tests across 9 classes
- requirements.txt: added markitdown[all]

### Results
- 58/58 total tests pass (25 existing + 33 new)

### Next
- Phase 2 (web2md) still running
- Will dispatch Phase 4 (img2md) + Phase 5 (html2md) + Phase 7 (rst2md) after Phase 2 lands

## 2026-03-10 Phase 2 DONE: web2md.py with ReaderLM-v2

### What was done
- web2md.py: URL → ReaderLM-v2 (mlx-lm) → markdown with frontmatter
- fetch_html() with httpx + urllib fallback
- extract_metadata() with trafilatura + regex fallback
- build_reader_prompt() via tokenizer.apply_chat_template
- 24 unit tests, all mocked (no model download needed)
- requirements.txt: added mlx-lm>=0.18.0, httpx>=0.24.0, trafilatura>=1.6.0

### Results
- 82/82 tests pass (25+33+24)

### Next
- Phase 4 (img2md) + Phase 5 (html2md) + Phase 7 (rst2md) running in parallel
- Phase 6 (enhanced pdf2md VLM) will follow Phase 4

## 2026-03-10 Phases 4+5+7 DONE: img2md, html2md, rst2md

### Phase 4 (img2md.py)
- 46 unit tests, mlx-vlm API verified: load() returns (model, processor), generate() returns plain str
- mlx-vlm 0.1.21 has torchvision ABI mismatch bug — all inference fully mocked in tests
- Symbols imported at module level with try/except fallback to None for testability

### Phase 5 (html2md.py)
- 28 tests; imports load_reader_model/html_to_markdown directly from web2md — zero duplication
- Batch mode: directory glob for *.html/*.htm, loads model once, processes sequentially
- extract_meta_tags() handles both attribute orderings for meta tags

### Phase 7 (rst2md.py)
- 37 tests; chose pypandoc (already installed) + pandoc binary (downloaded via pypandoc.download_pandoc())
- Fallback chain: pypandoc → docutils HTML→strip → RuntimeError
- Parses RST docinfo fields (:Author:, :Date:, :Version:) for frontmatter

### Results
- 196/196 tests pass
- All phases 1-5, 7 complete
- Phase 6 (pdf2md VLM fallback) worker running
- Phase 8 (polish) queued after Phase 6
.
