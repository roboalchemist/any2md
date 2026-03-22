# Goal: Best-in-class anything-to-markdown CLI for Apple Silicon

any2md converts any file format to clean markdown with YAML frontmatter. 18 converters, 8 zero-dependency. All AI inference local via MLX. Published on Homebrew.

## Phase 1 — Core Converters ✅ DONE

18 converters built and tested: yt, pdf, img, web, html, doc, rst, csv, data, db, sub, nb, eml, org, tex, man.

## Phase 2 — CLI Standards ✅ DONE

--version, --json, --quiet, NO_COLOR, structured errors, stdout/stderr separation, shell completions, llms.txt, Makefile, Homebrew formula, Claude skill, trckr project.

## Phase 3 — Quality & Polish 🔄 IN PROGRESS

### 3A — Fix pre-existing test failures
### 3B — Brew formula dependency UX (ANY2-9)
### 3C — Update CLAUDE.md with new line counts and CLI flags
### 3D — Push all work to GitHub
### 3E — Bump brew formula to v0.2.0

## Phase 4 — Test Coverage ⬜ NOT STARTED

### 4A — 90%+ unit test coverage
### 4B — --json output tests for every converter
### 4C — Fix all skipped/xfail tests

## Phase 5 — New Converters ⬜ NOT STARTED

### 5A — code2md (source code with syntax highlighting)
### 5B — rss2md (RSS/Atom feeds)
### 5C — repo2md (git repository overview)

## Phase N — Keep Going

Performance benchmarks, streaming output, batch progress bars, plugin system, MCP server.

## Success Criteria

- **Phase 3**: 0 test failures, brew v0.2.0 published, CLAUDE.md current
- **Phase 4**: 90%+ coverage, all converters have --json tests
- **Phase 5**: 3 new converters with tests
