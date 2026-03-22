# WORKLOG

## 2026-03-22 06:45: CLI Standards Upgrade Complete + Cant-Stop-Wont-Stop Initiated

### What was done
- Completed 8 CLI standards tickets (ANY2-1 through ANY2-8) via parallel plow agents
- `--version` / `-V` flag added
- `--json` / `-j` output mode for all 16 converters with `--fields` filtering
- Structured JSON error output to stderr in `--json` mode
- stdout/stderr separation standardized across all converters
- Help text footer with bug report URL
- `NO_COLOR` environment variable support
- `--quiet` / `-q` flag to suppress logs
- Shell completions enabled
- llms.txt and Makefile created
- Brew formula published at `roboalchemist/tap/any2md`
- trckr project created (key: ANY2)
- Claude skill created at `~/.claude/skills/any2md/`

### Key decisions
- Used `uv` in brew formula instead of pip (resolved missing `annotated-doc` transitive dep)
- Base brew install only includes typer core; AI deps are optional add-ons
- JSON output goes to stdout, all logs/status to stderr — clean agent piping

### Current state
- 244 tests passing, 1 pre-existing failure (test_doc.py markitdown issue)
- 9,454 lines of source across 18 modules
- GOAL.md outdated — needs refresh for post-CLI-standards era
- ANY2-9 open: brew formula missing optional deps guidance
- GitHub push may need retry (SSH was flaky earlier)
