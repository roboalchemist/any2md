#!/usr/bin/env python3
"""
repo.py - Repository → Markdown via repomix

Converts a git repository to a single markdown (or JSON) file by wrapping
the repomix CLI tool. Adds YAML frontmatter in markdown mode.

repomix must be installed separately:
    npm install -g repomix

Usage:
    any2md repo ./path/to/repo
    any2md repo ./path/to/repo --json
    any2md repo ./path/to/repo --compress --remove-comments
    any2md repo ./path/to/repo -o ~/notes/
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from any2md.common import (
    build_frontmatter,
    is_json_mode,
    setup_logging,
    write_json_error,
    write_output,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Pack a git repository into a single markdown file via repomix.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _check_repomix() -> None:
    """Verify repomix is installed; exit with instructions if not."""
    if not shutil.which("repomix"):
        logger.error("repomix is required. Install with: npm install -g repomix")
        if is_json_mode():
            write_json_error(
                "MISSING_DEPENDENCY",
                "repomix is required. Install with: npm install -g repomix",
            )
        raise typer.Exit(1)


def _build_repomix_cmd(
    input_path: Path,
    style: str,
    compress: bool,
    remove_comments: bool,
) -> list:
    """Build the repomix CLI command list."""
    cmd = ["repomix", "--style", style, "--stdout", str(input_path)]
    if compress:
        cmd.insert(-1, "--compress")
    if remove_comments:
        cmd.insert(-1, "--remove-comments")
    return cmd


def _run_repomix(cmd: list) -> subprocess.CompletedProcess:
    """Run repomix and return the CompletedProcess result."""
    return subprocess.run(cmd, capture_output=True, text=True)


def _extract_metadata(input_path: Path, compress: bool, remove_comments: bool) -> dict:
    """
    Run repomix with --style json to extract summary metadata.

    Falls back to a minimal metadata dict if the JSON run fails or the
    output cannot be parsed.
    """
    meta_cmd = _build_repomix_cmd(input_path, "json", compress, remove_comments)
    meta_result = _run_repomix(meta_cmd)

    metadata: dict = {
        "source": str(input_path),
        "repo_name": input_path.name,
        "converter": "repo",
    }

    if meta_result.returncode == 0:
        try:
            meta = json.loads(meta_result.stdout)
            summary = meta.get("fileSummary", {})
            if "totalFiles" in summary:
                metadata["total_files"] = summary["totalFiles"]
            if "totalTokens" in summary:
                metadata["total_tokens"] = summary["totalTokens"]
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.debug("Could not parse repomix JSON metadata; using minimal metadata")

    return metadata


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Path to git repository directory."),
    output_dir: Path = typer.Option(
        ".", "-o", "--output-dir", help="Directory to save output files."
    ),
    format: str = typer.Option(
        "md", "-f", "--format", help="Output format: md or txt."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output repomix JSON to stdout."
    ),
    fields: Optional[str] = typer.Option(
        None, "--fields", help="Not used for repo (repomix JSON is passthrough)."
    ),
    compress: bool = typer.Option(
        False, "--compress", help="Extract essential code structure only (Tree-sitter)."
    ),
    remove_comments: bool = typer.Option(
        False, "--remove-comments", help="Strip code comments."
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose (DEBUG) logging."
    ),
) -> None:
    """Pack a git repository into a single markdown file via repomix."""
    setup_logging(verbose)
    _check_repomix()

    input_path = Path(input_path).resolve()
    if not input_path.is_dir():
        logger.error(f"Not a directory: {input_path}")
        if json_output or is_json_mode():
            write_json_error("INVALID_INPUT", f"Not a directory: {input_path}")
        raise typer.Exit(1)

    # JSON mode: repomix JSON goes straight to stdout (no wrapping)
    if json_output or is_json_mode():
        cmd = _build_repomix_cmd(input_path, "json", compress, remove_comments)
        result = _run_repomix(cmd)
        if result.returncode != 0:
            logger.error(f"repomix failed: {result.stderr}")
            if is_json_mode():
                write_json_error(
                    "REPOMIX_ERROR", f"repomix failed: {result.stderr.strip()}"
                )
            raise typer.Exit(1)
        sys.stdout.write(result.stdout)
        return

    # Markdown mode: get repomix markdown, add frontmatter, write to file
    cmd = _build_repomix_cmd(input_path, "markdown", compress, remove_comments)
    result = _run_repomix(cmd)
    if result.returncode != 0:
        logger.error(f"repomix failed: {result.stderr}")
        raise typer.Exit(1)

    content = result.stdout

    metadata = _extract_metadata(input_path, compress, remove_comments)
    frontmatter = build_frontmatter(metadata)
    full_content = frontmatter + "\n" + content

    output_path = Path(output_dir) / f"{input_path.name}.md"
    write_output(full_content, output_path)
    typer.echo(f"Written: {output_path}", err=True)
