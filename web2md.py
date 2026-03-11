#!/usr/bin/env python3
"""
web2md.py — Convert web URLs to markdown via ReaderLM-v2 (mlx-lm)

Fetches a URL, extracts metadata (title, description, author, sitename),
and converts the HTML to clean markdown using ReaderLM-v2 running locally
on Apple Silicon via mlx-lm.

Usage:
    python web2md.py <url> [OPTIONS]

Examples:
    python web2md.py https://example.com/article
    python web2md.py https://example.com/page -o ~/notes/
    python web2md.py https://example.com -f txt
    python web2md.py https://example.com --model mlx-community/jinaai-ReaderLM-v2 -v
"""

import logging
import os
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
import typer
from typing_extensions import Annotated

from md_common import build_frontmatter, setup_logging, OutputFormat, write_output

# Configure logging (will be overridden by setup_logging in main)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mlx-community/ReaderLM-v2"
MAX_HTML_CHARS = 200_000   # Truncate HTML before passing to model
MAX_OUTPUT_TOKENS = 8192

# ReaderLM-v2 uses a chat template. The model is fine-tuned to convert
# HTML to clean markdown given the instruction below.
SYSTEM_PROMPT = (
    "Convert the HTML to clean markdown. "
    "Output only the converted markdown. "
    "Do not include any explanations or preamble."
)


# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------

def fetch_html(url: str) -> Tuple[str, dict]:
    """
    Fetch a URL and return (html_content, basic_metadata).

    Uses httpx if available, falls back to urllib.request.

    Args:
        url: The URL to fetch.

    Returns:
        Tuple of (html string, dict with url and fetched_at).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    fetched_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    try:
        import httpx
        logger.debug("Fetching URL with httpx: %s", url)
        with httpx.Client(headers=headers, follow_redirects=True, timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
    except ImportError:
        logger.debug("httpx not available, using urllib: %s", url)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.error("Failed to fetch URL %s: %s", url, exc)
        raise

    basic_meta = {
        "url": url,
        "fetched_at": fetched_at,
    }
    return html, basic_meta


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_meta_tag(html: str, name: str) -> Optional[str]:
    """Extract a <meta> tag value by name or property attribute."""
    patterns = [
        rf'<meta\s+name=["\'](?:og:)?{re.escape(name)}["\']\s+content=["\']([^"\']*)["\']',
        rf'<meta\s+content=["\']([^"\']*)["\'][^>]*name=["\'](?:og:)?{re.escape(name)}["\']',
        rf'<meta\s+property=["\']og:{re.escape(name)}["\']\s+content=["\']([^"\']*)["\']',
        rf'<meta\s+content=["\']([^"\']*)["\'][^>]*property=["\']og:{re.escape(name)}["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_title_tag(html: str) -> Optional[str]:
    """Extract the <title> tag content."""
    match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_metadata(html: str, url: str) -> dict:
    """
    Extract metadata from HTML.

    Uses trafilatura if available for richer extraction, falls back to
    manual <meta> tag parsing.

    Args:
        html: HTML content as a string.
        url: Source URL (used as fallback and for sitename).

    Returns:
        Dict with: title, description, author, sitename, url, fetched_at.
    """
    result: dict = {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    try:
        import trafilatura
        doc = trafilatura.extract_metadata(html, default_url=url)
        if doc is not None:
            meta_dict = doc.as_dict()
            if meta_dict.get("title"):
                result["title"] = meta_dict["title"]
            if meta_dict.get("description"):
                result["description"] = meta_dict["description"]
            if meta_dict.get("author"):
                result["author"] = meta_dict["author"]
            if meta_dict.get("sitename"):
                result["sitename"] = meta_dict["sitename"]
            logger.debug("Extracted metadata via trafilatura: %s", list(result.keys()))
            return result
    except ImportError:
        logger.debug("trafilatura not available, falling back to regex meta extraction")
    except Exception as exc:
        logger.debug("trafilatura metadata extraction failed: %s — falling back", exc)

    # Fallback: manual <meta> tag parsing
    title = _extract_meta_tag(html, "title") or _extract_title_tag(html)
    if title:
        result["title"] = title

    description = _extract_meta_tag(html, "description")
    if description:
        result["description"] = description

    author = _extract_meta_tag(html, "author")
    if author:
        result["author"] = author

    # Derive sitename from URL
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        result["sitename"] = match.group(1)

    logger.debug("Extracted metadata via regex: %s", list(result.keys()))
    return result


# ---------------------------------------------------------------------------
# ReaderLM-v2 inference
# ---------------------------------------------------------------------------

def load_reader_model(model_name: str = DEFAULT_MODEL):
    """
    Load ReaderLM-v2 model and tokenizer via mlx-lm.

    Args:
        model_name: HuggingFace model ID or local path.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ImportError: If mlx-lm is not installed.
    """
    try:
        from mlx_lm import load
    except ImportError:
        raise ImportError(
            "mlx-lm is required for web2md. Install it with: pip install mlx-lm"
        )

    logger.info("Loading model: %s", model_name)
    model, tokenizer = load(model_name)
    logger.info("Model loaded successfully")
    return model, tokenizer


def build_reader_prompt(html: str, tokenizer) -> str:
    """
    Build the prompt for ReaderLM-v2 using the tokenizer's chat template.

    ReaderLM-v2 is fine-tuned on a chat format. We use apply_chat_template
    if available, otherwise fall back to a manual format.

    Args:
        html: HTML content to convert.
        tokenizer: The loaded tokenizer.

    Returns:
        Formatted prompt string.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": html},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.debug("Built prompt via apply_chat_template (len=%d)", len(prompt))
            return prompt
        except Exception as exc:
            logger.debug("apply_chat_template failed: %s — using fallback format", exc)

    # Fallback: manual chat format
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n"
        f"<|user|>\n{html}\n<|end|>\n"
        f"<|assistant|>\n"
    )
    logger.debug("Built prompt via fallback format (len=%d)", len(prompt))
    return prompt


def html_to_markdown(html: str, model=None, tokenizer=None) -> str:
    """
    Convert HTML to markdown using ReaderLM-v2.

    Args:
        html: Raw HTML string to convert.
        model: Loaded mlx-lm model. If None, raises ValueError.
        tokenizer: Loaded tokenizer. If None, raises ValueError.

    Returns:
        Cleaned markdown string.
    """
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer must be provided to html_to_markdown")

    try:
        from mlx_lm import generate
    except ImportError:
        raise ImportError(
            "mlx-lm is required for web2md. Install it with: pip install mlx-lm"
        )

    # Truncate if too long
    if len(html) > MAX_HTML_CHARS:
        logger.warning(
            "HTML too long (%d chars), truncating to %d chars",
            len(html),
            MAX_HTML_CHARS,
        )
        html = html[:MAX_HTML_CHARS]

    prompt = build_reader_prompt(html, tokenizer)
    logger.info("Generating markdown from HTML (%d chars)...", len(html))

    markdown = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        verbose=False,
    )

    return markdown.strip()


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def page_to_markdown(content: str, metadata: dict) -> str:
    """
    Wrap content with YAML frontmatter.

    Args:
        content: Markdown body text.
        metadata: Metadata dict for frontmatter.

    Returns:
        Full markdown string with frontmatter.
    """
    parts = [build_frontmatter(metadata), "", content]
    return "\n".join(parts)


def page_to_text(content: str) -> str:
    """
    Return plain text content (no frontmatter).

    Args:
        content: Markdown/text body.

    Returns:
        Plain text string.
    """
    return content


# ---------------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------------

def url_to_filename(url: str) -> str:
    """
    Derive a safe filename stem from a URL.

    Args:
        url: Source URL.

    Returns:
        Filename-safe string (no extension).
    """
    # Strip scheme
    name = re.sub(r'^https?://', '', url)
    # Replace non-alphanumeric (except hyphen/underscore) with underscore
    name = re.sub(r'[^\w\-]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    # Truncate to 80 chars
    return name[:80]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert a web URL to markdown using ReaderLM-v2 (local MLX inference).",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    url: Annotated[str, typer.Argument(
        help="URL to fetch and convert to markdown.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path.cwd(),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="ReaderLM model (HuggingFace ID or local path).",
    )] = DEFAULT_MODEL,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert a web URL to markdown using ReaderLM-v2 (local MLX inference).

    Fetches the URL, extracts metadata (title, author, description, sitename),
    and converts the HTML to clean markdown using ReaderLM-v2 running locally
    on Apple Silicon via mlx-lm.
    """
    setup_logging(verbose)

    # Fetch HTML
    logger.info("Fetching URL: %s", url)
    html, basic_meta = fetch_html(url)
    logger.info("Fetched %d chars of HTML", len(html))

    # Extract metadata
    metadata = extract_metadata(html, url)
    logger.info("Extracted metadata: title=%s", metadata.get("title", "<none>"))

    # Load model
    reader_model, tokenizer = load_reader_model(model)

    # Convert HTML to markdown
    markdown_content = html_to_markdown(html, model=reader_model, tokenizer=tokenizer)
    logger.info("Generated %d chars of markdown", len(markdown_content))

    # Determine output filename
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    stem = url_to_filename(url)
    ext = format.value
    output_file = output_dir_path / f"{stem}.{ext}"

    # Format and write
    if format == OutputFormat.md:
        content = page_to_markdown(markdown_content, metadata)
    else:
        content = page_to_text(markdown_content)

    write_output(content, output_file)
    logger.info("Output saved to: %s", output_file)
    typer.echo(str(output_file))


if __name__ == "__main__":
    app()
