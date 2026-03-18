#!/usr/bin/env python3
"""
md_common.py - Shared utilities for the 2md toolkit

Contains shared code used across yt2md, pdf2md, and future tools:
- build_frontmatter: YAML frontmatter builder (no PyYAML dependency)
- setup_logging: Consistent logging configuration
- OutputFormat: Base enum for md/txt output formats
- write_output: Simple file writer helper
- load_vlm: Stub for future VLM tools
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logging. Call once at CLI startup.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Base output format enum for tools that produce md or txt output."""
    md = "md"
    txt = "txt"


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def write_output(content: str, output_path: Path) -> None:
    """
    Write content to a file, creating parent directories as needed.

    Args:
        content: Text content to write.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------

def build_frontmatter(metadata: Dict) -> str:
    """
    Build YAML frontmatter string from metadata dict.

    Handles scalars, lists, and nested dicts (chapters).
    Uses manual formatting to avoid a PyYAML dependency.

    Args:
        metadata: Dict of metadata fields

    Returns:
        YAML frontmatter string including --- delimiters
    """
    def yaml_scalar(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return str(v)
        s = str(v)
        # Quote if contains special chars or looks like a number/bool
        if any(c in s for c in ':#{}[]&*?|->!%@`"\',\n') or s in ('true', 'false', 'null', 'yes', 'no'):
            escaped = s.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return s

    lines = ["---"]

    # Ordered keys for clean output
    key_order = [
        'title', 'video_id', 'url', 'channel', 'channel_url', 'uploader',
        'upload_date', 'duration', 'duration_human', 'language', 'location',
        'availability', 'live_status', 'view_count', 'like_count',
        'comment_count', 'channel_follower_count', 'thumbnail',
        'categories', 'tags', 'subtitles', 'auto_captions',
        'chapters', 'description', 'fetched_at',
    ]
    # Include any keys not in the order list at the end
    all_keys = key_order + [k for k in metadata if k not in key_order]

    for key in all_keys:
        if key not in metadata:
            continue
        val = metadata[key]

        if val is None or val == [] or val == '':
            continue

        if key == 'description':
            # Multi-line string
            escaped = str(val).replace('\\', '\\\\')
            lines.append(f'{key}: |')
            for desc_line in escaped.split('\n'):
                lines.append(f'  {desc_line}')
        elif key == 'chapters' and isinstance(val, list) and val:
            lines.append(f'{key}:')
            for ch in val:
                lines.append(f'  - time: {yaml_scalar(ch["time"])}')
                lines.append(f'    title: {yaml_scalar(ch["title"])}')
        elif isinstance(val, list):
            if all(isinstance(item, str) for item in val) and len(val) <= 10:
                # Inline list for short string lists
                items = ", ".join(yaml_scalar(item) for item in val)
                lines.append(f'{key}: [{items}]')
            else:
                lines.append(f'{key}:')
                for item in val:
                    lines.append(f'  - {yaml_scalar(item)}')
        else:
            lines.append(f'{key}: {yaml_scalar(val)}')

    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VLM stub (for future img2md.py)
# ---------------------------------------------------------------------------

def load_vlm(model: str):
    """Load a VLM model via mlx-vlm. Stub for future use."""
    try:
        import mlx_vlm  # noqa: F401
        # Will be implemented in img2md.py phase
        raise NotImplementedError("load_vlm not yet implemented")
    except ImportError:
        raise ImportError("mlx-vlm not installed. Run: pip install mlx-vlm")
