#!/usr/bin/env python3
"""
pdf2md.py - PDF to Markdown Extraction Tool

Extracts text from PDF files to markdown (default), SRT-style, or plain text using
pymupdf4llm. Produces page-delineated output with YAML frontmatter from PDF metadata.

Usage:
    python pdf2md.py [options] <input.pdf>

Examples:
    python pdf2md.py document.pdf
    python pdf2md.py slides.pdf -o ~/notes/
    python pdf2md.py report.pdf --pages 1-10
    python pdf2md.py scanned.pdf -f txt
"""

import io
import os
import re
import sys
import tempfile
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from typing_extensions import Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from md_common import build_frontmatter

# fitz (PyMuPDF) — loaded at module level so tests can patch pdf2md.fitz.
# Falls back gracefully if not installed; ImportError is raised at call time.
try:
    import fitz
    _FITZ_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore[assignment]
    _FITZ_AVAILABLE = False

# Pillow — loaded at module level so tests can patch pdf2md.Image.
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

# mlx-vlm — loaded at module level so tests can patch pdf2md.generate etc.
# Falls back gracefully if mlx-vlm is not installed; ImportError is raised at call time.
try:
    from mlx_vlm import generate, load as _mlx_load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config as _mlx_load_config
    _MLX_VLM_AVAILABLE = True
except Exception:
    generate = None  # type: ignore[assignment]
    _mlx_load = None  # type: ignore[assignment]
    apply_chat_template = None  # type: ignore[assignment]
    _mlx_load_config = None  # type: ignore[assignment]
    _MLX_VLM_AVAILABLE = False

# Minimum chars on a page before flagging as image-heavy
THIN_PAGE_THRESHOLD = 50

# Default VLM model for OCR fallback
VLM_MODEL_DEFAULT = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

VLM_EXTRACTION_PROMPT = """Extract all text from this document page as clean markdown.
- Preserve tables as markdown tables
- Convert equations to LaTeX ($...$ inline, $$...$$ block)
- Use proper heading levels (#, ##, ###)
- Preserve code blocks with language tags
- Do NOT add text not present in the original
Output ONLY the markdown."""


def parse_page_range(page_range: str, total_pages: int) -> List[int]:
    """
    Parse a page range string into a list of 0-based page indices.

    Accepts formats like "1-10", "1,3,5", "1-5,8,10-12".
    Input is 1-based (user-facing), output is 0-based (fitz).

    Args:
        page_range: Page range string
        total_pages: Total number of pages in the document

    Returns:
        Sorted list of 0-based page indices
    """
    pages = set()
    for part in page_range.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start.strip()))
            end = min(total_pages, int(end.strip()))
            pages.update(range(start - 1, end))
        else:
            p = int(part.strip())
            if 1 <= p <= total_pages:
                pages.add(p - 1)
    return sorted(pages)


def extract_pdf_metadata(doc) -> Dict:
    """
    Extract metadata from a fitz Document object.

    Args:
        doc: fitz.Document object

    Returns:
        Cleaned metadata dict suitable for frontmatter
    """
    meta = doc.metadata or {}

    # Parse PDF date format: D:YYYYMMDDHHmmSS or similar
    def parse_pdf_date(date_str: str) -> Optional[str]:
        if not date_str:
            return None
        # Strip D: prefix
        date_str = date_str.lstrip("D:")
        # Take first 8 chars for YYYYMMDD
        if len(date_str) >= 8:
            try:
                dt = datetime.strptime(date_str[:8], "%Y%m%d")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        return date_str

    result = {
        'title': meta.get('title') or None,
        'author': meta.get('author') or None,
        'subject': meta.get('subject') or None,
        'keywords': meta.get('keywords') or None,
        'creator': meta.get('creator') or None,
        'producer': meta.get('producer') or None,
        'created': parse_pdf_date(meta.get('creationDate', '')),
        'modified': parse_pdf_date(meta.get('modDate', '')),
        'format': meta.get('format') or None,
        'pages': doc.page_count,
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    # Parse keywords into a list
    if result['keywords']:
        result['keywords'] = [k.strip() for k in result['keywords'].split(',') if k.strip()]

    # Remove None/empty values
    return {k: v for k, v in result.items() if v is not None and v != '' and v != []}


def render_page_as_image(pdf_path: Path, page_index: int, dpi: int = 150):
    """
    Render a PDF page as a PIL Image using fitz (PyMuPDF).

    Args:
        pdf_path: Path to the PDF file
        page_index: 0-based page index to render
        dpi: Resolution in dots per inch (default 150)

    Returns:
        PIL.Image.Image of the rendered page

    Raises:
        ImportError: If fitz (PyMuPDF) or Pillow is not installed
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required. Install it with: pip install pymupdf")
    if Image is None:
        raise ImportError("Pillow is required. Install it with: pip install Pillow")

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes))
    finally:
        doc.close()


def load_vlm_for_pdf(model_name: str) -> Tuple:
    """
    Load VLM model and processor via mlx-vlm for PDF OCR.

    Args:
        model_name: HuggingFace model ID or local path

    Returns:
        Tuple of (model, processor, config)

    Raises:
        ImportError: If mlx-vlm is not installed
    """
    if _mlx_load is None or _mlx_load_config is None:
        raise ImportError("mlx-vlm is required for --ocr. Install it with: pip install mlx-vlm")

    logger.info("Loading VLM model for OCR: %s", model_name)
    model, processor = _mlx_load(model_name)
    config = _mlx_load_config(model_name)
    return model, processor, config


def extract_page_via_vlm(pdf_path: Path, page_index: int, model, processor, config) -> str:
    """
    Extract text from a PDF page using VLM inference on a rendered image.

    Args:
        pdf_path: Path to the PDF file
        page_index: 0-based page index
        model: Loaded mlx-vlm model
        processor: Loaded mlx-vlm processor
        config: Model config dict from load_config()

    Returns:
        Extracted markdown text from the VLM

    Raises:
        ImportError: If mlx-vlm is not installed
    """
    if apply_chat_template is None or generate is None:
        raise ImportError("mlx-vlm is required for --ocr. Install it with: pip install mlx-vlm")

    img = render_page_as_image(pdf_path, page_index)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        tmp_path = f.name

    try:
        prompt = apply_chat_template(processor, config, VLM_EXTRACTION_PROMPT, num_images=1)
        result = generate(
            model,
            processor,
            prompt,
            image=tmp_path,
            max_tokens=2048,
            verbose=False,
        )
        # mlx-vlm >= 0.4.0 returns a GenerationResult dataclass instead of a plain str/dict.
        # Handle str, dict, and dataclass returns for backward compatibility.
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get('text', '')
        return getattr(result, 'text', '')
    finally:
        os.unlink(tmp_path)


def extract_pages_hybrid(
    pdf_path: str,
    page_indices: Optional[List[int]] = None,
    ocr: bool = False,
    force_ocr: bool = False,
    vlm_model=None,
    vlm_processor=None,
    vlm_config=None,
) -> List[Dict]:
    """
    Extract markdown from PDF pages, using VLM for thin/scanned pages when OCR is enabled.

    For each page:
    - If force_ocr=True: always use VLM
    - If ocr=True and page text < THIN_PAGE_THRESHOLD chars: use VLM
    - Otherwise: use pymupdf4llm text extraction

    Args:
        pdf_path: Path to the PDF file
        page_indices: Optional list of 0-based page indices to extract
        ocr: If True, use VLM for thin pages
        force_ocr: If True, use VLM on ALL pages (implies ocr=True)
        vlm_model: Loaded mlx-vlm model (required when ocr=True)
        vlm_processor: Loaded mlx-vlm processor (required when ocr=True)
        vlm_config: Model config dict (required when ocr=True)

    Returns:
        List of dicts with 'page' (1-based), 'text', 'is_thin', and 'ocr_used' keys
    """
    # First, get text extraction results from pymupdf4llm
    base_results = extract_pages(pdf_path, page_indices)

    if not ocr and not force_ocr:
        # No OCR requested — return base results with ocr_used=False
        for r in base_results:
            r['ocr_used'] = False
        return base_results

    pdf_path_obj = Path(pdf_path)
    results = []

    for page_data in base_results:
        page_index = page_data['page'] - 1  # convert 1-based back to 0-based
        use_vlm = force_ocr or (ocr and page_data['is_thin'])

        if use_vlm:
            logger.info("Page %d: using VLM (thin=%s, force=%s)", page_data['page'], page_data['is_thin'], force_ocr)
            try:
                vlm_text = extract_page_via_vlm(
                    pdf_path_obj, page_index, vlm_model, vlm_processor, vlm_config
                )
                results.append({
                    'page': page_data['page'],
                    'text': vlm_text.strip(),
                    'is_thin': page_data['is_thin'],
                    'ocr_used': True,
                })
            except Exception as exc:
                logger.warning("Page %d: VLM failed (%s), falling back to text extraction", page_data['page'], exc)
                page_data['ocr_used'] = False
                results.append(page_data)
        else:
            logger.debug("Page %d: using text extraction", page_data['page'])
            page_data['ocr_used'] = False
            results.append(page_data)

    return results


def extract_pages(pdf_path: str, page_indices: Optional[List[int]] = None) -> List[Dict]:
    """
    Extract markdown from PDF pages using pymupdf4llm.

    Args:
        pdf_path: Path to the PDF file
        page_indices: Optional list of 0-based page indices to extract

    Returns:
        List of dicts with 'page' (1-based) and 'text' keys
    """
    try:
        import pymupdf4llm
    except ImportError:
        logger.error("pymupdf4llm is required. Install it with: pip install pymupdf4llm")
        raise

    logger.info(f"Extracting text from: {pdf_path}")

    # Pass page_indices directly to pymupdf4llm to avoid processing all pages
    chunks = pymupdf4llm.to_markdown(pdf_path, pages=page_indices, page_chunks=True)

    results = []
    for chunk in chunks:
        page_num = chunk['metadata']['page'] + 1  # 0-based to 1-based
        text = chunk['text'].strip()
        is_thin = len(text) < THIN_PAGE_THRESHOLD

        results.append({
            'page': page_num,
            'text': text,
            'is_thin': is_thin,
        })

    return results


def pages_to_markdown(pages: List[Dict], metadata: Optional[Dict] = None,
                      title: Optional[str] = None) -> str:
    """
    Convert extracted pages to markdown format.

    Args:
        pages: List of page dicts from extract_pages()
        metadata: Optional metadata dict for YAML frontmatter
        title: Optional title for the document heading

    Returns:
        Markdown formatted string
    """
    lines = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append("")

    if title:
        lines.append(f"# {title}")
        lines.append("")

    thin_pages = []
    for page in pages:
        lines.append(f"## Page {page['page']}")
        lines.append("")

        if page['is_thin']:
            thin_pages.append(page['page'])
            if page['text']:
                lines.append(page['text'])
            else:
                lines.append("*[This page appears to be image-only — no extractable text]*")
            lines.append("")
        else:
            lines.append(page['text'])
            lines.append("")

    if thin_pages:
        logger.warning(
            f"Pages with little/no text (may need OCR): {thin_pages}"
        )

    return "\n".join(lines)


def pages_to_text(pages: List[Dict]) -> str:
    """
    Convert extracted pages to plain text (no formatting).

    Args:
        pages: List of page dicts from extract_pages()

    Returns:
        Plain text string
    """
    parts = []
    for page in pages:
        if page['text']:
            parts.append(page['text'])
    return "\n\n".join(parts)


# --- CLI ---

class OutputFormat(str, Enum):
    md = "md"
    txt = "txt"


app = typer.Typer(
    help="Extract text from PDFs to markdown or plain text using pymupdf4llm.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input: Annotated[str, typer.Argument(
        help="Path to a PDF file.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path.cwd(),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter + page headings), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    pages: Annotated[Optional[str], typer.Option(
        "--pages", "-p",
        help="Page range to extract, e.g. [bold]1-10[/bold], [bold]1,3,5[/bold], [bold]1-5,8,10-12[/bold]. Defaults to all pages.",
    )] = None,
    ocr: Annotated[bool, typer.Option(
        "--ocr",
        help="Use VLM for scanned/image pages (auto-detected by thin text threshold).",
    )] = False,
    force_ocr: Annotated[bool, typer.Option(
        "--force-ocr",
        help="Force VLM on ALL pages, regardless of text content.",
    )] = False,
    vlm_model: Annotated[str, typer.Option(
        "--vlm-model",
        help="VLM model HuggingFace ID or alias for OCR fallback.",
    )] = VLM_MODEL_DEFAULT,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
):
    """
    Extract text from a PDF file to markdown (default) or plain text.

    Produces page-delineated output with YAML frontmatter from PDF metadata.
    Pages with little/no extractable text are flagged as potentially image-only.
    Use [bold]--ocr[/bold] to automatically run VLM on thin/scanned pages.
    Use [bold]--force-ocr[/bold] to run VLM on every page.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    pdf_path = os.path.abspath(input)
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        raise typer.Exit(code=1)

    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"Not a PDF file: {pdf_path}")
        raise typer.Exit(code=1)

    # Open document for metadata
    if fitz is None:
        logger.error("PyMuPDF (fitz) is required. Install it with: pip install pymupdf")
        raise typer.Exit(code=1)
    doc = fitz.open(pdf_path)
    metadata = extract_pdf_metadata(doc)
    metadata['source'] = pdf_path
    total_pages = doc.page_count
    doc.close()

    logger.info(f"PDF has {total_pages} pages")

    # Parse page range
    page_indices = None
    if pages:
        page_indices = parse_page_range(pages, total_pages)
        logger.info(f"Extracting pages: {[i+1 for i in page_indices]}")

    # Load VLM model if OCR is requested
    vlm_model_obj = None
    vlm_processor = None
    vlm_config = None
    use_ocr = ocr or force_ocr
    if use_ocr:
        try:
            vlm_model_obj, vlm_processor, vlm_config = load_vlm_for_pdf(vlm_model)
        except ImportError as exc:
            logger.error("Cannot load VLM for OCR: %s", exc)
            raise typer.Exit(code=1)
        except Exception as exc:
            logger.error("Failed to load VLM model '%s': %s", vlm_model, exc)
            raise typer.Exit(code=1)

    # Extract (hybrid if OCR requested)
    extracted = extract_pages_hybrid(
        pdf_path,
        page_indices=page_indices,
        ocr=ocr,
        force_ocr=force_ocr,
        vlm_model=vlm_model_obj,
        vlm_processor=vlm_processor,
        vlm_config=vlm_config,
    )
    logger.info(f"Extracted {len(extracted)} pages")

    if use_ocr:
        ocr_count = sum(1 for p in extracted if p.get('ocr_used'))
        text_count = len(extracted) - ocr_count
        logger.info("Pages via VLM: %d, via text extraction: %d", ocr_count, text_count)

    # Determine output filename
    output_dir_str = str(output_dir)
    os.makedirs(output_dir_str, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ext = format.value
    output_file = os.path.join(output_dir_str, f"{base_name}.{ext}")

    # Format output
    title = metadata.get('title') or base_name
    if format == OutputFormat.md:
        content = pages_to_markdown(extracted, metadata=metadata, title=title)
    else:
        content = pages_to_text(extracted)

    # Write
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Output saved to: {output_file}")


if __name__ == "__main__":
    app()
