"""test_integration.py — Live integration tests requiring real model inference.

Tests are auto-skipped if the required model is not cached locally.

Run all integration tests:
    pytest test_integration.py -m integration -v

Run only the no-model integration tests (doc/rst — always run):
    pytest test_integration.py -m "integration and not requires_model" -v

Run everything including unit tests:
    pytest test_*.py test_integration.py
"""

import pytest
import sys
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------

def model_is_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is cached locally.

    Model cache dirs are named 'models--<org>--<name>' under HF_HOME/hub/.
    """
    import os
    cache_dir = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    model_dir = "models--" + model_id.replace("/", "--")
    return (cache_dir / model_dir).exists()


PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
READERLM_MODEL = "mlx-community/ReaderLM-v2"
QWEN_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


# ---------------------------------------------------------------------------
# Doc tools — no model needed (always run)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_doc2md_full_pipeline():
    """doc2md end-to-end: real DOCX → markdown with frontmatter."""
    pytest.importorskip("docx")
    if not (FIXTURES / "sample.docx").exists():
        pytest.skip("sample.docx fixture missing — run test/create_fixtures.py")

    from any2md.doc import convert_document, extract_doc_metadata
    from any2md.common import build_frontmatter

    path = FIXTURES / "sample.docx"
    content = convert_document(path)
    meta = extract_doc_metadata(path, "docx")
    output = build_frontmatter(meta) + "\n" + content

    assert output.startswith("---"), "Output must start with YAML frontmatter"
    assert "Test Document" in output, "Title must appear in output"
    assert len(content) > 20, "Content must be non-trivial"


@pytest.mark.integration
def test_doc2md_pptx_pipeline():
    """doc2md end-to-end: real PPTX → markdown with frontmatter."""
    pytest.importorskip("pptx")
    if not (FIXTURES / "sample.pptx").exists():
        pytest.skip("sample.pptx fixture missing — run test/create_fixtures.py")

    from any2md.doc import convert_document, extract_doc_metadata
    from any2md.common import build_frontmatter

    path = FIXTURES / "sample.pptx"
    content = convert_document(path)
    meta = extract_doc_metadata(path, "pptx")

    assert meta["slides"] == 2, "PPTX must report 2 slides"
    assert isinstance(content, str)


@pytest.mark.integration
def test_doc2md_xlsx_pipeline():
    """doc2md end-to-end: real XLSX → markdown."""
    pytest.importorskip("openpyxl")
    if not (FIXTURES / "sample.xlsx").exists():
        pytest.skip("sample.xlsx fixture missing — run test/create_fixtures.py")

    from any2md.doc import convert_document, extract_doc_metadata

    path = FIXTURES / "sample.xlsx"
    content = convert_document(path)
    meta = extract_doc_metadata(path, "xlsx")

    assert meta.get("title") == "Test Workbook"
    assert isinstance(content, str)


@pytest.mark.integration
def test_rst2md_full_pipeline():
    """rst2md end-to-end: real RST → markdown with frontmatter."""
    if not (FIXTURES / "sample.rst").exists():
        pytest.skip("sample.rst fixture missing — run test/create_fixtures.py")

    from any2md.rst import process_rst_file

    path = FIXTURES / "sample.rst"
    with tempfile.TemporaryDirectory() as tmpdir:
        out = process_rst_file(path, Path(tmpdir), "md")
        content = out.read_text()

    assert "---" in content, "Output must contain YAML frontmatter delimiters"
    assert "Test Author" in content, "Author must appear in frontmatter or body"
    assert len(content) > 50, "Output must be non-trivial"


@pytest.mark.integration
def test_rst2md_txt_mode():
    """rst2md txt mode: no frontmatter in plain-text output."""
    if not (FIXTURES / "sample.rst").exists():
        pytest.skip("sample.rst fixture missing — run test/create_fixtures.py")

    from any2md.rst import process_rst_file

    path = FIXTURES / "sample.rst"
    with tempfile.TemporaryDirectory() as tmpdir:
        out = process_rst_file(path, Path(tmpdir), "txt")
        content = out.read_text()

    assert "fetched_at:" not in content, "Plain text mode must not have YAML frontmatter"
    assert "Introduction" in content or "hello world" in content.lower()


@pytest.mark.integration
def test_pdf2md_text_path():
    """pdf2md end-to-end (text path): born-digital PDF → markdown."""
    if not (FIXTURES / "sample.pdf").exists():
        pytest.skip("sample.pdf fixture missing — run test/create_fixtures.py")

    from any2md.pdf import extract_pages, pages_to_markdown, extract_pdf_metadata
    import fitz

    path = FIXTURES / "sample.pdf"
    pages = extract_pages(str(path))
    doc = fitz.open(str(path))
    meta = extract_pdf_metadata(doc)
    meta["source"] = str(path)
    doc.close()

    output = pages_to_markdown(pages, metadata=meta)

    assert output.startswith("---"), "Output must start with YAML frontmatter"
    assert "Test PDF Document" in output
    assert len(output) > 50


# ---------------------------------------------------------------------------
# Web/HTML — requires ReaderLM-v2
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_html2md_full_pipeline():
    """html2md end-to-end: real HTML file → markdown with frontmatter via ReaderLM-v2."""
    if not model_is_cached(READERLM_MODEL):
        pytest.skip(f"ReaderLM-v2 not cached ({READERLM_MODEL}) — download it first")
    if not (FIXTURES / "sample.html").exists():
        pytest.skip("sample.html fixture missing — run test/create_fixtures.py")

    from any2md.html import process_html_file
    from any2md.web import load_reader_model
    from any2md.common import OutputFormat

    path = FIXTURES / "sample.html"
    model, tokenizer = load_reader_model(READERLM_MODEL)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = process_html_file(path, model, tokenizer, Path(tmpdir), OutputFormat.md)
        content = out.read_text()

    assert content.startswith("---"), "Output must start with YAML frontmatter"
    assert "Test Page" in content, "Title must appear in output"
    assert len(content) > 100, "Output must be non-trivial"


# ---------------------------------------------------------------------------
# Image/PDF OCR — requires Qwen VLM
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_img2md_full_pipeline():
    """img2md end-to-end: real JPEG → markdown via Qwen VLM."""
    if not model_is_cached(QWEN_MODEL):
        pytest.skip(f"Qwen VLM not cached ({QWEN_MODEL}) — download it first")
    if not (FIXTURES / "sample.jpg").exists():
        pytest.skip("sample.jpg fixture missing — run test/create_fixtures.py")

    from any2md.img import load_vlm_model, image_to_markdown_text, image_to_markdown, get_image_metadata
    from mlx_vlm.utils import load_config

    path = FIXTURES / "sample.jpg"
    model, processor, config = load_vlm_model(QWEN_MODEL)

    with tempfile.TemporaryDirectory() as tmpdir:
        text = image_to_markdown_text(path, model, processor, config, QWEN_MODEL)
        meta = get_image_metadata(path, QWEN_MODEL)
        output = image_to_markdown(text, meta)

    assert output.startswith("---"), "Output must start with YAML frontmatter"
    assert len(text) > 5, "VLM must produce non-empty output"


@pytest.mark.integration
def test_pdf2md_ocr_path():
    """pdf2md OCR path: render a PDF page as image and run VLM inference."""
    if not model_is_cached(QWEN_MODEL):
        pytest.skip(f"Qwen VLM not cached ({QWEN_MODEL}) — download it first")
    if not (FIXTURES / "sample.pdf").exists():
        pytest.skip("sample.pdf fixture missing — run test/create_fixtures.py")

    from any2md.pdf import load_vlm_for_pdf, extract_page_via_vlm
    from pathlib import Path

    path = FIXTURES / "sample.pdf"
    model, processor, config = load_vlm_for_pdf(QWEN_MODEL)
    text = extract_page_via_vlm(path, 0, model, processor, config)

    assert isinstance(text, str), "OCR output must be a string"
    assert len(text) > 5, "OCR output must be non-empty"


# ---------------------------------------------------------------------------
# Audio — requires Parakeet STT model
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_yt2md_transcription():
    """yt2md end-to-end: real audio file → markdown via Parakeet STT.

    Uses the high-level transcribe() function which handles model loading,
    audio conversion, and output formatting in a single call.
    """
    if not model_is_cached(PARAKEET_MODEL):
        pytest.skip(f"Parakeet model not cached ({PARAKEET_MODEL}) — download it first")

    audio = Path(__file__).parent / "audio" / "test_voice.mp3"
    if not audio.exists():
        pytest.skip(f"test_voice.mp3 not found at {audio}")

    from any2md.yt import transcribe, convert_audio_for_whisper, resolve_model

    model_id = resolve_model("parakeet-v3")

    with tempfile.TemporaryDirectory() as tmpdir:
        # convert_audio_for_whisper returns a string path; transcribe also takes/returns strings
        wav = convert_audio_for_whisper(str(audio), tmpdir)
        # transcribe() writes the output file and returns its path
        output_path = transcribe(wav, model_id, output_dir=tmpdir)
        output = Path(output_path).read_text(encoding="utf-8")

    assert isinstance(output, str), "Transcription output must be a string"
    assert len(output) > 20, "Transcription must produce non-trivial output"
