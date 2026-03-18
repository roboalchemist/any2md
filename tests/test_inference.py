"""test/test_inference.py — Real model inference tests.

These tests download and run actual ML models. They are slow (minutes each)
and require sufficient memory (16GB+ recommended, 36GB for all tests).

Run with:
    pytest test/test_inference.py -m slow -v -s

These tests MUST PASS (not skip) on the development machine.

## mlx-vlm torchvision ABI fix
The mlx-vlm package requires compatible torch + torchvision versions.
If you see "operator torchvision::nms does not exist" on import, run:
    pip install --upgrade torch torchvision
This upgrades torchvision to a version compatible with the installed torch.
Verified working with torch==2.10.0 (or later) and mlx-vlm==0.4.0.

## Models used
- STT:      mlx-community/parakeet-tdt-0.6b-v3   (~600MB)
- HTML→MD:  mlx-community/ReaderLM-v2             (~869MB 4-bit)
- VLM/OCR:  mlx-community/Qwen2.5-VL-3B-Instruct-4bit  (~2GB 4-bit)

All models auto-download from HuggingFace Hub on first use.
"""

import pytest
import sys
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
TEST_AUDIO = Path(__file__).parent / "audio" / "test_voice.mp3"
# 44.5s 2-speaker clip from VoxConverse diarization dataset (dev split, shard 0, row 4)
# Source: https://huggingface.co/datasets/diarizers-community/voxconverse
TWO_SPEAKERS_AUDIO = Path(__file__).parent / "audio" / "two_speakers.wav"
# 90s 2-speaker clip from VoxConverse diarization dataset (test split, shard 0, row 3)
# Source: https://huggingface.co/datasets/diarizers-community/voxconverse
YT_INTERVIEW_AUDIO = Path(__file__).parent / "audio" / "yt_interview.wav"

READERLM_MODEL = "mlx-community/jinaai-ReaderLM-v2"
# Use 3B as the smallest viable Qwen VL model with mlx-community support.
# If this model isn't available, fall back to 7B (see comment in VLM tests below).
QWEN_VL_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
SORTFORMER_MODEL = "mlx-community/diar_sortformer_4spk-v1-fp32"


# ---------------------------------------------------------------------------
# ReaderLM-v2 — HTML → Markdown
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_readerlm_html_to_markdown():
    """ReaderLM-v2 converts real HTML to markdown."""
    from any2md.web import load_reader_model, html_to_markdown

    html = (FIXTURES / "sample.html").read_text()
    model, tokenizer = load_reader_model(READERLM_MODEL)
    result = html_to_markdown(html, model, tokenizer)

    assert isinstance(result, str)
    assert len(result) > 20, f"Output too short: {repr(result)}"
    # sample.html has "Main Heading" and "paragraph" content
    lower = result.lower()
    assert any(word in lower for word in ["heading", "paragraph", "main", "section", "item"]), \
        f"Expected content not found in output: {repr(result[:200])}"


@pytest.mark.slow
def test_readerlm_full_html2md_pipeline():
    """html2md.py full pipeline: real HTML file → markdown file with frontmatter."""
    from any2md.html import process_html_file
    from any2md.web import load_reader_model
    from any2md.common import OutputFormat  # OutputFormat.md is the enum value

    path = FIXTURES / "sample.html"
    model, tokenizer = load_reader_model(READERLM_MODEL)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = process_html_file(path, model, tokenizer, Path(tmpdir), OutputFormat.md)
        content = out.read_text()

    assert content.startswith("---"), "Output must start with YAML frontmatter"
    assert "Test Page" in content, "Title from <title> tag must appear in frontmatter"
    assert len(content) > 100


@pytest.mark.slow
def test_readerlm_web_fetch_and_convert():
    """web2md.py fetches a real URL and converts via ReaderLM-v2.

    Uses urllib directly (not httpx) to avoid SSL cert issues with pyenv Python
    that doesn't have macOS root certs in its ssl module's default path.
    """
    import urllib.request
    from any2md.web import load_reader_model, html_to_markdown

    url = "https://example.com"
    # Fetch via stdlib urllib (always has access to system certs on macOS)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; test)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    assert len(html) > 100, "Fetched HTML is too short"

    model, tokenizer = load_reader_model(READERLM_MODEL)
    result = html_to_markdown(html, model, tokenizer)

    assert isinstance(result, str)
    assert len(result) > 20
    # example.com has "Example Domain" in content
    assert any(word in result.lower() for word in ["example", "domain", "illustrative"]), \
        f"example.com content not found: {repr(result[:300])}"


# ---------------------------------------------------------------------------
# Qwen2.5-VL — Image OCR
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vlm_image_ocr():
    """img2md.py: Qwen2.5-VL reads text from a real image.

    Uses QWEN_VL_MODEL (3B-Instruct-4bit). If that model ID doesn't exist on
    HuggingFace, the load() call will fail with a clear error pointing to the
    model name. In that case, update QWEN_VL_MODEL to 7B-Instruct-4bit.
    sample.jpg has "Hello World" drawn on it — the VLM should recognise it.
    """
    from any2md.img import load_vlm_model, image_to_markdown_text, get_image_metadata

    path = FIXTURES / "sample.jpg"

    try:
        model, processor, config = load_vlm_model(QWEN_VL_MODEL)
    except Exception as exc:
        pytest.fail(f"Failed to load VLM model {QWEN_VL_MODEL}: {exc}")

    text = image_to_markdown_text(path, model, processor, config, QWEN_VL_MODEL)

    assert isinstance(text, str)
    assert len(text) > 0, "VLM returned empty output"
    # The image has "Hello World" text — VLM should recognise it
    assert any(word in text.lower() for word in ["hello", "world"]), \
        f"'Hello World' not found in VLM output: {repr(text[:300])}"


# ---------------------------------------------------------------------------
# Qwen2.5-VL — PDF page OCR (via pdf2md VLM path)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vlm_pdf_scanned_page_ocr():
    """pdf2md.py: VLM extracts text from a PDF page rendered as image.

    load_vlm_for_pdf() returns (model, processor, config) — a 3-tuple.
    extract_page_via_vlm() renders page 0 as a PNG then runs VLM inference.
    sample.pdf contains "Test PDF Document".
    """
    from any2md.pdf import render_page_as_image, extract_page_via_vlm, load_vlm_for_pdf

    path = FIXTURES / "sample.pdf"

    try:
        model, processor, config = load_vlm_for_pdf(QWEN_VL_MODEL)
    except Exception as exc:
        pytest.fail(f"Failed to load VLM model for PDF OCR: {exc}")

    result = extract_page_via_vlm(path, 0, model, processor, config)

    assert isinstance(result, str)
    assert len(result) > 10, f"VLM OCR output too short: {repr(result)}"
    # sample.pdf contains "Test PDF Document"
    assert any(word in result.lower() for word in ["test", "pdf", "document", "paragraph"]), \
        f"Expected content not found: {repr(result[:300])}"


# ---------------------------------------------------------------------------
# Parakeet STT — Audio transcription
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_parakeet_transcription():
    """yt2md.py: Parakeet transcribes real audio to text.

    The high-level transcribe() function handles model loading, audio
    conversion, and output formatting in a single call. It returns a file
    path (str) to the written markdown file — we read that to inspect the
    transcript content.

    test_voice.mp3 says "this is a direct test of the voice to text script"
    (or similar) — we check for common words from that phrase.
    """
    assert TEST_AUDIO.exists(), f"Test audio not found: {TEST_AUDIO}"

    from any2md.yt import transcribe, convert_audio_for_whisper, resolve_model

    model_id = resolve_model("parakeet-v3")  # mlx-community/parakeet-tdt-0.6b-v3

    with tempfile.TemporaryDirectory() as tmpdir:
        # convert_audio_for_whisper: str → str (ffmpeg 16kHz mono WAV)
        wav_path = convert_audio_for_whisper(str(TEST_AUDIO), tmpdir)
        # transcribe: returns path to written output file (str)
        output_path = transcribe(wav_path, model_id, output_dir=tmpdir)
        full_text = Path(output_path).read_text(encoding="utf-8")

    assert isinstance(full_text, str)
    assert len(full_text) > 20, f"Transcript too short: {repr(full_text)}"
    # test_voice.mp3 says something about "test" or "script"
    lower = full_text.lower()
    assert any(word in lower for word in ["test", "this", "voice", "script", "direct"]), \
        f"Expected speech content not in transcript: {repr(full_text[:300])}"


# ---------------------------------------------------------------------------
# Sortformer Diarization
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_diarize_single_speaker_audio():
    """Sortformer on single-speaker audio returns 1 speaker."""
    assert TEST_AUDIO.exists(), f"Test audio not found: {TEST_AUDIO}"

    from any2md.yt import load_diarization_model, diarize, convert_audio_for_whisper

    model = load_diarization_model(SORTFORMER_MODEL)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = convert_audio_for_whisper(str(TEST_AUDIO), tmpdir)
        output = diarize(wav_path, model)

    assert output.num_speakers <= 1 or len({s.speaker for s in output.segments}) <= 1, \
        f"Expected <=1 speaker, got {output.num_speakers}"
    for seg in output.segments:
        assert seg.speaker == 0


@pytest.mark.slow
def test_diarize_multi_speaker_audio():
    """Sortformer on two-speaker synthetic audio detects >= 2 speakers."""
    if not TWO_SPEAKERS_AUDIO.exists():
        pytest.skip(f"Two-speaker fixture not found: {TWO_SPEAKERS_AUDIO}")

    from any2md.yt import load_diarization_model, diarize

    model = load_diarization_model(SORTFORMER_MODEL)
    output = diarize(str(TWO_SPEAKERS_AUDIO), model)

    speaker_ids = {s.speaker for s in output.segments}
    assert len(speaker_ids) >= 2, f"Expected >= 2 speakers, got {speaker_ids}"
    assert len(output.segments) >= 2, "Expected multiple segments"


@pytest.mark.slow
def test_diarize_end_to_end_single():
    """Full yt2md pipeline with --diarize on single-speaker audio."""
    assert TEST_AUDIO.exists()

    from any2md.yt import transcribe, convert_audio_for_whisper

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = convert_audio_for_whisper(str(TEST_AUDIO), tmpdir)
        metadata = {"title": "Test Voice"}
        output_path = transcribe(
            wav_path, PARAKEET_MODEL, output_dir=tmpdir,
            metadata=metadata, diarize_model_name=SORTFORMER_MODEL,
        )
        content = Path(output_path).read_text()

    assert content.startswith("---"), "Output must start with YAML frontmatter"
    assert "SPEAKER_0" in content
    assert "speakers:" in content


@pytest.mark.slow
def test_diarize_end_to_end_multi():
    """Full yt2md pipeline with --diarize on multi-speaker audio."""
    if not TWO_SPEAKERS_AUDIO.exists():
        pytest.skip(f"Two-speaker fixture not found: {TWO_SPEAKERS_AUDIO}")

    from any2md.yt import transcribe

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = {"title": "Two Speakers"}
        output_path = transcribe(
            str(TWO_SPEAKERS_AUDIO), PARAKEET_MODEL, output_dir=tmpdir,
            metadata=metadata, diarize_model_name=SORTFORMER_MODEL,
        )
        content = Path(output_path).read_text()

    assert "SPEAKER_0" in content
    assert "SPEAKER_1" in content or content.count("SPEAKER_") >= 2
    assert "speakers:" in content


@pytest.mark.slow
def test_diarize_real_youtube_interview():
    """Sortformer on a real YouTube interview clip detects multiple speakers."""
    if not YT_INTERVIEW_AUDIO.exists():
        pytest.skip(f"YouTube interview fixture not found: {YT_INTERVIEW_AUDIO}")

    from any2md.yt import load_diarization_model, diarize

    model = load_diarization_model(SORTFORMER_MODEL)
    output = diarize(str(YT_INTERVIEW_AUDIO), model)

    speaker_ids = {s.speaker for s in output.segments}
    assert len(speaker_ids) >= 2, f"Expected >= 2 speakers in interview, got {speaker_ids}"


@pytest.mark.slow
def test_diarize_end_to_end_youtube():
    """Full yt2md pipeline with --diarize on real YouTube interview."""
    if not YT_INTERVIEW_AUDIO.exists():
        pytest.skip(f"YouTube interview fixture not found: {YT_INTERVIEW_AUDIO}")

    from any2md.yt import transcribe

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = {"title": "YouTube Interview"}
        output_path = transcribe(
            str(YT_INTERVIEW_AUDIO), PARAKEET_MODEL, output_dir=tmpdir,
            metadata=metadata, diarize_model_name=SORTFORMER_MODEL,
        )
        content = Path(output_path).read_text()

    # Should have multiple speakers
    speaker_labels = [line for line in content.split("\n") if "SPEAKER_" in line]
    assert len(speaker_labels) >= 2, f"Expected multiple speaker labels, got {len(speaker_labels)}"
    # Should have real transcribed content
    text_lines = [line for line in content.split("\n") if line.strip() and not line.startswith("---") and not line.startswith("**") and not line.startswith("#") and ":" not in line[:20]]
    total_text = " ".join(text_lines)
    assert len(total_text) > 100, "Expected substantial transcription text"
