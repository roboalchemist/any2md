#!/usr/bin/env python3
"""
download_models.py - Pre-download MLX models for 2md tools

Downloads and caches models from HuggingFace so they are available for fast
startup in subsequent runs.

Usage:
    python download_models.py            # download all models (~15+ GB)
    python download_models.py --stt      # Parakeet STT models only
    python download_models.py --vlm      # Qwen3.5-27B VLM model only
    python download_models.py --reader   # ReaderLM-v2 only
    python download_models.py --docling  # SmolDocling-256M only
    python download_models.py --asr-zh   # Qwen3-ASR + ForcedAligner (~9.4 GB)
"""

import logging
import time
from typing import Optional

import typer
from typing_extensions import Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# STT models (Parakeet) + LID (Whisper-tiny used for --language auto detection)
STT_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-1.1b",
    "mlx-community/parakeet-ctc-0.6b",
]

# LID (language identification) models — tiny; needed for --language auto
LID_MODELS = [
    "mlx-community/whisper-tiny-asr-fp16",
]

# VLM models
VLM_MODELS = [
    "mlx-community/Qwen3.5-9B-MLX-4bit",
]

# Reader models (URL-to-markdown)
READER_MODELS = [
    "mlx-community/jinaai-ReaderLM-v2",
]

# Document layout models
# NOTE: the old mlx-community/SmolDocling-256M-preview-mlx-bf16 repo is gone;
# the real MLX conversion lives under the docling-project org.
DOCLING_MODELS = [
    "docling-project/SmolDocling-256M-preview-mlx-bf16",
]

# Diarization models (Nemotron-3-Diarization is the default; the streaming
# Sortformer v2.1 is kept for anyone passing it via --diarize-model)
DIARIZE_MODELS = [
    "mlx-community/Nemotron-3-Diarization",
    "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
]

# Qwen3-ASR models (Chinese / multilingual STT)
QWEN3_ASR_MODELS = [
    "mlx-community/Qwen3-ASR-1.7B-bf16",   # default alias from ANY2-30
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-ASR-1.7B-4bit",
]

# Qwen3-ForcedAligner models
QWEN3_ALIGNER_MODELS = [
    "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
]

app = typer.Typer(
    help="Download MLX models for 2md tools.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


def _download_stt_model(model_id: str) -> None:
    """Download a single STT model via mlx-audio."""
    try:
        from mlx_audio.stt import load_model
    except ImportError:
        logger.error("mlx-audio[stt] is required. Install it with: pip install mlx-audio[stt]")
        raise

    logger.info(f"Downloading STT model: {model_id}")
    start = time.time()
    try:
        load_model(model_id)
        logger.info(f"  Ready in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"  Failed: {e}")


def _download_via_snapshot(model_id: str) -> None:
    """Download a model directly via huggingface_hub snapshot_download.

    Use this for models whose repo names don't match mlx-audio's dispatch
    registry (e.g. Qwen3-ASR variants whose hyphenated names fail the
    underscore-keyed registry lookup).  Cache-population is all we need —
    no model instantiation required.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is required. Install it with: pip install huggingface-hub")
        raise

    logger.info(f"Downloading model: {model_id}")
    start = time.time()
    try:
        snapshot_download(repo_id=model_id)
        logger.info(f"  Ready in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"  Failed: {e}")


def _download_vlm_model(model_id: str) -> None:
    """Download a single VLM model via mlx-vlm."""
    try:
        import mlx_vlm
    except ImportError:
        logger.error("mlx-vlm is required. Install it with: pip install mlx-vlm")
        raise

    logger.info(f"Downloading VLM model: {model_id}")
    start = time.time()
    try:
        mlx_vlm.load(model_id)
        logger.info(f"  Ready in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"  Failed: {e}")


def _download_diarize_model(model_id: str) -> None:
    """Download a single diarization model via mlx-audio VAD."""
    try:
        from mlx_audio.vad import load
    except ImportError:
        logger.error("mlx-audio[stt] is required. Install it with: pip install mlx-audio[stt]")
        raise

    logger.info(f"Downloading diarization model: {model_id}")
    start = time.time()
    try:
        load(model_id)
        logger.info(f"  Ready in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"  Failed: {e}")


def _download_lm_model(model_id: str) -> None:
    """Download a text/reader model via mlx-lm."""
    try:
        import mlx_lm
    except ImportError:
        logger.error("mlx-lm is required. Install it with: pip install mlx-lm")
        raise

    logger.info(f"Downloading LM model: {model_id}")
    start = time.time()
    try:
        mlx_lm.load(model_id)
        logger.info(f"  Ready in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"  Failed: {e}")


@app.command()
def main(
    stt: Annotated[bool, typer.Option(
        "--stt",
        help="Download STT models (Parakeet v2/v3, 1.1B, CTC).",
    )] = False,
    vlm: Annotated[bool, typer.Option(
        "--vlm",
        help="Download VLM models (Qwen3.5-9B-MLX-4bit).",
    )] = False,
    reader: Annotated[bool, typer.Option(
        "--reader",
        help="Download ReaderLM-v2 for URL-to-markdown conversion.",
    )] = False,
    docling: Annotated[bool, typer.Option(
        "--docling",
        help="Download SmolDocling-256M for document layout analysis.",
    )] = False,
    diarize: Annotated[bool, typer.Option(
        "--diarize",
        help="Download Nemotron-3-Diarization (default) + streaming Sortformer models.",
    )] = False,
    asr_zh: Annotated[bool, typer.Option(
        "--asr-zh",
        help="Download Qwen3-ASR-1.7B + ForcedAligner + LID model for Chinese/multilingual (~9.4 GB).",
    )] = False,
    all_models: Annotated[bool, typer.Option(
        "--all",
        help="Download all models.",
    )] = False,
) -> None:
    """
    Download MLX models for 2md tools.

    If no flags are specified, all models are downloaded. Use individual flags
    to download only the models needed for specific tools.
    """
    # If no flags, download all
    if not any([stt, vlm, reader, docling, diarize, asr_zh, all_models]):
        all_models = True

    if stt or all_models:
        logger.info("--- STT models (Parakeet) ---")
        for model_id in STT_MODELS:
            _download_stt_model(model_id)

        logger.info("--- LID models (language detection, --language auto) ---")
        for model_id in LID_MODELS:
            _download_stt_model(model_id)

    if vlm or all_models:
        logger.info("--- VLM models ---")
        for model_id in VLM_MODELS:
            _download_vlm_model(model_id)

    if reader or all_models:
        logger.info("--- Reader models (ReaderLM-v2) ---")
        for model_id in READER_MODELS:
            _download_lm_model(model_id)

    if docling or all_models:
        logger.info("--- Docling models (SmolDocling) ---")
        for model_id in DOCLING_MODELS:
            _download_lm_model(model_id)

    if diarize or all_models:
        logger.info("--- Diarization models (Nemotron-3 + Sortformer) ---")
        for model_id in DIARIZE_MODELS:
            _download_diarize_model(model_id)

    if asr_zh or all_models:
        # Use snapshot_download directly for all Qwen3 models — mlx-audio's
        # load_model dispatcher splits on hyphens and looks up underscore-keyed
        # registry entries (e.g. 'qwen3_asr'), so hyphenated repo names like
        # 'Qwen3-ASR-1.7B-bf16' yield Model type None and raise ValueError.
        logger.info("--- Qwen3-ASR models (multilingual STT) ---")
        for model_id in QWEN3_ASR_MODELS:
            _download_via_snapshot(model_id)

        logger.info("--- Qwen3-ForcedAligner models ---")
        for model_id in QWEN3_ALIGNER_MODELS:
            _download_via_snapshot(model_id)

        logger.info("--- LID models (language detection) ---")
        for model_id in LID_MODELS:
            _download_via_snapshot(model_id)

    logger.info("Done.")


if __name__ == "__main__":
    app()
