#!/usr/bin/env python3
"""
Download mlx-audio models (Parakeet v3, Whisper variants)

This script pre-downloads mlx-audio models from HuggingFace so they are cached
locally for faster startup in subsequent runs.

Usage:
    python download_models.py
    python download_models.py --models parakeet-v3 parakeet-1.1b

Author: Joseph Schlesinger
"""

import os
import time
import logging
import argparse
from mlx_audio.stt import load
from yt2srt import resolve_model, MODEL_ALIASES, SUPPORTED_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default set of models to download
DEFAULT_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-1.1b",
    "mlx-community/parakeet-ctc-0.6b",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
]


def download_model(model_name: str):
    """
    Download (warm-up cache for) a model from HuggingFace via mlx-audio.

    Args:
        model_name (str): HuggingFace model ID or alias
    """
    resolved = resolve_model(model_name)
    logger.info(f"Downloading model: {resolved}")

    start_time = time.time()
    try:
        load(resolved)
        download_time = time.time() - start_time
        logger.info(f"Model {resolved} ready in {download_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error downloading model {resolved}: {str(e)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download mlx-audio models (Parakeet v3 / Whisper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to download. Accepts HuggingFace IDs or aliases: "
            + ", ".join(f"{k} -> {v}" for k, v in MODEL_ALIASES.items())
        ),
    )
    args = parser.parse_args()

    logger.info(f"Downloading {len(args.models)} models")
    for model in args.models:
        download_model(model)
        logger.info("-" * 80)

    logger.info("All models downloaded successfully")


if __name__ == "__main__":
    main()
