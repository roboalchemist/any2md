#!/usr/bin/env python3
"""
Model Test for mlx-audio (Parakeet v3 / Whisper)

With mlx-audio, quantization is baked into the model weights on HuggingFace —
there is no runtime quantization parameter. This script simply runs a quick
smoke-test transcription with the default Parakeet v3 model.

For quantized variants, use a different HuggingFace model ID directly.

Usage:
    python quant_test.py
    python quant_test.py --audio test_audio/test_voice.mp3
"""

import os
import time
import logging
import argparse
from mlx_audio.stt import load
from yt2srt import resolve_model, MODEL_ALIASES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model(model_name: str, audio_path: str = "test_audio/test_voice.mp3", is_warmup: bool = False):
    """Run a quick transcription test with the given model."""
    prefix = "[WARMUP] " if is_warmup else ""
    resolved = resolve_model(model_name)
    print(f"{prefix}Testing model: {resolved}")

    start_time = time.time()
    try:
        model = load(resolved)
        result = model.generate(audio=audio_path, chunk_duration=30.0)
        elapsed = time.time() - start_time

        print(f"{prefix}Transcription completed in {elapsed:.2f} seconds")
        print(f"{prefix}Result: {result.text[:200]}")
        print("-" * 80)
        return True
    except Exception as e:
        import traceback
        print(f"{prefix}Error testing {resolved}: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run smoke tests against available Parakeet models."""
    parser = argparse.ArgumentParser(description="mlx-audio model smoke test")
    parser.add_argument("--audio", default="test_audio/test_voice.mp3", help="Path to test audio file")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlx-community/parakeet-tdt-0.6b-v3"],
        help="Models to test (HuggingFace IDs or aliases)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        return

    print("Starting warmup phase...")
    for model_name in args.models:
        test_model(model_name, audio_path=args.audio, is_warmup=True)

    print("\nStarting main test phase...")
    for model_name in args.models:
        test_model(model_name, audio_path=args.audio, is_warmup=False)


if __name__ == "__main__":
    main()
