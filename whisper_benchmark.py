#!/usr/bin/env python3
"""
mlx-audio Benchmark

This script benchmarks the performance of different mlx-audio models
(Parakeet v3 and Whisper) for audio transcription.

Usage:
    python whisper_benchmark.py
    python whisper_benchmark.py --audio test_audio/yt_video.mp3 --models parakeet-v3 parakeet-1.1b
    python whisper_benchmark.py --simple

Author: Joseph Schlesinger
"""

import os
import time
import logging
import argparse
import subprocess
from tabulate import tabulate
from mlx_audio.stt import load
from yt2srt import resolve_model, MODEL_ALIASES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define available models (full HuggingFace IDs + aliases)
AVAILABLE_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-1.1b",
    "mlx-community/parakeet-ctc-0.6b",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
] + list(MODEL_ALIASES.keys())


def get_audio_duration(audio_path):
    """
    Get the duration of an audio file in seconds using ffprobe.

    Args:
        audio_path (str): Path to the audio file

    Returns:
        float: Duration of the audio file in seconds
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError) as e:
        logger.warning(f"Could not determine audio duration: {str(e)}")
        return None


def transcribe_audio(model_name, audio_path, chunk_duration=30.0):
    """
    Transcribe an audio file using the specified model.

    Args:
        model_name (str): HuggingFace model ID or alias
        audio_path (str): Path to the audio file
        chunk_duration (float): Chunk duration in seconds for long-form audio

    Returns:
        dict: Dictionary containing transcription results and timing information
    """
    resolved = resolve_model(model_name)
    logger.info(f"Transcribing with model: {resolved}")

    # Get audio duration
    audio_duration = get_audio_duration(audio_path)
    if audio_duration:
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")

    # Initialize model
    start_time = time.time()
    try:
        model = load(resolved)
        init_time = time.time() - start_time
        logger.info(f"Model initialization took {init_time:.2f} seconds")

        # Transcribe audio
        transcription_start = time.time()
        result = model.generate(audio=audio_path, chunk_duration=chunk_duration)
        transcription_time = time.time() - transcription_start

        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
        logger.info(f"Transcription result: {result.text[:100]}...")

        # Calculate real-time factor
        realtime_factor = None
        if audio_duration:
            realtime_factor = transcription_time / audio_duration
            logger.info(f"Real-time factor: {realtime_factor:.2f}x (1/{1/realtime_factor:.2f} of real-time)")

        return {
            "model": model_name,
            "resolved_model": resolved,
            "init_time": init_time,
            "transcription_time": transcription_time,
            "total_time": init_time + transcription_time,
            "audio_duration": audio_duration,
            "realtime_factor": realtime_factor,
            "text": result.text,
            "sentences": result.sentences,
        }
    except Exception as e:
        logger.error(f"Error with model {model_name}: {str(e)}")
        return {
            "model": model_name,
            "resolved_model": resolved,
            "init_time": time.time() - start_time,
            "transcription_time": None,
            "total_time": None,
            "audio_duration": audio_duration,
            "realtime_factor": None,
            "error": str(e)
        }


def run_benchmark(audio_path, models=None, chunk_duration=30.0):
    """
    Run a benchmark comparing different models.

    Args:
        audio_path (str): Path to the audio file
        models (list): List of models to benchmark
        chunk_duration (float): Chunk duration for long-form audio

    Returns:
        list: List of benchmark results
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return []

    if models is None or len(models) == 0:
        logger.warning("No models specified. Using default: parakeet-v3")
        models = ["mlx-community/parakeet-tdt-0.6b-v3"]

    logger.info(f"Running benchmark with models: {models}")
    results = []

    for model in models:
        result = transcribe_audio(model, audio_path, chunk_duration)
        results.append(result)
        logger.info("-" * 80)

    return results


def display_results(results):
    """
    Display benchmark results in a table.

    Args:
        results (list): List of benchmark results
    """
    table_data = []
    for result in results:
        if "error" in result:
            row = [
                result["model"],
                f"{result['init_time']:.2f}s" if result['init_time'] else "N/A",
                "Error",
                "Error",
                f"{result['audio_duration']:.2f}s" if result.get('audio_duration') else "N/A",
                "N/A",
                result["error"]
            ]
        else:
            realtime_info = "N/A"
            if result.get('realtime_factor'):
                realtime_info = f"{result['realtime_factor']:.2f}x (1/{1/result['realtime_factor']:.2f} real-time)"

            row = [
                result["model"],
                f"{result['init_time']:.2f}s",
                f"{result['transcription_time']:.2f}s",
                f"{result['total_time']:.2f}s",
                f"{result['audio_duration']:.2f}s" if result.get('audio_duration') else "N/A",
                realtime_info,
                result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]
            ]
        table_data.append(row)

    headers = ["Model", "Init Time", "Transcription Time", "Total Time", "Audio Duration", "Real-time Factor", "Result"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x["transcription_time"])
        print(f"\nFastest model: {fastest['model']}")
        print(f"Transcription time: {fastest['transcription_time']:.2f}s")
        if fastest.get('realtime_factor'):
            print(f"Real-time factor: {fastest['realtime_factor']:.2f}x (1/{1/fastest['realtime_factor']:.2f} of real-time)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="mlx-audio Benchmark (Parakeet v3 / Whisper)")
    parser.add_argument("--audio", default="test_audio/yt_video.mp3", help="Path to audio file")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to benchmark (HuggingFace IDs or aliases: " + ", ".join(MODEL_ALIASES.keys()) + ")"
    )
    parser.add_argument("--chunk-duration", type=float, default=30.0, help="Chunk duration in seconds")
    parser.add_argument("--simple", action="store_true", help="Run a simple transcription example with the default model")

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return

    # Print audio file information
    audio_size_mb = os.path.getsize(args.audio) / (1024 * 1024)
    logger.info(f"Audio file: {args.audio} ({audio_size_mb:.2f} MB)")

    if args.simple:
        # Simple example
        logger.info("Running simple transcription example with parakeet-v3")
        result = transcribe_audio("mlx-community/parakeet-tdt-0.6b-v3", args.audio, args.chunk_duration)
        print("\nTranscription result:")
        print(result["text"])
        if result.get('realtime_factor'):
            print(f"\nReal-time factor: {result['realtime_factor']:.2f}x (1/{1/result['realtime_factor']:.2f} of real-time)")
    else:
        # Run benchmark
        results = run_benchmark(args.audio, args.models, args.chunk_duration)
        display_results(results)

        # Save the full transcription from the first successful result
        valid = [r for r in results if "error" not in r]
        if valid:
            with open("transcription_result.txt", "w") as f:
                f.write(valid[0]["text"])
            logger.info("Full transcription saved to transcription_result.txt")


if __name__ == "__main__":
    main()
