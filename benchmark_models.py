#!/usr/bin/env python3
"""
Benchmark script for mlx-audio models (Parakeet v3 / Whisper).
Tests all specified models and reports transcription speed.
Each model gets a warmup run before its timed run.
"""

import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from yt2srt import resolve_model, MODEL_ALIASES

# Models to benchmark (full HuggingFace IDs or aliases)
MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-1.1b",
    "mlx-community/parakeet-ctc-0.6b",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
]


def get_audio_duration(file_path: str) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        file_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = json.loads(result.stdout)["format"]["duration"]
        return float(duration)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"Error getting duration: {e}")
        return 0.0


def run_transcription(
    input_file: str,
    model: str,
    output_dir: str = "benchmark-results",
    is_warmup: bool = False,
    chunk_duration: float = 30.0,
) -> Tuple[float, str]:
    """Run transcription via yt2srt.py subprocess and return processing time and output file path."""

    # Use a safe suffix for output filename
    model_safe = model.replace("/", "_").replace(":", "_")
    if is_warmup:
        model_safe += "-warmup"
    output_path = os.path.join(output_dir, f"{model_safe}.srt")

    # Build command
    cmd = [
        "python", "yt2srt.py",
        input_file,
        "--model", model,
        "-o", output_dir,
        "--chunk-duration", str(chunk_duration),
        "--verbose"
    ]

    # Time the transcription
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        return end_time - start_time, output_path
    except subprocess.CalledProcessError as e:
        print(f"Error running transcription: {e.stderr}")
        return 0.0, ""


def format_timing(proc_time: float, audio_duration: float) -> str:
    """Format timing details including speedup."""
    speedup = audio_duration / proc_time
    return f"took {proc_time:.2f}s ({speedup:.1f}x real-time)"


def benchmark_model(
    warmup_file: str,
    benchmark_file: str,
    model: str,
    output_dir: str,
    benchmark_duration: float,
    chunk_duration: float = 30.0,
) -> Optional[Dict]:
    """Run warmup and benchmark for a specific model."""
    model_desc = resolve_model(model)
    print(f"\nWarmup run for {model_desc}...")
    warmup_time, _ = run_transcription(warmup_file, model, output_dir, is_warmup=True, chunk_duration=chunk_duration)
    print(f"Warmup {format_timing(warmup_time, get_audio_duration(warmup_file))}")

    print(f"Benchmarking {model_desc}...")
    proc_time, output_path = run_transcription(benchmark_file, model, output_dir, chunk_duration=chunk_duration)
    print(f"Benchmark {format_timing(proc_time, benchmark_duration)}")

    if proc_time > 0:
        speed = benchmark_duration / proc_time
        return {
            "model": model,
            "time": proc_time,
            "speed": speed,
            "output": output_path,
            "warmup_time": warmup_time,
            "warmup_speed": get_audio_duration(warmup_file) / warmup_time if warmup_time > 0 else 0,
        }
    return None


def format_results(results: List[Dict], audio_duration: float) -> str:
    """Format benchmark results as markdown."""
    markdown = "# mlx-audio Benchmark Results\n\n"
    markdown += "## Test Configuration\n"
    markdown += f"- Audio duration: {audio_duration:.2f} seconds\n"
    markdown += "- Hardware: Apple Silicon Mac\n"
    markdown += "- Package: mlx-audio\n"
    markdown += "- Note: Each model had a warmup run before its timed run\n\n"

    markdown += "## Results\n\n"
    markdown += "| Model | Warmup Time (s) | Warmup Speed | Benchmark Time (s) | Benchmark Speed |\n"
    markdown += "|-------|----------------|--------------|-------------------|----------------|\n"

    # Sort results by benchmark speed (fastest first)
    sorted_results = sorted(results, key=lambda x: x["speed"], reverse=True)

    for result in sorted_results:
        markdown += (
            f"| {result['model']} | "
            f"{result['warmup_time']:.2f} | {result['warmup_speed']:.1f}x | "
            f"{result['time']:.2f} | {result['speed']:.1f}x |\n"
        )

    return markdown


def main():
    # Setup
    warmup_file = "test_audio/test_voice.mp3"
    benchmark_file = "test_audio/test_yt_video.wav"
    output_dir = "benchmark-results"
    chunk_duration = 30.0
    os.makedirs(output_dir, exist_ok=True)

    # Get audio durations
    benchmark_duration = get_audio_duration(benchmark_file)
    print(f"Benchmark file duration: {benchmark_duration:.2f} seconds")
    warmup_duration = get_audio_duration(warmup_file)
    print(f"Warmup file duration: {warmup_duration:.2f} seconds")

    # Run benchmarks
    results = []
    print("\nRunning benchmarks...")

    for model in MODELS:
        result = benchmark_model(warmup_file, benchmark_file, model, output_dir, benchmark_duration, chunk_duration)
        if result:
            results.append(result)

    # Generate and save report
    report = format_results(results, benchmark_duration)
    report_path = os.path.join(output_dir, "benchmark-results.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nBenchmark complete! Results saved to {report_path}")


if __name__ == "__main__":
    main()
