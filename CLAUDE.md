# 2md - Audio/Video to Markdown Transcription Toolkit

## Overview

A toolkit for transcribing audio/video (including YouTube videos) to markdown, SRT, or plain text using [mlx-audio](https://github.com/Blaizzy/mlx-audio) with Parakeet v3 on Apple Silicon. Includes benchmarking tools for comparing model performance.

## Project Structure

```
2md/
├── yt2md.py               # Main tool: YouTube/local audio/video → markdown/SRT/text
├── whisper_benchmark.py   # Interactive benchmark: compare models with tabulate output
├── benchmark_models.py    # Automated benchmark: warmup + timed runs, markdown report
├── download_models.py     # Pre-download mlx-audio models from HuggingFace
├── quant_test.py          # Quick model smoke test
├── transcription_cleanup_prompt.txt  # LLM prompt template for cleaning raw transcripts
├── requirements.txt       # Dependencies
├── test_yt2md.py          # Unit tests (extract_video_id, formatters, resolve_model)
├── test_benchmark.py      # Tests for whisper_benchmark.py
├── test_audio/            # Test audio files (mp3/wav)
├── benchmark-results/     # Generated benchmark reports
├── worklog.md             # Development history and findings
└── .cursor/mcp.json       # Cursor MCP config (project-name-mcp)
```

## Key Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `mlx-audio[stt]` | >=0.4.0 | Parakeet/Whisper inference optimized for Apple Silicon (MLX) |
| `yt-dlp` | >=2023.11.14 | YouTube audio download |
| `tabulate` | (any) | Benchmark result formatting |
| `ffmpeg`/`ffprobe` | system | Audio conversion and duration detection |

### llm-code-docs availability
- `mlx-lm`, `mlx-swift`: docs exist at `~/github/llm-code-docs/docs/github-scraped/`
- `mlx-audio`, `yt-dlp`: NOT in llm-code-docs

## Architecture

### yt2md.py (main tool)
- **Input auto-detection**: YouTube URL, YouTube ID (11 chars), or local file path
- **Pipeline**: Download (yt-dlp) → Convert to 16kHz mono WAV (ffmpeg) → Transcribe (mlx-audio) → output (md/srt/txt)
- **Default output**: Markdown (`--format md`). Also supports `srt` and `txt`.
- **Long audio**: mlx-audio handles chunking internally via `chunk_duration` parameter (default 30s). No manual splitting needed.
- **Model resolution**: Accepts full HuggingFace IDs (`mlx-community/parakeet-tdt-0.6b-v3`) or short aliases (`parakeet-v3`). See `MODEL_ALIASES` dict.
- **Quantization**: Baked into HuggingFace model weights — no runtime parameter. Choose a different model ID for a quantized variant.
- **AlignedResult**: `model.generate()` returns an `AlignedResult` with `.text` (str) and `.sentences` (list of `AlignedSentence` with `.start`, `.end`, `.text`, `.tokens`).

### Output Formats
- **md** (default): `# Title` heading + `**[MM:SS]** Sentence text.` paragraphs
- **srt**: Standard SRT subtitle format with `HH:MM:SS,mmm` timestamps
- **txt**: Plain text, no timestamps, double-newline separated paragraphs

### Available Models

| Alias | HuggingFace ID | Notes |
|-------|----------------|-------|
| `parakeet-v3` | `mlx-community/parakeet-tdt-0.6b-v3` | Default. 25 EU languages + RU/UK |
| `parakeet-v2` | `mlx-community/parakeet-tdt-0.6b-v2` | English only |
| `parakeet-1.1b` | `mlx-community/parakeet-tdt-1.1b` | Larger, more accurate |
| `parakeet-ctc` | `mlx-community/parakeet-ctc-0.6b` | CTC variant |
| `whisper-turbo` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | Whisper fallback |

### Benchmark scripts
- `whisper_benchmark.py`: Interactive CLI with `--simple` mode and `--models` selection. Uses tabulate for display.
- `benchmark_models.py`: Automated full benchmark. Warmup run per model, then timed run. Outputs markdown report to `benchmark-results/`.

## Testing

- **Framework**: `unittest` (stdlib)
- **Unit tests**: `python test_yt2md.py` — tests `extract_video_id`, `segments_to_srt`, `segments_to_markdown`, `segments_to_text`, `resolve_model` (no network/model required)
- **Benchmark tests**: `python test_benchmark.py` — requires mlx-audio installed and test audio files

## Installation

```bash
# System deps
brew install ffmpeg

# Python deps
pip install -r requirements.txt

# Optional: pre-download models
python download_models.py
```

Models are cached by the HuggingFace hub in `~/.cache/huggingface/`.

## Usage

```bash
# Transcribe YouTube video to markdown (default)
python yt2md.py https://www.youtube.com/watch?v=VIDEO_ID

# Transcribe local file
python yt2md.py my_video.mp4 --model parakeet-1.1b

# Output SRT instead
python yt2md.py podcast.mp3 --format srt

# Plain text
python yt2md.py lecture.mp4 --format txt

# Control chunk duration for long audio (default 30s)
python yt2md.py long_lecture.mp4 --chunk-duration 60

# Benchmark
python whisper_benchmark.py --audio test_audio/yt_video.mp3 --models parakeet-v3 parakeet-1.1b
python benchmark_models.py  # full automated benchmark
```
