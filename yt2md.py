#!/usr/bin/env python3
"""
yt2md.py - Audio/Video to Markdown Transcription Tool

This script transcribes audio to markdown (default), SRT, or plain text using the
mlx-audio library with Parakeet v3. It can process YouTube videos, local audio files,
or local video files.

Usage:
    python yt2md.py [options] <input>

Where <input> can be:
    - YouTube URL
    - YouTube video ID (11 characters)
    - Path to local audio/video file

Examples:
    python yt2md.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    python yt2md.py dQw4w9WgXcQ --model parakeet-v3
    python yt2md.py my_video.mp4 --format srt
    python yt2md.py podcast.mp3 --model mlx-community/parakeet-tdt-1.1b

The script can also be imported and used by other Python projects.
"""

import os
import re
import sys
import time
import json
import argparse
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
AUDIO_SAMPLE_RATE = 16000  # 16kHz as required by Parakeet/Whisper
MODELS_DIR = "mlx_models"  # Directory for cached models

# Short alias -> full HuggingFace model ID
MODEL_ALIASES = {
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
    "parakeet-v2": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-1.1b": "mlx-community/parakeet-tdt-1.1b",
    "parakeet-ctc": "mlx-community/parakeet-ctc-0.6b",
    "whisper-turbo": "mlx-community/whisper-large-v3-turbo-asr-fp16",
}

SUPPORTED_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-1.1b",
    "mlx-community/parakeet-ctc-0.6b",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
] + list(MODEL_ALIASES.keys())


def resolve_model(model_name: str) -> str:
    """Resolve a model alias or short name to its full HuggingFace model ID."""
    return MODEL_ALIASES.get(model_name, model_name)


def extract_video_id(url_or_id: str) -> str:
    """
    Extract the YouTube video ID from a URL or return the ID if already provided.

    Args:
        url_or_id: YouTube URL or video ID

    Returns:
        The YouTube video ID
    """
    # YouTube ID pattern
    youtube_id_pattern = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'

    # Check if it's already just an ID (11 characters)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    # Try to extract ID from URL
    match = re.search(youtube_id_pattern, url_or_id)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract YouTube video ID from: {url_or_id}")


def download_youtube_audio(url_or_id: str, output_dir: Optional[str] = None) -> Tuple[str, Dict]:
    """
    Download audio from a YouTube video.

    Args:
        url_or_id: YouTube URL or video ID
        output_dir: Directory to save the downloaded audio (default: temporary directory)

    Returns:
        Tuple containing (audio_file_path, metadata_dict)
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is required. Install it with: pip install yt-dlp")
        raise

    video_id = extract_video_id(url_or_id)

    # Use temporary directory if none specified
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output filename template
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        'extractaudio': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    logger.info(f"Downloading audio from YouTube video: {video_id}")

    # Download the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

    # The downloaded file path
    audio_file = os.path.join(output_dir, f"{video_id}.mp3")

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Failed to download audio file: {audio_file}")

    # Extract metadata
    metadata = extract_youtube_metadata(info)

    logger.info(f"Successfully downloaded audio to: {audio_file}")
    return audio_file, metadata


def extract_youtube_metadata(info: Dict) -> Dict:
    """
    Extract useful metadata from yt-dlp info dict.

    Args:
        info: Raw info dict from yt-dlp extract_info()

    Returns:
        Cleaned metadata dict suitable for frontmatter
    """
    # Format upload_date from YYYYMMDD to YYYY-MM-DD
    upload_date = info.get('upload_date')
    if upload_date and len(upload_date) == 8:
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    # Format duration to human-readable
    duration_secs = info.get('duration')
    if duration_secs:
        mins, secs = divmod(int(duration_secs), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            duration_human = f"{hours}h {mins}m {secs}s"
        elif mins:
            duration_human = f"{mins}m {secs}s"
        else:
            duration_human = f"{secs}s"
    else:
        duration_human = None

    # Format chapters
    chapters = info.get('chapters') or []
    formatted_chapters = []
    for ch in chapters:
        start = ch.get('start_time', 0)
        title = ch.get('title', '')
        if title:
            formatted_chapters.append({"time": format_timestamp_md(start), "title": title})

    # Collect available subtitle languages
    subtitles = list((info.get('subtitles') or {}).keys())
    auto_captions = list((info.get('automatic_captions') or {}).keys())

    metadata = {
        'title': info.get('title'),
        'video_id': info.get('id'),
        'url': info.get('webpage_url'),
        'channel': info.get('channel'),
        'channel_url': info.get('channel_url'),
        'uploader': info.get('uploader'),
        'upload_date': upload_date,
        'duration': duration_secs,
        'duration_human': duration_human,
        'description': info.get('description'),
        'categories': info.get('categories') or [],
        'tags': info.get('tags') or [],
        'view_count': info.get('view_count'),
        'like_count': info.get('like_count'),
        'comment_count': info.get('comment_count'),
        'channel_follower_count': info.get('channel_follower_count'),
        'thumbnail': info.get('thumbnail'),
        'language': info.get('language'),
        'availability': info.get('availability'),
        'live_status': info.get('live_status'),
        'location': info.get('location'),
        'chapters': formatted_chapters,
        'subtitles': subtitles,
        'auto_captions': auto_captions,
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    # Remove None values
    return {k: v for k, v in metadata.items() if v is not None}


def convert_audio_for_whisper(input_file: str, output_dir: Optional[str] = None) -> str:
    """
    Convert audio file to the format required by Parakeet/Whisper (16kHz, mono, WAV).

    Args:
        input_file: Path to the input audio file
        output_dir: Directory to save the converted audio (default: same as input file)

    Returns:
        Path to the converted audio file
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_whisper.wav")

    # FFmpeg command to convert to 16kHz mono WAV
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_file,
        "-ar", str(AUDIO_SAMPLE_RATE),  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        output_file
    ]

    logger.info(f"Converting audio to WAV format: {output_file}")

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        raise

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Failed to convert audio file: {output_file}")

    logger.info(f"Successfully converted audio to: {output_file}")
    return output_file


SUPPORTED_FORMATS = ["md", "srt", "txt"]
DEFAULT_FORMAT = "md"


def _extract_sentence_fields(sentence) -> tuple:
    """Extract (start, end, text) from an AlignedSentence object or dict."""
    if hasattr(sentence, 'start'):
        return sentence.start, sentence.end, sentence.text
    elif isinstance(sentence, dict):
        return sentence["start"], sentence["end"], sentence["text"]
    else:
        return None, None, None


def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


def format_timestamp_md(seconds: float) -> str:
    """Convert seconds to readable timestamp: HH:MM:SS or MM:SS"""
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def segments_to_srt(sentences: List) -> str:
    """
    Convert AlignedResult sentences to SRT format.

    Args:
        sentences: List of AlignedSentence objects (with .start, .end, .text attributes)
                   or dicts with "start", "end", "text" keys

    Returns:
        SRT formatted string
    """
    srt_lines = []

    for i, sentence in enumerate(sentences):
        start_time, end_time, text = _extract_sentence_fields(sentence)
        if start_time is None:
            logger.warning(f"Unknown sentence format: {sentence}")
            continue

        srt_lines.append(f"{i+1}")
        srt_lines.append(f"{format_timestamp_srt(start_time)} --> {format_timestamp_srt(end_time)}")
        srt_lines.append(text.strip())

        if i < len(sentences) - 1:
            srt_lines.append("")

    return "\n".join(srt_lines)


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


def segments_to_markdown(sentences: List, title: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> str:
    """
    Convert AlignedResult sentences to Markdown format.

    Produces a document with optional YAML frontmatter, a title heading,
    followed by timestamped paragraphs.

    Args:
        sentences: List of AlignedSentence objects or dicts
        title: Optional title for the markdown heading
        metadata: Optional metadata dict for YAML frontmatter

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

    for sentence in sentences:
        start_time, end_time, text = _extract_sentence_fields(sentence)
        if start_time is None:
            logger.warning(f"Unknown sentence format: {sentence}")
            continue

        ts = format_timestamp_md(start_time)
        lines.append(f"**[{ts}]** {text.strip()}")
        lines.append("")

    return "\n".join(lines)


def segments_to_text(sentences: List) -> str:
    """
    Convert AlignedResult sentences to plain text (no timestamps).

    Args:
        sentences: List of AlignedSentence objects or dicts

    Returns:
        Plain text string
    """
    parts = []
    for sentence in sentences:
        start_time, end_time, text = _extract_sentence_fields(sentence)
        if start_time is None:
            continue
        parts.append(text.strip())
    return "\n\n".join(parts)


def transcribe(
    audio_file: str,
    model_name: str = DEFAULT_MODEL,
    output_dir: Optional[str] = None,
    video_title: Optional[str] = None,
    video_id: Optional[str] = None,
    chunk_duration: float = 30.0,
    output_format: str = DEFAULT_FORMAT,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Transcribe audio file using mlx-audio (Parakeet/Whisper).

    Args:
        audio_file: Path to the audio file
        model_name: HuggingFace model ID or alias to use
        output_dir: Directory to save the output file (default: same as audio file)
        video_title: Title of the video (for naming the output file)
        video_id: YouTube video ID (for naming the output file)
        chunk_duration: Duration of each audio chunk in seconds (default: 30.0)
        output_format: Output format — "md" (default), "srt", or "txt"
        metadata: Optional metadata dict for YAML frontmatter (markdown only)

    Returns:
        Path to the generated output file
    """
    try:
        from mlx_audio.stt import load
    except ImportError:
        logger.error("mlx-audio is required. Install it with: pip install mlx-audio[stt]")
        raise

    model_name = resolve_model(model_name)

    if output_dir is None:
        output_dir = os.path.dirname(audio_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    ext = output_format if output_format != "md" else "md"
    if video_id and video_title:
        clean_title = re.sub(r'[\\/*?:"<>|]', "", video_title)
        output_file = os.path.join(output_dir, f"[{video_id}] {clean_title}.{ext}")
    else:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.{ext}")

    logger.info(f"Transcribing audio using model: {model_name}")
    start_time = time.time()

    # Load model
    model = load(model_name)

    # Transcribe — mlx-audio handles long audio via chunk_duration internally
    result = model.generate(
        audio=audio_file,
        chunk_duration=chunk_duration,
    )

    end_time = time.time()
    logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

    # Format output
    if output_format == "md":
        display_title = video_title or os.path.splitext(os.path.basename(audio_file))[0]
        content = segments_to_markdown(result.sentences, title=display_title, metadata=metadata)
    elif output_format == "txt":
        content = segments_to_text(result.sentences)
    else:
        content = segments_to_srt(result.sentences)

    # Write the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Output saved to: {output_file}")
    return output_file


def process_input_file(input_file: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Process an input audio/video file.

    Args:
        input_file: Path to the input audio/video file
        output_dir: Directory to save the processed audio (default: temporary directory)

    Returns:
        Tuple containing (audio_file_path, title)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Use temporary directory if none specified
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename without extension as title
    title = os.path.splitext(os.path.basename(input_file))[0]

    # Convert to mp3 if not already
    if not input_file.lower().endswith('.mp3'):
        output_file = os.path.join(output_dir, f"{title}.mp3")
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-i", input_file,
            "-vn",  # Disable video
            "-acodec", "libmp3lame",
            "-q:a", "4",  # High quality
            output_file
        ]

        logger.info(f"Converting input file to mp3: {output_file}")

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise

        audio_file = output_file
    else:
        audio_file = input_file

    return audio_file, title


def auto_detect_input(input_path: str) -> Tuple[str, bool]:
    """
    Auto-detect if the input is a YouTube URL, YouTube ID, or local file.

    Args:
        input_path: The input path or URL

    Returns:
        Tuple of (input_type, is_youtube) where:
        - input_type: 'youtube' or 'file'
        - is_youtube: True if it's a YouTube URL or ID
    """
    # Check if it's a YouTube URL
    youtube_url_pattern = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)'
    if re.search(youtube_url_pattern, input_path):
        logger.info(f"Detected YouTube URL: {input_path}")
        return 'youtube', True

    # Check if it's a YouTube ID (11 characters)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', input_path):
        logger.info(f"Detected YouTube ID: {input_path}")
        return 'youtube', True

    # Check if it's a local file
    if os.path.exists(input_path):
        logger.info(f"Detected local file: {input_path}")
        return 'file', False

    # If it's not a recognizable YouTube pattern and file doesn't exist
    # Check if it might be a relative path
    cwd_path = os.path.join(os.getcwd(), input_path)
    if os.path.exists(cwd_path):
        logger.info(f"Detected local file (relative path): {cwd_path}")
        return 'file', False

    # If nothing matches, assume it's a YouTube URL or ID that might be valid
    logger.warning(f"Could not definitively detect input type for '{input_path}'. Assuming YouTube ID/URL.")
    return 'youtube', True


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files or YouTube videos to markdown using mlx-audio (Parakeet v3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input",
        help="Input source: YouTube URL, YouTube ID, or path to local audio/video file"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model to use for transcription. Accepts full HuggingFace IDs or short aliases: "
            + ", ".join(f"{k} -> {v}" for k, v in MODEL_ALIASES.items())
        )
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=os.getcwd(),
        help="Directory to save output files"
    )

    parser.add_argument(
        "--keep-audio", "-k",
        action="store_true",
        help="Keep downloaded and converted audio files"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--format", "-f",
        choices=SUPPORTED_FORMATS,
        default=DEFAULT_FORMAT,
        help="Output format: md (markdown with timestamps, default), srt (subtitles), txt (plain text)"
    )

    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=30.0,
        help="Duration in seconds of each audio chunk for long-form transcription (default: 30.0)"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Auto-detect input type
    input_type, is_youtube = auto_detect_input(args.input)

    # Create temporary directory for audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Process input source based on detected type
            metadata = None
            if is_youtube:
                audio_file, metadata = download_youtube_audio(args.input, temp_dir if not args.keep_audio else args.output_dir)
                video_id = metadata.get('video_id') or extract_video_id(args.input)
                title = metadata.get('title', video_id)
            else:
                audio_file, title = process_input_file(args.input, temp_dir if not args.keep_audio else args.output_dir)
                video_id = None

            # Convert audio to 16kHz mono WAV format
            whisper_audio = convert_audio_for_whisper(audio_file, temp_dir)

            # Transcribe
            output_file = transcribe(
                whisper_audio,
                args.model,
                args.output_dir,
                title,
                video_id,
                args.chunk_duration,
                args.format,
                metadata,
            )

            logger.info(f"Transcription completed successfully. Output saved to: {output_file}")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
