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
import tempfile
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

import typer
from typing_extensions import Annotated

from md_common import build_frontmatter  # noqa: F401 — re-exported for backward compat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
AUDIO_SAMPLE_RATE = 16000  # 16kHz as required by Parakeet
MODELS_DIR = "mlx_models"  # Directory for cached models

# Short alias -> full HuggingFace model ID
MODEL_ALIASES = {
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
    "parakeet-v2": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-1.1b": "mlx-community/parakeet-tdt-1.1b",
    "parakeet-ctc": "mlx-community/parakeet-ctc-0.6b",
}

SUPPORTED_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-1.1b",
    "mlx-community/parakeet-ctc-0.6b",
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
    Convert audio file to the format required by Parakeet (16kHz, mono, WAV).

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


DEFAULT_DIARIZE_MODEL = "mlx-community/diar_sortformer_4spk-v1-fp32"


def load_diarization_model(model_name: str = DEFAULT_DIARIZE_MODEL):
    """Load Sortformer diarization model via mlx_audio.vad.load()."""
    try:
        from mlx_audio.vad import load
    except ImportError:
        logger.error("mlx-audio VAD module required. Install with: pip install mlx-audio[stt]")
        raise
    logger.info("Loading diarization model: %s", model_name)
    return load(model_name)


def diarize(audio_path: str, model):
    """Run speaker diarization on audio file. Returns DiarizationOutput."""
    logger.info("Running diarization on: %s", audio_path)
    start = time.time()
    result = model.generate(audio_path)
    logger.info("Diarization completed in %.2f seconds", time.time() - start)
    return result


def align_speakers(transcription_segments: List, diarization_segments: List) -> List[Dict]:
    """Align transcription segments with diarization speaker labels.

    For each transcription segment, finds the diarization speaker with the most
    temporal overlap and assigns that speaker. Then merges consecutive segments
    from the same speaker into speaker turns.

    Args:
        transcription_segments: List of AlignedSentence objects or dicts with start/end/text
        diarization_segments: List of DiarizationSegment objects with start/end/speaker

    Returns:
        List of dicts: {start, end, text, speaker}
    """
    if not transcription_segments:
        return []

    aligned = []
    for seg in transcription_segments:
        start_t, end_t, text = _extract_sentence_fields(seg)
        if start_t is None:
            continue

        # Find speaker with most overlap
        speaker = 0  # fallback
        best_overlap = 0.0
        for dseg in diarization_segments:
            overlap_start = max(start_t, dseg.start)
            overlap_end = min(end_t, dseg.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = dseg.speaker

        aligned.append({
            "start": start_t,
            "end": end_t,
            "text": text.strip(),
            "speaker": speaker,
        })

    # Merge consecutive segments from the same speaker
    if not aligned:
        return []

    merged = [aligned[0].copy()]
    for seg in aligned[1:]:
        if seg["speaker"] == merged[-1]["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg.copy())

    return merged


def segments_to_markdown_diarized(segments: List[Dict], title: Optional[str] = None,
                                   metadata: Optional[Dict] = None) -> str:
    """Convert diarized segments to markdown with speaker headers.

    Format:
        **SPEAKER_0** [00:00:00]

        text here...

        **SPEAKER_1** [00:05:30]

        other text...
    """
    lines = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append("")

    if title:
        lines.append(f"# {title}")
        lines.append("")

    for seg in segments:
        ts = format_timestamp_md(seg["start"])
        lines.append(f"**SPEAKER_{seg['speaker']}** [{ts}]")
        lines.append("")
        lines.append(seg["text"])
        lines.append("")

    return "\n".join(lines)


def segments_to_srt_diarized(segments: List[Dict]) -> str:
    """Convert diarized segments to SRT with speaker prefix."""
    srt_lines = []
    for i, seg in enumerate(segments):
        srt_lines.append(f"{i+1}")
        srt_lines.append(
            f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}"
        )
        srt_lines.append(f"[SPEAKER_{seg['speaker']}] {seg['text']}")
        if i < len(segments) - 1:
            srt_lines.append("")
    return "\n".join(srt_lines)


def segments_to_text_diarized(segments: List[Dict]) -> str:
    """Convert diarized segments to plain text with speaker prefix."""
    parts = []
    for seg in segments:
        parts.append(f"SPEAKER_{seg['speaker']}: {seg['text']}")
    return "\n\n".join(parts)


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
    diarize_model_name: Optional[str] = None,
) -> str:
    """
    Transcribe audio file using mlx-audio (Parakeet).

    Args:
        audio_file: Path to the audio file
        model_name: HuggingFace model ID or alias to use
        output_dir: Directory to save the output file (default: same as audio file)
        video_title: Title of the video (for naming the output file)
        video_id: YouTube video ID (for naming the output file)
        chunk_duration: Duration of each audio chunk in seconds (default: 30.0)
        output_format: Output format — "md" (default), "srt", or "txt"
        metadata: Optional metadata dict for YAML frontmatter (markdown only)
        diarize_model_name: If set, run speaker diarization with this Sortformer model

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

    # Optional diarization
    diarized_segments = None
    if diarize_model_name:
        diar_model = load_diarization_model(diarize_model_name)
        diar_output = diarize(audio_file, diar_model)
        diarized_segments = align_speakers(result.sentences, diar_output.segments)
        num_speakers = diar_output.num_speakers or len({s.speaker for s in diar_output.segments})
        if metadata is not None:
            metadata["speakers"] = num_speakers
        logger.info("Detected %d speaker(s)", num_speakers)

    # Format output
    if diarized_segments is not None:
        display_title = video_title or os.path.splitext(os.path.basename(audio_file))[0]
        if output_format == "md":
            content = segments_to_markdown_diarized(diarized_segments, title=display_title, metadata=metadata)
        elif output_format == "txt":
            content = segments_to_text_diarized(diarized_segments)
        else:
            content = segments_to_srt_diarized(diarized_segments)
    elif output_format == "md":
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


class OutputFormat(str, Enum):
    md = "md"
    srt = "srt"
    txt = "txt"


# Build model help text
_model_help_lines = "\n".join(f"  {alias} → {full}" for alias, full in MODEL_ALIASES.items())
_model_help = (
    "Model for transcription. Accepts a full HuggingFace ID or a short alias.\n\n"
    "Aliases:\n" + _model_help_lines
)

app = typer.Typer(
    help="Transcribe YouTube videos, audio, or video files to markdown, SRT, or plain text using mlx-audio (Parakeet).",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input: Annotated[str, typer.Argument(
        help="YouTube URL, video ID (11 chars), or local file path. [dim]Tip: use the video ID or quote URLs to avoid shell glob issues with ? and &.[/dim]",
    )],
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help=_model_help,
    )] = DEFAULT_MODEL,
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path.cwd(),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown + YAML frontmatter), [bold]srt[/bold] (subtitles), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    chunk_duration: Annotated[float, typer.Option(
        "--chunk-duration", "-c",
        help="Chunk length in seconds for long audio. mlx-audio splits internally.",
    )] = 30.0,
    keep_audio: Annotated[bool, typer.Option(
        "--keep-audio", "-k",
        help="Keep downloaded/converted audio files instead of cleaning up.",
    )] = False,
    diarize_flag: Annotated[bool, typer.Option(
        "--diarize/--no-diarize",
        help="Enable speaker diarization (identifies who is speaking).",
    )] = False,
    diarize_model: Annotated[str, typer.Option(
        "--diarize-model",
        help="Sortformer diarization model ID.",
    )] = DEFAULT_DIARIZE_MODEL,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
):
    """
    Transcribe audio to markdown (default), SRT, or plain text.

    Accepts a YouTube URL, a YouTube video ID (11 chars), or a path to a local
    audio/video file. YouTube sources get full YAML frontmatter in markdown output.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Auto-detect input type
    _input_type, is_youtube = auto_detect_input(input)

    output_dir_str = str(output_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            metadata = None
            if is_youtube:
                audio_file, metadata = download_youtube_audio(
                    input, temp_dir if not keep_audio else output_dir_str
                )
                video_id = metadata.get('video_id') or extract_video_id(input)
                title = metadata.get('title', video_id)
            else:
                audio_file, title = process_input_file(
                    input, temp_dir if not keep_audio else output_dir_str
                )
                video_id = None

            whisper_audio = convert_audio_for_whisper(audio_file, temp_dir)

            output_file = transcribe(
                whisper_audio,
                model,
                output_dir_str,
                title,
                video_id,
                chunk_duration,
                format.value,
                metadata,
                diarize_model_name=diarize_model if diarize_flag else None,
            )

            logger.info(f"Transcription completed successfully. Output saved to: {output_file}")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
