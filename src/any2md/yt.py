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
from typing import Any, Dict, Optional, Tuple, List
import logging

import typer
from typing_extensions import Annotated

from any2md.common import (  # noqa: F401 — build_frontmatter re-exported for backward compat
    build_frontmatter,
    configure_mlx_memory,
    is_json_mode,
    release_mlx_memory,
    write_json_output,
)

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
    "qwen3-asr": "mlx-community/Qwen3-ASR-1.7B-bf16",
    "qwen3-asr-bf16": "mlx-community/Qwen3-ASR-1.7B-bf16",
    "qwen3-asr-8bit": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "qwen3-asr-4bit": "mlx-community/Qwen3-ASR-1.7B-4bit",
}

SUPPORTED_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-1.1b",
    "mlx-community/parakeet-ctc-0.6b",
    "mlx-community/Qwen3-ASR-1.7B-bf16",
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-ASR-1.7B-4bit",
] + list(MODEL_ALIASES.keys())

# ---------------------------------------------------------------------------
# Qwen3-ForcedAligner constants, loader, and phrasifier
# ---------------------------------------------------------------------------

DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

# Languages supported by Qwen3-ForcedAligner (ISO 639-1 codes, full names, and
# regional variants that the model accepts).  Any language NOT in this set will
# fall back to coarse Qwen3-ASR segment-level timestamps with a logged warning.
ALIGNER_LANGS = {
    "zh", "chinese",
    "en", "english",
    "yue", "cantonese",
    "ja", "japanese",
    "ko", "korean",
    "ar", "arabic",
    "fr", "french",
    "de", "german",
    "es", "spanish",
    "pt", "portuguese",
}

# Qwen3-ASR and Qwen3-ForcedAligner expect language *names* ("Chinese"), while
# LID, YouTube metadata and --language usually give ISO codes ("zh", "en-US").
# Unknown strings are passed through verbatim (the aligner would then tokenize
# Chinese as a space-separated language), so normalize codes first.
QWEN3_LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "tl": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian",
    "mk": "Macedonian",
}


def _qwen3_language_name(language: str) -> str:
    """Map an ISO code like 'zh' or 'en-US' to the name Qwen3 models expect."""
    code = language.strip().lower()
    return QWEN3_LANGUAGE_NAMES.get(code) or QWEN3_LANGUAGE_NAMES.get(code.split("-")[0], language)


# (model_name, model) of the loaded aligner; released by unload_forced_aligner()
_aligner_model_cache: Optional[Tuple[str, Any]] = None


def load_forced_aligner(model_name: str = DEFAULT_ALIGNER_MODEL):
    """Lazy-load and cache the Qwen3-ForcedAligner model.

    mlx-audio's load_model() may not dispatch correctly on hyphenated HF repo
    names (ANY2-33 hit a similar issue).  We try load_model first; on
    ValueError/KeyError we fall back to snapshot_download + retry.

    Returns the loaded model, or None on any failure (callers must fall back
    gracefully to coarse segments in that case). Call unload_forced_aligner()
    when done so the weights don't stay resident for the rest of the process.
    """
    global _aligner_model_cache
    if _aligner_model_cache is not None and _aligner_model_cache[0] == model_name:
        return _aligner_model_cache[1]
    unload_forced_aligner()

    try:
        from mlx_audio.stt import load_model
    except ImportError:
        logger.warning("mlx-audio not available — forced aligner disabled")
        return None

    try:
        m = load_model(model_name)
        _aligner_model_cache = (model_name, m)
        return m
    except (ValueError, KeyError) as e:
        logger.warning(
            "Aligner load_model('%s') failed (%s); attempting snapshot_download fallback",
            model_name, e,
        )
    except Exception as e:
        logger.error("Aligner load failed unexpectedly (%s); skipping forced alignment", e)
        return None

    # Fallback: download via huggingface_hub and retry
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id=model_name)
        try:
            m = load_model(local_path)
            _aligner_model_cache = (model_name, m)
            return m
        except Exception as e2:
            logger.error(
                "Aligner instantiation failed even with local path '%s': %s — skipping",
                local_path, e2,
            )
            return None
    except Exception as e:
        logger.error("snapshot_download for aligner failed: %s — skipping", e)
        return None


def unload_forced_aligner() -> None:
    """Drop the cached aligner model and return its memory."""
    global _aligner_model_cache
    if _aligner_model_cache is not None:
        _aligner_model_cache = None
        release_mlx_memory()


def _load_wav_16k(audio_path: str):
    """Read audio as a mono float32 numpy array at 16 kHz."""
    import numpy as np
    try:
        from mlx_audio.audio_io import read as audio_read
        samples, _sr = audio_read(audio_path, dtype="float32", sample_rate=AUDIO_SAMPLE_RATE, nchannels=1)
        return np.asarray(samples, dtype=np.float32).reshape(-1)
    except ImportError:
        from mlx_audio.stt.utils import load_audio
        return np.array(load_audio(audio_path, sr=AUDIO_SAMPLE_RATE), dtype=np.float32).reshape(-1)


def _align_words_per_segment(aligner, audio_path: str, segments, language: str) -> list:
    """Word-align an ASR transcript one segment at a time.

    Qwen3-ForcedAligner aligns its whole input in a single forward pass and is
    trained on clips of up to ~5 minutes. Handing it a full recording plus its
    full transcript needs memory that grows with the recording length (and
    degrades alignment past 5 minutes). ASR segments are at most
    --chunk-duration long (30 s by default), so aligning them one by one keeps
    memory flat. Word timestamps are shifted back onto the recording timeline.
    """
    wav = _load_wav_16k(audio_path)
    words = []
    for seg in segments:
        start, end, text = _extract_sentence_fields(seg)
        if start is None or not (text or "").strip():
            continue
        clip = wav[int(start * AUDIO_SAMPLE_RATE):int(end * AUDIO_SAMPLE_RATE)]
        if clip.size == 0:
            continue
        result = aligner.generate(clip, text=text.strip(), language=language)
        for item in getattr(result, "items", result):
            item_start, item_end, item_text = _word_fields(item)
            if item_start is None or item_end is None:
                continue
            words.append({"start": start + item_start, "end": start + item_end, "text": item_text})
    return words


def _word_fields(w) -> tuple:
    """(start, end, text) of an aligner word: ForcedAlignItem, object, or dict."""
    if hasattr(w, "start_time"):  # mlx-audio ForcedAlignItem
        return w.start_time, w.end_time, w.text
    if hasattr(w, "start"):
        return w.start, w.end, w.text
    if isinstance(w, dict):
        return (
            w.get("start", w.get("start_time")),
            w.get("end", w.get("end_time")),
            w.get("text", ""),
        )
    return None, None, ""


def _phrasify_word_alignment(words, max_phrase_s: float = 15.0, gap_threshold_s: float = 0.4):
    """Group word-level aligner output into ~5–15s phrases.

    Breaks on:
      - Sentence-ending punctuation (CJK + Latin)
      - Silence gaps > gap_threshold_s between consecutive words
      - Phrase duration exceeding max_phrase_s

    Args:
        words: List of word dicts (keys: "start", "end", "text"), mlx-audio
               ForcedAlignItem objects (.start_time / .end_time / .text), or
               objects with .start / .end / .text attributes.
        max_phrase_s: Maximum phrase duration in seconds (default: 15.0).
        gap_threshold_s: Minimum silence gap that triggers a phrase break (default: 0.4s).

    Returns:
        List of dicts with "start", "end", "text" keys — compatible with
        _extract_sentence_fields() and align_speakers().
    """
    if not words:
        return []

    # CJK sentence-end + Latin sentence-end + comma as softer fallback
    PUNCT = set("。！？.!?")

    phrases = []
    cur: Dict = {"start": None, "end": None, "text": []}

    def flush():
        nonlocal cur
        if cur["start"] is not None and cur["text"]:
            phrases.append({
                "start": cur["start"],
                "end": cur["end"],
                "text": " ".join(cur["text"]).strip(),
            })
        cur = {"start": None, "end": None, "text": []}

    prev_end = None
    for w in words:
        ws, we, wt = _word_fields(w)

        if ws is None or we is None:
            continue

        if cur["start"] is None:
            cur["start"] = ws

        gap = (ws - prev_end) if prev_end is not None else 0.0
        cur_dur = we - cur["start"]

        if gap > gap_threshold_s or cur_dur > max_phrase_s:
            flush()
            cur["start"] = ws

        cur["end"] = we
        cur["text"].append(wt)

        if wt and wt[-1] in PUNCT:
            flush()

        prev_end = we

    flush()
    return phrases


def resolve_model(model_name: str) -> str:
    """Resolve a model alias or short name to its full HuggingFace model ID."""
    return MODEL_ALIASES.get(model_name, model_name)


# ---------------------------------------------------------------------------
# Language Identification (LID) constants, loader, and detector
# ---------------------------------------------------------------------------

LID_MIN_AUDIO_DURATION_S = 5.0        # skip LID below this; default to Parakeet
LID_SAMPLE_WINDOW_S = 30.0            # Whisper's native window (universal in production)
LID_CONFIDENCE_THRESHOLD = 0.5        # from faster-whisper's production default
LID_MAJORITY_VOTE_SEGMENTS = 3        # max segments for majority-vote (future use)
PARAKEET_LANGS = {"en", "english", "eng"}  # ISO 639-1, full name, ISO 639-2
# whisper-tiny-asr-* repackages the same whisper-tiny weights as the old
# whisper-tiny-mlx port but ships preprocessor_config.json/tokenizer.json, so
# no openai/whisper-tiny processor fallback is needed.
DEFAULT_LID_MODEL = "mlx-community/whisper-tiny-asr-fp16"

_lid_model_cache = None  # module-level singleton


def _audio_duration_s(wav_path: str) -> float:
    """Return duration of a WAV file in seconds using stdlib wave."""
    import wave as _wave
    try:
        with _wave.open(wav_path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def _extract_lid_window(wav_path: str, seconds: float) -> str:
    """Write the first *seconds* of a 16 kHz mono WAV to a temp file and return its path.

    LID only needs Whisper's 30 s window; slicing here keeps mel computation
    O(1) in recording length instead of running over an hours-long file.
    Falls back to the original path if the WAV can't be read with stdlib wave.
    """
    import shutil
    import tempfile
    import wave as _wave
    try:
        with _wave.open(wav_path, "rb") as src:
            frames = min(src.getnframes(), int(seconds * src.getframerate()))
            params = src.getparams()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            with _wave.open(tmp_path, "wb") as dst:
                dst.setparams(params)
                dst.writeframes(src.readframes(frames))
        return tmp_path
    except Exception:
        return wav_path


def _load_lid_model(model_name: str = DEFAULT_LID_MODEL):
    """Lazy-load and cache the LID (Whisper-tiny) model.

    Returns None if mlx-audio or transformers are unavailable, so callers
    can fall back to Parakeet without crashing.
    """
    global _lid_model_cache
    if _lid_model_cache is None:
        try:
            from mlx_audio.stt import load_model
        except ImportError:
            logger.warning("mlx-audio not available — LID disabled; defaulting to Parakeet")
            return None
        try:
            from transformers import WhisperProcessor
        except ImportError:
            logger.warning("transformers not available — LID disabled; defaulting to Parakeet")
            return None
        try:
            m = load_model(model_name)
            # Fallback patch: mlx-community Whisper ports may ship without preprocessor_config.json.
            # post_load_hook already tries this, but guard against None in case it failed.
            if getattr(m, "_processor", None) is None:
                try:
                    m._processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                except Exception as e:
                    logger.warning("Could not load WhisperProcessor: %s — LID may be unreliable", e)
            _lid_model_cache = m
        except Exception as e:
            logger.warning("Failed to load LID model %s: %s — defaulting to Parakeet", model_name, e)
            return None
    return _lid_model_cache


def _detect_language(wav_path: str) -> tuple:
    """Detect the spoken language in a WAV file.

    Returns (lang_code, confidence) where lang_code is a two-letter ISO 639-1
    code (e.g. 'en', 'zh') and confidence is in [0.0, 1.0].

    Defaults to ('en', 0.0) on any error, short clip, or silent/empty audio.
    Whisper internally pads audio to 30s — LID_SAMPLE_WINDOW_S is the window.

    v1: low-confidence → Parakeet (en) as safe fallback.
    Majority-vote across multiple 30s windows is documented as future work.
    """
    try:
        duration = _audio_duration_s(wav_path)
        if duration < LID_MIN_AUDIO_DURATION_S:
            logger.info(
                "Audio %.1fs < %.1fs minimum — skipping LID, defaulting to Parakeet (en)",
                duration, LID_MIN_AUDIO_DURATION_S,
            )
            return "en", 0.0

        model = _load_lid_model()
        if model is None:
            return "en", 0.0

        # Prepare mel spectrogram from the WAV path (uses model._prepare_audio internally)
        try:
            from mlx_audio.stt.models.whisper.audio import pad_or_trim, N_FRAMES
        except ImportError:
            try:
                from mlx_audio.stt.models.whisper.whisper import N_FRAMES
                from mlx_audio.stt.models.whisper.audio import pad_or_trim
            except ImportError:
                logger.warning("Cannot import Whisper audio utils — LID disabled")
                return "en", 0.0

        mel, _content_frames = model._prepare_audio(_extract_lid_window(wav_path, LID_SAMPLE_WINDOW_S + 1.0))
        # Trim/pad to a single 30s window (Whisper's native LID window)
        mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(model.dtype)

        try:
            _token, probs = model.detect_language(mel_segment)
        except Exception as e:
            logger.warning("LID detection failed: %s — defaulting to Parakeet (en)", e)
            return "en", 0.0

        # probs is a dict mapping language codes (e.g. "en", "zh") to float probabilities
        if not isinstance(probs, dict) or not probs:
            logger.warning("Unexpected LID result shape: %r — defaulting to Parakeet", type(probs))
            return "en", 0.0

        top_lang = max(probs, key=probs.get)
        top_prob = probs[top_lang]

        if top_prob >= LID_CONFIDENCE_THRESHOLD:
            logger.info("Detected language: %s (conf=%.2f)", top_lang, top_prob)
            return top_lang, top_prob

        # Low confidence — fall back to Parakeet (en)
        logger.warning(
            "LID confidence %.2f below threshold %.2f — defaulting to Parakeet (en)",
            top_prob, LID_CONFIDENCE_THRESHOLD,
        )
        return "en", top_prob

    except Exception as e:
        logger.warning("LID error: %s — defaulting to Parakeet (en)", e)
        return "en", 0.0


def _resolve_language_and_model(
    audio_path: str,
    lang_flag: Optional[str],
    model_flag: Optional[str],
    yt_metadata: Optional[Dict],
) -> tuple:
    """Resolve (language, model) from CLI flags + LID + YouTube metadata.

    Priority:
      1. Explicit --language (non-None, non-'auto') → use as-is, no LID
      2. YouTube metadata language field → use as-is, no LID
      3. Audio LID → detect from audio; low-conf/short → default 'en'

    Then:
      - Explicit --model (non-None) wins unconditionally
      - lang in PARAKEET_LANGS → DEFAULT_MODEL (Parakeet)
      - otherwise → 'qwen3-asr' (bf16 alias from ANY2-30)

    Note: model_flag=None means "user did not pass --model"; model_flag=DEFAULT_MODEL
    means the user explicitly passed --model parakeet-v3 (or the full ID). Using None
    as the sentinel is necessary to distinguish "default" from "explicit choice" because
    DEFAULT_MODEL is also a valid routing target for non-English languages.
    """
    # 1. Determine language
    if lang_flag and lang_flag.lower() != "auto":
        lang = lang_flag
    elif yt_metadata and yt_metadata.get("language"):
        lang = yt_metadata["language"]
        logger.info("Using YouTube metadata language: %s", lang)
    else:
        lang, conf = _detect_language(audio_path)
        # _detect_language already logged the result

    # 2. Determine model
    if model_flag is not None:
        # User explicitly specified --model — respect it unconditionally
        chosen_model = resolve_model(model_flag)
    elif lang.lower() in PARAKEET_LANGS:
        chosen_model = DEFAULT_MODEL  # Parakeet
    else:
        chosen_model = "qwen3-asr"   # → bf16 alias from ANY2-30

    if model_flag is None and chosen_model != DEFAULT_MODEL:
        logger.info("Auto-routing language '%s' → model %s", lang, chosen_model)

    return lang, chosen_model


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
        logger.error("yt-dlp is required. Install it with: uv pip install 'mlx-audio[stt]' yt-dlp")
        raise

    video_id = extract_video_id(url_or_id)

    # Use temporary directory if none specified
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output filename template
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    # Configure yt-dlp options — download best audio in its native format.
    # No mp3 postprocessor: convert_audio_for_whisper() handles the single
    # ffmpeg call to 16kHz mono WAV directly from any container format.
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
    }

    logger.info(f"Downloading audio from YouTube video: {video_id}")

    # Download the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

    # yt-dlp downloads in native format (webm/m4a/opus/etc) — find the actual file
    import glob
    candidates = glob.glob(os.path.join(output_dir, f"{video_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"Failed to download audio for video: {video_id}")
    audio_file = candidates[0]

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
    output_file = os.path.join(output_dir, f"{base_name}.wav")

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


def _get_segments(result) -> list:
    """Extract segment list from an STT result object.

    Handles both:
      - Parakeet: result.sentences (list of AlignedSentence objects)
      - Qwen3-ASR: result.segments (list of dicts with start/end/text keys)

    Order is load-bearing: check sentences FIRST. Parakeet's STTOutput also has
    a .segments field (base class attribute) that may be None or empty, so
    checking sentences first ensures Parakeet always uses the correct attribute.
    """
    return getattr(result, 'sentences', None) or getattr(result, 'segments', None) or []


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


DEFAULT_DIARIZE_MODEL = "mlx-community/Nemotron-3-Diarization"
DEFAULT_DIARIZE_CHUNK_DURATION = 10.0  # seconds per chunk; only used for Sortformer models

# Legacy diarization models (kept working; the pre-0.3.5 default was diar_sortformer_4spk-v1-fp32)
LEGACY_DIARIZE_MODELS = (
    "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
    "mlx-community/diar_sortformer_4spk-v1-fp32",
    "mlx-community/diar_sortformer_4spk-v1-fp16",
)


def _is_nemotron_diar_model(model_name: str) -> bool:
    """Return True for Nemotron-3-Diarization models.

    Nemotron's ``generate()`` processes the clip through bounded internal
    windows (NVIDIA's offline streaming preset) with an Arrival-Order Speaker
    Cache, so the full-clip call is safe for recordings of any length and
    keeps stable speaker IDs.
    """
    name = (model_name or "").lower()
    return "nemotron" in name and "diar" in name


def _is_sortformer_diar_model(model_name: str) -> bool:
    """Return True for Sortformer diarization models.

    Sortformer's full-clip ``generate()`` runs O(n²) self-attention over the
    whole recording — an 82-minute clip stalls growing past 20 GB. Those
    models must be driven through ``generate_stream(chunk_duration=...)``,
    which processes fixed-size windows while the model's internal
    Arrival-Order Speaker Cache keeps memory bounded AND speaker IDs stable
    across chunks (mlx-audio >= 0.5.6 threads the streaming state correctly).
    """
    name = (model_name or "").lower()
    return "sortformer" in name


def load_diarization_model(model_name: str = DEFAULT_DIARIZE_MODEL):
    """Load a diarization model (Nemotron-3-Diarization or Sortformer) via mlx_audio.vad.load()."""
    try:
        from mlx_audio.vad import load
    except ImportError:
        logger.error("mlx-audio VAD module required. Install with: uv pip install 'mlx-audio[stt]' yt-dlp")
        raise
    logger.info("Loading diarization model: %s", model_name)
    model = load(model_name)
    # Stash so diarize() can dispatch on the family without us having to
    # plumb model_name through every call site.
    setattr(model, "_any2md_model_name", model_name)
    return model


def diarize(audio_path: str, model, chunk_duration: float = DEFAULT_DIARIZE_CHUNK_DURATION):
    """Run speaker diarization on audio file with memory bounded by design.

    Dispatch on model family:

    - Nemotron-3-Diarization: ``generate()`` processes the clip through
      bounded internal windows with an Arrival-Order Speaker Cache — safe for
      recordings of any length, stable speaker IDs, up to 8 speakers.
    - Sortformer (streaming v2.1 or offline v1): the full-clip ``generate()``
      runs O(n²) self-attention and OOMs/stalls on long audio, so drive
      ``generate_stream(chunk_duration=...)`` instead and merge the per-chunk
      segments. mlx-audio (>= 0.5.6) threads the streaming state across
      chunks, so speaker IDs stay stable.

    Args:
        audio_path: Path to the audio file.
        model: Loaded diarization model from `load_diarization_model`.
        chunk_duration: Chunk size for the Sortformer streaming path.

    Returns:
        DiarizationOutput with merged speaker segments.
    """
    model_name = getattr(model, "_any2md_model_name", "")
    if not isinstance(model_name, str):
        model_name = ""

    if _is_nemotron_diar_model(model_name):
        logger.info("Running diarization on: %s (Nemotron-3, bounded streaming windows)", audio_path)
        start = time.time()
        result = model.generate(audio_path)
        elapsed = time.time() - start
        num_speakers = result.num_speakers or len({s.speaker for s in result.segments})
        logger.info(
            "Diarization completed in %.2f seconds (%d segments, %d speakers)",
            elapsed, len(result.segments), num_speakers,
        )
        return result

    logger.info("Running diarization on: %s (Sortformer streaming, chunk_duration=%.1fs)", audio_path, chunk_duration)
    start = time.time()

    all_segments = []
    num_chunks = 0
    for chunk_result in model.generate_stream(audio_path, chunk_duration=chunk_duration):
        all_segments.extend(chunk_result.segments)
        num_chunks += 1

    # Merge adjacent segments from the same speaker across chunk boundaries
    merged = _merge_diarization_segments(all_segments)
    num_speakers = len({s.speaker for s in merged})

    elapsed = time.time() - start
    logger.info(
        "Diarization completed in %.2f seconds (%d chunks, %d segments, %d speakers)",
        elapsed, num_chunks, len(merged), num_speakers,
    )

    # Import here to avoid top-level dependency
    from mlx_audio.vad.models.sortformer.sortformer import DiarizationOutput

    return DiarizationOutput(
        segments=merged,
        num_speakers=num_speakers,
        total_time=elapsed,
    )


def _merge_diarization_segments(segments, merge_gap: float = 0.3):
    """Merge adjacent diarization segments from the same speaker.

    Streaming chunks may produce back-to-back segments from the same speaker
    at chunk boundaries. This merges them if the gap is small enough.

    Args:
        segments: List of DiarizationSegment objects.
        merge_gap: Maximum gap (seconds) between segments to merge.

    Returns:
        List of merged DiarizationSegment objects.
    """
    if not segments:
        return []

    from mlx_audio.vad.models.sortformer.sortformer import DiarizationSegment

    # Sort by start time to handle interleaved chunk results
    sorted_segs = sorted(segments, key=lambda s: s.start)

    merged = [DiarizationSegment(start=sorted_segs[0].start, end=sorted_segs[0].end, speaker=sorted_segs[0].speaker)]
    for seg in sorted_segs[1:]:
        prev = merged[-1]
        if seg.speaker == prev.speaker and (seg.start - prev.end) <= merge_gap:
            merged[-1] = DiarizationSegment(start=prev.start, end=max(prev.end, seg.end), speaker=prev.speaker)
        else:
            merged.append(DiarizationSegment(start=seg.start, end=seg.end, speaker=seg.speaker))

    return merged


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
                                   metadata: Optional[Dict] = None,
                                   speaker_map: Optional[Dict] = None) -> str:
    """Convert diarized segments to markdown with speaker headers.

    Format (no identification):
        **SPEAKER_0** [00:00:00]

        text here...

    Format (with identification, high-conf match):
        **Alice** [00:00:00]

        text here...

    Format (with identification, medium-conf match):
        **Joe** (0.74) [00:24]

        text here...

    Args:
        segments: List of diarized segment dicts with 'start', 'end', 'speaker', 'text'.
        title: Optional title for the markdown header.
        metadata: Optional metadata dict for YAML frontmatter.
        speaker_map: Optional dict mapping speaker labels (e.g. 'SPEAKER_0') to
                     identification result dicts (as returned by identify_speakers()).
                     If None or a label is absent, falls back to 'SPEAKER_N' format.
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
        raw_label = seg["speaker"]

        # Build display name from speaker_map if available
        if speaker_map and raw_label in speaker_map:
            info = speaker_map[raw_label]
            if info.get("matched"):
                name = info["name"]
                if not info.get("high_conf") and info.get("distance") is not None:
                    # Medium confidence: show score
                    conf_score = round(1.0 - info["distance"], 2)
                    display = f"**{name}** ({conf_score:.2f}) [{ts}]"
                else:
                    display = f"**{name}** [{ts}]"
            else:
                # Unmatched — keep original label
                display = f"**{raw_label}** [{ts}]"
        else:
            # No speaker_map at all — legacy SPEAKER_N format
            display = f"**SPEAKER_{raw_label}** [{ts}]"

        lines.append(display)
        lines.append("")
        lines.append(seg["text"])
        lines.append("")

    return "\n".join(lines)


def segments_to_srt_diarized(segments: List[Dict], speaker_map: Optional[Dict] = None) -> str:
    """Convert diarized segments to SRT with speaker prefix.

    Args:
        segments: List of diarized segment dicts.
        speaker_map: Optional identification map from identify_speakers().
    """
    srt_lines = []
    for i, seg in enumerate(segments):
        srt_lines.append(f"{i+1}")
        srt_lines.append(
            f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}"
        )
        raw_label = seg["speaker"]
        if speaker_map and raw_label in speaker_map and speaker_map[raw_label].get("matched"):
            display = speaker_map[raw_label]["name"]
        else:
            display = f"SPEAKER_{raw_label}"
        srt_lines.append(f"[{display}] {seg['text']}")
        if i < len(segments) - 1:
            srt_lines.append("")
    return "\n".join(srt_lines)


def segments_to_text_diarized(segments: List[Dict], speaker_map: Optional[Dict] = None) -> str:
    """Convert diarized segments to plain text with speaker prefix.

    Args:
        segments: List of diarized segment dicts.
        speaker_map: Optional identification map from identify_speakers().
    """
    parts = []
    for seg in segments:
        raw_label = seg["speaker"]
        if speaker_map and raw_label in speaker_map and speaker_map[raw_label].get("matched"):
            display = speaker_map[raw_label]["name"]
        else:
            display = f"SPEAKER_{raw_label}"
        parts.append(f"{display}: {seg['text']}")
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
    identify: bool = False,
    audio_for_speaker: Optional[str] = None,
    auto_enroll: bool = False,
    no_enroll: bool = False,
    _unmatched_out: Optional[List] = None,
    speaker_names: Optional[List[str]] = None,
    language: Optional[str] = None,
    aligner_model: str = DEFAULT_ALIGNER_MODEL,
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
        identify: If True and diarization was run, extract WeSpeaker embeddings per segment
        audio_for_speaker: Path to 16kHz mono WAV to use for speaker embedding extraction;
            defaults to audio_file if identify is True and this is None
        auto_enroll: If True, auto-enroll unmatched speakers as Unknown_N without prompting.
            Mutually exclusive with interactive prompting.
        no_enroll: If True, skip enrollment prompts entirely and leave unmatched speakers
            as SPEAKER_N labels (current default for non-TTY environments).
        _unmatched_out: Optional mutable list; if provided, unmatched speaker dicts (with
            label, embedding, segments) are appended to it for the caller to use (e.g. JSON
            output). Internal use by main() only.
        speaker_names: Optional list of speaker names to restrict catalog matching to.
            Passed to identify_speakers(speaker_names=...). When None, all catalog
            speakers are searched.
        language: Optional language hint for STT (e.g. "Chinese", "English"). When set,
            passed to model.generate(language=...). When None, model auto-detects.
            Qwen3-ASR supports: Chinese, Cantonese, English, German, Spanish, French,
            Italian, Portuguese, Russian, Korean, Japanese. Parakeet ignores this via **kwargs.
        aligner_model: HuggingFace ID for the Qwen3-ForcedAligner model used to produce
            word-level timestamps when diarizing with Qwen3-ASR over a supported language.
            Defaults to DEFAULT_ALIGNER_MODEL. Only used when diarize_model_name is set,
            model_name is a Qwen3-ASR variant, and language is in ALIGNER_LANGS.

    Returns:
        Path to the generated output file
    """
    try:
        from mlx_audio.stt import load_model as load
    except ImportError:
        logger.error("mlx-audio is required. Install it with: uv pip install 'mlx-audio[stt]' yt-dlp")
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

    configure_mlx_memory()

    # Load model
    model = load(model_name)

    # Transcribe — mlx-audio handles long audio via chunk_duration internally
    generate_kwargs: Dict = {"audio": audio_file, "chunk_duration": chunk_duration}
    if language is not None:
        generate_kwargs["language"] = language
    result = model.generate(**generate_kwargs)

    end_time = time.time()
    logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

    # The STT model is no longer needed — drop it so its weights don't sit in
    # memory while diarization/aligner/speaker models load.
    del model
    release_mlx_memory()

    # Optional diarization
    diarized_segments = None
    if diarize_model_name:
        diar_model = load_diarization_model(diarize_model_name)
        diar_output = diarize(audio_file, diar_model)
        del diar_model
        release_mlx_memory()

        # For Qwen3-ASR + supported language, use ForcedAligner for word-level timestamps
        # so that speaker boundaries within a sentence are resolved correctly.
        # (Qwen3-ASR's native segments are ~30s chunks; Sortformer is fine-grained.)
        is_qwen3 = "qwen3-asr" in (model_name or "").lower()
        aligned_segments = None
        if diarize_model_name and is_qwen3 and language and language.lower() in ALIGNER_LANGS:
            aligner = load_forced_aligner(aligner_model)
            if aligner is not None:
                try:
                    # Align one ASR segment at a time: the aligner runs a single
                    # forward pass whose memory grows with clip length, and it is
                    # only reliable up to ~5 minutes of audio.
                    words = _align_words_per_segment(
                        aligner, audio_file, _get_segments(result),
                        _qwen3_language_name(language),
                    )
                    aligned_segments = _phrasify_word_alignment(words)
                    logger.info(
                        "Forced-aligner produced %d phrases (was %d coarse segments)",
                        len(aligned_segments), len(_get_segments(result)),
                    )
                except Exception as e:
                    logger.warning(
                        "Forced-aligner failed (%s); falling back to coarse segments", e
                    )
                    aligned_segments = None
                finally:
                    unload_forced_aligner()
        elif diarize_model_name and is_qwen3:
            logger.warning(
                "Qwen3-ForcedAligner does not support %r; using coarse segment-level "
                "timestamps for diarization (speaker boundaries will be coarse)", language,
            )

        segments_for_diarization = aligned_segments if aligned_segments is not None else _get_segments(result)
        diarized_segments = align_speakers(segments_for_diarization, diar_output.segments)
        num_speakers = diar_output.num_speakers or len({s.speaker for s in diar_output.segments})
        if metadata is not None:
            metadata["speakers"] = num_speakers
        logger.info("Detected %d speaker(s)", num_speakers)

        if identify:
            try:
                from any2md.speaker import (
                    load_speaker_model,
                    extract_embeddings_for_segments,
                    identify_speakers,
                    open_catalog,
                    close_catalog,
                )
                speaker_wav = audio_for_speaker or audio_file
                logger.info("Extracting speaker embeddings via WeSpeaker (ONNX)...")
                spk_model = load_speaker_model(device="mps")
                diarized_segments = extract_embeddings_for_segments(
                    spk_model, speaker_wav, diarized_segments
                )
            except ImportError as e:
                logger.error("Speaker identification requires any2md[speaker]: %s", e)
                raise typer.Exit(code=1)

    # Speaker identification: match embeddings against catalog
    speaker_map: Optional[Dict] = None
    if identify and diarized_segments is not None:
        try:
            from any2md.speaker import (
                identify_speakers,
                open_catalog,
                close_catalog,
                add_speaker,
                enroll,
                _next_unknown_name,
            )
            logger.info("Matching speakers against catalog...")
            catalog_path = None  # Use default catalog path
            conn = open_catalog(catalog_path)
            try:
                speaker_map = identify_speakers(
                    conn, diarized_segments, audio_for_speaker or audio_file,
                    speaker_names=speaker_names,
                )

                # --- Post-identification enrollment ---
                unmatched = [
                    (label, info)
                    for label, info in speaker_map.items()
                    if not info.get("matched")
                ]

                if unmatched and not no_enroll:
                    if auto_enroll:
                        # Auto-enroll all unmatched as Unknown_N
                        for label, info in unmatched:
                            avg_emb = info.get("avg_embedding")
                            if avg_emb is None:
                                logger.warning("No embedding for %s — skipping auto-enroll", label)
                                continue
                            new_name = _next_unknown_name(conn)
                            speaker_id = add_speaker(conn, new_name)
                            enroll(
                                conn,
                                speaker_id,
                                avg_emb,
                                source_file=audio_for_speaker or audio_file,
                                source_type="auto_enroll",
                            )
                            speaker_map[label]["name"] = new_name
                            speaker_map[label]["matched"] = True
                            logger.info("Auto-enrolled %s as %r", label, new_name)

                    elif is_json_mode() and _unmatched_out is not None:
                        # JSON mode: include unmatched speakers with embeddings for caller
                        for label, info in unmatched:
                            avg_emb = info.get("avg_embedding")
                            entry: Dict = {
                                "label": label,
                                "embedding": avg_emb.tolist() if avg_emb is not None else None,
                                "segments": info.get("segments", []),
                            }
                            _unmatched_out.append(entry)

                    elif sys.stdout.isatty():
                        # Interactive TTY: prompt for each unmatched speaker
                        for label, info in unmatched:
                            avg_emb = info.get("avg_embedding")
                            segs = info.get("segments", [])
                            total_secs = sum(
                                (s.get("end") or 0) - (s.get("start") or 0) for s in segs
                            )
                            print(  # noqa: T201 — intentional interactive output
                                f"\n{label} is unknown (spoke for {total_secs:.1f}s, "
                                f"{len(segs)} segment(s)).",
                                file=sys.stderr,
                            )
                            name = typer.prompt(
                                f"Enter name to enroll {label} (or press Enter to skip)",
                                default="",
                            )
                            name = name.strip()
                            if name and avg_emb is not None:
                                speaker_id = add_speaker(conn, name)
                                enroll(
                                    conn,
                                    speaker_id,
                                    avg_emb,
                                    source_file=audio_for_speaker or audio_file,
                                    source_type="interactive_enroll",
                                )
                                speaker_map[label]["name"] = name
                                speaker_map[label]["matched"] = True
                                logger.info("Enrolled %s as %r", label, name)
                            elif name and avg_emb is None:
                                logger.warning(
                                    "No embedding available for %s — cannot enroll", label
                                )
                    # else: non-TTY, not auto_enroll, not JSON → silently leave SPEAKER_N

            finally:
                close_catalog(catalog_path)

            # Build frontmatter lists from speaker_map (after any enrollment updates)
            if metadata is not None and speaker_map:
                identified = [info["name"] for info in speaker_map.values() if info.get("matched")]
                unidentified = [
                    label for label, info in speaker_map.items() if not info.get("matched")
                ]
                if identified:
                    metadata["identified_speakers"] = sorted(set(identified))
                if unidentified:
                    metadata["unidentified_speakers"] = sorted(set(unidentified))

        except ImportError as e:
            logger.warning("Speaker catalog matching unavailable (%s); using SPEAKER_N labels", e)
            speaker_map = None

    # Format output
    if diarized_segments is not None:
        display_title = video_title or os.path.splitext(os.path.basename(audio_file))[0]
        if output_format == "md":
            content = segments_to_markdown_diarized(
                diarized_segments, title=display_title, metadata=metadata,
                speaker_map=speaker_map,
            )
        elif output_format == "txt":
            content = segments_to_text_diarized(diarized_segments, speaker_map=speaker_map)
        else:
            content = segments_to_srt_diarized(diarized_segments, speaker_map=speaker_map)
    elif output_format == "md":
        display_title = video_title or os.path.splitext(os.path.basename(audio_file))[0]
        content = segments_to_markdown(_get_segments(result), title=display_title, metadata=metadata)
    elif output_format == "txt":
        content = segments_to_text(_get_segments(result))
    else:
        content = segments_to_srt(_get_segments(result))

    # Write the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Output saved to: {output_file}")
    release_mlx_memory()
    return output_file


def process_input_file(input_file: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Process an input audio/video file.

    Returns the original file path and extracted title. No intermediate
    conversion is done — convert_audio_for_whisper() handles the single
    ffmpeg call to 16kHz mono WAV directly from any input format.

    Args:
        input_file: Path to the input audio/video file
        output_dir: Directory to save the processed audio (unused, kept for API compat)

    Returns:
        Tuple containing (audio_file_path, title)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    title = os.path.splitext(os.path.basename(input_file))[0]

    return input_file, title


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
_model_help = (
    "Model for transcription. Accepts a full HuggingFace ID or a short alias.\n\n"
    "English (fast):\n"
    "  parakeet-v3, parakeet-v2, parakeet-1.1b, parakeet-ctc\n\n"
    "Multilingual (30 languages incl. Chinese):\n"
    "  qwen3-asr (alias of qwen3-asr-bf16), qwen3-asr-bf16, qwen3-asr-8bit, qwen3-asr-4bit"
)

app = typer.Typer(
    help="Transcribe YouTube videos, audio, or video files to markdown, SRT, or plain text using mlx-audio (Parakeet for English; Qwen3-ASR for multilingual).",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input: Annotated[str, typer.Argument(
        help="YouTube URL, video ID (11 chars), or local file path. [dim]Tip: use the video ID or quote URLs to avoid shell glob issues with ? and &.[/dim]",
    )],
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help=_model_help,
    )] = None,
    language: Annotated[Optional[str], typer.Option(
        "--language", "-l",
        help="Language for STT. 'auto' (default) detects from audio via Whisper-tiny LID, then routes to the appropriate model. Pass a name like 'Chinese' or code like 'zh' to skip detection and force a language. Qwen3-ASR supports: Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, Japanese.",
    )] = "auto",
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
    aligner_model: Annotated[str, typer.Option(
        "--aligner-model",
        help="Forced-aligner model for word-level timestamps (Qwen3-ASR + --diarize only). "
             "Runs Qwen3-ForcedAligner after transcription to produce fine-grained phrase "
             "timestamps so Sortformer speaker boundaries align correctly.",
    )] = DEFAULT_ALIGNER_MODEL,
    identify: Annotated[bool, typer.Option(
        "--identify/--no-identify",
        help="Extract WeSpeaker ResNet293 speaker embeddings per diarized segment. Requires --diarize and any2md[speaker].",
    )] = False,
    auto_enroll: Annotated[bool, typer.Option(
        "--auto-enroll",
        help="Auto-enroll unmatched speakers as Unknown_N without prompting. Requires --identify. Mutually exclusive with --no-enroll.",
    )] = False,
    no_enroll: Annotated[bool, typer.Option(
        "--no-enroll",
        help="Skip enrollment prompts entirely; leave unmatched speakers as SPEAKER_N labels. Requires --identify.",
    )] = False,
    speakers: Annotated[Optional[str], typer.Option(
        "--speakers",
        help="Comma-separated speaker names to match against (limits search space). Requires --identify.",
    )] = None,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
    json_output: Annotated[bool, typer.Option(
        "--json", "-j",
        help="Output as JSON to stdout instead of writing a file.",
    )] = False,
    fields: Annotated[Optional[str], typer.Option(
        "--fields",
        help="Comma-separated dot-notation fields to include in JSON output (e.g. 'frontmatter,content').",
    )] = None,
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

            # Auto-detect language + route to the best model.
            # Priority: explicit --language > YouTube metadata > audio LID > default (en/Parakeet).
            # model_flag=None means the user did not pass --model, so LID routing is active.
            resolved_lang, resolved_model = _resolve_language_and_model(
                whisper_audio,
                lang_flag=language,   # "auto" or explicit language from --language
                model_flag=model,     # None unless user passed --model explicitly
                yt_metadata=metadata if is_youtube else None,
            )
            # Use resolved values for transcription
            language = resolved_lang
            model = resolved_model

            if identify and not diarize_flag:
                logger.warning("--identify has no effect without --diarize; ignoring")
                identify = False

            if speakers and not identify:
                logger.warning("--speakers has no effect without --identify; ignoring")
                speakers = None

            # Parse comma-separated speaker names (may contain @group references) into a list
            speaker_names: Optional[List[str]] = None
            if speakers and identify:
                try:
                    from any2md.speaker import open_catalog as _open_catalog, resolve_speakers_arg
                    _tmp_conn = _open_catalog(None)
                    speaker_names = resolve_speakers_arg(_tmp_conn, speakers)
                except ValueError as exc:
                    logger.error("--speakers: %s", exc)
                    raise typer.Exit(code=1)
                except ImportError:
                    # wespeaker not installed — fall back to plain split
                    speaker_names = [s.strip() for s in speakers.split(",") if s.strip()]

            if auto_enroll and no_enroll:
                logger.error("--auto-enroll and --no-enroll are mutually exclusive")
                raise typer.Exit(code=1)

            if (auto_enroll or no_enroll) and not identify:
                logger.warning("--auto-enroll/--no-enroll have no effect without --identify; ignoring")
                auto_enroll = False
                no_enroll = False

            # Collect unmatched speakers for JSON output
            _unmatched: List[Dict] = []

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
                identify=identify,
                audio_for_speaker=whisper_audio if identify else None,
                auto_enroll=auto_enroll,
                no_enroll=no_enroll,
                _unmatched_out=_unmatched if (json_output or is_json_mode()) else None,
                speaker_names=speaker_names,
                language=language,
                aligner_model=aligner_model,
            )

            logger.info(f"Transcription completed successfully. Output saved to: {output_file}")

            if json_output or is_json_mode():
                import json as _json
                from pathlib import Path as _Path
                content = _Path(output_file).read_text(encoding='utf-8')
                fm = metadata or {}
                # Build output dict manually to support unmatched_speakers extension
                output_dict: Dict = {
                    "frontmatter": fm,
                    "content": content,
                    "source": input,
                    "converter": "yt",
                }
                if _unmatched:
                    output_dict["unmatched_speakers"] = _unmatched
                if speaker_names and identify:
                    # Build attendance report: who was expected vs. who was found
                    _identified = list(fm.get("identified_speakers", []))
                    _unknown = list(fm.get("unidentified_speakers", []))
                    output_dict["attendance"] = {
                        "expected": speaker_names,
                        "identified": _identified,
                        "unknown": _unknown,
                    }
                if fields:
                    from any2md.common import _filter_fields
                    output_dict = _filter_fields(output_dict, fields)
                _json.dump(output_dict, sys.stdout, indent=2, default=str)
                sys.stdout.write("\n")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
