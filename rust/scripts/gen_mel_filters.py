#!/usr/bin/env python3
"""Generate the mel-filterbank assets used by the native Rust front-ends.

The filterbanks are pure functions of (sample_rate, n_fft, n_mels, norm,
mel_scale). Rather than re-derive that float-sensitive maths in Rust, we
serialise the exact matrices mlx-audio uses into small safetensors files, so the
native side and the Python side are numerically identical by construction.

Run after changing model preprocessing:

    .venv/bin/python rust/scripts/gen_mel_filters.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "rust" / "assets"
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from mlx_audio.utils import mel_filters

    ASSETS.mkdir(parents=True, exist_ok=True)

    # Whisper LID: 80 mels, n_fft 400, Slaney scale/norm (mel_scale=None == slaney)
    whisper = mel_filters(16_000, 400, 80, norm="slaney", mel_scale=None)
    whisper = whisper.astype(mx.float32)
    mx.save_safetensors(str(ASSETS / "mel_whisper_80.safetensors"), {"mel_whisper_80": whisper})

    # NeMo family (Parakeet, Sortformer, Nemotron): 128 mels, n_fft 512
    nemo = mel_filters(16_000, 512, 128, norm="slaney", mel_scale="slaney").astype(mx.float32)
    mx.save_safetensors(str(ASSETS / "mel_nemo_512_128.safetensors"), {"mel_nemo_512_128": nemo})

    meta = {
        "mel_whisper_80": {"shape": list(whisper.shape), "dtype": "float32",
                           "source": "mlx_audio.utils.mel_filters(16000, 400, 80, norm=slaney, mel_scale=None)"},
        "mel_nemo_512_128": {"shape": list(nemo.shape), "dtype": "float32",
                             "source": "mlx_audio.utils.mel_filters(16000, 512, 128, norm=slaney, mel_scale=slaney)"},
    }
    (ASSETS / "README.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {ASSETS}/mel_whisper_80.safetensors {whisper.shape}")
    print(f"wrote {ASSETS}/mel_nemo_512_128.safetensors {nemo.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
