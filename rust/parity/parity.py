#!/usr/bin/env python3
"""Parity harness: native Rust (`any2md-rs`) vs the Python any2md.

Two jobs:

1. Produce the Python reference for a fixture (prompt tokens + generated text).
2. Compare a Rust `--stats-json` report against that reference and fail on
   quality drift.

This is the gate that keeps a model port from silently regressing: token-level
equality where the decode loop is greedy, and text similarity where float
reassociation can flip a token.

Usage:
    python3 rust/parity/parity.py reference readerlm --input tests/fixtures/sample.html \
        --max-tokens 64 --out /tmp/py.json
    python3 rust/parity/parity.py compare /tmp/py.json /tmp/rust.json --min-similarity 0.99
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys

READERLM = "mlx-community/jinaai-ReaderLM-v2"


def readerlm_reference(args) -> dict:
    """Run the Python any2md path for ReaderLM and capture the reference."""
    from any2md.web import build_reader_prompt, load_reader_model
    from mlx_lm import generate

    html = open(args.input, encoding="utf-8").read()
    if len(html) > 200_000:
        html = html[:200_000]

    model, tokenizer = load_reader_model(args.model)
    prompt = build_reader_prompt(html, tokenizer)
    ids = tokenizer.encode(prompt)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
    return {
        "model": args.model,
        "prompt_tokens": len(ids),
        "prompt_first": ids[:6],
        "prompt_last": ids[-6:],
        "text": text,
    }


def cmd_reference(args) -> int:
    report = readerlm_reference(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        f"reference: prompt_tokens={report['prompt_tokens']} chars={len(report['text'])} -> {args.out}"
    )
    return 0


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def cmd_compare(args) -> int:
    py = json.load(open(args.reference, encoding="utf-8"))
    rs = json.load(open(args.rust, encoding="utf-8"))

    failures: list[str] = []

    if py["prompt_tokens"] != rs["prompt_tokens"]:
        failures.append(f"prompt tokens differ: python={py['prompt_tokens']} rust={rs['prompt_tokens']}")
    if py["prompt_first"] != rs["prompt_first"] or py["prompt_last"] != rs["prompt_last"]:
        failures.append(
            f"prompt token boundaries differ: python={py['prompt_first']}..{py['prompt_last']} "
            f"rust={rs['prompt_first']}..{rs['prompt_last']}"
        )

    a, b = _normalise(py["text"]), _normalise(rs["text"])
    similarity = difflib.SequenceMatcher(None, a, b).ratio()
    identical = a == b

    print(f"prompt tokens: {py['prompt_tokens']} / {rs['prompt_tokens']}")
    print(f"python chars={len(a)} rust chars={len(b)} identical={identical} similarity={similarity:.4f}")

    if not identical:
        diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "python", "rust", lineterm=""))
        print("\n".join(diff[:40]))
    if similarity < args.min_similarity:
        failures.append(f"text similarity {similarity:.4f} < {args.min_similarity}")

    if failures:
        print("\nPARITY FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPARITY OK")
    return 0


def cmd_mel(args) -> int:
    """Compare the native mel front-end against the Python one for a WAV."""
    import subprocess
    import tempfile

    import numpy as np

    def load_bin(path):
        raw = open(path, "rb").read()
        nd = int.from_bytes(raw[:4], "little")
        dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "little") for i in range(nd)]
        return np.frombuffer(raw[4 + 4 * nd :], dtype="<f4").reshape(dims)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out = tmp.name
    subprocess.run(
        [args.binary, "mel", args.input, "--kind", args.kind, "--out", out],
        check=True,
        capture_output=True,
    )
    rust = load_bin(out)

    from mlx_audio.stt.utils import load_audio

    wav = load_audio(args.input)
    if args.kind == "whisper":
        from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram

        ref = np.array(log_mel_spectrogram(wav, n_mels=80, padding=0))
    else:
        from mlx_audio.vad.models.sortformer.sortformer import extract_mel_features

        ref = np.array(extract_mel_features(wav, n_mels=128, normalize="per_feature", pad_to=16))[0]

    frames = min(ref.shape[-1], rust.shape[-1])
    diff = float(np.abs(ref[:, :frames] - rust[:, :frames]).max())
    print(f"{args.kind} mel: python{ref.shape} rust{rust.shape} max|diff|={diff:.3e}")
    if diff > args.max_diff:
        print(f"MEL PARITY FAILED: {diff:.3e} > {args.max_diff:.1e}")
        return 1
    print("MEL PARITY OK")
    return 0


def cmd_lid(args) -> int:
    """Compare native Whisper LID against any2md's Python `_detect_language`."""
    import subprocess

    from any2md.yt import _detect_language

    failures = 0
    for wav in args.inputs:
        py_lang, py_conf = _detect_language(wav)
        out = subprocess.run(
            [args.binary, "lid", wav], check=True, capture_output=True, text=True
        ).stdout.split()
        rs_lang, rs_conf = out[0], float(out[1])
        ok = py_lang == rs_lang and abs(py_conf - rs_conf) <= args.max_prob_diff
        print(
            f"{'OK  ' if ok else 'FAIL'} {wav}: python={py_lang}/{py_conf:.4f} "
            f"rust={rs_lang}/{rs_conf:.4f} d={abs(py_conf - rs_conf):.4f}"
        )
        failures += 0 if ok else 1
    if failures:
        print(f"LID PARITY FAILED ({failures}/{len(args.inputs)})")
        return 1
    print("LID PARITY OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ref = sub.add_parser("reference", help="produce a Python reference report")
    ref.add_argument("model_kind", choices=["readerlm"])
    ref.add_argument("--input", required=True)
    ref.add_argument("--model", default=READERLM)
    ref.add_argument("--max-tokens", type=int, default=64)
    ref.add_argument("--out", required=True)
    ref.set_defaults(func=cmd_reference)

    cmp_ = sub.add_parser("compare", help="compare a Rust report against a Python reference")
    cmp_.add_argument("reference")
    cmp_.add_argument("rust")
    cmp_.add_argument("--min-similarity", type=float, default=0.99)
    cmp_.set_defaults(func=cmd_compare)

    mel = sub.add_parser("mel", help="compare the native mel front-end with Python")
    mel.add_argument("--input", required=True)
    mel.add_argument("--kind", choices=["whisper", "nemo"], default="whisper")
    mel.add_argument("--binary", default="rust/target/release/any2md-rs")
    mel.add_argument("--max-diff", type=float, default=1e-3)
    mel.set_defaults(func=cmd_mel)

    lid = sub.add_parser("lid", help="compare native Whisper language ID with Python")
    lid.add_argument("--binary", default="rust/target/release/any2md-rs")
    lid.add_argument("--max-prob-diff", type=float, default=0.02)
    lid.add_argument(
        "inputs",
        nargs="*",
        default=[
            "tests/audio/zh_short.wav",
            "tests/audio/zh_two_speakers.wav",
            "tests/audio/yt_interview.wav",
            "tests/audio/two_speakers.wav",
        ],
    )
    lid.set_defaults(func=cmd_lid)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
