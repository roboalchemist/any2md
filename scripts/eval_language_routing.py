#!/usr/bin/env python3
"""Evaluate ASR model routing: Parakeet v3 vs Qwen3-ASR 0.6B / 1.7B.

Routing decisions in ``any2md.yt`` are only as good as the evidence behind
them, so this script re-creates that evidence on reference corpora:

- **MLS** (``facebook/multilingual_librispeech``) for the European languages
  Parakeet v3 covers (es, fr, de, it, pt, nl, pl) — word-level WER.
- **FLEURS** (``google/fleurs``) for languages Parakeet cannot read
  (zh, ja, ko, ar, hi, ru) — character error rate, since CJK has no word
  boundaries.
- The repo's own fixtures for English.

It also runs the production router (``any2md.yt._resolve_language_and_model``)
over the same clips and reports whether each one landed on a model that can
actually read it, plus per-model speed and peak RSS.

    .venv/bin/python scripts/eval_language_routing.py --clips 6

Clips are cached under ``/tmp/lang_eval``; ``--skip-fetch`` reuses them.
FLEURS ships one tar.gz per language, so only the first N members are read and
the HTTP response is closed early (a few MB instead of 200-450 MB).
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import subprocess
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/tmp/lang_eval")
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
MLS_CONFIG = {
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "pl": "polish",
}
FLEURS_CONFIG = {
    "zh": "cmn_hans_cn",
    "ja": "ja_jp",
    "ko": "ko_kr",
    "ar": "ar_eg",
    "hi": "hi_in",
    "ru": "ru_ru",
}
# Languages Parakeet v3 can transcribe; ru is European and therefore in the set.
PARAKEET_LANGS = set(MLS_CONFIG) | {"en", "ru"}


def norm_words(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower()).replace("\u2019", "'")
    out = [ch if (ch.isalnum() or ch == "'") else " " for ch in text]
    return " ".join("".join(out).split())


def norm_chars(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFC", text.lower()) if ch.isalnum())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[len(b)]


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref).split(), norm_words(hyp).split()
    return 0.0 if not r else edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r, h = norm_chars(ref), norm_chars(hyp)
    return 0.0 if not r else edit_distance(list(r), list(h)) / len(r)


def fetch_mls(code: str, config: str, n: int) -> list[dict]:
    out = WORK / code
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.json"
    if index.exists():
        return json.loads(index.read_text())
    offsets = [int(i * 100 / n) for i in range(n)]
    cached = [out / f"{o}.json" for o in offsets]
    if all((out / f"{o}.wav").exists() and m.exists() for o, m in zip(offsets, cached)):
        clips = [
            {"wav": str(out / f"{o}.wav"), "transcript": json.loads(m.read_text())["transcript"]}
            for o, m in zip(offsets, cached)
        ]
        index.write_text(json.dumps(clips, ensure_ascii=False, indent=1))
        return clips
    clips = []
    for i in range(n):
        offset = int(i * 100 / n)
        meta, wav = out / f"{offset}.json", out / f"{offset}.wav"
        if not (meta.exists() and wav.exists()):
            url = (
                "https://datasets-server.huggingface.co/rows?"
                f"dataset=facebook/multilingual_librispeech&config={config}"
                f"&split=test&offset={offset}&length=1"
            )
            with urllib.request.urlopen(url, timeout=60) as resp:
                row = json.load(resp)["rows"][0]["row"]
            opus = out / f"{offset}.opus"
            urllib.request.urlretrieve(row["audio"][0]["src"], opus)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(opus), "-ar", "16000", "-ac", "1", str(wav)],
                check=True,
                capture_output=True,
            )
            opus.unlink()
            meta.write_text(json.dumps({"transcript": row["transcript"]}))
        clips.append({"wav": str(wav), "transcript": json.loads(meta.read_text())["transcript"]})
    index.write_text(json.dumps(clips, ensure_ascii=False, indent=1))
    return clips


def fetch_fleurs(code: str, config: str, n: int) -> list[dict]:
    out = WORK / code
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.json"
    if index.exists():
        return json.loads(index.read_text())
    with urllib.request.urlopen(f"{FLEURS}/{config}/test.tsv", timeout=60) as resp:
        text = resp.read().decode("utf-8")
    refs = {}
    for line in text.splitlines():  # headerless: id, filename, raw, normalised, ...
        cols = line.split("\t")
        if len(cols) >= 4:
            refs[cols[1]] = cols[3] or cols[2]
    clips: list[dict] = []
    with urllib.request.urlopen(f"{FLEURS}/{config}/audio/test.tar.gz", timeout=120) as resp:
        tar = tarfile.open(fileobj=resp, mode="r|gz")
        for member in tar:
            if not member.name.endswith(".wav"):
                continue
            wav = out / f"{len(clips)}.wav"
            wav.write_bytes(tar.extractfile(member).read())
            transcript = refs.get(Path(member.name).name, "")
            if transcript:
                clips.append({"wav": str(wav), "transcript": transcript})
            if len(clips) >= n:
                break
        resp.close()
    index.write_text(json.dumps(clips, ensure_ascii=False, indent=1))
    return clips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=6, help="clips per language")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--out", default="/tmp/lang_eval/routing.json")
    ap.add_argument(
        "--only",
        choices=["parakeet-v3", "qwen3-asr-0.6b", "qwen3-asr-1.7b"],
        help="run a single model (peak memory is only meaningful this way)",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from mlx_audio.stt import load

    data: dict[str, list[dict]] = {}
    if args.skip_fetch:
        for code in list(MLS_CONFIG) + list(FLEURS_CONFIG):
            data[code] = json.loads((WORK / code / "index.json").read_text())
    else:
        for code, config in MLS_CONFIG.items():
            data[code] = fetch_mls(code, config, args.clips)
        for code, config in FLEURS_CONFIG.items():
            data[code] = fetch_fleurs(code, config, args.clips)
        print(f"fetched {sum(len(v) for v in data.values())} clips across {len(data)} languages")

    models = {
        "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
        "qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "qwen3-asr-1.7b": "mlx-community/Qwen3-ASR-1.7B-4bit",
    }
    if args.only:
        models = {args.only: models[args.only]}
    import mlx.core as mx

    scores: dict[str, dict[str, float]] = {}
    for name, model_id in models.items():
        import time

        mx.reset_peak_memory()
        model = load(model_id)
        started = time.time()
        per_lang = {}
        for code, clips in data.items():
            err = cer if code in FLEURS_CONFIG else wer
            per_lang[code] = sum(err(c["transcript"], model.generate(c["wav"]).text) for c in clips) / len(clips)
        elapsed = time.time() - started
        peak = mx.get_peak_memory() / 1e9
        scores[name] = per_lang
        del model
        print(
            f"{name:16s} " + " ".join(f"{k}:{v:.1%}" for k, v in per_lang.items())
            + f"  | {elapsed / sum(len(v) for v in data.values()):.3f}s/clip  {peak:.2f} GB MLX peak"
        )

    from any2md.yt import _resolve_language_and_model

    routing = {}
    for code, clips in data.items():
        picks = []
        for clip in clips:
            _lang, model_id = _resolve_language_and_model(clip["wav"], None, None, None)
            picks.append(model_id)
        ok = sum("parakeet" in p for p in picks) == (len(picks) if code in PARAKEET_LANGS else 0)
        routing[code] = {"ok": ok, "models": sorted(set(picks))}
        print(f"route {code}: {'OK ' if ok else 'BAD'} {routing[code]['models']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"scores": scores, "routing": routing}, indent=2))
    print("wrote", args.out)
    bad = [k for k, v in routing.items() if not v["ok"]]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
