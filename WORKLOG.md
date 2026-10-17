# WORKLOG

## 2026-10-16 — ANY2-36: dep upgrades, memory fixes, model upgrades (v0.4.0)

Goal: update all libraries to latest, hunt the intermittent memory leaks, and apply clear 1:1 model upgrades.

### What was done

- **Deps upgraded to latest** (`pyproject.toml`, `uv.lock`, `requirements.txt`):
  mlx 0.31.1→0.32.4, mlx-audio 0.4.1→0.5.9, mlx-vlm 0.4.1→0.7.8, mlx-lm 0.31.1→0.32.2,
  transformers 5.3→5.20.1, huggingface-hub 1.7.2→2.2.0, pymupdf4llm 1.27.2→1.28.2,
  markitdown 0.1.5→0.1.8, typer 0.24.1→0.27.3, yt-dlp 2026.3.17→2026.8.19,
  onnxruntime 1.24.4→1.31.0, numpy→2.5.4, Pillow→13.0.0. Added real extras for
  rst (pypandoc/docutils), sub (pysubs2), and web (trafilatura) — they were used
  by the code but never declared. `[tool.uv] environments` pinned to darwin/arm64
  (MLX-only project), which also frees onnxruntime from magika's Windows pin.
- **Memory fixes** (the "leaks"):
  1. Diarization O(n²): Sortformer full-clip `generate()` ran attention over the
     whole recording — an 82-min podcast stalled at **21 GB footprint for 5+ min**
     and never finished. Root cause confirmed in upstream source (mlx-audio 0.5.9
     `sortformer.py::generate` computes full-file mel + one full attention pass).
     Fix: dispatch on model family. Nemotron-3-Diarization `generate()` streams
     internally (bounded windows); legacy Sortformer models are driven through
     `generate_stream()` fed by `_iter_wav_chunks()` (stdlib wave reader), because
     even `generate_stream(path)` does a full-file ConvSubsampling pre-encode
     (**14.5 GB transient** measured for 82 min). Result: 82-min podcast completes
     in ~45 s at **4.5 GB peak**; end-of-run footprint 0.6 GB.
  2. MLX buffer cache: freed buffers piled up to the memory limit and never
     shrank — looked exactly like a leak in long-lived usage. New
     `common.configure_mlx_memory()` caps it (2 GB default,
     `ANY2MD_MLX_CACHE_LIMIT_GB` override) and `common.release_mlx_memory()`
     (gc + `mx.clear_cache()`) runs between pipeline stages.
  3. Models stayed resident: STT/diarizer/aligner/speaker models now dropped
     (`del` + release) as soon as their stage finishes.
  4. ForcedAligner: aligned the whole recording + whole transcript in one
     forward pass (memory grows with clip length; unreliable past ~5 min).
     Now aligns **one ASR segment at a time** with timestamps re-offset.
  5. LID: whisper mel was computed over the whole file; now just the first 30 s
     window (stdlib wave slice).
  6. CLI startup: `any2md` imported every converter (mlx-vlm → transformers →
     torch: 3.1 s, 445 MB) even for `csv`. Lazy per-subcommand imports now:
     0.05 s, 15 MB.
  7. `pdf.py` imported the deprecated `fitz` alias, which prints a notice to
     **stdout**, corrupting `--json`. Imports `pymupdf` directly now.
  - Verified: 3 repeat runs of the yt pipeline → footprint flat at 0.61 GB,
    MLX cache 0.00 GB after each run (old stack: 4.6 GB resident + 4 GB cache
    retained). `scripts/memory_profile.py` added to reproduce measurements.
- **Model upgrades** (after HF API + source verification of every candidate):
  - Diarization → `mlx-community/Nemotron-3-Diarization` (NVIDIA's designated
    successor: 8 speakers, DER 19.09→12.73 DIHARD III; supported by
    mlx-audio >= 0.5.6). Measured on an 82-min podcast: 45 s, 4.5 GB peak.
    Note: it split the 3-host show into 6 speakers; legacy streaming Sortformer
    (still one `--diarize-model` flag away) gave 4. Kept both paths working.
  - LID → `mlx-community/whisper-tiny-asr-fp16` (same weights as the old
    whisper-tiny-mlx port but ships preprocessor_config.json, so the
    openai/whisper-tiny processor fallback is no longer needed).
  - img aliases: `qwen3.8-27b` (→ Qwen3.8-27B-4bit, same arch/tensor layout as
    Qwen3.5-27B) and `qwen3.6-35b` (→ Qwen3.6-35B-A3B-4bit). No successor exists
    for the 4B/9B sizes.
  - `smoldocling` alias repointed to `docling-project/SmolDocling-256M-preview-mlx-bf16`
    — the old `mlx-community/SmolDocling-256M-4bit` repo returns 401.
  - Explicitly **not** upgraded: Parakeet (v3 is still NVIDIA's newest TDT; the
    moondream retrain is third-party), Qwen3-ASR/ForcedAligner (no successors),
    ReaderLM-v2 (no v3 exists), WeSpeaker (wespeakerruntime 1.0.1 is current;
    docs corrected: it is ResNet34-LM, not ResNet293).
- **speaker extra**: torchaudio >= 2.9 routes `load()` through torchcodec, so
  `wespeakerruntime` was dead-on-arrival with latest torch. Added
  `torch/torchaudio/torchcodec` pins to the `[speaker]` extra; verified
  end-to-end embedding extraction works.
- Tests: 1105 unit + 16 slow + 11 integration all pass on the new stack
  (fixed pre-existing failures: missing rst/doc/sub deps, fitz stdout pollution,
  `test_zh_two_speakers_diarize` asserting frontmatter without metadata).

### Gotchas for next time

- PyPI `any2md` (1.1.1, rocklambros) is an **unrelated project** that squatted
  the name — never `uv pip install --upgrade any2md`.
- `mlx_audio.stt` STT models each handle long audio differently: parakeet
  chunks internally (30 s + 2 s overlap, `mx.clear_cache()`), qwen3_asr chunks
  at 1200 s, forced aligner does NOT chunk. Check the model source before
  assuming `generate()` is memory-safe.
- `mx.set_cache_limit()` is advisory — big one-shot evals can exceed it;
  `release_mlx_memory()` between stages is what actually keeps RSS flat.
- HF cache holds ~166 GB of models; before claiming a model is "missing",
  check `~/.cache/huggingface/hub`.
