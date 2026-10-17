# WORKLOG

## 2026-10-17 — ANY2-37: native Rust port (Option C), phases 1-5

Goal: a truly native `any2md-rs` on MLX via mlx-c/mlx-rs, with WER/DER parity gates.

Landed so far (branch `ANY2-37`, crate workspace under `rust/`):

| Phase | What | Evidence |
|---|---|---|
| 1 | mlx-rs 0.32 + safetensors; mlx-sys builds MLX from source | 4-bit quantized matmul on real ReaderLM weights **identical** to Python MLX |
| 2 | HF-cache loader, safetensors weights, tokenizers, WAV + mel | whisper mel 7e-6, NeMo mel 5e-5 max diff |
| 3 | Qwen2/ReaderLM decode loop (GQA, RoPE, KV cache, tied head) | `sample.html` markdown **byte-identical**, 145 tok/s, 1.24 GB |
| 4 | Whisper encoder + one-step decoder (LID) | 4/4 fixtures same language, prob Δ ≤ 8e-3 |
| 5 | Parakeet TDT (Conformer + LSTM + joint + chunk merge) | encoder 1e-6 rel; decode trace identical; **token-identical** on 3 fixtures (WER 0.00%); 82-min podcast 26 425 vs 26 388 tokens (0.9965) in 34 s at 3.9 GB |

Traps found while porting (worth remembering):
- Parakeet's LSTM state update must be **discarded** when the joint predicts blank;
  committing it drops repeated tokens (`Direct T Tia` → `Directia`).
- NeMo mel uses a **symmetric** Hann (denominator `size-1`), Whisper a periodic one.
- `mlx_audio.utils.load_audio` defaults to **24 kHz**; the STT paths use
  `mlx_audio.stt.utils.load_audio`. A wrong loader made a correct mel look broken.
- any2md's `generate()` casts the waveform to **bf16** before the STFT; that flips
  borderline joint decisions (one 30 s crop: 178 tokens at f32 vs 195 at bf16).
  `any2md-rs stt` therefore defaults to `--dtype bf16` to match the CLI.
- mlx-rs `try_index(range)` indexes **axis 0**; per-axis slicing needs
  `take_axis` with explicit index arrays (a silent empty mask otherwise).

Qwen3-ASR (done after the first pass): audio tower with per-100-frame conv
windows + block-attention mask, Qwen3 decoder with per-head q/k RMSNorm; the
MLX 4-bit conversion has no `tokenizer.json`, so `Qwen/Qwen3-ASR-1.7B-hf` is used
after verifying id-for-id equality. Token-identical on zh_short (12/12) and
zh_two_speakers (24/24).

Diarization notes for the next pass (Nemotron-3 default):
- weights: `encoder.pre_encode` is a plain linear (128*8 -> 512, bias-free),
  `embed_norm`/`final_norm`, 31 transformer blocks (w_qkv fused, RoPE on
  head_dim 64, gelu FFN), then `sortformer_modules`: encoder_proj 512->192,
  subpixel_upsample Conv1d(192->1536, k3, pad1) reshaping time by x8,
  first_hidden_to_hidden -> single_hidden_to_spks -> sigmoid; 8 speakers.
  `learnable_sil_emb` is 512-d (encoder space).
- the checkpoint ships its own mel front-end buffers
  (`preprocessor.fb` [1,128,257], `preprocessor.window` [400]) — use those
  instead of the generated asset for exactness.
- `generate()` streams 27.2 s windows (chunk_len 340 * factor 8 * hop 160),
  `feed()` internally slides 2720-frame central windows with 320-frame right
  context; AOSC compression (spkcache 264, fifo 40) kicks in after the first
  window. Segment conversion + stream merging are now ported and unit-tested
  (`any2md-models/src/diarize.rs`, vectors from
  `rust/parity/diarize_segments.py`).

Nemotron encoder + speaker head: ported and numerically verified
(`nemotron.rs`). Feature-stacking projection (`encoder.pre_encode.proj`,
bias-free) -> 31 blocks (LayerNorm, fused qkv `attn.w_qkv`, RoPE head_dim 64,
boolean key mask, gelu FFN) -> final_norm -> speaker head (encoder_proj
512->192, subpixel Conv1d 192->1536 with k3/pad1 followed by a reshape that
restores 8x time, ReLU -> 192 -> 8 -> sigmoid). Parity on a random 800-frame
mel: encoder max|diff| 9e-6 (rel 1.2e-6), speaker preds max|diff| 7e-9.

Nemotron streaming: ported and certified (`nemotron_stream.rs`). Snags worth
remembering:
- segments must be computed **once over the concatenated probability timeline**,
  not per window (upstream `generate()` concatenates every window's central
  preds and calls `_preds_to_segments` once) -- per-window conversion invents
  breaks at window boundaries and loses the global sort;
- `state.fifo_preds` is *rebuilt* each step (`low[cache..start]` +
  `low[start..start+n]`), not appended to;
- queued predictions must be masked past `(chunk_len + cache + fifo) * factor`,
  and segment times need the window start offset;
- the checkpoint's own `preprocessor.fb/window` are used, and the mel is
  globally indexed (frames `start..start+count`, zeroed past `total//hop`).
DER vs Python: 0.32% on the 44 s two-speaker clip (8/8 segments, 2/2 speakers),
10.5% on the 90 s interview where the only material difference is the final
speaker label (81.6-90 s) -- miss/FA stay below 0.5%.

Remaining: the Sortformer streaming path (any2md drives it through chunked
`generate_stream` + cross-chunk merge), packaging/release docs, merge to `main`.

Gates: `make rust-parity` → mel, LID, STT (WER) and ReaderLM parity.

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
     whole recording — an 82-min podcast stalled at **21 GB footprint for 17+ min**
     (never finished)
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
- Tests: 1107 unit + 16 slow + 11 integration all pass on the new stack
  (fixed pre-existing failures: missing rst/doc/sub deps, fitz stdout pollution,
  `test_zh_two_speakers_diarize` asserting frontmatter without metadata).
- **Upgrade regressions caught by the memory perf pass** (all fixed + regression
  tests):
  - `pymupdf4llm` 1.28 reports `page_number` **1-based**; the old `+1` shifted
    every page and made VLM OCR on the last page fail ("page 10 not in
    document" → 9/10 pages converted).
  - PyMuPDF 1.28 prints parser messages to **stdout**, corrupting `--json`
    (`pymupdf.set_messages(stream=sys.stderr)`).
  - `configure_mlx_memory()` originally imported mlx.core at startup, costing
    ~150 ms on every zero-dep run (csv 0.04s → 0.20s). Now it only acts when
    MLX is already loaded; each MLX converter configures/releases explicitly.

### Final measured numbers (all on the merged main)

| Workload | Before | After |
|---|---|---|
| 82-min podcast `--diarize` (Nemotron default) | 21 GB footprint, still running after 17 min, killed | **4.64 GB peak, 61 s**, 0.68 GB after |
| 82-min podcast `--diarize-model diar_streaming_sortformer…` | 24.7 GB peak (first pass), 21 GB stall on the old full-clip path | **4.64 GB peak**, 0.62 GB after |
| 3× repeat of `yt interview.wav --diarize` | 4.56 → 4.62 GB resident, 4.04 GB cache never released | **0.61 / 0.61 / 0.62 GB**, cache 0.00 GB |
| `any2md x.csv` (zero-dep converter) | 3.1 s / 445 MB (imported mlx-vlm+torch for every command) | **0.04 s / 15 MB** |
| VLM OCR, 12 pages in one process | untested | 3.69 GB flat, cache 0.09 GB |
| Speaker embeddings, 45 segments | untested | 0.92 → 0.46 GB (flat) |

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
