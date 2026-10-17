//! Native any2md CLI (Rust + MLX).
//!
//! First ported converter: `html` (ReaderLM-v2 → markdown), which exercises the
//! whole native stack: HF cache resolution, safetensors loading, tokenizer,
//! chat template, KV-cached decode loop, and output formatting.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tokenizers::Tokenizer;

use any2md_models::{Generator, Qwen2, Qwen2Config};

const READERLM: &str = "mlx-community/jinaai-ReaderLM-v2";
const MAX_HTML_CHARS: usize = 200_000;
const MAX_OUTPUT_TOKENS: usize = 8192;
/// `<|im_end|>` and `<|endoftext|>` for Qwen2-family tokenizers.
const EOS_IDS: [u32; 2] = [151645, 151643];

const SYSTEM_PROMPT: &str = "Convert the HTML to clean markdown. \
Output only the converted markdown. \
Do not include any explanations or preamble.";

#[derive(Parser)]
#[command(
    name = "any2md-rs",
    version,
    about = "Convert anything to markdown — native Rust/MLX build"
)]
struct Cli {
    #[arg(long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Transcribe a 16 kHz mono WAV with Parakeet TDT
    Stt {
        /// 16 kHz mono WAV file
        input: PathBuf,
        /// Parakeet model id
        #[arg(short = 'm', long, default_value = any2md_models::parakeet::DEFAULT_MODEL)]
        model: String,
        /// Print timings and memory to stderr
        #[arg(long)]
        stats: bool,
        /// Write the plain transcript here
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit token ids as JSON (parity checks)
        #[arg(long)]
        tokens_json: Option<PathBuf>,
        /// Write raw encoder features (f32, shape header) for parity debugging
        #[arg(long)]
        features_out: Option<PathBuf>,
        /// Write the per-step decode trace (time, pred, duration) as JSON
        #[arg(long)]
        trace_json: Option<PathBuf>,
        /// Chunk length in seconds (0 disables chunking; default 30 like any2md)
        #[arg(short = 'c', long, default_value_t = 30.0)]
        chunk_duration: f32,
        /// Audio precision before the STFT: bf16 matches any2md's generate() default
        #[arg(long, default_value = "bf16")]
        dtype: String,
    },
    /// Diarize a 16 kHz mono WAV (Nemotron-3)
    Diarize {
        /// 16 kHz mono WAV file
        input: PathBuf,
        /// Model id
        #[arg(short = 'm', long, default_value = any2md_models::nemotron::DEFAULT_MODEL)]
        model: String,
        /// Emit segments as JSON
        #[arg(long)]
        json: bool,
        /// Print timings and memory to stderr
        #[arg(long)]
        stats: bool,
        /// Write the full JSON report here
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the Nemotron encoder on a raw mel tensor (parity debugging)
    NemotronEncode {
        /// Model id
        #[arg(short = 'm', long, default_value = any2md_models::nemotron::DEFAULT_MODEL)]
        model: String,
        /// Input tensor: `[dims] + f32le` as written by write_f32_tensor
        #[arg(long)]
        input_tensor: PathBuf,
        /// Output tensor path (encoder features)
        #[arg(long)]
        out: PathBuf,
        /// Optional speaker-probability tensor path
        #[arg(long)]
        preds_out: Option<PathBuf>,
    },
    /// Transcribe with Qwen3-ASR (multilingual)
    Asr {
        /// 16 kHz mono WAV file
        input: PathBuf,
        /// Qwen3-ASR model id
        #[arg(short = 'm', long, default_value = any2md_models::qwen3asr::DEFAULT_MODEL)]
        model: String,
        /// Language name or ISO code (e.g. Chinese / zh); omit to auto-detect
        #[arg(short = 'l', long)]
        language: Option<String>,
        /// Maximum tokens to generate
        #[arg(long, default_value_t = 1024)]
        max_tokens: usize,
        /// Print timings and memory to stderr
        #[arg(long)]
        stats: bool,
        /// Write the plain transcript here
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit token ids as JSON (parity checks)
        #[arg(long)]
        tokens_json: Option<PathBuf>,
    },
    /// Detect spoken language with Whisper (mirrors --language auto)
    Lid {
        /// 16 kHz mono WAV file
        input: PathBuf,
        /// Whisper model id
        #[arg(short = 'm', long, default_value = any2md_models::whisper::DEFAULT_LID_MODEL)]
        model: String,
        /// Also print the full 30 s window mel shape
        #[arg(long)]
        stats: bool,
    },
    /// Dump log-mel features for a 16 kHz mono WAV (parity/verification helper)
    Mel {
        /// 16 kHz mono WAV file
        input: PathBuf,
        /// Front-end: whisper (Whisper LID) or nemo (Parakeet/Sortformer/Nemotron)
        #[arg(long, default_value = "whisper")]
        kind: String,
        /// Where to write the raw f32 little-endian tensor
        #[arg(long)]
        out: PathBuf,
    },
    /// Convert a local HTML file to markdown with ReaderLM-v2
    Html {
        /// HTML file to convert
        input: PathBuf,
        /// Output directory (default: current directory)
        #[arg(short = 'o', long, default_value = ".")]
        output_dir: PathBuf,
        /// Hugging Face model id
        #[arg(short = 'm', long, default_value = READERLM)]
        model: String,
        /// Maximum tokens to generate
        #[arg(long, default_value_t = MAX_OUTPUT_TOKENS)]
        max_tokens: usize,
        /// Print the prompt token count and boundary token ids (parity checks)
        #[arg(long)]
        dump_prompt: bool,
        /// Print generation stats (tokens/s, memory) to stderr
        #[arg(long)]
        stats: bool,
        /// Write a machine-readable parity report (prompt/generated tokens, text)
        #[arg(long)]
        stats_json: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    any2md_mlx::init()?;
    match cli.command {
        Command::Stt {
            input,
            model,
            stats,
            out,
            tokens_json,
            features_out,
            trace_json,
            chunk_duration,
            dtype,
        } => transcribe(
            &input,
            &model,
            stats,
            out.as_deref(),
            tokens_json.as_deref(),
            features_out.as_deref(),
            trace_json.as_deref(),
            chunk_duration,
            &dtype,
        ),
        Command::Diarize { input, model, json, stats, out } => {
            diarize_wav(&input, &model, json, stats, out.as_deref())
        }
        Command::NemotronEncode { model, input_tensor, out, preds_out } => {
            nemotron_encode(&model, &input_tensor, &out, preds_out.as_deref())
        }
        Command::Asr {
            input,
            model,
            language,
            max_tokens,
            stats,
            out,
            tokens_json,
        } => transcribe_asr(
            &input,
            &model,
            language.as_deref(),
            max_tokens,
            stats,
            out.as_deref(),
            tokens_json.as_deref(),
        ),
        Command::Lid { input, model, stats } => detect_language(&input, &model, stats),
        Command::Mel { input, kind, out } => dump_mel(&input, &kind, &out),
        Command::Html {
            input,
            output_dir,
            model,
            max_tokens,
            dump_prompt,
            stats,
            stats_json,
        } => html_to_markdown(
            &input,
            &output_dir,
            &model,
            max_tokens,
            dump_prompt,
            stats,
            stats_json.as_deref(),
        ),
    }
}

/// `qwen2` uses this prompt shape (see the model's `tokenizer_config.json`).
fn build_prompt(html: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{html}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn html_to_markdown(
    input: &Path,
    output_dir: &Path,
    model_id: &str,
    max_tokens: usize,
    dump_prompt: bool,
    stats: bool,
    stats_json: Option<&Path>,
) -> Result<()> {
    let html = std::fs::read_to_string(input)
        .with_context(|| format!("read {}", input.display()))?;
    let html = if html.len() > MAX_HTML_CHARS {
        tracing::warn!(len = html.len(), "truncating HTML to {MAX_HTML_CHARS} chars");
        &html[..floor_char_boundary(&html, MAX_HTML_CHARS)]
    } else {
        &html
    };

    let cfg_path = any2md_hub::get_file(model_id, "config.json")?;
    let config = Qwen2Config::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let tokenizer_path = any2md_hub::get_file(model_id, "tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let prompt = build_prompt(html);
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
    let ids: Vec<i32> = encoding.get_ids().iter().map(|v| *v as i32).collect();
    if dump_prompt {
        eprintln!(
            "prompt_tokens={} first={:?} last={:?}",
            ids.len(),
            &ids[..ids.len().min(6)],
            &ids[ids.len() - 6..]
        );
    }

    let dir = cfg_path.parent().context("config has no parent dir")?;
    tracing::info!(model = model_id, "loading weights");
    let model = Qwen2::load(dir, config)?;
    tracing::info!(prompt_tokens = ids.len(), "generating");

    let started = std::time::Instant::now();
    let prompt_arr = mlx_rs::Array::from_slice(&ids, &[1, ids.len() as i32]);
    let mut generator = Generator::new(&model);
    let mut streamed = String::new();
    let out_tokens = generator.generate(&prompt_arr, max_tokens, &EOS_IDS, |tok| {
        if let Ok(piece) = tokenizer.decode(&[tok], true) {
            streamed.push_str(&piece);
        }
    })?;
    let elapsed = started.elapsed();
    let text = tokenizer
        .decode(&out_tokens, true)
        .map_err(|e| anyhow::anyhow!("decode output: {e}"))?;

    if stats {
        let (active, cache, peak) = any2md_mlx::memory_stats()?;
        let gb = |v: usize| v as f64 / 1024.0 / 1024.0 / 1024.0;
        eprintln!(
            "tokens={} time={:.2}s tok/s={:.1} peak={:.2}GB active={:.2}GB cache={:.2}GB",
            out_tokens.len(),
            elapsed.as_secs_f64(),
            out_tokens.len() as f64 / elapsed.as_secs_f64().max(1e-9),
            gb(peak),
            gb(active),
            gb(cache),
        );
    }

    if let Some(path) = stats_json {
        let report = serde_json::json!({
            "model": model_id,
            "prompt_tokens": ids.len(),
            "prompt_first": &ids[..ids.len().min(6)],
            "prompt_last": &ids[ids.len().saturating_sub(6)..],
            "generated_tokens": out_tokens,
            "text": text,
            "seconds": elapsed.as_secs_f64(),
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    std::fs::create_dir_all(output_dir)?;
    let out_path = output_dir.join(format!("{stem}.md"));
    let mut file = std::fs::File::create(&out_path)?;
    writeln!(file, "---")?;
    writeln!(file, "source: {}", input.display())?;
    writeln!(file, "model: {model_id}")?;
    writeln!(file, "---\n")?;
    write!(file, "{}", text.trim())?;
    println!("{}", out_path.display());

    drop(model);
    any2md_mlx::release_memory();
    Ok(())
}

/// Largest char boundary <= `idx` (we slice HTML by chars, not bytes).
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Write `[dims]` + raw f32 data so a Python harness can `np.fromfile` it.
fn dump_mel(input: &Path, kind: &str, out: &Path) -> Result<()> {
    use anyhow::Context as _;
    let audio = any2md_audio::read_wav_16k(input)?;
    let mel = match kind {
        "whisper" => any2md_audio::whisper_mel(&audio, 80)?,
        "nemo" => any2md_audio::nemo_mel(&audio, &any2md_audio::NemoMel::parakeet())?,
        "nemotron" => {
            let cfg_path = any2md_hub::get_file(
                any2md_models::nemotron::DEFAULT_MODEL,
                "config.json",
            )?;
            let cfg = any2md_models::nemotron::DiarConfig::from_json(&std::fs::read_to_string(
                &cfg_path,
            )?)?;
            let w = any2md_mlx::Weights::load_dir(cfg_path.parent().context("dir")?)?;
            let fm = any2md_models::nemotron_stream::NemotronMel::from_weights(&w, &cfg)?;
            let samples: Vec<f32> = audio.as_slice::<f32>().to_vec();
            let total = samples.len() as i64;
            fm.features(&samples, 0, (samples.len() / 160 + 1) as i32, 0, total)?
                .squeeze_axes(&[0])?
                .transpose_axes(&[1, 0])?
        }
        other => anyhow::bail!("unknown mel kind {other:?} (whisper|nemo)"),
    };
    mel.eval()?;
    let shape = mel.shape().to_vec();
    let data = mel.as_slice::<f32>();
    let mut buf = Vec::with_capacity(8 + data.len() * 4);
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for d in &shape {
        buf.extend_from_slice(&(*d as u32).to_le_bytes());
    }
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, &buf).with_context(|| format!("write {}", out.display()))?;
    eprintln!("mel {:?} -> {}", shape, out.display());
    Ok(())
}

fn detect_language(input: &Path, model_id: &str, stats: bool) -> Result<()> {
    let cfg_path = any2md_hub::get_file(model_id, "config.json")?;
    let config = any2md_models::WhisperConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let dir = cfg_path.parent().context("config has no parent dir")?;

    let audio = any2md_audio::read_wav_16k(input)?;
    let mel = any2md_audio::whisper_mel(&audio, config.num_mel_bins as usize)?;
    let mel = any2md_audio::pad_or_trim(&mel, any2md_models::whisper::N_FRAMES, 0)?;
    mel.eval()?;

    let started = std::time::Instant::now();
    let model = any2md_models::Whisper::load(dir, config)?;
    let (token, prob) = model.detect_language(&mel)?;
    let code = any2md_models::whisper::lang_token_to_code(token).unwrap_or("??");
    if stats {
        eprintln!(
            "mel={:?} token={} lang={} prob={:.4} time={:.2}s",
            mel.shape(),
            token,
            code,
            prob,
            started.elapsed().as_secs_f64()
        );
    }
    println!("{code} {prob:.4}");
    Ok(())
}

fn transcribe(
    input: &Path,
    model_id: &str,
    stats: bool,
    out: Option<&Path>,
    tokens_json: Option<&Path>,
    features_out: Option<&Path>,
    trace_json: Option<&Path>,
    chunk_duration: f32,
    dtype: &str,
) -> Result<()> {
    let bf16_audio = match dtype {
        "bf16" | "bfloat16" => true,
        "f32" | "float32" => false,
        other => anyhow::bail!("unknown dtype {other:?} (bf16|f32)"),
    };
    let cfg_path = any2md_hub::get_file(model_id, "config.json")?;
    let config = any2md_models::ParakeetConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let dir = cfg_path.parent().context("config has no parent dir")?;

    let started = std::time::Instant::now();
    let audio = any2md_audio::read_wav_16k(input)?;
    let mel =
        any2md_models::parakeet::mel_from_audio(&audio, &config.preprocessor, bf16_audio)?;
    let mel_secs = started.elapsed().as_secs_f64();

    let model = any2md_models::Parakeet::load(dir, config)?;

    if let Some(path) = features_out {
        let feats = model.encode(&mel)?;
        feats.eval()?;
        write_f32_tensor(path, &feats)?;
        eprintln!("features {:?} -> {}", feats.shape(), path.display());
    }

    let tokens = if chunk_duration > 0.0 {
        any2md_models::chunk::transcribe_chunked_dtype(
            &model,
            &audio,
            chunk_duration,
            any2md_models::chunk::DEFAULT_OVERLAP_SECONDS,
            bf16_audio,
        )?
    } else {
        let mut trace = Vec::new();
        let tokens = model.decode_traced(&mel, trace_json.map(|_| &mut trace))?;
        if let Some(path) = trace_json {
            std::fs::write(path, serde_json::to_vec(&trace)?)?;
        }
        tokens
    };
    let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
    let text = text.trim().to_string();

    if stats {
        let (active, cache, peak) = any2md_mlx::memory_stats()?;
        let gb = |v: usize| v as f64 / 1024.0 / 1024.0 / 1024.0;
        eprintln!(
            "mel_frames={} tokens={} mel={mel_secs:.2}s total={:.2}s peak={:.2}GB active={:.2}GB cache={:.2}GB",
            mel.shape()[0],
            tokens.len(),
            started.elapsed().as_secs_f64(),
            gb(peak),
            gb(active),
            gb(cache),
        );
    }
    if let Some(path) = tokens_json {
        let report = serde_json::json!({
            "model": model_id,
            "mel_frames": mel.shape()[0],
            "text": text,
            "tokens": tokens.iter().map(|t| serde_json::json!({
                "id": t.id, "text": t.text, "start": t.start, "duration": t.duration,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    match out {
        Some(path) => {
            std::fs::write(path, text.as_bytes())?;
            println!("{}", path.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

/// Write a `[dims] + f32le` tensor blob (same format the mel dump uses).
fn write_f32_tensor(path: &Path, array: &any2md_mlx::Array) -> Result<()> {
    array.eval()?;
    let shape = array.shape().to_vec();
    let data = array.as_slice::<f32>();
    let mut buf = Vec::with_capacity(8 + data.len() * 4);
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for d in &shape {
        buf.extend_from_slice(&(*d as u32).to_le_bytes());
    }
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &buf)?;
    Ok(())
}

/// Multilingual transcription via the native Qwen3-ASR port.
fn transcribe_asr(
    input: &Path,
    model_id: &str,
    language: Option<&str>,
    max_tokens: usize,
    stats: bool,
    out: Option<&Path>,
    tokens_json: Option<&Path>,
) -> Result<()> {
    let cfg_path = any2md_hub::get_file(model_id, "config.json")?;
    let config = any2md_models::qwen3asr::AsrConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let dir = cfg_path.parent().context("config has no parent dir")?;
    // the MLX 4-bit conversion ships only vocab.json/merges.txt; the upstream
    // `-hf` repo has an equivalent tokenizer.json (verified id-for-id)
    let tokenizer_path = any2md_hub::get_file_with_fallback(
        model_id,
        "tokenizer.json",
        &["Qwen/Qwen3-ASR-1.7B-hf", "Qwen/Qwen3-ASR-0.6B-hf"],
    )?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let audio = any2md_audio::read_wav_16k(input)?;
    let mel = any2md_audio::whisper_mel(&audio, config.thinker_config.audio_config.num_mel_bins as usize)?;

    let model = any2md_models::Qwen3Asr::load(dir, config.clone())?;
    let started = std::time::Instant::now();
    let (features, n_tokens) = model.encode_audio(&mel)?;
    features.eval()?;

    // language: accept a name or an ISO code, matching the Python helper
    let lang_code = language.map(|l| any2md_models::qwen3asr::language_name(l, &config.support_languages));
    let prompt = any2md_models::qwen3asr::build_prompt(n_tokens, lang_code.as_deref());
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
    let ids: Vec<i32> = encoding.get_ids().iter().map(|v| *v as i32).collect();
    let audio_pad_id = 151676u32; // <|audio_pad|>

    let out_ids = model.transcribe(&ids, &features, audio_pad_id, max_tokens)?;
    let text = tokenizer
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?
        .trim()
        .to_string();
    let elapsed = started.elapsed().as_secs_f64();

    if stats {
        let (active, cache, peak) = any2md_mlx::memory_stats()?;
        let gb = |v: usize| v as f64 / 1024.0 / 1024.0 / 1024.0;
        eprintln!(
            "audio_tokens={n_tokens} prompt={} out={} time={elapsed:.2}s peak={:.2}GB active={:.2}GB cache={:.2}GB",
            ids.len(),
            out_ids.len(),
            gb(peak),
            gb(active),
            gb(cache),
        );
    }
    if let Some(path) = tokens_json {
        let report = serde_json::json!({
            "model": model_id,
            "audio_tokens": n_tokens,
            "prompt_tokens": ids.len(),
            "language": lang_code,
            "text": text,
            "tokens": out_ids,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    match out {
        Some(path) => {
            std::fs::write(path, text.as_bytes())?;
            println!("{}", path.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

/// Debug hook: encoder-only inference on a dumped mel tensor.
fn nemotron_encode(
    model_id: &str,
    input_path: &Path,
    out: &Path,
    preds_out: Option<&Path>,
) -> Result<()> {
    let cfg_path = any2md_hub::get_file(model_id, "config.json")?;
    let config = any2md_models::nemotron::DiarConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let dir = cfg_path.parent().context("config has no parent dir")?;
    let mel = read_f32_tensor(input_path)?;
    let t = mel.shape()[0];
    let model = any2md_models::Nemotron::load(dir, config)?;
    let (feats, len) = model.encode(&mel, Some(t))?;
    feats.eval()?;
    write_f32_tensor(out, &feats)?;
    if let Some(path) = preds_out {
        let preds = model.speakers.forward(&feats)?;
        write_f32_tensor(path, &preds)?;
        eprintln!("speaker preds {:?} -> {}", preds.shape(), path.display());
    }
    eprintln!("encoded {t} mel frames -> {len} frames {:?}", feats.shape());
    Ok(())
}

/// Read a `[dims] + f32le` tensor blob.
fn read_f32_tensor(path: &Path) -> Result<any2md_mlx::Array> {
    let raw = std::fs::read(path)?;
    let nd = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut dims = Vec::with_capacity(nd);
    for i in 0..nd {
        let off = 4 + 4 * i;
        dims.push(i32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
    }
    let data: Vec<f32> = raw[4 + 4 * nd..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(any2md_mlx::Array::from_slice(&data, &dims))
}

/// Diarize a WAV with the native Nemotron-3 port.
fn diarize_wav(input: &Path, model_id: &str, json: bool, stats: bool, out: Option<&Path>) -> Result<()> {
    let cfg_path = any2md_hub::get_file(model_id, "config.json")?;
    let config = any2md_models::nemotron::DiarConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let dir = cfg_path.parent().context("config has no parent dir")?;
    let audio = any2md_audio::read_wav_16k(input)?;

    let started = std::time::Instant::now();
    let diar = any2md_models::Diarizer::load(dir, config)?;
    let segments = diar.generate(&audio)?;
    let elapsed = started.elapsed().as_secs_f64();

    if stats {
        let (active, cache, peak) = any2md_mlx::memory_stats()?;
        let gb = |v: usize| v as f64 / 1024.0 / 1024.0 / 1024.0;
        let speakers: std::collections::BTreeSet<usize> =
            segments.iter().map(|s| s.speaker).collect();
        eprintln!(
            "segments={} speakers={} time={elapsed:.2}s peak={:.2}GB active={:.2}GB cache={:.2}GB",
            segments.len(),
            speakers.len(),
            gb(peak),
            gb(active),
            gb(cache),
        );
    }
    let report = serde_json::json!({
        "model": model_id,
        "source": input.display().to_string(),
        "seconds": elapsed,
        "segments": segments.iter().map(|s| serde_json::json!({
            "start": s.start, "end": s.end, "speaker": s.speaker,
        })).collect::<Vec<_>>(),
    });
    if let Some(path) = out {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        for s in &segments {
            println!("{:8.2} - {:8.2}  SPEAKER_{}", s.start, s.end, s.speaker);
        }
    }
    Ok(())
}
