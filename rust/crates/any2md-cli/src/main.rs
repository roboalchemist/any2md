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
        Command::Stt { input, model, stats, out, tokens_json } => {
            transcribe(&input, &model, stats, out.as_deref(), tokens_json.as_deref())
        }
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
) -> Result<()> {
    let cfg_path = any2md_hub::get_file(model_id, "config.json")?;
    let config = any2md_models::ParakeetConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let dir = cfg_path.parent().context("config has no parent dir")?;

    let started = std::time::Instant::now();
    let mel = any2md_models::parakeet::mel_for(input, &config.preprocessor)?;
    let mel_secs = started.elapsed().as_secs_f64();

    let model = any2md_models::Parakeet::load(dir, config)?;
    let tokens = model.decode(&mel)?;
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
