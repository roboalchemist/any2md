//! Audio I/O and log-mel front-ends.
//!
//! Three variants are needed by the ported models and each mirrors its upstream
//! Python implementation exactly (the models are sensitive to preprocessing):
//!
//! | variant | used by | notes |
//! |---|---|---|
//! | [`whisper_mel`] | Whisper LID | no pre-emphasis, log10, clamp to max-8, (x+4)/4, drops the Nyquist bin |
//! | [`nemo_mel`] | Parakeet, Sortformer, Nemotron | pre-emphasis 0.97, natural log, `log(x+guard)`, per-feature normalisation |
//!
//! Triangulated mel filterbanks are shipped as small safetensors assets (see
//! `scripts/gen_mel_filters.py`) so the Rust side never has to re-derive
//! float-sensitive filter maths; the STFT itself runs as MLX ops on the GPU.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use any2md_mlx::ops::indexing::TryIndexOp;
use any2md_mlx::{ops, Array, Weights};

pub use mlx_rs;

/// Read a mono 16 kHz PCM WAV into `[n_samples]` float32.
///
/// anything else is rejected: the CLI converts to 16 kHz mono WAV with ffmpeg
/// first, exactly like the Python implementation.
pub fn read_wav_16k(path: impl AsRef<Path>) -> Result<Array> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!(
            "{}: expected mono audio (got {} channels); convert to 16 kHz mono WAV first",
            path.display(),
            spec.channels
        );
    }
    if spec.sample_rate != 16_000 {
        bail!(
            "{}: expected 16 kHz audio (got {} Hz); convert with ffmpeg first",
            path.display(),
            spec.sample_rate
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .context("read float samples")?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<_, _>>()
                .context("read int samples")?
        }
    };
    Ok(Array::from_slice(&samples, &[samples.len() as i32]))
}

/// Location of the bundled mel-filterbank assets.
pub fn asset_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("ANY2MD_ASSETS") {
        return Ok(PathBuf::from(dir));
    }
    // crates/any2md-audio/src/lib.rs -> rust/assets
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    Ok(here.join("../../assets"))
}

/// Load one filterbank asset (`[n_mels, n_freqs]`, float32).
pub fn load_filterbank(name: &str) -> Result<Array> {
    let path = asset_dir()?.join(name);
    let weights = Weights::load_dir(path.parent().context("asset dir")?)?;
    let key = path
        .file_name()
        .and_then(|f| f.to_str())
        .and_then(|f| f.strip_suffix(".safetensors"))
        .context("asset name")?;
    Ok(weights.get(key)?.clone())
}

/// Hann window. `periodic=true` divides by `size` (spectral analysis, Whisper);
/// `periodic=false` divides by `size-1` (symmetric, NeMo/Parakeet/Sortformer).
pub fn hann(size: usize, periodic: bool) -> Array {
    let denom = if periodic { size as f32 } else { (size - 1) as f32 };
    let data: Vec<f32> = (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / denom).cos()))
        .collect();
    Array::from_slice(&data, &[size as i32])
}

fn reflect_pad(x: &Array, pad: usize) -> Result<Array> {
    // mlx_audio.dsp.stft: prefix = x[1:pad+1][::-1], suffix = x[-(pad+1):-1][::-1]
    let n = x.shape()[0] as usize;
    if n <= pad + 1 {
        bail!("signal too short for reflect padding ({n} samples, pad {pad})");
    }
    let prefix_idx: Vec<i32> = (1..=pad).rev().map(|i| i as i32).collect();
    let suffix_idx: Vec<i32> = ((n - pad - 1)..=(n - 2)).rev().map(|i| i as i32).collect();
    let prefix = x.take_axis(&Array::from_slice(&prefix_idx, &[pad as i32]), 0)?;
    let suffix = x.take_axis(&Array::from_slice(&suffix_idx, &[pad as i32]), 0)?;
    Ok(ops::concatenate(&[&prefix, x, &suffix], 0)?)
}

/// STFT magnitudes² with optional centre padding, mirroring `mlx_audio.dsp.stft`.
///
/// Returns `[n_freqs, n_frames]` power values (`n_freqs = n_fft/2 + 1`).
pub fn stft_power(
    x: &Array,
    n_fft: usize,
    hop: usize,
    win_length: usize,
    center: bool,
    reflect: bool,
) -> Result<Array> {
    stft_power_window(x, n_fft, hop, win_length, center, reflect, reflect)
}

/// STFT with an explicit window periodic/symmetric choice.
pub fn stft_power_window(
    x: &Array,
    n_fft: usize,
    hop: usize,
    win_length: usize,
    center: bool,
    reflect: bool,
    window_periodic: bool,
) -> Result<Array> {
    let mut window = hann(win_length, window_periodic);
    if win_length < n_fft {
        let left = (n_fft - win_length) / 2;
        let right = n_fft - win_length - left;
        let zeros_l = Array::zeros::<f32>(&[left as i32])?;
        let zeros_r = Array::zeros::<f32>(&[right as i32])?;
        window = ops::concatenate(&[&zeros_l, &window, &zeros_r], 0)?;
    }

    let mut signal = x.clone();
    if center {
        signal = if reflect {
            reflect_pad(&signal, n_fft / 2)?
        } else {
            let pad = Array::zeros::<f32>(&[(n_fft / 2) as i32])?;
            ops::concatenate(&[&pad, &signal, &pad], 0)?
        };
    }

    let n = signal.shape()[0];
    if n < n_fft as i32 {
        bail!("signal too short for STFT: {n} samples < n_fft {n_fft}");
    }
    let num_frames = 1 + (n as usize - n_fft) / hop;

    let frames = signal.as_strided(&[num_frames as i32, n_fft as i32], &[hop as i64, 1i64], None)?;
    let windowed = frames.multiply(&window)?;
    let spec = mlx_rs::fft::rfft(&windowed, n_fft as i32, -1)?;
    // [frames, n_freqs] power, same orientation as mlx_audio.dsp.stft
    Ok(spec.abs()?.square()?)
}

/// NeMo-style log-mel (Parakeet / Sortformer / Nemotron).
#[derive(Debug, Clone)]
pub struct NemoMel {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub preemph: f32,
    pub log_guard: f32,
    pub norm_constant: f32,
    /// `"per_feature"`, `"global"` or `"none"`.
    pub normalize: String,
    /// Pad feature frames up to a multiple of this (0 = no padding).
    pub pad_to: usize,
    /// Pad the audio itself to at least this many samples (0 = no padding).
    pub pad_to_samples: usize,
    pub pad_value: f32,
    pub reflect_pad: bool,
}

impl NemoMel {
    /// Parakeet's `preprocessor_config`.
    pub fn parakeet() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            hop: 160,
            win_length: 400,
            n_mels: 128,
            preemph: 0.97,
            log_guard: 5.960_464_5e-8,
            norm_constant: 1e-5,
            normalize: "per_feature".into(),
            pad_to: 16,
            pad_to_samples: 0,
            pad_value: 0.0,
            reflect_pad: false,
        }
    }

    /// Sortformer / Nemotron diarization front-end.
    pub fn sortformer() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            hop: 160,
            win_length: 400,
            n_mels: 128,
            preemph: 0.97,
            log_guard: 5.960_464_5e-8,
            norm_constant: 1e-5,
            normalize: "per_feature".into(),
            pad_to: 16,
            pad_to_samples: 0,
            pad_value: 0.0,
            reflect_pad: false,
        }
    }

    pub fn filterbank(&self) -> Result<Array> {
        load_filterbank(&format!("mel_nemo_{}_{}.safetensors", self.n_fft, self.n_mels))
    }
}

/// Compute NeMo log-mel features `[n_mels, n_frames]`.
pub fn nemo_mel(x: &Array, cfg: &NemoMel) -> Result<Array> {
    let mut audio = x.clone();
    if cfg.pad_to_samples > 0 && audio.shape()[0] < cfg.pad_to_samples as i32 {
        let pad = Array::from_slice(
            &vec![cfg.pad_value; cfg.pad_to_samples - audio.shape()[0] as usize],
            &[cfg.pad_to_samples as i32 - audio.shape()[0]],
        );
        audio = ops::concatenate(&[&audio, &pad], 0)?;
    }

    if cfg.preemph > 0.0 {
        let n = audio.shape()[0];
        // y[0] = x[0]; y[n] = x[n] - a*x[n-1]  (same as upstream's concat form)
        let first = audio.try_index(0..1)?;
        let rest = audio.try_index(1..n)?;
        let prev = audio.try_index(0..n - 1)?;
        let filtered = rest.subtract(&prev.multiply(&Array::from_f32(cfg.preemph))?)?;
        audio = ops::concatenate(&[&first, &filtered], 0)?;
    }

    let power = stft_power_window(&audio, cfg.n_fft, cfg.hop, cfg.win_length, true, cfg.reflect_pad, false)?;
    let filters = cfg.filterbank()?;
    // filters [mels, freqs] @ power^T [freqs, frames] -> [mels, frames]
    let mel = filters.matmul(&power.transpose_axes(&[1, 0])?)?;
    let log_guard = Array::from_f32(cfg.log_guard);
    let mut out = mel.add(&log_guard)?.log()?;

    let frames = out.shape()[1];
    match cfg.normalize.as_str() {
        "per_feature" => {
            let mean = out.mean_axes(&[1], true)?;
            let centered = out.subtract(&mean)?;
            let var = centered.square()?.sum_axes(&[1], true)?;
            let n = (frames - 1).max(1) as f32;
            let std = var.divide(&Array::from_f32(n))?.sqrt()?;
            out = centered.divide(&std.add(&Array::from_f32(cfg.norm_constant))?)?;
        }
        "global" => {
            let mean = out.mean(None)?;
            let centered = out.subtract(&mean)?;
            let var = centered.square()?.mean(None)?;
            let std = var.sqrt()?.add(&Array::from_f32(cfg.norm_constant))?;
            out = centered.divide(&std)?;
        }
        "none" => {}
        other => bail!("unknown mel normalisation {other:?}"),
    }

    if cfg.pad_to > 1 {
        let rem = frames as usize % cfg.pad_to;
        if rem != 0 {
            let pad = cfg.pad_to - rem;
            let zeros = Array::zeros::<f32>(&[out.shape()[0], pad as i32])?;
            out = ops::concatenate(&[&out, &zeros], 1)?;
        }
    }
    Ok(out)
}

/// Whisper log-mel `[n_mels, n_frames]` (LID path only needs 30 s).
pub fn whisper_mel(x: &Array, n_mels: usize) -> Result<Array> {
    const N_FFT: usize = 400;
    const HOP: usize = 160;
    let power = stft_power(x, N_FFT, HOP, N_FFT, true, true)?;
    // Whisper slices off the last frame (not the Nyquist bin): stft is [frames, freqs],
    // so `freqs[:-1]` in mlx_audio.whisper.audio drops a frame.
    let n_frames = power.shape()[0];
    let power = power.try_index(0..n_frames - 1)?;
    let filters = load_filterbank(&format!("mel_whisper_{n_mels}.safetensors"))?;
    // [frames, freqs] @ filters^T -> [frames, mels] (time-major, as the encoder expects)
    let mel = power.matmul(&filters.transpose_axes(&[1, 0])?)?;

    let log_spec = ops::maximum(&mel, &Array::from_f32(1e-10))?.log10()?;
    let max = log_spec.max(None)?;
    let floor = max.subtract(&Array::from_f32(8.0))?;
    let clamped = ops::maximum(&log_spec, &floor)?;
    Ok(clamped.add(&Array::from_f32(4.0))?.divide(&Array::from_f32(4.0))?)
}

/// Trim or zero-pad `axis` to `len`.
pub fn pad_or_trim(x: &Array, len: usize, axis: i32) -> Result<Array> {
    let ax = axis.rem_euclid(x.ndim() as i32) as usize;
    let cur = x.shape()[ax] as usize;
    if cur == len {
        return Ok(x.clone());
    }
    if cur > len {
        // Trim by taking the first `len` indices along `axis` (axis-generic).
        let idx: Vec<i32> = (0..len as i32).collect();
        let selector = Array::from_slice(&idx, &[len as i32]);
        return Ok(x.take_axis(&selector, ax as i32)?);
    }
    let mut pad_shape = x.shape().to_vec();
    pad_shape[ax] = (len - cur) as i32;
    let zeros = Array::zeros::<f32>(&pad_shape)?;
    Ok(ops::concatenate(&[x, &zeros], ax as i32)?)
}

/// Convenience: reject non-16 kHz inputs with an actionable message.
pub fn ensure_16k(path: &Path) -> Result<()> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(anyhow!(
            "{}: need 16 kHz mono WAV (got {} Hz, {} ch) — convert with ffmpeg first",
            path.display(),
            spec.sample_rate,
            spec.channels
        ));
    }
    Ok(())
}
