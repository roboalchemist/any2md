//! Qwen3-ASR (multilingual STT): Whisper-style audio tower + Qwen3 decoder.
//!
//! Mirrors `mlx_audio/stt/models/qwen3_asr/qwen3_asr.py`:
//! - the mel is split into 100-frame windows *before* the conv stack (per-window
//!   padding changes the boundary values, so full-audio conv is not equivalent),
//! - the encoder attends in blocks of `max_len_after_cnn * n_window_infer/(n_window*2)`
//!   frames with an additive -1e9 mask,
//! - the decoder is Qwen3 (q/k RMSNorm per head, GQA, RoPE) with audio features
//!   spliced into the `<|audio_pad|>` positions.

use std::path::Path;

use any2md_mlx::ops::indexing::TryIndexOp;
use any2md_mlx::{ops, Array, Dense, Dtype, QuantEmbedding, QuantLinear, Weights};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

pub const DEFAULT_MODEL: &str = "mlx-community/Qwen3-ASR-1.7B-4bit";
pub const EOS_IDS: [u32; 2] = [151643, 151645];

#[derive(Debug, Clone, Deserialize)]
pub struct AudioCfg {
    pub d_model: i32,
    pub encoder_layers: usize,
    pub encoder_attention_heads: i32,
    pub encoder_ffn_dim: i32,
    pub num_mel_bins: i32,
    pub max_source_positions: i32,
    pub downsample_hidden_size: i32,
    pub output_dim: i32,
    #[serde(default)]
    pub scale_embedding: bool,
    pub n_window: i32,
    pub n_window_infer: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextCfg {
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub intermediate_size: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerCfg {
    pub audio_config: AudioCfg,
    pub text_config: TextCfg,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct QuantCfg {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AsrConfig {
    pub thinker_config: ThinkerCfg,
    pub quantization: QuantCfg,
    #[serde(default)]
    pub support_languages: Vec<String>,
}

impl AsrConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text).context("parse qwen3-asr config.json")?)
    }
}

// ---------------------------------------------------------------------------
// Audio tower
// ---------------------------------------------------------------------------

struct LayerNorm {
    weight: Array,
    bias: Option<Array>,
    eps: f32,
}

impl LayerNorm {
    fn load(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?.clone(),
            bias: w.maybe(&format!("{prefix}.bias")).cloned(),
            eps,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::layer_norm(
            x,
            Some(&self.weight),
            self.bias.as_ref(),
            self.eps,
        )?)
    }
}

struct AudioLayer {
    attn_ln: LayerNorm,
    q: Dense,
    k: Dense,
    v: Dense,
    out: Dense,
    fc1: Dense,
    fc2: Dense,
    ff_ln: LayerNorm,
}

impl AudioLayer {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            attn_ln: LayerNorm::load(w, &format!("{prefix}.self_attn_layer_norm"), 1e-5)?,
            q: w.dense(&format!("{prefix}.self_attn.q_proj"))?,
            k: w.dense(&format!("{prefix}.self_attn.k_proj"))?,
            v: w.dense(&format!("{prefix}.self_attn.v_proj"))?,
            out: w.dense(&format!("{prefix}.self_attn.out_proj"))?,
            fc1: w.dense(&format!("{prefix}.fc1"))?,
            fc2: w.dense(&format!("{prefix}.fc2"))?,
            ff_ln: LayerNorm::load(w, &format!("{prefix}.final_layer_norm"), 1e-5)?,
        })
    }

    fn forward(&self, x: &Array, mask: &Array, n_heads: i32, head_dim: i32) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let scale = (head_dim as f32).powf(-0.5);

        let normed = self.attn_ln.forward(x)?;
        let q = self
            .q
            .forward(&normed)?
            .multiply(&Array::from_f32(scale))?
            .reshape(&[b, l, n_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k
            .forward(&normed)?
            .reshape(&[b, l, n_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&normed)?
            .reshape(&[b, l, n_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let attn = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0,
            mlx_rs::fast::ScaledDotProductAttentionMask::Array(mask),
            None,
        )?;
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        let mut h = x.add(&self.out.forward(&attn)?)?;

        let normed = self.ff_ln.forward(&h)?;
        let ff = self.fc2.forward(&mlx_rs::nn::gelu(&self.fc1.forward(&normed)?)?)?;
        h = h.add(&ff)?;
        Ok(h)
    }
}

struct AudioTower {
    conv1: (Array, Array),
    conv2: (Array, Array),
    conv3: (Array, Array),
    conv_out: Array,
    positional: Array,
    layers: Vec<AudioLayer>,
    ln_post: LayerNorm,
    proj1: Dense,
    proj2: Dense,
    cfg: AudioCfg,
}

impl AudioTower {
    fn load(w: &Weights, cfg: AudioCfg) -> Result<Self> {
        let pair = |name: &str| -> Result<(Array, Array)> {
            Ok((
                w.get(&format!("{name}.weight"))?.clone(),
                w.get(&format!("{name}.bias"))?.reshape(&[1, 1, 1, -1])?,
            ))
        };
        let layers = (0..cfg.encoder_layers)
            .map(|i| AudioLayer::load(w, &format!("audio_tower.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1: pair("audio_tower.conv2d1")?,
            conv2: pair("audio_tower.conv2d2")?,
            conv3: pair("audio_tower.conv2d3")?,
            conv_out: w.get("audio_tower.conv_out.weight")?.clone(),
            positional: sinusoids(cfg.max_source_positions as usize, cfg.d_model as usize)?,
            layers,
            ln_post: LayerNorm::load(w, "audio_tower.ln_post", 1e-5)?,
            proj1: w.dense("audio_tower.proj1")?,
            proj2: w.dense("audio_tower.proj2")?,
            cfg,
        })
    }

    /// `mel` is `[frames, n_mels]`; returns `[tokens, output_dim]`.
    fn forward(&self, mel: &Array) -> Result<(Array, usize)> {
        let cfg = &self.cfg;
        let frames = mel.shape()[0];
        let chunk = cfg.n_window * 2; // 100 frames per conv window

        // split into windows, pad the last one, conv each (boundary values differ)
        let n_chunks = (frames + chunk - 1) / chunk;
        let lengths: Vec<i64> = (0..n_chunks)
            .map(|i| {
                let start = i * chunk;
                (frames - start).min(chunk) as i64
            })
            .collect();
        let max_len = *lengths.iter().max().unwrap_or(&0) as i32;

        // [B, n_mels, T, 1] with zero padding on the right
        let mel_t = mel.transpose_axes(&[1, 0])?; // [n_mels, frames]
        let mut chunks: Vec<Array> = Vec::with_capacity(n_chunks as usize);
        for (i, len) in lengths.iter().enumerate() {
            let start = i as i32 * chunk;
            let idx = Array::from_slice(&(start..start + *len as i32).collect::<Vec<_>>(), &[*len as i32]);
            let piece = mel_t.take_axis(&idx, 1)?; // [n_mels, len]
            let padded = if *len as i32 == max_len {
                piece
            } else {
                let pad = Array::zeros::<f32>(&[mel_t.shape()[0], max_len - *len as i32])?;
                ops::concatenate(&[&piece, &pad], 1)?
            };
            chunks.push(padded.expand_dims(0)?.expand_dims(3)?); // [1, n_mels, T, 1]
        }
        let x = ops::concatenate(&chunks, 0)?;

        let x = mlx_rs::nn::gelu(&conv_bias(&x, &self.conv1, 2)?)?;
        let x = mlx_rs::nn::gelu(&conv_bias(&x, &self.conv2, 2)?)?;
        let x = mlx_rs::nn::gelu(&conv_bias(&x, &self.conv3, 2)?)?;

        let (b, f, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let x = x.transpose_axes(&[0, 2, 3, 1])?.reshape(&[b, t, c * f])?;
        let x = x.matmul(&self.conv_out.transpose_axes(&[1, 0])?)?; // conv_out has no bias

        // keep only the valid tokens of each window and concatenate
        let after: Vec<i64> = lengths.iter().map(|l| feat_len(*l)).collect();
        let mut pieces: Vec<Array> = Vec::new();
        for (i, len) in after.iter().enumerate() {
            let idx = Array::from_slice(&(0..*len as i32).collect::<Vec<_>>(), &[*len as i32]);
            pieces.push(x.take_axis(&Array::from_slice(&[i as i32], &[1]), 0)?.take_axis(&idx, 1)?);
        }
        let hidden = ops::concatenate(&pieces, 1)?; // [1, N, d_model]
        let n = hidden.shape()[1];
        let pos = self.positional.try_index(0..n as i32)?;
        let hidden = hidden.add(&pos.expand_dims(0)?)?;

        // block attention mask: window_aftercnn = max_after_cnn * (n_window_infer / chunk)
        let max_after = *after.iter().max().unwrap_or(&0);
        let window = (max_after * (cfg.n_window_infer as i64 / chunk as i64)).max(1);
        let mut mask = vec![-1e9f32; (n * n) as usize];
        let mut start = 0i64;
        while start < n as i64 {
            let end = (start + window).min(n as i64);
            for i in start..end {
                for j in start..end {
                    mask[(i * n as i64 + j) as usize] = 0.0;
                }
            }
            start = end;
        }
        let mask = Array::from_slice(&mask, &[1, 1, n, n]);

        let mut h = hidden;
        let n_heads = cfg.encoder_attention_heads;
        let head_dim = cfg.d_model / n_heads;
        for layer in &self.layers {
            h = layer.forward(&h, &mask, n_heads, head_dim)?;
        }

        let h = self.ln_post.forward(&h)?;
        let h = mlx_rs::nn::gelu(&self.proj1.forward(&h)?)?;
        let out = self.proj2.forward(&h)?;
        Ok((out.squeeze_axes(&[0])?, n as usize))
    }
}

fn conv_bias(x: &Array, (weight, bias): &(Array, Array), stride: i32) -> Result<Array> {
    let y = ops::conv2d(x, weight, (stride, stride), (1, 1), (1, 1), 1)?;
    Ok(y.add(bias)?)
}

/// Upstream `_get_feat_extract_output_lengths` (13 tokens per 100 mel frames).
fn feat_len(input: i64) -> i64 {
    let leave = input % 100;
    let feat = (leave - 1).div_euclid(2) + 1;
    let inner = (feat - 1).div_euclid(2) + 1 - 1;
    inner.div_euclid(2) + 1 + (input / 100) * 13
}

/// Sinusoidal positions (Whisper-style) for the audio tower.
fn sinusoids(length: usize, channels: usize) -> Result<Array> {
    let half = channels / 2;
    let inc = (10_000f32).ln() / (half as f32 - 1.0);
    let inv: Vec<f32> = (0..half).map(|i| (-inc * i as f32).exp()).collect();
    let pos = Array::from_slice(
        &(0..length).map(|i| i as f32).collect::<Vec<_>>(),
        &[length as i32],
    )
    .expand_dims(1)?;
    let scaled = pos.matmul(&Array::from_slice(&inv, &[half as i32]).expand_dims(0)?)?;
    Ok(ops::concatenate(&[&scaled.sin()?, &scaled.cos()?], 1)?)
}

// ---------------------------------------------------------------------------
// Text decoder (Qwen3)
// ---------------------------------------------------------------------------

struct TextLayer {
    input_ln: Array,
    post_ln: Array,
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    q_norm: Array,
    k_norm: Array,
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

pub struct Qwen3Asr {
    pub config: AsrConfig,
    tower: AudioTower,
    embed: QuantEmbedding,
    layers: Vec<TextLayer>,
    norm: Array,
}

impl Qwen3Asr {
    pub fn load(dir: impl AsRef<Path>, config: AsrConfig) -> Result<Self> {
        let w = Weights::load_dir(dir)?;
        let q = config.quantization;
        let t = &config.thinker_config.text_config;
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(TextLayer {
                input_ln: w.get(&format!("{p}.input_layernorm.weight"))?.clone(),
                post_ln: w
                    .get(&format!("{p}.post_attention_layernorm.weight"))?
                    .clone(),
                q: w.quant_linear(&format!("{p}.self_attn.q_proj"), q.group_size, q.bits)?,
                k: w.quant_linear(&format!("{p}.self_attn.k_proj"), q.group_size, q.bits)?,
                v: w.quant_linear(&format!("{p}.self_attn.v_proj"), q.group_size, q.bits)?,
                o: w.quant_linear(&format!("{p}.self_attn.o_proj"), q.group_size, q.bits)?,
                q_norm: w.get(&format!("{p}.self_attn.q_norm.weight"))?.clone(),
                k_norm: w.get(&format!("{p}.self_attn.k_norm.weight"))?.clone(),
                gate: w.quant_linear(&format!("{p}.mlp.gate_proj"), q.group_size, q.bits)?,
                up: w.quant_linear(&format!("{p}.mlp.up_proj"), q.group_size, q.bits)?,
                down: w.quant_linear(&format!("{p}.mlp.down_proj"), q.group_size, q.bits)?,
            });
        }
        Ok(Self {
            tower: AudioTower::load(&w, config.thinker_config.audio_config.clone())?,
            embed: w.quant_embedding("model.embed_tokens", q.group_size, q.bits)?,
            norm: w.get("model.norm.weight")?.clone(),
            layers,
            config,
        })
    }

    /// Encode audio: mel `[frames, 128]` -> features `[tokens, output_dim]`.
    pub fn encode_audio(&self, mel: &Array) -> Result<(Array, usize)> {
        self.tower.forward(mel)
    }

    /// Prefill + greedy decode. `prompt` contains the audio tokens.
    pub fn transcribe(
        &self,
        prompt_ids: &[i32],
        audio: &Array,
        audio_pad_id: u32,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let cfg = &self.config.thinker_config.text_config;
        let hidden = cfg.hidden_size;
        let mut caches: Vec<KvCache> = (0..cfg.num_hidden_layers).map(|_| KvCache::new()).collect();

        let ids = Array::from_slice(prompt_ids, &[1, prompt_ids.len() as i32]);
        let mut embeds = self.embed.forward(&ids)?;

        // splice audio features into the <|audio_pad|> run (always contiguous)
        let pads: Vec<usize> = prompt_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id as u32 == audio_pad_id)
            .map(|(i, _)| i)
            .collect();
        if !pads.is_empty() {
            if pads.len() as i32 != audio.shape()[0] {
                return Err(anyhow!(
                    "prompt has {} audio slots but encoder produced {} tokens",
                    pads.len(),
                    audio.shape()[0]
                ));
            }
            let first = pads[0] as i32;
            let last = *pads.last().unwrap() as i32;
            let head_idx: Vec<i32> = (0..first).collect();
            let tail_idx: Vec<i32> = (last + 1..prompt_ids.len() as i32).collect();
            let head = embeds.take_axis(
                &Array::from_slice(&head_idx, &[head_idx.len() as i32]),
                1,
            )?;
            let tail = embeds.take_axis(
                &Array::from_slice(&tail_idx, &[tail_idx.len() as i32]),
                1,
            )?;
            let audio_rows = audio.expand_dims(0)?; // [1, n_pad, hidden]
            embeds = ops::concatenate(&[&head, &audio_rows, &tail], 1)?;
        }

        let mut h = self.forward_layers(&embeds, &mut caches, hidden)?;
        let mut logits = self.logits(&h, &mut caches)?;
        let mut out = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            logits.eval()?;
            let next = argmax(&logits)?;
            if EOS_IDS.contains(&next) {
                break;
            }
            out.push(next);
            let ids = Array::from_slice(&[next as i32], &[1, 1]);
            let emb = self.embed.forward(&ids)?;
            let h = self.forward_layers(&emb, &mut caches, hidden)?;
            logits = self.logits(&h, &mut caches)?;
        }
        Ok(out)
    }

    fn forward_layers(
        &self,
        embeds: &Array,
        caches: &mut [KvCache],
        hidden: i32,
    ) -> Result<Array> {
        let cfg = &self.config.thinker_config.text_config;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let scale = (head_dim as f32).powf(-0.5);
        let l = embeds.shape()[1];

        let mut h = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = any2md_mlx::rms_norm(&h, &layer.input_ln, cfg.rms_norm_eps)?;
            let q = layer
                .q
                .forward(&normed)?
                .reshape(&[1, l, n_heads, head_dim])?;
            let k = layer
                .k
                .forward(&normed)?
                .reshape(&[1, l, n_kv, head_dim])?;
            let v = layer
                .v
                .forward(&normed)?
                .reshape(&[1, l, n_kv, head_dim])?;

            // per-head RMSNorm (Qwen3)
            let q = any2md_mlx::rms_norm(&q, &layer.q_norm, cfg.rms_norm_eps)?;
            let k = any2md_mlx::rms_norm(&k, &layer.k_norm, cfg.rms_norm_eps)?;

            let q = q.transpose_axes(&[0, 2, 1, 3])?;
            let k = k.transpose_axes(&[0, 2, 1, 3])?;
            let v = v.transpose_axes(&[0, 2, 1, 3])?;

            let offset = caches[i].offset();
            let q = mlx_rs::fast::rope(&q, head_dim, false, Some(cfg.rope_theta), 1.0, offset, None)?;
            let k = mlx_rs::fast::rope(&k, head_dim, false, Some(cfg.rope_theta), 1.0, offset, None)?;
            let (k, v) = caches[i].update_and_fetch(k, v)?;

            let attn = if l > 1 {
                mlx_rs::fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal),
                    None,
                )?
            } else {
                mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?
            };
            let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, l, hidden])?;
            h = h.add(&layer.o.forward(&attn)?)?;

            let normed = any2md_mlx::rms_norm(&h, &layer.post_ln, cfg.rms_norm_eps)?;
            let gate = layer.gate.forward(&normed)?;
            let up = layer.up.forward(&normed)?;
            let act = any2md_mlx::silu(&gate)?.multiply(&up)?;
            h = h.add(&layer.down.forward(&act)?)?;
        }
        Ok(h)
    }

    fn logits(&self, h: &Array, _caches: &mut [KvCache]) -> Result<Array> {
        let cfg = &self.config.thinker_config.text_config;
        let normed = any2md_mlx::rms_norm(h, &self.norm, cfg.rms_norm_eps)?;
        let logits = self.embed.as_linear(&normed)?;
        let last = logits.squeeze_axes(&[0])?;
        let n = last.shape()[0];
        let last = last.take_axis(&Array::from_slice(&[n - 1], &[1]), 0)?;
        Ok(last.as_dtype(Dtype::Float32)?)
    }
}

fn argmax(x: &Array) -> Result<u32> {
    let idx = mlx_rs::ops::indexing::argmax(x, None)?;
    idx.eval()?;
    Ok(idx.as_slice::<u32>().first().copied().unwrap_or(0))
}

/// Sliding KV cache (same shape convention as the Qwen2 port).
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl KvCache {
    fn new() -> Self {
        Self { keys: None, values: None, offset: 0 }
    }
    fn offset(&self) -> i32 {
        self.offset
    }
    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array)> {
        let len = keys.shape()[2];
        match (self.keys.take(), self.values.take()) {
            (None, None) => {
                self.offset = len;
                self.keys = Some(keys.clone());
                self.values = Some(values.clone());
                Ok((keys, values))
            }
            (Some(k), Some(v)) => {
                self.offset += len;
                let (k, v) = (
                    ops::concatenate(&[&k, &keys], 2)?,
                    ops::concatenate(&[&v, &values], 2)?,
                );
                self.keys = Some(k.clone());
                self.values = Some(v.clone());
                Ok((k, v))
            }
            _ => Err(anyhow!("inconsistent KV cache")),
        }
    }
}

/// Build the exact prompt `Qwen3ASRModel._build_prompt` uses.
pub fn build_prompt(num_audio_tokens: usize, language: Option<&str>) -> String {
    let assistant_prefix = match language {
        Some(l) => format!("language {l}<asr_text>"),
        None => String::new(),
    };
    format!(
        "<|im_start|>system\n<|im_end|>\n\
         <|im_start|>user\n<|audio_start|>{}<|audio_end|><|im_end|>\n\
         <|im_start|>assistant\n{assistant_prefix}",
        "<|audio_pad|>".repeat(num_audio_tokens)
    )
}

/// Canonical language name for codes/aliases (Qwen3-ASR expects names).
pub fn language_name(language: &str, supported: &[String]) -> String {
    let lower = language.to_lowercase();
    if let Some(hit) = supported.iter().find(|l| l.to_lowercase() == lower) {
        return hit.clone();
    }
    let code = iso_code(&lower).unwrap_or(lower.as_str());
    supported
        .iter()
        .find(|l| l.to_lowercase().starts_with(code))
        .cloned()
        .unwrap_or_else(|| language.to_string())
}

/// Minimal ISO 639-1 -> name mapping for the languages any2md routes here.
fn iso_code(code: &str) -> Option<&'static str> {
    Some(match code.split('-').next().unwrap_or(code) {
        "zh" => "chinese",
        "en" => "english",
        "yue" => "cantonese",
        "ja" => "japanese",
        "ko" => "korean",
        "ar" => "arabic",
        "de" => "german",
        "fr" => "french",
        "es" => "spanish",
        "pt" => "portuguese",
        "it" => "italian",
        "ru" => "russian",
        "id" => "indonesian",
        "th" => "thai",
        "vi" => "vietnamese",
        "tr" => "turkish",
        "hi" => "hindi",
        "ms" => "malay",
        "nl" => "dutch",
        "sv" => "swedish",
        "da" => "danish",
        "fi" => "finnish",
        "pl" => "polish",
        "cs" => "czech",
        "fil" | "tl" => "filipino",
        "fa" => "persian",
        "el" => "greek",
        "ro" => "romanian",
        "hu" => "hungarian",
        "mk" => "macedonian",
        _ => return None,
    })
}
