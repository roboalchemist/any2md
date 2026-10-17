//! Whisper (encoder + text decoder) — used by any2md purely for language ID.
//!
//! Mirrors `mlx_audio/stt/models/whisper/whisper.py` including its
//! hand-written attention (q·kᵀ with `(C/H)^-0.25` scaling on both sides,
//! `softmax(precise=True)`) rather than a fused SDPA, because the LID decision
//! is made from a single logit row and we want bit-comparable numbers.

use std::path::Path;

use any2md_mlx::ops::indexing::TryIndexOp;
use any2md_mlx::{ops, Array, Dense, Dtype, Weights};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

pub const DEFAULT_LID_MODEL: &str = "mlx-community/whisper-tiny-asr-fp16";
/// `<|startoftranscript|>` and the `<|lang|>` block for multilingual Whisper.
pub const SOT: u32 = 50258;
pub const LANG_FIRST: u32 = 50259;
pub const LANG_LAST: u32 = 50357;
pub const N_FRAMES: usize = 3000;

#[derive(Debug, Clone, Deserialize)]
pub struct WhisperConfig {
    pub d_model: i32,
    pub num_mel_bins: i32,
    pub encoder_layers: usize,
    pub encoder_attention_heads: i32,
    pub decoder_layers: usize,
    pub decoder_attention_heads: i32,
    pub max_source_positions: i32,
    pub max_target_positions: i32,
    pub vocab_size: i32,
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

impl WhisperConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text).context("parse whisper config.json")?)
    }

    fn dtype(&self) -> Dtype {
        match self.torch_dtype.as_deref() {
            Some("float16") | Some("half") => Dtype::Float16,
            Some("bfloat16") => Dtype::Bfloat16,
            _ => Dtype::Float32,
        }
    }
}

struct LayerNorm {
    weight: Array,
    bias: Array,
}

impl LayerNorm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?.clone(),
            bias: w.get(&format!("{prefix}.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-5)?)
    }
}

struct Mha {
    query: Dense,
    key: Dense,
    value: Dense,
    out: Dense,
    n_head: i32,
    n_state: i32,
}

impl Mha {
    fn load(w: &Weights, prefix: &str, n_state: i32, n_head: i32) -> Result<Self> {
        Ok(Self {
            query: w.dense(&format!("{prefix}.query"))?,
            key: w.dense(&format!("{prefix}.key"))?,
            value: w.dense(&format!("{prefix}.value"))?,
            out: w.dense(&format!("{prefix}.out"))?,
            n_head,
            n_state,
        })
    }

    /// `x` supplies queries (`[B, T, C]`); `xa` supplies keys/values when cross-attending.
    fn forward(&self, x: &Array, xa: Option<&Array>) -> Result<Array> {
        let (b, t, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let head_dim = self.n_state / self.n_head;
        let scale = (head_dim as f32).powf(-0.25);

        let q = self.query.forward(x)?;
        let (k_src, v_src) = (xa.unwrap_or(x), xa.unwrap_or(x));
        let k = self.key.forward(k_src)?;
        let v = self.value.forward(v_src)?;
        let s = k.shape()[1];

        let q = q
            .reshape(&[b, t, self.n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?
            .multiply(&Array::from_f32(scale))?;
        let k = k
            .reshape(&[b, s, self.n_head, head_dim])?
            .transpose_axes(&[0, 2, 3, 1])?
            .multiply(&Array::from_f32(scale))?;
        let v = v
            .reshape(&[b, s, self.n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let qk = q.matmul(&k)?;
        // `precise=True` in MLX: softmax computed in float32.
        let w = ops::softmax_axis(&qk.as_dtype(Dtype::Float32)?, -1, true)?;
        let out = w.as_dtype(qk.dtype())?.matmul(&v)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, self.n_state])?;
        self.out.forward(&out)
    }
}

struct Block {
    attn: Mha,
    attn_ln: LayerNorm,
    cross_attn: Option<Mha>,
    cross_attn_ln: Option<LayerNorm>,
    mlp1: Dense,
    mlp2: Dense,
    mlp_ln: LayerNorm,
}

impl Block {
    fn load(w: &Weights, prefix: &str, n_state: i32, n_head: i32, cross: bool) -> Result<Self> {
        let (cross_attn, cross_attn_ln) = if cross {
            (
                Some(Mha::load(w, &format!("{prefix}.cross_attn"), n_state, n_head)?),
                Some(LayerNorm::load(w, &format!("{prefix}.cross_attn_ln"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            attn: Mha::load(w, &format!("{prefix}.attn"), n_state, n_head)?,
            attn_ln: LayerNorm::load(w, &format!("{prefix}.attn_ln"))?,
            cross_attn,
            cross_attn_ln,
            mlp1: w.dense(&format!("{prefix}.mlp1"))?,
            mlp2: w.dense(&format!("{prefix}.mlp2"))?,
            mlp_ln: LayerNorm::load(w, &format!("{prefix}.mlp_ln"))?,
        })
    }

    fn forward(&self, x: &Array, xa: Option<&Array>) -> Result<Array> {
        let mut h = x.add(&self.attn.forward(&self.attn_ln.forward(x)?, None)?)?;
        if let (Some(cross), Some(cross_ln)) = (&self.cross_attn, &self.cross_attn_ln) {
            let xa = xa.ok_or_else(|| anyhow!("decoder block needs encoder features"))?;
            h = h.add(&cross.forward(&cross_ln.forward(&h)?, Some(xa))?)?;
        }
        let mlp = mlp_forward(&self.mlp1, &self.mlp2, &self.mlp_ln.forward(&h)?)?;
        Ok(h.add(&mlp)?)
    }
}

fn mlp_forward(mlp1: &Dense, mlp2: &Dense, x: &Array) -> Result<Array> {
    let h = mlp1.forward(x)?;
    let h = mlx_rs::nn::gelu(&h)?;
    mlp2.forward(&h)
}

pub struct Whisper {
    pub config: WhisperConfig,
    conv1: Dense, // stored as (out, k, in) with a bias; see conv_forward below
    conv2: Dense,
    conv1_kernel: i32,
    conv2_kernel: i32,
    positional: Array,
    enc_blocks: Vec<Block>,
    enc_ln_post: LayerNorm,
    token_embedding: Array,
    dec_positional: Array,
    dec_blocks: Vec<Block>,
    dec_ln: LayerNorm,
}

impl Whisper {
    pub fn load(dir: impl AsRef<Path>, config: WhisperConfig) -> Result<Self> {
        let w = Weights::load_dir(dir)?;
        let n_state = config.d_model;
        let dtype = config.dtype();

        let conv_weight = |name: &str| -> Result<Array> {
            let a = w.get(&format!("{name}.weight"))?.as_dtype(dtype)?;
            Ok(a)
        };
        let conv_bias = |name: &str| -> Result<Array> {
            Ok(w.get(&format!("{name}.bias"))?.as_dtype(dtype)?.reshape(&[1, 1, -1])?)
        };

        let enc_blocks = (0..config.encoder_layers)
            .map(|i| {
                Block::load(
                    &w,
                    &format!("encoder.blocks.{i}"),
                    n_state,
                    config.encoder_attention_heads,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let dec_blocks = (0..config.decoder_layers)
            .map(|i| {
                Block::load(
                    &w,
                    &format!("decoder.blocks.{i}"),
                    n_state,
                    config.decoder_attention_heads,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            conv1: Dense {
                weight: conv_weight("encoder.conv1")?,
                bias: Some(conv_bias("encoder.conv1")?),
            },
            conv2: Dense {
                weight: conv_weight("encoder.conv2")?,
                bias: Some(conv_bias("encoder.conv2")?),
            },
            conv1_kernel: 3,
            conv2_kernel: 3,
            positional: sinusoids(config.max_source_positions as usize, n_state as usize)?
                .as_dtype(dtype)?,
            enc_blocks,
            enc_ln_post: LayerNorm::load(&w, "encoder.ln_post")?,
            token_embedding: w.get("decoder.token_embedding.weight")?.as_dtype(dtype)?,
            dec_positional: w.get("decoder.positional_embedding")?.as_dtype(dtype)?,
            dec_blocks,
            dec_ln: LayerNorm::load(&w, "decoder.ln")?,
            config,
        })
    }

    fn conv(&self, x: &Array, conv: &Dense, kernel: i32, stride: i32) -> Result<Array> {
        let y = ops::conv1d(x, &conv.weight, stride, kernel / 2, 1, 1)?;
        let y = y.add(conv.bias.as_ref().expect("conv bias"))?;
        Ok(mlx_rs::nn::gelu(&y)?)
    }

    /// `mel` must be `[frames, n_mels]` (time-major) and is padded/trimmed to 3000.
    pub fn encode(&self, mel: &Array) -> Result<Array> {
        let cfg = &self.config;
        let dtype = cfg.dtype();
        let mel = any2md_audio_pad_or_trim(mel, N_FRAMES)?;
        let x = mel.as_dtype(dtype)?.expand_dims(0)?; // [1, 3000, 80]
        let x = self.conv(&x, &self.conv1, self.conv1_kernel, 1)?;
        let x = self.conv(&x, &self.conv2, self.conv2_kernel, 2)?;
        // [1, 1500, 384] + positional [1500, 384]
        let mut h = x.add(&self.positional)?;
        for block in &self.enc_blocks {
            h = block.forward(&h, None)?;
        }
        self.enc_ln_post.forward(&h)
    }

    /// Decoder logits for `tokens` `[n]` against encoder output `audio`.
    pub fn logits(&self, tokens: &[u32], audio: &Array) -> Result<Array> {
        let dtype = self.config.dtype();
        let idx = Array::from_slice(
            &tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
            &[1, tokens.len() as i32],
        );
        let emb = self.token_embedding.take_axis(&idx, 0)?;
        let pos = self
            .dec_positional
            .try_index(0..tokens.len() as i32)?
            .expand_dims(0)?;
        let mut h = emb.add(&pos)?.as_dtype(dtype)?;
        for block in &self.dec_blocks {
            h = block.forward(&h, Some(audio))?;
        }
        let h = self.dec_ln.forward(&h)?;
        // logits = x @ token_embedding.T
        let proj = h.matmul(&self.token_embedding.transpose_axes(&[1, 0])?.as_dtype(dtype)?)?;
        Ok(proj.as_dtype(Dtype::Float32)?)
    }

    /// Language identification: `(language_token_id, probability)`.
    ///
    /// Mirrors `mlx_audio...whisper.decoding.detect_language` — a single decoder
    /// step on `<|startoftranscript|>`, then the softmax restricted to the
    /// `<|lang|>` token block.
    pub fn detect_language(&self, mel: &Array) -> Result<(u32, f32)> {
        let audio = self.encode(mel)?;
        let logits = self.logits(&[SOT], &audio)?.squeeze()?;

        let n_vocab = logits.shape()[0];
        let mut masked = vec![f32::NEG_INFINITY; n_vocab as usize];
        let all = logits.as_slice::<f32>();
        for id in LANG_FIRST..=LANG_LAST {
            masked[id as usize] = all[id as usize];
        }
        let mask = Array::from_slice(&masked, &[n_vocab]);
        let probs = ops::softmax_axis(&mask, -1, true)?;
        probs.eval()?;
        let p = probs.as_slice::<f32>();

        let mut best = (LANG_FIRST, f32::NEG_INFINITY);
        for id in LANG_FIRST..=LANG_LAST {
            let v = p[id as usize];
            if v > best.1 {
                best = (id, v);
            }
        }
        Ok(best)
    }
}

/// Whisper's sinusoidal positional embedding (computed, not stored).
pub fn sinusoids(length: usize, channels: usize) -> Result<Array> {
    let half = channels / 2;
    let log_increment = (10_000f32).ln() / (half as f32 - 1.0);
    let inv: Vec<f32> = (0..half)
        .map(|i| (-log_increment * i as f32).exp())
        .collect();
    let inv_ts = Array::from_slice(&inv, &[half as i32]);
    let time = Array::from_slice(
        &(0..length).map(|i| i as f32).collect::<Vec<_>>(),
        &[length as i32],
    )
    .expand_dims(1)?;
    let scaled = time.matmul(&inv_ts.expand_dims(0)?)?;
    Ok(ops::concatenate(&[&scaled.sin()?, &scaled.cos()?], 1)?)
}

/// Pad/trim frames to Whisper's 30 s window (axis 0, time-major).
fn any2md_audio_pad_or_trim(mel: &Array, frames: usize) -> Result<Array> {
    any2md_audio::pad_or_trim(mel, frames, 0)
}

/// Map a Whisper language token id to an ISO 639-1 code.
pub fn lang_token_to_code(id: u32) -> Option<&'static str> {
    // Order of mlx_audio's whisper tokenizer LANGUAGES (100 entries)
    const LANGS: [&str; 100] = ["en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv","it","id","hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn","bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc","ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn","mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su","yue"];
    LANGS.get((id - LANG_FIRST) as usize).copied()
}
