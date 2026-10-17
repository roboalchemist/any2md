//! Nemotron-3-Diarization: FastConformer-style encoder + Sortformer speaker head.
//!
//! Mirrors `mlx_audio/vad/models/nemotron_diarization/nemotron_diarization.py`:
//! a bias-free feature-stacking projection (128*8 -> 512), 31 transformer
//! blocks with fused qkv + RoPE and a valid-length key mask, then the speaker
//! modules (512 -> 192 projection, subpixel upsampling conv that restores the
//! 8x time resolution, ReLU -> 192 -> 8 -> sigmoid).

use std::path::Path;

use any2md_mlx::{ops, Array, Dense, Dtype, Weights};
use anyhow::{Context, Result};
use serde::Deserialize;

pub const DEFAULT_MODEL: &str = "mlx-community/Nemotron-3-Diarization";

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderCfg {
    pub feat_in: i32,
    pub d_model: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub subsampling_factor: i32,
    pub ff_expansion: f32,
    #[serde(default)]
    pub qkv_bias: bool,
    #[serde(default)]
    pub qk_norm: bool,
    #[serde(default)]
    pub pre_block_norm: bool,
    #[serde(default)]
    pub xscaling: bool,
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    #[serde(default = "default_rotary")]
    pub rotary_fraction: f32,
}

fn default_rope_base() -> f32 {
    10_000.0
}
fn default_rotary() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModulesCfg {
    pub num_speakers: usize,
    pub fc_d_model: i32,
    pub tf_d_model: i32,
    pub subsampling_factor: i32,
    pub chunk_len: i32,
    pub fifo_len: i32,
    pub spkcache_len: i32,
    pub spkcache_update_period: i32,
    pub chunk_right_context: i32,
    #[serde(default)]
    pub spkcache_sil_frames_per_spk: i32,
    #[serde(default)]
    pub pred_score_threshold: f32,
    #[serde(default)]
    pub scores_boost_latest: f32,
    #[serde(default)]
    pub strong_boost_rate: f32,
    #[serde(default)]
    pub weak_boost_rate: f32,
    #[serde(default)]
    pub min_pos_scores_rate: f32,
    #[serde(default)]
    pub max_index: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorCfg {
    pub sampling_rate: usize,
    pub feature_size: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    #[serde(default)]
    pub preemphasis: f32,
    #[serde(default)]
    pub pad_to: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiarConfig {
    pub num_speakers: usize,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    pub encoder_config: EncoderCfg,
    pub modules_config: ModulesCfg,
    #[serde(default)]
    pub processor_config: Option<ProcessorCfg>,
}

fn default_dtype() -> String {
    "float32".into()
}

impl DiarConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text).context("parse nemotron config.json")?)
    }
    pub fn dtype(&self) -> Dtype {
        match self.dtype.as_str() {
            "bfloat16" => Dtype::Bfloat16,
            "float16" => Dtype::Float16,
            _ => Dtype::Float32,
        }
    }
}

struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?.clone(),
            bias: w.get(&format!("{prefix}.bias"))?.clone(),
            eps: 1e-5,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::layer_norm(
            x,
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
        )?)
    }
}

struct Block {
    norm1: LayerNorm,
    w_qkv: Dense,
    out_proj: Dense,
    norm2: LayerNorm,
    lin1: Dense,
    lin2: Dense,
}

impl Block {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::load(w, &format!("{prefix}.norm1"))?,
            w_qkv: w.dense(&format!("{prefix}.attn.w_qkv"))?,
            out_proj: w.dense(&format!("{prefix}.attn.out_proj"))?,
            norm2: LayerNorm::load(w, &format!("{prefix}.norm2"))?,
            lin1: w.dense(&format!("{prefix}.ffn.linear1"))?,
            lin2: w.dense(&format!("{prefix}.ffn.linear2"))?,
        })
    }

    fn forward(&self, x: &Array, mask: &Array, n_heads: i32, head_dim: i32, base: f32) -> Result<Array> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let scale = (head_dim as f32).powf(-0.5);

        let normed = self.norm1.forward(x)?;
        let qkv = self.w_qkv.forward(&normed)?;
        // [B, T, 3, H, D] -> three [B, H, T, D]
        let qkv = qkv.reshape(&[b, t, 3, n_heads, head_dim])?;
        let mut parts: Vec<Array> = Vec::with_capacity(3);
        for i in 0..3 {
            let idx = Array::from_slice(&[i as i32], &[1]);
            parts.push(
                qkv.take_axis(&idx, 2)?
                    .squeeze_axes(&[2])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            );
        }
        let q = mlx_rs::fast::rope(&parts[0], head_dim, false, Some(base), 1.0, 0, None)?;
        let k = mlx_rs::fast::rope(&parts[1], head_dim, false, Some(base), 1.0, 0, None)?;
        let v = &parts[2];

        let attn = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            v,
            scale,
            mlx_rs::fast::ScaledDotProductAttentionMask::Array(mask),
            None,
        )?;
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, -1])?;
        let h = x.add(&self.out_proj.forward(&attn)?)?;

        let ff = self.lin2.forward(&mlx_rs::nn::gelu(&self.lin1.forward(&self.norm2.forward(&h)?)?)?)?;
        Ok(h.add(&ff)?)
    }
}

/// Speaker head (SortformerModules + Nemotron's subpixel upsampling).
pub struct SpeakerModules {
    encoder_proj: Dense,
    subpixel: (Array, Array), // Conv1d weight (out, k, in) + bias
    first_hidden: Dense,
    single_hidden_to_spks: Dense,
    activity_ln: Option<LayerNorm>,
    activity_proj: Option<Dense>,
    pub learnable_sil_emb: Array,
    upsample_factor: i32,
}

impl SpeakerModules {
    fn load(w: &Weights, cfg: &ModulesCfg) -> Result<Self> {
        let (activity_ln, activity_proj) = match w.maybe("sortformer_modules.activity_head.2.weight") {
            Some(_) => (
                Some(LayerNorm::load(w, "sortformer_modules.activity_head.0")?),
                Some(w.dense("sortformer_modules.activity_head.2")?),
            ),
            None => (None, None),
        };
        Ok(Self {
            encoder_proj: w.dense("sortformer_modules.encoder_proj")?,
            subpixel: (
                w.get("sortformer_modules.subpixel_upsample.weight")?.clone(),
                w.get("sortformer_modules.subpixel_upsample.bias")?.clone(),
            ),
            first_hidden: w.dense("sortformer_modules.first_hidden_to_hidden")?,
            single_hidden_to_spks: w.dense("sortformer_modules.single_hidden_to_spks")?,
            activity_ln,
            activity_proj,
            learnable_sil_emb: w.get("sortformer_modules.learnable_sil_emb")?.clone(),
            upsample_factor: cfg.subsampling_factor,
        })
    }

    /// `x` is `[B, T, fc_d_model]`; returns speaker probabilities `[B, T*factor, n_spk]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let proj = self.encoder_proj.forward(x)?;
        let (b, t, h) = (proj.shape()[0], proj.shape()[1], proj.shape()[2]);
        let up = ops::conv1d(&proj, &self.subpixel.0, 1, 1, 1, 1)?;
        let up = up.add(&self.subpixel.1)?;
        let up = up.reshape(&[b, t * self.upsample_factor, h])?;
        Ok(self.sigmoid_forward(&up)?)
    }

    fn sigmoid_forward(&self, x: &Array) -> Result<Array> {
        let h = ops::maximum(x, &Array::from_f32(0.0))?;
        let h = ops::maximum(&self.first_hidden.forward(&h)?, &Array::from_f32(0.0))?;
        let logits = self.single_hidden_to_spks.forward(&h)?;
        Ok(mlx_rs::nn::sigmoid(&logits)?)
    }

    /// Activity head `[B, T, 3]` logits (speech / silence / overlap), if present.
    pub fn activity(&self, x: &Array) -> Result<Option<Array>> {
        match (&self.activity_ln, &self.activity_proj) {
            (Some(ln), Some(proj)) => Ok(Some(proj.forward(&ln.forward(x)?)?)),
            _ => Ok(None),
        }
    }
}

pub struct Nemotron {
    pub config: DiarConfig,
    pre_encode: Dense,
    embed_norm: Option<LayerNorm>,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
    pub speakers: SpeakerModules,
}

impl Nemotron {
    pub fn load(dir: impl AsRef<Path>, config: DiarConfig) -> Result<Self> {
        let w = Weights::load_dir(dir)?;
        let enc = &config.encoder_config;
        let blocks = (0..enc.n_layers)
            .map(|i| Block::load(&w, &format!("encoder.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pre_encode: w.dense("encoder.pre_encode.proj")?,
            embed_norm: if enc.pre_block_norm {
                Some(LayerNorm::load(&w, "encoder.embed_norm")?)
            } else {
                None
            },
            blocks,
            final_norm: LayerNorm::load(&w, "encoder.final_norm")?,
            speakers: SpeakerModules::load(&w, &config.modules_config)?,
            config,
        })
    }

    /// Feature stacking: `[1, n_mels, T]` (checkpoint convention) -> `[1, T/8, d_model]`.
    pub fn pre_encode(&self, features: &Array) -> Result<(Array, i32)> {
        let enc = &self.config.encoder_config;
        let factor = enc.subsampling_factor;
        // checkpoint layout is [B, mels, T]; move time to the last axis
        let x = features.transpose_axes(&[0, 2, 1])?; // [1, T, mels]
        let t = x.shape()[1];
        let pad = (factor - t % factor) % factor;
        let x = if pad > 0 {
            let zeros = Array::zeros::<f32>(&[1, pad, enc.feat_in])?;
            ops::concatenate(&[&x, &zeros], 1)?
        } else {
            x
        };
        let t2 = x.shape()[1];
        let x = x.reshape(&[1, t2 / factor, enc.feat_in * factor])?;
        let h = self.pre_encode.forward(&x)?; // bias-free linear
        Ok((h, t2 / factor))
    }

    /// Run the transformer stack over stacked features, masking positions >= len.
    pub fn encoder_forward(&self, stacked: &Array, len: i32) -> Result<Array> {
        let enc = &self.config.encoder_config;
        let mut h = stacked.clone();
        if enc.xscaling {
            h = h.multiply(&Array::from_f32((enc.d_model as f32).sqrt()))?;
        }
        if let Some(norm) = &self.embed_norm {
            h = norm.forward(&h)?;
        }
        let tmax = h.shape()[1];
        let clamped = len.min(tmax).max(0);
        let mask_vec: Vec<bool> = (0..tmax).map(|i| i < clamped).collect();
        let mask = Array::from_slice(&mask_vec, &[1, 1, 1, tmax]);

        let head_dim = enc.d_model / enc.n_heads;
        for block in &self.blocks {
            h = block.forward(&h, &mask, enc.n_heads, head_dim, enc.rope_base)?;
        }
        Ok(self.final_norm.forward(&h)?)
    }

    /// `mel` is `[T, n_mels]` (time-major, our mel pipeline); returns features
    /// `[1, T/8, d_model]` and the valid encoder-frame count.
    pub fn encode(&self, mel: &Array, lengths: Option<i32>) -> Result<(Array, i32)> {
        let factor = self.config.encoder_config.subsampling_factor;
        let (stacked, full_len) = self.pre_encode(&mel.expand_dims(0)?.transpose_axes(&[0, 2, 1])?)?;
        let len = lengths
            .map(|l| (l + factor - 1) / factor)
            .unwrap_or(full_len);
        Ok((self.encoder_forward(&stacked, len)?, len))
    }
}

/// Number of encoder frames for a mel length (subsampling ceil division).
pub fn encoder_frames(mel_frames: i32, factor: i32) -> i32 {
    (mel_frames + factor - 1) / factor
}
