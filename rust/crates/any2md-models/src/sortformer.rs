//! Sortformer streaming diarization (`diar_streaming_sortformer_4spk-v2.1`).
//!
//! Mirrors `mlx_audio/vad/models/sortformer/sortformer.py` along the path any2md
//! drives it: an iterable of ~10 s audio chunks is fed through
//! `Model.feed()` (per-chunk mel, no normalisation for the AOSC variant), the
//! pre-encoded embeddings roll through `spkcache` + `fifo`, and the speaker
//! cache is compressed with AOSC once it overflows.
//!
//! Architecture: dw-striding Conv2d subsampling (8x) -> 17 FastConformer layers
//! with relative-position attention -> linear projection to 192 -> 18 post-LN
//! Transformer layers with learned positions -> speaker sigmoids.
//!
//! Frame rate is `hop * subsampling = 1280` samples = 80 ms per diarization
//! frame, which is what segment times are expressed in.

use any2md_audio::{nemo_mel, NemoMel};
use any2md_mlx::{ops, Array, Dense, Dtype, Weights};
use anyhow::{Context, Result};
use serde::Deserialize;

use crate::aosc::{self, AoscCfg};
use crate::diarize::{merge_same_speaker, preds_to_segments, Segment};

pub const DEFAULT_MODEL: &str = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16";

#[derive(Debug, Clone, Deserialize)]
pub struct FcCfg {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub intermediate_size: i32,
    pub num_mel_bins: i32,
    pub conv_kernel_size: i32,
    pub subsampling_factor: i32,
    pub subsampling_conv_channels: i32,
    pub subsampling_conv_kernel_size: i32,
    pub subsampling_conv_stride: i32,
    #[serde(default)]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub scale_input: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TfCfg {
    pub d_model: i32,
    pub encoder_layers: i32,
    pub encoder_attention_heads: i32,
    pub encoder_ffn_dim: i32,
    pub max_source_positions: i32,
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub k_proj_bias: bool,
}

fn default_ln_eps() -> f32 {
    1e-5
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModulesCfg {
    pub num_speakers: i32,
    pub fc_d_model: i32,
    pub tf_d_model: i32,
    pub chunk_len: i32,
    #[serde(default)]
    pub fifo_len: i32,
    pub spkcache_len: i32,
    #[serde(default)]
    pub spkcache_update_period: i32,
    #[serde(default)]
    pub chunk_left_context: i32,
    #[serde(default)]
    pub chunk_right_context: i32,
    #[serde(default)]
    pub spkcache_sil_frames_per_spk: i32,
    #[serde(default)]
    pub sil_threshold: f32,
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
    #[serde(default)]
    pub use_aosc: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorCfg {
    pub feature_size: i32,
    pub sampling_rate: i32,
    pub hop_length: i32,
    pub n_fft: i32,
    pub win_length: i32,
    pub preemphasis: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SortformerConfig {
    pub num_speakers: i32,
    pub dtype: String,
    pub fc_encoder_config: FcCfg,
    pub tf_encoder_config: TfCfg,
    pub modules_config: ModulesCfg,
    pub processor_config: ProcessorCfg,
}

impl SortformerConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    pub fn dtype(&self) -> Dtype {
        match self.dtype.as_str() {
            "float16" => Dtype::Float16,
            "bfloat16" => Dtype::Bfloat16,
            _ => Dtype::Float32,
        }
    }

    /// Diarization frame duration in seconds (80 ms for this checkpoint).
    pub fn frame_duration(&self) -> f32 {
        (self.processor_config.hop_length * self.fc_encoder_config.subsampling_factor) as f32
            / self.processor_config.sampling_rate as f32
    }
}

struct Subsampling {
    conv0: (Array, Array),
    conv2: (Array, Array),
    conv3: (Array, Array),
    conv5: (Array, Array),
    conv6: (Array, Array),
    linear: Dense,
}

impl Subsampling {
    fn load(w: &Weights) -> Result<Self> {
        let pair = |name: &str| -> Result<(Array, Array)> {
            Ok((
                w.get(&format!("{name}.weight"))?.clone(),
                w.get(&format!("{name}.bias"))?.reshape(&[1, 1, 1, -1])?,
            ))
        };
        Ok(Self {
            conv0: pair("fc_encoder.subsampling.layers_0")?,
            conv2: pair("fc_encoder.subsampling.layers_2")?,
            conv3: pair("fc_encoder.subsampling.layers_3")?,
            conv5: pair("fc_encoder.subsampling.layers_5")?,
            conv6: pair("fc_encoder.subsampling.layers_6")?,
            linear: w.dense("fc_encoder.subsampling.linear")?,
        })
    }

    /// `mel` is `[1, n_mels, T]`; returns `[1, T/8, d_model]`.
    fn forward(&self, mel: &Array) -> Result<(Array, i32)> {
        // [1, F, T] -> [1, T, F, 1] (NHWC)
        let x = mel.transpose_axes(&[0, 2, 1])?.expand_dims(3)?;
        let mut h = relu(&conv_bn(&x, &self.conv0, 2, 1, 1)?)?;
        h = conv_bn(&h, &self.conv2, 2, 1, self.conv2.0.shape()[0])?; // depthwise
        h = conv_bn(&h, &self.conv3, 1, 0, 1)?;
        h = relu(&h)?;
        h = conv_bn(&h, &self.conv5, 2, 1, self.conv5.0.shape()[0])?;
        h = conv_bn(&h, &self.conv6, 1, 0, 1)?;
        h = relu(&h)?;
        // [b, T, F, C] -> [b, T, C*F] (NeMo's channel-major flatten)
        let (b, t, f, c) = (h.shape()[0], h.shape()[1], h.shape()[2], h.shape()[3]);
        let h = h.transpose_axes(&[0, 1, 3, 2])?.reshape(&[b, t, c * f])?;
        Ok((self.linear.forward(&h)?, t))
    }
}

fn conv_bn(
    x: &Array,
    (weight, bias): &(Array, Array),
    stride: i32,
    padding: i32,
    groups: i32,
) -> Result<Array> {
    let y = ops::conv2d(x, weight, (stride, stride), (padding, padding), (1, 1), groups)?;
    Ok(y.add(bias)?)
}

fn relu(x: &Array) -> Result<Array> {
    Ok(ops::maximum(x, &Array::from_f32(0.0))?)
}

fn silu(x: &Array) -> Result<Array> {
    any2md_mlx::silu(x)
}

struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    fn load(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?.clone(),
            bias: w.get(&format!("{prefix}.bias"))?.clone(),
            eps,
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

struct FeedForward {
    linear1: Dense,
    linear2: Dense,
}

impl FeedForward {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear1: w.dense(&format!("{prefix}.linear1"))?,
            linear2: w.dense(&format!("{prefix}.linear2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.linear2.forward(&silu(&self.linear1.forward(x)?)?)
    }
}

/// Relative-position Conformer self-attention (Transformer-XL style, with
/// learned `bias_u` / `bias_v` and a shared relative-key projection).
struct RelPosAttention {
    q_proj: Dense,
    k_proj: Dense,
    v_proj: Dense,
    o_proj: Dense,
    rel_k: Dense,
    bias_u: Array,
    bias_v: Array,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl RelPosAttention {
    fn load(w: &Weights, prefix: &str, n_heads: i32, d_model: i32) -> Result<Self> {
        Ok(Self {
            q_proj: w.dense(&format!("{prefix}.q_proj"))?,
            k_proj: w.dense(&format!("{prefix}.k_proj"))?,
            v_proj: w.dense(&format!("{prefix}.v_proj"))?,
            o_proj: w.dense(&format!("{prefix}.o_proj"))?,
            rel_k: w.dense(&format!("{prefix}.relative_k_proj"))?,
            bias_u: w.get(&format!("{prefix}.bias_u"))?.clone(),
            bias_v: w.get(&format!("{prefix}.bias_v"))?.clone(),
            n_heads,
            head_dim: d_model / n_heads,
            scale: (d_model as f32 / n_heads as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array, pos_emb: &Array) -> Result<Array> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let (h, dk) = (self.n_heads, self.head_dim);
        let q = self.q_proj.forward(x)?.reshape(&[b, t, h, dk])?;
        let k = self.k_proj.forward(x)?.reshape(&[b, t, h, dk])?;
        let v = self.v_proj.forward(x)?.reshape(&[b, t, h, dk])?;
        let p = self.rel_k.forward(pos_emb)?;
        let pos_len = p.shape()[1];
        let p = p.reshape(&[1, pos_len, h, dk])?.transpose_axes(&[0, 2, 1, 3])?;

        let q_u = q.add(&self.bias_u.expand_dims(0)?)?.transpose_axes(&[0, 2, 1, 3])?;
        let q_v = q.add(&self.bias_v.expand_dims(0)?)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let matrix_bd = q_v.matmul(&p.transpose_axes(&[0, 1, 3, 2])?)?;
        let matrix_bd = rel_shift(&matrix_bd)?;
        // keep the first `t` relative-position columns
        let cols = Array::from_slice(&(0..t).collect::<Vec<_>>(), &[t]);
        let matrix_bd = matrix_bd
            .take_axis(&cols, 3)?
            .multiply(&Array::from_f32(self.scale))?;

        let out = mlx_rs::fast::scaled_dot_product_attention(
            &q_u,
            &k,
            &v,
            self.scale,
            mlx_rs::fast::ScaledDotProductAttentionMask::Array(&matrix_bd),
            None,
        )?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, -1])?;
        self.o_proj.forward(&out)
    }
}

/// Conformer relative-position shift.
fn rel_shift(x: &Array) -> Result<Array> {
    let (b, h, tq) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let pos_len = x.shape()[3];
    let pad = Array::zeros::<f32>(&[b, h, tq, 1])?;
    let padded = ops::concatenate(&[&pad, x], 3)?;
    let reshaped = padded.reshape(&[b, h, pos_len + 1, tq])?;
    let rows = Array::from_slice(&(1..=pos_len).collect::<Vec<_>>(), &[pos_len]);
    let sliced = reshaped.take_axis(&rows, 2)?;
    Ok(sliced.reshape(&[b, h, tq, pos_len])?)
}

/// Conformer convolution module: pointwise -> GLU -> depthwise -> BatchNorm ->
/// SiLU -> pointwise.
struct ConvModule {
    pointwise1: (Array, Array),
    depthwise: (Array, Array),
    pointwise2: (Array, Array),
    bn_weight: Array,
    bn_bias: Array,
    bn_mean: Array,
    bn_var: Array,
    kernel: i32,
}

impl ConvModule {
    fn load(w: &Weights, prefix: &str, kernel: i32) -> Result<Self> {
        let conv = |name: &str| -> Result<(Array, Array)> {
            Ok((
                w.get(&format!("{name}.weight"))?.clone(),
                w.get(&format!("{name}.bias"))?.clone(),
            ))
        };
        Ok(Self {
            pointwise1: conv(&format!("{prefix}.pointwise_conv1"))?,
            depthwise: conv(&format!("{prefix}.depthwise_conv"))?,
            pointwise2: conv(&format!("{prefix}.pointwise_conv2"))?,
            bn_weight: w.get(&format!("{prefix}.norm.weight"))?.clone(),
            bn_bias: w.get(&format!("{prefix}.norm.bias"))?.clone(),
            bn_mean: w.get(&format!("{prefix}.norm.running_mean"))?.clone(),
            bn_var: w.get(&format!("{prefix}.norm.running_var"))?.clone(),
            kernel,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = ops::conv1d(&x, &self.pointwise1.0, 1, 0, 1, 1)?.add(&self.pointwise1.1)?;
        // GLU over the channel axis: x1 * sigmoid(x2)
        let h = mlx_rs::nn::glu(&h, 2)?;
        let mut h = ops::conv1d(
            &h,
            &self.depthwise.0,
            1,
            self.kernel / 2,
            1,
            self.depthwise.0.shape()[0],
        )?
        .add(&self.depthwise.1)?;
        // BatchNorm (inference)
        let denom = self.bn_var.add(&Array::from_f32(1e-5))?.sqrt()?;
        h = h
            .subtract(&self.bn_mean)?
            .divide(&denom)?
            .multiply(&self.bn_weight)?
            .add(&self.bn_bias)?;
        h = silu(&h)?;
        Ok(ops::conv1d(&h, &self.pointwise2.0, 1, 0, 1, 1)?.add(&self.pointwise2.1)?)
    }
}

struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_att: LayerNorm,
    attn: RelPosAttention,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    fn load(w: &Weights, prefix: &str, cfg: &FcCfg) -> Result<Self> {
        Ok(Self {
            norm_ff1: LayerNorm::load(w, &format!("{prefix}.norm_feed_forward1"), 1e-5)?,
            ff1: FeedForward::load(w, &format!("{prefix}.feed_forward1"))?,
            norm_att: LayerNorm::load(w, &format!("{prefix}.norm_self_att"), 1e-5)?,
            attn: RelPosAttention::load(
                w,
                &format!("{prefix}.self_attn"),
                cfg.num_attention_heads,
                cfg.hidden_size,
            )?,
            norm_conv: LayerNorm::load(w, &format!("{prefix}.norm_conv"), 1e-5)?,
            conv: ConvModule::load(w, &format!("{prefix}.conv"), cfg.conv_kernel_size)?,
            norm_ff2: LayerNorm::load(w, &format!("{prefix}.norm_feed_forward2"), 1e-5)?,
            ff2: FeedForward::load(w, &format!("{prefix}.feed_forward2"))?,
            norm_out: LayerNorm::load(w, &format!("{prefix}.norm_out"), 1e-5)?,
        })
    }

    fn forward(&self, x: &Array, pos_emb: &Array) -> Result<Array> {
        let half = Array::from_f32(0.5);
        let mut h = x.add(&self.ff1.forward(&self.norm_ff1.forward(x)?)?.multiply(&half)?)?;
        h = h.add(&self.attn.forward(&self.norm_att.forward(&h)?, pos_emb)?)?;
        h = h.add(&self.conv.forward(&self.norm_conv.forward(&h)?)?)?;
        h = h.add(&self.ff2.forward(&self.norm_ff2.forward(&h)?)?.multiply(&half)?)?;
        self.norm_out.forward(&h)
    }
}

struct TfAttention {
    q_proj: Dense,
    k_proj: Dense,
    v_proj: Dense,
    out_proj: Dense,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl TfAttention {
    fn load(w: &Weights, prefix: &str, d_model: i32, n_heads: i32) -> Result<Self> {
        Ok(Self {
            q_proj: w.dense(&format!("{prefix}.q_proj"))?,
            k_proj: w.dense(&format!("{prefix}.k_proj"))?,
            v_proj: w.dense(&format!("{prefix}.v_proj"))?,
            out_proj: w.dense(&format!("{prefix}.out_proj"))?,
            n_heads,
            head_dim: d_model / n_heads,
            scale: (d_model as f32 / n_heads as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let (h, dk) = (self.n_heads, self.head_dim);
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, t, h, dk])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, t, h, dk])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, t, h, dk])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let out = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            self.scale,
            Option::<mlx_rs::fast::ScaledDotProductAttentionMask>::None,
            None,
        )?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, -1])?;
        self.out_proj.forward(&out)
    }
}

struct TfLayer {
    attn: TfAttention,
    attn_norm: LayerNorm,
    fc1: Dense,
    fc2: Dense,
    final_norm: LayerNorm,
}

impl TfLayer {
    fn load(w: &Weights, prefix: &str, cfg: &TfCfg) -> Result<Self> {
        Ok(Self {
            attn: TfAttention::load(
                w,
                &format!("{prefix}.self_attn"),
                cfg.d_model,
                cfg.encoder_attention_heads,
            )?,
            attn_norm: LayerNorm::load(
                w,
                &format!("{prefix}.self_attn_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            fc1: w.dense(&format!("{prefix}.fc1"))?,
            fc2: w.dense(&format!("{prefix}.fc2"))?,
            final_norm: LayerNorm::load(w, &format!("{prefix}.final_layer_norm"), cfg.layer_norm_eps)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.attn_norm.forward(&x.add(&self.attn.forward(x)?)?)?;
        let f = self.fc2.forward(&relu(&self.fc1.forward(&h)?)?)?;
        self.final_norm.forward(&h.add(&f)?)
    }
}

pub struct Sortformer {
    pub config: SortformerConfig,
    subsampling: Subsampling,
    layers: Vec<ConformerLayer>,
    tf_layers: Vec<TfLayer>,
    pos_weight: Array,
    encoder_proj: Dense,
    first_hidden: Dense,
    single_out: Dense,
    dtype: Dtype,
}

impl Sortformer {
    pub fn load(dir: impl AsRef<std::path::Path>, config: SortformerConfig) -> Result<Self> {
        let w = Weights::load_dir(dir.as_ref())?;
        let fc = &config.fc_encoder_config;
        let tf = &config.tf_encoder_config;
        let mut layers = Vec::with_capacity(fc.num_hidden_layers as usize);
        for i in 0..fc.num_hidden_layers {
            layers.push(ConformerLayer::load(
                &w,
                &format!("fc_encoder.layers.{i}"),
                fc,
            )?);
        }
        let mut tf_layers = Vec::with_capacity(tf.encoder_layers as usize);
        for i in 0..tf.encoder_layers {
            tf_layers.push(TfLayer::load(&w, &format!("tf_encoder.layers.{i}"), tf)?);
        }
        Ok(Self {
            dtype: config.dtype(),
            subsampling: Subsampling::load(&w)?,
            layers,
            tf_layers,
            pos_weight: w.get("tf_encoder.embed_positions.weight")?.clone(),
            encoder_proj: w.dense("sortformer_modules.encoder_proj")?,
            first_hidden: w.dense("sortformer_modules.first_hidden_to_hidden")?,
            single_out: w.dense("sortformer_modules.single_hidden_to_spks")?,
            config,
        })
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Relative positional encoding for a sequence of `t` frames.
    fn pos_encoding(&self, t: i32) -> Result<Array> {
        let d = self.config.fc_encoder_config.hidden_size as usize;
        let len = (2 * t - 1) as usize;
        let half = d / 2;
        let log_timescale = 10_000f32.ln() / d as f32;
        let mut pe = vec![0f32; len * d];
        for (row, pos) in (0..len).map(|i| (i, (t - 1 - i as i32) as f32)) {
            for j in 0..half {
                let angle = pos * (-((2 * j) as f32) * log_timescale).exp();
                pe[row * d + 2 * j] = angle.sin();
                pe[row * d + 2 * j + 1] = angle.cos();
            }
        }
        Ok(Array::from_slice(&pe, &[1, len as i32, d as i32]).as_dtype(self.dtype)?)
    }

    /// FastConformer stack over pre-encoded embeddings `[1, T, 512]`.
    pub fn fc_encode(&self, embs: &Array) -> Result<Array> {
        let fc = &self.config.fc_encoder_config;
        let mut x = embs.clone();
        if fc.scale_input {
            x = x.multiply(&Array::from_f32((fc.hidden_size as f32).sqrt()))?;
        }
        let pos = self.pos_encoding(x.shape()[1])?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos)?;
        }
        Ok(x)
    }

    /// Transformer encoder with learned positional embeddings.
    pub fn tf_encode(&self, x: &Array) -> Result<Array> {
        let t = x.shape()[1];
        let ids = Array::from_slice(&(0..t).collect::<Vec<_>>(), &[t]);
        let pos = self.pos_weight.take_axis(&ids, 0)?;
        let mut h = x.add(&pos)?;
        for layer in &self.tf_layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }

    /// Speaker sigmoids `[1, T, n_spk]`.
    fn speaker_probs(&self, x: &Array) -> Result<Array> {
        let h = self.first_hidden.forward(&relu(x)?)?;
        let logits = self.single_out.forward(&relu(&h)?)?;
        Ok(mlx_rs::nn::sigmoid(&logits)?)
    }

    /// ConvSubsampling only (streaming pre-encode).
    pub fn pre_encode(&self, features: &Array) -> Result<(Array, i32)> {
        self.subsampling.forward(features)
    }

    /// Full forward over `[1, n_mels, T]` features (non-streaming).
    pub fn forward(&self, features: &Array) -> Result<Array> {
        let features = features.as_dtype(self.dtype)?;
        let (embs, len) = self.subsampling.forward(&features)?;
        let embs = embs.take_axis(&Array::from_slice(&(0..len).collect::<Vec<_>>(), &[len]), 1)?;
        let fc = self.fc_encode(&embs)?;
        let proj = self.encoder_proj.forward(&fc)?;
        let tf = self.tf_encode(&proj)?;
        Ok(self.speaker_probs(&tf)?.as_dtype(Dtype::Float32)?)
    }
}

/// Streaming state: pre-encoded embedding cache + FIFO, plus the AOSC silence
/// profile.
pub struct StreamState {
    pub spkcache: Option<Array>,
    pub spkcache_preds: Option<Array>,
    pub fifo: Option<Array>,
    pub fifo_preds: Option<Array>,
    pub frames_processed: i32,
    pub mean_sil_emb: Array,
    pub n_sil_frames: f32,
}

impl StreamState {
    pub fn new(emb_dim: i32) -> Result<Self> {
        Ok(Self {
            spkcache: None,
            spkcache_preds: None,
            fifo: None,
            fifo_preds: None,
            frames_processed: 0,
            mean_sil_emb: Array::zeros::<f32>(&[1, emb_dim])?,
            n_sil_frames: 0.0,
        })
    }

    fn cache_len(&self) -> i32 {
        self.spkcache.as_ref().map(|a| a.shape()[1]).unwrap_or(0)
    }

    fn fifo_len(&self) -> i32 {
        self.fifo.as_ref().map(|a| a.shape()[1]).unwrap_or(0)
    }
}

/// Chunked streaming diarizer matching `any2md.yt.diarize()` for Sortformer
/// models: 10 s WAV chunks, per-chunk segments, then a 0.3 s same-speaker merge.
pub struct SortformerDiarizer {
    pub model: Sortformer,
    mel: NemoMel,
    cfg: SortformerConfig,
    chunk_duration: f32,
}

impl SortformerDiarizer {
    pub fn load(dir: impl AsRef<std::path::Path>, cfg: SortformerConfig) -> Result<Self> {
        let model = Sortformer::load(dir, cfg.clone())?;
        let proc = &cfg.processor_config;
        let mut mel = NemoMel::sortformer();
        mel.sample_rate = proc.sampling_rate as usize;
        mel.n_fft = proc.n_fft as usize;
        mel.hop = proc.hop_length as usize;
        mel.win_length = proc.win_length as usize;
        mel.n_mels = proc.feature_size as usize;
        mel.preemph = proc.preemphasis;
        // v2.1 streaming front-end: no per-feature normalisation, no padding
        mel.normalize = "none".into();
        mel.pad_to = 0;
        Ok(Self {
            model,
            mel,
            cfg,
            chunk_duration: 10.0,
        })
    }

    fn frame_duration(&self) -> f32 {
        self.cfg.frame_duration()
    }

    fn aosc_cfg(&self) -> AoscCfg {
        let mc = &self.cfg.modules_config;
        AoscCfg {
            num_speakers: mc.num_speakers as usize,
            spkcache_len: mc.spkcache_len as usize,
            sil_frames_per_spk: mc.spkcache_sil_frames_per_spk.max(0) as usize,
            strong_boost_rate: mc.strong_boost_rate,
            weak_boost_rate: mc.weak_boost_rate,
            min_pos_scores_rate: mc.min_pos_scores_rate,
            pred_score_threshold: mc.pred_score_threshold,
            scores_boost_latest: mc.scores_boost_latest,
        }
    }

    /// Feed one raw audio chunk; returns that chunk's segments (times already
    /// shifted to the absolute timeline).
    pub fn feed(&self, state: &mut StreamState, chunk: &[f32]) -> Result<Vec<Segment>> {
        if chunk.is_empty() {
            return Ok(Vec::new());
        }
        let audio = Array::from_slice(chunk, &[chunk.len() as i32]);
        let features = nemo_mel(&audio, &self.mel)?.expand_dims(0)?; // [1, mels, T]
        let features = features.as_dtype(self.model.dtype())?;
        let offset = state.frames_processed as f32 * self.frame_duration();
        let preds = self.streaming_step(state, &features)?;
        let mut segments = preds_to_segments(&preds, self.frame_duration(), 0.5, 0.0, 0.0);
        for s in &mut segments {
            s.start += offset;
            s.end += offset;
        }
        self.maybe_compress(state)?;
        Ok(segments)
    }

    /// One streaming step; returns `[frames, n_spk]` probabilities for the new
    /// chunk only.
    fn streaming_step(&self, state: &mut StreamState, features: &Array) -> Result<Vec<Vec<f32>>> {
        let mc = &self.cfg.modules_config;
        let (chunk_embs, chunk_len) = self.model.pre_encode(features)?;
        let chunk_embs = chunk_embs.take_axis(
            &Array::from_slice(&(0..chunk_len).collect::<Vec<_>>(), &[chunk_len]),
            1,
        )?;

        let cache_len = state.cache_len();
        let fifo_len = state.fifo_len();

        // left context: last `chunk_left_context` frames of the FIFO
        let left_len = if mc.chunk_left_context > 0 && fifo_len > 0 {
            mc.chunk_left_context.min(fifo_len)
        } else {
            0
        };
        let left_ctx = if left_len > 0 {
            let fifo = state.fifo.as_ref().expect("fifo");
            Some(fifo.take_axis(
                &Array::from_slice(&(fifo_len - left_len..fifo_len).collect::<Vec<_>>(), &[left_len]),
                1,
            )?)
        } else {
            None
        };

        let mut parts: Vec<&Array> = Vec::new();
        if cache_len > 0 {
            parts.push(state.spkcache.as_ref().expect("cache"));
        }
        if fifo_len > 0 {
            parts.push(state.fifo.as_ref().expect("fifo"));
        }
        if let Some(ctx) = &left_ctx {
            parts.push(ctx);
        }
        parts.push(&chunk_embs);
        let all = ops::concatenate(&parts, 1)?;

        let fc_out = self.model.fc_encode(&all)?;
        let proj = self.model.encoder_proj.forward(&fc_out)?;
        let tf_out = self.model.tf_encode(&proj)?;
        let preds = self.model.speaker_probs(&tf_out)?; // [1, total, n_spk]

        let chunk_start = cache_len + fifo_len + left_len;
        let chunk_preds = preds.take_axis(
            &Array::from_slice(
                &(chunk_start..chunk_start + chunk_len).collect::<Vec<_>>(),
                &[chunk_len],
            ),
            1,
        )?;
        let cache_preds = if cache_len > 0 {
            Some(preds.take_axis(
                &Array::from_slice(&(0..cache_len).collect::<Vec<_>>(), &[cache_len]),
                1,
            )?)
        } else {
            None
        };
        let fifo_preds = if fifo_len > 0 {
            Some(preds.take_axis(
                &Array::from_slice(
                    &(cache_len..cache_len + fifo_len).collect::<Vec<_>>(),
                    &[fifo_len],
                ),
                1,
            )?)
        } else {
            None
        };

        // _update_streaming_state: FIFO grows, cache stays put
        if let Some(cp) = cache_preds {
            state.spkcache_preds = Some(cp);
        }
        let base_preds = fifo_preds.or_else(|| state.fifo_preds.clone());
        state.fifo_preds = Some(match &base_preds {
            Some(p) => ops::concatenate(&[p, &chunk_preds], 1)?,
            None => chunk_preds.clone(),
        });
        state.fifo = Some(match &state.fifo {
            Some(f) => ops::concatenate(&[f, &chunk_embs], 1)?,
            None => chunk_embs,
        });
        state.frames_processed += chunk_len;

        let rows = to_rows(&chunk_preds)?;
        Ok(rows)
    }

    /// Move FIFO overflow into the speaker cache, compressing with AOSC.
    fn maybe_compress(&self, state: &mut StreamState) -> Result<()> {
        let mc = &self.cfg.modules_config;
        // any2md passes the checkpoint defaults: spkcache_len, and fifo_max
        // falls back to the 188 default because this checkpoint's fifo_len is 0.
        let spkcache_max = mc.spkcache_len;
        let fifo_max = if mc.fifo_len > 0 { mc.fifo_len } else { 188 };
        if state.fifo_len() <= fifo_max {
            return Ok(());
        }
        let mut pop = state.fifo_len() - fifo_max;
        if mc.use_aosc {
            pop = pop.min(mc.spkcache_update_period.max(1));
        }
        let fifo = state.fifo.as_ref().expect("fifo");
        let fifo_preds = state.fifo_preds.as_ref().expect("fifo_preds");
        let popped = fifo.take_axis(&Array::from_slice(&(0..pop).collect::<Vec<_>>(), &[pop]), 1)?;
        let popped_preds =
            fifo_preds.take_axis(&Array::from_slice(&(0..pop).collect::<Vec<_>>(), &[pop]), 1)?;

        if mc.use_aosc {
            let (mean_sil, n_sil) = aosc::update_silence_profile(
                &state.mean_sil_emb,
                state.n_sil_frames,
                &popped,
                &popped_preds,
                if mc.sil_threshold > 0.0 { mc.sil_threshold } else { 0.2 },
            )?;
            state.mean_sil_emb = mean_sil;
            state.n_sil_frames = n_sil;
        }

        let mut cache = match &state.spkcache {
            Some(c) => ops::concatenate(&[c, &popped], 1)?,
            None => popped,
        };
        let mut cache_preds = match &state.spkcache_preds {
            Some(p) => ops::concatenate(&[p, &popped_preds], 1)?,
            None => popped_preds,
        };
        if cache.shape()[1] > spkcache_max {
            let (c, p) = aosc::compress(&cache, &cache_preds, &state.mean_sil_emb, &self.aosc_cfg())?;
            cache = c;
            cache_preds = p;
        }
        let remaining = state.fifo_len() - pop;
        state.spkcache = Some(cache);
        state.spkcache_preds = Some(cache_preds);
        state.fifo = Some(fifo.take_axis(
            &Array::from_slice(&(pop..pop + remaining).collect::<Vec<_>>(), &[remaining]),
            1,
        )?);
        state.fifo_preds = Some(fifo_preds.take_axis(
            &Array::from_slice(&(pop..pop + remaining).collect::<Vec<_>>(), &[remaining]),
            1,
        )?);
        Ok(())
    }

    /// Diarize a whole 16 kHz mono recording.
    pub fn generate(&self, audio: &[f32]) -> Result<Vec<Segment>> {
        Ok(self.generate_with_probs(audio)?.0)
    }

    /// Like [`Self::generate`], also returning the per-chunk speaker
    /// probabilities (used by the parity harness to bisect divergence).
    pub fn generate_with_probs(
        &self,
        audio: &[f32],
    ) -> Result<(Vec<Segment>, Vec<Vec<Vec<f32>>>)> {
        let mut state = StreamState::new(self.cfg.fc_encoder_config.hidden_size)?;
        let step = (self.chunk_duration * self.cfg.processor_config.sampling_rate as f32) as usize;
        let mut segments = Vec::new();
        let mut probs: Vec<Vec<Vec<f32>>> = Vec::new();
        for chunk in audio.chunks(step.max(1)) {
            let audio_arr = Array::from_slice(chunk, &[chunk.len() as i32]);
            let features = nemo_mel(&audio_arr, &self.mel)?.expand_dims(0)?;
            let features = features.as_dtype(self.model.dtype())?;
            let offset = state.frames_processed as f32 * self.frame_duration();
            let chunk_probs = self.streaming_step(&mut state, &features)?;
            let mut chunk_segments = preds_to_segments(&chunk_probs, self.frame_duration(), 0.5, 0.0, 0.0);
            for s in &mut chunk_segments {
                s.start += offset;
                s.end += offset;
            }
            segments.extend(chunk_segments);
            probs.push(chunk_probs);
            self.maybe_compress(&mut state)?;
        }
        Ok((merge_same_speaker(segments, 0.3), probs))
    }
}

fn to_rows(a: &Array) -> Result<Vec<Vec<f32>>> {
    let a = a.as_dtype(Dtype::Float32)?;
    a.eval()?;
    let shape = a.shape().to_vec();
    let (frames, spk) = (shape[1] as usize, shape[2] as usize);
    let data = a.as_slice::<f32>();
    Ok((0..frames)
        .map(|i| data[i * spk..(i + 1) * spk].to_vec())
        .collect())
}

/// Convenience: load a checkpoint directory id (HF id) into a streaming diarizer.
pub fn load_diarizer(model_id: &str) -> Result<SortformerDiarizer> {
    let cfg_path = any2md_hub::get_file(model_id, "config.json")
        .with_context(|| format!("resolving {model_id}/config.json"))?;
    let cfg = SortformerConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let dir = cfg_path.parent().context("config has no parent")?;
    SortformerDiarizer::load(dir, cfg)
}
