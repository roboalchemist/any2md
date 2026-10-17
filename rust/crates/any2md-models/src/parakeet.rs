//! Parakeet TDT (FastConformer + LSTM predictor + TDT joint) — the English STT
//! model any2md defaults to.
//!
//! Mirrors `mlx_audio/stt/models/parakeet/{conformer,attention,rnnt,parakeet}.py`:
//! dw-striding Conv2d subsampling, relative-position Conformer blocks with
//! batch-norm convolution modules, a 2-layer LSTM prediction network, and the
//! duration-conditioned TDT greedy loop (max_symbols cap included).

use std::path::Path;

use any2md_mlx::ops::indexing::TryIndexOp;
use any2md_mlx::{ops, Array, Dense, Dtype, Weights};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

pub const DEFAULT_MODEL: &str = "mlx-community/parakeet-tdt-0.6b-v3";
/// Encoder frames are 80 ms at 16 kHz with subsampling 8 / hop 160.
const SAMPLE_RATE: f32 = 16_000.0;

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub feat_in: i32,
    pub n_layers: usize,
    pub d_model: i32,
    pub n_heads: i32,
    pub ff_expansion_factor: i32,
    pub subsampling_factor: i32,
    pub subsampling_conv_channels: i32,
    pub conv_kernel_size: i32,
    pub pos_emb_max_len: i32,
    #[serde(default)]
    pub xscaling: bool,
    #[serde(default)]
    pub mask_padding: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    pub features: usize,
    pub n_fft: usize,
    pub window_size: f32,
    pub window_stride: f32,
    #[serde(default)]
    pub pad_to: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TdtDecoding {
    pub durations: Vec<i32>,
    #[serde(default)]
    pub greedy: Option<GreedyCfg>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GreedyCfg {
    #[serde(default)]
    pub max_symbols: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JointConfig {
    pub vocabulary: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParakeetConfig {
    pub preprocessor: PreprocessorConfig,
    pub encoder: EncoderConfig,
    pub decoding: TdtDecoding,
    pub joint: JointConfig,
}

impl ParakeetConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text).context("parse parakeet config.json")?)
    }

    pub fn blank_id(&self) -> u32 {
        self.joint.vocabulary.len() as u32
    }

    fn max_symbols(&self) -> Option<usize> {
        self.decoding.greedy.as_ref().and_then(|g| g.max_symbols)
    }
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

struct LayerNorm {
    weight: Array,
    bias: Option<Array>,
}

impl LayerNorm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?.clone(),
            bias: w.maybe(&format!("{prefix}.bias")).cloned(),
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::layer_norm(x, Some(&self.weight), self.bias.as_ref(), 1e-5)?)
    }
}

fn silu(x: &Array) -> Result<Array> {
    any2md_mlx::silu(x)
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

/// Relative-position Conformer self-attention.
struct RelPosAttention {
    linear_q: Dense,
    linear_k: Dense,
    linear_v: Dense,
    linear_out: Dense,
    linear_pos: Dense,
    pos_bias_u: Array,
    pos_bias_v: Array,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl RelPosAttention {
    fn load(w: &Weights, prefix: &str, n_heads: i32, d_model: i32) -> Result<Self> {
        let head_dim = d_model / n_heads;
        Ok(Self {
            linear_q: w.dense(&format!("{prefix}.linear_q"))?,
            linear_k: w.dense(&format!("{prefix}.linear_k"))?,
            linear_v: w.dense(&format!("{prefix}.linear_v"))?,
            linear_out: w.dense(&format!("{prefix}.linear_out"))?,
            linear_pos: w.dense(&format!("{prefix}.linear_pos"))?,
            pos_bias_u: w.get(&format!("{prefix}.pos_bias_u"))?.clone(),
            pos_bias_v: w.get(&format!("{prefix}.pos_bias_v"))?.clone(),
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array, pos_emb: &Array) -> Result<Array> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;
        let p = self.linear_pos.forward(pos_emb)?;
        let pos_len = p.shape()[1];

        let q = q
            .reshape(&[b, t, self.n_heads, self.head_dim])?
            .add(&self.pos_bias_u.expand_dims(0)?)?;
        let q_u = q.transpose_axes(&[0, 2, 1, 3])?;
        let q_v = q
            .subtract(&self.pos_bias_u.expand_dims(0)?)?
            .add(&self.pos_bias_v.expand_dims(0)?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, t, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, t, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let p = p
            .reshape(&[p.shape()[0], pos_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let matrix_bd = q_v.matmul(&p.transpose_axes(&[0, 1, 3, 2])?)?;
        let matrix_bd = rel_shift(&matrix_bd)?;
        // keep the first `t` relative-position columns (upstream: [:, :, :, :k_len])
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
        self.linear_out.forward(&out)
    }
}

/// Conformer relative-position shift (as in the upstream implementation).
fn rel_shift(x: &Array) -> Result<Array> {
    let (b, h, tq) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let pos_len = x.shape()[3];
    // pad the last axis on the left with a zero column, then transpose-shift
    let pad = Array::zeros::<f32>(&[b, h, tq, 1])?;
    let padded = ops::concatenate(&[&pad, x], 3)?;
    let reshaped = padded.reshape(&[b, h, pos_len + 1, tq])?;
    // drop the first row of the (pos_len+1) axis — axis 2, not the batch axis
    let rows = Array::from_slice(&(1..=pos_len).collect::<Vec<_>>(), &[pos_len]);
    let sliced = reshaped.take_axis(&rows, 2)?;
    Ok(sliced.reshape(&[b, h, tq, pos_len])?)
}

struct ConvModule {
    pointwise1: Dense, // (2*d_model, 1, d_model)
    depthwise: Dense,  // (d_model, k, 1)
    bn_weight: Array,
    bn_bias: Array,
    bn_mean: Array,
    bn_var: Array,
    pointwise2: Dense,
    kernel: i32,
}

impl ConvModule {
    fn load(w: &Weights, prefix: &str, kernel: i32) -> Result<Self> {
        Ok(Self {
            pointwise1: w.dense(&format!("{prefix}.pointwise_conv1"))?,
            depthwise: w.dense(&format!("{prefix}.depthwise_conv"))?,
            bn_weight: w.get(&format!("{prefix}.batch_norm.weight"))?.clone(),
            bn_bias: w.get(&format!("{prefix}.batch_norm.bias"))?.clone(),
            bn_mean: w.get(&format!("{prefix}.batch_norm.running_mean"))?.clone(),
            bn_var: w.get(&format!("{prefix}.batch_norm.running_var"))?.clone(),
            pointwise2: w.dense(&format!("{prefix}.pointwise_conv2"))?,
            kernel,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        // pointwise 1x1 -> GLU over the channel axis, then depthwise (kernel)
        // encoder.use_bias=false for parakeet, so these convs have no bias
        let mut x = ops::conv1d(x, &self.pointwise1.weight, 1, 0, 1, 1)?;
        if let Some(b) = &self.pointwise1.bias {
            x = x.add(b)?;
        }
        let x = mlx_rs::nn::glu(&x, 2)?;
        let mut x = ops::conv1d(
            x.as_ref(),
            &self.depthwise.weight,
            1,
            self.kernel / 2,
            1,
            self.depthwise_groups(),
        )?;
        if let Some(b) = &self.depthwise.bias {
            x = x.add(b)?;
        }
        // BatchNorm (inference): (x - mean) / sqrt(var + eps) * w + b
        let var = self.bn_var.add(&Array::from_f32(1e-5))?;
        let denom = var.sqrt()?;
        let x = x
            .subtract(&self.bn_mean)?
            .divide(&denom)?
            .multiply(&self.bn_weight)?
            .add(&self.bn_bias)?;
        let x = silu(&x)?;
        let mut x = ops::conv1d(&x, &self.pointwise2.weight, 1, 0, 1, 1)?;
        if let Some(b) = &self.pointwise2.bias {
            x = x.add(b)?;
        }
        Ok(x)
    }

    fn depthwise_groups(&self) -> i32 {
        // (out, k, 1) with one input channel per group
        self.depthwise.weight.shape()[0]
    }
}

struct ConformerBlock {
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

impl ConformerBlock {
    fn load(w: &Weights, prefix: &str, cfg: &EncoderConfig) -> Result<Self> {
        Ok(Self {
            norm_ff1: LayerNorm::load(w, &format!("{prefix}.norm_feed_forward1"))?,
            ff1: FeedForward::load(w, &format!("{prefix}.feed_forward1"))?,
            norm_att: LayerNorm::load(w, &format!("{prefix}.norm_self_att"))?,
            attn: RelPosAttention::load(
                w,
                &format!("{prefix}.self_attn"),
                cfg.n_heads,
                cfg.d_model,
            )?,
            norm_conv: LayerNorm::load(w, &format!("{prefix}.norm_conv"))?,
            conv: ConvModule::load(w, &format!("{prefix}.conv"), cfg.conv_kernel_size)?,
            norm_ff2: LayerNorm::load(w, &format!("{prefix}.norm_feed_forward2"))?,
            ff2: FeedForward::load(w, &format!("{prefix}.feed_forward2"))?,
            norm_out: LayerNorm::load(w, &format!("{prefix}.norm_out"))?,
        })
    }

    fn forward(&self, x: &Array, pos_emb: &Array) -> Result<Array> {
        let mut h = x.add(&self.ff1.forward(&self.norm_ff1.forward(x)?)?.multiply(&Array::from_f32(0.5))?)?;
        let normed = self.norm_att.forward(&h)?;
        h = h.add(&self.attn.forward(&normed, pos_emb)?)?;
        h = h.add(&self.conv.forward(&self.norm_conv.forward(&h)?)?)?;
        h = h.add(&self.ff2.forward(&self.norm_ff2.forward(&h)?)?.multiply(&Array::from_f32(0.5))?)?;
        self.norm_out.forward(&h)
    }
}

/// dw-striding Conv2d subsampling (8x) + output projection.
struct Subsampling {
    conv0: (Array, Array),
    conv2: (Array, Array),
    conv3: (Array, Array),
    conv5: (Array, Array),
    conv6: (Array, Array),
    out: Dense,
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
            conv0: pair("encoder.pre_encode.conv.0")?,
            conv2: pair("encoder.pre_encode.conv.2")?,
            conv3: pair("encoder.pre_encode.conv.3")?,
            conv5: pair("encoder.pre_encode.conv.5")?,
            conv6: pair("encoder.pre_encode.conv.6")?,
            out: w.dense("encoder.pre_encode.out")?,
        })
    }

    /// `mel` is `[T, n_mels]`; returns `[T/8, d_model]`.
    fn forward(&self, mel: &Array) -> Result<(Array, i32)> {
        // [T, F] -> [1, T, F, 1] (NHWC)
        let x = mel.expand_dims(0)?.expand_dims(3)?;
        let mut h = conv_bn(&x, &self.conv0, 2, 1, 1)?;
        h = relu(&h)?;
        h = conv_bn(&h, &self.conv2, 2, 1, self.conv2.0.shape()[0])?; // depthwise
        h = conv_bn(&h, &self.conv3, 1, 0, 1)?;
        h = relu(&h)?;
        h = conv_bn(&h, &self.conv5, 2, 1, self.conv5.0.shape()[0])?;
        h = conv_bn(&h, &self.conv6, 1, 0, 1)?;
        h = relu(&h)?;
        // [1, T', F', C] -> [1, T', C*F'] as upstream does (channel-major flatten)
        let (b, t, f, c) = (h.shape()[0], h.shape()[1], h.shape()[2], h.shape()[3]);
        let h = h
            .transpose_axes(&[0, 1, 3, 2])?
            .reshape(&[b, t, c * f])?;
        Ok((self.out.forward(&h)?, t))
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

// ---------------------------------------------------------------------------
// LSTM prediction network + joint
// ---------------------------------------------------------------------------

struct LstmLayer {
    wx: Array,   // (4*hidden, input)
    wh: Array,   // (4*hidden, hidden)
    bias: Array, // (4*hidden,)
    hidden: i32,
}

#[derive(Clone)]
struct LstmState {
    h: Vec<Array>,
    c: Vec<Array>,
}

impl LstmState {
    fn zeros(batch: i32, hidden: i32, layers: usize) -> Self {
        let h = (0..layers).map(|_| Array::zeros::<f32>(&[batch, hidden]).unwrap()).collect();
        let c = (0..layers).map(|_| Array::zeros::<f32>(&[batch, hidden]).unwrap()).collect();
        Self { h, c }
    }
}

/// 2-layer LSTM (MLX gate order: input, forget, cell, output).
struct PredictNetwork {
    embed: Array, // (vocab+1, pred_hidden)
    layers: Vec<LstmLayer>,
}

impl PredictNetwork {
    fn load(w: &Weights, hidden: i32, layers: usize) -> Result<Self> {
        let mut out = Vec::new();
        for i in 0..layers {
            let p = format!("decoder.prediction.dec_rnn.lstm.{i}");
            out.push(LstmLayer {
                wx: w.get(&format!("{p}.Wx"))?.clone(),
                wh: w.get(&format!("{p}.Wh"))?.clone(),
                bias: w.get(&format!("{p}.bias"))?.clone(),
                hidden,
            });
        }
        Ok(Self {
            embed: w.get("decoder.prediction.embed.weight")?.clone(),
            layers: out,
        })
    }

    /// One step for `token` (blank id zeroes the embedding, as upstream does).
    fn step(&self, token: u32, blank: u32, state: &mut LstmState) -> Result<Array> {
        let idx = Array::from_slice(&[token as i32], &[1]);
        let mut x = self.embed.take_axis(&idx, 0)?;
        if token == blank {
            x = Array::zeros::<f32>(&x.shape())?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let gates = x
                .matmul(&layer.wx.transpose_axes(&[1, 0])?)?
                .add(&state.h[i].matmul(&layer.wh.transpose_axes(&[1, 0])?)?)?
                .add(&layer.bias)?;
            let parts = gates.split_equal(4, 1)?;
            let (i_gate, f_gate, g_gate, o_gate) = (
                mlx_rs::nn::sigmoid(&parts[0])?,
                mlx_rs::nn::sigmoid(&parts[1])?,
                ops::tanh(&parts[2])?,
                mlx_rs::nn::sigmoid(&parts[3])?,
            );
            let c = f_gate.multiply(&state.c[i])?.add(&i_gate.multiply(&g_gate)?)?;
            let h = o_gate.multiply(&ops::tanh(&c)?)?;
            state.c[i] = c.clone();
            state.h[i] = h.clone();
            x = h;
        }
        Ok(x)
    }
}

struct JointNetwork {
    enc: Dense,
    pred: Dense,
    /// `joint.joint_net.2` — the only learned layer (index 0 is ReLU, 1 Identity)
    out: Dense,
}

impl JointNetwork {
    fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            enc: w.dense("joint.enc")?,
            pred: w.dense("joint.pred")?,
            out: w.dense("joint.joint_net.2")?,
        })
    }

    fn forward(&self, enc: &Array, pred: &Array) -> Result<Array> {
        // enc [B, 1, 640] + pred [B, 640] -> [B, 1, 1, 640] -> ReLU -> Linear
        let x = self
            .enc
            .forward(enc)?
            .expand_dims(2)?
            .add(&self.pred.forward(pred)?.expand_dims(1)?)?;
        self.out.forward(&relu(&x)?)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Parakeet {
    pub config: ParakeetConfig,
    subsampling: Subsampling,
    enc_blocks: Vec<ConformerBlock>,
    pos_embedding: Array,
    predictor: PredictNetwork,
    joint: JointNetwork,
}

/// One decoded token with its start time and duration in seconds.
#[derive(Debug, Clone)]
pub struct AlignedToken {
    pub id: u32,
    pub text: String,
    pub start: f32,
    pub duration: f32,
}

impl Parakeet {
    pub fn load(dir: impl AsRef<Path>, config: ParakeetConfig) -> Result<Self> {
        let dir = dir.as_ref();
        let w = Weights::load_dir(dir)?;
        let enc_blocks = (0..config.encoder.n_layers)
            .map(|i| ConformerBlock::load(&w, &format!("encoder.layers.{i}"), &config.encoder))
            .collect::<Result<Vec<_>>>()?;
        let pos_embedding = relative_positional(
            config.encoder.d_model as usize,
            config.encoder.pos_emb_max_len as usize,
        )?;
        Ok(Self {
            subsampling: Subsampling::load(&w)?,
            enc_blocks,
            pos_embedding,
            predictor: PredictNetwork::load(&w, 640, 2)?,
            joint: JointNetwork::load(&w)?,
            config,
        })
    }

    /// Encode log-mel `[T, n_mels]` into `[T', d_model]`.
    pub fn encode(&self, mel: &Array) -> Result<Array> {
        let (mut h, t) = self.subsampling.forward(mel)?;
        if self.config.encoder.xscaling {
            let scale = (self.config.encoder.d_model as f32).sqrt();
            h = h.multiply(&Array::from_f32(scale))?;
        }
        // centred slice of the relative-position buffer, matching RelPositionalEncoding
        let buf_len = self.pos_embedding.shape()[1];
        let start = buf_len / 2 - (t - 1);
        let end = buf_len / 2 + (t - 1) + 1;
        let idx = Array::from_slice(&(start..end).collect::<Vec<_>>(), &[end - start]);
        let pos = self.pos_embedding.take_axis(&idx, 1)?;
        for block in &self.enc_blocks {
            h = block.forward(&h, &pos)?;
        }
        Ok(h)
    }

    /// Greedy TDT decode of one encoded utterance.
    pub fn decode(&self, mel: &Array) -> Result<Vec<AlignedToken>> {
        let features = self.encode(mel)?;
        let blank = self.config.blank_id();
        let frame_secs = self.config.encoder.subsampling_factor as f32 / SAMPLE_RATE * 160.0;
        let max_symbols = self.config.max_symbols();

        let mut state = LstmState::zeros(1, 640, self.predictor.layers.len());
        let mut last_token = blank;
        let mut tokens = Vec::new();
        let mut time: i32 = 0;
        let mut new_symbols = 0usize;
        let total = features.shape()[1];

        while time < total {
            let idx = Array::from_slice(&[time], &[1]);
            let feature = features.take_axis(&idx, 1)?;
            // Upstream only commits the LSTM state when a symbol is emitted;
            // a blank prediction keeps the previous hidden/cell state.
            let mut next_state = state.clone();
            let pred = self.predictor.step(last_token, blank, &mut next_state)?;
            let logits = self.joint.forward(&feature, &pred)?;
            logits.eval()?;

            // [1, 1, 1, vocab+1+durations]
            let logits = logits.squeeze_axes(&[0, 1, 2])?;
            let all = logits.as_slice::<f32>();
            let vocab_dim = (blank + 1) as usize;
            let (pred_id, _) = argmax(&all[..vocab_dim]);
            let (dur_idx, _) = argmax(&all[vocab_dim..]);
            let duration = *self
                .config
                .decoding
                .durations
                .get(dur_idx)
                .ok_or_else(|| anyhow!("duration index {dur_idx} out of range"))?;

            if pred_id as u32 != blank {
                state = next_state;
                last_token = pred_id as u32;
                let piece = self
                    .config
                    .joint
                    .vocabulary
                    .get(last_token as usize)
                    .cloned()
                    .unwrap_or_default();
                if !is_special_piece(&piece) {
                    tokens.push(AlignedToken {
                        id: last_token,
                        text: piece.replace('▁', " "),
                        start: time as f32 * frame_secs,
                        duration: duration as f32 * frame_secs,
                    });
                }
            }

            time += duration;
            new_symbols += 1;
            if duration != 0 {
                new_symbols = 0;
            } else if max_symbols.is_some_and(|m| m <= new_symbols) {
                time += 1;
                new_symbols = 0;
            }
        }
        Ok(tokens)
    }

    /// Plain transcript for a log-mel input.
    pub fn transcribe(&self, mel: &Array) -> Result<String> {
        Ok(self
            .decode(mel)?
            .into_iter()
            .map(|t| t.text)
            .collect::<String>()
            .trim()
            .to_string())
    }
}

fn argmax(values: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, v) in values.iter().enumerate() {
        if *v > best.1 {
            best = (i, *v);
        }
    }
    best
}

fn is_special_piece(piece: &str) -> bool {
    (piece.starts_with("<|") && piece.ends_with("|>")) || piece == "<unk>" || piece == "<pad>"
}

/// NeMo relative positional encoding buffer, `[1, 2*max_len-1, d_model]`.
fn relative_positional(d_model: usize, max_len: usize) -> Result<Array> {
    let count = 2 * max_len - 1;
    let mut pe = vec![0f32; count * d_model];
    let log_inc = (10_000f32).ln() / d_model as f32;
    for (row, pos) in (0..count).map(|i| (i, (max_len - 1) as f32 - i as f32)) {
        for j in 0..(d_model / 2) {
            let div = (-((2 * j) as f32) * log_inc).exp();
            let v = pos * div;
            pe[row * d_model + 2 * j] = v.sin();
            pe[row * d_model + 2 * j + 1] = v.cos();
        }
    }
    Ok(Array::from_slice(&pe, &[1, count as i32, d_model as i32]))
}

/// Load a 16 kHz WAV and compute the model's own log-mel (NeMo params).
pub fn mel_for(path: &Path, cfg: &PreprocessorConfig) -> Result<Array> {
    let audio = any2md_audio::read_wav_16k(path)?;
    let mut nemo = any2md_audio::NemoMel::parakeet();
    nemo.n_fft = cfg.n_fft;
    nemo.n_mels = cfg.features;
    nemo.win_length = (cfg.window_size * cfg.sample_rate as f32) as usize;
    nemo.hop = (cfg.window_stride * cfg.sample_rate as f32) as usize;
    nemo.pad_to = cfg.pad_to.max(1);
    let mel = any2md_audio::nemo_mel(&audio, &nemo)?;
    // NeMo returns [mels, frames]; the encoder wants [frames, mels]
    Ok(mel.transpose_axes(&[1, 0])?)
}

/// Guard against silently running with a non-f32 checkpoint (we mirror f32 math).
pub fn check_dtype(w: &Weights) -> Result<()> {
    let dt = w.get("encoder.pre_encode.out.weight")?.dtype();
    if dt != Dtype::Float32 {
        bail!("parakeet port expects an f32 checkpoint, got {dt:?}");
    }
    Ok(())
}
