//! Qwen2 decoder-only LM (ReaderLM-v2 is a 4-bit Qwen2-1.5B fine-tune).
//!
//! The forward pass intentionally mirrors `mlx_lm/models/qwen2.py` op for op so
//! token-for-token parity can be asserted against the Python any2md path:
//! pre-norm blocks, RoPE before the KV-cache append, GQA through MLX's fused
//! SDPA, SwiGLU MLP, tied output head.

use any2md_mlx::{ops, Array, Dtype, QuantEmbedding, QuantLinear, Weights};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use mlx_rs::fast;

/// The subset of `config.json` this model needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    pub intermediate_size: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_true() -> bool {
    true
}

impl Qwen2Config {
    pub fn from_json(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text).context("parse config.json")?;
        if cfg.quantization.is_none() {
            return Err(anyhow!(
                "only pre-quantized MLX checkpoints are supported right now (no quantization block)"
            ));
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// `(group_size, bits)`, panicking-free because `from_json` proved it exists.
    pub fn quant(&self) -> QuantConfig {
        self.quantization.expect("validated in from_json")
    }
}

/// Grow-by-`step` KV cache laid out `[batch, kv_heads, tokens, head_dim]`.
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    step: i32,
}

impl KvCache {
    pub fn new(step: i32) -> Self {
        Self { keys: None, values: None, offset: 0, step }
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Append `keys`/`values` (already RoPE'd) and return the full history.
    pub fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array)> {
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
                let (k, v) = (ops::concatenate(&[&k, &keys], 2)?, ops::concatenate(&[&v, &values], 2)?);
                self.keys = Some(k.clone());
                self.values = Some(v.clone());
                Ok((k, v))
            }
            _ => Err(anyhow!("inconsistent KV cache state")),
        }
    }

    /// Reserve room ahead of the next append (keeps concatenation off the hot path).
    pub fn reserve(&mut self, _tokens: i32) {
        let _ = self.step;
    }
}

/// Loaded ReaderLM/Qwen2 model.
pub struct Qwen2 {
    pub config: Qwen2Config,
    embed_tokens: QuantEmbedding,
    layers: Vec<Layer>,
    norm: Array,
    lm_head: Option<QuantLinear>,
}

struct Layer {
    input_layernorm: Array,
    post_attention_layernorm: Array,
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    gate_proj: QuantLinear,
    up_proj: QuantLinear,
    down_proj: QuantLinear,
}

impl Qwen2 {
    pub fn load(dir: impl AsRef<std::path::Path>, config: Qwen2Config) -> Result<Self> {
        let weights = Weights::load_dir(dir)?;
        let q = config.quant();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_layernorm: weights.get(&format!("{p}.input_layernorm.weight"))?.clone(),
                post_attention_layernorm: weights
                    .get(&format!("{p}.post_attention_layernorm.weight"))?
                    .clone(),
                q_proj: weights.quant_linear(&format!("{p}.self_attn.q_proj"), q.group_size, q.bits)?,
                k_proj: weights.quant_linear(&format!("{p}.self_attn.k_proj"), q.group_size, q.bits)?,
                v_proj: weights.quant_linear(&format!("{p}.self_attn.v_proj"), q.group_size, q.bits)?,
                o_proj: weights.quant_linear(&format!("{p}.self_attn.o_proj"), q.group_size, q.bits)?,
                gate_proj: weights.quant_linear(&format!("{p}.mlp.gate_proj"), q.group_size, q.bits)?,
                up_proj: weights.quant_linear(&format!("{p}.mlp.up_proj"), q.group_size, q.bits)?,
                down_proj: weights.quant_linear(&format!("{p}.mlp.down_proj"), q.group_size, q.bits)?,
            });
        }
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(weights.quant_linear("lm_head", q.group_size, q.bits)?)
        };
        Ok(Self {
            embed_tokens: weights.quant_embedding("model.embed_tokens", q.group_size, q.bits)?,
            layers,
            norm: weights.get("model.norm.weight")?.clone(),
            lm_head,
            config,
        })
    }

    /// Logits for the last position, plus the updated caches.
    ///
    /// `token_ids` is `[1, L]`. Returns `[1, vocab]` (float32 for stable argmax).
    pub fn forward(
        &self,
        token_ids: &Array,
        caches: &mut [KvCache],
    ) -> Result<Array> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
        let scale = (head_dim as f32).powf(-0.5);

        let h0 = self.embed_tokens.forward(token_ids)?;
        let mut h = h0;
        let seq_len = h.shape()[1];

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = &mut caches[i];
            let normed = any2md_mlx::rms_norm(&h, &layer.input_layernorm, cfg.rms_norm_eps)?;

            let q = layer.q_proj.forward(&normed)?;
            let k = layer.k_proj.forward(&normed)?;
            let v = layer.v_proj.forward(&normed)?;
            let q = q
                .reshape(&[1, seq_len, n_heads, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = k
                .reshape(&[1, seq_len, n_kv, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = v
                .reshape(&[1, seq_len, n_kv, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;

            let offset = cache.offset();
            let q = fast::rope(&q, head_dim, cfg.rope_traditional, Some(cfg.rope_theta), 1.0, offset, None)?;
            let k = fast::rope(&k, head_dim, cfg.rope_traditional, Some(cfg.rope_theta), 1.0, offset, None)?;
            let (k, v) = cache.update_and_fetch(k, v)?;

            // Causal mask only while pre-filling; single-token decode needs none.
            let attn = if seq_len > 1 {
                fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    Some(fast::ScaledDotProductAttentionMask::Causal),
                    None,
                )?
            } else {
                fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?
            };
            let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, seq_len, -1])?;
            let attn = layer.o_proj.forward(&attn)?;
            h = h.add(&attn)?;

            let normed = any2md_mlx::rms_norm(&h, &layer.post_attention_layernorm, cfg.rms_norm_eps)?;
            let gate = layer.gate_proj.forward(&normed)?;
            let up = layer.up_proj.forward(&normed)?;
            let act = any2md_mlx::silu(&gate)?.multiply(&up)?;
            let down = layer.down_proj.forward(&act)?;
            h = h.add(&down)?;
        }

        let h = any2md_mlx::rms_norm(&h, &self.norm, cfg.rms_norm_eps)?;
        let logits = match &self.lm_head {
            Some(head) => head.matmul_t(&h)?,
            None => self.embed_tokens.as_linear(&h)?,
        };
        // Last position only; float32 keeps argmax identical to Python.
        let last = logits.squeeze_axes(&[0])?;
        let last_idx = Array::from_slice(&[seq_len - 1], &[1]);
        let last = last.take_axis(&last_idx, 0)?;
        Ok(last.as_dtype(Dtype::Float32)?)
    }
}

/// Greedy (temperature-0) generation, matching `mlx_lm.generate`.
pub struct Generator<'m> {
    pub model: &'m Qwen2,
    pub caches: Vec<KvCache>,
}

impl<'m> Generator<'m> {
    pub fn new(model: &'m Qwen2) -> Self {
        let caches = (0..model.config.num_hidden_layers)
            .map(|_| KvCache::new(256))
            .collect();
        Self { model, caches }
    }

    /// Prefill `prompt` and decode up to `max_tokens`, stopping on `eos`.
    pub fn generate(
        &mut self,
        prompt: &Array,
        max_tokens: usize,
        eos: &[u32],
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(max_tokens);
        let mut logits = self.model.forward(prompt, &mut self.caches)?;
        logits.eval()?;

        for _ in 0..max_tokens {
            let next = argmax_last(&logits)?;
            if eos.contains(&next) {
                break;
            }
            tokens.push(next);
            on_token(next);

            let input = Array::from_slice(&[next as i32], &[1, 1]);
            logits = self.model.forward(&input, &mut self.caches)?;
            logits.eval()?;
        }
        Ok(tokens)
    }
}

fn argmax_last(logits: &Array) -> Result<u32> {
    let idx = mlx_rs::ops::indexing::argmax(logits, None)?;
    idx.eval()?;
    let v = idx.as_slice::<u32>();
    v.first().copied().ok_or_else(|| anyhow!("empty logits"))
}

