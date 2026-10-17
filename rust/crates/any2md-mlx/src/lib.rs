//! MLX plumbing shared by every any2md model: device + memory policy, reading
//! safetensors into [`mlx_rs::Array`]s, and the quantized layer kinds MLX
//! checkpoints use (4/8-bit affine groups).
//!
//! Memory policy mirrors the Python implementation: the buffer cache is capped
//! (2 GiB by default, `ANY2MD_MLX_CACHE_LIMIT_GB`) and callers release it
//! between pipeline stages, because MLX otherwise keeps freed Metal buffers
//! alive until the process exits.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
pub use mlx_rs::{ops, Array, Dtype};

pub use mlx_rs;

pub const DEFAULT_CACHE_LIMIT_GB: f32 = 2.0;

/// Initialise MLX: pick the GPU, then apply the cache-limit policy.
///
/// Must run before any array is created so the limit applies to the device
/// allocator. Safe to call more than once.
pub fn init() -> Result<()> {
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    configure_memory()
}

/// Apply `ANY2MD_MLX_CACHE_LIMIT_GB` (default [`DEFAULT_CACHE_LIMIT_GB`]).
pub fn configure_memory() -> Result<()> {
    let limit_gb = std::env::var("ANY2MD_MLX_CACHE_LIMIT_GB")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(DEFAULT_CACHE_LIMIT_GB)
        .max(0.0);
    let limit = (limit_gb as f64 * 1024.0 * 1024.0 * 1024.0) as usize;
    mlx_rs::memory::set_cache_limit(limit).context("set MLX cache limit")?;
    Ok(())
}

/// Drop unreferenced arrays and hand cached Metal buffers back to the OS.
pub fn release_memory() {
    let _ = mlx_rs::memory::clear_cache();
}

/// Current allocator counters, for benchmark output.
pub fn memory_stats() -> Result<(usize, usize, usize)> {
    Ok((
        mlx_rs::memory::active_memory()?,
        mlx_rs::memory::cache_memory()?,
        mlx_rs::memory::peak_memory()?,
    ))
}

/// All tensors of one MLX checkpoint (all `*.safetensors` shards merged).
pub struct Weights {
    map: HashMap<String, Array>,
}

impl Weights {
    /// Load every `*.safetensors` file in `dir`, merging shards by tensor name.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("read model dir {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(anyhow!("no .safetensors files in {}", dir.display()));
        }
        let mut map = HashMap::new();
        for file in files {
            let bytes = std::fs::read(&file)
                .with_context(|| format!("read weights {}", file.display()))?;
            let st = safetensors::SafeTensors::deserialize(&bytes)
                .with_context(|| format!("parse safetensors {}", file.display()))?;
            for (name, view) in st.tensors() {
                let array: Array = view
                    .try_into()
                    .with_context(|| format!("tensor {name} -> mlx array"))?;
                map.insert(name, array);
            }
        }
        tracing::debug!(tensors = map.len(), "loaded weights");
        Ok(Self { map })
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        self.map
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor: {name}"))
    }

    /// Return `Some` when the tensor exists (used for optional biases/tied heads).
    pub fn maybe(&self, name: &str) -> Option<&Array> {
        self.map.get(name)
    }

    /// A quantized linear layer stored under `prefix` (`{prefix}.weight`,
    /// `.scales`, `.biases` and an optional float `.bias`).
    pub fn quant_linear(&self, prefix: &str, group_size: i32, bits: i32) -> Result<QuantLinear> {
        Ok(QuantLinear {
            weight: self.get(&format!("{prefix}.weight"))?.clone(),
            scales: self.get(&format!("{prefix}.scales"))?.clone(),
            qbiases: self.maybe(&format!("{prefix}.biases")).cloned(),
            bias: self.maybe(&format!("{prefix}.bias")).cloned(),
            group_size,
            bits,
        })
    }

    /// A quantized embedding table stored under `prefix`.
    pub fn quant_embedding(
        &self,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<QuantEmbedding> {
        Ok(QuantEmbedding {
            weight: self.get(&format!("{prefix}.weight"))?.clone(),
            scales: self.get(&format!("{prefix}.scales"))?.clone(),
            qbiases: self.maybe(&format!("{prefix}.biases")).cloned(),
            group_size,
            bits,
        })
    }

    /// A plain (unquantized) linear weight under `prefix`.
    pub fn dense(&self, prefix: &str) -> Result<Dense> {
        Ok(Dense {
            weight: self.get(&format!("{prefix}.weight"))?.clone(),
            bias: self.maybe(&format!("{prefix}.bias")).cloned(),
        })
    }
}

/// Affine-quantized linear layer: `y = x @ dequant(w).T + bias`.
pub struct QuantLinear {
    pub weight: Array,
    pub scales: Array,
    pub qbiases: Option<Array>,
    pub bias: Option<Array>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantLinear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = self.matmul_t(x)?;
        match &self.bias {
            Some(b) => Ok(y.add(b)?),
            None => Ok(y),
        }
    }

    /// `x @ dequant(w).T` without the float bias (used for tied LM heads).
    pub fn matmul_t(&self, x: &Array) -> Result<Array> {
        Ok(ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            self.qbiases.as_ref(),
            true,
            self.group_size,
            self.bits,
        )?)
    }
}

/// Affine-quantized embedding table.
pub struct QuantEmbedding {
    pub weight: Array,
    pub scales: Array,
    pub qbiases: Option<Array>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantEmbedding {
    /// `table[ids]`, dequantized to the table's float dtype.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let rows = self.weight.take_axis(ids, 0)?;
        let scales = self.scales.take_axis(ids, 0)?;
        let biases = match &self.qbiases {
            Some(b) => Some(b.take_axis(ids, 0)?),
            None => None,
        };
        Ok(ops::dequantize(
            &rows,
            &scales,
            biases.as_ref(),
            self.group_size,
            self.bits,
        )?)
    }

    /// Tied output projection: `x @ dequant(w).T` (no bias).
    pub fn as_linear(&self, x: &Array) -> Result<Array> {
        Ok(ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            self.qbiases.as_ref(),
            true,
            self.group_size,
            self.bits,
        )?)
    }
}

/// Unquantized (fp16/bf16/fp32) linear weight.
pub struct Dense {
    pub weight: Array,
    pub bias: Option<Array>,
}

impl Dense {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let w = self.weight.transpose_axes(&[1, 0])?;
        let y = x.matmul(&w)?;
        match &self.bias {
            Some(b) => Ok(y.add(b)?),
            None => Ok(y),
        }
    }
}

/// Cast an array to another float dtype (`as_dtype` in MLX).
pub fn cast(x: &Array, dtype: Dtype) -> Result<Array> {
    Ok(x.as_dtype(dtype)?)
}

/// Silu activation, matching `mx.nn.silu`.
pub fn silu(x: &Array) -> Result<Array> {
    Ok(x.multiply(&mlx_rs::nn::sigmoid(x)?)?)
}

/// RMS norm with a learned weight (MLX `fast.rms_norm`).
pub fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    Ok(mlx_rs::fast::rms_norm(x, Some(weight), eps)?)
}
