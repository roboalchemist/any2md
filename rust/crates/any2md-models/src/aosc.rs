//! Arrival-Order Speaker Cache (AOSC) compression.
//!
//! Shared by both diarization ports (Sortformer and Nemotron-3), which call the
//! same upstream routine (`Sortformer._compress_spkcache_aosc`).
//!
//! The cache holds a few hundred frames, so the whole selection runs on the CPU
//! in f32: scoring, speech/overlap masking, latest/strong/weak boosts, silence
//! padding and the global top-k. Doing it with explicit loops also pins down the
//! flattening order that upstream uses — frame indices come from a
//! **speaker-major** flattening (`transpose(scores, [0, 2, 1]).reshape(b, -1)`),
//! which is easy to get wrong with an ad-hoc index enumeration.

use any2md_mlx::{Array, Dtype};
use anyhow::Result;

/// AOSC knobs, mirroring the checkpoint's `modules_config`.
#[derive(Debug, Clone)]
pub struct AoscCfg {
    pub num_speakers: usize,
    pub spkcache_len: usize,
    pub sil_frames_per_spk: usize,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
    pub pred_score_threshold: f32,
    pub scores_boost_latest: f32,
}

/// Compress a speaker cache to `cfg.spkcache_len` frames.
///
/// `embs` is `[1, n_frames, emb_dim]`, `preds` is `[1, n_frames, n_spk]` and
/// `mean_sil_emb` is `[1, emb_dim]`. Returns arrays with the input dtypes.
pub fn compress(
    embs: &Array,
    preds: &Array,
    mean_sil_emb: &Array,
    cfg: &AoscCfg,
) -> Result<(Array, Array)> {
    let e_dtype = embs.dtype();
    let p_dtype = preds.dtype();
    let embs_f = embs.as_dtype(Dtype::Float32)?;
    let preds_f = preds.as_dtype(Dtype::Float32)?;
    let sil_f = mean_sil_emb.as_dtype(Dtype::Float32)?;
    embs_f.eval()?;
    preds_f.eval()?;
    sil_f.eval()?;

    let eshape = embs_f.shape().to_vec();
    let pshape = preds_f.shape().to_vec();
    let n = eshape[1] as usize;
    let emb_dim = eshape[2] as usize;
    let n_spk = cfg.num_speakers;
    assert_eq!(pshape[2] as usize, n_spk, "speaker count mismatch");

    let e = embs_f.as_slice::<f32>();
    let p = preds_f.as_slice::<f32>();
    let sil = sil_f.as_slice::<f32>();

    let per_spk = cfg.spkcache_len / n_spk - cfg.sil_frames_per_spk;
    let strong = (per_spk as f32 * cfg.strong_boost_rate).floor() as usize;
    let weak = (per_spk as f32 * cfg.weak_boost_rate).floor() as usize;
    let min_pos = (per_spk as f32 * cfg.min_pos_scores_rate).floor() as usize;
    let threshold = cfg.pred_score_threshold;
    let ln_half = 0.5f32.ln();
    let neg_inf = f32::NEG_INFINITY;

    // 1. log-likelihood-ratio scores
    let mut scores = vec![0f32; n * n_spk];
    for f in 0..n {
        let mut sum_log_1p = 0f32;
        for k in 0..n_spk {
            let v = p[f * n_spk + k];
            sum_log_1p += (1.0 - v).max(threshold).ln();
        }
        for k in 0..n_spk {
            let v = p[f * n_spk + k];
            scores[f * n_spk + k] = v.max(threshold).ln() - (1.0 - v).max(threshold).ln()
                + sum_log_1p
                - ln_half;
        }
    }

    // 2. disable non-speech, then overlapped speech for speakers that already
    //    have enough clean frames
    let mut pos_count = vec![0usize; n_spk];
    for f in 0..n {
        for k in 0..n_spk {
            if p[f * n_spk + k] > 0.5 && scores[f * n_spk + k] > 0.0 {
                pos_count[k] += 1;
            }
        }
    }
    for f in 0..n {
        for k in 0..n_spk {
            let idx = f * n_spk + k;
            if p[idx] <= 0.5 {
                scores[idx] = neg_inf;
            } else if scores[idx] <= 0.0 && pos_count[k] >= min_pos {
                scores[idx] = neg_inf;
            }
        }
    }

    // 3. boost the frames newer than the current cache length
    if cfg.scores_boost_latest > 0.0 && n > cfg.spkcache_len {
        for f in cfg.spkcache_len..n {
            for k in 0..n_spk {
                scores[f * n_spk + k] += cfg.scores_boost_latest;
            }
        }
    }

    // 4./5. strong then weak per-speaker top-k boosts
    for (k_boost, scale) in [(strong, 2.0f32), (weak, 1.0f32)] {
        if k_boost == 0 {
            continue;
        }
        let boost = -scale * ln_half;
        for k in 0..n_spk {
            let mut idx: Vec<usize> = (0..n).filter(|f| scores[f * n_spk + k].is_finite()).collect();
            idx.sort_by(|a, b| {
                scores[*b * n_spk + k]
                    .partial_cmp(&scores[*a * n_spk + k])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            for f in idx.into_iter().take(k_boost) {
                scores[f * n_spk + k] += boost;
            }
        }
    }

    // 6. silence padding rows with +inf
    let n_total = n + cfg.sil_frames_per_spk;
    let mut padded = vec![neg_inf; n_total * n_spk];
    padded[..n * n_spk].copy_from_slice(&scores);
    for j in 0..cfg.sil_frames_per_spk {
        for k in 0..n_spk {
            padded[(n + j) * n_spk + k] = f32::INFINITY;
        }
    }

    // 7. global top-k over the speaker-major flattening
    //    (`transpose(scores, [0, 2, 1]).reshape(b, -1)`)
    let mut candidates: Vec<(f32, usize, usize)> = Vec::with_capacity(n_total * n_spk);
    for k in 0..n_spk {
        for f in 0..n_total {
            candidates.push((padded[f * n_spk + k], k, f));
        }
    }
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    let k = cfg.spkcache_len.min(candidates.len());
    let mut frames: Vec<usize> = candidates[..k]
        .iter()
        .map(|(score, _, f)| if score.is_finite() { *f } else { usize::MAX })
        .collect();
    frames.sort_unstable();

    let n_frames_no_sil = n;
    let mut out_e = vec![0f32; k * emb_dim];
    let mut out_p = vec![0f32; k * n_spk];
    for (slot, frame) in frames.iter().enumerate() {
        let disabled = *frame == usize::MAX || *frame >= n_frames_no_sil;
        if disabled {
            out_e[slot * emb_dim..(slot + 1) * emb_dim].copy_from_slice(&sil[..emb_dim]);
        } else {
            let f = *frame;
            out_e[slot * emb_dim..(slot + 1) * emb_dim]
                .copy_from_slice(&e[f * emb_dim..(f + 1) * emb_dim]);
            out_p[slot * n_spk..(slot + 1) * n_spk]
                .copy_from_slice(&p[f * n_spk..(f + 1) * n_spk]);
        }
    }

    let out_e = Array::from_slice(&out_e, &[1, k as i32, emb_dim as i32]).as_dtype(e_dtype)?;
    let out_p = Array::from_slice(&out_p, &[1, k as i32, n_spk as i32]).as_dtype(p_dtype)?;
    Ok((out_e, out_p))
}

/// Update the running mean silence embedding with newly popped frames.
///
/// A frame counts as silence when the sum of its speaker probabilities is below
/// `sil_threshold` (upstream `_get_silence_profile`). Returns the new mean and
/// the new silence-frame count.
pub fn update_silence_profile(
    mean_sil_emb: &Array,
    n_sil_frames: f32,
    embs: &Array,
    preds: &Array,
    sil_threshold: f32,
) -> Result<(Array, f32)> {
    let dtype = embs.dtype();
    let embs_f = embs.as_dtype(Dtype::Float32)?;
    let preds_f = preds.as_dtype(Dtype::Float32)?;
    let mean_f = mean_sil_emb.as_dtype(Dtype::Float32)?;
    embs_f.eval()?;
    preds_f.eval()?;
    mean_f.eval()?;

    let n = embs_f.shape()[1] as usize;
    let emb_dim = embs_f.shape()[2] as usize;
    let n_spk = preds_f.shape()[2] as usize;
    let e = embs_f.as_slice::<f32>();
    let p = preds_f.as_slice::<f32>();
    let mean = mean_f.as_slice::<f32>();

    let mut count = 0f32;
    let mut sum = vec![0f32; emb_dim];
    for f in 0..n {
        let total: f32 = (0..n_spk).map(|k| p[f * n_spk + k]).sum();
        if total < sil_threshold {
            count += 1.0;
            for d in 0..emb_dim {
                sum[d] += e[f * emb_dim + d];
            }
        }
    }

    let new_n = n_sil_frames + count;
    let denom = new_n.max(1.0);
    let mut updated = vec![0f32; emb_dim];
    for d in 0..emb_dim {
        updated[d] = (mean[d] * n_sil_frames + sum[d]) / denom;
    }
    let updated = Array::from_slice(&updated, &[1, emb_dim as i32]).as_dtype(dtype)?;
    Ok((updated, new_n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_keeps_the_strongest_frames_in_order() {
        // one speaker, 4 frames, cache holds 3 (2 speech + 1 silence slot)
        let preds = Array::from_slice(&[0.9f32, 0.1, 0.8, 0.05], &[1, 4, 1]);
        let embs = Array::from_slice(
            &[1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
            &[1, 4, 2],
        );
        let sil = Array::from_slice(&[0.0f32, 0.0], &[1, 2]);
        let cfg = AoscCfg {
            num_speakers: 1,
            spkcache_len: 3,
            sil_frames_per_spk: 1,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.05,
        };
        let (out_e, out_p) = compress(&embs, &preds, &sil, &cfg).unwrap();
        assert_eq!(out_e.shape(), &[1, 3, 2]);
        assert_eq!(out_p.shape(), &[1, 3, 1]);
        out_e.eval().unwrap();
        let p = out_p.as_slice::<f32>();
        // frame 0 (p=0.9) outranks frame 2 (p=0.8): kept first, in temporal order
        assert_eq!(&p[..1], &[0.9]);
        assert_eq!(&p[1..2], &[0.8]);
        // the third slot is the silence placeholder
        assert_eq!(&p[2..3], &[0.0]);
    }

    #[test]
    fn compression_falls_back_to_silence_when_nothing_scores() {
        // all-silent input: every slot becomes the mean silence embedding
        let preds = Array::from_slice(&[0.0f32, 0.0], &[1, 2, 1]);
        let embs = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[1, 2, 2]);
        let sil = Array::from_slice(&[0.5f32, -0.5], &[1, 2]);
        let cfg = AoscCfg {
            num_speakers: 1,
            spkcache_len: 4,
            sil_frames_per_spk: 1,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.05,
        };
        let (out_e, out_p) = compress(&embs, &preds, &sil, &cfg).unwrap();
        out_e.eval().unwrap();
        out_p.eval().unwrap();
        let e = out_e.as_slice::<f32>();
        assert_eq!(&e[..2], &[0.5, -0.5]);
        let p = out_p.as_slice::<f32>();
        assert!(p.iter().all(|v| *v == 0.0));
    }
}
