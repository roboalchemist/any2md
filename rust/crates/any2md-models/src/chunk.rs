//! Long-audio chunked transcription and overlap merging.
//!
//! any2md's `yt` path calls Parakeet with `chunk_duration=30.0`, so audio is
//! cut into 30 s windows with a 2 s overlap and the per-chunk token streams are
//! stitched back together. This mirrors `mlx_audio/stt/models/nemo/alignment.py`
//! (`merge_longest_contiguous`, with `merge_longest_common_subsequence` as the
//! fallback) so the merged transcript matches the Python implementation.

use anyhow::Result;

use crate::parakeet::{AlignedToken, Parakeet};

pub const DEFAULT_CHUNK_SECONDS: f32 = 30.0;
pub const DEFAULT_OVERLAP_SECONDS: f32 = 2.0;

impl AlignedToken {
    /// End time in seconds (upstream exposes `.end` as start + duration).
    pub fn end(&self) -> f32 {
        self.start + self.duration
    }
}

/// Transcribe `audio` in overlapping chunks and merge the token streams.
pub fn transcribe_chunked(
    model: &Parakeet,
    audio: &any2md_mlx::Array,
    chunk_seconds: f32,
    overlap_seconds: f32,
) -> Result<Vec<AlignedToken>> {
    transcribe_chunked_dtype(model, audio, chunk_seconds, overlap_seconds, true)
}

/// Chunked transcription with an explicit audio dtype choice (see [`mel_from_audio`]).
pub fn transcribe_chunked_dtype(
    model: &Parakeet,
    audio: &any2md_mlx::Array,
    chunk_seconds: f32,
    overlap_seconds: f32,
    bf16_audio: bool,
) -> Result<Vec<AlignedToken>> {
    let n = audio.shape()[0] as usize;
    let sample_rate = model.config.preprocessor.sample_rate;
    let chunk_samples = (chunk_seconds * sample_rate as f32) as usize;
    let overlap_samples = (overlap_seconds * sample_rate as f32) as usize;
    let step = chunk_samples.saturating_sub(overlap_samples).max(1);

    let nemo = model.nemo_mel_config();
    let mut all: Vec<AlignedToken> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk_samples).min(n);
        let chunk = audio.take_axis(
            &any2md_mlx::Array::from_slice(
                &(start..end).map(|i| i as i32).collect::<Vec<_>>(),
                &[(end - start) as i32],
            ),
            0,
        )?;
        let chunk = if bf16_audio {
            chunk.as_dtype(any2md_mlx::Dtype::Bfloat16)?.as_dtype(any2md_mlx::Dtype::Float32)?
        } else {
            chunk
        };
        let mel = any2md_audio::nemo_mel(&chunk, &nemo)?;
        let mel = mel.transpose_axes(&[1, 0])?;
        let mut tokens = model.decode(&mel)?;

        let offset = start as f32 / sample_rate as f32;
        for t in &mut tokens {
            t.start += offset;
        }
        if tokens.is_empty() {
            // nothing recognised in this window; keep going
        } else if all.is_empty() {
            all = tokens;
        } else {
            all = match merge_longest_contiguous(&all, &tokens, overlap_seconds) {
                Ok(merged) => merged,
                Err(_) => merge_longest_common_subsequence(&all, &tokens, overlap_seconds),
            };
        }

        if end == n {
            break;
        }
        start += step;
    }
    Ok(all)
}

fn cutoff_split(a: &[AlignedToken], b: &[AlignedToken]) -> Vec<AlignedToken> {
    let a_end = a.last().map(|t| t.end()).unwrap_or(0.0);
    let b_start = b.first().map(|t| t.start).unwrap_or(0.0);
    let cutoff = (a_end + b_start) / 2.0;
    let mut out: Vec<AlignedToken> = a.iter().filter(|t| t.end() <= cutoff).cloned().collect();
    out.extend(b.iter().filter(|t| t.start >= cutoff).cloned());
    out
}

fn splice(
    a: &[AlignedToken],
    b: &[AlignedToken],
    pairs: &[(usize, usize)],
    a_start_idx: usize,
) -> Vec<AlignedToken> {
    let idx_a: Vec<usize> = pairs.iter().map(|(i, _)| a_start_idx + i).collect();
    let idx_b: Vec<usize> = pairs.iter().map(|(_, j)| *j).collect();

    let mut out: Vec<AlignedToken> = a[..idx_a[0]].to_vec();
    for k in 0..pairs.len() {
        out.push(a[idx_a[k]].clone());
        if k + 1 < pairs.len() {
            let gap_a = &a[idx_a[k] + 1..idx_a[k + 1]];
            let gap_b = &b[idx_b[k] + 1..idx_b[k + 1]];
            if gap_b.len() > gap_a.len() {
                out.extend_from_slice(gap_b);
            } else {
                out.extend_from_slice(gap_a);
            }
        }
    }
    out.extend_from_slice(&b[idx_b[idx_b.len() - 1] + 1..]);
    out
}

/// Upstream `merge_longest_contiguous` — errors when too few overlapping pairs.
pub fn merge_longest_contiguous(
    a: &[AlignedToken],
    b: &[AlignedToken],
    overlap_duration: f32,
) -> Result<Vec<AlignedToken>> {
    if a.is_empty() || b.is_empty() {
        return Ok(if a.is_empty() { b.to_vec() } else { a.to_vec() });
    }
    let a_end = a[a.len() - 1].end();
    let b_start = b[0].start;
    if a_end <= b_start {
        let mut out = a.to_vec();
        out.extend_from_slice(b);
        return Ok(out);
    }

    let overlap_a: Vec<usize> = (0..a.len())
        .filter(|i| a[*i].end() > b_start - overlap_duration)
        .collect();
    let overlap_b: Vec<usize> = (0..b.len())
        .filter(|j| b[*j].start < a_end + overlap_duration)
        .collect();
    let enough_pairs = overlap_a.len() / 2;
    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        return Ok(cutoff_split(a, b));
    }

    let mut best: Vec<(usize, usize)> = Vec::new();
    for (i, ai) in overlap_a.iter().enumerate() {
        for (j, bj) in overlap_b.iter().enumerate() {
            let (ta, tb) = (&a[*ai], &b[*bj]);
            if ta.id == tb.id && (ta.start - tb.start).abs() < overlap_duration / 2.0 {
                let mut cur = Vec::new();
                let (mut k, mut l) = (i, j);
                while k < overlap_a.len()
                    && l < overlap_b.len()
                    && a[overlap_a[k]].id == b[overlap_b[l]].id
                    && (a[overlap_a[k]].start - b[overlap_b[l]].start).abs() < overlap_duration / 2.0
                {
                    cur.push((k, l));
                    k += 1;
                    l += 1;
                }
                if cur.len() > best.len() {
                    best = cur;
                }
            }
        }
    }

    if best.len() >= enough_pairs {
        Ok(splice(a, b, &best, a.len() - overlap_a.len()))
    } else {
        anyhow::bail!("no contiguous overlap of {enough_pairs} pairs")
    }
}

/// Upstream `merge_longest_common_subsequence` — the fallback path.
pub fn merge_longest_common_subsequence(
    a: &[AlignedToken],
    b: &[AlignedToken],
    overlap_duration: f32,
) -> Vec<AlignedToken> {
    if a.is_empty() || b.is_empty() {
        return if a.is_empty() { b.to_vec() } else { a.to_vec() };
    }
    let a_end = a[a.len() - 1].end();
    let b_start = b[0].start;
    if a_end <= b_start {
        let mut out = a.to_vec();
        out.extend_from_slice(b);
        return out;
    }

    let overlap_a: Vec<usize> = (0..a.len())
        .filter(|i| a[*i].end() > b_start - overlap_duration)
        .collect();
    let overlap_b: Vec<usize> = (0..b.len())
        .filter(|j| b[*j].start < a_end + overlap_duration)
        .collect();
    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        return cutoff_split(a, b);
    }

    let (m, n) = (overlap_a.len(), overlap_b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            let (ta, tb) = (&a[overlap_a[i - 1]], &b[overlap_b[j - 1]]);
            dp[i][j] = if ta.id == tb.id && (ta.start - tb.start).abs() < overlap_duration / 2.0 {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    let mut pairs = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        let (ta, tb) = (&a[overlap_a[i - 1]], &b[overlap_b[j - 1]]);
        if ta.id == tb.id && (ta.start - tb.start).abs() < overlap_duration / 2.0 {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    if pairs.is_empty() {
        return cutoff_split(a, b);
    }
    splice(a, b, &pairs, a.len() - overlap_a.len())
}
