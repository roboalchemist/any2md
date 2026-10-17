//! Nemotron-3-Diarization streaming inference.
//!
//! Reproduces `Model.generate()` from
//! `mlx_audio/vad/models/nemotron_diarization/nemotron_diarization.py`:
//! audio is fed in 27.2 s windows (chunk_len 340 * factor 8 * hop 160), each
//! window is processed as 2720-frame central blocks with 320 frames of right
//! context, and the arrival-order speaker cache (AOSC) is compressed whenever
//! the speaker cache exceeds 264 frames.
//!
//! The mel front-end is the checkpoint's own `MelFeatures`: globally indexed
//! frames over a retained PCM suffix (so window boundaries match bit-for-bit)
//! with pre-emphasis applied before the STFT and the checkpoint's filterbank.

use any2md_mlx::{ops, Array};
use anyhow::{Context, Result};

use crate::diarize::{preds_to_segments, Segment};
use crate::nemotron::{DiarConfig, Nemotron};

/// Global-index mel front-end (checkpoint filterbank + window).
pub struct NemotronMel {
    fb: Array,     // [n_mels, n_freqs]
    window: Array, // [n_fft] (win_length padded, centred)
    n_fft: i32,
    hop: i32,
    n_mels: i32,
    preemphasis: f32,
}

impl NemotronMel {
    /// Build from the checkpoint's `preprocessor.fb` / `preprocessor.window`.
    pub fn from_weights(w: &any2md_mlx::Weights, cfg: &DiarConfig) -> Result<Self> {
        let proc = cfg
            .processor_config
            .as_ref()
            .context("nemotron config lacks processor_config")?;
        // fb is stored as [1, n_mels, n_freqs]; drop the batch axis
        let fb = w.get("preprocessor.fb")?.squeeze_axes(&[0])?;
        let win = w.get("preprocessor.window")?.as_dtype(any2md_mlx::Dtype::Float32)?;
        let n_fft = proc.n_fft as i32;
        let wl = proc.win_length as i32;
        let window = if wl < n_fft {
            let left = (n_fft - wl) / 2;
            let right = n_fft - wl - left;
            let zeros_l = Array::zeros::<f32>(&[left])?;
            let zeros_r = Array::zeros::<f32>(&[right])?;
            ops::concatenate(&[&zeros_l, &win, &zeros_r], 0)?
        } else {
            win
        };
        Ok(Self {
            fb,
            window,
            n_fft,
            hop: proc.hop_length as i32,
            n_mels: proc.feature_size as i32,
            preemphasis: if proc.preemphasis > 0.0 { proc.preemphasis } else { 0.97 },
        })
    }

    /// Frames `start..start+count` from `audio` (which begins at global sample
    /// `sample_offset`; `total_samples` is the full stream length).
    pub fn features(
        &self,
        audio: &[f32],
        start: i32,
        count: i32,
        sample_offset: i64,
        total_samples: i64,
    ) -> Result<Array> {
        if count <= 0 {
            return Ok(Array::zeros::<f32>(&[1, self.n_mels, 0])?);
        }
        let n = self.n_fft as usize;
        let len = audio.len() as i64;
        let mut indices = Vec::with_capacity(count as usize * n);
        let mut mask = Vec::with_capacity(count as usize * n);
        let mut prev_indices = Vec::with_capacity(count as usize * n);
        for f in 0..count {
            let frame = start + f;
            for p in 0..n {
                let pos = frame as i64 * self.hop as i64 + p as i64 - (self.n_fft as i64) / 2;
                indices.push(pos);
                prev_indices.push(pos - 1);
                mask.push(if pos >= 0 && pos < total_samples { 1.0f32 } else { 0.0 });
            }
        }
        let local = |g: i64| -> i32 {
            let l = g - sample_offset;
            l.clamp(0, (len - 1).max(0)) as i32
        };
        let idx_arr = Array::from_slice(
            &indices.iter().map(|g| local(*g)).collect::<Vec<_>>(),
            &[(count as usize * n) as i32],
        );
        let prev_arr = Array::from_slice(
            &prev_indices.iter().map(|g| local(*g)).collect::<Vec<_>>(),
            &[(count as usize * n) as i32],
        );
        let audio_arr = Array::from_slice(audio, &[audio.len() as i32]);
        let cur = audio_arr.take_axis(&idx_arr, 0)?.reshape(&[count, n as i32])?;
        let prev = audio_arr.take_axis(&prev_arr, 0)?.reshape(&[count, n as i32])?;

        let mut frames = cur.subtract(&prev.multiply(&Array::from_f32(self.preemphasis))?)?;
        // zero frames whose centre sample falls outside the stream
        let mask_arr = Array::from_slice(&mask, &[count, n as i32]);
        frames = frames.multiply(&mask_arr)?;
        // valid_frames = total_samples // hop
        let valid_frames = (total_samples / self.hop as i64) as i32;
        let frame_ids: Vec<f32> = (start..start + count)
            .map(|f| if f < valid_frames { 1.0 } else { 0.0 })
            .collect();
        let frame_mask = Array::from_slice(&frame_ids, &[count]);

        let windowed = frames.multiply(&self.window)?;
        let spec = mlx_rs::fft::rfft(&windowed, self.n_fft, -1)?;
        let power = spec.abs()?.square()?;
        let mel = power.matmul(&self.fb.transpose_axes(&[1, 0])?)?;
        let mel = mel.add(&Array::from_f32(2f32.powi(-24)))?.log()?;
        let mel = mel.multiply(&frame_mask.expand_dims(1)?)?; // [count, n_mels]
        Ok(mel.transpose_axes(&[1, 0])?.expand_dims(0)?) // [1, n_mels, count]
    }
}

/// Streaming state (AOSC cache + FIFO + retained PCM suffix).
pub struct StreamState {
    spkcache: Option<Array>,
    spkcache_preds: Option<Array>,
    fifo: Option<Array>,
    fifo_preds: Option<Array>,
    compressed: bool,
    frames_processed: i32,
    samples_received: i64,
    sample_offset: i64,
    audio: Vec<f32>,
    finished: bool,
}

impl StreamState {
    pub fn new() -> Self {
        Self {
            spkcache: None,
            spkcache_preds: None,
            fifo: None,
            fifo_preds: None,
            compressed: false,
            frames_processed: 0,
            samples_received: 0,
            sample_offset: 0,
            audio: Vec::new(),
            finished: false,
        }
    }
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Diarizer {
    pub model: Nemotron,
    pub mel: NemotronMel,
    cfg: DiarConfig,
}

impl Diarizer {
    pub fn load(dir: impl AsRef<std::path::Path>, cfg: DiarConfig) -> Result<Self> {
        let w = any2md_mlx::Weights::load_dir(dir.as_ref())?;
        let mel = NemotronMel::from_weights(&w, &cfg)?;
        let model = Nemotron::load(dir, cfg.clone())?;
        Ok(Self { model, mel, cfg })
    }

    /// Consume one audio chunk; returns the central speaker probabilities
    /// (`[frames, n_spk]`) for this chunk. Segments are produced once over the
    /// whole clip in [`Diarizer::generate`], matching upstream.
    pub fn feed(
        &self,
        state: &mut StreamState,
        chunk: &[f32],
        final_chunk: bool,
    ) -> Result<Vec<Array>> {
        let proc = self
            .cfg
            .processor_config
            .as_ref()
            .context("nemotron config lacks processor_config")?;
        let modc = &self.cfg.modules_config;
        let hop = proc.hop_length as i64;
        let n_fft = proc.n_fft as i64;
        let factor = self.cfg.encoder_config.subsampling_factor;
        let central = modc.chunk_len * factor;
        let right = modc.chunk_right_context * factor;

        state.audio.extend_from_slice(chunk);
        state.samples_received += chunk.len() as i64;

        let mut outputs: Vec<Array> = Vec::new();
        // window start time for this feed (upstream passes it to _output)
        let offset = state.frames_processed as f32 * hop as f32 / proc.sampling_rate as f32;
        loop {
            let available = state.samples_received / hop - state.frames_processed as i64;
            let needed = ((state.frames_processed + central + right - 1) as i64) * hop + n_fft / 2;
            if available <= 0 || (!final_chunk && state.samples_received < needed) {
                break;
            }
            let n = central.min(available as i32);
            let window_frames = if final_chunk {
                let total_frames = state.samples_received / hop + 1;
                let pad_to = proc.pad_to as i64;
                let total_frames = total_frames + if pad_to > 0 { (-total_frames).rem_euclid(pad_to) } else { 0 };
                ((central + right) as i64)
                    .min(total_frames - state.frames_processed as i64) as i32
            } else {
                central + right
            };
            let features = self.mel.features(
                &state.audio,
                state.frames_processed,
                window_frames,
                state.sample_offset,
                state.samples_received,
            )?;
            let preds = self.streaming_step(state, &features, n)?;
            outputs.push(preds);

            // drop audio that can no longer be referenced (keep n_fft/2 + 1 overlap)
            let keep_from = (state.frames_processed as i64 * hop - n_fft / 2 - 1).max(0);
            let drop = (keep_from - state.sample_offset).max(0) as usize;
            if drop > 0 && drop <= state.audio.len() {
                state.audio.drain(0..drop);
                state.sample_offset = keep_from;
            }
        }
        state.finished = final_chunk;
        if final_chunk {
            state.audio.clear();
        }

        let _ = offset;
        Ok(outputs)
    }

    /// One 2720-frame central window; updates the AOSC cache and returns the
    /// high-resolution speaker probabilities for the new central frames.
    fn streaming_step(&self, state: &mut StreamState, features: &Array, n: i32) -> Result<Array> {
        let enc = &self.cfg.encoder_config;
        let modc = &self.cfg.modules_config;
        let factor = enc.subsampling_factor;
        let preds_per_frame = self.cfg.num_speakers as i32;

        // features are [1, mels, T] (checkpoint convention)
        let (stacked, _len) = self.model.pre_encode(features)?;
        let chunk = stacked; // [1, Tc, fc]
        let cache_len = state.spkcache.as_ref().map(|a| a.shape()[1]).unwrap_or(0);
        let fifo_len = state.fifo.as_ref().map(|a| a.shape()[1]).unwrap_or(0);

        let combined = match (&state.spkcache, &state.fifo) {
            (Some(c), Some(f)) => ops::concatenate(&[c, f, &chunk], 1)?,
            (Some(c), None) => ops::concatenate(&[c, &chunk], 1)?,
            (None, Some(f)) => ops::concatenate(&[f, &chunk], 1)?,
            (None, None) => chunk.clone(),
        };
        let total_len = cache_len + fifo_len + chunk.shape()[1];
        let high = self.model.encoder_forward(&combined, total_len)?; // [1, total, 512]
        let mut preds = self.model.speakers.forward(&high)?; // [1, total*8, n_spk]
        // mask predictions past the valid chunk length (upstream: high * valid)
        let valid_limit = total_len * factor;
        let np = preds.shape()[1];
        if valid_limit < np {
            let mask: Vec<f32> = (0..np).map(|i| if i < valid_limit { 1.0 } else { 0.0 }).collect();
            preds = preds.multiply(&Array::from_slice(&mask, &[1, np, 1]))?;
        }

        // low-resolution (per encoder frame) predictions for cache bookkeeping
        let low = preds
            .reshape(&[1, -1, factor, preds_per_frame])?
            .mean_axes(&[2], false)?;
        let start = (cache_len + fifo_len) * factor;
        let take = n.min(preds.shape()[1] - start).max(0);
        let central_preds = preds.take_axis(
            &Array::from_slice(
                &(start..start + take).collect::<Vec<_>>(),
                &[take],
            ),
            1,
        )?;

        // append the new chunk frames to the FIFO
        let n_enc = ((n + factor - 1) / factor).min(chunk.shape()[1]);
        let chunk_keep = chunk.take_axis(&Array::from_slice(&(0..n_enc).collect::<Vec<_>>(), &[n_enc]), 1)?;
        let fifo = match &state.fifo {
            Some(f) => ops::concatenate(&[f, &chunk_keep], 1)?,
            None => chunk_keep.clone(),
        };
        let start_enc = cache_len + fifo_len;
        let fifo_refreshed = if fifo_len > 0 {
            low.take_axis(
                &Array::from_slice(
                    &(cache_len..start_enc).collect::<Vec<_>>(),
                    &[fifo_len],
                ),
                1,
            )?
        } else {
            Array::zeros::<f32>(&[1, 0, preds_per_frame])?
        };
        let low_new = low.take_axis(
            &Array::from_slice(
                &(start_enc..start_enc + n_enc).collect::<Vec<_>>(),
                &[n_enc],
            ),
            1,
        )?;
        let fifo_preds = ops::concatenate(&[&fifo_refreshed, &low_new], 1)?;
        state.fifo = Some(fifo);
        state.fifo_preds = Some(fifo_preds);

        // overflow -> speaker cache (compressed with AOSC when over the limit)
        if state.fifo.as_ref().map(|f| f.shape()[1]).unwrap_or(0) > modc.fifo_len {
            let fifo = state.fifo.take().expect("fifo");
            let fifo_preds = state.fifo_preds.take().expect("fifo_preds");
            let cur = fifo.shape()[1];
            let pop = cur.min((modc.spkcache_update_period).max(cur - modc.fifo_len));
            let popped = fifo.take_axis(&Array::from_slice(&(0..pop).collect::<Vec<_>>(), &[pop]), 1)?;
            let popped_preds =
                fifo_preds.take_axis(&Array::from_slice(&(0..pop).collect::<Vec<_>>(), &[pop]), 1)?;

            let mut cache = match &state.spkcache {
                Some(c) => ops::concatenate(&[c, &popped], 1)?,
                None => popped,
            };
            let mut cache_preds = match &state.spkcache_preds {
                Some(p) => ops::concatenate(&[p, &popped_preds], 1)?,
                None => popped_preds,
            };
            if cache.shape()[1] > modc.spkcache_len {
                let (c, p) = aosc_compress(
                    &cache,
                    &cache_preds,
                    &self.model.speakers.learnable_sil_emb.reshape(&[1, -1])?,
                    modc,
                )?;
                cache = c;
                cache_preds = p;
                state.compressed = true;
            }
            state.spkcache = Some(cache);
            state.spkcache_preds = Some(cache_preds);
            state.fifo = Some(fifo.take_axis(
                &Array::from_slice(&(pop..cur).collect::<Vec<_>>(), &[cur - pop]),
                1,
            )?);
            state.fifo_preds = Some(fifo_preds.take_axis(
                &Array::from_slice(&(pop..cur).collect::<Vec<_>>(), &[cur - pop]),
                1,
            )?);
        }

        state.frames_processed += n;
        Ok(central_preds.squeeze_axes(&[0])?)
    }

    /// Diarize a whole recording (16 kHz mono).
    pub fn generate(&self, audio: &Array) -> Result<Vec<Segment>> {
        let n = audio.shape()[0] as usize;
        let proc = self
            .cfg
            .processor_config
            .as_ref()
            .context("nemotron config lacks processor_config")?;
        let step = (self.cfg.modules_config.chunk_len
            * self.cfg.encoder_config.subsampling_factor
            * proc.hop_length as i32) as usize;

        let samples: Vec<f32> = audio.as_slice::<f32>().to_vec();
        let mut state = StreamState::new();
        let mut chunks: Vec<Array> = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + step).min(n);
            chunks.extend(self.feed(&mut state, &samples[start..end], false)?);
            start = end;
        }
        chunks.extend(self.feed(&mut state, &[], true)?);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let probs = ops::concatenate(&chunks.iter().collect::<Vec<_>>(), 0)?;
        let stride = proc.hop_length as f32 / proc.sampling_rate as f32; // output_subsampling_factor = 1
        let mut segments = preds_to_segments(&to_rows(&probs)?, stride, 0.5, 0.0, 0.0);
        let end_limit = state.frames_processed as f32 * stride;
        for s in &mut segments {
            s.end = s.end.min(end_limit);
        }
        Ok(segments)
    }
}

fn to_rows(a: &Array) -> Result<Vec<Vec<f32>>> {
    // predictions may be bf16 (checkpoint dtype); CPU post-processing is f32
    let a = a.as_dtype(any2md_mlx::Dtype::Float32)?;
    a.eval()?;
    let shape = a.shape().to_vec();
    let (frames, spk) = (shape[0] as usize, shape[1] as usize);
    let data = a.as_slice::<f32>();
    Ok((0..frames).map(|i| data[i * spk..(i + 1) * spk].to_vec()).collect())
}

/// AOSC compression, mirroring `Sortformer._compress_spkcache_aosc`.
///
/// Kept in plain Rust on the CPU: the cache is at most a few hundred frames, and
/// this avoids depending on argpartition/take_along_axis semantics.
pub fn aosc_compress(
    embs: &Array,
    preds: &Array,
    sil_emb: &Array,
    cfg: &crate::nemotron::ModulesCfg,
) -> Result<(Array, Array)> {
    let embs = &embs.as_dtype(any2md_mlx::Dtype::Float32)?;
    let preds = &preds.as_dtype(any2md_mlx::Dtype::Float32)?;
    let sil_emb = &sil_emb.as_dtype(any2md_mlx::Dtype::Float32)?;
    embs.eval()?;
    preds.eval()?;
    sil_emb.eval()?;
    let esc = embs.shape().to_vec();
    let psc = preds.shape().to_vec();
    let (n_frames, emb_dim) = (esc[1], esc[2]);
    let n_spk = psc[2];
    let e = embs.as_slice::<f32>();
    let p = preds.as_slice::<f32>();
    let sil = sil_emb.as_slice::<f32>();

    let n_spk_us = n_spk as usize;
    let cache_len = cfg.spkcache_len as usize;
    let sil_per_spk = 1usize; // spkcache_sil_frames_per_spk for this checkpoint
    let per_spk = cache_len / n_spk_us - sil_per_spk;
    let strong = (per_spk as f32 * 0.75).floor() as usize;
    let weak = (per_spk as f32 * 1.5).floor() as usize;
    let min_pos = (per_spk as f32 * 0.5).floor() as usize;
    let threshold = 0.25f32;
    let boost_latest = 0.05f32;
    let n = n_frames as usize;

    // 1. log-likelihood-ratio scores
    let mut scores = vec![0f32; n * n_spk_us];
    for i in 0..n {
        let mut sum_log_1p = 0f32;
        for k in 0..n_spk_us {
            let v = p[i * n_spk_us + k];
            sum_log_1p += (1.0 - v).max(threshold).ln();
        }
        for k in 0..n_spk_us {
            let v = p[i * n_spk_us + k];
            scores[i * n_spk_us + k] =
                v.max(threshold).ln() - (1.0 - v).max(threshold).ln() + sum_log_1p
                    - 0.5f32.ln();
        }
    }
    // 2. drop non-speech, then overlapped frames when a speaker has enough clean frames
    let mut pos_count = vec![0usize; n_spk_us];
    for i in 0..n {
        for k in 0..n_spk_us {
            if p[i * n_spk_us + k] > 0.5 && scores[i * n_spk_us + k] > 0.0 {
                pos_count[k] += 1;
            }
        }
    }
    for i in 0..n {
        for k in 0..n_spk_us {
            let speech = p[i * n_spk_us + k] > 0.5;
            let idx = i * n_spk_us + k;
            if !speech {
                scores[idx] = f32::NEG_INFINITY;
            } else if scores[idx] <= 0.0 && pos_count[k] >= min_pos {
                scores[idx] = f32::NEG_INFINITY;
            }
        }
    }
    // 3. boost the newest frames
    if boost_latest > 0.0 && n > cache_len {
        for i in cache_len..n {
            for k in 0..n_spk_us {
                scores[i * n_spk_us + k] += boost_latest;
            }
        }
    }
    // 4/5. strong then weak top-k boost per speaker
    for (k, n_boost) in [(0usize, strong), (1usize, weak)] {
        if n_boost == 0 {
            continue;
        }
        let scale = if k == 0 { 2.0f32 } else { 1.0 };
        let boost_val = -scale * 0.5f32.ln();
        for spk in 0..n_spk_us {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|a, b| {
                scores[*b * n_spk_us + spk]
                    .partial_cmp(&scores[*a * n_spk_us + spk])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &i in order.iter().take(n_boost) {
                let idx = i * n_spk_us + spk;
                if scores[idx] > f32::NEG_INFINITY {
                    scores[idx] += boost_val;
                }
            }
        }
    }
    // 6. append silence padding with +inf
    let n_total = n + sil_per_spk;
    let mut padded = vec![f32::NEG_INFINITY; n_total * n_spk_us];
    padded[..n * n_spk_us].copy_from_slice(&scores);
    for j in 0..sil_per_spk {
        for k in 0..n_spk_us {
            padded[(n + j) * n_spk_us + k] = f32::INFINITY;
        }
    }
    // 7. top-k over the flattened (speaker, frame) space
    let mut flat: Vec<(usize, f32)> = (0..n_total * n_spk_us)
        .map(|i| (i, padded[i]))
        .collect();
    flat.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = cache_len.min(flat.len());
    let max_index = 99_999usize;
    let mut indices: Vec<usize> = flat[..k]
        .iter()
        .map(|(i, v)| if *v > f32::NEG_INFINITY { *i % n_total } else { max_index })
        .collect();
    indices.sort_unstable();
    let mut is_disabled: Vec<bool> = indices.iter().map(|i| *i == max_index || *i >= n).collect();
    for (idx, d) in indices.iter_mut().zip(is_disabled.iter()) {
        if *d {
            *idx = 0;
        }
    }
    // 8. gather embeddings/preds, replacing disabled slots with the silence embedding
    let mut out_e = vec![0f32; k * emb_dim as usize];
    let mut out_p = vec![0f32; k * n_spk_us];
    for (slot, i) in indices.iter().enumerate() {
        if is_disabled[slot] {
            out_e[slot * emb_dim as usize..(slot + 1) * emb_dim as usize]
                .copy_from_slice(&sil[..emb_dim as usize]);
        } else {
            out_e[slot * emb_dim as usize..(slot + 1) * emb_dim as usize]
                .copy_from_slice(&e[i * emb_dim as usize..(i + 1) * emb_dim as usize]);
            out_p[slot * n_spk_us..(slot + 1) * n_spk_us]
                .copy_from_slice(&p[i * n_spk_us..(i + 1) * n_spk_us]);
        }
    }
    Ok((
        Array::from_slice(&out_e, &[1, k as i32, emb_dim]),
        Array::from_slice(&out_p, &[1, k as i32, n_spk]),
    ))
}
