//! Native model implementations.
//!
//! Started with ReaderLM (Qwen2); Parakeet/Qwen3-ASR/Sortformer/Nemotron-3 and
//! Whisper LID follow as the port progresses.

pub mod qwen2;

pub use qwen2::{Generator, KvCache, Qwen2, Qwen2Config};
