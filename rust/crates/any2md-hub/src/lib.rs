//! Resolve Hugging Face model files from the shared cache, downloading on miss.
//!
//! The cache layout is the one `huggingface_hub` (Python) uses:
//!
//! ```text
//! $HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/<file>
//! ```
//!
//! so anything already pulled by the Python any2md (or by `any2md deps`) is
//! reused as-is. Downloads go to `blobs/` first and are then copied into the
//! snapshot directory; that is enough for our read-only use.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

/// Root of the HF cache (`$HF_HOME/hub`, else `~/.cache/huggingface/hub`).
pub fn hub_dir() -> Result<PathBuf> {
    if let Ok(home) = std::env::var("HF_HOME") {
        return Ok(Path::new(&home).join("hub"));
    }
    let home = std::env::var("HOME").context("HOME is not set")?;
    Ok(Path::new(&home).join(".cache/huggingface/hub"))
}

fn repo_dir(repo_id: &str) -> Result<PathBuf> {
    let name = repo_id.replace('/', "--");
    Ok(hub_dir()?.join(format!("models--{name}")))
}

/// Locate an already-downloaded file, if any snapshot has it.
pub fn find_cached(repo_id: &str, file: &str) -> Option<PathBuf> {
    let snaps = repo_dir(repo_id).ok()?.join("snapshots");
    let mut revisions: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    revisions.sort();
    // Latest revision last alphabetically is not meaningful; prefer any hit.
    revisions.reverse();
    for rev in revisions {
        let candidate = rev.join(file);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Path to `file` inside `repo_id`, downloading it into the HF cache on miss.
pub fn get_file(repo_id: &str, file: &str) -> Result<PathBuf> {
    if let Some(hit) = find_cached(repo_id, file) {
        return Ok(hit);
    }

    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{file}");
    tracing::info!(%repo_id, %file, "downloading from Hugging Face");
    let mut response = ureq::get(&url)
        .header("User-Agent", "any2md-rs")
        .call()
        .with_context(|| format!("GET {url}"))?;

    let repo = repo_dir(repo_id)?;
    let blobs = repo.join("blobs");
    let snapshot = repo.join("snapshots").join("main");
    std::fs::create_dir_all(&blobs)?;
    std::fs::create_dir_all(&snapshot)?;

    // Read fully then write once: model files are large but this keeps the
    // cache consistent if the process is interrupted.
    let mut body = Vec::new();
    response
        .body_mut()
        .as_reader()
        .read_to_end(&mut body)
        .with_context(|| format!("read body of {url}"))?;

    let blob = blobs.join(file.replace('/', "_"));
    std::fs::write(&blob, &body)?;
    let target = snapshot.join(file);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&blob, &target)?;
    Ok(target)
}

/// Shorthand for the common "single-file JSON config" case.
pub fn read_json<T: serde::de::DeserializeOwned>(repo_id: &str, file: &str) -> Result<T> {
    let path = get_file(repo_id, file)?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// Fail fast with a helpful message when the cache has no model at all.
pub fn require_snapshot(repo_id: &str) -> Result<PathBuf> {
    let snaps = repo_dir(repo_id)?.join("snapshots");
    let any = std::fs::read_dir(&snaps)
        .ok()
        .and_then(|it| {
            it.filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir())
        })
        .ok_or_else(|| {
            anyhow!(
                "model {repo_id} is not in the HF cache; run `python scripts/download_models.py` \
                 or let any2md-rs download it"
            )
        })?;
    Ok(any)
}

/// Like [`get_file`], but tries `fallbacks` (other repos) when the file is
/// missing from `repo_id`. MLX conversions sometimes omit `tokenizer.json`.
pub fn get_file_with_fallback(repo_id: &str, file: &str, fallbacks: &[&str]) -> Result<PathBuf> {
    match get_file(repo_id, file) {
        Ok(path) => Ok(path),
        Err(primary) => {
            for candidate in fallbacks {
                if *candidate == repo_id {
                    continue;
                }
                if let Ok(path) = get_file(candidate, file) {
                    tracing::info!(repo = candidate, %file, "using fallback repo for {file}");
                    return Ok(path);
                }
            }
            Err(primary)
        }
    }
}
