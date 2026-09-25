//! Whisper model catalog: which models the settings page offers, where each
//! lives on disk, and downloading the ones that aren't bundled.
//!
//! Bundled models ship in `resources/`; downloaded ones go to
//! `<app data>/models/`. Files are whisper.cpp's quantized ggml builds.
use serde::Serialize;
use std::io::{Read, Write};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager};

/// (id, file, approximate size in MB).
const MODELS: &[(&str, &str, u32)] = &[
    ("tiny.en", "ggml-tiny.en-q5_1.bin", 32),
    ("base.en", "ggml-base.en-q5_1.bin", 60),
    ("small.en", "ggml-small.en-q5_1.bin", 190),
    ("medium.en", "ggml-medium.en-q5_0.bin", 539),
    ("large-v3-turbo", "ggml-large-v3-turbo-q5_0.bin", 574),
];
const BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

fn file(id: &str) -> Result<&'static str, String> {
    MODELS
        .iter()
        .find(|m| m.0 == id)
        .map(|m| m.1)
        .ok_or_else(|| format!("unknown model {id}"))
}

fn bundled(app: &AppHandle, file: &str) -> Option<PathBuf> {
    app.path()
        .resolve(format!("resources/{file}"), tauri::path::BaseDirectory::Resource)
        .ok()
        .filter(|p| p.exists())
}

fn downloaded(app: &AppHandle, file: &str) -> Result<PathBuf, String> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("models").join(file))
}

/// On-disk path of an installed model, bundled first.
pub fn path(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    let f = file(id)?;
    if let Some(p) = bundled(app, f) {
        return Ok(p);
    }
    let p = downloaded(app, f)?;
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("model {id} is not downloaded"))
    }
}

#[derive(Serialize)]
pub struct ModelInfo {
    id: &'static str,
    size_mb: u32,
    installed: bool,
    bundled: bool,
}

#[tauri::command]
pub fn list_models(app: AppHandle) -> Vec<ModelInfo> {
    MODELS
        .iter()
        .map(|&(id, f, size_mb)| {
            let bundled = bundled(&app, f).is_some();
            let installed = bundled || downloaded(&app, f).map(|p| p.exists()).unwrap_or(false);
            ModelInfo { id, size_mb, installed, bundled }
        })
        .collect()
}

#[derive(Serialize, Clone)]
struct Progress {
    id: String,
    done: u64,
    total: u64,
}

/// Downloads a model, emitting `model_download` progress events. Writes to a
/// `.part` file and renames on success, so a failed or partial download never
/// shows up as installed.
#[tauri::command]
pub async fn download_model(app: AppHandle, id: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dest = downloaded(&app, file(&id)?)?;
        std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
        let part = dest.with_extension("part");
        let result = fetch(&app, &id, &part).and_then(|_| std::fs::rename(&part, &dest).map_err(|e| e.to_string()));
        if result.is_err() {
            let _ = std::fs::remove_file(&part);
        }
        result
    })
    .await
    .map_err(|e| e.to_string())?
}

fn fetch(app: &AppHandle, id: &str, part: &PathBuf) -> Result<(), String> {
    // ureq defaults to rustls; only native-tls is compiled in (see Cargo.toml).
    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    let agent: ureq::Agent = ureq::Agent::config_builder().tls_config(tls).build().into();
    let mut resp = agent.get(&format!("{BASE_URL}{}", file(id)?)).call().map_err(|e| e.to_string())?;
    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok()?.parse().ok())
        .unwrap_or(0);
    let mut reader = resp.body_mut().as_reader();
    let mut out = std::fs::File::create(part).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 1 << 16];
    let (mut done, mut emitted) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        done += n as u64;
        // ~1MB granularity keeps the event rate sane on a fast connection.
        if done - emitted >= 1 << 20 {
            emitted = done;
            let _ = app.emit("model_download", Progress { id: id.into(), done, total });
        }
    }
    if total > 0 && done != total {
        return Err(format!("download of {id} incomplete ({done} of {total} bytes)"));
    }
    out.sync_all().map_err(|e| e.to_string())?;
    let _ = app.emit("model_download", Progress { id: id.into(), done, total: done });
    Ok(())
}

/// Deletes a downloaded model. Bundled models can't be deleted.
#[tauri::command]
pub fn delete_model(app: AppHandle, id: String) -> Result<(), String> {
    std::fs::remove_file(downloaded(&app, file(&id)?)?).map_err(|e| e.to_string())
}
