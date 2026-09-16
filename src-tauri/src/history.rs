//! Local session history. Each finished recording's summary (stats + sub-scores,
//! built on the frontend) is appended to a single JSON file in the app data dir,
//! so progress-over-time, personal bests and streaks survive restarts. Local-only
//! by design — nothing leaves the machine.
//!
//! ponytail: whole-file read-modify-write of one JSON array. Sessions are small
//! and infrequent (one per recording), so this is fine; move to append-only JSONL
//! or SQLite only if a user ever accumulates enough history to make the rewrite
//! cost show up.

use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

fn sessions_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join("sessions.json"))
}

fn load(app: &AppHandle) -> Result<Vec<Value>, String> {
    let path = sessions_path(app)?;
    match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| e.to_string()),
        // Missing file = no history yet.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.to_string()),
    }
}

/// Appends one session summary (an already-serialized JSON object) and returns
/// the full updated list, so the frontend can refresh its history view in one
/// round trip.
#[tauri::command]
pub fn save_session(app: AppHandle, session: String) -> Result<String, String> {
    let value: Value = serde_json::from_str(&session).map_err(|e| e.to_string())?;
    let mut sessions = load(&app)?;
    sessions.push(value);
    let text = serde_json::to_string(&sessions).map_err(|e| e.to_string())?;
    fs::write(sessions_path(&app)?, &text).map_err(|e| e.to_string())?;
    Ok(text)
}

/// Returns all saved sessions as a JSON array string (`[]` when none).
#[tauri::command]
pub fn list_sessions(app: AppHandle) -> Result<String, String> {
    let sessions = load(&app)?;
    serde_json::to_string(&sessions).map_err(|e| e.to_string())
}
