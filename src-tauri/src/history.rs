//! Local session history. Each finished recording's summary (stats + sub-scores,
//! built on the frontend) is appended to a single JSON file in the app data dir,
//! so progress-over-time, personal bests and streaks survive restarts. Local-only
//! by design — nothing leaves the machine.
//!
//! Each session also gets a detail file (`sessions/<ts>.json`: per-line timing,
//! analysis and transcript markup) so History can redraw its full report. Those
//! are kept only for the newest `KEEP_RECENT` sessions and starred ones; older
//! sessions fall back to what their summary holds (incl. a per-second pace series).
//!
//! ponytail: whole-file read-modify-write of one JSON array. Sessions are small
//! and infrequent (one per recording), so this is fine; move to append-only JSONL
//! or SQLite only if a user ever accumulates enough history to make the rewrite
//! cost show up.

use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

fn sessions_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join("sessions.json"))
}

/// Sessions whose detail file survives pruning, besides starred ones.
const KEEP_RECENT: usize = 5;

fn detail_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("sessions");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Sessions (by ts) whose detail file is kept: the newest `KEEP_RECENT`, plus starred.
fn detail_keep(sessions: &[Value]) -> HashSet<i64> {
    let mut ts: Vec<i64> = sessions.iter().filter_map(|s| s["ts"].as_i64()).collect();
    ts.sort_unstable();
    let mut keep: HashSet<i64> = ts.iter().rev().take(KEEP_RECENT).copied().collect();
    keep.extend(
        sessions
            .iter()
            .filter(|s| s["saved"].as_bool() == Some(true))
            .filter_map(|s| s["ts"].as_i64()),
    );
    keep
}

/// Deletes detail files for sessions outside `detail_keep`.
fn prune_details(app: &AppHandle, sessions: &[Value]) -> Result<(), String> {
    let keep = detail_keep(sessions);
    for entry in fs::read_dir(detail_dir(app)?).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let id = path.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<i64>().ok());
        if id.is_some_and(|id| !keep.contains(&id)) {
            fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
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

/// Appends one session summary (an already-serialized JSON object), writes its
/// detail file, and returns the full updated list, so the frontend can refresh
/// its history view in one round trip.
#[tauri::command]
pub fn save_session(app: AppHandle, session: String, detail: String) -> Result<String, String> {
    let value: Value = serde_json::from_str(&session).map_err(|e| e.to_string())?;
    let ts = value["ts"].as_i64();
    let mut sessions = load(&app)?;
    sessions.push(value);
    let text = store(&app, &sessions)?;
    // The summary is already safe on disk; a failed detail write only costs
    // the full report for this session.
    let details = ts
        .ok_or_else(|| "session has no ts".to_string())
        .and_then(|ts| fs::write(detail_dir(&app)?.join(format!("{ts}.json")), detail).map_err(|e| e.to_string()))
        .and_then(|_| prune_details(&app, &sessions));
    if let Err(e) = details {
        eprintln!("session detail: {e}");
    }
    Ok(text)
}

/// The detail file of the session started at `ts`, or None once pruned (or for
/// sessions saved before details existed).
#[tauri::command]
pub fn load_session_detail(app: AppHandle, ts: i64) -> Result<Option<String>, String> {
    match fs::read_to_string(detail_dir(&app)?.join(format!("{ts}.json"))) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// Writes the list back and returns it serialized, for the frontend to adopt.
fn store(app: &AppHandle, sessions: &[Value]) -> Result<String, String> {
    let text = serde_json::to_string(sessions).map_err(|e| e.to_string())?;
    fs::write(sessions_path(app)?, &text).map_err(|e| e.to_string())?;
    Ok(text)
}

/// Stars or unstars the session started at `ts` (its id) and returns the full
/// updated list. Unstarring an older session drops its detail file.
#[tauri::command]
pub fn set_session_saved(app: AppHandle, ts: i64, saved: bool) -> Result<String, String> {
    let mut sessions = load(&app)?;
    for s in sessions.iter_mut().filter(|s| s["ts"].as_i64() == Some(ts)) {
        s["saved"] = Value::Bool(saved);
    }
    let text = store(&app, &sessions)?;
    prune_details(&app, &sessions)?;
    Ok(text)
}

/// Deletes all saved sessions and their detail files.
#[tauri::command]
pub fn clear_sessions(app: AppHandle) -> Result<(), String> {
    fs::remove_dir_all(detail_dir(&app)?).map_err(|e| e.to_string())?;
    match fs::remove_file(sessions_path(&app)?) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
        _ => Ok(()),
    }
}

/// Returns all saved sessions as a JSON array string (`[]` when none).
#[tauri::command]
pub fn list_sessions(app: AppHandle) -> Result<String, String> {
    let sessions = load(&app)?;
    serde_json::to_string(&sessions).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keeps_newest_five_and_starred() {
        let sessions: Vec<Value> = (1..=8)
            .map(|ts| json!({ "ts": ts, "saved": ts == 2 }))
            .collect();
        let mut keep: Vec<i64> = detail_keep(&sessions).into_iter().collect();
        keep.sort_unstable();
        assert_eq!(keep, vec![2, 4, 5, 6, 7, 8]);
    }
}
