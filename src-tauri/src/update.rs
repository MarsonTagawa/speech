//! Settings → Updates: self-update from GitHub releases with tauri-plugin-updater.
//!
//! The release workflow (.github/workflows/release.yml) publishes signed bundles
//! and a `latest.json` listing them; the endpoint and the public key that checks
//! the signatures are in tauri.conf.json. Only checks when asked: nothing goes
//! out at launch.
use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tauri_plugin_updater::{Error, UpdaterExt};

#[derive(Serialize)]
pub struct Available {
    version: String,
    notes: Option<String>,
}

#[derive(Clone, Serialize)]
struct Progress {
    done: u64,
    total: u64,
}

fn message(e: Error) -> String {
    match e {
        Error::TargetsNotFound(_) | Error::UnsupportedOs | Error::UnsupportedArch => {
            "no prebuilt update for this platform or package type; pull and rebuild instead".into()
        }
        // GitHub answered, but the latest release has no latest.json (made
        // before release.yml, or by hand).
        Error::ReleaseNotFound => "the latest release has no update package; download it from GitHub instead".into(),
        Error::Reqwest(_) | Error::Network(_) => "couldn't reach GitHub; check your connection".into(),
        e => e.to_string(),
    }
}

async fn latest(app: &AppHandle) -> Result<Option<tauri_plugin_updater::Update>, String> {
    app.updater().map_err(message)?.check().await.map_err(message)
}

/// The newer release, if there is one.
#[tauri::command]
pub async fn update_check(app: AppHandle) -> Result<Option<Available>, String> {
    Ok(latest(&app).await?.map(|u| Available { version: u.version, notes: u.body }))
}

/// Downloads the newer release, emitting `update_download` progress and then
/// `update_installing`, installs it over this one and restarts. On Windows the
/// NSIS installer takes over and relaunches the app itself.
#[tauri::command]
pub async fn update_install(app: AppHandle) -> Result<(), String> {
    // A dev build's executable sits in target/, not in a bundle the updater can replace.
    if cfg!(debug_assertions) {
        return Err("this is a dev build; pull and rebuild instead".into());
    }
    let update = latest(&app).await?.ok_or("already up to date")?;
    // The updater backs the AppImage up by renaming it into $TMPDIR, checking
    // only st_dev. NixOS's appimage-run (bwrap) binds /tmp and /home as
    // separate mounts of one filesystem, so the check passes and rename fails
    // with EXDEV. Back up next to the AppImage instead; we restart right after.
    #[cfg(target_os = "linux")]
    if let Some(dir) = std::env::var_os("APPIMAGE").as_ref().and_then(|p| std::path::Path::new(p).parent()) {
        std::env::set_var("TMPDIR", dir);
    }
    let (mut done, mut emitted) = (0u64, 0u64);
    update
        .download_and_install(
            |chunk, total| {
                done += chunk as u64;
                if done - emitted >= 1 << 20 {
                    emitted = done;
                    let _ = app.emit("update_download", Progress { done, total: total.unwrap_or(0) });
                }
            },
            || {
                let _ = app.emit("update_installing", ());
            },
        )
        .await
        .map_err(message)?;
    app.restart()
}
