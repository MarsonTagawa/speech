//! Screen replay: records one display (plus the camera when it's on) during a
//! session as change-driven JPEG frames in `sessions/<ts>.frames`, replayed in
//! the report in sync with the session audio. See
//! docs/superpowers/specs/2026-10-20-screen-replay-design.md.
//!
//! While the toggle is on, a capture thread reads the display (screen_prepare
//! opens it once, so later sessions don't re-prompt). During a session it
//! samples every SAMPLE and saves frames that changed. The camera thread hands
//! frames to a writer thread through a small queue, so it never waits on disk.
mod frames;
pub mod stream;

use frames::{Rec, Recorder, CAMERA, SCREEN};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tauri::{ipc::Response, AppHandle, Emitter, Manager};

const SAMPLE: Duration = Duration::from_millis(250);
const CAMERA_QUEUE: usize = 4;
const CAMERA_EVERY: u32 = 3; // of the camera's ~30 fps

struct Session {
    rec: Recorder,
    clock: Option<Instant>, // the session's first audio chunk; None until then
}

impl Session {
    fn ms(&self) -> u32 {
        self.clock.map_or(0, |c| c.elapsed().as_millis() as u32)
    }
}

static SESSION: Mutex<Option<Session>> = Mutex::new(None);
/// Separate from SESSION so the camera thread never waits on a frame being encoded.
static CAMERA_TX: Mutex<Option<SyncSender<Option<Vec<u8>>>>> = Mutex::new(None);
/// The replay index of the most recently viewed session.
static INDEX: Mutex<Option<(i64, Vec<Rec>)>> = Mutex::new(None);

struct Capture {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

#[derive(Default)]
pub struct ScreenState(Mutex<Option<Capture>>);

impl ScreenState {
    pub fn release(&self) {
        if let Some(c) = self.0.lock().unwrap().take() {
            c.stop.store(true, Relaxed); // the thread exits within one `next` timeout
        }
    }
    fn live(&self) -> bool {
        self.0.lock().unwrap().as_ref().is_some_and(|c| !c.thread.is_finished())
    }
}

// --- Session hooks (audio.rs, vision) -------------------------------------------

/// Session `ts` starts (audio::start_recording). Records only while a capture
/// is live; a failure is reported to the UI, never to the recording.
pub fn begin(app: &AppHandle, ts: i64) {
    let live = app.state::<ScreenState>().live();
    if let Err(e) = crate::history::detail_dir(app).and_then(|d| start_session(&d, ts, live)) {
        let _ = app.emit("screen_ended", format!("couldn't save the video — {e}"));
    }
}

/// Opens `<dir>/<ts>.frames` and the camera writer when `live`. Returns whether it did.
fn start_session(dir: &Path, ts: i64, live: bool) -> Result<bool, String> {
    if !live {
        return Ok(false);
    }
    let rec = Recorder::create(&dir.join(format!("{ts}.frames"))).map_err(|e| e.to_string())?;
    *SESSION.lock().unwrap() = Some(Session { rec, clock: None });
    let (tx, rx) = mpsc::sync_channel::<Option<Vec<u8>>>(CAMERA_QUEUE);
    thread::spawn(move || {
        for frame in rx {
            if let Some(s) = SESSION.lock().unwrap().as_mut() {
                let ms = s.ms();
                if let Err(e) = s.rec.camera(ms, frame.as_deref()) {
                    eprintln!("[screen] camera frame: {e}");
                }
            }
        }
    });
    *CAMERA_TX.lock().unwrap() = Some(tx);
    Ok(true)
}

/// The session's first audio chunk arrived: ms 0 of its `.pcm`. Idempotent.
pub fn mark_clock() {
    if let Some(s) = SESSION.lock().unwrap().as_mut() {
        s.clock.get_or_insert_with(Instant::now);
    }
}

/// Session stops (audio::stop_recording).
pub fn end() {
    CAMERA_TX.lock().unwrap().take(); // the writer thread drains and exits
    if let Some(s) = SESSION.lock().unwrap().take() {
        if let Err(e) = s.rec.finish() {
            eprintln!("[screen] finishing the video: {e}");
        }
    }
}

/// Every camera frame (vision::capture); keeps every CAMERA_EVERY-th during a session.
pub fn camera_frame(n: u32, rgb: &[u8]) {
    if !n.is_multiple_of(CAMERA_EVERY) {
        return;
    }
    if let Some(tx) = CAMERA_TX.lock().unwrap().as_ref() {
        let _ = tx.try_send(Some(rgb.to_vec())); // full → drop the frame
    }
}

/// The camera turned off (vision::stop_running): the replay hides the bubble from here.
pub fn camera_off() {
    if let Some(tx) = CAMERA_TX.lock().unwrap().as_ref() {
        let _ = tx.send(None); // must not be dropped like a frame
    }
}

fn screen_sample(w: usize, h: usize, rgb: &[u8]) {
    let mut g = SESSION.lock().unwrap();
    if let Some(s) = g.as_mut() {
        let ms = s.ms();
        if let Err(e) = s.rec.screen(ms, w, h, rgb) {
            eprintln!("[screen] writing a frame failed, video stops here: {e}");
            *g = None; // keep recording audio; drop the replay's tail
        }
    }
}

fn stream_ended() {
    if let Some(s) = SESSION.lock().unwrap().as_mut() {
        let ms = s.ms();
        let _ = s.rec.screen_ended(ms);
    }
}

fn run(app: AppHandle, stream: stream::Stream, stop: Arc<AtomicBool>) {
    let mut last: Option<Instant> = None;
    while !stop.load(Relaxed) {
        match stream.next(Duration::from_millis(500)) {
            Ok(Some(f)) if last.is_none_or(|t| t.elapsed() >= SAMPLE) => {
                last = Some(Instant::now());
                screen_sample(f.w, f.h, &f.rgb);
            }
            Ok(_) => {}
            Err(e) => {
                stream_ended();
                let _ = app.emit("screen_ended", e);
                return;
            }
        }
    }
}

// --- Commands -------------------------------------------------------------------

/// Toggle on: asks for the display/permission and starts reading it. async so
/// the picker doesn't block the main thread.
#[tauri::command]
pub async fn screen_prepare(app: AppHandle, display: Option<String>) -> Result<(), String> {
    app.state::<ScreenState>().release();
    let stream = tauri::async_runtime::spawn_blocking(move || stream::open(display)).await.map_err(|e| e.to_string())??;
    let stop = Arc::new(AtomicBool::new(false));
    let thread = {
        let (app, stop) = (app.clone(), stop.clone());
        thread::spawn(move || run(app, stream, stop))
    };
    *app.state::<ScreenState>().0.lock().unwrap() = Some(Capture { stop, thread });
    Ok(())
}

#[tauri::command]
pub fn screen_release(app: AppHandle) {
    app.state::<ScreenState>().release();
}

#[tauri::command]
pub fn screen_displays() -> Vec<stream::Display> {
    stream::displays()
}

fn with_index<T>(dir: &Path, ts: i64, f: impl FnOnce(&Path, &[Rec]) -> T) -> Result<T, String> {
    let path = dir.join(format!("{ts}.frames"));
    let missing = || "Video isn't kept for this session".to_string();
    if !path.exists() {
        return Err(missing());
    }
    let mut g = INDEX.lock().unwrap();
    if g.as_ref().is_none_or(|(t, _)| *t != ts) {
        *g = Some((ts, frames::index(&path).map_err(|_| missing())?));
    }
    Ok(f(&path, &g.as_ref().unwrap().1))
}

fn frame_at(dir: &Path, ts: i64, kind: u8, ms: u32) -> Result<Vec<u8>, String> {
    with_index(dir, ts, |path, idx| frames::latest(idx, kind, ms).map_or(Ok(vec![]), |r| frames::read(path, r)))?
        .map_err(|_| "Video isn't kept for this session".to_string())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
    screen: bool,
    camera: bool,
    ends_ms: Option<u32>,
}

fn summary(dir: &Path, ts: i64) -> Option<Summary> {
    with_index(dir, ts, |_, idx| Summary {
        screen: idx.iter().any(|r| r.kind == SCREEN && r.len > 0),
        camera: idx.iter().any(|r| r.kind == CAMERA && r.len > 0),
        ends_ms: idx.iter().find(|r| r.kind == SCREEN && r.len == 0).map(|r| r.ms),
    })
    .ok()
}

/// The JPEG of session `ts`'s last `kind` frame at or before `ms`; empty when
/// there's none or it's an "off" marker.
#[tauri::command]
pub fn session_frame(app: AppHandle, ts: i64, kind: u8, ms: u32) -> Result<Response, String> {
    Ok(Response::new(frame_at(&crate::history::detail_dir(&app)?, ts, kind, ms)?))
}

/// What session `ts`'s replay holds; None if it has no `.frames`.
#[tauri::command]
pub fn session_has_frames(app: AppHandle, ts: i64) -> Option<Summary> {
    summary(&crate::history::detail_dir(&app).ok()?, ts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // SESSION is process-wide; tests that touch it take turns.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("speech-screen-{}-{name}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn one_record(path: &std::path::Path, body: &[u8]) {
        let mut out = Vec::new();
        frames::write_record(&mut out, frames::SCREEN, 0, body).unwrap();
        fs::write(path, out).unwrap();
    }

    // Review focus 5: switching replays never shows the other session's frames,
    // and a pruned file reports missing instead of serving the stale index.
    #[test]
    fn index_cache_follows_the_session_and_notices_pruning() {
        let d = dir("cache");
        one_record(&d.join("1.frames"), b"one");
        one_record(&d.join("2.frames"), b"two");
        assert_eq!(frame_at(&d, 1, frames::SCREEN, 0).unwrap(), b"one");
        assert_eq!(frame_at(&d, 2, frames::SCREEN, 0).unwrap(), b"two");
        assert_eq!(frame_at(&d, 1, frames::SCREEN, 0).unwrap(), b"one");
        fs::remove_file(d.join("1.frames")).unwrap();
        assert_eq!(frame_at(&d, 1, frames::SCREEN, 0).unwrap_err(), "Video isn't kept for this session");
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn summary_reports_kinds_and_the_cut() {
        let d = dir("summary");
        let mut out = Vec::new();
        frames::write_record(&mut out, frames::SCREEN, 0, b"s").unwrap();
        frames::write_record(&mut out, frames::SCREEN, 4200, b"").unwrap(); // stream ended
        fs::write(d.join("7.frames"), out).unwrap();
        let s = summary(&d, 7).unwrap();
        assert!(s.screen && !s.camera);
        assert_eq!(s.ends_ms, Some(4200));
        assert!(summary(&d, 8).is_none()); // never recorded / pruned → no panel
        fs::remove_dir_all(d).unwrap();
    }

    // Review focus 4: no live capture → begin writes nothing, so a session can't
    // fail or leave an empty replay because of the screen.
    #[test]
    fn begin_without_capture_writes_no_file() {
        let _g = TEST_LOCK.lock().unwrap();
        let d = dir("nocap");
        assert!(!start_session(&d, 5, false).unwrap());
        assert!(!d.join("5.frames").exists());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn session_records_on_the_audio_clock_and_marks_the_end() {
        let _g = TEST_LOCK.lock().unwrap();
        let d = dir("clock");
        assert!(start_session(&d, 9, true).unwrap());
        let rgb = vec![128u8; 64 * 36 * 3];
        screen_sample(64, 36, &rgb); // before the first audio chunk → ms 0
        mark_clock();
        std::thread::sleep(std::time::Duration::from_millis(30));
        stream_ended();
        end();
        let idx = frames::index(&d.join("9.frames")).unwrap();
        assert_eq!(idx[0].ms, 0);
        assert!(idx[0].len > 0);
        assert_eq!(idx.last().unwrap().len, 0);
        assert!(idx.last().unwrap().ms >= 30);
        fs::remove_dir_all(d).unwrap();
    }
}
