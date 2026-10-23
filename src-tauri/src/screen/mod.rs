//! Screen replay: records one display (plus the camera when it's on) during a
//! session as change-driven JPEG frames in `sessions/<ts>.frames`, replayed in
//! the report in sync with the session audio. See
//! docs/superpowers/specs/2026-10-20-screen-replay-design.md.
//!
//! While the toggle is on, a capture thread reads the display (screen_prepare
//! opens it once, so later sessions don't re-prompt). During a session it
//! samples every SAMPLE and saves frames that changed. The camera thread hands
//! frames to a writer thread through a small queue, so it never waits on disk.
pub mod export;
mod frames;
pub(crate) mod ocr;
pub mod stream;

use frames::{Rec, Recorder, CAMERA, SCREEN};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tauri::{ipc::Response, AppHandle, Emitter, Manager};

const SAMPLE: Duration = Duration::from_millis(250);
/// How often an idle capture checks for stop and a new session.
const IDLE_POLL: Duration = Duration::from_millis(50);
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
/// SESSION and CAMERA_TX, past any poisoning: the audio and camera threads call
/// into this module, and a panic elsewhere must not take their recordings down too.
fn session() -> MutexGuard<'static, Option<Session>> {
    SESSION.lock().unwrap_or_else(|e| e.into_inner())
}
fn camera_tx() -> MutexGuard<'static, Option<SyncSender<Option<Vec<u8>>>>> {
    CAMERA_TX.lock().unwrap_or_else(|e| e.into_inner())
}
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
    let models = crate::models::ocr_dir(app).ok();
    let notify = {
        let app = app.clone(); // live slide laps (main.ts screen_frame_text)
        move |ms: u32, text: &str| {
            let _ = app.emit("screen_frame_text", (ts, ms, text));
        }
    };
    if let Err(e) = crate::history::detail_dir(app).and_then(|d| start_session(&d, ts, live, models.as_deref(), notify)) {
        let _ = app.emit("screen_ended", format!("couldn't save the video — {e}"));
    }
}

/// Opens `<dir>/<ts>.frames` and the camera writer when `live`, with screen OCR
/// (reporting each frame's text to `notify`) when the `models` are there.
/// Returns whether it did.
fn start_session(
    dir: &Path,
    ts: i64,
    live: bool,
    models: Option<&Path>,
    notify: impl Fn(u32, &str) + Send + 'static,
) -> Result<bool, String> {
    if !live {
        return Ok(false);
    }
    // Before SESSION is set: the capture thread may hand the session its first
    // frame the moment it is.
    if let Some(m) = models {
        ocr::start(ts, m.to_path_buf(), notify);
    }
    let rec = Recorder::create(&dir.join(format!("{ts}.frames"))).map_err(|e| {
        ocr::close();
        e.to_string()
    })?;
    *session() = Some(Session { rec, clock: None });
    let (tx, rx) = mpsc::sync_channel::<Option<Vec<u8>>>(CAMERA_QUEUE);
    thread::spawn(move || {
        for frame in rx {
            if let Some(s) = session().as_mut() {
                let ms = s.ms();
                if let Err(e) = s.rec.camera(ms, frame.as_deref()) {
                    eprintln!("[screen] camera frame: {e}");
                }
            }
        }
    });
    *camera_tx() = Some(tx);
    Ok(true)
}

/// The session's first audio chunk arrived: ms 0 of its `.pcm`. Idempotent.
pub fn mark_clock() {
    if let Some(s) = session().as_mut() {
        s.clock.get_or_insert_with(Instant::now);
    }
}

/// Session stops (audio::stop_recording).
pub fn end() {
    ocr::close();
    camera_tx().take(); // the writer thread drains and exits
    if let Some(s) = session().take() {
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
    if let Some(tx) = camera_tx().as_ref() {
        let _ = tx.try_send(Some(rgb.to_vec())); // full → drop the frame
    }
}

/// The camera turned off (vision::stop_running): the replay hides the bubble from here.
pub fn camera_off() {
    if let Some(tx) = camera_tx().as_ref() {
        let _ = tx.send(None); // must not be dropped like a frame
    }
}

fn screen_sample(w: usize, h: usize, rgb: &[u8]) {
    let mut g = session();
    if let Some(s) = g.as_mut() {
        let ms = s.ms();
        match s.rec.screen(ms, w, h, rgb) {
            Ok(true) => ocr::feed(ms, w, h, rgb),
            Ok(false) => {}
            Err(e) => {
                eprintln!("[screen] writing a frame failed, video stops here: {e}");
                *g = None; // keep recording audio; drop the replay's tail
            }
        }
    }
}

fn stream_ended() {
    if let Some(s) = session().as_mut() {
        let ms = s.ms();
        let _ = s.rec.screen_ended(ms);
    }
}

/// Whether the session has no screen frame yet (it gets the held one at once).
fn session_wants_first() -> bool {
    session().as_ref().is_some_and(|s| !s.rec.has_screen())
}

/// Feeds sessions from `next` until `stop` (None) or the stream ends (its
/// reason). Every SAMPLE it records the newest frame received since the last
/// sample, holding frames that arrive in between rather than dropping them:
/// the compositor may not send another once the screen settles.
fn feed(next: impl Fn(Duration) -> Result<Option<stream::Frame>, String>, stop: &AtomicBool) -> Option<String> {
    let mut held: Option<stream::Frame> = None;
    let mut fresh = false; // `held` not yet sampled
    let mut last = Instant::now() - SAMPLE;
    while !stop.load(Relaxed) {
        let wait = if fresh { SAMPLE.saturating_sub(last.elapsed()) } else { IDLE_POLL };
        match next(wait) {
            Ok(Some(f)) => (held, fresh) = (Some(f), true),
            Ok(None) => {}
            Err(e) => return Some(e),
        }
        if let Some(f) = &held {
            if (fresh && last.elapsed() >= SAMPLE) || session_wants_first() {
                screen_sample(f.w, f.h, &f.rgb);
                (fresh, last) = (false, Instant::now());
            }
        }
    }
    None
}

fn run(app: AppHandle, stream: stream::Stream, stop: Arc<AtomicBool>) {
    if let Some(e) = feed(|t| stream.next(t), &stop) {
        stream_ended();
        let _ = app.emit("screen_ended", e);
    }
}

// --- Commands -------------------------------------------------------------------

/// Toggle on: asks for the display/permission and starts reading it. async so
/// the picker doesn't block the main thread.
#[tauri::command]
pub async fn screen_prepare(app: AppHandle, display: Option<String>) -> Result<(), String> {
    app.state::<ScreenState>().release();
    {
        let app = app.clone(); // models for the OCR worker; sessions without them just skip OCR
        thread::spawn(move || {
            if let Err(e) = crate::models::ocr_models(&app) {
                eprintln!("[ocr] model download: {e}");
            }
        });
    }
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

/// Session `ts`'s screen text by frame ms (screen/ocr.rs), once its OCR is
/// done or after 60 s; empty without OCR.
#[tauri::command]
pub async fn screen_text(ts: i64) -> Vec<(u32, String)> {
    tauri::async_runtime::spawn_blocking(move || ocr::results(ts, Duration::from_secs(60))).await.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // SESSION and ocr's JOB are process-wide (end() closes the OCR job); tests

    // that touch either take turns, on one lock so they can't interleave.

    use super::ocr::TEST_LOCK;

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

    // Review I-3: OCR is running before the capture thread can see the session,
    // so the frame a new session gets at once (a static first slide) is read.
    #[test]
    fn start_session_starts_ocr_first() {
        let _g = TEST_LOCK.lock().unwrap();
        let d = dir("ocr");
        let models = d.join("models");
        fs::create_dir_all(&models).unwrap();
        for f in [ocr::DETECTION, ocr::RECOGNITION] {
            fs::write(models.join(f), b"").unwrap();
        }
        assert!(start_session(&d, 31, true, Some(&models), |_, _| {}).unwrap());
        assert!(ocr::running(31));
        end();
        fs::remove_dir_all(d).unwrap();
    }

    // Review focus 4: no live capture → begin writes nothing, so a session can't
    // fail or leave an empty replay because of the screen.
    #[test]
    fn begin_without_capture_writes_no_file() {
        let _g = TEST_LOCK.lock().unwrap();
        let d = dir("nocap");
        assert!(!start_session(&d, 5, false, None, |_, _| {}).unwrap());
        assert!(!d.join("5.frames").exists());
        fs::remove_dir_all(d).unwrap();
    }

    fn flat(v: u8) -> stream::Frame {
        stream::Frame { w: 64, h: 36, rgb: vec![v; 64 * 36 * 3] }
    }

    /// A fake capture: hands out each (at, frame) once its time has come, then goes quiet.
    fn script(frames: Vec<(u64, stream::Frame)>) -> impl Fn(Duration) -> Result<Option<stream::Frame>, String> {
        let t0 = Instant::now();
        let q = Mutex::new(std::collections::VecDeque::from(frames));
        move |timeout| {
            let mut q = q.lock().unwrap();
            match q.front() {
                Some((at, _)) if t0.elapsed() >= Duration::from_millis(*at) => Ok(q.pop_front().map(|(_, f)| f)),
                _ => {
                    thread::sleep(timeout.min(Duration::from_millis(10)));
                    Ok(None)
                }
            }
        }
    }

    // Review #1: the last frame of a burst (a slide settling) arrives inside the
    // sample gap, and nothing follows it. It must still be recorded.
    #[test]
    fn feed_records_the_last_frame_of_a_burst() {
        let _g = TEST_LOCK.lock().unwrap();
        let d = dir("burst");
        assert!(start_session(&d, 11, true, None, |_, _| {}).unwrap());
        mark_clock();
        let stop = AtomicBool::new(false);
        thread::scope(|sc| {
            sc.spawn(|| feed(script(vec![(0, flat(10)), (100, flat(120)), (150, flat(240))]), &stop));
            thread::sleep(Duration::from_millis(700));
            stop.store(true, Relaxed);
        });
        end();
        let idx = frames::index(&d.join("11.frames")).unwrap();
        assert_eq!(idx.len(), 2, "{idx:?}"); // the first frame, then the settled one
        fs::remove_dir_all(d).unwrap();
    }

    // Review #1: a session that starts on a static screen gets the held frame at
    // once, not a blank replay until something changes.
    #[test]
    fn feed_gives_a_new_session_the_current_screen() {
        let _g = TEST_LOCK.lock().unwrap();
        let d = dir("static");
        let stop = AtomicBool::new(false);
        thread::scope(|sc| {
            sc.spawn(|| feed(script(vec![(0, flat(10))]), &stop));
            thread::sleep(Duration::from_millis(300)); // idle: frame already received
            assert!(start_session(&d, 12, true, None, |_, _| {}).unwrap());
            thread::sleep(Duration::from_millis(300));
            stop.store(true, Relaxed);
        });
        end();
        let idx = frames::index(&d.join("12.frames")).unwrap();
        assert_eq!(idx.len(), 1, "{idx:?}");
        fs::remove_dir_all(d).unwrap();
    }

    // Review #7: the audio thread calls these hooks; a panic elsewhere while
    // holding SESSION must not take the session's audio down with it.
    #[test]
    fn hooks_survive_a_poisoned_session_lock() {
        let _g = TEST_LOCK.lock().unwrap();
        let _ = thread::spawn(|| {
            let _s = SESSION.lock().unwrap();
            panic!("poison SESSION");
        })
        .join();
        assert!(SESSION.is_poisoned());
        mark_clock();
        camera_frame(3, &[0; 3]);
        camera_off();
        end();
        SESSION.clear_poison();
    }

    #[test]
    fn session_records_on_the_audio_clock_and_marks_the_end() {
        let _g = TEST_LOCK.lock().unwrap();
        let d = dir("clock");
        assert!(start_session(&d, 9, true, None, |_, _| {}).unwrap());
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
