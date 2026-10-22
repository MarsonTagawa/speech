//! Screen OCR for "reading off the slides": the text in each changed screen
//! frame (ocrs), so the report can compare speech to what was on screen. A
//! worker per session reads frames as the recorder saves them; the report
//! fetches the results at Stop (screen_text). See
//! docs/superpowers/specs/2026-10-21-screen-reading-design.md.
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const DETECTION: &str = "text-detection.rten";
pub const RECOGNITION: &str = "text-recognition.rten";
pub const MODEL_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/";

/// ponytail: a frame replaced within this long is a transient (scroll,
/// animation, slide transition) and isn't read; tune from real sessions.
const TRANSIENT_MS: u32 = 1000;
/// ponytail: frames waiting for OCR at most (≤ ~6 MB each at OCR_MAX_W); a
/// screen busy for longer than OCR keeps up loses its oldest frames.
const QUEUE_CAP: usize = 8;
/// Frames are read at most this wide: slide text stays legible, memory bounded.
const OCR_MAX_W: usize = 1920;

pub struct Engine(OcrEngine);

impl Engine {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let model = |f: &str| Model::load_file(dir.join(f)).map_err(|e| format!("{f}: {e}"));
        let params = OcrEngineParams {
            detection_model: Some(model(DETECTION)?),
            recognition_model: Some(model(RECOGNITION)?),
            ..Default::default()
        };
        OcrEngine::new(params).map(Engine).map_err(|e| e.to_string())
    }

    /// The text in a packed-RGB frame, lines joined with "\n".
    pub fn text(&self, w: usize, h: usize, rgb: &[u8]) -> Result<String, String> {
        let src = ImageSource::from_bytes(rgb, (w as u32, h as u32)).map_err(|e| e.to_string())?;
        let input = self.0.prepare_input(src).map_err(|e| e.to_string())?;
        self.0.get_text(&input).map_err(|e| e.to_string())
    }
}

pub struct Frame {
    ms: u32,
    w: usize,
    h: usize,
    rgb: Vec<u8>,
    at: Instant, // when it was queued (a transient is judged in wall time)
}

/// Queues `f`: it replaces a tail it follows within TRANSIENT_MS, and the
/// oldest frame goes when the queue is over QUEUE_CAP.
fn push(q: &mut VecDeque<Frame>, f: Frame) {
    if q.back().is_some_and(|b| f.ms.saturating_sub(b.ms) < TRANSIENT_MS) {
        q.pop_back();
    }
    q.push_back(f);
    if q.len() > QUEUE_CAP {
        q.pop_front();
    }
}

struct Queue {
    frames: VecDeque<Frame>,
    open: bool,
}

struct Job {
    ts: i64,
    queue: Arc<(Mutex<Queue>, Condvar)>,
    out: Arc<Mutex<Vec<(u32, String)>>>,
    thread: JoinHandle<()>,
}

/// The current (or last) session's OCR, past any poisoning like screen's SESSION.
static JOB: Mutex<Option<Job>> = Mutex::new(None);
fn job() -> MutexGuard<'static, Option<Job>> {
    JOB.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts session `ts`'s worker with the models in `models` (loaded on the
/// worker, so the session start never waits), calling `notify(ms, text)` as
/// each frame is read (live slide laps). No models → no worker.
// ponytail: loads the engine once per session (fresh worker thread); keep one across sessions if load time shows.
pub fn start(ts: i64, models: PathBuf, notify: impl Fn(u32, &str) + Send + 'static) {
    if !models.join(DETECTION).exists() || !models.join(RECOGNITION).exists() {
        return;
    }
    let mut engine: Option<Result<Engine, String>> = None;
    let read = move |w, h, rgb: &[u8]| match engine.get_or_insert_with(|| Engine::load(&models)) {
        Ok(e) => e.text(w, h, rgb),
        Err(err) => Err(err.clone()),
    };
    start_with(ts, read, notify);
}

fn start_with(
    ts: i64,
    mut read: impl FnMut(usize, usize, &[u8]) -> Result<String, String> + Send + 'static,
    notify: impl Fn(u32, &str) + Send + 'static,
) {
    close(); // a previous session's worker finishes on its own
    let queue = Arc::new((Mutex::new(Queue { frames: VecDeque::new(), open: true }), Condvar::new()));
    let out = Arc::new(Mutex::new(Vec::new()));
    let thread = {
        let (queue, out) = (queue.clone(), out.clone());
        thread::spawn(move || {
            while let Some(f) = next(&queue) {
                match read(f.w, f.h, &f.rgb) {
                    Ok(text) => {
                        notify(f.ms, &text);
                        out.lock().unwrap().push((f.ms, text));
                    }
                    Err(e) => eprintln!("[ocr] frame at {} ms: {e}", f.ms),
                }
            }
        })
    };
    *job() = Some(Job { ts, queue, out, thread });
}

/// The next frame to read: the oldest, once it can't be a transient any more
/// (a newer frame follows it, it's TRANSIENT_MS old, or the session ended).
/// None once the session ended and the queue is empty.
fn next(queue: &(Mutex<Queue>, Condvar)) -> Option<Frame> {
    let (lock, cv) = queue;
    let mut q = lock.lock().unwrap();
    loop {
        let wait = match q.frames.front() {
            None if !q.open => return None,
            None => None,
            Some(f) => {
                let left = Duration::from_millis(TRANSIENT_MS as u64).saturating_sub(f.at.elapsed());
                if q.frames.len() > 1 || !q.open || left.is_zero() {
                    return q.frames.pop_front();
                }
                Some(left)
            }
        };
        q = match wait {
            Some(t) => cv.wait_timeout(q, t).unwrap().0,
            None => cv.wait(q).unwrap(),
        };
    }
}

/// A changed screen frame of the running session (screen_sample). Cheap when
/// no worker runs.
pub fn feed(ms: u32, w: usize, h: usize, rgb: &[u8]) {
    let g = job();
    let Some(j) = g.as_ref() else { return };
    let (lock, cv) = &*j.queue;
    let mut q = lock.lock().unwrap();
    if !q.open {
        return;
    }
    let (nw, nh) = if w > OCR_MAX_W { (OCR_MAX_W, h * OCR_MAX_W / w) } else { (w, h) };
    let rgb = if nw == w { rgb.to_vec() } else { super::frames::resize(rgb, w, h, nw, nh) };
    push(&mut q.frames, Frame { ms, w: nw, h: nh, rgb, at: Instant::now() });
    cv.notify_one();
}

/// The session ended: the worker reads what's queued, then exits.
pub fn close() {
    if let Some(j) = job().as_ref() {
        j.queue.0.lock().unwrap().open = false;
        j.queue.1.notify_one();
    }
}

/// Session `ts`'s text by frame ms, in time order, waiting up to `wait` for
/// its worker to finish; whatever's done by then otherwise. Empty for any
/// other session.
pub fn results(ts: i64, wait: Duration) -> Vec<(u32, String)> {
    let deadline = Instant::now() + wait;
    loop {
        let g = job();
        let Some(j) = g.as_ref().filter(|j| j.ts == ts) else { return Vec::new() };
        if j.thread.is_finished() || Instant::now() >= deadline {
            let mut out = j.out.lock().unwrap().clone();
            out.sort_by_key(|r| r.0);
            return out;
        }
        drop(g);
        thread::sleep(Duration::from_millis(50));
    }
}

/// Tests anywhere in the crate that touch JOB take turns.
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn running(ts: i64) -> bool {
    job().as_ref().is_some_and(|j| j.ts == ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(ms: u32) -> Frame {
        Frame { ms, w: 1, h: 1, rgb: vec![0; 3], at: std::time::Instant::now() }
    }
    fn ms(q: &VecDeque<Frame>) -> Vec<u32> {
        q.iter().map(|f| f.ms).collect()
    }

    // A frame replaced within TRANSIENT_MS is a transient: the newer one takes its place.
    #[test]
    fn push_replaces_a_transient_tail() {
        let mut q = VecDeque::new();
        for m in [0, 1500, 1900, 2300, 4000] {
            push(&mut q, f(m));
        }
        assert_eq!(ms(&q), [0, 2300, 4000]);
    }

    // Review focus 5: a busy screen can't grow the queue without bound, and the
    // newest frame (where the screen settles) is the one kept.
    #[test]
    fn push_caps_the_queue_keeping_the_newest() {
        let mut q = VecDeque::new();
        for m in (0..20).map(|i| i * 2000) {
            push(&mut q, f(m));
        }
        assert_eq!(q.len(), QUEUE_CAP);
        assert_eq!(q.back().unwrap().ms, 38000);
    }

    // Review focus 4: a new session's results never include the last one's.
    #[test]
    fn results_are_per_session() {
        let _g = TEST_LOCK.lock().unwrap();
        start_with(1, |_, _, _| Ok("slide one".into()), |_, _| {});
        feed(0, 1, 1, &[0, 0, 0]);
        close();
        assert_eq!(results(1, Duration::from_secs(5)), [(0, "slide one".to_string())]);
        assert!(results(2, Duration::from_secs(1)).is_empty());
    }

    // The worker skips a frame that fails and keeps going.
    #[test]
    fn a_failed_frame_is_skipped() {
        let _g = TEST_LOCK.lock().unwrap();
        start_with(3, |w, _, _| if w == 2 { Err("bad".into()) } else { Ok("ok".into()) }, |_, _| {});
        feed(0, 2, 1, &[0; 6]);
        feed(5000, 1, 1, &[0; 3]);
        close();
        assert_eq!(results(3, Duration::from_secs(5)), [(5000, "ok".to_string())]);
    }


    // Live laps: each frame read is reported as it lands, in order.
    #[test]
    fn each_read_frame_is_reported_as_it_lands() {
        let _g = TEST_LOCK.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        start_with(4, |_, _, _| Ok("slide".into()), move |ms, text| {
            let _ = tx.send((ms, text.to_string()));
        });
        feed(0, 1, 1, &[0; 3]);
        feed(5000, 1, 1, &[0; 3]);
        close();
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (0, "slide".to_string()));
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (5000, "slide".to_string()));
    }

    // Where the app downloads the models (Linux app data dir).
    fn models() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/com.marsont.speech/models")
    }

    // Needs the models: turn Record screen on once in the app, then
    // `cargo test --release screen::ocr -- --ignored`.
    #[test]
    #[ignore]
    fn reads_a_rendered_slide() {
        let img = image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/resources/ocr-test/slide.jpg")).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let t0 = std::time::Instant::now();
        let engine = Engine::load(&models()).unwrap();
        let loaded = t0.elapsed().as_millis();
        let text = engine.text(w, h, img.as_raw()).unwrap().to_lowercase();
        eprintln!("[ocr] load {loaded} ms, read {} ms: {text:?}", t0.elapsed().as_millis() - loaded);
        assert!(text.contains("revenue grew 40 percent in q3"), "{text}");
        assert!(text.contains("driven by enterprise renewals"), "{text}");
    }
}
