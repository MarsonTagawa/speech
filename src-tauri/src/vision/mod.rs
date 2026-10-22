//! Camera capture and face/pose/hand inference for presence grading, run in
//! Rust so the webview's UI thread stays free (see
//! docs/superpowers/specs/2026-10-08-rust-vision-design.md). The webview gets
//! one JSON result per processed frame (hands 25 Hz, face and body 12.5 Hz)
//! and a JPEG preview of every camera frame (30 fps).
mod capture;
pub mod face;
pub mod geometry;
#[cfg(test)]
mod golden_util;
pub mod hands;
mod model;
pub mod pose;

use capture::{Latest, H, W};
use face::{Face, BLENDSHAPE_NAMES};
use hands::Hands;
use pose::Pose;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tauri::ipc::{Channel, InvokeResponseBody};
use tauri::{AppHandle, Manager, State};

/// Hands run every tick (25 Hz): a fast hand leaves the ROI it was tracked in
/// between slower ticks. Face and body change slower and cost more (~20 ms vs
/// ~7 per hand), so they run every BODY_EVERY ticks (12.5 Hz).
const TICK: Duration = Duration::from_millis(40);
const BODY_EVERY: u32 = 2;
const MAX_FAILS: u32 = 30;

#[derive(Serialize, Clone, Copy)]
pub struct Pt {
    x: f32,
    y: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    z: Option<f32>, // hands only: depth, normalised by width like x
}

#[derive(Serialize, Clone)]
pub struct Head {
    yaw: f32,
    pitch: f32,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum VisionMsg {
    Frame { t: u64, blend: Option<BTreeMap<&'static str, f32>>, head: Option<Head>, pose: Option<Vec<Pt>>, hands: Vec<Vec<Pt>>, world: Vec<Vec<[f32; 3]>>, handed: Vec<f32> },
    Ended { ended: bool },
}

pub struct Pipelines {
    face: Face,
    pose: Pose,
    hands: Hands,
    body: Body,
}

/// The last face and pose results, sent again on hands-only ticks.
type Body = (Option<BTreeMap<&'static str, f32>>, Option<Head>, Option<Vec<Pt>>);

impl Pipelines {
    pub fn load(dir: &Path) -> Result<Pipelines, String> {
        Ok(Pipelines { face: Face::load(dir)?, pose: Pose::load(dir)?, hands: Hands::load(dir)?, body: (None, None, None) })
    }

    /// One frame's results; face and pose only when `body`, else last time's.
    pub fn frame(&mut self, t: u64, rgb: &[u8], body: bool) -> Result<VisionMsg, String> {
        let norm = |p: &[f32; 2]| Pt { x: p[0] / W as f32, y: p[1] / H as f32, z: None };
        let norm3 = |p: &[f32; 3]| Pt { x: p[0] / W as f32, y: p[1] / H as f32, z: Some(p[2] / W as f32) };
        if body {
            let face = self.face.process(rgb, W, H)?;
            let pose = self.pose.process(rgb, W, H)?;
            self.body = (
                face.as_ref().map(|f| BLENDSHAPE_NAMES.iter().copied().zip(f.blend.iter().copied()).collect()),
                face.as_ref().map(|f| Head { yaw: f.yaw, pitch: f.pitch }),
                pose.map(|p| p.iter().map(norm).collect()),
            );
        }
        let hands = self.hands.process(rgb, W, H)?;
        let (blend, head, pose) = self.body.clone();
        Ok(VisionMsg::Frame {
            t,
            blend,
            head,
            pose,
            hands: hands.iter().map(|hd| hd.pts.iter().map(norm3).collect()).collect(),
            world: hands.iter().map(|hd| hd.world.clone()).collect(),
            handed: hands.iter().map(|hd| hd.handed).collect(),
        })
    }
}

/// Inference thread: every TICK on the newest frame, one result each. (The
/// preview goes out from the capture thread at the camera's 30 fps.)
fn infer(mut pipes: Pipelines, latest: Arc<Latest>, stop: Arc<AtomicBool>, ended: Arc<AtomicBool>, results: Channel<VisionMsg>) {
    let mut fails = 0;
    let mut tick = 0u32;
    let mut last = Instant::now() - TICK;
    while !stop.load(Relaxed) && !ended.load(Relaxed) {
        thread::sleep(TICK.saturating_sub(last.elapsed()));
        last = Instant::now();
        let frame = {
            let mut g = latest.slot.lock().unwrap();
            while g.is_none() && !stop.load(Relaxed) && !ended.load(Relaxed) {
                g = latest.ready.wait_timeout(g, Duration::from_millis(200)).unwrap().0;
            }
            g.take()
        };
        let Some((t, rgb)) = frame else { continue };
        let body = tick % BODY_EVERY == 0;
        tick += 1;
        match pipes.frame(t, &rgb, body) {
            Ok(msg) => {
                fails = 0;
                let _ = results.send(msg);
            }
            Err(e) => {
                eprintln!("[vision] frame failed: {e}");
                fails += 1;
                if fails >= MAX_FAILS {
                    ended.store(true, Relaxed);
                }
            }
        }
    }
    if ended.load(Relaxed) && !stop.load(Relaxed) {
        let _ = results.send(VisionMsg::Ended { ended: true });
    }
}

struct Running {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

#[derive(Default)]
pub struct VisionState(Mutex<Option<Running>>);

impl VisionState {
    /// Signals both threads and waits for them; the camera is released when
    /// the capture thread returns.
    pub fn stop(&self) {
        stop_running(&mut self.0.lock().unwrap());
    }
}

fn stop_running(slot: &mut Option<Running>) {
    if let Some(r) = slot.take() {
        r.stop.store(true, Relaxed);
        for t in r.threads {
            let _ = t.join();
        }
        crate::screen::camera_off();
    }
}

// async: model loading and opening the camera take a few hundred ms and must
// not block the main thread (sync Tauri commands run on it). The state lock is
// held for the whole start (there are no awaits), so overlapping start/stop
// calls — a quick off→on click — run one after another instead of two starts
// both opening the camera (EBUSY) and one session leaking. `camera` is the
// camera's name (Settings → Camera); None = the first one.
#[tauri::command]
pub async fn start_vision(app: AppHandle, state: State<'_, VisionState>, results: Channel<VisionMsg>, preview: Channel<InvokeResponseBody>, camera: Option<String>) -> Result<(), String> {
    let mut slot = state.0.lock().unwrap();
    stop_running(&mut slot);
    let dir = app.path().resolve("resources/vision", tauri::path::BaseDirectory::Resource).map_err(|e| e.to_string())?;
    let pipes = Pipelines::load(&dir)?;
    let latest = Arc::new(Latest::default());
    let (stop, ended) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (tx, rx) = mpsc::channel();
    let cap = {
        let (latest, stop, ended) = (latest.clone(), stop.clone(), ended.clone());
        thread::spawn(move || capture::run(camera, latest, stop, ended, tx, preview))
    };
    if let Err(e) = rx.recv().map_err(|e| e.to_string()).and_then(|r| r) {
        let _ = cap.join();
        return Err(e);
    }
    let inf = {
        let stop = stop.clone();
        thread::spawn(move || infer(pipes, latest, stop, ended, results))
    };
    *slot = Some(Running { stop, threads: vec![cap, inf] });
    Ok(())
}

/// A gesture report (the webview's JSON: the last few seconds of results and
/// what the gesture code made of them) to <app data>/gesture-reports/<ms>.json,
/// for debugging a wrong gesture. Returns the file's path.
#[tauri::command]
pub fn save_gesture_report(app: AppHandle, json: String) -> Result<String, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("gesture-reports");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_millis();
    let file = dir.join(format!("{ms}.json"));
    std::fs::write(&file, json).map_err(|e| e.to_string())?;
    Ok(file.display().to_string())
}

/// Camera names for Settings → Camera. async: the query talks to the OS
/// camera stack, which can take a moment.
#[tauri::command]
pub async fn list_cameras() -> Vec<String> {
    capture::names()
}

#[tauri::command]
pub async fn stop_vision(state: State<'_, VisionState>) -> Result<(), String> {
    state.stop();
    Ok(())
}
