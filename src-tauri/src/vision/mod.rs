//! Camera capture and face/pose/hand inference for presence grading, run in
//! Rust so the webview's UI thread stays free (see
//! docs/superpowers/specs/2026-10-08-rust-vision-design.md). The webview gets
//! one JSON result and one JPEG preview per processed frame (~12.5 Hz).
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

const TICK: Duration = Duration::from_millis(80);
const MAX_FAILS: u32 = 30;

#[derive(Serialize, Clone, Copy)]
pub struct Pt {
    x: f32,
    y: f32,
}

#[derive(Serialize)]
pub struct Head {
    yaw: f32,
    pitch: f32,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum VisionMsg {
    Frame { t: u64, blend: Option<BTreeMap<&'static str, f32>>, head: Option<Head>, pose: Option<Vec<Pt>>, hands: Vec<Vec<Pt>> },
    Ended { ended: bool },
}

pub struct Pipelines {
    face: Face,
    pose: Pose,
    hands: Hands,
}

impl Pipelines {
    pub fn load(dir: &Path) -> Result<Pipelines, String> {
        Ok(Pipelines { face: Face::load(dir)?, pose: Pose::load(dir)?, hands: Hands::load(dir)? })
    }

    pub fn frame(&mut self, t: u64, rgb: &[u8]) -> Result<VisionMsg, String> {
        let norm = |p: &[f32; 2]| Pt { x: p[0] / W as f32, y: p[1] / H as f32 };
        let face = self.face.process(rgb, W, H)?;
        let pose = self.pose.process(rgb, W, H)?;
        let hands = self.hands.process(rgb, W, H)?;
        Ok(VisionMsg::Frame {
            t,
            blend: face.as_ref().map(|f| BLENDSHAPE_NAMES.iter().copied().zip(f.blend.iter().copied()).collect()),
            head: face.as_ref().map(|f| Head { yaw: f.yaw, pitch: f.pitch }),
            pose: pose.map(|p| p.iter().map(norm).collect()),
            hands: hands.iter().map(|hd| hd.iter().map(norm).collect()).collect(),
        })
    }
}

fn jpeg(rgb: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = jpeg_encoder::Encoder::new(&mut buf, 70).encode(rgb, W as u16, H as u16, jpeg_encoder::ColorType::Rgb);
    buf
}

/// Inference thread: ~12.5 Hz on the newest frame; one result + one preview per frame.
fn infer(mut pipes: Pipelines, latest: Arc<Latest>, stop: Arc<AtomicBool>, ended: Arc<AtomicBool>, results: Channel<VisionMsg>, preview: Channel<InvokeResponseBody>) {
    let mut fails = 0;
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
        match pipes.frame(t, &rgb) {
            Ok(msg) => {
                fails = 0;
                let _ = results.send(msg);
                let _ = preview.send(InvokeResponseBody::Raw(jpeg(&rgb)));
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
        if let Some(r) = self.0.lock().unwrap().take() {
            r.stop.store(true, Relaxed);
            for t in r.threads {
                let _ = t.join();
            }
        }
    }
}

// async: model loading and opening the camera take a few hundred ms and must
// not block the main thread (sync Tauri commands run on it).
#[tauri::command]
pub async fn start_vision(app: AppHandle, state: State<'_, VisionState>, results: Channel<VisionMsg>, preview: Channel<InvokeResponseBody>) -> Result<(), String> {
    state.stop();
    let dir = app.path().resolve("resources/vision", tauri::path::BaseDirectory::Resource).map_err(|e| e.to_string())?;
    let pipes = Pipelines::load(&dir)?;
    let latest = Arc::new(Latest::default());
    let (stop, ended) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (tx, rx) = mpsc::channel();
    let cap = {
        let (latest, stop, ended) = (latest.clone(), stop.clone(), ended.clone());
        thread::spawn(move || capture::run(latest, stop, ended, tx))
    };
    if let Err(e) = rx.recv().map_err(|e| e.to_string()).and_then(|r| r) {
        let _ = cap.join();
        return Err(e);
    }
    let inf = {
        let stop = stop.clone();
        thread::spawn(move || infer(pipes, latest, stop, ended, results, preview))
    };
    *state.0.lock().unwrap() = Some(Running { stop, threads: vec![cap, inf] });
    Ok(())
}

#[tauri::command]
pub async fn stop_vision(state: State<'_, VisionState>) -> Result<(), String> {
    state.stop();
    Ok(())
}
