//! Camera thread: YUYV 640×360@30 via nokhwa, decoded to RGB, kept in a
//! one-frame slot so the inference thread always takes the newest frame.
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

pub const W: usize = 640;
pub const H: usize = 360;

#[derive(Default)]
pub struct Latest {
    pub slot: Mutex<Option<(u64, Vec<u8>)>>, // (ms since start, RGB)
    pub ready: Condvar,
}

/// Runs until `stop`; sets `ended` if the camera fails mid-stream. Reports the
/// open result on `opened` first so start_vision can fail fast (busy/missing).
pub fn run(latest: Arc<Latest>, stop: Arc<AtomicBool>, ended: Arc<AtomicBool>, opened: Sender<Result<(), String>>) {
    let fmt = CameraFormat::new(Resolution::new(W as u32, H as u32), FrameFormat::YUYV, 30);
    let cam = Camera::new(CameraIndex::Index(0), RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt))).and_then(|mut c| c.open_stream().map(|_| c));
    let mut cam = match cam {
        Ok(c) => {
            let _ = opened.send(Ok(()));
            c
        }
        Err(e) => {
            let _ = opened.send(Err(format!("Camera unavailable — {e}")));
            return;
        }
    };
    let start = Instant::now();
    while !stop.load(Relaxed) {
        match cam.frame().and_then(|f| f.decode_image::<RgbFormat>()) {
            Ok(img) => {
                *latest.slot.lock().unwrap() = Some((start.elapsed().as_millis() as u64, img.into_raw()));
                latest.ready.notify_one();
            }
            Err(e) => {
                eprintln!("[vision] camera read failed: {e}");
                ended.store(true, Relaxed);
                break;
            }
        }
    }
    let _ = cam.stop_stream();
    latest.ready.notify_one();
}
