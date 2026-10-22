//! Camera thread: YUYV 640×360@30 via nokhwa, decoded to RGB, kept in a
//! one-frame slot so the inference thread always takes the newest frame; every
//! frame also goes to the webview as a JPEG, so the video runs at 30 fps while
//! inference runs at its own rate.
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Instant;
use tauri::ipc::{Channel, InvokeResponseBody};

/// Frame timestamps share one process-wide epoch so they keep increasing
/// across camera sessions (presence.ts compares times between frames, and
/// main.ts throttles the live card by them).
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

pub fn now_ms() -> u64 {
    EPOCH.elapsed().as_millis() as u64
}

pub const W: usize = 640;
pub const H: usize = 360;

#[derive(Default)]
pub struct Latest {
    pub slot: Mutex<Option<(u64, Vec<u8>)>>, // (now_ms(), RGB)
    pub ready: Condvar,
}

/// Runs until `stop`; sets `ended` if the camera fails mid-stream. Reports the
/// open result on `opened` first so start_vision can fail fast (busy/missing).
pub fn run(latest: Arc<Latest>, stop: Arc<AtomicBool>, ended: Arc<AtomicBool>, opened: Sender<Result<(), String>>, preview: Channel<InvokeResponseBody>) {
    let fmt = CameraFormat::new(Resolution::new(W as u32, H as u32), FrameFormat::YUYV, 30);
    let cam = Camera::new(CameraIndex::Index(0), RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt))).and_then(|mut c| c.open_stream().map(|_| c));
    let mut cam = match cam {
        Ok(c) => {
            let _ = opened.send(Ok(()));
            c
        }
        Err(e) => {
            let _ = opened.send(Err(e.to_string())); // the UI adds "Camera unavailable — "
            return;
        }
    };
    let mut n = 0u32;
    while !stop.load(Relaxed) {
        match cam.frame().and_then(|f| f.decode_image::<RgbFormat>()) {
            Ok(img) => {
                let rgb = img.into_raw();
                n = n.wrapping_add(1);
                crate::screen::camera_frame(n, &rgb); // replay's camera bubble
                let _ = preview.send(InvokeResponseBody::Raw(jpeg(&rgb)));
                *latest.slot.lock().unwrap() = Some((now_ms(), rgb));
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

fn jpeg(rgb: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = jpeg_encoder::Encoder::new(&mut buf, 70).encode(rgb, W as u16, H as u16, jpeg_encoder::ColorType::Rgb);
    buf
}

#[cfg(test)]
mod tests {
    // main.ts/presence.ts compare frame times across camera sessions (live
    // card throttle, motion between frames), so a new session must not
    // restart the clock at 0.
    #[test]
    fn timestamps_keep_counting_across_sessions() {
        let first_session = super::now_ms();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let second_session = super::now_ms();
        assert!(second_session >= first_session + 20, "{first_session} → {second_session}");
    }
}
