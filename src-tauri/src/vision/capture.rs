//! Camera thread: 640×360@30 via nokhwa (see `pick`/`fit`), decoded to RGB, kept in a
//! one-frame slot so the inference thread always takes the newest frame; every
//! frame also goes to the webview as a JPEG, so the video runs at 30 fps while
//! inference runs at its own rate.
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
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
    // macOS asks for camera permission here (once); elsewhere it answers true at once.
    let (tx, rx) = std::sync::mpsc::channel();
    nokhwa::nokhwa_initialize(move |ok| {
        let _ = tx.send(ok);
    });
    if !rx.recv().unwrap_or(false) {
        let _ = opened.send(Err("camera permission denied".into()));
        return;
    }
    let mut cam = match open() {
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
                let (w, h) = img.dimensions();
                let rgb = fit(img.into_raw(), w as usize, h as usize);
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

/// Opens camera 0 in the mode `pick` chooses from the ones it lists.
fn open() -> Result<Camera, nokhwa::NokhwaError> {
    let mut cam = Camera::new(CameraIndex::Index(0), RequestedFormat::new::<RgbFormat>(RequestedFormatType::None))?;
    if let Some(fmt) = pick(&cam.compatible_camera_formats()?) {
        cam.set_camera_requset(RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt)))?;
    }
    cam.open_stream()?;
    Ok(cam)
}

/// 640×360 YUYV@30 where the camera has it (most Linux webcams). Otherwise
/// the smallest mode that covers 640×360 at 30 fps, since `fit` crops and scales it
/// down. Mac and Windows cameras mostly offer NV12/MJPEG at 720p or 640×480.
/// (nokhwa's own `Closest` can't do this: it looks up frame rates at the
/// requested resolution, so it fails whenever that resolution is missing.)
fn pick(formats: &[CameraFormat]) -> Option<CameraFormat> {
    formats
        .iter()
        .filter(|f| nokhwa::utils::color_frame_formats().contains(&f.format()))
        .min_by_key(|f| {
            let (w, h) = (f.resolution().width() as usize, f.resolution().height() as usize);
            let rank = match f.format() {
                FrameFormat::YUYV => 0,
                FrameFormat::NV12 => 1,
                _ => 2, // MJPEG costs a JPEG decode per frame
            };
            (w < W || h < H, f.frame_rate() < 30, w * h, rank)
        })
        .copied()
}

/// Center-crops a w×h RGB frame to 16:9 and scales it (nearest neighbour) to
/// W×H, the size everything downstream assumes. A no-op at W×H.
fn fit(rgb: Vec<u8>, w: usize, h: usize) -> Vec<u8> {
    if (w, h) == (W, H) {
        return rgb;
    }
    let (cw, ch) = if w * H > h * W { (h * W / H, h) } else { (w, w * H / W) };
    let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
    let mut out = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        let row = (y0 + y * ch / H) * w;
        for x in 0..W {
            let i = (row + x0 + x * cw / W) * 3;
            out.extend_from_slice(&rgb[i..i + 3]);
        }
    }
    out
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

    #[test]
    fn picks_a_mode_fit_can_scale_down() {
        use nokhwa::utils::{CameraFormat, FrameFormat::*, Resolution};
        let f = |w, h, fmt, fps| CameraFormat::new(Resolution::new(w, h), fmt, fps);
        // Linux webcam: the exact mode wins.
        let linux = [f(1280, 720, MJPEG, 30), f(640, 360, YUYV, 30), f(320, 180, YUYV, 30)];
        assert_eq!(super::pick(&linux), Some(linux[1]));
        // Mac-style: no 640×360, no YUYV; smallest 30 fps mode that covers it.
        let mac = [f(1920, 1080, NV12, 30), f(1280, 720, NV12, 30), f(640, 480, NV12, 15), f(320, 240, NV12, 30)];
        assert_eq!(super::pick(&mac), Some(mac[1]));
    }

    #[test]
    fn fit_crops_to_16_9_and_scales() {
        use super::{fit, H, W};
        // 4:3 at 2x: pixel value = source row, so the crop's top edge shows up.
        let (w, h) = (W * 2, W * 2 * 3 / 4); // 1280×960 → crop 1280×720 from row 120
        let src: Vec<u8> = (0..h).flat_map(|y| std::iter::repeat((y / 4) as u8).take(w * 3)).collect();
        let out = fit(src, w, h);
        assert_eq!(out.len(), W * H * 3);
        assert_eq!(out[0], (120 / 4) as u8);
        assert_eq!(out[(H - 1) * W * 3], ((120 + (H - 1) * 2) / 4) as u8);
    }
}
