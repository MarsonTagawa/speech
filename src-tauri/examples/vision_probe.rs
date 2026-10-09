// Runs the vision pipelines on the live camera for 5 s without the UI and prints
// throughput: `cargo run --example vision_probe --manifest-path src-tauri/Cargo.toml`
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use speech_lib::vision_probe::Pipelines;
use std::time::Instant;

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/vision");
    let mut pipes = Pipelines::load(&dir).expect("models");
    let fmt = CameraFormat::new(Resolution::new(640, 360), FrameFormat::YUYV, 30);
    let mut cam = Camera::new(CameraIndex::Index(0), RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt))).expect("camera");
    cam.open_stream().expect("stream");
    let (start, mut n, mut ms) = (Instant::now(), 0, Vec::new());
    while start.elapsed().as_secs() < 5 {
        let rgb = cam.frame().unwrap().decode_image::<RgbFormat>().unwrap().into_raw();
        let t = Instant::now();
        let msg = serde_json::to_string(&pipes.frame(0, &rgb).unwrap()).unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
        n += 1;
        if n % 10 == 0 {
            println!("{}", &msg[..msg.len().min(160)]);
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{n} frames, inference p50 {:.1} ms, p95 {:.1} ms", ms[ms.len() / 2], ms[ms.len() * 95 / 100]);
}
