//! Helpers for the golden tests: MediaPipe's test images and the official
//! MediaPipe Python output on them (scripts/convert-vision-models.sh --goldens).
use std::path::PathBuf;

pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/vision")
}

pub fn image(name: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(dir().join("golden").join(name)).expect("run scripts/convert-vision-models.sh --goldens").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    (img.into_raw(), w, h)
}

pub fn goldens() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("golden/goldens.json")).unwrap()).unwrap()
}

/// Mean distance between our pixel points and the expected normalised points,
/// as a fraction of the image width.
pub fn mean_err(got: &[[f32; 2]], want: &serde_json::Value, w: usize, h: usize) -> f32 {
    let want = want.as_array().unwrap();
    got.iter()
        .zip(want)
        .map(|(g, e)| {
            let (ex, ey) = (e[0].as_f64().unwrap() as f32 * w as f32, e[1].as_f64().unwrap() as f32 * h as f32);
            (g[0] - ex).hypot(g[1] - ey) / w as f32
        })
        .sum::<f32>()
        / got.len() as f32
}
