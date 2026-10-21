//! The replay file, `sessions/<ts>.frames`: records back to back,
//! `[kind u8][ms u32 LE][len u32 LE][jpeg]`, kind SCREEN or CAMERA, ms on the
//! session audio's clock. A zero-length record means "off from here" (screen
//! stream ended; camera turned off). Plus the pure image helpers the recorder needs.
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const SCREEN: u8 = 0;
pub const CAMERA: u8 = 1;
const HEADER: u64 = 9;

/// Screen frames are stored at most this wide.
const MAX_W: usize = 1280;
/// Camera frames arrive at 640×360 (vision::capture) and are stored at half that.
pub const CAM_W: usize = 640;
pub const CAM_H: usize = 360;
/// Change-detection thumbnail, compared in 8×6-pixel cells (an 8×6 grid).
pub const TW: usize = 64;
pub const TH: usize = 36;
/// Tuning knob: a cell's mean grey (0–255) must move more than this to count as a change.
const CHANGE_THRESHOLD: u32 = 6;

pub fn write_record(out: &mut impl Write, kind: u8, ms: u32, jpeg: &[u8]) -> io::Result<()> {
    out.write_all(&[kind])?;
    out.write_all(&ms.to_le_bytes())?;
    out.write_all(&(jpeg.len() as u32).to_le_bytes())?;
    out.write_all(jpeg)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rec {
    pub kind: u8,
    pub ms: u32,
    pub offset: u64,
    pub len: u32,
}

/// Every complete record's header, in file order. A truncated last record
/// (crash mid-write) is dropped.
pub fn index(path: &Path) -> io::Result<Vec<Rec>> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    let (mut out, mut at, mut h) = (Vec::new(), 0u64, [0u8; HEADER as usize]);
    while at + HEADER <= size {
        f.seek(SeekFrom::Start(at))?;
        f.read_exact(&mut h)?;
        let len = u32::from_le_bytes([h[5], h[6], h[7], h[8]]);
        let offset = at + HEADER;
        if offset + len as u64 > size {
            break;
        }
        out.push(Rec { kind: h[0], ms: u32::from_le_bytes([h[1], h[2], h[3], h[4]]), offset, len });
        at = offset + len as u64;
    }
    Ok(out)
}

/// The last record of `kind` at or before `ms`. Each kind has one writer, so
/// its records are in time order.
// ponytail: linear scan from the end (~6k camera records per 10 min, ≤ 10 calls/s); binary-search per kind if it shows up.
pub fn latest(idx: &[Rec], kind: u8, ms: u32) -> Option<Rec> {
    idx.iter().rev().find(|r| r.kind == kind && r.ms <= ms).copied()
}

pub fn read(path: &Path, r: Rec) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(r.offset))?;
    let mut buf = vec![0; r.len as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

/// Area-average resize of packed RGB; each output pixel averages its source rect.
pub fn resize(rgb: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(nw * nh * 3);
    for y in 0..nh {
        let (y0, y1) = (y * h / nh, ((y + 1) * h / nh).max(y * h / nh + 1));
        for x in 0..nw {
            let (x0, x1) = (x * w / nw, ((x + 1) * w / nw).max(x * w / nw + 1));
            let mut sum = [0u32; 3];
            for yy in y0..y1 {
                for p in rgb[(yy * w + x0) * 3..(yy * w + x1) * 3].chunks_exact(3) {
                    sum[0] += p[0] as u32;
                    sum[1] += p[1] as u32;
                    sum[2] += p[2] as u32;
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            out.extend(sum.map(|s| (s / n) as u8));
        }
    }
    out
}

/// Stored size of a w×h screen frame: at most MAX_W wide, aspect kept.
pub fn fit(w: usize, h: usize) -> (usize, usize) {
    if w <= MAX_W {
        (w, h)
    } else {
        (MAX_W, (h * MAX_W / w).max(1))
    }
}

/// Grey TW×TH thumbnail for change detection.
pub fn thumb(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    resize(rgb, w, h, TW, TH).chunks_exact(3).map(|p| ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8).collect()
}

/// True if any 8×6 cell's mean grey moved more than CHANGE_THRESHOLD. Cells
/// rather than the whole image, so a new bullet point counts but a moving
/// cursor doesn't.
pub fn changed(a: &[u8], b: &[u8]) -> bool {
    (0..TH / 6).any(|cy| {
        (0..TW / 8).any(|cx| {
            let (mut sa, mut sb) = (0u32, 0u32);
            for y in cy * 6..cy * 6 + 6 {
                for x in cx * 8..cx * 8 + 8 {
                    sa += a[y * TW + x] as u32;
                    sb += b[y * TW + x] as u32;
                }
            }
            sa.abs_diff(sb) > CHANGE_THRESHOLD * 48
        })
    })
}

fn jpeg(rgb: &[u8], w: usize, h: usize, quality: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = jpeg_encoder::Encoder::new(&mut buf, quality).encode(rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb);
    buf
}

/// Writes one session's records. Times come from the caller (the session clock).
pub struct Recorder {
    out: BufWriter<File>,
    thumb: Option<Vec<u8>>,
}

impl Recorder {
    pub fn create(path: &Path) -> io::Result<Self> {
        Ok(Recorder { out: BufWriter::new(File::create(path)?), thumb: None })
    }

    /// Saves the frame if it differs from the last saved one (the first always
    /// saves). Returns whether it saved.
    // ponytail: full-resolution area average every sample (~10 ms for 4K in release); subsample the thumbnail if the capture thread shows up in profiles.
    pub fn screen(&mut self, ms: u32, w: usize, h: usize, rgb: &[u8]) -> io::Result<bool> {
        let t = thumb(rgb, w, h);
        if self.thumb.as_ref().is_some_and(|old| !changed(old, &t)) {
            return Ok(false);
        }
        let (nw, nh) = fit(w, h);
        let data = if nw == w { jpeg(rgb, w, h, 70) } else { jpeg(&resize(rgb, w, h, nw, nh), nw, nh, 70) };
        write_record(&mut self.out, SCREEN, ms, &data)?;
        self.thumb = Some(t);
        Ok(true)
    }

    /// Whether a screen frame has been saved yet.
    pub fn has_screen(&self) -> bool {
        self.thumb.is_some()
    }

    pub fn screen_ended(&mut self, ms: u32) -> io::Result<()> {
        write_record(&mut self.out, SCREEN, ms, &[])
    }

    /// A CAM_W×CAM_H frame, stored at half size; None marks the camera turned off.
    pub fn camera(&mut self, ms: u32, rgb: Option<&[u8]>) -> io::Result<()> {
        let data = rgb.map(|p| jpeg(&resize(p, CAM_W, CAM_H, CAM_W / 2, CAM_H / 2), CAM_W / 2, CAM_H / 2, 60)).unwrap_or_default();
        write_record(&mut self.out, CAMERA, ms, &data)
    }

    pub fn finish(mut self) -> io::Result<()> {
        self.out.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("speech-frames-{}-{name}", std::process::id()))
    }

    #[test]
    fn index_finds_the_last_record_at_or_before() {
        let p = tmp("index");
        let mut out = Vec::new();
        write_record(&mut out, SCREEN, 0, b"a").unwrap();
        write_record(&mut out, CAMERA, 50, b"bb").unwrap();
        write_record(&mut out, SCREEN, 1000, b"ccc").unwrap();
        write_record(&mut out, CAMERA, 1100, b"").unwrap(); // camera off
        fs::write(&p, &out).unwrap();
        let idx = index(&p).unwrap();
        assert_eq!(idx.len(), 4);
        let at = |kind, ms| latest(&idx, kind, ms).map(|r| read(&p, r).unwrap());
        assert_eq!(at(SCREEN, 999), Some(b"a".to_vec()));
        assert_eq!(at(SCREEN, 1000), Some(b"ccc".to_vec()));
        assert_eq!(at(CAMERA, 10), None); // before the first camera record
        assert_eq!(at(CAMERA, 60), Some(b"bb".to_vec()));
        assert_eq!(at(CAMERA, 5000), Some(vec![])); // off marker: UI hides the bubble
        fs::remove_file(p).unwrap();
    }

    #[test]
    fn index_ignores_a_truncated_last_record() {
        let p = tmp("trunc");
        let mut out = Vec::new();
        write_record(&mut out, SCREEN, 0, b"whole").unwrap();
        write_record(&mut out, SCREEN, 250, &[7; 100]).unwrap();
        out.truncate(out.len() - 90); // crashed mid-write
        fs::write(&p, &out).unwrap();
        assert_eq!(index(&p).unwrap().len(), 1);
        fs::remove_file(p).unwrap();
    }

    #[test]
    fn fits_wide_odd_and_portrait_displays() {
        assert_eq!(fit(3841, 2161), (1280, 720));
        assert_eq!(fit(1080, 1920), (1080, 1920));
        assert_eq!(fit(1280, 800), (1280, 800));
        let rgb = vec![200u8; 3841 * 2161 * 3];
        let (nw, nh) = fit(3841, 2161);
        let small = resize(&rgb, 3841, 2161, nw, nh);
        assert_eq!(small.len(), nw * nh * 3);
        assert!(small.iter().all(|&v| v == 200)); // area average of a flat image is flat
    }

    #[test]
    fn change_detection_sees_one_cell_and_ignores_faint_noise() {
        let base = vec![100u8; TW * TH];
        assert!(!changed(&base, &base));
        let mut one_cell = base.clone();
        for y in 6..12 {
            for x in 8..16 {
                one_cell[y * TW + x] = 110; // a whole cell +10 > threshold 6
            }
        }
        assert!(changed(&base, &one_cell));
        let faint: Vec<u8> = base.iter().map(|v| v + 3).collect(); // everywhere +3 < 6
        assert!(!changed(&base, &faint));
    }

    #[test]
    fn recorder_saves_first_and_changed_frames_only() {
        let p = tmp("rec");
        let (w, h) = (320, 180);
        let grey = vec![90u8; w * h * 3];
        let white = vec![250u8; w * h * 3];
        let mut r = Recorder::create(&p).unwrap();
        assert!(r.screen(0, w, h, &grey).unwrap()); // first frame always
        assert!(!r.screen(250, w, h, &grey).unwrap()); // identical
        assert!(r.screen(500, w, h, &white).unwrap()); // changed
        r.camera(600, Some(&vec![0u8; CAM_W * CAM_H * 3])).unwrap();
        r.camera(700, None).unwrap();
        r.screen_ended(800).unwrap();
        r.finish().unwrap();
        let idx = index(&p).unwrap();
        let kinds: Vec<(u8, u32, bool)> = idx.iter().map(|r| (r.kind, r.ms, r.len > 0)).collect();
        assert_eq!(kinds, vec![(SCREEN, 0, true), (SCREEN, 500, true), (CAMERA, 600, true), (CAMERA, 700, false), (SCREEN, 800, false)]);
        let jpeg = read(&p, idx[0]).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]); // JPEG SOI
        fs::remove_file(p).unwrap();
    }
}
