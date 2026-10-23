//! Report → Download: a session's audio (`sessions/<ts>.pcm`) as a WAV, or its
//! replay (`sessions/<ts>.frames`) composited the way replay.ts paints it —
//! screen letterboxed into 1280×720, camera bubble bottom-right — with that
//! audio, as a QuickTime movie. Both land in Downloads.
//!
//! The movie is Photo-JPEG video plus 16-bit PCM ('sowt'): every replay record
//! is already a JPEG and the audio is raw PCM, so no codec is needed, and
//! QuickTime Player, VLC and most editors open it. One video sample per record
//! time, each lasting until the next, so a still screen costs one frame.
use super::frames::{self, Rec, CAMERA, SCREEN};
use std::fs::{self, File};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter, Manager};

const W: usize = 1280;
const H: usize = 720;
const RATE: u32 = 16_000; // the .pcm's sample rate (audio.rs), mono i16 LE
const QUALITY: u8 = 75;

/// Saves session `ts` to Downloads as `name` (a file name, extension added):
/// the movie when `video` (and the session has a replay), else the audio.
/// Returns the saved file's name. Emits `export_progress` (0–100) while the
/// movie encodes.
#[tauri::command]
pub async fn export_session(app: AppHandle, ts: i64, video: bool, name: String) -> Result<String, String> {
    let dir = crate::history::detail_dir(&app)?;
    let downloads = app.path().download_dir().map_err(|e| e.to_string())?;
    // File name only: nothing outside Downloads.
    let stem = Path::new(&name).file_name().ok_or("bad file name")?.to_string_lossy().into_owned();
    tauri::async_runtime::spawn_blocking(move || {
        let pcm = fs::read(dir.join(format!("{ts}.pcm"))).map_err(|_| "Audio isn't kept for this session".to_string())?;
        let out = free_name(&downloads, &stem, if video { "mov" } else { "wav" });
        let part = out.with_extension("part");
        let res = if video {
            let idx = frames::index(&dir.join(format!("{ts}.frames"))).map_err(|_| "Video isn't kept for this session".to_string())?;
            let src = dir.join(format!("{ts}.frames"));
            write_mov(&part, &src, &idx, &pcm, |pct| {
                let _ = app.emit("export_progress", pct);
            })
        } else {
            File::create(&part).and_then(|f| write_wav(BufWriter::new(f), &pcm))
        };
        res.and_then(|_| fs::rename(&part, &out)).map_err(|e| {
            let _ = fs::remove_file(&part);
            e.to_string()
        })?;
        Ok(out.file_name().unwrap_or_default().to_string_lossy().into_owned())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// `dir/stem.ext`, or `stem (2).ext` and so on when that's taken.
fn free_name(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    (1..)
        .map(|n| dir.join(if n == 1 { format!("{stem}.{ext}") } else { format!("{stem} ({n}).{ext}") }))
        .find(|p| !p.exists())
        .unwrap()
}

fn write_wav(mut out: impl Write, pcm: &[u8]) -> io::Result<()> {
    let len = pcm.len() as u32;
    out.write_all(b"RIFF")?;
    out.write_all(&(36 + len).to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?; // PCM
    out.write_all(&1u16.to_le_bytes())?; // mono
    out.write_all(&RATE.to_le_bytes())?;
    out.write_all(&(RATE * 2).to_le_bytes())?; // bytes per second
    out.write_all(&2u16.to_le_bytes())?; // block align
    out.write_all(&16u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&len.to_le_bytes())?;
    out.write_all(pcm)?;
    out.flush()
}

// --- Compositing (mirrors replay.ts paint) --------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// An iw×ih image letterboxed into W×H (replay.ts fitRect).
fn fit_rect(iw: usize, ih: usize) -> Rect {
    let s = (W as f64 / iw as f64).min(H as f64 / ih as f64);
    let (w, h) = (((iw as f64 * s).round() as usize).clamp(1, W), ((ih as f64 * s).round() as usize).clamp(1, H));
    Rect { x: (W - w) / 2, y: (H - h) / 2, w, h }
}

/// The camera bubble (replay.ts bubbleRect).
fn bubble_rect() -> Rect {
    let w = (W as f64 * 0.22).round() as usize;
    let h = (w as f64 * 9.0 / 16.0).round() as usize;
    let m = (W as f64 * 0.02).round() as usize;
    Rect { x: W - w - m, y: H - h - m, w, h }
}

fn decode(jpeg: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg).ok()?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Some((img.into_raw(), w, h))
}

/// `rgb` (r.w×r.h) into the W×H canvas at r, outside corners of radius `radius` left as is.
fn blit(canvas: &mut [u8], rgb: &[u8], r: Rect, radius: f64) {
    for y in 0..r.h {
        for x in 0..r.w {
            if radius > 0.0 && !inside_rounded(x as f64 + 0.5, y as f64 + 0.5, r.w as f64, r.h as f64, radius) {
                continue;
            }
            let (s, d) = ((y * r.w + x) * 3, ((r.y + y) * W + r.x + x) * 3);
            canvas[d..d + 3].copy_from_slice(&rgb[s..s + 3]);
        }
    }
}

fn inside_rounded(x: f64, y: f64, w: f64, h: f64, r: f64) -> bool {
    let cx = x.clamp(r, w - r);
    let cy = y.clamp(r, h - r);
    (x - cx).powi(2) + (y - cy).powi(2) <= r * r
}

/// Video sample times: every record's ms (plus 0), each sample lasting until
/// the next; the last one runs to `end_ms`. Returns (start, duration) pairs.
fn sample_times(idx: &[Rec], end_ms: u32) -> Vec<(u32, u32)> {
    let mut at: Vec<u32> = idx.iter().map(|r| r.ms.min(end_ms)).chain([0]).collect();
    at.sort_unstable();
    at.dedup();
    let ends = at.iter().skip(1).copied().chain([end_ms]);
    at.iter().zip(ends).filter(|(a, b)| b > a).map(|(&a, b)| (a, b - a)).collect()
}

/// The movie: replay frames composited per sample time, plus the session audio.
fn write_mov(path: &Path, src: &Path, idx: &[Rec], pcm: &[u8], progress: impl Fn(u32)) -> io::Result<()> {
    let audio_ms = (pcm.len() / 2 * 1000 / RATE as usize) as u32;
    let end_ms = audio_ms.max(idx.iter().map(|r| r.ms + 100).max().unwrap_or(0));
    let times = sample_times(idx, end_ms);
    let ends_ms = idx.iter().find(|r| r.kind == SCREEN && r.len == 0).map(|r| r.ms);
    let bubble = bubble_rect();

    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(&atom(b"ftyp", &[b"qt  ".as_slice(), &0x2005_0300u32.to_be_bytes(), b"qt  "].concat()))?;
    let mdat_at = out.stream_position()?;
    out.write_all(&1u32.to_be_bytes())?; // size 1: a 64-bit size follows
    out.write_all(b"mdat")?;
    out.write_all(&0u64.to_be_bytes())?; // patched below

    let mut video: Vec<(u64, u32, u32)> = Vec::with_capacity(times.len()); // offset, size, duration
    let (mut screen_key, mut base) = (None::<Rec>, vec![0u8; W * H * 3]);
    let (mut cam_key, mut cam) = (None::<Rec>, None::<Vec<u8>>);
    let mut last_pct = 0;
    for (i, &(ms, dur)) in times.iter().enumerate() {
        let ended = ends_ms.is_some_and(|e| ms >= e);
        // Past the cut the screen keeps its last frame (the record there is empty).
        let screen = idx.iter().rev().find(|r| r.kind == SCREEN && r.ms <= ms && r.len > 0).copied();
        if screen != screen_key {
            base.fill(0);
            if let Some((rgb, w, h)) = screen.and_then(|r| frames::read(src, r).ok()).and_then(|j| decode(&j)) {
                let r = fit_rect(w, h);
                blit(&mut base, &frames::resize(&rgb, w, h, r.w, r.h), r, 0.0);
            }
            screen_key = screen;
        }
        let camera = frames::latest(idx, CAMERA, ms).filter(|_| !ended);
        if camera != cam_key {
            cam = camera
                .filter(|r| r.len > 0)
                .and_then(|r| frames::read(src, r).ok())
                .and_then(|j| decode(&j))
                .map(|(rgb, w, h)| frames::resize(&rgb, w, h, bubble.w, bubble.h));
            cam_key = camera;
        }
        let mut canvas = base.clone();
        if let Some(c) = &cam {
            blit(&mut canvas, c, bubble, bubble.h as f64 * 0.18);
        }
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, QUALITY)
            .encode(&canvas, W as u16, H as u16, jpeg_encoder::ColorType::Rgb)
            .map_err(io::Error::other)?;
        video.push((out.stream_position()?, jpeg.len() as u32, dur));
        out.write_all(&jpeg)?;
        let pct = (i * 100 / times.len()) as u32;
        if pct != last_pct {
            progress(pct);
            last_pct = pct;
        }
    }

    // Audio: one chunk per second, after the video.
    let mut chunks: Vec<(u64, u32)> = Vec::new(); // offset, samples
    for c in pcm[..pcm.len() / 2 * 2].chunks(RATE as usize * 2) {
        chunks.push((out.stream_position()?, (c.len() / 2) as u32));
        out.write_all(c)?;
    }
    let mdat_end = out.stream_position()?;
    out.seek(SeekFrom::Start(mdat_at + 8))?;
    out.write_all(&(mdat_end - mdat_at).to_be_bytes())?;
    out.seek(SeekFrom::Start(mdat_end))?;
    out.write_all(&moov(&video, &chunks, end_ms))?;
    out.flush()?;
    progress(100);
    Ok(())
}

// --- QuickTime atoms -------------------------------------------------------------

fn atom(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    [&(body.len() as u32 + 8).to_be_bytes(), kind.as_slice(), body].concat()
}

/// A full atom: version 0, then 24-bit flags.
fn full(kind: &[u8; 4], flags: u32, body: &[u8]) -> Vec<u8> {
    atom(kind, &[&(flags & 0xFF_FFFF).to_be_bytes(), body].concat())
}

const MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

fn be32(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_be_bytes()).collect()
}

fn moov(video: &[(u64, u32, u32)], audio: &[(u64, u32)], ms: u32) -> Vec<u8> {
    let mvhd = full(b"mvhd", 0, &[
        be32(&[0, 0, 1000, ms, 0x0001_0000]),
        vec![0x01, 0x00], // volume 1.0
        vec![0; 10],
        be32(&MATRIX),
        vec![0; 24], // preview, poster, selection, current time
        be32(&[3]), // next track id
    ].concat());

    // Video: Photo-JPEG, one sample per chunk, ms timescale.
    let mut stts: Vec<(u32, u32)> = Vec::new();
    for &(_, _, d) in video {
        match stts.last_mut() {
            Some((n, dd)) if *dd == d => *n += 1,
            _ => stts.push((1, d)),
        }
    }
    let jpeg_desc = [
        vec![0; 6],
        vec![0, 1], // data reference index
        be32(&[0, 0, 0, 0x200]), // version + revision, vendor, temporal quality, spatial quality
        vec![(W >> 8) as u8, W as u8, (H >> 8) as u8, H as u8],
        be32(&[72 << 16, 72 << 16, 0]), // resolution, data size
        vec![0, 1], // frames per sample
        pascal32("Photo - JPEG"),
        vec![0, 24, 0xFF, 0xFF], // depth, no color table
    ]
    .concat();
    let video_stbl = [
        full(b"stsd", 0, &[be32(&[1]), atom(b"jpeg", &jpeg_desc)].concat()),
        full(b"stts", 0, &[be32(&[stts.len() as u32]), stts.iter().flat_map(|&(n, d)| be32(&[n, d])).collect()].concat()),
        full(b"stsc", 0, &be32(&[1, 1, 1, 1])),
        full(b"stsz", 0, &[be32(&[0, video.len() as u32]), video.iter().flat_map(|v| v.1.to_be_bytes()).collect()].concat()),
        co64(video.iter().map(|v| v.0)),
    ]
    .concat();
    let video_trak = trak(1, ms, 0, (W as u32, H as u32), 1000, ms, b"vide", &full(b"vmhd", 1, &[0; 8]), &video_stbl);

    // Audio: 16 kHz mono 16-bit little-endian, one chunk per second.
    let samples: u32 = audio.iter().map(|c| c.1).sum();
    let sowt_desc = [vec![0; 6], vec![0, 1], be32(&[0, 0]), vec![0, 1, 0, 16, 0, 0, 0, 0], be32(&[RATE << 16])].concat();
    let mut stsc = vec![(1u32, audio.first().map_or(0, |c| c.1))];
    if let (Some(last), true) = (audio.last(), audio.len() > 1) {
        if last.1 != stsc[0].1 {
            stsc.push((audio.len() as u32, last.1));
        }
    }
    let audio_stbl = [
        full(b"stsd", 0, &[be32(&[1]), atom(b"sowt", &sowt_desc)].concat()),
        full(b"stts", 0, &be32(&[1, samples, 1])),
        full(b"stsc", 0, &[be32(&[stsc.len() as u32]), stsc.iter().flat_map(|&(f, n)| be32(&[f, n, 1])).collect()].concat()),
        full(b"stsz", 0, &be32(&[2, samples])),
        co64(audio.iter().map(|c| c.0)),
    ]
    .concat();
    let audio_ms = (samples as u64 * 1000 / RATE as u64) as u32;
    let audio_trak = trak(2, audio_ms, 0x0100, (0, 0), RATE, samples, b"soun", &full(b"smhd", 0, &[0; 4]), &audio_stbl);

    atom(b"moov", &[mvhd, video_trak, audio_trak].concat())
}

#[allow(clippy::too_many_arguments)]
fn trak(id: u32, ms: u32, volume: u16, (w, h): (u32, u32), timescale: u32, duration: u32, kind: &[u8; 4], media_header: &[u8], stbl: &[u8]) -> Vec<u8> {
    let tkhd = full(b"tkhd", 0x3, &[
        be32(&[0, 0, id, 0, ms, 0, 0]),
        vec![0, 0, 0, 0], // layer, alternate group
        volume.to_be_bytes().to_vec(),
        vec![0, 0],
        be32(&MATRIX),
        be32(&[w << 16, h << 16]),
    ].concat());
    let mdhd = full(b"mdhd", 0, &[be32(&[0, 0, timescale, duration]), vec![0, 0, 0, 0]].concat());
    let hdlr = |class: &[u8; 4], sub: &[u8; 4]| full(b"hdlr", 0, &[class.as_slice(), sub, &[0; 12], &[0]].concat());
    let dinf = atom(b"dinf", &full(b"dref", 0, &[be32(&[1]), full(b"url ", 1, &[])].concat()));
    let minf = atom(b"minf", &[media_header, &hdlr(b"dhlr", b"url "), &dinf, &atom(b"stbl", stbl)].concat());
    atom(b"trak", &[tkhd, atom(b"mdia", &[mdhd, hdlr(b"mhlr", kind), minf].concat())].concat())
}

fn co64(offsets: impl ExactSizeIterator<Item = u64>) -> Vec<u8> {
    let n = offsets.len() as u32;
    full(b"co64", 0, &[n.to_be_bytes().to_vec(), offsets.flat_map(|o| o.to_be_bytes()).collect()].concat())
}

/// A Pascal string padded to 32 bytes (the compressor name).
fn pascal32(s: &str) -> Vec<u8> {
    let mut v = vec![0u8; 32];
    v[0] = s.len() as u8;
    v[1..=s.len()].copy_from_slice(s.as_bytes());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rects_match_the_replay_canvas() {
        assert_eq!(fit_rect(1280, 800), Rect { x: 64, y: 0, w: 1152, h: 720 });
        assert_eq!(fit_rect(640, 360), Rect { x: 0, y: 0, w: 1280, h: 720 });
        assert_eq!(bubble_rect(), Rect { x: 972, y: 535, w: 282, h: 159 });
    }

    #[test]
    fn samples_run_until_the_next_record() {
        let rec = |kind, ms| Rec { kind, ms, offset: 0, len: 1 };
        let idx = [rec(SCREEN, 0), rec(CAMERA, 100), rec(CAMERA, 100), rec(SCREEN, 250)];
        assert_eq!(sample_times(&idx, 1000), vec![(0, 100), (100, 150), (250, 750)]);
        // Nothing at 0: the movie still starts there (black).
        assert_eq!(sample_times(&[rec(SCREEN, 40)], 100), vec![(0, 40), (40, 60)]);
    }

    #[test]
    fn writes_a_wav_header() {
        let mut out = Vec::new();
        write_wav(&mut out, &[1, 0, 2, 0]).unwrap();
        assert_eq!(out.len(), 48);
        assert_eq!(&out[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(out[4..8].try_into().unwrap()), 40);
        assert_eq!(u32::from_le_bytes(out[24..28].try_into().unwrap()), RATE);
        assert_eq!(&out[44..], &[1, 0, 2, 0]);
    }

    #[test]
    fn writes_a_movie() {
        let dir = std::env::temp_dir().join(format!("speech-export-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("s.frames");
        let mut rec = frames::Recorder::create(&src).unwrap();
        rec.screen(0, 320, 200, &vec![200u8; 320 * 200 * 3]).unwrap();
        rec.camera(100, Some(&vec![50u8; frames::CAM_W * frames::CAM_H * 3])).unwrap();
        rec.screen(300, 320, 200, &vec![20u8; 320 * 200 * 3]).unwrap();
        rec.finish().unwrap();
        let idx = frames::index(&src).unwrap();
        let pcm = vec![0u8; RATE as usize * 2 * 2]; // 2 s of silence
        let out = dir.join("s.mov");
        write_mov(&out, &src, &idx, &pcm, |_| {}).unwrap();
        let mov = fs::read(&out).unwrap();
        assert_eq!(&mov[4..8], b"ftyp");
        let mdat = u32::from_be_bytes(mov[..4].try_into().unwrap()) as usize; // right after ftyp
        assert_eq!(&mov[mdat + 4..mdat + 8], b"mdat");
        let mdat_len = u64::from_be_bytes(mov[mdat + 8..mdat + 16].try_into().unwrap()) as usize;
        assert_eq!(&mov[mdat + mdat_len + 4..mdat + mdat_len + 8], b"moov");
        assert_eq!(mdat + mdat_len + u32::from_be_bytes(mov[mdat + mdat_len..mdat + mdat_len + 4].try_into().unwrap()) as usize, mov.len());
        assert_eq!(&mov[mdat + 16..mdat + 18], &[0xFF, 0xD8]); // first frame is a JPEG
        fs::remove_dir_all(dir).unwrap();
    }
}
