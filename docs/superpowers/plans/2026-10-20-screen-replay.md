# Screen Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With a "Screen" toggle on, each session saves a replay of one display, with the camera in a corner bubble. The report plays the replay in sync with the session's saved audio.

**Architecture:**
- A capture crate (`scap`) streams the display into a Rust capture thread. The thread samples at 4 Hz, keeps only frames that changed, and appends them as JPEG records to `sessions/<ts>.frames`.
- The camera thread hands every third RGB frame to a writer thread, which appends 320×180 JPEGs to the same file.
- Record times count from the session's first audio chunk, the same zero as the `.pcm`.
- The report's canvas asks Rust for "the last frame at or before ms" while the existing audio replay plays.

**Tech Stack:** Rust (tauri 2, `scap` 0.0.8, `jpeg-encoder` 0.7), TypeScript/Vite, bun for `*.check.ts`.

**Spec:** `docs/superpowers/specs/2026-10-20-screen-replay-design.md`

## Global Constraints

- Commit messages: short, lowercase, repo style (`screen: …`). **No `Co-Authored-By` or any Claude attribution** (user preference).
- Never build in the shared `src-tauri/target`: the user runs `tauri dev` alongside, and our builds break its linking. Every cargo command in this plan is:
  `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo <cmd> --manifest-path src-tauri/Cargo.toml …`
  where `$SCRATCH` is your session scratchpad directory. Run from the repo root `/home/marsont/speech`.
- The sandbox has no network. The first `cargo` call after adding `scap` must run with the sandbox disabled, so it can fetch crates.
- Only commit files this plan touches. The working tree has unrelated uncommitted work (deck, gestures, google, drive). Never `git add -A`, `git add .` or `git commit -a`.
- Local-only: nothing leaves the machine.
- Limits from the spec:
  - screen sampled at most every 250 ms;
  - screen JPEG ≤ 1280 px wide at quality 70;
  - camera JPEG 320×180 at quality 60, every 3rd camera frame (~10 fps);
  - change threshold 6/255 on a cell mean (a tuning knob);
  - camera handoff queue capacity 4, dropped when full;
  - sync ≤ 150 ms.
- Error copy:
  - `"Video isn't kept for this session"` (missing `.frames`);
  - `"Screen recording isn't available — …"` (prepare failed);
  - `"Screen recording stopped — …"` (stream ended);
  - `"Video ended here"` (replay past the cut).
- **Two clarifications of the spec, made here:**
  1. The thumbnail is 64×36, compared in 8×6-pixel cells (an 8×6 grid), because 36 isn't divisible by 8.
  2. A **zero-length record** means "off from here": for screen, the stream ended (this is `endsMs`); for camera, the camera was turned off.

  The persisted preference is the last chosen display id. The toggle itself always starts off.
- Linux is the only platform exercised here. macOS and Windows must compile, which is the crate's job, but are verified by the user on those machines.

## Review Focus

1. **The user stops sharing from the portal or Hyprland mid-session.** Expected: audio keeps recording, a zero-length screen record marks `endsMs`, the toggle turns off and the UI says "Screen recording stopped". Task 3 pins the end marker with a `Recorder` test, and Task 6 checks it manually.
2. **Camera turned off (or on) mid-session.** Expected: no frozen stale bubble. After "off" the bubble disappears; after "on" it comes back. Task 2 pins that `latest` returns the zero-length camera record (so the UI hides the bubble), and Task 3 makes `vision::stop_running` send it.
3. **Odd display sizes: 4K, odd dimensions, portrait monitors, padded row strides.** Expected: frames scale to ≤ 1280 wide with the aspect ratio kept and no panic. Task 2 tests `resize`/`fit` on 3841×2161 and 1080×1920; Task 1 tests `pack` with a padded stride.
4. **The stream dies while idle between sessions (permission revoked, display unplugged).** Expected: the next session starts normally, audio-only. `start_recording` never fails because of the screen. Task 3 tests that `begin` with no live capture writes no file.
5. **Switching between two sessions' replays, or a `.frames` pruned after its index was cached.** Expected: each replay shows its own frames, and a missing file gives "Video isn't kept for this session" instead of stale frames. Task 3 tests `with_index` alternating between two files and after deleting one.

---

### Task 1: Capture crate spike and stream adapter

Proves `scap` on this machine. This is the go/no-go gate: if the probe can't get frames on Hyprland, stop and report back instead of continuing.

**Files:**
- Modify: `flake.nix` (buildInputs)
- Modify: `src-tauri/Cargo.toml` (`scap` dependency)
- Create: `src-tauri/src/screen/stream.rs`
- Create: `src-tauri/src/screen/mod.rs` (just `pub mod stream;` for now)
- Modify: `src-tauri/src/lib.rs` (`mod screen;`, plus a `pub mod screen_probe` re-export for the example)
- Create: `src-tauri/examples/screen_probe.rs`

**Interfaces:**
- Produces, in `screen::stream`:
  - `pub struct Frame { pub w: usize, pub h: usize, pub rgb: Vec<u8> }` (packed RGB, `w*h*3` bytes)
  - `pub struct Display { pub id: String, pub name: String }` (Serialize)
  - `pub struct Stream`, with `pub fn next(&self, timeout: Duration) -> Result<Option<Frame>, String>`. `Ok(None)` means a timeout; `Err` means the stream ended.
  - `pub fn open(display: Option<String>) -> Result<Stream, String>`. It blocks until the OS picker or permission resolves.
  - `pub fn displays() -> Vec<Display>`, which is empty on Linux.
  - `fn pack(w: usize, h: usize, data: &[u8], order: [usize; 3]) -> Vec<u8>`: 4-byte pixels to RGB; the stride is `data.len() / h`.

- [ ] **Step 1: Add the system libraries and the crate**

In `flake.nix`, add to `buildInputs` (after `dbus`):

```nix
            # Screen replay (scap): PipeWire screencast via xdg-desktop-portal
            pipewire
```

In `src-tauri/Cargo.toml` under `[dependencies]`, after `jpeg-encoder`:

```toml
# Screen replay (screen/): one API over PipeWire portal, ScreenCaptureKit and WGC.
scap = "0.0.8"
```

- [ ] **Step 2: Write the failing test for `pack`**

Create `src-tauri/src/screen/mod.rs`:

```rust
//! Screen replay: records one display (plus the camera when it's on) during a
//! session as change-driven JPEG frames in `sessions/<ts>.frames`, replayed in
//! the report in sync with the session audio. See
//! docs/superpowers/specs/2026-10-20-screen-replay-design.md.
pub mod stream;
```

Create `src-tauri/src/screen/stream.rs` with only the test and a stub:

```rust
//! The capture crate behind one small interface: open a display, pull RGB
//! frames. Everything crate-specific lives in this file.

fn pack(_w: usize, _h: usize, _data: &[u8], _order: [usize; 3]) -> Vec<u8> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn packs_bgra_with_padded_rows() {
        // 2×2 BGRA, rows padded to 12 bytes (stride 12, 4 bytes junk each row).
        let data = [
            3, 2, 1, 255, 6, 5, 4, 255, 9, 9, 9, 9, //
            12, 11, 10, 255, 15, 14, 13, 255, 9, 9, 9, 9,
        ];
        assert_eq!(super::pack(2, 2, &data, [2, 1, 0]), vec![1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]);
    }
}
```

In `src-tauri/src/lib.rs`, add `mod screen;` after `mod screenshot;`.

- [ ] **Step 3: Run it and check that it fails**

Run (sandbox disabled, because this first build fetches `scap`): `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo test --manifest-path src-tauri/Cargo.toml packs_bgra`

Expected: FAIL (panics with `not implemented`).

If the *build* fails in `scap`'s dependencies instead (missing `libspa`, `xcb`, `dbus` and so on), add the named nixpkgs package to `buildInputs` in `flake.nix` and retry. Record each one you add in the commit message.

- [ ] **Step 4: Implement the adapter**

Replace `src-tauri/src/screen/stream.rs` with:

```rust
//! The capture crate behind one small interface: open a display, pull RGB
//! frames. Everything crate-specific lives in this file.
//!
//! The crate's capturer is built, read and stopped on one pump thread (its
//! types aren't Send on every platform), which forwards frames over a
//! one-slot channel so a busy consumer drops frames instead of queueing them.
use scap::capturer::{Capturer, Options};
use scap::frame::{Frame as ScapFrame, FrameType};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

#[derive(Serialize)]
pub struct Display {
    pub id: String,
    pub name: String,
}

pub struct Stream {
    rx: Receiver<Result<Frame, String>>,
    stop: Arc<AtomicBool>,
}

impl Stream {
    /// `Ok(None)` on timeout; `Err` once the stream has ended.
    pub fn next(&self, timeout: Duration) -> Result<Option<Frame>, String> {
        match self.rx.recv_timeout(timeout) {
            Ok(r) => r.map(Some),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err("the screen share ended".into()),
        }
    }
}

// ponytail: the pump only sees `stop` after its next frame; on a fully static
// screen that can lag. Fine because nothing waits on it; add a crate-level
// interrupt if the OS "sharing" indicator lingers noticeably after toggling off.
impl Drop for Stream {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
    }
}

/// Opens `display` (an id from `displays`; None = the OS default or, on
/// Linux, the portal picker). Blocks until the user picks or permission resolves.
pub fn open(display: Option<String>) -> Result<Stream, String> {
    let (tx, rx) = mpsc::sync_channel(1);
    let (opened_tx, opened) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let pump_stop = stop.clone();
    std::thread::spawn(move || {
        let mut cap = match build(display) {
            Ok(c) => {
                let _ = opened_tx.send(Ok(()));
                c
            }
            Err(e) => {
                let _ = opened_tx.send(Err(e));
                return;
            }
        };
        cap.start_capture();
        while !pump_stop.load(Relaxed) {
            match cap.get_next_frame() {
                Ok(f) => {
                    if let Some(f) = to_rgb(f) {
                        if let Err(mpsc::TrySendError::Disconnected(_)) = tx.try_send(Ok(f)) {
                            break;
                        }
                    }
                }
                Err(_) => {
                    let _ = tx.send(Err("the screen share ended".into()));
                    break;
                }
            }
        }
        cap.stop_capture();
    });
    opened.recv().map_err(|_| "screen capture failed to start".to_string())??;
    Ok(Stream { rx, stop })
}

fn build(display: Option<String>) -> Result<Capturer, String> {
    if !scap::is_supported() {
        return Err("this system has no supported screen capture".into());
    }
    if !scap::has_permission() && !scap::request_permission() {
        return Err("permission to record the screen was denied".into());
    }
    let target = display.and_then(|id| {
        scap::get_all_targets().into_iter().find(|t| matches!(t, scap::Target::Display(d) if d.id.to_string() == id))
    });
    let options = Options { fps: 4, target, show_cursor: true, show_highlight: false, output_type: FrameType::BGRAFrame, ..Default::default() };
    Capturer::build(options).map_err(|e| format!("{e:?}"))
}

/// Displays to choose from on macOS/Windows. Linux's portal has its own picker.
pub fn displays() -> Vec<Display> {
    if cfg!(target_os = "linux") {
        return vec![];
    }
    scap::get_all_targets()
        .into_iter()
        .filter_map(|t| match t {
            scap::Target::Display(d) => Some(Display { id: d.id.to_string(), name: d.title }),
            _ => None,
        })
        .collect()
}

fn to_rgb(f: ScapFrame) -> Option<Frame> {
    let (w, h, data, order) = match f {
        ScapFrame::BGRA(f) => (f.width, f.height, f.data, [2, 1, 0]),
        ScapFrame::BGRx(f) => (f.width, f.height, f.data, [2, 1, 0]),
        ScapFrame::BGR0(f) => (f.width, f.height, f.data, [2, 1, 0]),
        ScapFrame::RGBx(f) => (f.width, f.height, f.data, [0, 1, 2]),
        ScapFrame::XBGR(f) => (f.width, f.height, f.data, [3, 2, 1]),
        _ => return None,
    };
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 || data.len() < w * h * 4 {
        return None;
    }
    Some(Frame { w, h, rgb: pack(w, h, &data, order) })
}

/// 4-byte pixels → packed RGB, taking channels at `order`. Rows may be padded:
/// the stride is `data.len() / h`.
fn pack(w: usize, h: usize, data: &[u8], order: [usize; 3]) -> Vec<u8> {
    let stride = data.len() / h;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in data.chunks_exact(stride).take(h) {
        for p in row[..w * 4].chunks_exact(4) {
            rgb.extend([p[order[0]], p[order[1]], p[order[2]]]);
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    #[test]
    fn packs_bgra_with_padded_rows() {
        // 2×2 BGRA, rows padded to 12 bytes (stride 12, 4 bytes junk each row).
        let data = [
            3, 2, 1, 255, 6, 5, 4, 255, 9, 9, 9, 9, //
            12, 11, 10, 255, 15, 14, 13, 255, 9, 9, 9, 9,
        ];
        assert_eq!(super::pack(2, 2, &data, [2, 1, 0]), vec![1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]);
    }
}
```

**If this doesn't compile against the resolved `scap` version:** read the crate source (`find ~/.cargo/registry/src -maxdepth 1 -name 'scap-*'`) and fix only `build`, `displays` and `to_rgb` in this file to match its real names (the `Frame` variants, `Options` fields, `Capturer::build` vs `new`, `Target::Display` fields). Keep the public interface above exactly as it is. List what changed in the commit message.

- [ ] **Step 5: Run the test and check that it passes**

Run: `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo test --manifest-path src-tauri/Cargo.toml packs_bgra`

Expected: PASS.

- [ ] **Step 6: Write the probe**

In `src-tauri/src/lib.rs`, next to the existing `pub mod vision_probe`, add:

```rust
/// For `examples/screen_probe.rs` only.
pub mod screen_probe {
    pub use crate::screen::stream;
}
```

and change `mod screen;` to `pub(crate) mod screen;` if the re-export complains about visibility.

Create `src-tauri/examples/screen_probe.rs`:

```rust
//! Manual check of screen capture: opens the default display (the portal
//! picker on Linux), reads frames for ~10 s, then reports size, rate and how
//! fast dropping the stream ends it.
use speech_lib::screen_probe::stream;
use std::time::{Duration, Instant};

fn main() {
    let t = Instant::now();
    let s = stream::open(None).expect("open");
    println!("opened in {:?}", t.elapsed());
    let (start, mut n, mut size) = (Instant::now(), 0, (0, 0));
    while start.elapsed() < Duration::from_secs(10) {
        if let Some(f) = s.next(Duration::from_millis(500)).expect("stream") {
            n += 1;
            size = (f.w, f.h);
        }
    }
    println!("{n} frames in 10 s ({:.1} fps), {}×{}", n as f64 / 10.0, size.0, size.1);
    let t = Instant::now();
    drop(s);
    println!("dropped in {:?}", t.elapsed());
}
```

(Check that the library crate's name is `speech_lib` with `grep -n '^name' src-tauri/Cargo.toml` under `[lib]`, and use whatever it is.)

- [ ] **Step 7: Run the probe on Hyprland**

Run, outside the sandbox: `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo run --manifest-path src-tauri/Cargo.toml --example screen_probe`

The user picks a display in the portal dialog, so ask them to watch for it. Move the mouse during the 10 s.

Expected: `opened`, then **≥ 3.5 fps** at the display's native size, then `dropped`.

**Gate:** if `open` errors or 0 frames arrive, run it again with `xcap`'s equivalent API. If that fails too, **stop and report to the user**, as the spec requires.

- [ ] **Step 8: Commit**

```bash
git add flake.nix src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/src/screen/ src-tauri/examples/screen_probe.rs src-tauri/src/lib.rs
git commit -m "screen: capture adapter over scap, probe example"
```

Caution: `flake.nix`, `Cargo.toml`, `Cargo.lock` and `lib.rs` already have unrelated uncommitted changes. Stage only your hunks with `git add -p <file>`. That isn't interactive-blocked, but if the harness can't drive it, write a patch with `git diff <file> > $SCRATCH/x.patch`, cut it down to your hunks, then `git apply --cached $SCRATCH/x.patch`. For `Cargo.lock`, stage the whole file only if `git diff --stat src-tauri/Cargo.lock` shows that nothing else changed before your edit. Otherwise leave the lock out and say so in the commit message.

---

### Task 2: Frames file, resize and change detection (pure)

**Files:**
- Create: `src-tauri/src/screen/frames.rs`
- Modify: `src-tauri/src/screen/mod.rs` (add `mod frames;`)

**Interfaces:**
- Produces, in `screen::frames`:
  - `pub const SCREEN: u8 = 0; pub const CAMERA: u8 = 1;`
  - `pub struct Rec { pub kind: u8, pub ms: u32, pub offset: u64, pub len: u32 }` (Copy, Debug, PartialEq)
  - `pub fn index(path: &Path) -> io::Result<Vec<Rec>>`
  - `pub fn latest(idx: &[Rec], kind: u8, ms: u32) -> Option<Rec>`
  - `pub fn read(path: &Path, r: Rec) -> io::Result<Vec<u8>>`
  - `pub fn resize(rgb: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8>`
  - `pub fn fit(w: usize, h: usize) -> (usize, usize)` (≤ 1280 wide, aspect kept)
  - `pub fn thumb(rgb: &[u8], w: usize, h: usize) -> Vec<u8>`
  - `pub fn changed(a: &[u8], b: &[u8]) -> bool`
  - `pub struct Recorder`, with:
    - `create(path) -> io::Result<Self>`
    - `screen(&mut self, ms, w, h, rgb) -> io::Result<bool>`
    - `screen_ended(&mut self, ms) -> io::Result<()>`
    - `camera(&mut self, ms, rgb: Option<&[u8]>) -> io::Result<()>` (640×360 in)
    - `finish(self) -> io::Result<()>`

- [ ] **Step 1: Write the failing tests**

Create `src-tauri/src/screen/frames.rs` with these tests at the bottom. Above them, add the signatures from **Interfaces** with `unimplemented!()` bodies, so the tests compile and fail:

```rust
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
```

Add `mod frames;` to `src-tauri/src/screen/mod.rs`.

- [ ] **Step 2: Run them and check that they fail**

Run: `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo test --manifest-path src-tauri/Cargo.toml screen::frames`

Expected: 5 FAIL (`not implemented`).

- [ ] **Step 3: Implement**

Put this above the tests in `frames.rs`, replacing the stubs:

```rust
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
```

- [ ] **Step 4: Run them and check that they pass**

Run: `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo test --manifest-path src-tauri/Cargo.toml screen::frames`

Expected: 5 PASS. Ignore `dead_code` warnings until Task 3 uses these items.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/screen/frames.rs src-tauri/src/screen/mod.rs
git commit -m "screen: frames file, resize and change detection"
```

---

### Task 3: Capture thread, session hooks and commands

**Files:**
- Modify: `src-tauri/src/screen/mod.rs`
- Modify: `src-tauri/src/audio.rs` (`start_recording` ~line 191, `stop_recording` ~line 210, `run_processing_loop` top of its `while` loop ~line 575)
- Modify: `src-tauri/src/vision/capture.rs` (`run` loop)
- Modify: `src-tauri/src/vision/mod.rs` (`stop_running`, ~line 153)
- Modify: `src-tauri/src/lib.rs` (manage the state, register commands, release on close)
- Modify: `src-tauri/src/history.rs` (module doc comment only)

**Interfaces:**
- Consumes: `stream::{open, displays, Stream, Frame, Display}` (Task 1); `frames::{Recorder, index, latest, read, Rec, SCREEN, CAMERA}` (Task 2); `crate::history::detail_dir(&AppHandle) -> Result<PathBuf, String>`.
- Produces:
  - Tauri commands:
    - `screen_prepare(display: Option<String>) -> Result<(), String>` (async)
    - `screen_release()`
    - `screen_displays() -> Vec<Display>`
    - `session_frame(ts: i64, kind: u8, ms: u32) -> Response` (raw JPEG; an empty body if there's no frame or it's an off marker)
    - `session_has_frames(ts: i64) -> Option<{ screen: bool, camera: bool, endsMs: Option<u32> }>`
  - The event `"screen_ended"` with payload `String` (the reason).
  - Hooks for Rust callers: `screen::begin(&AppHandle, i64)`, `screen::end()`, `screen::mark_clock()`, `screen::camera_frame(n: u32, rgb: &[u8])` (keeps every 3rd n), `screen::camera_off()`.
  - `pub struct ScreenState`.

- [ ] **Step 1: Write the failing tests**

Append to `src-tauri/src/screen/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("speech-screen-{}-{name}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn one_record(path: &std::path::Path, body: &[u8]) {
        let mut out = Vec::new();
        frames::write_record(&mut out, frames::SCREEN, 0, body).unwrap();
        fs::write(path, out).unwrap();
    }

    // Review focus 5: switching replays never shows the other session's frames,
    // and a pruned file reports missing instead of serving the stale index.
    #[test]
    fn index_cache_follows_the_session_and_notices_pruning() {
        let d = dir("cache");
        one_record(&d.join("1.frames"), b"one");
        one_record(&d.join("2.frames"), b"two");
        assert_eq!(frame_at(&d, 1, frames::SCREEN, 0).unwrap(), b"one");
        assert_eq!(frame_at(&d, 2, frames::SCREEN, 0).unwrap(), b"two");
        assert_eq!(frame_at(&d, 1, frames::SCREEN, 0).unwrap(), b"one");
        fs::remove_file(d.join("1.frames")).unwrap();
        assert_eq!(frame_at(&d, 1, frames::SCREEN, 0).unwrap_err(), "Video isn't kept for this session");
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn summary_reports_kinds_and_the_cut() {
        let d = dir("summary");
        let mut out = Vec::new();
        frames::write_record(&mut out, frames::SCREEN, 0, b"s").unwrap();
        frames::write_record(&mut out, frames::SCREEN, 4200, b"").unwrap(); // stream ended
        fs::write(d.join("7.frames"), out).unwrap();
        let s = summary(&d, 7).unwrap();
        assert!(s.screen && !s.camera);
        assert_eq!(s.ends_ms, Some(4200));
        assert!(summary(&d, 8).is_none()); // never recorded / pruned → no panel
        fs::remove_dir_all(d).unwrap();
    }

    // Review focus 4: no live capture → begin writes nothing, so a session can't
    // fail or leave an empty replay because of the screen.
    #[test]
    fn begin_without_capture_writes_no_file() {
        let d = dir("nocap");
        assert!(!start_session(&d, 5, false).unwrap());
        assert!(!d.join("5.frames").exists());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn session_records_on_the_audio_clock_and_marks_the_end() {
        let d = dir("clock");
        assert!(start_session(&d, 9, true).unwrap());
        let rgb = vec![128u8; 64 * 36 * 3];
        screen_sample(64, 36, &rgb); // before the first audio chunk → ms 0
        mark_clock();
        std::thread::sleep(std::time::Duration::from_millis(30));
        stream_ended();
        end();
        let idx = frames::index(&d.join("9.frames")).unwrap();
        assert_eq!(idx[0].ms, 0);
        assert!(idx[0].len > 0);
        assert_eq!(idx.last().unwrap().len, 0);
        assert!(idx.last().unwrap().ms >= 30);
        fs::remove_dir_all(d).unwrap();
    }
}
```

These tests share the `SESSION` and `INDEX` statics. Make them run serially by adding a `static TEST_LOCK: Mutex<()> = Mutex::new(());` in the test module and `let _g = TEST_LOCK.lock().unwrap();` as the first line of each test that touches `SESSION` (the last two).

- [ ] **Step 2: Run them and check that they fail**

Run: `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo test --manifest-path src-tauri/Cargo.toml screen::tests`

Expected: compile errors (`frame_at`, `summary`, `start_session`, `screen_sample`, `mark_clock`, `stream_ended` and `end` aren't defined).

- [ ] **Step 3: Implement `screen/mod.rs`**

Replace everything above the tests in `src-tauri/src/screen/mod.rs` with:

```rust
//! Screen replay: records one display (plus the camera when it's on) during a
//! session as change-driven JPEG frames in `sessions/<ts>.frames`, replayed in
//! the report in sync with the session audio. See
//! docs/superpowers/specs/2026-10-20-screen-replay-design.md.
//!
//! While the toggle is on, a capture thread reads the display (screen_prepare
//! opens it once, so later sessions don't re-prompt). During a session it
//! samples every SAMPLE and saves frames that changed. The camera thread hands
//! frames to a writer thread through a small queue, so it never waits on disk.
mod frames;
pub mod stream;

use frames::{Rec, Recorder, CAMERA, SCREEN};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tauri::{ipc::Response, AppHandle, Emitter, Manager};

const SAMPLE: Duration = Duration::from_millis(250);
const CAMERA_QUEUE: usize = 4;
const CAMERA_EVERY: u32 = 3; // of the camera's ~30 fps

struct Session {
    rec: Recorder,
    clock: Option<Instant>, // the session's first audio chunk; None until then
}

impl Session {
    fn ms(&self) -> u32 {
        self.clock.map_or(0, |c| c.elapsed().as_millis() as u32)
    }
}

static SESSION: Mutex<Option<Session>> = Mutex::new(None);
/// Separate from SESSION so the camera thread never waits on a frame being encoded.
static CAMERA_TX: Mutex<Option<SyncSender<Option<Vec<u8>>>>> = Mutex::new(None);
/// The replay index of the most recently viewed session.
static INDEX: Mutex<Option<(i64, Vec<Rec>)>> = Mutex::new(None);

struct Capture {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

#[derive(Default)]
pub struct ScreenState(Mutex<Option<Capture>>);

impl ScreenState {
    pub fn release(&self) {
        if let Some(c) = self.0.lock().unwrap().take() {
            c.stop.store(true, Relaxed); // the thread exits within one `next` timeout
        }
    }
    fn live(&self) -> bool {
        self.0.lock().unwrap().as_ref().is_some_and(|c| !c.thread.is_finished())
    }
}

// --- Session hooks (audio.rs, vision) -------------------------------------------

/// Session `ts` starts (audio::start_recording). Records only while a capture
/// is live; a failure is reported to the UI, never to the recording.
pub fn begin(app: &AppHandle, ts: i64) {
    let live = app.state::<ScreenState>().live();
    if let Err(e) = crate::history::detail_dir(app).and_then(|d| start_session(&d, ts, live)) {
        let _ = app.emit("screen_ended", format!("couldn't save the video — {e}"));
    }
}

/// Opens `<dir>/<ts>.frames` and the camera writer when `live`. Returns whether it did.
fn start_session(dir: &Path, ts: i64, live: bool) -> Result<bool, String> {
    if !live {
        return Ok(false);
    }
    let rec = Recorder::create(&dir.join(format!("{ts}.frames"))).map_err(|e| e.to_string())?;
    *SESSION.lock().unwrap() = Some(Session { rec, clock: None });
    let (tx, rx) = mpsc::sync_channel::<Option<Vec<u8>>>(CAMERA_QUEUE);
    thread::spawn(move || {
        for frame in rx {
            if let Some(s) = SESSION.lock().unwrap().as_mut() {
                let ms = s.ms();
                if let Err(e) = s.rec.camera(ms, frame.as_deref()) {
                    eprintln!("[screen] camera frame: {e}");
                }
            }
        }
    });
    *CAMERA_TX.lock().unwrap() = Some(tx);
    Ok(true)
}

/// The session's first audio chunk arrived: ms 0 of its `.pcm`. Idempotent.
pub fn mark_clock() {
    if let Some(s) = SESSION.lock().unwrap().as_mut() {
        s.clock.get_or_insert_with(Instant::now);
    }
}

/// Session stops (audio::stop_recording).
pub fn end() {
    CAMERA_TX.lock().unwrap().take(); // the writer thread drains and exits
    if let Some(s) = SESSION.lock().unwrap().take() {
        if let Err(e) = s.rec.finish() {
            eprintln!("[screen] finishing the video: {e}");
        }
    }
}

/// Every camera frame (vision::capture); keeps every CAMERA_EVERY-th during a session.
pub fn camera_frame(n: u32, rgb: &[u8]) {
    if n % CAMERA_EVERY != 0 {
        return;
    }
    if let Some(tx) = CAMERA_TX.lock().unwrap().as_ref() {
        let _ = tx.try_send(Some(rgb.to_vec())); // full → drop the frame
    }
}

/// The camera turned off (vision::stop_running): the replay hides the bubble from here.
pub fn camera_off() {
    if let Some(tx) = CAMERA_TX.lock().unwrap().as_ref() {
        let _ = tx.send(None); // must not be dropped like a frame
    }
}

fn screen_sample(w: usize, h: usize, rgb: &[u8]) {
    let mut g = SESSION.lock().unwrap();
    if let Some(s) = g.as_mut() {
        let ms = s.ms();
        if let Err(e) = s.rec.screen(ms, w, h, rgb) {
            eprintln!("[screen] writing a frame failed, video stops here: {e}");
            *g = None; // keep recording audio; drop the replay's tail
        }
    }
}

fn stream_ended() {
    if let Some(s) = SESSION.lock().unwrap().as_mut() {
        let ms = s.ms();
        let _ = s.rec.screen_ended(ms);
    }
}

fn run(app: AppHandle, stream: stream::Stream, stop: Arc<AtomicBool>) {
    let mut last: Option<Instant> = None;
    while !stop.load(Relaxed) {
        match stream.next(Duration::from_millis(500)) {
            Ok(Some(f)) if last.is_none_or(|t| t.elapsed() >= SAMPLE) => {
                last = Some(Instant::now());
                screen_sample(f.w, f.h, &f.rgb);
            }
            Ok(_) => {}
            Err(e) => {
                stream_ended();
                let _ = app.emit("screen_ended", e);
                return;
            }
        }
    }
}

// --- Commands -------------------------------------------------------------------

/// Toggle on: asks for the display/permission and starts reading it. async so
/// the picker doesn't block the main thread.
#[tauri::command]
pub async fn screen_prepare(app: AppHandle, display: Option<String>) -> Result<(), String> {
    app.state::<ScreenState>().release();
    let stream = tauri::async_runtime::spawn_blocking(move || stream::open(display)).await.map_err(|e| e.to_string())??;
    let stop = Arc::new(AtomicBool::new(false));
    let thread = {
        let (app, stop) = (app.clone(), stop.clone());
        thread::spawn(move || run(app, stream, stop))
    };
    *app.state::<ScreenState>().0.lock().unwrap() = Some(Capture { stop, thread });
    Ok(())
}

#[tauri::command]
pub fn screen_release(app: AppHandle) {
    app.state::<ScreenState>().release();
}

#[tauri::command]
pub fn screen_displays() -> Vec<stream::Display> {
    stream::displays()
}

fn with_index<T>(dir: &Path, ts: i64, f: impl FnOnce(&Path, &[Rec]) -> T) -> Result<T, String> {
    let path = dir.join(format!("{ts}.frames"));
    let missing = || "Video isn't kept for this session".to_string();
    if !path.exists() {
        return Err(missing());
    }
    let mut g = INDEX.lock().unwrap();
    if g.as_ref().is_none_or(|(t, _)| *t != ts) {
        *g = Some((ts, frames::index(&path).map_err(|_| missing())?));
    }
    Ok(f(&path, &g.as_ref().unwrap().1))
}

fn frame_at(dir: &Path, ts: i64, kind: u8, ms: u32) -> Result<Vec<u8>, String> {
    with_index(dir, ts, |path, idx| frames::latest(idx, kind, ms).map_or(Ok(vec![]), |r| frames::read(path, r)))?
        .map_err(|_| "Video isn't kept for this session".to_string())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
    screen: bool,
    camera: bool,
    ends_ms: Option<u32>,
}

fn summary(dir: &Path, ts: i64) -> Option<Summary> {
    with_index(dir, ts, |_, idx| Summary {
        screen: idx.iter().any(|r| r.kind == SCREEN && r.len > 0),
        camera: idx.iter().any(|r| r.kind == CAMERA && r.len > 0),
        ends_ms: idx.iter().find(|r| r.kind == SCREEN && r.len == 0).map(|r| r.ms),
    })
    .ok()
}

/// The JPEG of session `ts`'s last `kind` frame at or before `ms`; empty when
/// there's none or it's an "off" marker.
#[tauri::command]
pub fn session_frame(app: AppHandle, ts: i64, kind: u8, ms: u32) -> Result<Response, String> {
    Ok(Response::new(frame_at(&crate::history::detail_dir(&app)?, ts, kind, ms)?))
}

/// What session `ts`'s replay holds; None if it has no `.frames`.
#[tauri::command]
pub fn session_has_frames(app: AppHandle, ts: i64) -> Option<Summary> {
    summary(&crate::history::detail_dir(&app).ok()?, ts)
}
```

`is_none_or` needs Rust ≥ 1.82. If `nix develop -c rustc --version` is older, write `!matches!(…, Some(x) if …)` instead.

- [ ] **Step 4: Run them and check that they pass**

Run: `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo test --manifest-path src-tauri/Cargo.toml screen::`

Expected: all PASS (Task 2's 5, these 4, and Task 1's 1).

- [ ] **Step 5: Wire up the hooks**

`src-tauri/src/audio.rs`, `start_recording`: just before `let capture_thread = std::thread::spawn(...)`, add:

```rust
    // Screen replay (screen.rs): opens sessions/<ts>.frames if the toggle is on.
    crate::screen::begin(&app, ts);
```

`stop_recording`: after the `processing_thread` join, before `Ok(())`:

```rust
    crate::screen::end();
```

`run_processing_loop`: as the first statement inside `while let Ok(chunk) = audio_rx.recv() {`:

```rust
        if pcm.is_empty() {
            crate::screen::mark_clock(); // the replay's ms 0 is the .pcm's
        }
```

`src-tauri/src/vision/capture.rs`, `run`: before the `while`, add `let mut n = 0u32;`. In the `Ok(img)` arm, after `let rgb = img.into_raw();`, add:

```rust
                n = n.wrapping_add(1);
                crate::screen::camera_frame(n, &rgb); // replay's camera bubble
```

`src-tauri/src/vision/mod.rs`, `stop_running`: inside `if let Some(r) = slot.take() {`, after the join loop:

```rust
        crate::screen::camera_off();
```

`src-tauri/src/lib.rs`:
- After `app.manage(vision::VisionState::default());`, add `app.manage(screen::ScreenState::default());`.
- In the `CloseRequested` arm, add `window.state::<screen::ScreenState>().release();`.
- Add to `generate_handler!`: `screen::screen_prepare, screen::screen_release, screen::screen_displays, screen::session_frame, screen::session_has_frames,`.

`src-tauri/src/history.rs`, module doc: after the sentence about `sessions/<ts>.pcm`, add ` and, when the screen was recorded, its replay (sessions/<ts>.frames, screen/)`. `prune_details` already deletes any `<id>.*`.

- [ ] **Step 6: Build and test everything**

Run:
- `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo test --manifest-path src-tauri/Cargo.toml`
- `nix develop -c env CARGO_TARGET_DIR=$SCRATCH/target cargo clippy --manifest-path src-tauri/Cargo.toml`

Expected: all tests pass, and no new clippy warnings in `screen/`, `audio.rs`, `vision/` or `lib.rs`.

- [ ] **Step 7: Commit**

```bash
git add src-tauri/src/screen/mod.rs src-tauri/src/history.rs
git add -p src-tauri/src/audio.rs src-tauri/src/vision/capture.rs src-tauri/src/vision/mod.rs src-tauri/src/lib.rs
git commit -m "screen: capture thread, session hooks and replay commands"
```

`vision/*` and `lib.rs` carry unrelated uncommitted changes, so stage only your hunks (see the Task 1 Step 8 note).

---

### Task 4: Screen toggle in the UI

**Files:**
- Modify: `index.html` (button after `#rec-label` ~line 94; badge after `#clock` ~line 66; dialog next to `#google-dialog` ~line 385)
- Modify: `src/main.ts` (new block near `setRecordingUi` ~line 1579; listener and button wiring near the `video-btn` wiring ~line 5007)
- Modify: `src/styles.css`

**Interfaces:**
- Consumes the commands `screen_displays`, `screen_prepare({ display })`, `screen_release` and the event `screen_ended` (Task 3).
- Produces nothing used by later tasks.

- [ ] **Step 1: Markup**

`index.html`, after `<span id="rec-label" class="rec-label">REC</span>`:

```html
          <button id="screen-btn" class="tool-btn" type="button" aria-pressed="false" data-tip="Record screen — each session also saves a replay of your screen, with your camera in a corner">Screen</button>
```

After the `#clock` div:

```html
            <span id="screen-badge" class="screen-badge" hidden data-tip="Recording the screen with each session">● SCREEN</span>
```

Next to `#google-dialog`:

```html
    <dialog id="screen-dialog" class="modal" aria-labelledby="sd-title">
      <div class="panel">
        <span id="sd-title" class="label">Record which display?</span>
        <div id="screen-displays" class="sd-list"></div>
        <button type="button" class="ctl">Cancel</button>
      </div>
    </dialog>
```

- [ ] **Step 2: Logic**

`src/main.ts`, after `setRecordingUi`:

```ts
// Record screen (screen/ in Rust): while on, each session also saves a replay
// of one display. Turning it on asks the OS — the portal's picker on Linux, a
// display menu here on macOS/Windows. The capture stays open while on, so
// sessions don't re-prompt; the choice of display is remembered, the toggle isn't.
const SCREEN_DISPLAY_KEY = "speech.screenDisplay";
function setScreenUi(on: boolean) {
  $("screen-btn")?.setAttribute("aria-pressed", String(on));
  $("screen-badge")?.toggleAttribute("hidden", !on);
}

// The display to record: an id, null for the OS's own choice, undefined if cancelled.
async function pickDisplay(): Promise<string | null | undefined> {
  const list = await invoke<{ id: string; name: string }[]>("screen_displays");
  if (list.length < 2) return list[0]?.id ?? null;
  const dlg = $<HTMLDialogElement>("screen-dialog");
  const box = $("screen-displays");
  if (!dlg || !box) return null;
  const last = load(SCREEN_DISPLAY_KEY);
  box.innerHTML = list
    .map((d) => `<button type="button" class="ctl${d.id === last ? " primary" : ""}" data-id="${escapeHtml(d.id)}">${escapeHtml(d.name)}</button>`)
    .join("");
  return new Promise((resolve) => {
    dlg.onclick = (e) => {
      const b = (e.target as Element).closest<HTMLButtonElement>("button");
      if (!b) return;
      dlg.close();
      resolve(b.dataset.id); // Cancel has no id → undefined
    };
    dlg.oncancel = () => resolve(undefined);
    dlg.showModal();
  });
}

async function toggleScreen() {
  const btn = $<HTMLButtonElement>("screen-btn");
  if (!btn || recording) return;
  if (btn.getAttribute("aria-pressed") === "true") {
    setScreenUi(false);
    return void invoke("screen_release");
  }
  const display = await pickDisplay();
  if (display === undefined) return;
  if (display) save(SCREEN_DISPLAY_KEY, display);
  btn.disabled = true;
  try {
    await invoke("screen_prepare", { display });
    setScreenUi(true);
  } catch (e) {
    appendError(`Screen recording isn't available — ${e}`);
  } finally {
    btn.disabled = false;
  }
}
```

In `setRecordingUi`, add a line so the toggle can't change mid-session (a session's replay has one source):

```ts
  const screenBtn = $<HTMLButtonElement>("screen-btn");
  if (screenBtn) screenBtn.disabled = on;
```

Near the `video-btn`/`camera-btn` click wiring (~line 5007):

```ts
  $("screen-btn")?.addEventListener("click", () => void toggleScreen());
  void listen<string>("screen_ended", (e) => {
    setScreenUi(false);
    appendError(`Screen recording stopped — ${e.payload}`);
  });
```

- [ ] **Step 3: Styles**

Append to `src/styles.css`, reusing the existing tokens (check `--` variables near the `.rec-label.on` rule and use the same red):

```css
/* Record screen: badge by the clock while the toggle is on. */
.screen-badge { font-size: 10px; letter-spacing: 0.08em; color: var(--rec, #e5484d); margin-left: 8px; }
#screen-btn[aria-pressed="true"] { color: var(--rec, #e5484d); }
.sd-list { display: flex; flex-direction: column; gap: 6px; margin: 10px 0; }
```

If `--rec` doesn't exist, use the variable that `.rec-label.on` uses.

- [ ] **Step 4: Typecheck**

Run: `nix develop -c bunx tsc --noEmit`

Expected: no errors.

- [ ] **Step 5: Manual check**

Ask the user to restart `tauri dev`, since Rust changed. In the app:
- click **Screen**: the portal picker appears; pick a display; the button turns red and the badge shows;
- click it again: it's off and the badge is hidden;
- cancel the picker: an error appears and the toggle stays off;
- start recording: the Screen button is disabled.

- [ ] **Step 6: Commit**

```bash
git add -p index.html src/main.ts src/styles.css
git commit -m "screen: record-screen toggle, display menu and stop notice"
```

---

### Task 5: Replay panel in the report

**Files:**
- Create: `src/replay.ts`
- Create: `src/replay.check.ts`
- Modify: `src/main.ts` (`transcriptCopyHtml` ~line 2776, `renderReport` ~line 2720, the session popup ~line 4268, the history page ~line 4278, `playClip` ~line 1756)
- Modify: `src/styles.css`

**Interfaces:**
- Consumes the commands `session_has_frames({ ts })` → `{ screen, camera, endsMs } | null` and `session_frame({ ts, kind, ms })` → `ArrayBuffer` (Task 3).
- Produces, in `src/replay.ts`:
  - `showReplay(canvas: HTMLCanvasElement, ts: number): Promise<void>`
  - `followClip(ts: number, startMs: number, ctx: AudioContext): number` (returns a token)
  - `stopFollow(token: number): void`
  - `fitRect(iw, ih, cw, ch)`, `bubbleRect(cw, ch)`, `due(last, ms, period)` (pure)

- [ ] **Step 1: Write the failing check**

Create `src/replay.check.ts`:

```ts
// Run: bun src/replay.check.ts
import assert from "node:assert";
import { fitRect, bubbleRect, due } from "./replay";

// 16:9 into 16:9 fills; 4:3 into 16:9 pillarboxes; portrait letterboxes centred.
assert.deepEqual(fitRect(1280, 720, 640, 360), { x: 0, y: 0, w: 640, h: 360 });
assert.deepEqual(fitRect(1024, 768, 640, 360), { x: 80, y: 0, w: 480, h: 360 });
assert.deepEqual(fitRect(1080, 1920, 640, 360), { x: 219, y: 0, w: 203, h: 360 });
// Bubble: 22% wide, 16:9, inset 2% from the bottom-right corner.
assert.deepEqual(bubbleRect(1000, 562), { x: 760, y: 418, w: 220, h: 124 });
// Fetch when never fetched, when far enough from the last fetch, or after seeking back.
assert.equal(due(-1, 0, 250), true);
assert.equal(due(1000, 1100, 250), false);
assert.equal(due(1000, 1250, 250), true);
assert.equal(due(5000, 1000, 250), true);
console.log("replay ok");
```

- [ ] **Step 2: Run it and check that it fails**

Run: `bun src/replay.check.ts`

Expected: FAIL (cannot find module `./replay`).

- [ ] **Step 3: Implement `src/replay.ts`**

```ts
// Screen replay in the report: the session's saved screen frames, with the
// camera as a bubble, drawn for the moment the session audio is playing
// (screen/ in Rust records them; session_frame returns the last frame at or
// before a time). Without playback the canvas holds the last frame drawn.
import { invoke } from "@tauri-apps/api/core";

const SCREEN = 0;
const CAMERA = 1;
const SCREEN_EVERY = 250; // ms of session time between fetches; frames are ≥ 250 ms apart anyway
const CAMERA_EVERY = 100; // ~10 fps, as recorded

interface Rect { x: number; y: number; w: number; h: number }

// An iw×ih image letterboxed into cw×ch.
export function fitRect(iw: number, ih: number, cw: number, ch: number): Rect {
  const s = Math.min(cw / iw, ch / ih);
  const w = Math.round(iw * s);
  const h = Math.round(ih * s);
  return { x: Math.round((cw - w) / 2), y: Math.round((ch - h) / 2), w, h };
}

// The camera bubble: 22% of the width, 16:9, 2% in from the bottom-right.
export function bubbleRect(cw: number, ch: number): Rect {
  const w = Math.round(cw * 0.22);
  const h = Math.round((w * 9) / 16);
  const m = Math.round(cw * 0.02);
  return { x: cw - w - m, y: ch - h - m, w, h };
}

// Whether a frame fetched at session time `last` is stale at `ms` (or never fetched).
export const due = (last: number, ms: number, period: number) => last < 0 || ms < last || ms - last >= period;

interface View {
  ts: number;
  endsMs: number | null;
  screen: ImageBitmap | null;
  cam: ImageBitmap | null;
  screenAt: number;
  camAt: number;
  busy: boolean;
}
const views = new WeakMap<HTMLCanvasElement, View>();

async function frame(ts: number, kind: number, ms: number): Promise<ImageBitmap | null> {
  const buf = await invoke<ArrayBuffer>("session_frame", { ts, kind, ms: Math.max(0, Math.round(ms)) });
  return buf.byteLength ? createImageBitmap(new Blob([buf], { type: "image/jpeg" })) : null;
}

function paint(canvas: HTMLCanvasElement, v: View, ended: boolean) {
  const cx = canvas.getContext("2d");
  if (!cx) return;
  const { width: cw, height: ch } = canvas;
  cx.fillStyle = "#000";
  cx.fillRect(0, 0, cw, ch);
  if (v.screen) {
    const r = fitRect(v.screen.width, v.screen.height, cw, ch);
    cx.drawImage(v.screen, r.x, r.y, r.w, r.h);
  }
  if (v.cam && !ended) {
    const b = bubbleRect(cw, ch);
    cx.save();
    cx.beginPath();
    cx.roundRect(b.x, b.y, b.w, b.h, b.h * 0.18);
    cx.clip();
    cx.drawImage(v.cam, b.x, b.y, b.w, b.h);
    cx.restore();
  }
  if (ended) {
    cx.fillStyle = "rgba(0,0,0,0.55)";
    cx.fillRect(0, ch - 28, cw, 28);
    cx.fillStyle = "#fff";
    cx.font = "12px 'IBM Plex Mono', monospace";
    cx.fillText("Video ended here", 10, ch - 10);
  }
}

// Brings `canvas` to session time `ms`, fetching only what's stale. A fetch
// already in flight makes this a no-op (the next tick catches up).
async function drawAt(canvas: HTMLCanvasElement, ms: number) {
  const v = views.get(canvas);
  if (!v || v.busy) return;
  const ended = v.endsMs !== null && ms >= v.endsMs;
  v.busy = true;
  try {
    // Past the cut the screen keeps its last frame (the record there is empty).
    if (!ended && due(v.screenAt, ms, SCREEN_EVERY)) {
      v.screen = (await frame(v.ts, SCREEN, ms)) ?? v.screen;
      v.screenAt = ms;
    }
    if (due(v.camAt, ms, CAMERA_EVERY)) {
      v.cam = await frame(v.ts, CAMERA, ms); // empty = camera off → no bubble
      v.camAt = ms;
    }
  } catch {
    canvas.hidden = true; // pruned since it was shown
  } finally {
    v.busy = false;
  }
  paint(canvas, v, ended);
}

// Shows `canvas` with session `ts`'s first frame if the session has a replay.
export async function showReplay(canvas: HTMLCanvasElement, ts: number) {
  const f = await invoke<{ screen: boolean; camera: boolean; endsMs: number | null } | null>("session_has_frames", { ts }).catch(() => null);
  if (!f?.screen) return;
  canvas.width = 1280;
  canvas.height = 720;
  canvas.dataset.ts = String(ts);
  views.set(canvas, { ts, endsMs: f.endsMs, screen: null, cam: null, screenAt: -1, camAt: -1, busy: false });
  canvas.hidden = false;
  await drawAt(canvas, 0);
}

// Follows audio playback of session `ts` from `startMs` on `ctx`'s clock,
// drawing every shown replay canvas of that session. Returns a token for
// stopFollow; a newer follow replaces an older one.
let following = 0;
export function followClip(ts: number, startMs: number, ctx: AudioContext): number {
  const token = ++following;
  const t0 = ctx.currentTime;
  const tick = () => {
    if (token !== following) return;
    const ms = startMs + (ctx.currentTime - t0) * 1000;
    document.querySelectorAll<HTMLCanvasElement>(`canvas.rep-video[data-ts="${ts}"]:not([hidden])`).forEach((c) => void drawAt(c, ms));
    requestAnimationFrame(tick);
  };
  tick();
  return token;
}

export function stopFollow(token: number) {
  if (token === following) following++;
}
```

- [ ] **Step 4: Run the check and check that it passes**

Run: `bun src/replay.check.ts`

Expected: `replay ok`.

If bun fails to import `@tauri-apps/api/core`, move `fitRect`, `bubbleRect` and `due` into the check's import path unchanged (a `replayGeom.ts` that `replay.ts` re-exports), and say so in the commit message.

- [ ] **Step 5: Wire it into the report**

`src/main.ts`:

Import: `import { showReplay, followClip, stopFollow } from "./replay";`

`transcriptCopyHtml`: insert the canvas right after the panel-head `</div>`:

```ts
    `<div class="panel-head">…unchanged…</div>` +
    `<canvas class="rep-video" hidden aria-label="Screen replay"></canvas>` +
    lines.map(([, html]) => html).join("")
```

Add a helper next to it:

```ts
// Session ts's screen replay, if it was recorded, in the transcripts under `root`.
const replays = (root: ParentNode, ts: number) =>
  root.querySelectorAll<HTMLCanvasElement>("canvas.rep-video").forEach((c) => void showReplay(c, ts));
```

Call it after each place that renders `transcriptCopyHtml` into the DOM:
- `renderReport`: after `if (rt) rt.innerHTML = transcriptCopyHtml(d.lines);`, add `if (rt) replays(rt, d.ts);`. (`Detail` has `ts`; confirm with `grep -n "interface Detail" -A5 src/main.ts`.)
- The session popup (~line 4268): after `el.innerHTML = sessionPageHtml(i, popDetail, head);`, add `replays(el, history[i].ts);`.
- The history page (~line 4278): after `el.innerHTML = i < 0 ? historyListHtml() : historyPageHtml(i);`, add `if (i >= 0) replays(el, history[i].ts);`.

`playClip`: make the replay follow the audio:

```ts
async function playClip(ts: number, el: HTMLElement, startMs: number, endMs: number) {
  if (el.classList.contains("playing")) return void clipSrc?.stop();
  try {
    const buf = await invoke<ArrayBuffer>("session_clip", { ts, startMs: Math.round(startMs), endMs: Math.round(endMs) });
    el.classList.add("playing");
    let follow = 0;
    playSamples(new Float32Array(buf), () => {
      el.classList.remove("playing");
      stopFollow(follow);
    });
    follow = followClip(ts, startMs, micAudio!);
  } catch (e) {
    el.dataset.tip = String(e);
  }
}
```

(`playSamples` creates `micAudio` before returning, so the `!` holds. The old clip's `onended` fires asynchronously after `stop()`. It passes *its own* token, which `stopFollow` ignores once a newer follow exists.)

- [ ] **Step 6: Styles**

Append to `src/styles.css`:

```css
/* Screen replay above the report transcript (replay.ts draws it at 1280×720). */
.rep-video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 8px; margin: 8px 0 12px; }
```

- [ ] **Step 7: Typecheck and run the checks**

Run:
- `nix develop -c bunx tsc --noEmit`
- `bun src/replay.check.ts`

Expected: no type errors, and `replay ok`.

- [ ] **Step 8: Commit**

```bash
git add src/replay.ts src/replay.check.ts
git add -p src/main.ts src/styles.css
git commit -m "screen: replay panel in the report, synced to session audio"
```

---

### Task 6: End-to-end verification on Linux

No new code unless a check fails. If one does, fix it in the task that owns that code's file and recommit.

**Files:** none, except fixes.

- [ ] **Step 1: Clock sync (goal: ≤ 150 ms)**

1. With the user: turn on Screen, show a stopwatch with ms on the recorded display (`watch -n 0.1 date +%T.%N` in a terminal), and record a 2-minute session.
2. Read the time aloud at roughly 0:10, 1:00 and 1:50.
3. In the report, click those transcript lines. The replayed frame should show the spoken time within 150 ms.

If it's consistently off by a fixed amount, add a `CLOCK_OFFSET_MS` const in `screen/mod.rs` `Session::ms` with a `ponytail:` note.

- [ ] **Step 2: Size and load (goal: ≤ 10 MB screen, ≤ 60 MB camera per 10 min; < 10% core)**

1. Record 10 minutes of slides with the camera on.
2. `ls -l ~/.local/share/com.marsont.speech/sessions/*.frames`.
3. Get the split between kinds from a quick `index` over the file: add a temporary `#[ignore]` test, or use `examples/`. Don't commit it.
4. While recording, watch the process in `top -H`: the screen thread should stay under 10% (a release build is the honest figure: `bun tauri build --debug false`, or the user's call).

- [ ] **Step 3: Failure modes**

- Mid-session, stop sharing from Hyprland or the portal. Expected: the toggle turns off, "Screen recording stopped — …" appears, audio continues, and the replay shows "Video ended here" past that point.
- Turn the camera off mid-session. Expected: the bubble disappears from that time on in the replay.
- Record with the camera off throughout. Expected: no bubble.
- Turn Screen on, revoke it while idle (close the portal share), then record. Expected: the session records normally, audio-only, with no replay panel.
- Open History on a session older than the 5 kept. Expected: no replay panel.

- [ ] **Step 4: Report**

Tell the user the measured sync error, the sizes, the CPU figure and any fixes. macOS and Windows remain unverified until they're built on those machines.
