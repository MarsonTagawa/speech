# Screen replay — design

Date: 2026-10-20
Status: approved in conversation, pending spec review

## Problem

Users rehearse presentations in other apps (PowerPoint, Keynote, Google Slides,
Zoom). Speech and presence grading already keep running while another app is in
front, because audio (`audio.rs`) and vision (`vision/`) live in Rust and aren't
throttled with the webview. What's missing is a way to *watch* the run
afterwards: the session report has the transcript and audio, but no picture of
what was on screen or how the speaker looked while saying it.

## Goals and success criteria

- A **Record screen** toggle. Sessions recorded with it on save a replay of one
  display, with the camera in a corner bubble when the camera is on.
- The report plays the replay **in sync with the existing session audio**:
  starting the whole-session replay or clicking a transcript line shows the
  frames for that moment. Video and audio drift ≤ 150 ms over a 10-minute session.
- Works on **Linux (Wayland via xdg-desktop-portal + PipeWire), macOS
  (ScreenCaptureKit) and Windows (Windows Graphics Capture)**.
- Storage: a 10-minute slide talk is ≤ 10 MB for the screen and ≤ 60 MB for the
  camera bubble.
- Recording the screen doesn't slow the live view or whisper. The capture thread
  stays under 10% of one core while slides are static.
- Local-only, like everything else: nothing leaves the machine.

## Non-goals

- Exporting MP4 or any shareable file, or muxing audio into the video.
- Grading from the screen (slide timing, OCR, matching speech to slide text).
- Capturing a single window instead of a display.
- Smooth full-motion video of the screen. Screen frames are change-driven and
  sampled at 4 Hz, which suits slides and most demos but makes fast motion choppy.

## Approach

Native capture in Rust, with change-driven JPEG frames written to one
file per session next to the session's audio. A canvas in the report draws the
frame for the audio's current time.

Rejected:
- **`getDisplayMedia` + `MediaRecorder` in the webview.** WKWebView doesn't
  expose `getDisplayMedia`, WebKitGTK support depends on version and settings, and
  WebKitGTK throttles hidden pages, which is the normal state while presenting
  elsewhere.
- **Periodic screenshots.** On Wayland each screenshot goes through the portal:
  it's slow and may prompt.
- **A real video codec (openh264 + MP4).** Only needed for export, which is a
  non-goal. JPEG frames reuse `jpeg-encoder`, which we already depend on.

### Capture path

Settled by the Task 1 spike (2026-10-21). Neither cross-platform crate works
on Wayland as published:
- `scap` 0.0.8 times the portal picker out after 10 s, panics, and leaves a
  stuck picker that blocks later requests.
- `xcap` 0.9.8 calls Start before the user's pick registers ("Sources not
  selected"), and its PipeWire format request didn't match Hyprland's offer.

So `screen/stream.rs` has two paths behind one interface:
- **Linux:** the ScreenCast portal via `ashpd` (it waits for the pick), then
  the portal's PipeWire stream via `pipewire`. It accepts any 4-byte RGB layout
  at any size and frame rate. Verified on Hyprland: 1920×1080 frames, and
  stopping closes the session in ~0.3 s.
- **macOS/Windows:** `xcap`'s recorder (ScreenCaptureKit / WGC). Unverified here.

Hyprland only sends frames when the screen changes, so a static slide yields
no new frames, which is exactly what the change-driven recorder wants.

## Components

### `src-tauri/src/screen.rs` (new)

State: `ScreenState(Mutex<Option<Capture>>)`, managed by Tauri like `RecordingState`.

- `screen_prepare(display: Option<String>) -> Result<(), String>`
  - Called when the user turns the toggle on.
  - Opens the capture stream and asks for permission: the portal display picker
    on Linux, the one-time Screen Recording permission on macOS, nothing on
    Windows. `display` picks a display on macOS/Windows, defaulting to the
    primary; Linux ignores it because the portal picker chooses.
  - The stream stays open while the toggle is on, so later sessions don't
    re-prompt. Frames are read and dropped until a session starts.
  - Errors (denied, picker cancelled, unsupported) come back as strings, and the
    UI turns the toggle off.
- `screen_displays() -> Vec<{id, name}>`: for the macOS/Windows display menu.
  Returns an empty list on Linux.
- `screen_release()`: toggle off. Closes the stream.
- Session hooks, called from `audio::start_recording` / `stop_recording` (no new
  frontend calls):
  - `begin(ts)` opens `sessions/<ts>.frames` and sets the session clock (below).
  - `end()` flushes and closes the file, and records whether capture ended early.

**Capture thread.** For each frame from the crate, at most every 250 ms (4 Hz):
1. Downscale to a 64×36 grayscale thumbnail.
2. Compare it with the last *saved* thumbnail, one 8×6 cell at a time (an 8×6 grid). Save the
   frame if any cell's mean differs by more than `CHANGE_THRESHOLD`
   (start at 6/255; it's a tuning knob).
3. To save, scale the frame to at most 1280 px wide, JPEG quality 70, and append
   a screen record.

The first frame of a session is always saved, so seeking to 0 shows something.

**Camera bubble.** When a session is being recorded and the camera is on,
`vision/capture.rs` passes every third RGB frame it already decodes (~10 fps)
to `screen::camera_frame(rgb)`. The writer thread downsizes it to 320×180
(a 2×2 box filter), encodes JPEG at quality 60 and appends a camera record.
That avoids decoding the preview JPEG, which would need a new dependency. If the camera is
off, nothing is written and the replay has no bubble.
The camera thread must never block on the file: hand off through a bounded
channel (capacity 4, drop on full) to the writer thread.

**Session clock.** Record times are milliseconds since the first audio
sample of the session, because playback is driven by the PCM (16 samples/ms).
`audio.rs` sets a shared `Instant` when its processing loop receives the first
audio chunk. Frames that arrive before that are kept with ms = 0.
`// ponytail:` the device's buffering latency (tens of ms) isn't compensated.
Add an offset if replays look out of sync.

**File format** (`sessions/<ts>.frames`), records back to back, little-endian:

```
[kind: u8 (0 = screen, 1 = camera)][ms: u32][len: u32][jpeg: len bytes]
```

A zero-length record means "off from here": for screen, the stream ended (its ms is `endsMs`); for camera, the camera turned off (the replay hides the bubble). A trailing partial record (crash mid-write) is ignored on read.

- `session_frame(ts, kind, ms) -> Response`: the JPEG of the last record of
  `kind` with time ≤ `ms`, or an empty body if there's none.
  - The index (kind, ms, offset, len) comes from walking the record headers with
    seeks. It's cached for the most recently opened `ts` so seeking during
    playback doesn't rescan.
  - Error `"Video isn't kept for this session"` if the file is missing (same
    wording as `session_clip`).
- `session_has_frames(ts) -> {screen: bool, camera: bool, endsMs: u32}`: the
  report uses this to decide whether to show the replay panel and where "video
  ends" falls.

### `history.rs`

No code change: `prune_details` already deletes every `sessions/<id>.*` outside
the keep set, so `.frames` files follow the `.pcm` rule (newest 5 plus starred).
Update the module doc comment to mention `.frames`.

### `audio.rs`

- `start_recording` calls `screen::begin(ts)` when a capture is prepared.
- `stop_recording` calls `screen::end()`.
- The processing loop marks the session clock on the first chunk.

### Frontend (`main.ts`, `index.html`, `styles.css`)

- **Toggle.** A "Record screen" toggle beside Record.
  - On macOS/Windows it opens a small display menu (`screen_displays`) before
    `screen_prepare`.
  - While it's on, a small red "screen" badge shows next to the timer.
  - The last chosen display persists in localStorage, but the stream
    doesn't: after a restart the toggle starts off.
- **Report replay panel.** Shown when `session_has_frames` says screen frames exist.
  - A canvas draws the screen frame letterboxed, with the camera frame as a
    rounded bubble in the bottom-right (~22% of the width).
  - The clock is the whole-session audio replay already in the Transcript panel:
    while it plays, a rAF loop asks for frames at the audio's current position
    (screen ≤ 4 Hz, camera ≤ 10 Hz, skipping a request while one is in flight).
    Clicking a transcript line seeks both.
  - Past `endsMs`, the canvas keeps the last frame and shows "Video ended here".
  - Paused, the panel shows the frame at the current position. It works the
    same in History for any session whose `.frames` is kept.

## Errors

| Situation | Behaviour |
|---|---|
| Permission denied / picker cancelled | `screen_prepare` errs; toggle flips off; message via `appendError` |
| Crate unsupported (old macOS, no portal) | Same as above, with "Screen recording isn't available on this system — …" |
| Stream ends mid-session (stop sharing in the portal, display unplugged) | Session keeps going audio-only. The frames so far are kept and `endsMs` marks the cut. Toggle flips off with a message |
| Disk write fails | Log, stop writing frames for the session, keep recording audio |
| `.frames` missing at replay | No replay panel (pruned or never recorded) |

## Testing

- **Rust unit tests in `screen.rs`:**
  - Write → index → `session_frame` for screen and camera records at given times,
    including before the first record (empty), exactly on a record, and after the last.
  - A truncated trailing record is ignored.
  - The change detector: identical thumbnails aren't saved; a change confined to
    one 8×8 cell above the threshold is saved; uniform noise below the threshold
    isn't.
- **Clock check.** Record a 2-minute session while a clock is shown on screen.
  Click transcript lines near the start, middle and end, and check that the
  replayed frame matches the spoken time within 150 ms.
- **Manual, Linux (this machine):** portal picker, a 10-minute slide session
  (check file size against the goals), stopping sharing mid-session,
  camera on and off.
- **macOS and Windows:** compiled and run on those machines by the user. I can't
  exercise them here, so they're unverified until then.
- **Build hygiene:** test with `CARGO_TARGET_DIR` in the scratchpad so the user's
  running `tauri dev` isn't disturbed.
