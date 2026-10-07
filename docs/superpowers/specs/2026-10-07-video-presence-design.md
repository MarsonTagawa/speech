# Video presence grading — design

Date: 2026-10-07
Status: approved in conversation, pending spec review

## Goal

Add an optional camera mode to a practice session that grades four on-camera
behaviours: **eye contact, facial expressiveness, head/body stillness, hand
gestures**. The result is one more scored dimension, **Presence**, in the live
view, the session report, the coach tips and history.

With the camera off, the app behaves exactly as it does today. Everything runs
locally. No video, frames or landmarks are saved, only summary numbers.

## Feasibility (spike, 2026-10-07)

Tested in the real Tauri/WebKitGTK window on this machine (AMD iGPU, webkitgtk
2.54, GStreamer 1.28.7):

- wry ignores WebKitGTK permission requests, so `getUserMedia` never resolves
  unless we enable media-stream and allow the request ourselves.
- Once the camera runs, the WebKitWebProcess segfaults after ~20s in
  `_dma_buf_upload_accept` (libgstgl). `WEBKIT_GST_DMABUF_SINK_DISABLED=1` fixes it.
- The camera ignores the `ideal` 640×480 request and delivers 1920×1080.
  Running MediaPipe on full frames gives about 6.5 fps. Downscaling to a
  640×360 canvas first gives **12–14 fps** with the GPU delegate
  (face + pose + hands every frame, ~55 ms total) and ~10 fps on CPU.
- Running tiny.en GPU decodes alongside didn't change MediaPipe fps. Whisper
  p50 went 54→61 ms and p95 83→107 ms, which is acceptable.
- Loading `@mediapipe/tasks-vision` from a CDN failed inside the webview.
  Local files work.

## Approach

MediaPipe Tasks Vision runs in the webview: `FaceLandmarker` (blendshapes and
the facial transformation matrix), `PoseLandmarker` lite, and `HandLandmarker`
(2 hands), all on the GPU delegate in `VIDEO` mode. A Rust-side `nokhwa` + `ort`
pipeline was rejected as 5–10× more code with no blendshapes.

## Components

### 1. Rust — `src-tauri/src/lib.rs`

- At the top of `run()`, before any webview exists, call
  `std::env::set_var("WEBKIT_GST_DMABUF_SINK_DISABLED", "1")` on Linux, with a
  comment pointing to the segfault.
- In `setup`, on Linux, `get_webview_window("main")` → `with_webview`:
  `settings.set_enable_media_stream(true)` and
  `connect_permission_request(|_, req| { req.allow(); true })`. The
  `webkit2gtk` crate is already a dependency (used by `screenshot.rs`).
- No new commands. Camera capture and inference stay in the webview.

### 2. Assets

- `bun add @mediapipe/tasks-vision`. Vite bundles the JS.
- `public/mediapipe/` (git-ignored) holds the WASM files
  (`vision_wasm_internal.{js,wasm}`, plus the nosimd pair as a fallback) and the
  three `.task` models: `face_landmarker.task`, `pose_landmarker_lite.task`,
  `hand_landmarker.task`, about 30 MB in total.
- Add the curl/cp steps to the README's "Models (required, not in git)"
  section. If the files are missing, camera enable fails with a clear
  message (see Error handling).

### 3. `src/presence.ts` (new)

Two parts:

**Capture loop (impure, small):**
- `startCamera(video: HTMLVideoElement): Promise<void>` calls `getUserMedia`
  with `{ width: {ideal: 640}, height: {ideal: 360}, frameRate: {ideal: 30} }`,
  then lazily creates the three landmarkers (once per app run, then reused).
- `stopCamera()` stops the tracks and the loop. The landmarkers stay loaded.
- The loop paces itself to about 12 fps. Each tick draws the video onto a
  640×360 canvas, runs the three detectors on that canvas, reduces the results
  to a `Frame`, and passes it to the accumulator **only while recording**. A
  `live` callback receives each `Frame` for the live cue whether or not we're
  recording.

**Pure scoring (unit-checked):**
- `Frame` holds `{ t, face: boolean, yaw, pitch, eyeLookMax, smile, brows,
  shoulderMid?: [x,y], shoulderW?, nose?: [x,y], wrists: [x,y][] }`.
- `toFrame(face, pose, hands, t): Frame` reduces the MediaPipe results.
  Yaw and pitch come from the facial transformation matrix.
- `summarize(frames: Frame[], opts: { scriptOpen: boolean }): Presence`
  returns the metrics and sub-scores below.
- `PRESENCE_TUNING` is a single const holding every threshold, marked with a
  `ponytail:` comment: initial guesses, to be tuned against real sessions.

### 4. Metrics and scoring

All are computed over the frames recorded during the session.

| Signal | Per-frame | Session metric | Sub-score |
|---|---|---|---|
| Eye contact | `face && |yaw| ≤ 15° && |pitch| ≤ 15° && eyeLookMax < 0.5` | `eyeContact` = fraction of frames (no face counts as not looking) | `scoreLinear(eyeContact, 0.8, 0.3)` |
| Expressiveness | `smile` = mean(mouthSmileLeft/Right); `brows` = max(browInnerUp, browOuterUpLeft/Right) | `smileFrac` = fraction of face frames with smile > 0.3; `exprRange` = mean over 1 s windows of the std-dev of (smile + brows) | mean of `scoreLinear(smileFrac, 0.25, 0)` and `scoreLinear(exprRange, 0.08, 0.01)` |
| Stillness | shoulder midpoint and nose, divided by shoulder width | `motion` = mean per-second path length of shoulder-mid + nose, in shoulder widths | `scoreLinear(motion, 0.15, 0.6)` |
| Gestures | some wrist moved > 0.02 (normalised) since the last frame | `gestureFrac` = fraction of frames gesturing | band: 100 for 0.2–0.6, falling linearly to 0 at 0 and to 70 at 1.0 |

`jawOpen` is not used, because speaking moves the jaw.

`presence` = mean of the four sub-scores. **If a script is open during the
session, eye contact is reported but left out of that mean**, because reading
from a screen below the camera means looking down.

A session needs at least 3 s of frames to be scored. Shorter sessions get
`presence: null`.

### 5. Integration into `main.ts`

- **Toggle:** a new camera `tool-btn icon` in the strip, next to `ribbon-btn`,
  with `aria-pressed`, persisted as `speech.camera` via the existing
  `load`/`save`. Turning it on calls `startCamera`, turning it off calls
  `stopCamera`. On launch, the camera only starts again if the key is on.
- **Self-view:** a mirrored `<video>` (~160×90, rounded) in the scope's
  top-right under the clock, shown only while the camera is on. No overlays.
- **Live cue:** a new dock card, **Presence**. Its big number is the session's
  eye-contact % so far. Its sub-line shows "look at the camera" after more than
  2 s of continuous looking away, and otherwise names the currently weakest
  other signal ("still", "use your hands", "smile more").
- **Scores:** `Scores` gains `presence: number | null`. `SCORE_WEIGHTS` gains
  `presence: 1.5`. `computeSummary` pushes it into the composite only when it
  isn't null, the same way `articulation` works.
- **Summary:** `SessionSummary` gains an optional `presence?: Presence`
  (metrics + sub-scores). Sessions saved before this field existed still load
  unchanged.
- **Live ring:** it already derives from the summary score, so it includes
  presence automatically.

### 6. Report, coach, history

- **Report tile:** `reportTilesHtml` adds a Presence tile when `s.presence` is
  set. It shows the score, the change from the previous session (same mechanism
  as the other tiles), and four rows: eye contact %, smile %, movement
  (low/ok/high), gesturing %.
- **Tips:** `generateTips` gets up to four presence candidates with
  `impact = SCORE_WEIGHTS.presence × (100 − subScore)`, each firing only below
  a threshold (eye contact < 60%, expressiveness sub-score < 60, motion > 0.4,
  gestureFrac < 0.1 or > 0.8). The text gives the actual number and one
  concrete fix. No drill links.
- **Speech overview:** add `["presence", "Presence"]` to `DIMENSIONS` with one
  `detail` line (average eye contact and the weakest sub-signal).
- **History:** no new chart. Presence feeds `overall`, which is already plotted.

## Error handling

- `getUserMedia` rejects (no device, busy, denied): the toggle switches off,
  the status pill shows "Camera unavailable: <reason>", and recording
  continues audio-only.
- MediaPipe assets missing or a landmarker fails to load: same as above, with
  the message "Camera models missing — see README".
- The GPU delegate fails to init: retry once with the CPU delegate (~10 fps,
  still usable).
- The camera stream ends mid-session (unplugged): the loop stops, and the
  frames collected so far are still summarized if there are at least 3 s.

## Testing

- `src/presence.check.ts` (run: `bun src/presence.check.ts`) uses synthetic
  `Frame` sequences to assert:
  - all-looking → eye contact 1.0 / sub-score 100; no-face → 0;
  - script open → eye contact excluded from `presence`;
  - jittering shoulders score lower stillness than steady ones;
  - the gesture band: 0%, 40% and 100% gesturing give low / 100 / ~70;
  - fewer than 3 s of frames → `null`.
- Manual check in the real app: toggle on, record ~60 s while varying gaze,
  smile, movement and hands; confirm the live card responds and the report
  tile and tips make sense; confirm no web-process crash (`coredumpctl list`).
- `tsc` and `vite build` are clean; existing `.check.ts` files still pass.

## Out of scope

- Saving video, landmarks or per-second presence timelines.
- "Looked away at 1:23" markers in the report's moments list (needs a stored
  timeline).
- Landmark or skeleton overlays on the self-view.
- Presence drills, and a separate presence trend chart.
- macOS/Windows camera permission specifics. The Rust permission handler
  is Linux-only. Other platforms rely on wry's default behaviour and are
  untested.
